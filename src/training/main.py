# Copyright (c) Meta Platforms, Inc. and affiliates

import sys
sys.path.append("./") # trick to avoid bash env. on PYTHONPATH

import re
import logging
import os
import random
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler


try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None


from src.open_clip.factory import create_model_and_transforms
from src.open_clip.transform import get_mean_std
from src.open_clip.model import CLIP, VisualTransformer, Transformer, ResidualAttentionBlock
from src.training.data import get_data
from src.training.distributed import is_master, init_distributed_device, world_info_from_env
from src.training.logger import setup_logging
from src.training.scheduler import cosine_lr
from src.training import train
from src.training.checkpoint import load_checkpoint, unwrap_model


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main(args):
    # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
    args.model = args.model.replace('/', '-')

    # get the name of the experiments
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and args.resume is None and not hasattr(args, "eval"):
            print(
                "Error. Experiment already exists. `rm -rf logs/{args.name}` ?"
            )
            return -1

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # fully initialize distributed device environment
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = init_distributed_device(args)

    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    if is_master(args):
        args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
        args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''
        args.checkpoint_path = ''

    assert args.precision in ['amp', 'fp16', 'fp32', 'amp_bf16']
    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    random_seed(args.seed, 0)
    mean, std = get_mean_std()

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        pretrained_image=args.pretrained_image,
        mean=mean, std=std,
        gpu_trans=hasattr(args, "gpu_trans"),
        clip_model=args.clip_model,
    )

    composed_model = model
    if hasattr(args, "cap_model"):
        from src.training.train_altogether import create_captioner
        clip_model, model = create_captioner(args, model, device)
        composed_model = clip_model, model

    random_seed(args.seed, args.rank)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(dir(args)):
                if name.startswith('__'):
                    continue
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed:
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None
    if args.train_data:
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n or 'norm' in n or 'layer_norm' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    step, positions = -1, None
    
    
    if args.resume is not None:
        if os.path.isfile(args.resume):
            model_to_load = unwrap_model(model)
            step, positions = load_checkpoint(args.resume, model_to_load, optimizer=optimizer, scaler=scaler)
            logging.info(f"=> resuming checkpoint '{args.resume}' (step {step})")
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    # initialize datasets
    data = get_data(args, (preprocess_train, preprocess_val), positions)
    assert len(data), 'At least one train or eval dataset must be specified.'

    if hasattr(args, "torchcompile") and args.torchcompile:
        logging.info('Compiling model...')
        try:
            model = torch.compile(model)
        except Exception:
            logging.warn("please use PyTorch 2.0")

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = data["train"].dataloader.num_batches * args.epochs
        if "endsft" in data:
            endsft_steps = data["endsft"].dataloader.num_batches * args.endsft_epochs
            logging.info(f"appending {endsft_steps} endsft steps")
            total_steps += endsft_steps

        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if 'train' not in data or hasattr(args, "eval") and args.eval:  # huxu: merge native/SLIP eval.
        # TODO: move to below first.
        from src.training.slip_evaluate import slip_evaluate
        from src.open_clip.tokenizer import tokenize
        # in case a downloaded model.
        os.makedirs(args.output_dir, exist_ok=True)
        slip_evaluate(args, model, preprocess_val, tokenize)
        evaluate(model, data, start_epoch, args, writer)
        return

    start_step = step + 1

    if hasattr(args, "engine"):
        engine = args.engine

        import importlib
        for model_code in os.listdir(f"src/training"):
            if model_code.startswith("train"):
                module = importlib.import_module("src.training." + model_code[:-len(".py")])
                if hasattr(module, engine):
                    engine_cls = getattr(module, engine)
                    break
        else:
            raise ValueError(f"{engine} not found.")
    else:
        engine_cls = train.train_one_epoch_ex

    engine_cls(args, composed_model, data, start_step, total_steps, optimizer, scaler, scheduler, writer)

    
    if hasattr(args, "eval") and args.eval and any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
        from src.training.slip_evaluate import slip_evaluate
        from src.open_clip import tokenize

        slip_evaluate(args, model, preprocess_val, tokenize)


if __name__ == "__main__":
    import sys
    sys.path.append("./")
    from configs import search_config
    config = search_config(sys.argv[1])
    if len(sys.argv) == 3:
        config.resume = os.path.join(config.output_dir, "checkpoints", sys.argv[2])
    main(config)
