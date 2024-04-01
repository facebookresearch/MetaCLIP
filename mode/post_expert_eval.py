# Copyright (c) Meta Platforms, Inc. and affiliates

import sys
sys.path.append("./")
sys.path.append("src")

import os
import logging
import torch
import json

from open_clip import tokenize
from open_clip import create_model_and_transforms, get_mean_std

from training.distributed import init_distributed_device
from training.logger import setup_logging
from clipeval.eval_zeroshot import validate_zeroshot, mean_per_class, accuracy, roc_auc


def evaluate_logits(d, val_loader, templates, labels, model, tokenizer, classnorm=False):
    print('Evaluating {}'.format(d))

    outputs = validate_zeroshot(val_loader, templates, labels, model, tokenizer, False, classnorm)

    if d in ['FGVCAircraft', 'OxfordPets', 'Caltech101', 'Flowers102']:
        metric = mean_per_class(*outputs)
    elif d == 'Kinetics700':
        top1, top5 = accuracy(*outputs, topk=(1, 5))
        metric = (top1 + top5) / 2
        metric = metric.item()
    elif d == 'HatefulMemes':
        metric = roc_auc(*outputs)
    else:
        pred = outputs[0].argmax(dim=1)
        correct = pred.eq(outputs[1]).sum()
        metric = correct.item() / float(pred.size(0)) * 100.0

    return metric, outputs


@torch.no_grad()
def slip_evaluate_expert(args, model, val_transform, tokenizer, idx):
    from clipeval import datasets, eval_zeroshot

    catalog, all_templates, all_labels = eval_zeroshot.load_metadata("clipeval")

    if hasattr(model, "module"):
        model = model.module

    metrics = {}
    for d in catalog:
        result_fn = os.path.join(args.output_dir, 'eval_outputs', f'{d}_pred-{idx}.pth')
        if os.path.exists(result_fn):
            'logits' in torch.load(result_fn)
            continue
        
        val_dataset = datasets.get_downstream_dataset(
            catalog, d, is_train=False, transform=val_transform)
        templates = all_templates[d]
        labels = all_labels[d]

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size//2, shuffle=False,
            num_workers=args.workers, pin_memory=False, drop_last=False)
        
        metric, logits = evaluate_logits(d, val_loader, templates, labels, model, tokenizer)
        metrics[d] = metric
        json_str = json.dumps({"model": idx, "task": d, "acc": metric})
        torch.save({'logits':logits[0], 'targets':logits[1]}, result_fn)
        logging.info(json_str)
    return metrics


def main(args):

    device = init_distributed_device(args)
    mean, std = get_mean_std(args)
    args.log_path = os.path.join(args.output_dir, f'expert.log')
    args.log_level = logging.INFO
    setup_logging(args.log_path, args.log_level)
    os.makedirs(os.path.join(args.output_dir,'eval_outputs'), exist_ok=True)

    model, _, preprocess_val = create_model_and_transforms(
            args.model,
            args.pretrained,
            precision=args.precision,
            device=device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            pretrained_image=args.pretrained_image,
            mean=mean, std=std,
            inmem=hasattr(args, "inmem"),
            clip_model=args.clip_model
        )

    ckpt_list = [os.path.join(args.output_dir,f'expert_{i}','checkpoints','epoch_latest.pt') for i in range(args.mode_size)]
    logging.info('There are {} ckpts to be ensembled for exp {}.'.format(len(ckpt_list),os.path.dirname(args.name)))
    idxes = [i for i in range(len(ckpt_list))]
    if args.world_size > 1:
        assert len(ckpt_list) % args.world_size == 0
        seg = len(ckpt_list) // args.world_size
        idxes = idxes[args.rank*seg:(args.rank+1)*seg]
        ckpt_list = ckpt_list[args.rank*seg:(args.rank+1)*seg]

    for idx, ckpt_path in zip(idxes,ckpt_list):
        logging.info(f'Loading Model {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=torch.device('cuda'))
        
        if next(iter(ckpt['state_dict'].items()))[0].startswith('_orig_mod'): 
            ckpt['state_dict'] = {k[len('_orig_mod.'):]: v for k, v in ckpt['state_dict'].items()}
        if next(iter(ckpt['state_dict'].items()))[0].startswith('module'): 
            ckpt['state_dict'] = {k[len('module.'):]: v for k, v in ckpt['state_dict'].items()}
        
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        slip_evaluate_expert(args, model, preprocess_val, tokenize, idx)


if __name__ == "__main__":
    from configs_mode import search_config
    config = search_config(sys.argv[1])
    config.output_dir = os.path.dirname(config.output_dir)
    main(config)
