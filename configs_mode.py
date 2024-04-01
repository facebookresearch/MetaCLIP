# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import inspect

from collections import OrderedDict

import sys
sys.path.append("src")

from training.params import get_default_params
from mode.get_prep_parser import get_default_paths


class Config:
    train_data = None
    val_data = None
    train_num_samples = None
    val_num_samples = None
    dataset_type = "auto"
    dataset_resampled = False
    csv_separator = "\t"
    csv_img_key = "filepath"
    csv_caption_key = "title"
    imagenet_val = "/datasets01/imagenet_full_size/061417/val"
    imagenet_v2 = None
    logs = "./logs/"
    log_local = False
    name = None
    workers = 8
    batch_size = 64
    epochs = 32
    lr = None
    beta1 = None
    beta2 = None
    eps = None
    wd = 0.2
    warmup = 2000  # 10000
    use_bn_sync = False
    skip_scheduler = False
    save_frequency = 1
    save_most_recent = True  # False
    zeroshot_frequency = 1
    val_frequency = 1
    resume = None
    precision = "amp"
    clip_model = "CLIP"
    model = "RN50"
    pretrained = ''
    pretrained_image = False
    lock_image = False
    lock_image_unlocked_groups = 0
    lock_image_freeze_bn_stats = False
    grad_checkpointing = False
    local_loss = False
    gather_with_grad = False
    force_quick_gelu = False
    torchscript = False
    trace = False
    dist_url = "env://"
    dist_backend = "nccl"
    report_to = "tensorboard"
    wandb_notes = ''
    debug = False
    copy_codebase = False
    horovod = False
    ddp_static_graph = False
    no_set_device_rank = False
    seed = 0
    norm_gradient_clip = None

    fine_index = ''
    hrchy_assign = ''
    ooc_ratio = 0.02 # slightly better than 0.0
    dist_type = 'euclidean'

    def __init__(self, overwrite=None, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        if overwrite is not None or len(overwrite) > 0:
            for key in overwrite:
                setattr(self, key, overwrite[key])

        if not hasattr(self, "output_dir"):
            import sys
            self.name = sys.argv[1]
            print("config.name (sys.argv[1]) =", self.name)
            if 'mode' in self.name:
                assert self.coarse_idx >=0 and self.coarse_idx < self.mode_size
                sub_str=f'expert_{self.coarse_idx}'
                self.name = '{}_n{}m{}/{}'.format(self.name, self.mode_size, self.mode_fine, sub_str)

                datakey = ''
                paths = get_default_paths()[datakey]
                self.train_data = paths['root']
                self.fine_index = paths['assign']
                self.hrchy_assign = paths['cluster']

                if self.resume is None:
                    self.resume = os.path.join(self.seed_exp, 'checkpoints', f'epoch_{self.quick_init}.pt')

        self.output_dir = os.path.join(self.logs, self.name)
        self.extra_from_params()

    def extra_from_params(self):
        args = self
        # If some params are not passed, we use the default values based on model name.
        default_params = get_default_params(args.model)
        for name, val in default_params.items():
            if getattr(args, name) is None:
                setattr(args, name, val)

    def add_cmd_args(self, cmd_args):
        for key, value in vars(cmd_args).items():
            if not key.startswith("__"):
                setattr(self, key, value)
        return self

    def __str__(self):
        return "\n".join([f"{k}={v}" for k, v in vars(self).items()])


def parse_start_end(shards):
    start, end = os.path.basename(shards).split("{")[1].split("}")[0].split("..")
    return int(start), int(end)


def search_config(config_name):
    import importlib
    project_dir = os.path.dirname(__file__)
    all_configs = {}
    for code in os.listdir(project_dir):
        if code.endswith(".py") and code.startswith("run_configs"):
            print(f"searching config in {code}")
            module = importlib.import_module(code[:-3])
            for _config_name in dir(module):
                if _config_name in ["Config"] or _config_name.startswith("__") or _config_name.startswith("run_config"):
                    continue
                if _config_name not in all_configs:
                    all_configs[_config_name] = module
    print(f"launching {config_name} from {all_configs[config_name].__file__}")
    config = getattr(all_configs[config_name], config_name)()
    return config
