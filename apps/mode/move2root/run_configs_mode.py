# Copyright (c) Meta Platforms, Inc. and affiliates

# usage:
# python src/training/main.py b32_mode
# torchrun --nproc_per_node=8 src/training/main.py b32_mode
# python submitit_mode.py b32_mode

from dataclasses import dataclass
from configs_mode import Config

# ================================
# Configs for MoDE
# set the following checkpoint paths before running
# more explanation can be found in configs_mode.py
# ================================

b32_demo_full=''
b16_demo_full=''
l14_demo_full=''

@dataclass
class b32_mode(Config):
    one_iter=True
    inmem=True
    engine="train_one_epoch_ex"
    eval_steps=5000
    save_frequency=1
    train_data="data/demo/{0..200000}.tar"
    workers=8
    train_num_samples=400000000
    batch_size=512
    epochs=32
    model="ViT-B-32-quickgelu"
    name="ViT-B-32"
    force_quick_gelu=True
    warmup=2000
    seed=0
    local_loss=True
    gather_with_grad=True
    nodes=8
    ngpus=8
    # Configs for MoDE
    dataset_type='cluster'
    mode_size=4 
    coarse_idx=1 # (change from 0 to mode_size-1 index the data experts)
    mode_fine=1024
    quick_init=27
    seed_exp=f'logs/{b32_demo_full}'

@dataclass
class b16_mode(b32_mode):
    model="ViT-B-16-quickgelu"
    name="ViT-B-16"
    grad_checkpointing=True
    seed_exp=f'logs/{b16_demo_full}'

@dataclass
class l14_mode(b32_mode):
    model="ViT-L-14-quickgelu"
    name="ViT-L-14"
    lr=0.0004
    batch_size=256
    grad_checkpointing=True
    nodes=16
    ngpus=8
    seed_exp=f'logs/{l14_demo_full}'


if __name__ == "__main__":
    import inspect
    import sys
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(obj):
            print(name)
