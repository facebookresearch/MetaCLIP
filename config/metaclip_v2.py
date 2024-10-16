# Copyright (c) Meta Platforms, Inc. and affiliates

# usage:
# python src/training/main.py b32_v2
# torchrun --nproc_per_node=8 src/training/main.py b32_v2
# python submit.py b32_v2

from dataclasses import dataclass
from configs import Config


@dataclass
class b32_v2_online(Config):
    gpu_trans=True
    engine="train_one_epoch_ex"
    eval_steps=5000
    save_frequency=1
    train_data="data/metaclip_v2_5B/{0..400000}.tar"
    workers=8
    train_num_samples=400_000_000   # assume same freq. of validation as 400M.
    batch_size=512
    epochs=32
    model="ViT-B-32-quickgelu"
    name="ViT-B-32"
    force_quick_gelu=True
    warmup=2000
    seed=0
    local_loss=True
    gather_with_grad=True
    online_curation="data/textprob_v2"
    nodes=8
    ngpus=8


@dataclass
class b32_v2_syn(b32_v2_online):
    cap_dir="data/cap_altogether"
    syn_ratio = 0.15


@dataclass
class h14_v2_syn(b32_v2_syn):
    precision = "amp_bf16"
    model="ViT-H-14"
    name="ViT-H-14"
    force_quick_gelu = False
    grad_checkpointing = True

    lr=0.0004
    beta2 = 0.95
    wd = 0.1

    train_num_samples = 1_600_000_000
    batch_size=512    
    nodes=32
    ngpus=8
