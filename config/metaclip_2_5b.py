# Copyright (c) Meta Platforms, Inc. and affiliates

# usage:
# python src/training/main.py b32_fullcc
# torchrun --nproc_per_node=8 src/training/main.py b32_fullcc
# python submit.py b32_fullcc

from dataclasses import dataclass
from configs import Config


@dataclass
class b32_fullcc(Config):
    gpu_trans=True
    engine="train_one_epoch_ex"
    eval_steps=5000
    save_frequency=1
    train_data="data/metaclip_v1_2_5B/{0..200000}.tar"
    workers=8
    train_num_samples=400000000   # assume same freq. of validation as 400M.
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


@dataclass
class b16_fullcc(b32_fullcc):
    model="ViT-B-16-quickgelu"
    name="ViT-B-16"
    grad_checkpointing=True


@dataclass
class l14_fullcc(b32_fullcc):
    model="ViT-L-14-quickgelu"
    name="ViT-L-14"
    lr=0.0004
    batch_size=256
    grad_checkpointing=True
    nodes=16
    ngpus=8


@dataclass
class h14_fullcc(b32_fullcc):
    model="ViT-H-14-quickgelu"
    name="ViT-H-14"
    lr=0.0004
    batch_size=256
    grad_checkpointing=True
    nodes=16
    ngpus=8


@dataclass
class G14_fullcc(b32_fullcc):
    model="ViT-bigG-14-quickgelu"
    name="ViT-bigG-14"
    lr=0.0004
    batch_size=128
    force_quick_gelu=True
    grad_checkpointing=True
    nodes=32
    ngpus=8


if __name__ == "__main__":
    import inspect
    import sys
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(obj):
            print(name)
