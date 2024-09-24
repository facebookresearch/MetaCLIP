# Copyright (c) Meta Platforms, Inc. and affiliates

# usage:
# python src/training/main.py b32_400m
# torchrun --nproc_per_node=8 src/training/main.py b32_400m
# python submit.py b32_400m

from dataclasses import dataclass
from configs import Config


@dataclass
class b32_400m(Config):
    gpu_trans=True
    engine="train_one_epoch_ex"
    eval_steps=5000
    save_frequency=1
    train_data="data/400m/{0..60800}.tar"
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
    nodes=16
    ngpus=4


@dataclass
class b16_400m(b32_400m):
    model="ViT-B-16-quickgelu"
    name="ViT-B-16"
    grad_checkpointing=True


@dataclass
class l14_400m(b32_400m): 
    model="ViT-L-14-quickgelu"
    name="ViT-L-14"
    lr=0.0004
    grad_checkpointing=True
    batch_size=256
    nodes=16
    ngpus=8


if __name__ == "__main__":
    import inspect
    import sys
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(obj):
            print(name)
