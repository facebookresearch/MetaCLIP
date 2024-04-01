# Copyright (c) Meta Platforms, Inc. and affiliates

from configs_mode import Config


def b32_fullcc(**overwrite):
    return Config(
        overwrite,
        one_iter=True,
        inmem=True,
        engine="train_one_epoch_ex",
        eval_steps=5000,
        save_frequency=1,
        train_data="data/metaclip_v1_2_5B/{0..200000}.tar",
        workers=8,
        train_num_samples=400000000,
        batch_size=512,
        epochs=32,
        model="ViT-B-32-quickgelu",
        name="ViT-B-32",
        force_quick_gelu=True,
        warmup=2000,
        seed=0,
        local_loss=True,
        gather_with_grad=True,
        nodes=8, ngpus=8,
    )


# ================================
# Configs for MoDE
# (change coarse_idx to from 0 to mode_size-1 the experts)
# align paths for fine_index and hrchy_assign
# in mode/get_prep_parser.py and configs_mode.py
# ================================

def b32_mode():
    return b32_fullcc(
        dataset_type='cluster',
        mode_size=4, coarse_idx=1,
        mode_fine=1024,
        quick_init=27, seed_exp='logs/b32_fullcc',
    )

def b16_mode():
    return b32_fullcc(
        model="ViT-B-16-quickgelu",
        name="ViT-B-16",
        grad_checkpointing=True,
        dataset_type='cluster',
        mode_size=4, coarse_idx=0,
        mode_fine=1024,
        quick_init=27, seed_exp='logs/b16_fullcc',
    )

def l14_mode():
    return b32_fullcc(
        model="ViT-L-14-quickgelu",
        name="ViT-L-14",
        lr=0.0004,
        batch_size=256,
        grad_checkpointing=True,
        nodes=16, ngpus=8,
        dataset_type='cluster',
        mode_size=4, coarse_idx=0,
        mode_fine=1024,
        quick_init=27, seed_exp='logs/l14_fullcc',
    )


if __name__ == "__main__":
    import inspect
    import sys
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(obj):
            print(name)
