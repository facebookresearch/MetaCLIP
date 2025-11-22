# Copyright (c) Meta Platforms, Inc. and affiliates

import ast
import json
import logging
import math
import os
import random
import sys
import torch

from dataclasses import dataclass
from multiprocessing import Value

from dataclasses import dataclass
from multiprocessing import Value

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path
        
        import torchvision.datasets as datasets
        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def get_metaclip_dataset(args, preprocess_fn, is_train, positions=None):
    # a switcher func for different versions of dataloader.
    from src.training.metaclip_wds import get_metaclip_iter_wds_dataset
    return get_metaclip_iter_wds_dataset(args, preprocess_fn, is_train, positions)


def get_mode_dataset(args, preprocess_fn, is_train, positions=None):
    # a switcher func for different versions of dataloader.
    from src.training.mode_wds import get_mode_iter_wds_dataset
    return get_mode_iter_wds_dataset(args, preprocess_fn, is_train, positions)


def get_dataset_fn(args, data_path, dataset_type):
    if hasattr(args, "dataset_cls") and data_path == args.train_data:
        return get_dataset

    if dataset_type == "cluster":
        return get_mode_dataset
    elif data_path.endswith(".tar"):
        return get_metaclip_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, positions=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_dataset_fn(args, args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, positions=positions)
    
    if args.val_data:
        data["val"] = get_dataset_fn(args, args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data