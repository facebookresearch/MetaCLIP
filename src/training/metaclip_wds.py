# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import mmap
import os
import json

from typing import Any, Callable, Optional

import numpy as np
import random

import tarfile
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from io import BytesIO
from PIL import Image, ImageDraw, ImageFilter
from training.distributed import world_info_from_env

from open_clip import tokenize
from .data import DataInfo


class IterativeWebDataset(torch.utils.data.IterableDataset):
    def __init__(self, args, transform, tokenize):
        self.args = args
        start, end = os.path.basename(args.train_data).split("{")[1].split("}")[0].split("..")
        self.num_shards = int(end) - int(start)
        self.root_dir = os.path.dirname(args.train_data)
        self.transform = transform
        self.tokenizer = tokenize
        self.start_shard_id = 0
        self.shard_ids = list(range(self.num_shards))

    def set_epoch(self, epoch, num_batches, step=0):
        random.seed(epoch+step)
        self.shard_ids = list(range(self.num_shards))
        random.shuffle(self.shard_ids)
        self.start_shard_id = (num_batches * epoch) % self.num_shards

    def _get_tarball_path(self, shard_id):
        return os.path.join(self.root_dir, f"{shard_id % 100}", f"{shard_id}.tar")

    def _get_next_shard_id(self, shard_id):
        shard_id += self.group_size
        return shard_id % self.num_shards

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        _, global_rank, world_size = world_info_from_env()
        self.group_size = int(num_workers * world_size)
        shard_id = num_workers * global_rank + worker_id
        shard_id = (shard_id + self.start_shard_id) % self.num_shards
        shard_id = self.shard_ids[shard_id]

        while True:
            tarball_path = self._get_tarball_path(shard_id)
            if not os.path.exists(tarball_path):
                shard_id = self._get_next_shard_id(shard_id)
                continue

            with tarfile.open(tarball_path) as tar:
                members = tar.getmembers()

                # metaclip_v1 can be iterative but the paper uses mmap for random access.
                json_uuid, img_uuid = -1, -2
                for member in members:
                    if member.name.endswith(".json"):
                        json_uuid = member.name[:-len(".json")]
                        with tar.extractfile(member) as f:
                            text_json = json.load(f)

                    if member.name.endswith(".jpeg"):
                        img_uuid = member.name[:-len(".jpeg")]
                        with tar.extractfile(member) as f:
                            img = f.read()

                    if img_uuid != json_uuid:
                        # assume uuid is json even and img ord;
                        continue

                    txt = random.choice(text_json["texts"])[1]
                    txt = self.tokenizer([txt])[0]

                    with Image.open(BytesIO(img)) as img:
                        image = img.convert("RGB")
                        image = self.transform(image)

                    yield image, txt

            shard_id = self._get_next_shard_id(shard_id)


def get_metaclip_iter_wds_dataset(args, preprocess_fn, is_train, epoch=0):
    # borrowed from get_csv_dataset
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = IterativeWebDataset(
        args,
        preprocess_fn,
        tokenize,
    )

    assert is_train
    num_samples = args.train_num_samples
    sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=True,
    )

    dataloader.num_samples = num_samples
    dataloader.num_batches = int(num_samples / (args.batch_size * args.world_size))

    return DataInfo(dataloader, sampler)
