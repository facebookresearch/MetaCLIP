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

from src.open_clip.tokenizer import tokenize
from src.training.data import DataInfo
from src.training.distributed import world_info_from_env


class IterativeWebDataset(torch.utils.data.IterableDataset):
    """
    The dataset are organized in the following structure:
        `<dataset_dir>/{shard_id % 100}/{shard_id}.tar`.
    Each tar contains files in the following order (WebDatadet compatible):
    ```
        uuid1.json
        uuid1.jpeg
        uuid2.json
        uuid2.jpeg
    ```
    Each json has a `text` field with a list of texts associated with the image (uuid):
    [
        ['alt', 'this is a caption.'],
        ['alt', 'this is another caption for the same image.'],
        ...
    ]
    """

    def __init__(self, args, transform, tokenize):
        self.args = args
        start, end = os.path.basename(args.train_data).split("{")[1].split("}")[0].split("..")
        self.num_shards = int(end) - int(start)
        self.root_dir = os.path.dirname(args.train_data)
        self.transform = transform
        self.tokenizer = tokenize
        self.positions = None

    def set_positions(self, positions):
        self.positions = positions

    def _get_tarball_path(self, shard_id):
        return os.path.join(self.root_dir, f"{shard_id % 100}", f"{shard_id}.tar")

    def _get_next_shard_id(self, shard_id):
        self.global_shard_id += self.worker_size
        self.global_shard_id = shard_id % self.num_shards
        return self.global_shard_id

    def _get_worker_info(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        _, global_rank, world_size = world_info_from_env()
        self.worker_size = int(num_workers * world_size)
        return (global_rank, worker_id), num_workers, self.worker_size

    def __iter__(self):
        (global_rank, worker_id), num_workers, worker_size = self._get_worker_info()
        
        if self.positions is not None and self.positions[f"{global_rank}_{worker_id}"] != -1:
            self.global_shard_id = self.positions[f"{global_rank}_{worker_id}"]
            print(f"{global_rank}_{worker_id} restore {self.global_shard_id}")
            shard_id = self._get_next_shard_id()
        else:
            self.global_shard_id = global_rank * num_workers + worker_id
            shard_id = self.global_shard_id

        while True:
            tarball_path = self._get_tarball_path(shard_id)
            if not os.path.exists(tarball_path):
                shard_id = self._get_next_shard_id(shard_id)
                continue

            if hasattr(self.args, "online_curation"):
                with open(f"{self.args.online_curation}/{shard_id % 100}/{shard_id}.json") as f:
                    online_txts = json.load(f)

            if hasattr(self.args, "syn_ratio"):
                with open(f"{self.args.cap_dir}/{shard_id % 100}/{shard_id}.json") as f:
                    syn_cap = json.load(f)

            with tarfile.open(tarball_path) as tar:
                members = tar.getmembers()

                # metaclip_v1 can be iterative but the paper uses mmap for random access.
                json_uuid, img_uuid = None, None
                for member in members:
                    if member.name.endswith(".json"):
                        json_uuid = member.name[:-len(".json")]
                        if json_uuid.startswith("./"):
                            json_uuid = json_uuid[len('./'):]
                        with tar.extractfile(member) as f:
                            text_json = json.load(f)

                    if member.name.endswith(".jpeg"):
                        img_uuid = member.name[:-len(".jpeg")]
                        if img_uuid.startswith("./"):
                            img_uuid = img_uuid[len('./'):]
                        with tar.extractfile(member) as f:
                            img = f.read()

                    if img_uuid != json_uuid or img_uuid is None or json_uuid is None:
                        continue

                    if hasattr(self.args, "online_curation"):
                        if json_uuid not in online_txts:
                            json_uuid, img_uuid = None, None
                            continue
                        txt, prob = random.choice(online_txts[json_uuid])
                        if prob < random.random():
                            json_uuid, img_uuid = None, None
                            continue
                    else:
                        txt = random.choice(text_json["texts"])[1]
                    
                    if hasattr(self.args, "syn_ratio"):
                        if random.random() < self.args.syn_ratio:
                            if json_uuid in syn_cap:
                                txt = syn_cap[json_uuid]

                    txt = self.tokenizer([txt])[0]

                    with Image.open(BytesIO(img)) as img:
                        image = img.convert("RGB")
                        image = self.transform(image)

                    yield image, txt, worker_id, shard_id
                    json_uuid, img_uuid = None, None

            shard_id = self._get_next_shard_id(shard_id)


def get_metaclip_iter_wds_dataset(args, preprocess_fn, is_train, positions=None):
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

    dataset.set_positions(positions)

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
