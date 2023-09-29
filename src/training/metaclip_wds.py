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


def fairblurbbox(image, faces):
    if len(faces) == 0:
        return image

    mask = Image.new(mode="L", size=image.size, color="white")
    width, height = image.size
    max_diagonal = 0
    for face in faces:
        x0, y0, x1, y1 = faces[face]["facial_area"]
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        x0 = min(x0, width)
        x1 = min(x1, width)
        y0 = min(y0, height)
        y1 = min(y1, height)
        bbox = [x0, y0, x1, y1]
        erosion = 0.1
        blur_radius = 1/17
        mask_radius = 0.1

        diagonal = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        max_diagonal = max(max_diagonal, diagonal)
        bbox = [
            bbox[0] - erosion * diagonal,
            bbox[1] - erosion * diagonal,
            bbox[2] + erosion * diagonal,
            bbox[3] + erosion * diagonal,
        ]
        draw = ImageDraw.Draw(mask)
        draw.rectangle(bbox, fill="black")
    blurred_img = image.filter(ImageFilter.GaussianBlur(blur_radius * max_diagonal))
    blurred_mask = mask.filter(ImageFilter.GaussianBlur(mask_radius * max_diagonal))
    img = Image.composite(image, blurred_img, blurred_mask)
    return img


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

                    # with Image.open(BytesIO(img)) as img:
                    with Image.open(BytesIO(img)) as img:
                        image = img.convert("RGB")
                        if "face_bbox" in text_json:
                            image = fairblurbbox(image, text_json["face_bbox"])
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
