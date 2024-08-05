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
import pdb


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
        return os.path.join(self.root_dir, str(shard_id % 100), f'{shard_id}.tar')

    def _get_next_shard_id(self, shard_id):
        shard_id += self.group_size
        return shard_id % self.num_shards
    
    def dataset_filter(self, shard_id, members):
        # Placeholder function, to be inherited
        return members

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
                # MoDE_v1 can be iterative but the paper uses mmap for random access.
                members = self.dataset_filter(shard_id, members)

                json_uuid, img_uuid = -1, -2
                for member in members:
                    if member.name.endswith(".json"):
                        json_uuid = member.name[:-len(".json")]
                        with tar.extractfile(member) as f:
                            text_json = json.load(f)

                    if member.name.endswith(".jpeg") or member.name.endswith(".jpg"):
                        suffix = len(member.name.split('.')[-1])+1
                        img_uuid = member.name[:-suffix]
                        with tar.extractfile(member) as f:
                            img = f.read()

                    if img_uuid != json_uuid:
                        # assume uuid is json even and img ord;
                        continue

                    txt = random.choice(text_json["texts"])
                    txt = self.tokenizer([txt])[0]

                    with Image.open(BytesIO(img)) as img:
                        image = img.convert("RGB")
                        image = self.transform(image)

                    yield image, txt

            shard_id = self._get_next_shard_id(shard_id)
    
    def iter_func(self, shard_id):
        tarball_path = self._get_tarball_path(shard_id)

        with tarfile.open(tarball_path) as tar:
            members = tar.getmembers()
            members = self.dataset_filter(shard_id, members)



class IterativeMoDEWebDataset(IterativeWebDataset):
    # cluster agnostic
    def __init__(self, args, transform, tokenize):
        super(IterativeMoDEWebDataset,self).__init__(args, transform, tokenize)

        hrchy_filepath = os.path.join(self.args.hrchy_assign, self.args.dist_type, 'F{}-C{}.pth'.format(self.args.mode_fine, self.args.mode_size))
        hrchy_file = torch.load(hrchy_filepath)
        self.ic_idx = [i for i,a in enumerate(hrchy_file['assign'].tolist()) if a==self.args.coarse_idx]

        self.iter_func(0)

    def dataset_filter(self, shard_id, members):
        assert os.path.exists(self.args.fine_index)
        group_file_path = os.path.join(self.args.fine_index, str(shard_id%100), f'{shard_id}_assign_dist.json')
        with open(group_file_path, 'r') as json_file:
            group_file = json.load(json_file)
        
        cluster_key = f'{self.args.dist_type[0].upper()}{self.args.mode_fine}' # e.g., E1024, C512, !!! f'K{self.args.mode_fine}B1'
        file_grp = group_file[cluster_key]
        file_idx = {'txt':group_file['key'],'image':group_file['image']}

        index = (np.array(file_grp['assign'])[:,None]==np.array(self.ic_idx)[None]).sum(axis=-1)==1
        kept_keys = {key:np.array(value)[index] for key,value in file_idx.items()}
        
        if self.args.ooc_ratio == 0.0:
            all_keys = kept_keys
        else:
            out_group_keys = {key:np.array(value)[~index] for key,value in file_idx.items()}
            num_ooc_samples = int(len(out_group_keys['txt']) * self.args.ooc_ratio)
            the_index = np.random.permutation(len(out_group_keys['txt']))[:num_ooc_samples]
            the_keys = {key:value[the_index] for key,value in out_group_keys.items()}
            all_keys = {key:np.concatenate([kept_keys[key],the_keys[key]]) for key in the_keys}

        index = np.argsort(all_keys['txt'])
        kept_keys_txt = all_keys['txt'][index]
        kept_keys_image = all_keys['image'][index]

        # MoDE_v1 can be iterative but the paper uses mmap for random access.
        # The indexing of files can be done at kmeans inference 
        # together with cluster assignment
        new_members = []
        for i,j in zip(kept_keys_txt.tolist(),kept_keys_image.tolist()):
            new_members.append(members[i])
            new_members.append(members[j])

        return new_members
    

def get_mode_iter_wds_dataset(args, preprocess_fn, is_train, epoch=0):
    # borrowed from get_csv_dataset
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = IterativeMoDEWebDataset(
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
