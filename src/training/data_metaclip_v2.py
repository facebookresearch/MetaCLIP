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

import pickle

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from io import BytesIO
from PIL import Image, ImageDraw, ImageFilter

from src.mini_clip.factory import get_tokenizer
from src.training.data import DataInfo
from src.training.distributed import world_info_from_env

# check whether the following are valid now.
from src.training.data_utils import tar_iter, transform_img_member


from metaclip.curation.substr_matching import lid_langcode_to_metadata_langcode 


class IterativeWebDatasetWorldWide(torch.utils.data.IterableDataset):
    """
    Meta CLIP 2 datasets are organized in the following structure.
    
    `train_data` contains tars of jpeg files:
        `{train_data}/<shard_group>/<shard_id>.tar`.

        Each tar contains files in the following order (WebDataSet compatible):
        ```
            uuid1.jpeg
            uuid2.jpeg
        ```

    `pkg_json` contains jsons with the following format:
        `{pkg_json}/<shard_group>/<shard_id>.json`.
        Each json is a dict with keys of image uuid and values of a list:
        [
            ['this is a caption.', 'en', [<index_of_matched_metadata_entry1>, <index_of_matched_metadata_entry2> ...]],
            ...
        ]

    `{entry_count_dir}/per_lang_prob` contains numpy arrays corresponding to all (LID-aligned, see paper for details) languages:
        rows are indexes (offset) of metadata entries (matching the order in json list or `.pkl` automaton);
        columns are probabilities of metadata entries (see Algorithm 1 in paper for more details).    
    """

    def __init__(self, args, transform):
        self.args = args
        start, end = os.path.basename(args.train_data).split("{")[1].split("}")[0].split("..")
        self.num_shards = int(end) - int(start)
        self.root_dir = [os.path.dirname(root_dir) for root_dir in args.train_data.split(",")]
        self.transform = transform
        self.tokenizer = get_tokenizer(args.tokenizer if hasattr(args, "tokenizer") else None)
        self.positions = None

    def set_positions(self, positions):
        self.positions = positions

    def _get_next_shard_id(self, shard_id):
        shard_id += self.worker_size
        shard_id = shard_id % self.num_shards
        return shard_id

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

    def _shard_iter(self):
        (global_rank, worker_id), num_workers, worker_size = self._get_worker_info()
        if self.positions is not None and f"{global_rank}_{worker_id}" in self.positions and self.positions[f"{global_rank}_{worker_id}"] != -1:
            shard_id = self.positions[f"{global_rank}_{worker_id}"]
            print(f"{global_rank}_{worker_id} restore {shard_id}")
            shard_id = self._get_next_shard_id(shard_id)
        else:
            shard_id = global_rank * num_workers + worker_id

        while True:
            yield shard_id, worker_id
            shard_id = self._get_next_shard_id(shard_id)

    def __iter__(self):    
        entry_probs_per_lang = {}

        for shard_id, worker_id in self._shard_iter():
            assert hasattr(self.args, "pkg_json")
            with open(f"{self.args.pkg_json}/{shard_id - shard_id % 2000}/{shard_id}.json") as f:
                uuid_to_text = json.load(f)

            for uuid, member, tar in tar_iter([f"{root_dir}/{shard_id - shard_id % 2000}/{shard_id}.tar" for root_dir in self.root_dir]):
                if uuid not in uuid_to_text:
                    continue

                txt_tuples = uuid_to_text[uuid]

                if len(txt_tuples) == 0:
                    continue

                txt, lang_id, matched_entry_ids_list = random.choice(txt_tuples)

                lang_id = lid_langcode_to_metadata_langcode(lang_id)

                if lang_id not in entry_probs_per_lang:
                    entry_probs_per_lang[lang_id] = np.load(f"{self.args.entry_prob_dir}/per_lang_t/{self.args.t}_{lang_id}.npy", mmap_mode="r")

                prob = 1. - np.prod(1. - entry_probs_per_lang[lang_id][matched_entry_ids_list])
                if prob < random.random():
                    continue

                image = transform_img_member(self.transform, member, tar)
                txt = self.tokenizer([txt])[0]
                yield image, txt, worker_id, shard_id
