# Copyright (c) Meta Platforms, Inc. and affiliates

import os
import random
import numpy as np
import time
import math

from pathlib import Path
from tqdm import tqdm


def balance_sampling(entry_count, inverted_index, max_match):
    if max_match == "inf":
        return np.sort(np.unique(inverted_index[:, 1]))

    entry_ids = inverted_index[:, 0]
    entry_id_count = entry_count[entry_ids]
    entry_id_count[entry_id_count < max_match] = max_match
    entry_probs = max_match / entry_id_count
    entry_selector = np.random.random_sample((inverted_index.shape[0],)) < entry_probs
    pair_offsets = np.sort(np.unique(inverted_index[:, 1][entry_selector]))
    return pair_offsets


def build_subset_index(args):
    entry_count = np.load(f"{args.index_dir}/entry_count.npy")
    print(f"entry_count.sum()={entry_count.sum()}")

    print(f"max_match={args.max_match}")
    dataset_index = []
    total_size = 0
    valid_shards = 0
    for shard_id in tqdm(range(args.start_shard, args.end_shard)):
        shard_folder = shard_id % 100
        inverted_index_fn = f"{args.index_dir}/{shard_folder}/{shard_id}_inverted.npy"
        if not os.path.exists(inverted_index_fn):
            continue
        inverted_index = np.load(inverted_index_fn, mmap_mode="r")
        pair_offsets = balance_sampling(entry_count, inverted_index, max_match=args.max_match)
        shard_offset = np.empty(shape=(pair_offsets.shape[0], 2), dtype=np.uint32)
        shard_offset[:, 0] = shard_id
        shard_offset[:, 1] = pair_offsets

        Path(f"{args.index_dir}/subset_{args.max_match}/{shard_folder}").mkdir( parents=True, exist_ok=True )
        np.save(f"{args.index_dir}/subset_{args.max_match}/{shard_folder}/{shard_id}.npy", pair_offsets)
        
        total_size += shard_offset.shape[0]
        valid_shards += 1

    print(f"total_size(pairs)={total_size}, valid_shards={valid_shards}")
