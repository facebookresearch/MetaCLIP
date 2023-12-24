# Copyright (c) Meta Platforms, Inc. and affiliates

import re
import math
import json
import time
import numpy as np
import os

from tqdm import tqdm
from pathlib import Path


def entry_count(args):
    with open("metadata.json") as fr:
        metadata = json.load(fr)

    entry_count = np.zeros(shape=(len(metadata),), dtype=np.uint64)  #  uint64 to be safe for scaling.
    total_missing_shards = 0
    for shard_id in range(args.start_shard, args.end_shard):
        shard_group = shard_id % 100
        index_fn = f"{args.index_dir}/{shard_group}/{shard_id}_inverted.npy"

        if not Path(index_fn).is_file():
            total_missing_shards += 1
            continue

        entry_ids, counts = np.unique(np.load(index_fn, mmap_mode="r")[:, 0], return_counts=True)
        entry_count[entry_ids] += counts.astype(np.uint64)
    
    print(f"total_missing_shards={total_missing_shards}")    
    np.save(f"{args.index_dir}/entry_count.npy", entry_count)
