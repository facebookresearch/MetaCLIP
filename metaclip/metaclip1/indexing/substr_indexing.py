# Copyright (c) Meta Platforms, Inc. and affiliates

import json
import numpy as np

from pathlib import Path

from metaclip.substr_matching import substr_matching


def build_index(output_index_fn, metadata, texts):
    meta_index = []
    print("here", output_index_fn)
    for text_id, text in enumerate(texts):
        matched_entry_ids = substr_matching(text, metadata)
        meta_index.extend([[entry_id, text_id] for entry_id in matched_entry_ids])

    meta_index = np.array(meta_index, dtype=np.uint32)
    Path(output_index_fn).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_index_fn, meta_index)
    return meta_index


def build_shards_index(index_dir, metadata, load_texts_fn, start_shard, end_shard):
    for shard_id in range(start_shard, end_shard):
        shard_group = shard_id % 100
        output_index_fn = f"{index_dir}/{shard_group}/{shard_id}_inverted.npy"
        if Path(output_index_fn).is_file():
            try:
                np.load(output_index_fn)  # trial loading of existing one.
                continue
            except Exception as e:
                print(f"error on existing {output_index_fn}: {e}")
                pass

        texts, tar_info = load_texts_fn(shard_id)
        # TODO: also save tar_info for random access of image-text pairs.
        build_index(output_index_fn, metadata, texts)
