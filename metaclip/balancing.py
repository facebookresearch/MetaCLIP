# Copyright (c) Meta Platforms, Inc. and affiliates

import json
import numpy as np
import os
import random

from tqdm import tqdm


def balance_sampling(matched_entry_ids, entry_prob):
    # this can be placed in a pipeline or on-the-fly in a data loader.
    # see a numpy impl. at metaclip.indexing.balance_sampling.balance_sampling
    for entry_id in matched_entry_ids:
        if random.random() < entry_prob[entry_id]:
            return True
    return False


def main(input_dir, balanced_dir, t):
    # this func is for demo purpose of how the algorithm works, see metaclip.pipeline for an efficient impl.
    with open("metadata.json") as f:
        metadata = json.load(f)

    entry_count = np.zeros(shape=(len(metadata),), dtype=np.uint64)  # uint64 to be safe for scaling.
    
    # TODO: add cross json global dedup
    D = []
    for json_file in tqdm(os.listdir(input_dir)):
        with open(f"{input_dir}/{json_file}") as f:
            parsed_json = json.load(f)
        for rec in parsed_json:
            # this is a pure-python impl.: we use `numpy.unique` in the paper.
            for texts in rec["texts"]:
                for entry in texts[2]:  # index 2 is a list of substr matched entry ids.
                    entry_count[entry] += 1
            D.append(rec)

    os.makedirs(balanced_dir, exist_ok=True)
    np.save(f"{balanced_dir}/entry_count.npy", entry_count)

    entry_count[entry_count < t] = t
    entry_prob = t / entry_count

    D_star = []
    for rec in D:
        for texts in rec["texts"]:
            if balance_sampling(texts[2], entry_prob):
                D_star.append(rec)

    with open(f"{balanced_dir}/curated.json", "w") as fw:
        json.dump(D_star, fw)


if __name__ == '__main__':
    import sys
    input_dir = sys.argv[1]
    balanced_dir = sys.argv[2]
    t = int(sys.argv[3])
    main(input_dir, balanced_dir, t)
