# Copyright (c) Meta Platforms, Inc. and affiliates


import math
import json
import time
import numpy as np
import os
import tarfile

from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict

import pickle

import sys
sys.path.append("./")

from metaclip.curation.substr_matching import (
    lid_langcode_to_metadata_langcode,
    get_spaced_metadata,
    initialize_automaton,
    spacing,
    substr_match
)


index_json_dir = f"data/index_json"
curated_index_json_dir = f"data/curated_index_json"
automaton_dir = f"data/metadata"
text_json_dir = f"data/pkg_json"
valid_uuids_dir = f"data/valid_uuids"


def p_to_t(entry_counts, p=0.1):
    """
    convert p (portion of tail counts) into t (threshold of head / tail);

    Parameters:
    entry_counts (np.array): counts indexed by indexes of metadata;
    p (float): portion of tail counts;

    Returns:
    t (int): threshold of head / tail count;
    """
    counts = np.sort(entry_counts)
    cumsum_counts = np.cumsum(counts)
    portion = cumsum_counts / counts.sum()
    p_idx = (np.abs(portion - p)).argmin()
    t = counts[p_idx]
    return t


def t_to_p(entry_counts, t=20000):
    """
    convert t (threshold of head / tail) into p (portion of tail counts);

    Parameters:
    entry_counts (np.array): counts indexed by indexes of metadata;
    t (int): threshold of head / tail count;
    Returns:
    p (float): portion of tail counts;
    """
    return float(entry_counts[entry_counts < t].sum() / entry_counts.sum())


def count_to_prob(entry_count, t):
    """
    convert counts into entry probability by t (entries with counts smaller than t will have prob. 1.0; otherwise t / count;

    Parameters:
    entry_count (np.array): counts indexed by indexes of metadata;
    t (int): threshold of head / tail count;
    Returns:
    entry_probs (np.array): entry's probability;
    """
    p_entry_counts = np.array(entry_count)
    p_entry_counts[p_entry_counts < t] = t
    entry_probs = t / p_entry_counts
    return entry_probs


def count_per_shard(start, end):
    """
    Stage 1: substring match.

    Parameters:
    start (int): starting shard_id;
    end (int): (exclusive) end shard_id;
    """

    automaton_ml = {}
    entry_counts_ml = {}

    if Path(f"{index_json_dir}/{start}.npz").exists():
        return

    for shard_id in range(start, end):
        if Path(f"{index_json_dir}/{shard_id - shard_id % 2000}/{shard_id}.json").exists():
            try:  # test reading if failed redo.
                with open(f"{index_json_dir}/{shard_id - shard_id % 2000}/{shard_id}.json") as f:
                    text_index = json.load(f)
            except:
                Path(f"{index_json_dir}/{shard_id - shard_id % 2000}/{shard_id}.json").unlink()

        if not Path(f"{index_json_dir}/{shard_id - shard_id % 2000}/{shard_id}.json").exists():
            with open(f"{valid_uuids_dir}/{shard_id - shard_id % 2000}/{shard_id}.json") as f:
                valid_uuids = json.load(f)

            with open(f"{text_json_dir}/{shard_id - shard_id % 2000}/{shard_id}.json") as f:
                uuid_to_text = json.load(f)

            text_index = {}
            for uuid in uuid_to_text:
                if uuid not in valid_uuids:
                    continue

                texts = []

                for rec in uuid_to_text[uuid]:
                    _, txt, lang = rec
                    txt = txt.strip()
                    lang_id = lid_langcode_to_metadata_langcode(lang_id)

                    matched_entry_ids_list = substr_match(lang_id, txt, automaton_dir, automaton_ml, matching_fn="iter")
                    if len(matched_entry_ids_list) == 0:
                        continue
    
                    texts.append([txt, lang, matched_entry_ids_list])
                
                if len(texts) > 0:
                    text_index[uuid] = texts

            Path(f"{index_json_dir}/{shard_id - shard_id % 2000}").mkdir(parents=True, exist_ok=True)
            with open(f"{index_json_dir}/{shard_id - shard_id % 2000}/{shard_id}.json", "w") as f:
                json.dump(text_index, f)

        # text_index exists now.
        for uuid in text_index:
            for txt, lang, matched_entry_ids_list in text_index[uuid]:
                # TODO: refactor this into a function.
                lang_id = lid_langcode_to_metadata_langcode(lang)

                if lang_id not in entry_counts_ml:
                    with open(f'{automaton_dir}/{lang_id}.pkl', 'rb') as f:
                        automaton = pickle.load(f)
                    entry_counts_ml[lang_id] = np.zeros((len(automaton),), dtype=np.uint64)
                entry_counts_ml[lang_id][matched_entry_ids_list] += 1

    np.savez(f"{index_json_dir}/{start}.npz", **entry_counts_ml)


def global_count(t_en=170000):
    """
    Stage 2: global count of matches.

    Parameters:
    t_en (int): threshold of head / tail count for English (OpenAI CLIP: 20k; MetaCLIP: 170k)
    """
    all_lang_entry_counts_ml = {}
    for shard_group in range(0, 1600000, 2000):
        entry_counts_ml = np.load(f"{index_json_dir}/{shard_group}.npz")

        for lang_code in entry_counts_ml:
            if lang_code not in all_lang_entry_counts_ml:
                all_lang_entry_counts_ml[lang_code] = entry_counts_ml[lang_code]
            else:
                all_lang_entry_counts_ml[lang_code] += entry_counts_ml[lang_code]

    Path(f"{index_json_dir}/per_lang").mkdir(parents=True, exist_ok=True)

    p_t = t_to_p(all_lang_entry_counts_ml['en'], t=t_en)

    Path(f"{index_json_dir}/per_lang_prob").mkdir(parents=True, exist_ok=True)
    for lang_id in all_lang_entry_counts_ml:
        if lang_id == "en":
            t = t_en
        else:
            t = p_to_t(all_lang_entry_counts_ml[lang_id], p_t)

        entry_probs = count_to_prob(all_lang_entry_counts_ml[lang_id], t)
        np.save(f"{index_json_dir}/per_lang_prob/{t_en}_{lang_id}.npy", entry_probs.astype(np.float32))


def curate(start, end):
    """
    Stage 3: curate an img(uuid) text pair.

    Parameters:
    start (int): starting shard_id;
    end (int): (exclusive) end shard_id;
    """
    entry_probs_per_lang = {}

    for shard_id in range(start, end):
        with open(f"{index_json_dir}/{shard_id - shard_id % 2000}/{shard_id}.json") as f:
            uuid_to_text = json.load(f)

        curated_text_index = {}
        for uuid in uuid_to_text:
            txt_tuples = uuid_to_text[uuid]

            txt, lang_id, matched_entry_ids_list = random.choice(txt_tuples)

            lang_id = lid_langcode_to_metadata_langcode(lang_id)

            if lang_id not in entry_probs_per_lang:
                entry_probs_per_lang[lang_id] = np.load(f"{index_json_dir}/per_lang_t/{self.args.t}_{lang_id}.npy", mmap_mode="r")

            prob = 1. - np.prod(1. - entry_probs_per_lang[lang_id][matched_entry_ids_list])
            if prob < random.random():
                continue

            curated_text_index[uuid] = [txt, lang_id, matched_entry_ids_list]

        Path(f"{curated_index_json_dir}/{shard_id - shard_id % 2000}").mkdir(parents=True, exist_ok=True)
        with open(f"{curated_index_json_dir}/{shard_id - shard_id % 2000}/{shard_id}.json", "w") as f:
            json.dump(curated_text_index, f)


if __name__ == '__main__':
    print(f"index_json_dir={index_json_dir}")
    
    import sys
    if len(sys.argv) == 1:  # global count
        global_count()

    else:
        import os
        import submitit
        import sys

        shards_per_thread = 2000
        job_plans = [
            ("data", (0, 400000)),
        ]

        if sys.argv == "curate":
            submitit_fn = curate
        elif sys.argv == "count_per_shard":
            submitit_fn = count_per_shard
        else:
            raise ValueError(f'unknown {sys.argv[1]} as submitit func.')
        
        for partition, (start_shard, end_shard) in job_plans:
            params = dict(
                name=f"curate",
                gpus_per_node=0,
                mem_gb=80,
                cpus_per_task=4,
                nodes=1,
                slurm_partition=partition,
                timeout_min=4320,
            )

            executor = submitit.AutoExecutor(
                folder="submitit/%j"
            )
            executor.update_parameters(**params)

            jobs = []
            with executor.batch():
                starts = list(range(start_shard, end_shard, shards_per_thread))
                ends = list(range(start_shard + shards_per_thread, end_shard + shards_per_thread, shards_per_thread))
                for start_shard, end_shard in zip(starts, ends):
                    job = executor.submit(
                        submitit_fn,
                        start_shard, end_shard
                    )
                    jobs.append(job)

            if len(jobs) > 0:
                print(partition, len(jobs), jobs[0].job_id, jobs[-1].job_id)
