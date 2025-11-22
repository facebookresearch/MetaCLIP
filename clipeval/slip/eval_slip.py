# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import math
import torch
import torch.nn.functional as F

import json
import os

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from clipeval.slip import datasets, eval_zeroshot


@torch.no_grad()
def slip_evaluate(model, val_transform, tokenizer, batch_size, output_fn, test=False):
    catalog, all_templates, all_labels = eval_zeroshot.load_metadata("clipeval/slip")

    if hasattr(model, "module"):
        model = model.module

    metrics = {}
    for d in catalog:
        if test and d != 'imagenet':
            print(f"skip {d} for testing")
            continue

        val_dataset = datasets.get_downstream_dataset(
            catalog, d, is_train=False, transform=val_transform)
        templates = all_templates[d]
        labels = all_labels[d]

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=6, pin_memory=False, drop_last=False)

        metric = eval_zeroshot.evaluate(d, val_loader, templates, labels, model, tokenizer)
        metrics[d] = metric
        json_str = json.dumps({"task": d, "acc": metric})
        print(json_str)
        if not test:
            with open(output_fn, mode="a+", encoding="utf-8") as f:
                f.write(json_str + "\n")
    return metrics


def parse_results(results, result_json):
    with open(result_json) as f:
        slip_results = []
        headline = []
        for line in f:
            rec = json.loads(line.strip())
            if rec['task'] not in headline:
                headline.append(rec['task'])
            slip_results.append(rec['acc'])
        assert len(slip_results) == 26
    avg = (sum(slip_results) / len(slip_results)) / 100
    print('slip avg:', (sum(slip_results) / len(slip_results)) / 100)
    results['slip'] = avg


def main(model, preprocess_val, tokenizer, result_json):
    slip_evaluate(model, preprocess_val, tokenizer, 128, result_json)
