# Copyright (c) Meta Platforms, Inc. and affiliates

import math
import torch
import torch.nn.functional as F

import json
import os

from tqdm import tqdm
from collections import defaultdict


@torch.no_grad()
def slip_evaluate(args, model, val_transform, tokenizer):
    from clipeval import datasets, eval_zeroshot

    catalog, all_templates, all_labels = eval_zeroshot.load_metadata("clipeval")

    if hasattr(model, "module"):
        model = model.module

    metrics = {}
    for d in catalog:
        val_dataset = datasets.get_downstream_dataset(
            catalog, d, is_train=False, transform=val_transform)
        templates = all_templates[d]
        labels = all_labels[d]

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size//2, shuffle=False,
            num_workers=args.workers, pin_memory=False, drop_last=False)

        metric = eval_zeroshot.evaluate(d, val_loader, templates, labels, model, tokenizer)
        metrics[d] = metric
        json_str = json.dumps({"task": d, "acc": metric})
        if args.rank == 0:
            print(json_str)
            with open(os.path.join(args.output_dir, "slip.txt"), mode="a+", encoding="utf-8") as f:
                f.write(json_str + "\n")
    return metrics
