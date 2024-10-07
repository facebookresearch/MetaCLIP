import torch
import os
import json
import random
import numpy as np
import tarfile
import mmap

import torchvision.transforms.functional as F

from io import BytesIO
from pathlib import Path

from typing import Any, Callable, Optional
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, CenterCrop

from PIL import Image

from src.open_clip.tokenizer import tokenize

from src.training.distributed import world_info_from_env
from src.training.data import DataInfo
from src.open_clip.transform import _convert_to_rgb


def pad_tokens(tokens: int, prefix_length, pad_token_id, max_seq_len = 77):
    padding = max_seq_len - tokens.shape[0]
    if padding > 0:
        tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))  # padding is negative.
    elif padding < 0:
        tokens = tokens[:max_seq_len]
    mask = tokens.ge(0)  # mask is zero where we out of sequence
    tokens[~mask] = pad_token_id
    mask = mask.float()
    if prefix_length > 0:
        mask = torch.cat((torch.ones(prefix_length), mask), dim=0)  # adding prefix mask
    return tokens, mask


class Altogether_FT(torch.utils.data.IterableDataset):
    """
    Altogether finetuning set format:
    A JSON file contains a list of dict w/ the following keys:
    `round3` round 3 annotation;
    `url`: url of the image;
    `img_path`: url of the image;
    `alt`: original alt text;
    `source`: source of the image, either `wit` or `datacomp`.
    """

    def __init__(self, args):
        self.args = args

        # no centercropping to ensure all information aligned in annotated dataset.
        self.transform = Compose([
            Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            _convert_to_rgb,
            F.pil_to_tensor,
        ])

        from transformers import AutoTokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.tokenizer = AutoTokenizer.from_pretrained(args.clipcap_args["decoder"], use_fast=True)

        with open(args.endsft_train_data) as f:
            records = json.load(f)["data"]

        self.records = []
        for record in records:
            self.records.append(record)

    def __len__(self):
        return len(self.records)

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

    def _get_next_idx(self, idx):
        idx += self.worker_size
        return idx % len(self)

    def __iter__(self):
        (global_rank, worker_id), num_workers, worker_size = self._get_worker_info()
        idx = global_rank * num_workers + worker_id
        while True:
            yield self[idx]
            idx = self._get_next_idx(idx)

    def __getitem__(self, idx):
        rec = self.records[idx]

        with Image.open(rec["img_path"]) as img:
            image = img.convert("RGB")
            from src.training.fb import fairblurbbox
            image = fairblurbbox(image, rec["face_bbox"])

            image = self.transform(image)

        pad_token_id = self.args.clipcap_args["pad_token_id"]

        prefix_length = self.args.clipcap_args["prefix_length"]

        tokens = self.tokenizer.encode(rec[self.args.endsft_response], add_special_tokens=False) + [self.tokenizer.eos_token_id]
        
        if hasattr(self.args, "rewrite_prompt") and self.args.rewrite_prompt:
            prompt = random.choice([rec["alt"], ""])
            
            prompt_input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)[:self.args.rewrite_prompt]
            padding_len = self.args.rewrite_prompt - len(prompt_input_ids)
            prompt_input_ids = [pad_token_id] * padding_len + prompt_input_ids
            tokens = prompt_input_ids + tokens

        tokens = tokens[:self.args.max_seq_len]
        padding_len = max(0, self.args.max_seq_len - len(tokens))
        tokens += [pad_token_id] * padding_len
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        attention_mask = torch.zeros(prefix_length + len(tokens))
        attention_mask[:-padding_len] = 1.

        # target mask
        if hasattr(self.args, "rewrite_prompt") and self.args.rewrite_prompt:
            mask = torch.zeros(self.args.max_seq_len - self.args.rewrite_prompt)
        else:
            mask = torch.zeros(self.args.max_seq_len)
        mask[:-padding_len] = 1.

        return image, tokens, attention_mask, mask, rec[self.args.endsft_response], rec["alt"], 0, 0
