import torch
import mmap
import os
import json
import random
import numpy as np
import tarfile

from io import BytesIO
from pathlib import Path

from typing import Any, Callable, Optional

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.open_clip.tokenizer import tokenize
from src.training.distributed import world_info_from_env
from src.training.data import DataInfo


class Altogether_PT(torch.utils.data.IterableDataset):
    """
    Read data similar as `metaclip_wds.py` but output format used by `train_altogether.py:train_altogether`.

    The dataset are organized in the following structure:
        `<dataset_dir>/{shard_id % 100}/{shard_id}.tar`.
    Each tar contains files in the following order (WebDatadet compatible):
    ```
        uuid1.json
        uuid1.jpeg
        uuid2.json
        uuid2.jpeg
    ```
    Each json has a `text` field with a list of texts associated with the image (uuid):
    [
        ['alt', 'this is a caption.'],
        ['alt', 'this is another caption for the same image.'],
        ...
    ]
    """

    def __init__(self, args):
        self.args = args

        import torchvision.transforms.functional as F

        from src.open_clip.transform import _convert_to_rgb
        from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, CenterCrop

        self.transform = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),  # smaller edge of the image will be matched to this number
            CenterCrop(224),
            _convert_to_rgb,
            F.pil_to_tensor,
        ])

        from transformers import AutoTokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.tokenizer = AutoTokenizer.from_pretrained(args.clipcap_args["decoder"], use_fast=True)

        start, end = os.path.basename(args.train_data).split("{")[1].split("}")[0].split("..")
        self.num_shards = int(end) - int(start)
        self.start_shard_id = int(start)
        self.root_dir = os.path.dirname(args.train_data)
        self.positions = None

    def __len__(self):
        assert hasattr(self.args, "train_data_len")
        return self.args.train_data_len

    def set_positions(self, positions):
        self.positions = positions

    def _get_next_shard_id(self, shard_id):
        shard_id += self.worker_size
        return self.start_shard_id + (shard_id % self.num_shards)

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

    def __iter__(self):
        (global_rank, worker_id), num_workers, worker_size = self._get_worker_info()
        
        if self.positions is not None and self.positions[f"{global_rank}_{worker_id}"] != -1:
            shard_id = self.positions[f"{global_rank}_{worker_id}"]
            print(f"{global_rank}_{worker_id} restore {shard_id}")
            shard_id = self._get_next_shard_id(shard_id)
        else:
            shard_id = self.start_shard_id + (global_rank * num_workers + worker_id)

        while True:
            with tarfile.open(f"{self.root_dir}/{shard_id % 100}/{shard_id}.tar") as tar:
                img_uuid, json_uuid = None, None
                members = tar.getmembers()
                for member in members:
                    # read jpeg first and json next
                    if member.name.endswith(".jpeg"):
                        img_uuid = member.name[:-len(".jpeg")]
                        if img_uuid.startswith("./"):
                            img_uuid = img_uuid[len("./"):]
                        with tar.extractfile(member) as f:
                            img = f.read()

                    elif member.name.endswith(".json"):
                        json_uuid = member.name[:-len(".json")]
                        if json_uuid.startswith("./"):
                            json_uuid = json_uuid[len("./"):]
                        with tar.extractfile(member) as f:
                            text_json = json.load(f)
                    else:
                        print(f"unknown file ext {member.name}")
                        continue

                    if img_uuid is None or json_uuid is None:
                        continue

                    assert img_uuid == json_uuid, f"{img_uuid}-{json_uuid}"

                    alt_list = list(set([txt_tuple[1] for txt_tuple in text_json["texts"]]))
                    alt_text = "; ".join(alt_list)
                    
                    # for a list of alt_list: generate the longest one from other random one;
                    max_len, max_len_idx = 0, -1 
                    for alt_idx, alt in enumerate(alt_list):
                        if len(alt) > max_len:
                            max_len = len(alt)
                            max_len_idx = alt_idx

                    if len(alt_list) == 1:
                        alt_text = ""
                    else:
                        alt_text = random.choice([alt_list[alt_idx] for alt_idx in range(len(alt_list)) if alt_idx != max_len_idx])
                    txt = alt_list[max_len_idx]

                    with Image.open(BytesIO(img)) as img:
                        image = img.convert("RGB")
                        # assert "face_bbox" in text_json
                        # image = fairblurbbox(image, text_json["face_bbox"])
                        image = self.transform(image)
                    
                    pad_token_id = self.args.clipcap_args["pad_token_id"]
                    prefix_length = self.args.clipcap_args["prefix_length"]

                    tokens = self.tokenizer.encode(txt, add_special_tokens=False) + [self.tokenizer.eos_token_id]

                    if hasattr(self.args, "rewrite_prompt"):
                        prompts = [alt_text, ""]
                        prompt = random.choice(prompts)

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
                    if hasattr(self.args, "rewrite_prompt"):
                        mask = torch.zeros(self.args.max_seq_len - self.args.rewrite_prompt)
                    else:
                        mask = torch.zeros(self.args.max_seq_len)
                    mask[:-padding_len] = 1.

                    yield image, tokens, attention_mask, mask, txt, alt_text, worker_id, shard_id
                    img_uuid, json_uuid = None, None

            shard_id = self._get_next_shard_id(shard_id)
