# Copyright (c) Meta Platforms, Inc. and affiliates

import os
import json
import torch
import time
import tarfile

from pathlib import Path
from collections import defaultdict
from torch.nn import functional as F

from io import BytesIO
from PIL import Image

from tqdm import tqdm

from transformers import AutoTokenizer

import sys
sys.path.append(".")

from src.training.train import to_device
from src.open_clip.factory import create_model_and_transforms
from src.open_clip import model_altogether
from src.training.train_altogether import llm_decode
from src.training.checkpoint import load_checkpoint


def seeding(seed=0):
    import random
    import torch
    import numpy as np

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed=seed)


def load_model(args, checkpoint_path):
    seeding()

    clip_model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_name=args.model, pretrained=args.pretrained,
        precision=args.precision, device=args.clipcap_args["device"],
        force_quick_gelu=args.force_quick_gelu, clip_model=args.clip_model
    )

    clip_model = clip_model.eval()

    model_cls = getattr(model_altogether, args.cap_model)

    model = model_cls(**args.clipcap_args)

    model_to_load = model
    
    load_checkpoint(checkpoint_path, model_to_load)

    model = model.to(args.clipcap_args["device"])
    model = model.eval()
    
    from transformers import AutoTokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(args.clipcap_args["decoder"], use_fast=True)

    return (clip_model, model), tokenizer, preprocess_train, preprocess_val


class SingleShardIterativeWebDataset(torch.utils.data.IterableDataset):
    def __init__(self, args, data_path, shard_id, transform, tokenize):
        self.args = args
        self.root_dir = data_path
        self.transform = transform
        self.tokenizer = tokenize
        self.shard_id = shard_id
    
    def _get_tarball_path(self, shard_id):
        return f"{self.root_dir}/{shard_id % 100}/{shard_id}.tar"

    def __iter__(self):
        shard_id = self.shard_id

        # v2 374px is face blurred. iterate over all image-text pairs until needs legal mitigation;
        tarball_path = self._get_tarball_path(shard_id)

        with tarfile.open(tarball_path) as tar:        
            img_uuid, json_uuid = None, None
            members = tar.getmembers()
            # metaclip_v1 can be iterative but the paper uses mmap for random access.
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

                assert img_uuid == json_uuid
    
                with Image.open(BytesIO(img)) as img:
                    image = img.convert("RGB")
                    # assert "face_bbox" in text_json
                    # image = fairblurbbox(image, text_json["face_bbox"])
                    image = self.transform(image)
                
                pad_token_id = self.args.clipcap_args["pad_token_id"]
                prefix_length = self.args.clipcap_args["prefix_length"]

                alt_text = "; ".join(list(set([txt_tuple[1] for txt_tuple in text_json["texts"]])))

                prompt = alt_text

                prompt_input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)[:self.args.rewrite_prompt]
                padding_len = self.args.rewrite_prompt - len(prompt_input_ids)
                prompt_input_ids = [pad_token_id] * padding_len + prompt_input_ids
                tokens = prompt_input_ids
                tokens = torch.tensor(tokens, dtype=torch.long)
                
                yield image, tokens, img_uuid
                img_uuid, json_uuid = None, None


def get_dataloader(args, batch_size, data_path, shard_id, transform, tokenize):
    dataloader_args = dict(
        dataset = SingleShardIterativeWebDataset(args, data_path, shard_id, transform, tokenize),
        batch_size = batch_size,
        num_workers = 1,
        pin_memory = False,
        drop_last = False,
    )    
    return torch.utils.data.DataLoader(**dataloader_args)


def inference(args, batch_iter, model, clip_model, tokenizer):
    outputs = {}
    
    import time
    t0 = time.time()

    total_size = 0
    with torch.no_grad(), torch.cuda.amp.autocast():
        while True:
            try:
                batch = next(batch_iter)
            except StopIteration:
                print(f"qps: {total_size / (time.time() - t0)}")
                return outputs

            imgs, prompt_input_ids, uuids = to_device(batch, args.clipcap_args["device"])
            batch_size = imgs.size(0)
            prefix_length = args.clipcap_args["prefix_length"]
            pad_token_id = args.clipcap_args["pad_token_id"]

            assert hasattr(args, "rewrite_prompt")
            num_new_token = args.max_seq_len - args.rewrite_prompt

            image_features = clip_model.encode_image(imgs)
            image_features = F.normalize(image_features, dim=-1)

            embedding_image = model.clip_project(image_features).view(batch_size, args.clipcap_args["prefix_length"], -1)
            
            embedding_text = model.gpt.get_input_embeddings()(prompt_input_ids)
            embedding_cat = torch.cat((embedding_image, embedding_text), dim=1)

            gen_ids = model.gpt.generate(
                inputs_embeds=embedding_cat,
                max_new_tokens=num_new_token,
                temperature=0.2,
                do_sample=True,
                top_p=0.7,
                use_cache = True,
            )

            cap_strs = llm_decode(tokenizer, gen_ids, remove_new_line=True)
            cap_strs = [cap_str.split(tokenizer.eos_token)[0] for cap_str in cap_strs]
            
            for img_id, cap_str in enumerate(cap_strs):
                uuid = uuids[img_id]
                outputs[uuid] = {"altogether": f"{cap_strs[img_id]}"}
            total_size += batch_size


def main(config_name, checkpoint_name, batch_size, data_path, cap_path, todo):
    
    import sys
    sys.path.append("./")

    from configs import search_config
    
    args = search_config(config_name)
    args.distributed = False
    args.clipcap_args["device"] = torch.device('cuda')
    
    checkpoint_path = f"logs/{args.__class__.__name__}/checkpoints/{checkpoint_name}"

    print(f"m: {checkpoint_path}")
    (clip_model, model), tokenizer, _, transform = load_model(args, checkpoint_path)

    for shard_id in todo:

        if not Path(f"{data_path}/{shard_id % 100}/{shard_id}.tar").exists():
            continue

        shard_folder = f"{cap_path}/{shard_id % 100}"

        Path(shard_folder).mkdir(parents=True, exist_ok=True)

        shard_json = f"{shard_folder}/{shard_id}.json"
        if Path(shard_json).exists():
            continue

        batch_iter = iter(get_dataloader(args, batch_size, data_path, shard_id, transform, tokenizer))

        outputs = inference(args, batch_iter, model, clip_model, tokenizer)

        with open(shard_json, "w") as fw:
            json.dump(outputs, fw)


if __name__ == "__main__":
    # python altogether/infer.py altogether:epoch_ft.pt <your_wds_path> <output_path>

    import sys
    assert len(sys.argv) == 4
    
    config_name, data_path, cap_path = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f"config_name={config_name}", f"data_path={data_path}", f"cap_path={cap_path}")

    batch_size = 128
    
    if ":" in config_name:
        config_name, checkpoint_name = config_name.split(":")
    else:
        checkpoint_name = "epoch_ft.pt"
    
    if False:  # testing
        main(config_name, checkpoint_name, batch_size, data_path, cap_path, [0])
    else:
        import os
        import submitit
        import sys

        job_plans = [
            ("my_partition", (0, 200000), 256),
        ]
        
        for partition, (start_shard, end_shard), world_size in job_plans:
            todo = []
            for shard_id in range(start_shard, end_shard):
                shard_json = f"{cap_path}/{shard_id % 100}/{shard_id}.json"
                if Path(shard_json).exists():
                    continue
                if not Path(f"{data_path}/{shard_id % 100}/{shard_id}.tar").exists():
                    continue
                todo.append(shard_id)
            print(len(todo))

            params = dict(
                name=f"gen_cap",
                gpus_per_node=1,
                mem_gb=4,
                cpus_per_task=2,  # one for dataloader one for main loop.
                nodes=1,
                slurm_partition='my_partition',
                timeout_min=4320,
            )

            executor = submitit.AutoExecutor(
                folder="submitit/%j"
            )
            executor.update_parameters(**params)

            import math
            jobs = []
            with executor.batch():
                shards_per_rank = math.ceil(len(todo) / world_size)
                for rank in range(world_size):
                    job = executor.submit(
                        main, config_name, checkpoint_name, batch_size, data_path, cap_path, 
                        todo[int(shards_per_rank * rank):int(shards_per_rank * (rank+1))]
                    )
                    jobs.append(job)
            if len(jobs) > 0:
                print(partition, len(jobs), jobs[0].job_id, jobs[-1].job_id)
