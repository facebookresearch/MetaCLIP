# Copyright (c) Meta Platforms, Inc. and affiliates

# usage:
# python src/training/main.py altogether  # local
# torchrun --nproc_per_node=8 src/training/main.py altogether  # one node
# python submit.py altogether  # slurm job


from dataclasses import dataclass
from configs import Config


@dataclass
class altogether(Config):
    gpu_trans = True
    train_data = "data/metaclip_v1_2_5B/{0..200000}.tar"
    train_data_len = 22_807_024
    dataset_cls = "Altogether_PT"

    clipcap_args = {
        "prefix_length": 40, "pad_token_id": 1, "clip_length": 40, "clip_emb_size": 1024,
        "decoder": "facebook/opt-1.3b", "use_flash_attention_2": True,
    }
    
    model = "ViT-H-14-quickgelu"
    force_quick_gelu = True
    pretrained = "metaclip_2_5b"

    cap_model = "Altogether"

    max_seq_len = 320
    rewrite_prompt = 128

    batch_size = 32
    epochs = 1
    workers = 6
    
    engine = "train_altogether"
    lr = 2e-4
    warmup = 200
    min_ratio = 0.1
    seed = 0

    save_steps = 2000
    ngpus = 8
    nodes = 2

    endsft_train_data = "your_ft_dataset"
    endsft_response = "your_response_key"
    endsft_dataset_cls = "Altogether_FT"
    endsft_epochs = 2
