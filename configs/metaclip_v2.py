from dataclasses import dataclass
from configs import Config


@dataclass
class h14_worldwide(Config):
    local_loss = True
    gather_with_grad = True

    torchcompile=True
    gpu_trans=True
    engine="train_one_epoch_ex"
    eval_steps = 5000
    save_frequency = 1
    save_steps = 2000
    workers = 6
    
    dataset_cls = "IterativeWebDatasetWorldWide"
    train_data = f"data/metaclip2_tar/" + "{0..400000}.tar"
    pkg_json = f"data/metaclip2_json"
    entry_prob_dir = f"data/metaclip2_prob"

    t = 170000

    train_num_samples = 920_000_000
    batch_size = 196
    grad_checkpointing = True

    epochs = 32
    model = "ViT-H-14-quickgelu-worldwide@WorldWideCLIP"
    name = "ViT-H-14-quickgelu-worldwide"
    tokenizer = "facebook/xlm-v-base"

    force_quick_gelu = True

    lr = 0.0004
    beta2 = 0.95
    wd = 0.1
    warmup = 2000
    precision = "amp_bf16"
    nodes = 48
    ngpus = 8
