# Copyright (c) Meta Platforms, Inc. and affiliates

import json
import logging
import os
import pathlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple

import torch

from src.open_clip import model as model_zoo
from src.open_clip.model import CLIP, convert_weights_to_fp16, resize_pos_embed
from src.open_clip.openai import load_openai_model
from src.open_clip.pretrained import get_pretrained_url, download_pretrained
from src.open_clip.transform import image_transform
from src.training.checkpoint import load_checkpoint


_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def create_model(
    model_name: str,
    pretrained: str = '',
    precision: str = 'fp32',
    device: torch.device = torch.device('cpu'),
    jit: bool = False,
    force_quick_gelu: bool = False,
    pretrained_image: bool = False,
    clip_model: str = "CLIP",
):
    model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names

    if pretrained.lower() == 'openai':
        logging.info(f'Loading pretrained {model_name} from OpenAI.')
        model = load_openai_model(model_name, device=device, jit=jit)
        # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
        if precision == "amp" or precision == "fp32":
            model = model.float()
    else:
        if model_name in _MODEL_CONFIGS:
            logging.info(f'Loading {model_name} model config.')
            model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
        else:
            logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
            raise RuntimeError(f'Model config for {model_name} not found.')

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if pretrained_image:
            if 'timm_model_name' in model_cfg.get('vision_cfg', {}):
                # pretrained weight loading for timm models set via vision_cfg
                model_cfg['vision_cfg']['timm_model_pretrained'] = True
            else:
                assert False, 'pretrained image towers currently only supported for timm models'

        import importlib
        for model_code in os.listdir(f"src/open_clip"):
            if not model_code.endswith("model.py"):
                continue
            module_name = "src.open_clip." + model_code[:-len(".py")]
            module = importlib.import_module(module_name)
            if hasattr(module, clip_model):
                model_cls = getattr(module, clip_model)
                model = model_cls(**model_cfg)
                break
        else:
            raise ValueError(f"{clip_model} not found with *model.py")

        if pretrained:
            checkpoint_path = ''
            url = get_pretrained_url(model_name, pretrained)
            if url:
                checkpoint_path = download_pretrained(url)
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained

            if checkpoint_path:
                logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
                load_checkpoint(checkpoint_path, model, resize_pos_embed=True)
            else:
                logging.warning(f'Pretrained weights ({pretrained}) not found for model {model_name}.')
                raise RuntimeError(f'Pretrained weights ({pretrained}) not found for model {model_name}.')

        model.to(device=device)
        if precision == "fp16":
            assert device.type != 'cpu'
            convert_weights_to_fp16(model)

        if jit:
            model = torch.jit.script(model)

    return model


def create_model_and_transforms(
    model_name: str,
    pretrained: str = '',
    precision: str = 'fp32',
    device: torch.device = torch.device('cpu'),
    jit: bool = False,
    force_quick_gelu: bool = False,
    pretrained_image: bool = False,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    gpu_trans = False,
    clip_model: str = "CLIP",
):
    model = create_model(
        model_name, pretrained, precision, device, jit,
        force_quick_gelu=force_quick_gelu,
        pretrained_image=pretrained_image,
        clip_model=clip_model,
    )
    preprocess_train = image_transform(model.visual.image_size, is_train=True, mean=mean, std=std, gpu_trans=gpu_trans)
    preprocess_val = image_transform(model.visual.image_size, is_train=False, mean=mean, std=std)
    return model, preprocess_train, preprocess_val


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()
