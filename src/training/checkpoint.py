# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
import logging

from src.open_clip.model import resize_pos_embed as _resize_pos_embed


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def unwrap_state_dict(sd):
    if next(iter(sd.items()))[0].startswith('_orig_mod'):
        sd = {k[len('_orig_mod.'):]: v for k, v in sd.items()}
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    return sd


def load_checkpoint(checkpoint_path, model, map_location='cpu', resize_pos_embed=False, strict=True, optimizer=None, scaler=None):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    step, positions = -1, None

    if isinstance(checkpoint, dict):
        state_dict = unwrap_state_dict(checkpoint["state_dict"])
        if resize_pos_embed:
            _resize_pos_embed(state_dict, model)

        model.load_state_dict(state_dict, strict=strict)

        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])

        if "step" in checkpoint:
            step = checkpoint["step"]

        if "positions" in checkpoint:
            positions = checkpoint["positions"]
    else:
        # loading a bare (model only) checkpoint for fine-tune or evaluation
        model.load_state_dict(unwrap_state_dict(checkpoint))
    return step, positions


def save_checkpoint(checkpoint_path, model, optimizer=None, scaler=None, step=None, positions_dict=None):
    checkpoint_dict = {
        "step": step,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    if scaler is not None:
        checkpoint_dict["scaler"] = scaler.state_dict()

    if positions_dict is not None:
        checkpoint_dict["positions"] = positions_dict

    # Saving checkpoints. use eval_steps to save a checkpoint.
    torch.save(checkpoint_dict, checkpoint_path)


def agg_positions(positions, worker_ids, shard_ids):
    if positions is None or worker_ids is None or shard_ids is None:
        return None
    assert sum(worker_ids) == worker_ids[0] * worker_ids.shape[0]  # pt dataloader should iter over worker for each batch;
    positions[worker_ids[0]] = shard_ids.max()
    return positions


def collect_positions(args, positions):
    if positions is None:
        return None
    if args.distributed:
        import torch.distributed as dist
        from src.training.distributed import world_info_from_env

        _, _, world_size = world_info_from_env()

        gathered_tensors = [torch.zeros_like(positions, device=args.device) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, positions.to(args.device))
    else:
        gathered_tensors = [positions]
    gathered_tensors = [gathered_tensor.cpu() for gathered_tensor in gathered_tensors]
    positions = {f"{rank}_{worker_id}": shard_id for rank, gathered_tensor in enumerate(gathered_tensors) for worker_id, shard_id in enumerate(gathered_tensor)}
    return positions
