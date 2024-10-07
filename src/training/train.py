# Copyright (c) Meta Platforms, Inc. and affiliates

import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict

from src.open_clip.loss import ClipLoss
from src.open_clip.transform import get_mean_std
from src.training.distributed import is_master, world_info_from_env
from src.training.zero_shot import zero_shot_eval
from src.training.checkpoint import save_checkpoint, agg_positions, collect_positions, unwrap_model
from src.training.precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def images_to_device(images, device):
    images = images.to(device=device, non_blocking=True)
    if images.dtype == torch.uint8:
        images = images.to(torch.float32).div_(255.)  # b, 3, 224, 224
        mean, std = get_mean_std()
        mean = torch.as_tensor(mean, device=images.device)[None, :, None, None]
        std = torch.as_tensor(std, device=images.device)[None, :, None, None]
        images.sub_(mean).div_(std)
    return images


def to_device(batch, device):
    if len(batch) == 2:
        images, texts = batch
    else:
        images, texts, metadata = batch
    images = images_to_device(images, device)
    texts = texts.to(device=device, non_blocking=True)
    if len(batch) == 2:
        return images, texts
    else:
        return images, texts, metadata


def build_loss(args):
    from src.open_clip import loss as loss_module
    loss_cls = getattr(loss_module, "ClipLoss")

    loss = loss_cls(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size)
    return loss


def backward(args, total_loss, scaler, optimizer, model):
    if torch.isfinite(total_loss).all():
        if scaler is not None:
            scaler.scale(total_loss).backward()
            # if args.world_size == 1:
            #    from src.training.detect import detect_unused_parameters
            #    detect_unused_parameters(model)
            if args.norm_gradient_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            # if args.world_size == 1:
            #    from src.training.detect import detect_unused_parameters
            #    detect_unused_parameters(model)
            # detect_nan(model, optimizer)
            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        if hasattr(unwrap_model(model), "logit_scale"):
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))
    else:
        logging.warn(f"Loss is {total_loss}, skip back prop.")
        import sys
        sys.exit(1)  # protect the checkpoint for debugging.


def train_one_epoch_ex(args, model, data, start_step, total_steps, optimizer, scaler, scheduler, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    model.train()

    loss = build_loss(args)

    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    epoch = start_step // num_batches_per_epoch
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    positions = torch.full((args.workers,), fill_value=-1, dtype=torch.long)

    batch_iter = iter(dataloader)

    for step in range(start_step, total_steps):
        batch = next(batch_iter)
        scheduler(step)

        if len(batch) == 2:
            (images, texts), worker_ids, shard_ids = batch, None, None
        else:
            images, texts, worker_ids, shard_ids = batch

        images, texts = to_device((images, texts), device)
        
        positions = agg_positions(positions, worker_ids, shard_ids)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            total_loss = loss(image_features, text_features, logit_scale)

        backward(args, total_loss, scaler, optimizer, model)

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = step + 1
        if is_master(args) and (step % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

        if hasattr(args, "save_steps") and (step + 1) % args.save_steps == 0:
            positions_dict = collect_positions(args, positions)
            if args.save_logs:
                save_checkpoint(f"{args.checkpoint_path}/epoch_latest.pt", model, optimizer, scaler, step, positions_dict=positions_dict)
    
        # TODO: copied from main.py, wrap as a function call.
        if hasattr(args, "eval_steps") and (step + 1) % args.eval_steps == 0: # TODO (huxu): put eval on master only?
            if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
                evaluate_ex(args, model, data, step, tb_writer)  # completed_epoch -> epoch, writer -> tb_writer
            model.train()  # evaluate won't turn model back to train.
            positions_dict = collect_positions(args, positions)
            if args.save_logs:
                save_checkpoint(f"{args.checkpoint_path}/epoch_latest.pt", model, optimizer, scaler, step, positions_dict=positions_dict)
    # end for
    positions_dict = collect_positions(args, positions)
    if is_master(args):
        positions_dict = collect_positions(args, positions)
        save_checkpoint(f"{args.checkpoint_path}/epoch_latest.pt", model, optimizer, scaler, step, positions_dict=positions_dict)


def evaluate_ex(args, model, data, step, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(args, model, data, 0)  # huxu: epoch = 0 as a trick to bypass checking.
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    if 'val' in data:  # and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):  # huxu: val anytime called.
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = to_device(batch, device)

                with autocast():
                    image_features, text_features, logit_scale = model(images, texts)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Step: {step} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "step": step, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Step: {step} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val_step/{name}", val, step)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
