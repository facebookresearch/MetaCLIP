# Copyright (c) Meta Platforms, Inc. and affiliates

import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import collections
from collections import defaultdict

from transformers import AutoTokenizer

from src.open_clip.transform import get_mean_std
from src.open_clip.tokenizer import tokenize as clip_tokenizer
from src.training.distributed import is_master, world_info_from_env
from src.training.zero_shot import zero_shot_eval
from src.training.precision import get_autocast
from src.training.train import to_device, images_to_device, backward, AverageMeter
from src.training.checkpoint import unwrap_model, save_checkpoint, agg_positions, collect_positions


def create_captioner(args, model, device):
    assert hasattr(args, "clipcap_args")

    clip_model = model

    import importlib  # TODO: wrap as a file search func.
    for model_code in os.listdir(f"src/open_clip"):
        if model_code.endswith(".py") and "model" in model_code:
            module = importlib.import_module("src.open_clip." + model_code[:-len(".py")])
            if hasattr(module, args.cap_model):
                model_cls = getattr(module, args.cap_model)
                break
    else:
        raise ValueError(f"{args.cap_model} not found.")
    
    args.clipcap_args["device"] = device
    model = model_cls(**args.clipcap_args)
    model = model.to(device)
    return clip_model, model


def train_altogether(args, model, data, start_step, total_steps, optimizer, scaler, scheduler, tb_writer=None):
    clip_model, model = model

    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    clip_model.eval()
    model.train()

    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches

    llm_tokenizer = dataloader.dataset.tokenizer

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    positions = torch.full((args.workers,), fill_value=-1, dtype=torch.long)

    prefix_length = args.clipcap_args["prefix_length"]  # this includes prompts
    pad_token_id = args.clipcap_args["pad_token_id"]

    if hasattr(args, "rewrite_prompt"):
        num_new_token = args.max_seq_len - args.rewrite_prompt
    else:
        num_new_token = args.max_seq_len

    batch_iter = iter(dataloader)
    
    for step in range(start_step, total_steps):
        if "endsft" in data and step == (total_steps - (data["endsft"].dataloader.num_batches * args.endsft_epochs)):
            logging.info(f"begin of endsft")
            del batch_iter, dataloader.dataset, dataloader
            batch_iter = iter(data["endsft"].dataloader)  # switch to SFT mode;
        
        batch = next(batch_iter)  # set_epoch made a new shuffled list based on epoch+step.
        scheduler(step)

        images, tokens, attention_masks, masks, raw_texts, raw_alt_texts, worker_ids, shard_ids = batch

        if hasattr(args, "rewrite_prompt"):
            assert masks.size(1) < tokens.size(1)
            assert torch.all(masks.sum(-1) < masks.size(1))

        raw_texts = list(raw_texts)
        raw_alt_texts = list(raw_alt_texts)
        assert isinstance(raw_texts[0], str) and isinstance(raw_alt_texts[0], str)

        positions = agg_positions(positions, worker_ids, shard_ids)
        
        batch_size = images.size(0)

        images = images_to_device(images, device)
        tokens = tokens.to(device)
        attention_masks = attention_masks.to(device)
        masks = masks.to(device)
        
        data_time_m.update(time.time() - end)

        optimizer.zero_grad()
        
        # step 1: get image clip_emb
        with torch.no_grad(), autocast():
            image_features = clip_model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            
            if is_master(args) and ((step % 1000 == 0) or (step + 1) == total_steps):
                gen_sample0(
                    args, model, llm_tokenizer, image_features, tokens, num_new_token, raw_texts, raw_alt_texts, prefix_length, device)
                logging.info(f"tokens {tokens.size()}, [0]={str(tokens[0])}")
                logging.info(f"attention_masks {attention_masks.size()}, [0]={str(attention_masks[0])}")
                logging.info(f"masks {masks.size()}, [0]={str(masks[0])}")

        with autocast():
            # attention_mask contains masks for visual tokens, but tokens is text only; max_seq_len is text part only.
            logits = model(tokens[:, :-1], image_features, attention_masks[:, :-1]).logits[:, -num_new_token:]
            total_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens[:, -num_new_token:].flatten(), reduction="none")
            masks = masks.flatten()
            total_loss = total_loss * masks
            total_loss = total_loss.sum() / (masks.sum() + 1e-6)
        
        backward(args, total_loss, scaler, optimizer, model)

        batch_time_m.update(time.time() - end)
        end = time.time()
        
        if is_master(args) and step % 200 == 0:
            loss_m.update(total_loss.item(), batch_size)

            percent_complete = (step / total_steps) * 100
            logging.info(
                f"step: {step}/{total_steps}({percent_complete:.0f}%)), "
                f"loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}), "
                f"data_time: {data_time_m.avg:.3f}, "
                f"batch_time: {batch_time_m.avg:.3f}({args.batch_size*args.world_size / batch_time_m.val:#g}/s), "
                f"lr: {optimizer.param_groups[0]['lr']:5f}, "
            )
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": args.batch_size*args.world_size / batch_time_m.val,
                "lr": optimizer.param_groups[0]['lr'],
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
            
            batch_time_m.reset()
            data_time_m.reset()

        if hasattr(args, "save_steps") and (step + 1) % args.save_steps == 0:
            positions_dict = collect_positions(args, positions)
            if is_master(args):
                model_to_save = model
                save_checkpoint(f"{args.checkpoint_path}/epoch_latest.pt", model, optimizer, scaler, step, positions_dict=positions_dict)

        if "endsft" in data and (step + 1) == (total_steps - (data["endsft"].dataloader.num_batches * args.endsft_epochs)): 
            # end for
            positions_dict = collect_positions(args, positions)
            if is_master(args):
                model_to_save = model
                save_checkpoint(f"{args.checkpoint_path}/epoch_pt.pt", model, optimizer, scaler, step, positions_dict=positions_dict)
    
    # end for
    positions_dict = collect_positions(args, positions)
    if is_master(args):
        model_to_save = model
        save_checkpoint(f"{args.checkpoint_path}/epoch_ft.pt", model, optimizer, scaler, step, positions_dict=positions_dict)


def llm_decode(llm_tokenizer, gen_ids, remove_new_line=True):
    gen_decoded_strs = []
    for _gen_ids in gen_ids:
        gen_decoded_str = llm_tokenizer.decode(_gen_ids, skip_special_tokens=True)
        if remove_new_line:
            gen_decoded_str = gen_decoded_str.replace("\n", " ")
        gen_decoded_strs.append(gen_decoded_str)
    return gen_decoded_strs


def gen_sample0(args, model, llm_tokenizer, image_features, tokens, num_new_token, raw_texts, raw_alt_texts, prefix_length, device):
    model = unwrap_model(model)
    batch_size = tokens.size(0)

    embedding_image = model.clip_project(image_features).reshape(batch_size, -1, model.gpt_embedding_size)

    # add rewrite_prompt and can be integrated into interactive_*.py
    if hasattr(args, "rewrite_prompt"):
        embedding_text = model.gpt.get_input_embeddings()(tokens[:, :args.rewrite_prompt])
        embedding_cat = torch.cat((embedding_image, embedding_text), dim=1)
    else:
        embedding_cat = embedding_image

    gen_ids = model.gpt.generate(
        inputs_embeds=embedding_cat,
        max_new_tokens=num_new_token,
        temperature=0.2,
        do_sample=True,
        top_p=0.7,
    )

    print(gen_ids)
    
    gen_strs = llm_decode(llm_tokenizer, gen_ids, remove_new_line=True)
    assert len(gen_strs) == len(raw_texts)

    logging.info(f"[alt]{raw_alt_texts[0]}")
    logging.info(f"[gt]{raw_texts[0]}")
    logging.info(f"[gen]{gen_strs[0]}")
