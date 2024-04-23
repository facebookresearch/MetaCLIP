# Copyright (c) Meta Platforms, Inc. and affiliates

import sys
sys.path.append("src")
sys.path.append("./")
import os

import torch
import torch.nn.functional as F
import argparse
from get_prep_parser import get_args_parser, get_default_paths
from prep_caption import get_tarfile_path

import json,pdb

from multiprocessing import Pool
from tqdm import tqdm


def build_assignment(feat_dir, shard_id, assign_dir, overwrite=True):

    shard_folder = shard_id % 100 
    output_fn_group = os.path.join(assign_dir, f"{shard_folder}", f'{shard_id}_assign_dist.json')
    os.makedirs(os.path.dirname(output_fn_group), exist_ok=True)
    
    feat_fn = os.path.join(feat_dir, f"{shard_folder}", f'{shard_id}_feat.pth')
    if not os.path.exists(feat_fn):
        print(feat_fn, 'Not Found')
        return None
    
    if os.path.exists(output_fn_group) and not overwrite:
        print(f'{output_fn_group} Written already')
        return True
    
    feature = torch.load(feat_fn, map_location='cpu')
    assign = {'key':feature['filekeys'],'image':feature['img_midx']}
    feature = F.normalize(feature['feat'],dim=-1) 

    for key,ccenter in ccenters.items():
        if key[0] == 'E': # for euclidean
            dist = torch.cdist(feature[None], ccenter[None])[0]
            min_dist,assign_tensor = dist.min(dim=-1)
            min_dist = min_dist.numpy().tolist()
        elif key[0] == 'C': # for cosine
            sim = torch.mm(feature, ccenter.T)
            max_sim,assign_tensor = sim.max(dim=-1)
            min_dist = (1.0 - max_sim).numpy().tolist()
        # add "'dist':min_dist" in the dict if needed
        assign[key] = {'assign':assign_tensor.numpy().tolist()}

    with open(output_fn_group,'w') as json_file:
        json.dump(assign, json_file)
    print('Newly written', shard_id)
    return assign


def func(args, _start, _end):
    missing_shards = []

    if isinstance(_start, list):
        warc_iter = _start
    else:
        warc_iter = (
            tqdm(range(_start, _end)) if _start == 0 else range(_start, _end)
        )

    for idx, shard_id in enumerate(warc_iter):

        wds_fn = get_tarfile_path(wds_dir, shard_id)
        if not os.path.exists(wds_fn):
            continue

        status = build_assignment(
            args.feature_dir, shard_id, args.cassign_dir, overwrite=False,
        )
        if status:
            pass
        elif status is None:
            missing_shards.append(shard_id)
        else:
            raise ValueError('No Implementation Error')

    return missing_shards


def main(args):

    shard_ids = [[] for _ in range(args.num_threads)]
    for shard_id in range(args.tar_init, args.tar_end):
        group_offset = shard_id % args.num_threads
        shard_ids[group_offset].append(shard_id)
    
    print(f"shard_ids[0]={shard_ids[0]}")
    starts = shard_ids
    ends = [None for _ in range(len(starts))]

    argss = [args for _ in range(len(starts))]
    assert len(argss) == len(starts) == len(ends)
    assert len(starts) <= args.num_threads

    global wds_dir
    wds_dir = os.path.dirname(args.root)

    global ccenters
    ccenters = {} 
    # MoDE originally uses euclidean dist in clustering
    # The dict structure below provides flexibility 
    # for cluster assginment with different cm and and cosine
    for dist_type in ['euclidean']: 
        for cm in [args.cm,]:
            path = os.path.join(args.ccenter_dir,dist_type,f'F{cm}.pth')
            if os.path.exists(path):
                key = '{}{}'.format(dist_type[0].upper(),args.cm)
                ccenters[key] = torch.load(path)['center']
                if 'cos' in dist_type:
                    ccenters[key] = F.normalize(ccenters[key],dim=-1)

    with Pool(len(starts)) as p:
        results = p.starmap(
            func,
            zip(
                argss,
                starts,
                ends
            ),
        )

    all_results = []
    for result in results:
        all_results.extend(result)
    print("missing npy", len(all_results))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Clustering evaluation', parents=[get_args_parser()])
    config = parser.parse_args()

    paths = get_default_paths()[config.dataset]
    config.root = paths['root']
    config.feature_dir = paths['feature']
    config.cassign_dir = paths['assign']
    config.ccenter_dir = paths['cluster']

    config.num_threads = 40
    if config.tar_end == -1:
        config.tar_end = int(os.path.basename(config.root).split("{")[1].split("}")[0].split("..")[1])
    
    os.makedirs(config.cassign_dir, exist_ok=True)
    main(config)
