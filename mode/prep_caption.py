# Copyright (c) Meta Platforms, Inc. and affiliates

import sys
sys.path.append("src")
sys.path.append("./")
import os
import random

from tqdm import tqdm
from multiprocessing import Pool
from get_prep_parser import get_args_parser, get_default_paths

import json
import argparse
import tarfile

import pdb

def get_tarfile_path(wds_dir,tar_id):
    # File Organization
    return os.path.join(wds_dir, str(tar_id % 100), f'{tar_id}.tar')


def gather_caption_from_tarfile(root_dir, tar_id):
    tarball_path = get_tarfile_path(root_dir, tar_id)

    if not os.path.exists(tarball_path):
        return None, 'file not exists'

    captions,tmems,imems = [],[],[]
    with tarfile.open(tarball_path) as tar:
        members = tar.getmembers()
        
        json_cnt = 0
        json_mid,img_mid = -1,-1
        iuid, uuid = None, None
        for midx, member in enumerate(members):
            if member.name.endswith(".json"):
                uuid = member.name[:-len(".json")]
                with tar.extractfile(member) as f:
                    text_json = json.load(f)

                if 'demo' in root_dir:
                    txt = random.choice(text_json["texts"])
                else:
                    raise ValueError('Please Implement by yourself and uncomment this line in prep_caption.py')
                json_cnt += 1
                json_mid = midx

            if member.name.endswith(".jpeg") or member.name.endswith(".jpg"):
                suffix = len(member.name.split('.')[-1]) + 1
                iuid = member.name[:-suffix]
                img_mid = midx

            if iuid is not None and iuid == uuid:
                if txt is None or len(txt)==0 or txt in ['"',]:
                    continue
                if json_mid in tmems or img_mid in imems:
                    continue
                captions.append(txt)
                tmems.append(json_mid)
                imems.append(img_mid)
    if len(set(tmems)) == len(set(imems)) and len(set(imems)) == len(captions):
        return {'tmems':tmems, 'imems':imems, 'caption':captions}, 'success'
    else:
        return None, 'fail'


def build_caption(wds_dir, shard_id, caption_dir, overwrite=True):
    shard_folder = shard_id % 100 
    os.makedirs(os.path.join(caption_dir, f"{shard_folder}"), exist_ok=True)

    output_fn_group = os.path.join(caption_dir, f"{shard_folder}", f"{shard_id}_caption.json")
    if os.path.exists(output_fn_group) and not overwrite:
        return True
    
    data, status = gather_caption_from_tarfile(wds_dir, shard_id)
    
    if status == 'success':
        with open(output_fn_group,'w') as json_file:
            json.dump(data, json_file)
        print('Newly write {} with {} items'.format(shard_id, len(data['caption'])))
        return True
    else:
        return None


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
            missing_shards.append(shard_id)
            continue

        status = build_caption(
            wds_dir, shard_id, args.caption_dir, overwrite=False,
        )
        if status:
            pass
        elif status is None:
            missing_shards.append(shard_id)
        else:
            raise ValueError('No Implementation Error')

    return missing_shards


def main(args):
    global wds_dir

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
    wds_dir = os.path.dirname(args.root)

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
    print("missing file", len(all_results), all_results)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Clustering evaluation', parents=[get_args_parser()])
    config = parser.parse_args()

    paths = get_default_paths()[config.dataset]
    config.root = paths['root']
    config.caption_dir = paths['caption']

    config.num_threads = 40
    if config.tar_end == -1:
        config.tar_end = int(os.path.basename(config.root).split("{")[1].split("}")[0].split("..")[1])
    
    os.makedirs(config.caption_dir, exist_ok=True)
    main(config)
