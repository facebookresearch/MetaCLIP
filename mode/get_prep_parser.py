# Copyright (c) Meta Platforms, Inc. and affiliates

import argparse

PATH_TO_DEMO = '/demo'
def get_default_paths():
    mm = {
        'root':'data/mm/{0..200000}.tar',
        'caption':f'{PATH_TO_DEMO}/caption_mm/',
        'feature':f'{PATH_TO_DEMO}/feature_mm/',
        'assign':f'{PATH_TO_DEMO}/assign_mm/',
        'cluster':f'{PATH_TO_DEMO}/cluster_center/meta/',
    }
    return {'mm': mm}

def get_args_parser():
    parser = argparse.ArgumentParser(description='MoDE Data Preparation', add_help=False)
    parser.add_argument('--dataset', default='clipeval', type=str, choices=['clipeval', 'mm'])
    parser.add_argument('--root', default="data/mm/{0..200000}.tar", type=str,
                        help='path to dataset root')
    parser.add_argument('--caption-dir', default='demo/caption_mm/', type=str, help='caption dir, highly recommended')
    parser.add_argument('--feature-dir', default='demo/feature_mm/', type=str, help='feature output dir')
    
    # Below arguments are only for pre-processing pre-train data on feature extraction 
    parser.add_argument('--file-mode', default='tarfile', type=str, choices=['caption', 'tarfile'],
                        help='processing extracted captions or tarfiles direction')
    parser.add_argument('--tar-init', default=0, type=int, help='tarfile_id to start')
    parser.add_argument('--tar-end', default=-1, type=int, help='tarfile_id to end')
    parser.add_argument('--tar-per-gpu', default=-1, type=int, help='number of tarfiles to process per GPU')
    parser.add_argument('--chunk-size', default=400, type=int, help='number of captions to be processed')
    parser.add_argument('--horovod', default=False, type=bool, help='placeholder, needed to pass ddp initialization')
    parser.add_argument('--dist-url', default="env://", type=str, help='placeholder, needed to pass ddp initialization')
    parser.add_argument('--dist-backend', default="nccl", type=str, help='placeholder, needed to pass ddp initialization')
    parser.add_argument('--no-set-device-rank', default=False, type=bool, help='placeholder, needed to pass ddp initialization')

    # Arguments on clustering and assignment
    parser.add_argument('--cm', default=1024, type=int, help='number of fine-grained cluster centers')
    parser.add_argument('--cn', default=4, type=int, help='number of coarse-grained cluster centers')
    parser.add_argument('--cd', default='euclidean', type=str, help='cluster distance, euc or cos')
    parser.add_argument('--cassign-dir', default='demo/assign_meta/', type=str, help='feature output dir')
    parser.add_argument('--ccenter-dir', default='demo/cluster_center/', type=str, help='cluster center')

    # Arguments on intermediate variables at inference time
    parser.add_argument('--logits-dir', default='./logs/clip_eval', type=str, help='cluster center')
    return parser