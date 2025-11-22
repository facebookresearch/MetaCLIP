# Copyright (c) Meta Platforms, Inc. and affiliates

import argparse

PATH_TO_DEMO = '/demo'
def get_default_paths():
    demo = {
        'root':'data/demo/{0..200000}.tar',
        'caption':f'{PATH_TO_DEMO}/caption/',
        'feature':f'{PATH_TO_DEMO}/feature/',
        'assign':f'{PATH_TO_DEMO}/assign/',
        'cluster':f'{PATH_TO_DEMO}/cluster_center/',
    }
    return {'demo': demo}

def get_args_parser():
    parser = argparse.ArgumentParser(description='MoDE Data Preparation', add_help=False)
    parser.add_argument('--dataset', default='demo', type=str, choices=['clipeval', 'demo'])
    parser.add_argument('--root', default="data/demo/{0..200000}.tar", type=str,
                        help='path to dataset root')
    parser.add_argument('--caption-dir', default='caption/', type=str, help='caption dir, highly recommended')
    parser.add_argument('--feature-dir', default='feature/', type=str, help='feature output dir')
    
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
    parser.add_argument('--cassign-dir', default='assign/', type=str, help='dir for cluster assignment')
    parser.add_argument('--ccenter-dir', default='cluster_center/', type=str, help='dir for cluster centers')

    # Arguments on intermediate variables at inference time
    parser.add_argument('--logits-dir', default='./logs/clip_eval', type=str, help='cluster center')
    return parser