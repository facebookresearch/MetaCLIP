# Copyright (c) Meta Platforms, Inc. and affiliates

import sys
sys.path.append("src")
sys.path.append("./")
import os
import random

import torch
import torch.nn.functional as F
import argparse, pickle, pdb
import numpy as np

from kmeans_pytorch import KMeans as BalancedKMeans
from get_prep_parser import get_args_parser, get_default_paths


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def cluster_fine(n_clusters, args, balanced=1):

    path_to_fine = os.path.join(args.ccenter_dir, args.cd, f'F{n_clusters}.pth')
    os.makedirs(os.path.dirname(path_to_fine), exist_ok=True)
    if os.path.exists(path_to_fine):
        print(f'{path_to_fine} is written')
        return True
    
    print(f'Preparing data for file {path_to_fine}')
    file_for_run = []
    for i in range(100):
        feat_files = os.listdir(os.path.join(args.feature_dir,str(i)))
        num_files = len(feat_files)
        num_fun_files =  int(num_files*0.05) + 1 # num_files
        files = np.random.choice(feat_files,num_fun_files).tolist()
        file_for_run.extend([os.path.join(args.feature_dir,str(i),file) for file in files])
    
    np.random.shuffle(file_for_run)
    print('{} files are selected'.format(len(file_for_run)))
    kmeans = BalancedKMeans(n_clusters=n_clusters, device=device, balanced=(balanced==1))
    total_size = 0
    for i,file in enumerate(file_for_run):
        print(i, file)
        feat = F.normalize(torch.load(file)['feat'].cuda(),dim=-1) 

        total_size += feat.size(0)
        if 'cos' in args.cd:
            kmeans.fit(feat, distance='cosine', iter_limit=50, online=True, iter_k=i)
        elif 'euc' in args.cd.lower(): # euclidean
            kmeans.fit(feat, distance='euclidean', iter_limit=50, online=True, iter_k=i)
        else:
            raise ValueError('Not Implemented')
        if (i+1) % 100 == 0:
            print(f'checkpointing at step {i}')
            torch.save({'center':kmeans.cluster_centers.cpu()},path_to_fine)

    print('there are {} files involved in clustering'.format(total_size))
    with open(path_to_fine.replace('.pth','.pkl'),  'wb+') as f:
        _ = pickle.dump(kmeans, f)
    torch.save({'center':kmeans.cluster_centers.cpu()},path_to_fine)
    return True


def cluster_coarse(n_clusters, args, balanced=1):

    path_to_fine = os.path.join(args.ccenter_dir, args.cd, f'F{args.cm}.pth')
    centers = torch.load(path_to_fine)['center']

    path_to_coarse = os.path.join(args.ccenter_dir, args.cd, f'F{args.cm}-C{n_clusters}.pth')
    if os.path.exists(path_to_coarse):
        print(f'{path_to_coarse} is written')
        return True
    
    kmeans = BalancedKMeans(n_clusters=n_clusters, device=device, balanced=(balanced==1))

    if 'cos' in args.cd:
        kmeans.fit(F.normalize(centers.cuda(),dim=-1), distance='cosine', iter_limit=100, online=False)
    elif 'euc' in args.cd.lower(): # euclidean
        kmeans.fit(centers.cuda(), distance='euclidean', iter_limit=100, online=False)
    else:
        raise ValueError('Not Implemented')

    assign = kmeans.predict(centers.cuda(), args.cd)
    torch.save({'coarse':kmeans.cluster_centers.cpu(),'assign':assign.cpu()}, path_to_coarse)
    return True

parser = argparse.ArgumentParser('Clustering Evaluation', parents=[get_args_parser()])
args = parser.parse_args()
paths = get_default_paths()[args.dataset]
args.feature_dir = paths['feature']
args.ccenter_dir = paths['cluster']

cluster_fine(args.cm, args)
cluster_coarse(args.cn, args)
