# Copyright (c) Meta Platforms, Inc. and affiliates

import sys
sys.path.append("src")
sys.path.append("./")
import os
import random

import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
import argparse
from transformers import AutoModel, AutoTokenizer

from src.training.distributed import init_distributed_device

from get_prep_parser import get_args_parser, get_default_paths
from prep_caption import gather_caption_from_tarfile, get_tarfile_path
from clipeval import eval_zeroshot


@torch.no_grad()
def build_text_indfeatures(templates, labels, model, tokenizer):
    text_features = []
    for i,label in enumerate(labels):
        if isinstance(label, list):
            texts = [t.format(l) for t in templates for l in label]
        else:
            texts = [t.format(label) for t in templates]

        texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        texts = {key:item.cuda() for key,item in texts.items()}
        class_embeddings = model(**texts, output_hidden_states=True, return_dict=True).pooler_output.cpu()
        
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        text_features.append(F.normalize(class_embeddings.mean(dim=0),dim=-1))
    text_features = torch.stack(text_features,dim=0)
    return text_features


def main(args):
    device = init_distributed_device(args)
    os.makedirs(args.feature_dir,exist_ok=True)

    if 'demo' in config.dataset:
        _, tar_end = os.path.basename(config.root).split("{")[1].split("}")[0].split("..")
        if config.tar_end == -1:
            config.tar_end = int(tar_end)
        else:
            config.tar_end = min(config.tar_end, int(tar_end))
        if config.tar_per_gpu == -1:
            config.tar_per_gpu = int((config.tar_end - config.tar_init) / config.world_size)

    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").cuda()

    stat = {'non_exist':[], 'failed':[], 'success':[]}
    if 'demo' in args.dataset:

        current_init = args.tar_init+args.rank*args.tar_per_gpu
        current_end = current_init + args.tar_per_gpu
        shard_id_list = [i for i in range(current_init,min(current_end,args.tar_end))]
        random.shuffle(shard_id_list)
        
        for shard_id in shard_id_list:
            save_path = os.path.join(args.feature_dir, str(shard_id % 100),'{}_feat.pth'.format(shard_id))
            if os.path.exists(save_path):
                stat['success'].append(shard_id)
                print(shard_id, f'already written in {save_path}')
                continue
            
            txtfeats = {'feat':[]}
            with torch.no_grad():
                os.makedirs(os.path.dirname(save_path),exist_ok=True)

                if args.file_mode == 'caption':
                    caption_file = os.path.join(args.caption_dir,str(shard_id % 100),f'{shard_id}_caption.json')
                    if not os.path.isfile(caption_file):
                        tarpath = get_tarfile_path(args.root, shard_id)
                        if os.path.isfile(tarpath):
                            stat['failed'].append(shard_id)
                        else:
                            stat['non_exist'].append(shard_id)
                        continue
                    captions = json.load(open(caption_file,'r'))
                else:
                    captions, status = gather_caption_from_tarfile(os.path.dirname(args.root), shard_id)
                    if captions is None:
                        stat['non_exist'].append(shard_id)
                        continue

                chunks = [captions['caption'][x:x+args.chunk_size] for x in range(0, len(captions['caption']), args.chunk_size)]
                for chunk in chunks:
                    inputs = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")
                    inputs = {key:item.cuda() for key,item in inputs.items()}
                    embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu()
                    txtfeats['feat'].append(embeddings)

            txtfeats['filekeys'] = captions['tmems']
            txtfeats['img_midx'] = captions['imems']
            txtfeats['feat'] = torch.cat(txtfeats['feat'],dim=0)
            if len(txtfeats['feat']) != len(txtfeats['filekeys']):
                print(f'check {shard_id}')
                stat['failed'].append(shard_id)
            else:
                torch.save(txtfeats, save_path)
                stat['success'].append(shard_id)
                print('write feature for {} with {} items'.format(shard_id, len(txtfeats['feat'])))

    if args.dataset == 'clipeval':

        catalog, all_templates, all_labels = eval_zeroshot.load_metadata("clipeval")
        for d in catalog:
            feat_file = os.path.join(args.feature_dir, f'{d}.pth')
            if os.path.exists(feat_file):
                continue
            templates = all_templates[d]
            labels = all_labels[d]
            text_embeddings = build_text_indfeatures(templates, labels, model, tokenizer)
            torch.save(text_embeddings, feat_file)

    else:
        raise ValueError('Please comment this command and customize the code for yourself')


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Clustering evaluation', parents=[get_args_parser()])
    config = parser.parse_args()

    if config.dataset != 'clipeval':
        paths = get_default_paths()[config.dataset]
        config.root = paths['root']
        config.caption_dir = paths['caption']
        config.feature_dir = paths['feature']
    
    os.makedirs(config.feature_dir, exist_ok=True)
    main(config)
