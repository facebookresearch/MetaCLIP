# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import numpy as np
import json

from tqdm import tqdm
from pathlib import Path

from datasets import load_from_disk
from PIL import Image


def parse_results(results, result_json):
    with open(result_json) as f:
        cvqa_result = json.load(f)
        print(eval_fn, cvqa_result)
        results['cvqa_en'] = cvqa_result['EN']
        results['cvqa_local'] = cvqa_result['LOCAL']


def main(model, preprocess_val, tokenizer, result_json):
    batch_size = 16
    key_to_subtask = {'': 'LOCAL', "Translated ": 'EN'}    
    dataset = load_from_disk('data/cvqa/test')
    
    results = {}
    
    for key in key_to_subtask:
        correct = 0
        n = 0
        from collections import defaultdict
        subset_acc = defaultdict(lambda : [0, 0])
        with torch.no_grad():
            for start in tqdm(range(0, len(dataset), batch_size)):
                end = min(start + batch_size, len(dataset))
                labels = []
                texts = []
                batch_imgs = []
                subsets = []
                for i in range(start, end):
                    data = dataset[i]
                    batch_imgs.append(data['image'])
                    labels.append(data['Label'])
                    subsets.append(data['Subset'])
                    tmp_text = [data[f'{key}Question'] + ' ' + opt for opt in data[f'{key}Options']]
                    assert(len(tmp_text) == 4)
                    texts.extend(tmp_text)

                texts = tokenizer(texts).cuda(non_blocking=True)
                text_embs = model.encode_text(texts)
                images = torch.stack([preprocess_val(img).cuda(non_blocking=True) for img in batch_imgs]) 
                image_embs = model.encode_image(images)

                text_embs /= text_embs.norm(dim=-1, keepdim=True)
                image_embs /= image_embs.norm(dim=-1, keepdim=True)
                N, D = image_embs.shape
                text_embs_reshaped = text_embs.view(N, 4, D)          # [N, 4, D]
                image_embs_reshaped = image_embs.view(N, 1, D)            # [N, 1, D]
                # batched matrix multiply: [N, 1, D] x [N, D, 4] -> [N, 1, 4]
                similarities = torch.bmm(image_embs_reshaped, text_embs_reshaped.transpose(1, 2)).squeeze(1)
                
                match = np.argmax(similarities.cpu().numpy(), axis=1) == labels
                correct += sum(match)

                for ix, subset in enumerate(subsets):
                    subset_acc[subset][1] += 1
                    subset_acc[subset][0] += match[ix]

            print("subsets", len(subset_acc))
            for subset in subset_acc:
                subset_acc[subset] = float(subset_acc[subset][0] / subset_acc[subset][1])

            results[key_to_subtask[key]] = float(correct/len(dataset))
            results[f"{key_to_subtask[key]}_subset"] = subset_acc
    print(results)

    with open(result_json, 'w') as f:
        json.dump(results, f)
