# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import os
import torch
import torch.nn.functional as F
import time
import numpy as np
from urllib.request import urlopen
from PIL import Image
from pathlib import Path
from transformers import AutoTokenizer


import sys
if "external/big_vision" not in sys.path:
    sys.path.append("external/big_vision") 
# or directly copy the functions from https://github.com/google-research/big_vision/blob/main/big_vision/evaluators/proj/image_text/image_text_retrieval.py

from big_vision.evaluators.proj.image_text import image_text_retrieval


data_dir = 'data/XM3600'

def evaluate_xm3600(model, preprocess, tokenizer):
    LOCALES = ['ar', 'bn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fil', 'fr', 'hi', 'hr', 'hu', 'id', 'it', 'he', 'ja', 'ko', 'mi', 'nl', 'no', 'pl', 'pt', 'quz', 'ro', 'ru', 'sv', 'sw', 'te', 'th', 'tr', 'uk', 'vi', 'zh']

    image_labels = []
    image_features = []
    with torch.no_grad():
        images = []
        for img_path in Path(f'{data_dir}/images/').glob('*.jpg'):  # risky to list file this way?
            with Image.open(img_path) as img:
                image = preprocess(img)
            images.append(image)
            image_labels.append(img_path.stem)
            if len(images) >= 128:
                images = torch.stack(images)
                images = images.cuda(non_blocking=True)
                image_features.append(model.encode_image(images))
                images = []
        if len(images) > 0:
            images = torch.stack(images)
            images = images.cuda(non_blocking=True)
            image_features.append(model.encode_image(images))
            images = []

        image_features = torch.cat(image_features, dim=0)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)

    eval_results = {}
    
    key = 'Recall@1'
    
    for locale in LOCALES:
        text_features = []
        text_labels = []
        
        text_ids = []
        with torch.no_grad():
            # load texts and image_ids as labels
            with open(f'{data_dir}/captions.jsonl', 'r') as f:
                for line in f:
                    data = json.loads(line)
                    
                    captions = data[locale]['caption']
                    text_labels.extend([data['image/key']] * len(captions))

                    captions = tokenizer(captions)
                    text_ids.append(captions)

                    if len(text_ids) >= 128:
                        captions = torch.cat(text_ids).to('cuda:0', non_blocking=True)
                        text_features.append(model.encode_text(captions))
                        text_ids = []

                if len(text_ids) > 0:
                    captions = torch.cat(text_ids).to('cuda:0', non_blocking=True)
                    text_features.append(model.encode_text(captions))
                    text_ids = []

            text_features = torch.cat(text_features, dim=0)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarities = image_features @ text_features.t()

            results_numpy = similarities.cpu().numpy()
            
            text_image = np.argmax(results_numpy, axis=0)
            image_text = np.argmax(results_numpy, axis=1)
    
        id2img = {id_: i for i, id_ in enumerate(image_labels)}
        text_image_correspondence = [id2img[id_] for id_ in text_labels]
        img2txt = image_text_retrieval.image_to_text_retrieval_eval(-results_numpy, text_image_correspondence)
        txt2img = image_text_retrieval.text_to_image_retrieval_eval(-results_numpy, text_image_correspondence)
        eval_results[locale]={"img2txt": float(img2txt[key]), "txt2img": float(txt2img[key])}
    
    eval_results['avg'] = {"img2txt": float(np.mean([eval_results[lang]["img2txt"] for lang in eval_results])), "txt2img": float(np.mean([eval_results[lang]["txt2img"] for lang in eval_results]))}
    return eval_results


def parse_results(results, result_json):
    with open(result_json) as f:
        xm3600_result = json.load(f)
        print("xm3600:", xm3600_result['avg'])
        results['xm3600_t2i'] = xm3600_result['avg']['txt2img']
        results['xm3600_i2t'] = xm3600_result['avg']['img2txt']


def main(model, preprocess_val, tokenizer, result_json):
    eval_results = evaluate_xm3600(model, preprocess_val, tokenizer)
    print(eval_results['avg'])
    with open(result_json, "w") as f:
        json.dump(eval_results, f)
