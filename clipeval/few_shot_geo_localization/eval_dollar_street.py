import torch
import json
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np

import sys
if "external/big_vision" not in sys.path:
    sys.path.append("external/big_vision") 
# or directly copy the functions from https://github.com/google-research/big_vision/blob/main/big_vision/evaluators/fewshot_lsr.py

from big_vision.evaluators.fewshot_lsr import _precompute_cache, _eig_fewshot_acc_fn


data_dir = 'data/DollarStreet/dataset_dollarstreet/'

# Evaluation Function
def evaluate(model, preprocess_val):
    train_df = pd.read_csv(data_dir + 'images_v2_imagenet_train.csv')
    test_df = pd.read_csv(data_dir + 'images_v2_imagenet_test.csv')
    print("done load data", len(train_df), len(test_df))

    batch_size = 16
    device = torch.cuda.current_device()

    ## train classification probe
    classification_probes = []
    country_ids_list = [] # each n_shot has a list, theoretically should be the same, but just in case
    for n_shot in [5, 10, 25]:
        train_sampled = train_df.groupby('country.id', group_keys=False).apply(lambda x: x.sample(n=min(len(x), n_shot), random_state=42))
        country_ids = sorted(list(set(train_sampled['country.id'])))

        df = train_sampled
        with torch.no_grad():
            all_features = []
            all_labels = []
            for start in tqdm(range(0, len(df), batch_size)):
                end = min(start + batch_size, len(df))
                batch_imgs = []
                for i in range(start, end):
                    data = df.iloc[i]
                    batch_imgs.append(Image.open(data_dir + data['imageRelPath']).convert("RGB"))
                    all_labels.append(country_ids.index(data['country.id']))


                images = torch.stack([preprocess_val(img).to(device) for img in batch_imgs]) 
                image_embs = model.encode_image(images)
                image_embs /= image_embs.norm(dim=-1, keepdim=True)

                all_features.append(image_embs)

            all_features = torch.cat(all_features, dim=0)
            print(all_features.shape)

        classification_probes.append(_precompute_cache(all_features.cpu().numpy(), all_labels, len(set(all_labels))))
        country_ids_list.append(country_ids)

    ## start eval
    n = 0
    correct = [0] * len(classification_probes)

    with torch.no_grad():
        for local_start in tqdm(range(0, len(test_df), batch_size)):
            local_end = min(local_start + batch_size, len(test_df))
            batch_imgs = []
            country_labels = []
     
            for i in range(local_start, local_end):
                data = test_df.iloc[i]
                batch_imgs.append(Image.open(data_dir + data['imageRelPath']).convert("RGB"))
                country_labels.append(data['country.id'])

            images = torch.stack([preprocess_val(img).to(device) for img in batch_imgs]) 
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            for ind, cache in enumerate(classification_probes):
                labels = [country_ids_list[ind].index(c) for c in country_labels]
                correct[ind] += _eig_fewshot_acc_fn(cache, image_features.cpu().numpy(), labels, 2.0 ** 10).item()
            
            n += len(labels)
    
    print(f"few_shot [5, 10, 25] geo-localization on DollarStreet, {correct}, {n}, {np.array(correct)/n}")        
    return correct, n

def parse_results(results, result_json):
    with open(result_json) as f:
        result = json.load(f)
        print("few-shot geo-localization dollar street:", result['acc'])
        results['few_shot_geo_loc_dollar_street'] = result['acc']

def main(model, preprocess_val, tokenizer, result_json):
    correct, n = evaluate(model, preprocess_val)
    with open(result_json, "w") as f:
        json.dump({"correct": correct, "total": n, "acc": (np.array(correct)/n).tolist()}, f)
