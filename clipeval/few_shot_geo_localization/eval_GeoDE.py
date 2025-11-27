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

data_dir = 'data/geode/'
GROUP_KEY = 'ip_country' # 'ip_country'

# Evaluation Function
def evaluate(model, preprocess_val):
    geo_df = pd.read_csv(data_dir + 'index.csv')
    geo_df = geo_df.sample(frac=1).reset_index(drop=True) #shuffle
    train_df = geo_df.iloc[:20000]
    test_df = geo_df.iloc[20000:]
    print("done load data", len(geo_df), len(train_df), len(test_df))

    batch_size = 16
    device = torch.cuda.current_device()

    ## train classification probe
    classification_probes = []
    country_ids_list = [] # each n_shot has a list, theoretically should be the same, but GeoDE is special, some countries are very rare
    for n_shot in [5, 10, 25]:
        train_sampled = train_df.groupby(GROUP_KEY, group_keys=False).apply(lambda x: x.sample(n=min(len(x), n_shot), random_state=42))
        country_ids = sorted(list(set(train_sampled[GROUP_KEY])))

        df = train_sampled
        with torch.no_grad():
            all_features = []
            all_labels = []
            for start in tqdm(range(0, len(df), batch_size)):
                end = min(start + batch_size, len(df))
                batch_imgs = []
                for i in range(start, end):
                    data = df.iloc[i]
                    try:
                        batch_imgs.append(Image.open(data_dir + 'images/' + data['file_path']).convert("RGB"))
                        all_labels.append(country_ids.index(data[GROUP_KEY]))
                    except:
                        print(f"missing image {data['file_path']}")

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
                try:
                    batch_imgs.append(Image.open(data_dir + 'images/' + data['file_path']).convert("RGB"))
                    country_labels.append(data[GROUP_KEY])
                except:
                    print(f"missing image {data['file_path']}")

            images = torch.stack([preprocess_val(img).to(device) for img in batch_imgs]) 
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            for ind, cache in enumerate(classification_probes):
                labels = [country_ids_list[ind].index(c) if c in country_ids_list[ind] else -1 for c in country_labels]
                if labels.count(-1) > 0:
                    print(f"WARNING: there are {labels.count(-1)} out of {len(labels)} samples country are not in the training set.")
                correct[ind] += _eig_fewshot_acc_fn(cache, image_features.cpu().numpy(), labels, 2.0 ** 10).item()
            
            n += len(labels)
    
    print(f"few_shot [5, 10, 25] geo-localization on GeoDE, {correct}, {n}, {np.array(correct)/n}")        
    return correct, n

def parse_results(results, result_json):
    with open(result_json) as f:
        result = json.load(f)
        print("few-shot geo-localization GeoDE:", result['acc'])
        results['few_shot_geo_loc_GeoDE'] = result['acc']

def main(model, preprocess_val, tokenizer, result_json):
    correct, n = evaluate(model, preprocess_val)
    with open(result_json, "w") as f:
        json.dump({"correct": correct, "total": n, "acc": (np.array(correct)/n).tolist()}, f)
