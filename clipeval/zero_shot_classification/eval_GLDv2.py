import torch
import json
from PIL import Image
from tqdm import tqdm
from collections import Counter
import pandas as pd

import sys
if "external/open_clip" not in sys.path:
    sys.path.append("external/open_clip") 
from src.open_clip.open_clip_train.zero_shot import accuracy # https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/zero_shot.py

data_dir = 'data/GLDv2/'

# return top landmark or all retrieved landmarks
def get_landmark(row, image_to_land_id_map, land_id_to_name):
    tmp = [image_to_land_id_map[id] for id in row['images'].split()]
    counter = Counter(tmp)
    landmark_id, count = counter.most_common(1)[0]
    return land_id_to_name[landmark_id], [land_id_to_name[x] for x in set(tmp)] # most_voting landmark, all landmarks

def encode_texts(model, tokenizer, texts, device):
    texts = tokenizer(texts).to(device)        
    text_embs = model.encode_text(texts)
    text_embs /= text_embs.norm(dim=-1, keepdim=True)
    return text_embs

# Evaluation Function
def evaluate(model, preprocess_val, tokenizer):
    # prepare for GLDv2 data
    df = pd.read_csv(data_dir + 'retrieval_solution_v2.1.csv')
    df = df[df['Usage'].isin(['Private', 'Public'])]
    ids = set(df['id'])
    print("test images count: ", len(ids))
    image_ids = set([id for x in df['images'] for id in x.split()])
    print("retrieved images count: ", len(image_ids))

    image_to_landmark_df = pd.read_csv(data_dir + 'index_image_to_landmark.csv')
    image_to_landmark_df = image_to_landmark_df[image_to_landmark_df['id'].isin(image_ids)]
    image_to_land_id_map = {}
    for _, row in image_to_landmark_df.iterrows():
        image_to_land_id_map[row['id']] = row['landmark_id']

    landmark_to_category_df = pd.read_csv(data_dir + 'index_label_to_category.csv')
    landmark_to_category_df = landmark_to_category_df[landmark_to_category_df['landmark_id'].isin(set(image_to_landmark_df['landmark_id']))]
    land_id_to_name = {}
    for _, row in landmark_to_category_df.iterrows():
        category = row['category']
        name = category[category.rfind(':') + 1:].replace("_", " ").rstrip('"')
        land_id_to_name[row['landmark_id']] = name

    landmarks = list(land_id_to_name.values())
    print("number of landmarks: ", len(landmarks))

    batch_size = 16
    device = torch.cuda.current_device()

    top1 = 0
    n = 0

    text_features = encode_texts(model, tokenizer, landmarks, device)

    with torch.no_grad():
        for local_start in tqdm(range(0, len(df), batch_size)):
            local_end = min(local_start + batch_size, len(df))
            batch_imgs = []
            labels = []

            for i in range(local_start, local_end):
                data = df.iloc[i]
                batch_imgs.append(Image.open(data_dir + 'test/' + data['id'] + '.jpg').convert("RGB"))
                landmark_name, retrieved_landmarks = get_landmark(data, image_to_land_id_map, land_id_to_name) 
                labels.append(landmarks.index(landmark_name))

            labels = torch.tensor(labels).to(device)
            images = torch.stack([preprocess_val(img).to(device) for img in batch_imgs]) 
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            probs = image_features @ text_features.T
            top1 += accuracy(probs, labels)[0]
            n += images.size(0)

    print(f"results {top1}, {n}, {top1/n}")
    return top1, n

def parse_results(results, result_json):
    with open(result_json) as f:
        result = json.load(f)
        print("zero-shot classification GLDv2:", result['acc'])
        results['zero_shot_classification_GLDv2'] = result['acc']

def main(model, preprocess_val, tokenizer, result_json):
    top1, n = evaluate(model, preprocess_val, tokenizer)
    with open(result_json, "w") as f:
        json.dump({"top1": top1, "total": n, "acc": top1/n}, f)
