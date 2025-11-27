import torch
import json
from PIL import Image
from tqdm import tqdm
import pandas as pd

import sys
if "external/open_clip" not in sys.path:
    sys.path.append("external/open_clip") 
from src.open_clip.zero_shot_classifier import build_zero_shot_classifier # https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/zero_shot_classifier.py
from src.open_clip.open_clip_train.zero_shot import accuracy # https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/zero_shot.py
from src.open_clip.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES # https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/zero_shot_metadata.py


data_dir = 'data/GeoDE/geode/'
# Evaluation Function
def evaluate(model, preprocess_val, tokenizer):
    df = pd.read_csv(data_dir + 'index.csv')
    classnames = df['object'].unique().tolist()
    print("done load data", len(df))

    batch_size = 16
    device = torch.cuda.current_device()

    top1 = 0
    n = 0

    classifier = build_zero_shot_classifier(
        model,
        tokenizer=tokenizer,
        classnames=classnames,
        templates=OPENAI_IMAGENET_TEMPLATES,
        num_classes_per_batch=10,
        device=device,
        use_tqdm=True,
    )

    with torch.no_grad():
        for local_start in tqdm(range(0, len(df), batch_size)):
            local_end = min(local_start + batch_size, len(df))
            batch_imgs = []
            labels = []

            for i in range(local_start, local_end):
                data = df.iloc[i]
                try:
                    batch_imgs.append(Image.open(data_dir + 'images/' + data['file_path']).convert("RGB"))
                    labels.append(classnames.index(data['object']))
                except:
                    print(f"missing image {data['file_path']}")

            labels = torch.tensor(labels).to(device)
            images = torch.stack([preprocess_val(img).to(device) for img in batch_imgs]) 
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ classifier
            top1 += accuracy(logits, labels)[0]
            n += images.size(0)

    print(f"results {top1}, {n}, {top1/n}")
    return top1, n

def parse_results(results, result_json):
    with open(result_json) as f:
        result = json.load(f)
        print("zero-shot classification GeoDE:", result['acc'])
        results['zero_shot_classification_GeoDE'] = result['acc']

def main(model, preprocess_val, tokenizer, result_json):
    top1, n = evaluate(model, preprocess_val, tokenizer)
    with open(result_json, "w") as f:
        json.dump({"top1": top1, "total": n, "acc": top1/n}, f)
