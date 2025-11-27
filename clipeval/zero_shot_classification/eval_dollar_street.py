import ast
import torch
import json
from PIL import Image
from tqdm import tqdm
import pandas as pd

import sys
if "external/open_clip" not in sys.path:
    sys.path.append("external/open_clip") 
from src.open_clip.zero_shot_classifier import build_zero_shot_classifier # https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/zero_shot_classifier.py
from src.open_clip.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES, IMAGENET_CLASSNAMES # https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/zero_shot_metadata.py


data_dir = 'data/DollarStreet/dataset_dollarstreet/'

def match_any_accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1] # [B, k]
    pred_exp = pred.unsqueeze(2)       # [B, k, 1]
    target_exp = target.unsqueeze(1)  # [B, 1, N]
    # Compare: broadcasted over [B, k, N]
    correct = pred_exp.eq(target_exp).any(dim=2).t() # [k, B] — True if any label matches
    return [float(correct[:k].reshape(-1).float().sum().item()) for k in topk]

# Evaluation Function
def evaluate(model, preprocess_val, tokenizer):
    ds_train_df = pd.read_csv(data_dir + 'images_v2_imagenet_train.csv')
    ds_test_df = pd.read_csv(data_dir + 'images_v2_imagenet_test.csv')
    df = pd.concat([ds_train_df, ds_test_df])
    print("done load data", len(df))

    batch_size = 16
    device = torch.cuda.current_device()

    top1 = 0
    top5 = 0
    n = 0

    classifier = build_zero_shot_classifier(
        model,
        tokenizer=tokenizer,
        classnames=IMAGENET_CLASSNAMES,
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
                batch_imgs.append(Image.open(data_dir + data['imageRelPath']).convert("RGB"))
                labels.append(ast.literal_eval(data['imagenet_sysnet_id']))

            max_len = max(len(x) for x in labels)
            padded_labels = [x + [-1] * (max_len - len(x)) for x in labels]

            labels = torch.tensor(padded_labels).to(device)
            images = torch.stack([preprocess_val(img).to(device) for img in batch_imgs]) 
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ classifier
            tmp1, tmp5 = match_any_accuracy(logits, labels, (1, 5))
            top1 += tmp1
            top5 += tmp5
            n += images.size(0)

    print(f"results {top1}, {top5}, {n}, {top1/n}, {top5/n}")
    return top1, top5, n

def parse_results(results, result_json):
    with open(result_json) as f:
        result = json.load(f)
        print("zero-shot classification dollar street:", result['acc'])
        results['zero_shot_classification_dollar_street'] = result['acc']

def main(model, preprocess_val, tokenizer, result_json):
    top1, top5, n = evaluate(model, preprocess_val, tokenizer)
    with open(result_json, "w") as f:
        json.dump({"top1": top1, "top5": top5, "total": n, "acc": top1/n}, f)
