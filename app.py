# Copyright (c) Meta Platforms, Inc. and affiliates

import os
import json
import numpy as np
import gradio as gr
import torch
import torch.nn.functional as F
import pandas as pd

import sys
sys.path.append("./src")
sys.path.append("./")

from src.open_clip.factory import create_model_and_transforms
from src.training.zero_shot import zero_shot_classifier
from src.training.imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template

from PIL import Image
from metaclip.substr_matching import substr_matching
from metaclip.balancing import balance_sampling


entry_count = None
metadata = None
model = None
preprocess = None
classifier = None


def init_demo():
    global metadata
    with open("metadata.json") as f:
        metadata = json.load(f)
    
    # entry counts for our 1.6B(pool) -> 400M(curated); please check balance_sampling:main and substr match and count on your own data.
    with open("metaclip/entry_counts_400m.json") as f:
        entry_count_json = json.load(f)
    global entry_count
    entry_count = np.array([entry_count_json[entry] for entry in metadata], dtype=np.uint64)  # uint64 to be safe for scaling.

    from configs import search_config
    args = search_config("h14_fullcc")
    args.device = torch.device("cpu")
    args.distributed = None

    global model
    global preprocess
    global classifier
    
    model, _, preprocess = create_model_and_transforms(args.model, pretrained='metaclip_2_5b')
    if not os.path.exists(f"{args.model}_classifier.pt"):
        classifier = zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template, args)
        torch.save(classifier, f"{args.model}_classifier.pt")
    else:
        classifier = torch.load(f"{args.model}_classifier.pt")


def curation(texts):
    t = 20000  # TODO: make this part of the UI
    _entry_count = np.array(entry_count)
    _entry_count[_entry_count < t] = t
    entry_prob = t / _entry_count

    texts = texts.split("\n")
    results = []
    
    for text in texts:
        text = text.strip()
        if len(text) > 0:
            matched_entry_ids = substr_matching(text, metadata)
            curation_prob = min(entry_prob[matched_entry_ids].sum(), 1.0)
            curated = balance_sampling(matched_entry_ids, entry_prob)
            results.append((text, curated, curation_prob))
    
    return "\n".join([f"{curated}, curation_prob.={curation_prob:.2f}" for text, curated, curation_prob in results])


def zeroshot_classification():
    images = preprocess(Image.open(imagenet_list[selected_index])).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        logits = 100. * image_features @ classifier
    # top1 = int(logits.argmax(dim=-1)[0])
    probs = torch.softmax(logits, dim=-1)
    probs, index = probs.topk(k=5, dim=-1)
    probs, index = probs[0].tolist(), index[0].tolist()
    # imagenet_classnames[top1]
    frame = pd.DataFrame({
        "class": [imagenet_classnames[idx] for idx in index],
        "prob": probs,
    })
    return frame


init_demo()

import random

imagenet_list = [f"gradio/imagenet/{synset}/{fn}" for synset in os.listdir("gradio/imagenet") for fn in os.listdir(f"gradio/imagenet/{synset}")]
random.shuffle(imagenet_list)
imagenet_list = imagenet_list[:30]

selected_index = 0


def get_select_index(evt: gr.SelectData):
    global selected_index
    selected_index = evt.index
    print("selected_index", selected_index)
    return evt.index


with gr.Blocks(theme='nuttea/Softblue') as demo:
    gr.Image("/file=metaclip_logo.jpg", width=498, height=102, show_download_button=False, show_label=False)
    with gr.Tab("Curation Algorithm"):
        with gr.Row(equal_height=True):
            alt_box = gr.Textbox(label="alt text", lines=5, max_lines=5)
            prob_box = gr.Textbox(label="Curation Probability", lines=5, max_lines=5)
        btn = gr.Button("Curate!")
        btn.click(fn=curation, inputs=alt_box, outputs=prob_box)
    

    with gr.Tab("MetaCLIP Model"):
        gr.Markdown("Find an Image from ImageNet")
        with gr.Row(equal_height=True):
            gallery = gr.Gallery(
                imagenet_list,
                show_label=False,
                columns=[5], rows=[2],
                object_fit="scale-down", height="auto",
                show_download_button=False,
                interactive=True,
                preview=True,
            )

        gallery.select(get_select_index)

        btn = gr.Button("Classify!")
        barplot = gr.BarPlot(
            label="Top-5 Prob.",
            height=256,
            width=1024,
            y_lim=[0.0, 1.0],
            x="class", 
            y="prob",
            x_title="",
            y_title="",
            vertical=False
        )
        btn.click(fn=zeroshot_classification, outputs=barplot)


if __name__ == "__main__":
    demo.launch(show_api=False, allowed_paths=["metaclip_logo.jpg"] + imagenet_list)  
