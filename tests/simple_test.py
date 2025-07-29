# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
from PIL import Image
from src.mini_clip.factory import get_tokenizer, create_model_and_transforms
from src.mini_clip import tokenizer

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def test_inference():
    model, _, preprocess = create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')

    current_dir = os.path.dirname(os.path.realpath(__file__))

    image = preprocess(Image.open(current_dir + "/../docs/CLIP.png")).unsqueeze(0)
    text = tokenizer.tokenize(["a diagram", "a dog", "a cat"])

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    assert text_probs.cpu().numpy()[0].tolist() == [1.0, 0.0, 0.0]


def test_metaclip2_inference():
    model, _, preprocess = create_model_and_transforms(
        "ViT-H-14-quickgelu-worldwide@WorldWideCLIP", pretrained="ckpts/metaclip2_h14_quickgelu_224px_worldwide.pt"
    )

    tokenizer = get_tokenizer("facebook/xlm-v-base")

    current_dir = os.path.dirname(os.path.realpath(__file__))

    image = preprocess(Image.open(current_dir + "/../docs/CLIP.png")).unsqueeze(
        0
    )
    text = tokenizer(["a diagram", "a dog", "a cat"])
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    text_probs = text_probs.cpu().numpy()[0].tolist()
    assert text_probs[0] > text_probs[1] and text_probs[0] > text_probs[2]


if __name__ == "__main__":

    test_metaclip2_inference()
