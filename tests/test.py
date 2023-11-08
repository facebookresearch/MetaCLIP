# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
from PIL import Image
from open_clip import tokenizer
import open_clip
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def test_inference():
    for model_name in ["ViT-B-32", "ViT-B-32-quickgelu", "ViT-B-16", "ViT-L-14"]:
        for pretrained in ["metaclip400m", "metaclip2_5b"]:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )

            current_dir = os.path.dirname(os.path.realpath(__file__))

            image = preprocess(Image.open(current_dir + "/../docs/CLIP.png")).unsqueeze(
                0
            )
            text = tokenizer.tokenize(["a diagram", "a dog", "a cat"])

            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)

                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            assert text_probs.cpu().numpy()[0].tolist() == [1.0, 0.0, 0.0]
