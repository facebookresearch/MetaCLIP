# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""refactored from `main` in `eval_zeroshot.py` (SLIP) for clarity.
"""

import torch
import json
import os

from sklearn import metrics


def load_metadata(metadir="clipeval"):
    with open(os.path.join(metadir, 'dataset_catalog.json')) as f:
        catalog = json.load(f)

    with open(os.path.join(metadir, 'templates.json')) as f:
        all_templates = json.load(f)

    with open(os.path.join(metadir, 'labels.json')) as f:
        all_labels = json.load(f)
    return catalog, all_templates, all_labels


def evaluate(d, val_loader, templates, labels, model, tokenizer, classnorm=False):
    print('Evaluating {}'.format(d))

    is_acc = d not in ['FGVCAircraft', 'OxfordPets', 'Caltech101', 'Flowers102', 'Kinetics700', 'HatefulMemes']

    acc_or_outputs = validate_zeroshot(val_loader, templates, labels, model, tokenizer, is_acc, classnorm)

    if d in ['FGVCAircraft', 'OxfordPets', 'Caltech101', 'Flowers102']:
        metric = mean_per_class(*acc_or_outputs)
    elif d == 'Kinetics700':
        top1, top5 = accuracy(*acc_or_outputs, topk=(1, 5))
        metric = (top1 + top5) / 2
        metric = metric.item()
    elif d == 'HatefulMemes':
        metric = roc_auc(*acc_or_outputs)
    else:
        metric = acc_or_outputs

    return metric


@torch.no_grad()
def build_text_features(templates, labels, model, tokenizer, skip_text_projection=False, classnorm=False):
    # TODO: add device
    text_features = []
    for label in labels:
        if isinstance(label, list):
            texts = [t.format(l) for t in templates for l in label]
        else:
            texts = [t.format(label) for t in templates]

        texts = tokenizer(texts).to(next(model.parameters()).device, non_blocking=True)
        class_embeddings = model.encode_text(texts)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embeddings = class_embeddings.mean(dim=0)
        text_features.append(class_embeddings)
    text_features = torch.stack(text_features, dim=0)
    mean, std = None, None
    if classnorm:
        mean, std = text_features.mean(dim=0)[None, :], text_features.std(dim=0)[None, :]
        text_features = (text_features - mean) / std
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features, mean, std


@torch.no_grad()
def validate_zeroshot(val_loader, templates, labels, model, tokenizer, is_acc, classnorm=False):
    # switch to evaluate mode
    model.cuda()
    model.eval()

    total_top1 = 0
    total_images = 0

    all_outputs = []
    all_targets = []

    text_features = None

    for samples in val_loader:
        if text_features is None:
            print('=> encoding captions')
            text_features, mean, std = build_text_features(templates, labels, model, tokenizer, classnorm=classnorm)

        if isinstance(samples, tuple) or isinstance(samples, list):
            images, target = samples[0], samples[1]
        elif isinstance(samples, dict):
            images, target = samples["pixel_values"], samples["targets"]
        else:
            raise ValueError("unknown sample type", type(samples))

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # encode images
        image_features = model.encode_image(images)
        
        if classnorm:
            image_features = (image_features - mean) / std

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # cosine similarity as logits
        logits_per_image = image_features @ text_features.t()
        logits_per_image = logits_per_image.cpu()
        target = target.cpu()
        if is_acc:
            # measure accuracy and record loss
            pred = logits_per_image.argmax(dim=1)
            correct = pred.eq(target).sum()
            total_top1 += correct.item()
            total_images += images.size(0)
        else:
            all_outputs.append(logits_per_image)
            all_targets.append(target)

    if is_acc:
        return 100 * total_top1 / total_images
    else:
        return torch.cat(all_outputs), torch.cat(all_targets)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mean_per_class(outputs, targets):
    pred = outputs.argmax(1)
    confusion_matrix = metrics.confusion_matrix(targets, pred)
    per_classes = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)

    return 100 * per_classes.mean()


def roc_auc(outputs, targets):
    pos_score = outputs[:, 1] - outputs[:, 0]
    metric = metrics.roc_auc_score(targets, pos_score)

    return 100 * metric


if __name__ == '__main__':
    logits = torch.randn(128, 10)
    targets = torch.randint(size=(128,), low=0, high=10)

    evaluate("imagenet", logits, targets)
