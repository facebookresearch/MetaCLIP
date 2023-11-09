# Demystifying CLIP Data

This repository contains the code for the MetaCLIP, described in the paper [Demystifying CLIP Data](https://arxiv.org/abs/2309.16671) that formalizes CLIP data curation as a simple algorithm. The main contributions are:
  - Curating data from scratch without filtering via prior models (e.g., different from existing open source [efforts](https://arxiv.org/abs/2111.02114) ) that uses the original CLIP model as a teacher for filtering **student** data.
  - Making training data more transparent, we released our **training data distribution** over [metadata](metadata.json);
  - A scalable algorithm running in the data pipeline, allowing to scale the **data pool** to the whole CommonCrawl (CC) w/ 300+B image-text pairs. We observe that data quality is **much more** important than quantity (different from existing [open source efforts](https://arxiv.org/abs/2210.08402) or [ALIGN](https://arxiv.org/abs/2102.05918) that mostly scale quantity);
  - [standard CLIP training setup](run_configs_400m.py) for controlled experiments and fair comparisons under fixed training and model configuration.

We conclude that:
  - Effective pretraining data should **maximally preserve signal and mitigate noise**, instead of hard removal of noise with blackbox filters that lead to unknown distribution
  - Our algorithm is simpler and scalable to curate the whole Internet
  - Open-sourcing does not just entail a trained model checkpoint but more importantly the **pre-training data distribution**.

MetaCLIP is trained w/ face blurred images.

```bibtex
@inproceedings{xu2023metaclip,
   title={Demystifying CLIP Data},
   author={Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu, Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer and Christoph Feichtenhofer},
   journal={arXiv preprint arXiv:2309.16671},
   year={2023}
}
```

## Updates
* 09/28/2023: initial release.


## Quick Links

  - [Getting Started](#getting-started)
  - [Metadata](#metadata)
  - [Pre-trained Models](#pre-trained-models)
  - [Curation](#curation)
  - [Training](#training)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)
  - [Reference](#reference)


## Getting Started

This code is developed with minimal changes on top of [OpenCLIP](https://github.com/mlfoundations/open_clip). The following command should install requirements for OpenCLIP and `submitit=1.2.1` used by this repo:

```bash
conda create -n python=3.10 pytorch torchvision pytorch-cuda=11.7 tqdm ftfy braceexpand regex pandas submitit=1.2.1 \
    -c pytorch-nightly \
    -c nvidia \
    -c conda-forge \
    -c anaconda
```

## Metadata

MetaCLIP uses 500,000 queries as [metadata](metadata.json) to align the training data to distribution over quality writing of Wikipedia/WordNet terms. This metadata also allows us to release training data distribution of a released model as **data card**.

### Pre-trained Models

We change OpenCLIP to match training in the default CLIP model setup (w/ [ViT-B-16-quickgelu](src/open_clip/model_configs/ViT-B-16-quickgelu.json), [ViT-L-14-quickgelu](src/open_clip/model_configs/ViT-L-14-quickgelu.json) and [ViT-H-14-quickgelu](src/open_clip/model_configs/ViT-H-14-quickgelu.json)). Most OpenCLIP models use `nn.GELU` not `quickgelu` used by vanilla CLIP. We hope this helps research w/ controlled experiments in the "CLIP era of ImageNet".

```python
import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='metaclip_400m')  # or 'metaclip_fullcc'

image = preprocess(Image.open("CLIP.png")).unsqueeze(0)
text = open_clip.tokenize(["a diagram", "a dog", "a cat"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
```

|              Model              | Data Card | # of Seen Pairs | Res. | GPUs | IN ZS Acc. |
|:--------------------------------|:---------:|:---------:|:---------:|:---------:|:--------------:|
| [MetaCLIP B32 400M](https://dl.fbaipublicfiles.com/MMPT/metaclip/b32_400m.pt) | [data card](https://dl.fbaipublicfiles.com/MMPT/metaclip/datacard_400m.json) | 12.8B | 224 | 64 x V100 | 65.5 |
| [MetaCLIP B16 400M](https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_400m.pt) | [data card](https://dl.fbaipublicfiles.com/MMPT/metaclip/datacard_400m.json) | 12.8B | 224 | 64 x V100 | 70.8 |
| [MetaCLIP L14 400M](https://dl.fbaipublicfiles.com/MMPT/metaclip/l14_400m.pt) | [data card](https://dl.fbaipublicfiles.com/MMPT/metaclip/datacard_400m.json) | 12.8B | 224 | 128 x V100 | 76.2 |
| [MetaCLIP B32 2.5B](https://dl.fbaipublicfiles.com/MMPT/metaclip/b32_fullcc2.5b.pt) | [data card](https://dl.fbaipublicfiles.com/MMPT/metaclip/datacard_fullcc2.5b.json) | 12.8B | 224 | 64 x V100 | 67.6 |
| [MetaCLIP B16 2.5B](https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_fullcc2.5b.pt) | [data card](https://dl.fbaipublicfiles.com/MMPT/metaclip/datacard_fullcc2.5b.json) | 12.8B | 224 | 64 x V100 | 72.1 |
| [MetaCLIP L14 2.5B](https://dl.fbaipublicfiles.com/MMPT/metaclip/l14_fullcc2.5b.pt) | [data card](https://dl.fbaipublicfiles.com/MMPT/metaclip/datacard_fullcc2.5b.json) | 12.8B | 224 | 128 x V100 | 79.2 |
| [MetaCLIP H14 2.5B](https://dl.fbaipublicfiles.com/MMPT/metaclip/h14_fullcc2.5b.pt) | [data card](https://dl.fbaipublicfiles.com/MMPT/metaclip/datacard_fullcc2.5b.json) | 12.8B | 224 | 256 x A100 | 80.5 |
| MetaCLIP G14 2.5B | [data card](https://dl.fbaipublicfiles.com/MMPT/metaclip/datacard_fullcc2.5b.json) | 12.8B | 224 | 256 x A100 | ongoing |

## How to Curate ?

We have a [demo notebook](demo.ipynb) to show how the proposed algorithm works.


### I already have a (head distributed) dataset:
CLIP curation can still help as online balancing (Table 6 in the paper). We wrap CLIP curation in two key functions: [substring matching](metaclip/substr_matching.py) (recommended to run offline) and [balancing](metaclip/balancing.py) (either offline or online, please check `metaclip.balancing:main`).

```python
import json
import numpy as np
from metaclip.substr_matching import substr_matching
from metaclip.balancing import balance_sampling

with open("metadata.json") as f:
  metadata = json.load(f)
# entry counts for our 1.6B(pool) -> 400M(curated); please check balance_sampling:main and substr match and count on your own data.
with open("metaclip/entry_counts_400m.json") as f:
  entry_count_json = json.load(f)
entry_count = np.array([entry_count_json[entry] for entry in metadata], dtype=np.uint64)  # uint64 to be safe for scaling.

t = 20000
entry_count[entry_count < t] = t
entry_prob = t / entry_count

for text in ["jacksons chameleon", "battery plate"]:
  matched_entry_ids = substr_matching(text, metadata)
  if balance_sampling(matched_entry_ids, entry_prob):
    print(f"'{text}' curated")
```

### I want to curate data from scratch:
We release a skeleton code for [sub-string matching](metaclip/cc_matching.py) from CommonCrawl WAT or WARC and [balancing](metaclip/balancing.py). Check [here](metaclip/README.md) for details.

## Training

```python
python submitit_openclip.py b32_400m
```
Please config the corresponding `training_data` in `run_configs_400m.py`.


## Bugs or questions?

If you have any questions related to the code or the paper, feel free to email Hu Xu (`huxu@meta.com`).


## Citation

Please cite our paper if MetaCLIP helps your work:

```bibtex
@inproceedings{xu2023metaclip,
   title={Demystifying CLIP Data},
   author={Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu, Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer and Christoph Feichtenhofer},
   journal={arXiv preprint arXiv:2309.16671},
   year={2023}
}
```

## Reference

The training code is developed based on [OpenCLIP](https://github.com/mlfoundations/open_clip), modified to the vanilla CLIP training setup.

## TODO
- cross-json URL dedup in skeleton code;
- numpy implementation for matching and balancing;
- support online downloading;
- support vanilla CLIP API;
- (welcome your use cases or suggestions to update this codebase regularly)


## License

The majority of MetaCLIP is licensed under CC-BY-NC, however portions of the project are available under separate license terms: open_clip is licensed under the https://github.com/mlfoundations/open_clip license.

