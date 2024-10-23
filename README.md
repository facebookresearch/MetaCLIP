# Demystifying CLIP Data

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/activebus/MetaCLIP) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V0Rv1QQJkcolTjiwJuRsqWycROvYjOwg?usp=sharing)

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
   author={Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer and Christoph Feichtenhofer},
   journal={arXiv preprint arXiv:2309.16671},
   year={2023}
}

@inproceedings{xu2024altogether,
   title={Altogether: Image Captioning via Re-aligning Alt-text},
   author={Hu Xu, Po-Yao Huang, Xiaoqing Ellen Tan, Ching-Feng Yeh, Jacob Kahn, Christine Jou, Gargi Ghosh, Omer Levy, Luke Zettlemoyer, Wen-tau Yih, Shang-Wen Li, Saining Xie, Christoph Feichtenhofer},
   journal={arXiv preprint arXiv:2410.17251},
   year={2024}
}
```

## Updates
* 10/09/2024: 🔥 [Altogether: Image Captioning via Re-aligning Alt-text](https://arxiv.org/abs/2410.17251) (aka MetaCLIPv2) is accepted by EMNLP 2024 with [code](altogether/README.md) released.
* 08/15/2024: [v0.1](https://github.com/facebookresearch/MetaCLIP/releases/tag/v0.1) released.
* 04/25/2024: 🔥 paper [MoDE: CLIP Data Experts via Clustering](https://arxiv.org/abs/2404.16030) is accepted by CVPR 2024 with [code](mode/README.md) released.
* 01/18/2024: 🔥 add [code](metaclip/README_metadata.md) for building metadata.
* 01/16/2024: 🔥 paper accepted by ICLR as [spotlight presentation](https://openreview.net/group?id=ICLR.cc/2024/Conference#tab-accept-spotlight).
* 12/25/2023: [Huggingface Space](https://huggingface.co/spaces/activebus/MetaCLIP) demo and [Colab](https://colab.research.google.com/drive/1V0Rv1QQJkcolTjiwJuRsqWycROvYjOwg?usp=sharing) released.
* 12/21/2023: ViT-G/14 released.
* 09/28/2023: initial release.


## Quick Links
  - [Quick Start](#quick-start)
  - [Pre-trained Models](#pre-trained-models)
  - [Development](#development)
    - [Metadata](#metadata)
    - [Curation](#curation)
    - [Training](#training)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)
  - [Reference](#reference)


## Quick Start
The pre-trained MetaCLIP models are available in
<details>
<summary>Huggingface</summary>

```python
from PIL import Image
from transformers import AutoProcessor, AutoModel

processor = AutoProcessor.from_pretrained("facebook/metaclip-b32-400m")
model = AutoModel.from_pretrained("facebook/metaclip-b32-400m")

image = Image.open("docs/CLIP.png")
inputs = processor(text=["a diagram", "a dog", "a cat"], images=image, return_tensors="pt", padding=True)

with torch.no_grad():
  outputs = model(**inputs)
  logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
  text_probs = logits_per_image.softmax(dim=-1)
print("Label probs:", text_probs)
```
</details>

<details>
<summary>This repo or (OpenCLIP)</summary>

```python
import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='metaclip_400m')  # for 2.5B use 'metaclip_fullcc' in OpenCLIP or 'metaclip_2_5b' in this repo

image = preprocess(Image.open("docs/CLIP.png")).unsqueeze(0)
text = open_clip.tokenize(["a diagram", "a dog", "a cat"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
```
</details>

## Pre-trained Models

All MetaCLIP adhere to OpenAI CLIP training setup: we hope to bring back controlled experiments in the "CLIP era of ImageNet". Specifically, we use OpenAI CLIP's `quickgelu` activation for all model configs (which was missing in older versions of OpenCLIP that mainly uses `nn.GELU` instead). We add [ViT-B-16-quickgelu](src/open_clip/model_configs/ViT-B-16-quickgelu.json), [ViT-L-14-quickgelu](src/open_clip/model_configs/ViT-L-14-quickgelu.json), [ViT-H-14-quickgelu](src/open_clip/model_configs/ViT-H-14-quickgelu.json) and [ViT-bigG-14-quickgelu](src/open_clip/model_configs/ViT-bigG-14-quickgelu.json) in this repo.

|    `model_name`     | `pretrained` | Data Card | # of Seen Pairs | Res. | GPUs | IN ZS Acc. |
|:--------------------|:------------:|:---------:|:---------:|:---------:|:---------:|:--------------:|
| `ViT-B-32-quickgelu` | [`metaclip_400m`](https://dl.fbaipublicfiles.com/MMPT/metaclip/b32_400m.pt) | [data card](https://dl.fbaipublicfiles.com/MMPT/metaclip/datacard_400m.json) | 12.8B | 224 | 64 x V100 | 65.5 |
| `ViT-B-16-quickgelu` | [`metaclip_400m`](https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_400m.pt) | [data card](https://dl.fbaipublicfiles.com/MMPT/metaclip/datacard_400m.json) | 12.8B | 224 | 64 x V100 | 70.8 |
| `ViT-L-14-quickgelu` | [`metaclip_400m`](https://dl.fbaipublicfiles.com/MMPT/metaclip/l14_400m.pt) | [data card](https://dl.fbaipublicfiles.com/MMPT/metaclip/datacard_400m.json) | 12.8B | 224 | 128 x V100 | 76.2 |
| `ViT-B-32-quickgelu` | [`metaclip_2_5b`](https://dl.fbaipublicfiles.com/MMPT/metaclip/b32_fullcc2.5b.pt) | [data card](https://dl.fbaipublicfiles.com/MMPT/metaclip/datacard_fullcc2.5b.json) | 12.8B | 224 | 64 x V100 | 67.6 |
| `ViT-B-16-quickgelu` | [`metaclip_2_5b`](https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_fullcc2.5b.pt) | [data card](https://dl.fbaipublicfiles.com/MMPT/metaclip/datacard_fullcc2.5b.json) | 12.8B | 224 | 64 x V100 | 72.1 |
| `ViT-L-14-quickgelu` | [`metaclip_2_5b`](https://dl.fbaipublicfiles.com/MMPT/metaclip/l14_fullcc2.5b.pt) | [data card](https://dl.fbaipublicfiles.com/MMPT/metaclip/datacard_fullcc2.5b.json) | 12.8B | 224 | 128 x V100 | 79.2 |
| `ViT-H-14-quickgelu` | [`metaclip_2_5b`](https://dl.fbaipublicfiles.com/MMPT/metaclip/h14_fullcc2.5b.pt) | [data card](https://dl.fbaipublicfiles.com/MMPT/metaclip/datacard_fullcc2.5b.json) | 12.8B | 224 | 256 x A100 | 80.5 |
| `ViT-bigG-14-quickgelu` | [`metaclip_2_5b`](https://dl.fbaipublicfiles.com/MMPT/metaclip/G14_fullcc2.5b.pt) | [data card](https://dl.fbaipublicfiles.com/MMPT/metaclip/datacard_fullcc2.5b.json) | 12.8B | 224 | 256 x A100 | 82.1 |




## Development 

This code is customized from [OpenCLIP](https://github.com/mlfoundations/open_clip) and will be maintained separately for research on MetaCLIP. The following command should install requirements for OpenCLIP and `submitit=1.2.1` used by this repo:

```bash
conda create -n metaclip python=3.10 pytorch torchvision pytorch-cuda=11.7 tqdm ftfy braceexpand regex pandas submitit=1.2.1 \
    -c pytorch-nightly \
    -c nvidia \
    -c conda-forge \
    -c anaconda
```

### Metadata

MetaCLIP uses 500,000 queries as [metadata](metadata.json) to align the training data to distribution over quality writing of Wikipedia/WordNet terms. This metadata also allows us to release training data distribution of a released model as **data card**.


### How to Curate ?

We have a [demo notebook](demo.ipynb) to show how the proposed algorithm works.

#### I already have a (head distributed) dataset
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
  matched_entry_ids = substr_matching(text, metadata)  # this is for demo purpose that redo substr_matching; see metaclip/README.md.
  curation_prob = min(entry_prob[matched_entry_ids].sum(), 1.0)
  curated = balance_sampling(matched_entry_ids, entry_prob)
  print(f"[curation_prob={curation_prob:.3f}, curated={curated}] {text}")
```

#### I want to curate data from scratch
We release a skeleton code for [sub-string matching](metaclip/cc_matching.py) from CommonCrawl WAT or WARC and [balancing](metaclip/balancing.py). Check [here](metaclip/README.md) for details.


#### Numpy Impl.
A numpy impl. of the algorithm can be found at [`metaclip.pipeline`](metaclip/pipeline.py), close to the impl. used by the paper.


### Training

```python
python submitit_openclip.py b32_400m
```
Please config the corresponding `training_data` in `run_configs_400m.py`.


### Build Your Own Metadata
Consider start from our [code](metaclip/README_metadata.md) for building CLIP's 500k metadata. 


## Bugs or questions?

If you have any questions related to the code or the paper, feel free to email Hu Xu (`huxu@meta.com`).


## Citation

Please cite our paper (accepted by ICLR2024 as spotlight presentation) if MetaCLIP helps your work:

```bibtex
@inproceedings{xu2023metaclip,
   title={Demystifying CLIP Data},
   author={Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer and Christoph Feichtenhofer},
   journal={arXiv preprint arXiv:2309.16671},
   year={2023}
}
```

## Reference

The training code is developed based on [OpenCLIP](https://github.com/mlfoundations/open_clip), modified to the vanilla CLIP training setup.

## TODO
- pip installation of metaclip package;


## License

The majority of MetaCLIP is licensed under CC-BY-NC, however portions of the project are available under separate license terms: open_clip is licensed under the https://github.com/mlfoundations/open_clip license.

## Acknowledgement
We gratefully acknowledge the [OpenCLIP](https://github.com/mlfoundations/open_clip) team for initial CLIP codebase and integration and [NielsRogge](https://github.com/NielsRogge)'s integration into [Huggingface](https://huggingface.co/models?other=metaclip).
