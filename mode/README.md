# MoDE: CLIP Data Experts via Clustering

This repository contains the code for the Mixture of Data Experts, described in the paper [MoDE: CLIP Data Experts via Clustering](https://arxiv.org/abs/2309.16671) that provides the first multi-modal understanding system based on independent CLIP data expert models. The main contributions are:
  - Investigating the quality negative samples in contrastive language-image pretraining, and in particular, the noise of false negatives in web-crawled image-caption pairs.
  - Making the MoDE framework where several small models are separately learned and the dynamic ensembling via task-level adaptation can hit a single large model. In this way, instead of aggressively making a model deeper, the framework studies how to build a **wider** system.
  - A scalable and robust algorithm that can consistently include new data experts and without forgetting, which acknowledges the temporal dynamics of our online data in real world and the system can be continuously updated.  
  - As a side benefit, since the networks can be trained sequentially, this algorithm significantly reduce the mximum computation required for network training via data clustering, which sheds light on research based on limited computation resource. 

We conclude that:
  - Effective pretraining data should **carefully examine the data distribution and semantics**, instead of aggressively training on the whole dataset.
  - how to scaling the specialization to reach expandable generalization 
  - Our algorithm is simpler and scalable to comsume the data in the whole Internet

MoDE is trained w/ face blurred images.

```bibtex
@inproceedings{ma2023mode,
   title={MoDE: CLIP Data Experts via Clustering},
   author={Jiawei Ma, Po-Yao Huang, Saining Xie, Shang-Wen Li, Luke Zettlemoyer, Shih-Fu Chang, Wen-Tau Yih, and Hu Xu},
   journal={openreview preprint},
   year={2023}
}
@inproceedings{xu2023metaclip,
   title={Demystifying CLIP Data},
   author={Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu, Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer and Christoph Feichtenhofer},
   journal={arXiv preprint arXiv:2309.16671},
   year={2023}
}
```

## Updates
* 03/31/2024: initial release.


## Quick Links

  - [Getting Started](#getting-started)
  - [Pre-trained Models](#pre-trained-models)
  - [Clustering](#clustering)
  - [Training](#training)
  - [Task-Level Adaptation/Ensembling](#ensembling)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)
  - [Reference](#reference)


## Getting Started

This code is developed with minimal changes on top of [OpenCLIP](https://github.com/mlfoundations/open_clip) and [MetaCLIP](https://github.com/facebookresearch/MetaCLIP). The following command should install requirements for OpenCLIP and `submitit=1.2.1` used by this repo:

```bash
conda create -n python=3.10 pytorch torchvision pytorch-cuda=11.7 tqdm ftfy braceexpand regex pandas submitit=1.2.1 \
    -c pytorch-nightly \
    -c nvidia \
    -c conda-forge \
    -c anaconda
```

## Clustering

The clustering is conducted on the language embeddings of captions. As a consequence, this section mainly explains feature extraction and and data clustering.
For large-scale processing, we provide the optimized code below to separate the steps and enable easy parallel processing to speed up the processing.

We have a detailed command example under `prep-steps` in `run_mode.sh` to explain the steps for clustering. 
The configuration and the paths for intermediate data storing are summarized in `mode/get_prep_parser.py`.
When you run the code, please make sure to be in the root directory of the whole project.

### Step 0 Preparing Captions

This step considers the tarfile where the image-caption pairs are stored together. 
As caption extraction is CPU-only, we provide the function below to enable multi-thread caption collection (This is highly recommended for large-scale data processing).

```python
python mode/prep_caption.py --dataset ${DATASET}
```

### Step 1 Preparing Features

This step extracts the language embeddings of captions and the features for caption in one tarfile will be stored in a single pth file. For large-scale data where thouands of tarfiles are presented, we choose to organize the features in sharded folders for each access.

When the captions are pre-collected, run the command below to extract the features for captions where each thread is allocated on one GPU chip.

```python
torchrun --nproc_per_node=8 mode/prep_feature.py --dataset ${DATASET} --file-mode caption 
```

As an alternative, you can skip step 0 and directly do feature extraction from the tarfiles.

```python
torchrun --nproc_per_node=8 mode/prep_feature.py --dataset ${DATASET} --file-mode tarfile 
```

For both step 0 and step 1, please check the organization of tarfiles. Specifically, the code assumes the data organization as follows.
```
mm: tarfiles organized in sharded folders, 
           items in tarfile follow: ., json, jpeg, json, jpeg ...
```

### Step 2 Two-Step Clustering

Once the features are ready, perform two-step clustering to obtain the finegrained clusters and the coarse-grained condition. Note we only use a fraction of the whole data to do the clustering on a single GPU chip. Once finished, both the finegrained clusters, coarse-grained clusters can be provided. 

```python
torchrun --nproc_per_node=1 mode/prep_hrchy.py --dataset ${DATASET}
```

### Step 3 Cluster Assignment

Once the cluster centers are obtained, use nearest neighborhood search to determine the cluster assignment for each pair. This process is CPU-only and the code below supports multi-thread processing.

```python
python mode/prep_inference.py --dataset ${DATASET} 
```

## Training

Once the cluster assignment is ready, we do normal training as CLIP but just alter the data sampling. Please check the config file `run_configs_mode.py` and manually change the expert ID via `coarse_idx` to determine the expert model to be trained.

```bash
torchrun --nproc_per_node=8 src/training/main.py b32_mm_mode
```

## Ensembling

Given the well-trained expert models, for comprehensive evaluation, we gather the outputs from each expert model as well as the ensembled output, and summarize them as a report in original experiment log folder.

Firstly, we evaluate each model and gather their outputs for ensembling.

```bash
torchrun --master_port=29600 --nproc_per_node=4 mode/post_expert_eval.py b32_mm_mode
```

Then, as a preparation for ensembling, we extract the language embeddings of task metadata, e.g., class names. We reuse the feature extraction file but pass different arguments.

```bash
python mode/post_report_ensemble.py b32_mm_mode ${DIR_CLIPEVAL}
```

Lastly, we use the similarity between metadata embeddings and cluster centers to determine ensembling weights for evaluation. By running the command below, all results will be summarized in a csv file.

```bash
python mode/post_report_ensemble.py b32_mm_mode ${DIR_CLIPEVAL}
```

## Bugs or questions?

If you have any questions related to the code or the paper, feel free to email Jiawei Ma () and Hu Xu (`huxu@meta.com`).


## Citation

Please cite our paper if MoDE helps your work:

```bibtex
@inproceedings{ma2023mode,
   title={MoDE: CLIP Data Experts via Clustering},
   author={Jiawei Ma, Po-Yao Huang, Saining Xie, Shang-Wen Li, Luke Zettlemoyer, Shih-Fu Chang, Wen-Tau Yih, and Hu Xu},
   journal={openreview preprint},
   year={2023}
}
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
- (welcome your use cases or suggestions to update this codebase regularly)


## License

The majority of MetaCLIP is licensed under CC-BY-NC, however portions of the project are available under separate license terms: open_clip is licensed under the https://github.com/mlfoundations/open_clip license.

