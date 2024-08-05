# MoDE: CLIP Data Experts via Clustering

This repository contains the code for the Mixture of Data Experts, described in the paper [MoDE: CLIP Data Experts via Clustering](https://arxiv.org/abs/2404.16030) that provides the first multi-modal understanding system based on independent CLIP models. The main contributions are:
  - Introducing the concept of **data expert** and making the MoDE framework where several small models are separately learned but adaptively ensembled for each task. 
  - Studying how to build a **wider** system, rather than a deeper network. The system is scalable and capable of integrating new data experts, without compromising the extablished ability, which can thus be applied to online data and be continuously updated.  
  - Investigating the quality negative samples in contrastive language-image pretraining, and in particular, the false negatives in web-crawled image-caption pairs.
  - Demonstrating that a set of small data experts can be comparable with a single large model. As the data experts can be trained asynchorously, MoDE significantly reduces the mximum computation requirement, shedding light on research based on limited computation resource. 

We conclude that:
  - Effective pretraining should **carefully examine the data distribution**, instead of aggressively learning from the whole dataset.
  - Data can be used to explain the model capability and determine the ensemble of models (deep learning is data driven). 
  - Our algorithm is simpler and easily scalable to comsume the data in the whole Internet

MoDE is trained w/ face blurred images.

```bibtex
@inproceedings{ma2024mode,
   title={MoDE: CLIP Data Experts via Clustering},
   author={Ma, Jiawei and Huang, Po-Yao and Xie, Saining and Li, Shang-Wen and Zettlemoyer, Luke and Chang, Shih-Fu and Yih, Wen-Tau and Xu, Hu},
   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
   year={2024}
}

@inproceedings{xu2023metaclip,
   title={Demystifying CLIP Data},
   author={Xu, Hu and Xie, Saining and Tan, Xiaoqing and Huang, Po-Yao and Howes, Russell and Sharma, Vasu and Li, Shang-Wen and Ghosh, Gargi and Zettlemoyer, Luke and Feichtenhofer, Christoph},
   booktitle={The Twelfth International Conference on Learning Representations},
   year={2023}
}
```

## Quick Links

  - [Getting Started](#getting-started)
  - [Data Preparation](#data-preparation)
  - [Clustering](#clustering)
  - [Training](#training)
  - [Inference-Time Task Adaptation (Ensemble)](#ensemble)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)
  - [Reference](#reference)


## Getting Started

This code is developed with minimal changes on top of [MetaCLIP](https://github.com/facebookresearch/MetaCLIP). The following command should install requirements for MetaCLIP and `submitit=1.2.1` used by this repo:

```bash
conda create -n metaclip python=3.10 pytorch torchvision pytorch-cuda=11.7 tqdm ftfy braceexpand webdataset regex pandas submitit=1.2.1 \
    -c pytorch-nightly \
    -c nvidia \
    -c conda-forge \
    -c anaconda
```

Then, please refer to the following repos to install the code for kmeans clustering
```bash
https://github.com/kernelmachine/cbtm.git
https://github.com/kernelmachine/balanced-kmeans.git
```
Specifically, please refer the following command
```bash
git clone https://github.com/kernelmachine/balanced-kmeans.git
cd balanced-kmeans
pip3 install -e .
```
Next, please move the dataset file from this folder to src/training.
```bash
mv move2train/* ../src/training/
rm -r move2train
```

Finally, please move the config-related files from this folder to the root
```bash
mv move2root/* ../
rm -r move2root
```

## Data Preparation

In this example code, we assume the dataset is called `demo` and all of the image-caption pairs are saved in a bunch of tarfiles while all tarfiles are tarfiles are organized in sharded folders
```
'demo':
      '0':
         '0.tar'
         '100.tar'
         ...
      '1':
         '1.tar'
         '101.tar'
         ...
      ...
      '99':
         '99.tar'
         '199.tar'
         ...
```
Within each tarfile, the image-caption pairs are saved in sequence.
```
., json, jpeg, json, jpeg ...
```
where for each pair, the text is first stored in a `json` file and the image is then saved in `jpeg`. 

For the following steps, we have provided a detailed command example under `prep-steps` in `run_mode.sh` for explanation & usage. 
The configuration and the paths for intermediate data storing are summarized in `mode/get_prep_parser.py`. When you run the code, please make sure to be in the root directory of the whole project. For the customization of your own data, you can also modify the `get_default_paths` function in the `py` file.

## Clustering

Data clustering is performed on the language embeddings of captions. This section mainly explains feature extraction and data clustering.
For large-scale data processing, we provide the optimized code below to separate the steps and enable multi-thread processing.

### Step 0 Preparing Captions

This step considers the tarfile where the image-caption pairs are stored together. 
As caption extraction is CPU-only, we provide the function below to enable multi-thread caption collection (This is highly recommended for large-scale data processing).

```bash
python mode/prep_caption.py 
```

### Step 1 Preparing Features

This step extracts the language embeddings of captions, and the features for captions in one tarfile will be stored in a single pth file. Following the organization of tarfiles, we also organize the features in sharded folders.

When the captions are pre-collected (via step 0), run the command below to extract the features for captions where each thread is allocated on one GPU chip.

```bash
torchrun --nproc_per_node=8 mode/prep_feature.py  --file-mode caption 
```

As an alternative, you can skip step 0 and directly do feature extraction from the tarfiles.

```bash
torchrun --nproc_per_node=8 mode/prep_feature.py  --file-mode tarfile 
```

### Step 2 Two-Step Clustering

Once the features are ready, perform two-step clustering to obtain the finegrained clusters and the coarse-grained condition. Note we only use a fraction of the whole data to do the clustering on a single GPU chip. Once finished, both the finegrained clusters, coarse-grained clusters can be provided. 

```bash
torchrun --nproc_per_node=1 mode/prep_hrchy.py
```

### Step 3 Cluster Assignment

Once the cluster centers are obtained, use nearest neighborhood search to determine the cluster assignment for each pair. This process is CPU-only and the code below supports multi-thread processing.

```bash
python mode/prep_inference.py
```

## Training

Once the cluster assignment is ready, we do normal training as CLIP but just alter the data sampling. Please check the config file `run_configs_mode.py` and manually change the expert ID via `coarse_idx` to determine the data expert model to be trained.

```bash
torchrun --nproc_per_node=8 src/training/main.py b32_mode
```

## Ensemble

Given the well-trained expert models, for comprehensive evaluation, we gather the outputs from each expert model as well as the ensembled output, and summarize them as a report in original experiment log folder.

Firstly, we evaluate each model and gather their outputs for ensembling.

```bash
torchrun --master_port=29600 --nproc_per_node=4 mode/post_expert_eval.py b32_mode
```

Then, as a preparation for ensembling, we extract the language embeddings of task metadata, e.g., class names. We reuse the feature extraction file but pass different arguments.

```bash
python mode/post_report_ensemble.py b32_mode ${DIR_CLIPEVAL}
```

Lastly, we use the similarity between metadata embeddings and cluster centers to determine ensembling weights for evaluation. By running the command below, all results will be summarized in a csv file.

```bash
python mode/post_report_ensemble.py b32_mode ${DIR_CLIPEVAL}
```

## Bugs or questions?

If you have any questions related to the code or the paper, feel free to email Jiawei Ma (`jiawei.m@columbia.edu`) Hu Xu (`huxu@meta.com`).


## Citation

Please cite our papers (accepted by CVPR 2024 & ICLR 2024) if MoDE helps your work:

```bibtex
@inproceedings{ma2024mode,
   title={MoDE: CLIP Data Experts via Clustering},
   author={Ma, Jiawei and Huang, Po-Yao and Xie, Saining and Li, Shang-Wen and Zettlemoyer, Luke and Chang, Shih-Fu and Yih, Wen-Tau and Xu, Hu},
   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
   year={2024}
}
@inproceedings{xu2023metaclip,
   title={Demystifying CLIP Data},
   author={Xu, Hu and Xie, Saining and Tan, Xiaoqing and Huang, Po-Yao and Howes, Russell and Sharma, Vasu and Li, Shang-Wen and Ghosh, Gargi and Zettlemoyer, Luke and Feichtenhofer, Christoph},
   booktitle={The Twelfth International Conference on Learning Representations},
   year={2023}
}
```

## Reference

The code is based on [MetaCLIP](https://github.com/facebookresearch/MetaCLIP), and only the data loading & sampling is modified.

## TODO
- (welcome your use cases or suggestions to update this codebase regularly)


## License

The MoDE is licensed under CC-BY-NC. 

## Acknowledgement
We gratefully acknowledge the [OpenCLIP](https://github.com/mlfoundations/open_clip) team for initial CLIP codebase and [MetaCLIP](https://github.com/facebookresearch/MetaCLIP) for the careful data distribution examination.
