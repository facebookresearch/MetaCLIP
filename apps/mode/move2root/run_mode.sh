# Copyright (c) Meta Platforms, Inc. and affiliates

# To change the data path, please check get_default_paths func in mode/get_prep_parser.py
# ==============================================
# Run the commands below to prepare data clustering
# ==============================================

DATASET=demo
DIR_CLIPEVAL=demo_verify/feature_clipeval/
TEND=-1 # choose a small positive number for test purpose if needed

## Prep-Step 0 Prepare captions 
# (only for pre-train data, not necessary but highly recommended)
# python mode/prep_caption.py --dataset ${DATASET} --tar-end ${TEND}

## Prep-Step 1 Prepare features 
# 1.1 for pre-train dataset (if you skiped step 0, remove --file-mode caption)
# torchrun --master_port=29600 --nproc_per_node=8 mode/prep_feature.py \
#         --dataset ${DATASET} --tar-end ${TEND} --file-mode caption 
# 1.2 for downstream dataset
# torchrun --master_port=29500 --nproc_per_node=1 mode/prep_feature.py \
#     --dataset clipeval --feature-dir ${DIR_CLIPEVAL}

## Prep-Step 2 Two-Level Clustering
# torchrun --master_port=29500 --nproc_per_node=1 mode/prep_hrchy.py --dataset ${DATASET}

## Prep-Step 3 Fine-grained cluster Assignment
# python mode/prep_inference.py --dataset ${DATASET} --tar-end ${TEND}

# ==============================================
# Run the commands below to prepare data clustering
# ==============================================
## Training-Step 1 
# For details, please refer to run_configs_mode.py
# torchrun --nproc_per_node=8 src/training/main.py b32_mode

# ==============================================
# Run the commands below when all ckpts are ready
# ==============================================
## Post-Step 1 
# torchrun --master_port=29600 --nproc_per_node=4 mode/post_expert_eval.py b32_mode

## Post-Step 2
# python mode/post_report_ensemble.py b32_mode ${DIR_CLIPEVAL}
