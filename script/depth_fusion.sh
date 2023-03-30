#!/usr/bin/env bash

DATASET="./DTU_TEST"

ROOT_DIR="./outputs"

python depth_fusion.py --dataset DTU --geo_mask_thres 4 --full_fusion \
--dataset_dir=$DATASET --root_dir=$ROOT_DIR $@
