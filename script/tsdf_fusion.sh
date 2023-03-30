#!/usr/bin/env bash

ROOT_DIR="./outputs"

python tsdf_fusion.py --n_view 3 --voxel_size 1.5 \
--root_dir=$ROOT_DIR $@
