#!/usr/bin/env bash


DATASET="../CUSTOM"

LOAD_CKPT="checkpoints/epoch=15-step=193199.ckpt" 

OUT_DIR="./outputs_general"

python main.py --extract_geometry --test_general \
--test_n_view 5 --test_ray_num 400 --volume_reso 96 \
--test_dir=$DATASET --load_ckpt=$LOAD_CKPT --out_dir=$OUT_DIR $@
