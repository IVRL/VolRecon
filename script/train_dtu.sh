#!/usr/bin/env bash

DATASET="./dtu_training/"

LOG_DIR="./checkpoints"

python main.py --max_epochs 16 --batch_size 2 --lr 0.0001 \
--weight_rgb 1.0 --weight_depth 1.0 \
--train_ray_num 1024 --volume_reso 96 \
--root_dir=$DATASET --logdir=$LOG_DIR $@
