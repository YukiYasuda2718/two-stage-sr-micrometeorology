#!/bin/bash

HOME_DIR=/home/yuki_yasuda/workspace_hub/two-stage-sr-micrometeorology
ROOT_DIR=/data1/yuki_yasuda/workspace_hub/two-stage-sr-micrometeorology

SCRIPT_PATH=$HOME_DIR/python/scripts/train_ddp_model.py
CONFIG_PATH=$HOME_DIR/python/configs/lr-inference/default_lr.yml

CUDA_VISIBLE_DEVICES="0,1"
WORLD_SIZE=2

singularity exec \
        --nv \
        --bind $ROOT_DIR:$HOME_DIR \
        --env PYTHONPATH=$HOME_DIR/python \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    $ROOT_DIR/pytorch.sif python3 $SCRIPT_PATH --config_path $CONFIG_PATH --world_size $WORLD_SIZE
