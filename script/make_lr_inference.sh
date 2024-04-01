#!/bin/bash

HOME_DIR=/home/yuki_yasuda/workspace_hub/two-stage-sr-micrometeorology
ROOT_DIR=/data1/yuki_yasuda/workspace_hub/two-stage-sr-micrometeorology

SCRIPT_PATH=$HOME_DIR/python/scripts/make_lr_inference.py
CUDA_VISIBLE_DEVICES="0"

singularity exec \
        --nv \
        --bind $ROOT_DIR:$HOME_DIR \
        --env PYTHONPATH=$HOME_DIR/python \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    $ROOT_DIR/pytorch.sif python3 $SCRIPT_PATH
