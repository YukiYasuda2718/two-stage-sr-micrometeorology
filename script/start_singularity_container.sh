#!/bin/bash

PORT=10169
HOME_DIR=/home/yuki_yasuda/workspace_hub/two-stage-sr-micrometeorology
ROOT_DIR=/data1/yuki_yasuda/workspace_hub/two-stage-sr-micrometeorology

CUDA_VISIBLE_DEVICES="0"

singularity exec \
        --nv \
        --bind $ROOT_DIR:$HOME_DIR \
        --env PYTHONPATH=$HOME_DIR/python \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    $ROOT_DIR/pytorch.sif jupyter lab \
        --no-browser --ip=0.0.0.0 --allow-root --LabApp.token='' --port=$PORT
