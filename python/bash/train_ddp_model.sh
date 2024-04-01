#!/bin/bash

HOSTNAME=$(hostname)

case $HOSTNAME in
  "oni03") ROOT_DIR="/data1/yuki_yasuda/workspace_lab/micrometeorology-sr-simulation-2023-yasuda" ;;
  *) exit ;;
esac

HOME_DIR=/home/yuki_yasuda/workspace_lab/micrometeorology-sr-simulation-2023-yasuda

IMAGE_PATH="${ROOT_DIR}/pytorch_es.sif"
SCRIPT_PATH="${HOME_DIR}/python/scripts/train_ddp_model.py"

echo "host name = ${HOSTNAME}"
echo "image path = ${IMAGE_PATH}"
echo "script path = ${SCRIPT_PATH}"

CONFIG_PATH="${HOME_DIR}/python/configs/summary-lr-inference-snapshot/testA.yml"
echo "config path = ${CONFIG_PATH}"

export CUDA_VISIBLE_DEVICES="2,3"
WORLD_SIZE=2

case $HOSTNAME in
  "oni03")
    singularity exec \
      --nv \
      --bind $ROOT_DIR:$HOME_DIR \
      --env PYTHONPATH=$HOME_DIR/python \
      --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
      ${IMAGE_PATH} python3 ${SCRIPT_PATH} --config_path ${CONFIG_PATH} --world_size $WORLD_SIZE
    ;;
  *) exit;;
esac
