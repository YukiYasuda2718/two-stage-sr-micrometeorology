# Two-Stage Super-Resolution Simulation Method for Three-Dimensional Flow Fields Around Buildings for Real-Time Prediction of Urban Micrometeorology <!-- omit in toc -->

- [Setup](#setup)
  - [Singularity](#singularity)
  - [Docker](#docker)
- [Deep learning](#deep-learning)
  - [U-Net for LR inference](#u-net-for-lr-inference)
  - [U-Net for HR inference](#u-net-for-hr-inference)

## Setup

- The experiments have been conducted in the Singularity container.
- The same experimental environment can be made using Docker.

### Singularity

1. Check if the command `singularity` works. If necessary, install Singularity.
2. Build a container: `$ singularity build -f pytorch.sif ./singularity/pytorch.def`
3. Change preferences in `./script/start_singularity_container.sh` if needed.
4. Run a container: `$ ./script/start_singularity_container.sh`

### Docker

1. Check if the command `docker compose` works. If necessary, install Docker.
2. Change preferences in `docker-compose.yml` if needed.
3. Build a container: `$ docker compose build`
4. Run a container: `$ docker compose up -d`

## Deep learning

- This repository contains only the deep-learning code, not the data.
- If the data were in an appropriate directory, the following scripts would work.

### U-Net for LR inference

- Training:
  - `$ ./script/train_unet_lr.sh`
  - Two NVIDIA A100 GPU cards are assumed.
- Inference:
  - `$ ./script/make_lr_inference.sh`
  -  A single NVIDIA A100 GPU cards are assumed.
- Evaluation
  - Start JupyterLab: `$ ./script/start_singularity_container.sh`
  - Run a notebook `python/notebooks/evaluate_lr_models.ipynb`

### U-Net for HR inference

- Training:
  - `$ ./script/train_unet_hr.sh`
  - Two NVIDIA A100 GPU cards are assumed.
- Inference:
  -  `$ ./script/make_hr_inference.sh`
  -  A single NVIDIA A100 GPU cards are assumed.
- Evaluation
  - Start JupyterLab: `$ ./script/start_singularity_container.sh`
  - Run a notebook `python/notebooks/evaluate_hr_models.ipynb`
