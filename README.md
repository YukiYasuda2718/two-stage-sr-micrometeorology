# Two-Stage Super-Resolution Simulation Method for Three-Dimensional Flow Fields Around Buildings for Real-Time Prediction of Urban Micrometeorology <!-- omit in toc -->

- [Description](#description)
- [Setup](#setup)
  - [Singularity](#singularity)
  - [Docker](#docker)
- [Deep learning](#deep-learning)
  - [U-Net for LR inference](#u-net-for-lr-inference)
  - [U-Net for HR inference](#u-net-for-hr-inference)
- [Citation](#citation)

## Description

This repository contains the source code used in "Two-stage super-resolution simulation method of three-dimensional street-scale atmospheric flows for real-time urban micrometeorology prediction" by Yuki Yasuda and Ryo Onishi.
- [Link](https://doi.org/10.1016/j.uclim.2025.102300) to our article in Urban Climate
- [Link](https://arxiv.org/abs/2404.02631) to our preprint in arXiv

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

## Citation

```txt
@article{
  title = {Two-stage super-resolution simulation method of three-dimensional street-scale atmospheric flows for real-time urban micrometeorology prediction},
  journal = {Urban Climate},
  volume = {59},
  pages = {102300},
  year = {2025},
  issn = {2212-0955},
  doi = {https://doi.org/10.1016/j.uclim.2025.102300},
  url = {https://www.sciencedirect.com/science/article/pii/S2212095525000161},
  author = {Yuki Yasuda and Ryo Onishi},
  keywords = {Urban micrometeorology, Street canyon, Real-time prediction, Super-resolution, Image inpainting, Convolutional neural network},
}
```