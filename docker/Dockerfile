FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV HOME=/workspace \
    LD_LIBRARY_PATH=/usr/local/lib/ \
    PIP_NO_CACHE_DIR="off"

ARG DEBIAN_FRONTEND="noninteractive"

EXPOSE 6006 8888 5000

WORKDIR $HOME

# Install python 3.9 and libraries necessary for `xgrads`
RUN set -eux \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        zsh \
        less \
        git \
        wget \
        curl \
        ttf-mscorefonts-installer \
        libeccodes-dev \
        libgeos-dev \
        libgdal-dev \
        build-essential \
        python3.9 \
        python3.9-dev \
        python3-pip \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Change python version to 3.9
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

COPY Pipfile Pipfile.lock ${HOME}/

# Install python libraries
# PyYaml is already installed using `apt-get`, but pipenv will also install it.
# To avoid conflict, `apt-get purge python3-yaml` is execute to remove the installed PyYaml.
RUN cd ${HOME} \
    && pip3 freeze > uninstall.txt \
    && pip3 uninstall -y -r uninstall.txt \
    && pip3 install -U pip==23.0.1 pipenv==2023.2.18 \
    && export PATH="$PATH:${HOME}/.local/bin" \
    && cp /usr/lib/python3/dist-packages/apt_pkg.cpython-38-x86_64-linux-gnu.so /usr/lib/python3/dist-packages/apt_pkg.so \
    && apt-get purge -y python3-yaml \
    && pipenv sync --system --dev \
    && rm -rf ${HOME}/.cache/pip ${HOME}/.cache/pipenv

COPY .jupyter/ ${HOME}/.jupyter/
