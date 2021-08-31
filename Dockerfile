FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

SHELL [ "/bin/bash", "-c"]
# Install required packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -yq software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt install -yq apt-transport-https \
    gcc \
    g++ \
    wget \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY /. /home/xview3/
WORKDIR /home/xview3/

# Changing parameter back to default
ENV DEBIAN_FRONTEND=newt
# Install Miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda update -n base -c defaults conda \
    && conda env create -f ./environment.yml \ 
    && conda init bash

# Set entrypoint
ENTRYPOINT [ "/home/xview3/run_inference.sh" ]