FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get install -y libtcmalloc-minimal4
RUN apt-get install -y git
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && echo "Etc/UTC" > /etc/timezone \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended xzdec dvipng cm-super -y \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y libglib2.0-0 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:git-core/ppa && \
    apt-get update && \
    apt-get install -y git build-essential cmake && \
    git --version

RUN apt-get update && \
    apt-get install -y \
    curl \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    vim \
    mercurial \
    subversion \
    cmake \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    gcc \
    g++

RUN apt update && \
    apt install unzip

# Add OpenCL ICD files for LightGBM
RUN mkdir -p /etc/OpenCL/vendors && \
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

##############################################################################
# LightGBM-GPU
##############################################################################

RUN cd /usr/local/src && mkdir lightgbm && cd lightgbm && \
git clone -b v2.3.1 https://github.com/microsoft/LightGBM && \
cd LightGBM && mkdir build && cd build && \
    cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ .. && \
    make OPENCL_HEADERS=/usr/local/cuda/targets/x86_64-linux/include LIBOPENCL=/usr/local/cuda/targets/x86_64-linux/lib

ENV PATH /usr/local/src/lightgbm/LightGBM:${PATH}

RUN /bin/bash -c "cd /usr/local/src/lightgbm/LightGBM/python-package && python setup.py install --precompile"

RUN pip install --upgrade pip

WORKDIR /workspace

RUN mkdir /assets

COPY requirements.txt /assets/requirements.txt
RUN pip install -r /assets/requirements.txt --upgrade --no-cache-dir

COPY . /workspace/
RUN git config --global --add safe.directory '*'

ENV PYTHONPATH "$PYTHONPATH:./"
