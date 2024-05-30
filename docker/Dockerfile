ARG CUDA_VERSION=11.7.1
ARG OS_VERSION=22.04
ARG CUDNN_VERSION=8

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${OS_VERSION}
LABEL org.opencontainers.image.authors="john.lee@robotics.utias.utoronto.ca"

ENV TZ Canada/Eastern
ENV DEBIAN_FRONTEND noninteractive
ENV CUDA_VERSION=${CUDA_VERSION}
ENV OS_VERSION=${OS_VERSION}
ENV CUDNN_VERSION=${CUDNN_VERSION}

SHELL ["/bin/bash", "-c"]

#? System Packages
COPY apt_packages.txt apt_packages.txt
RUN apt-get update && \
    xargs -a apt_packages.txt apt-get install -y --no-install-recommends && \
    rm apt_packages.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    ln -s /usr/bin/pip3 pip;

#? Install basic python packages
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt && \
    rm requirements.txt

#? Install PyTorch
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

#? Install Other Packages
RUN git clone https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && \
    pip3 install -e .
RUN pip3 install git+https://github.com/bdd100k/bdd100k.git

#? Install mmdet and mmtrack
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
RUN pip install mmdet==2.28.2
RUN git clone https://github.com/open-mmlab/mmtracking.git && \
    cd mmtracking && \
    pip install -r requirements/build.txt && \
    pip install -v -e . && \
    pip install numpy==1.23.5

WORKDIR /home