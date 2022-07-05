FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

# base packages
RUN apt-get update -y && apt-get install -y \
    ca-certificates \
    software-properties-common \
    sudo \
    git \
    wget \
    curl \
    bzip2 \
    libgtk2.0-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# imagemagik and opencv dependencies
RUN apt-get update -y && apt-get install -y \
    libmagickwand-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# librosa and pydub dependencies
RUN apt-get update -y && apt-get install -y libsndfile1-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV HOME=/opt
ENV PATH=/opt/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.5 \
 && conda clean -ya

# CUDA and pytorch
RUN conda install -y -c pytorch \
    cudatoolkit=10.2 \
    pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 \
    && conda clean -ya 

# python-magic and ffmpeg4.4 for augly
RUN conda install -y -c conda-forge python-magic \
    && conda clean -ya

ENV PATH "${HOME}/ffmpeg-4.4-amd64-static:$PATH"
RUN curl -sLo ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
    && tar xvf ffmpeg.tar.xz && rm -f ffmpeg.tar.xz

# imagenet c
RUN git clone https://github.com/hendrycks/robustness.git && cd robustness/ImageNet-C/imagenet_c/ && pip install -e . 

# pip packages
ADD ./requirements.txt ./
RUN pip install -r requirements.txt

RUN apt-get autoremove -y && apt-get autoclean -y
