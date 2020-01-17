FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime
RUN apt-get update && apt-get install -y \ 
    build-essential \
    cmake \
    gfortran \
    git \
    libatlas-base-dev \
    libav-tools  \
    libgtk2.0-dev \
    libjasper-dev \
    libjpeg-dev \
    libopencv-dev \
    libpng-dev \
    libtiff-dev \
    libvtk6-dev \
    pkg-config \
    python-dev \
    python-numpy \
    python-opencv \
    python-pycurl \
    qt5-default \
    unzip \
    webp \
    wget \
    zlib1g-dev 
    #&& apt-get clean && rm -rf /tmp/* /var/tmp/*

ENV SRC_DIR /src

COPY src $SRC_DIR
WORKDIR $SRC_DIR

RUN chmod +x ./train.sh ./inference.sh
RUN pip install -r requirements.txt