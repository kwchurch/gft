FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

RUN apt update && apt install -y python3-pip && apt install -y libsndfile1
RUN apt install -y ffmpeg

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /root/gft
COPY . .

RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN pip install --upgrade pip
RUN pip install gft
