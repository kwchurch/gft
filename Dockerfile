FROM nvidia/cuda:10.2-base-ubuntu18.04 
RUN apt update && apt install -y python3-pip && apt install -y libsndfile1 
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
RUN ln -s /usr/bin/pip3 /usr/bin/pip
WORKDIR /root/gft
COPY . .
RUN pip install --upgrade pip
RUN pip install gft
