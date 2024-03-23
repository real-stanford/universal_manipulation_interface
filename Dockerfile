FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBAIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
RUN apt-get -y install curl
RUN apt-get install -y wget

# RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.12.0-Linux-x86_64.sh -O /tmp/miniconda.sh
# RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
#     rm /tmp/miniconda.sh && \
#     echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh
# ENV PATH /opt/conda/bin:$PATH

# ENV PATH="/root/miniconda3/bin:${PATH}"
# ARG PATH="/root/miniconda3/bin:${PATH}"
# RUN apt-get update

# RUN wget \
#     https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && mkdir /root/.conda \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b \
#     && rm -f Miniconda3-latest-Linux-x86_64.sh 
# RUN conda --version

RUN apt-get update && apt-get -y install git
RUN apt-get update && apt-get -y install python3.9 python3-pip
RUN apt install -y libprotobuf-dev protobuf-compiler
RUN apt-get update && apt-get -y install cmake

# Set the non-root user as the default user
ARG UID=1000
ARG GID=1000

# Update the package list, install sudo, create a non-root user, and grant password-less sudo permissions
RUN apt update && \
    apt install -y sudo && \
    addgroup --gid $GID nonroot && \
    adduser --uid $UID --gid $GID --disabled-password --gecos "" nonroot && \
    echo 'nonroot ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

USER nonroot
ENV PATH="/home/nonroot/miniconda3/bin:${PATH}"
ARG PATH="/home/nonroot/miniconda3/bin:${PATH}"
WORKDIR /tmp
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

WORKDIR /ws
RUN git clone https://github.com/baishibona/universal_manipulation_interface.git
WORKDIR /ws/universal_manipulation_interface

RUN conda env create -f conda_environment.yaml
WORKDIR /ws/universal_manipulation_interface/example_demo_session
RUN wget --recursive --no-parent --no-host-directories --cut-dirs=2 --relative --reject="index.html*" https://real.stanford.edu/umi/data/example_demo_session/
ADD run.sh /ws/universal_manipulation_interface
WORKDIR /ws/universal_manipulation_interface
RUN sudo chmod a+x run.sh
