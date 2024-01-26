FROM nvcr.io/nvidia/pytorch:22.03-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN git clone https://github.com/astra-toolbox/astra-toolbox.git /astra-toolbox

RUN apt update && apt install -y build-essential \
    python3-dev \
    autotools-dev \
    autoconf \
    libtool

RUN curl -LO 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara-holoscan/holoscan_dev_deb/v0.6.0-arm64/files?redirect=true&path=holoscan_0.6.0_arm64.deb'

RUN apt-get install '/workspace/files?redirect=true&path=holoscan_0.6.0_arm64.deb' -y

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install matplotlib
RUN apt-get install -y python3-tk

RUN python -m pip install setuptools==58.2.0
RUN python -m pip install Cython six scipy

RUN cd /astra-toolbox/build/linux/ && ./autogen.sh && ./configure --with-cuda=/usr/local/cuda \
    --with-python \
    --with-install-type=module && \
    make && make install

RUN pip3 install git+https://github.com/ahendriksen/tomosipo@develop
RUN pip3 install monai
RUN pip3 install utils

