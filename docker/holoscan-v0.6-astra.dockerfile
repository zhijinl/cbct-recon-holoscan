FROM nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-dgpu

ARG DEBIAN_FRONTEND=noninteractive

RUN git clone https://github.com/astra-toolbox/astra-toolbox.git /astra-toolbox

RUN apt update && apt install -y build-essential \
    python3-dev \
    autotools-dev \
    autoconf \
    libtool

RUN python -m pip install setuptools==58.2.0
RUN python -m pip install Cython six scipy

RUN cd /astra-toolbox/build/linux/ && ./autogen.sh && ./configure --with-cuda=/usr/local/cuda \
    --with-python \
    --with-install-type=module && \
    make && make install
