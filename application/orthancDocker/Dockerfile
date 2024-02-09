FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt install --no-install-recommends -y \
    software-properties-common \
    wget \
    build-essential \
    uuid-dev \
    unzip \
    cmake

# Build Orthanc
RUN mkdir -p /tmp/orthanc && cd /tmp/orthanc/ && \
    wget https://orthanc.uclouvain.be/downloads/sources/orthanc/Orthanc-1.12.2.tar.gz -O orthanc && \
    tar xzvf orthanc

RUN mkdir -p /tmp/orthanc/build && \
    cd /tmp/orthanc/build && \
    cmake -DSTATIC_BUILD=ON -DCMAKE_BUILD_TYPE=Release ../Orthanc-1.12.2/OrthancServer/ && \
    make -j && \
    make install

# Build the DICOMWeb plugin
RUN mkdir -p /tmp/orthanc/plugins && cd /tmp/orthanc/plugins && \
    wget https://orthanc.uclouvain.be/downloads/sources/orthanc-dicomweb/OrthancDicomWeb-1.16.tar.gz \
    -O OrthancDicomWeb-1.16.tar.gz && \
    tar xzvf OrthancDicomWeb-1.16.tar.gz

RUN mkdir -p /tmp/orthanc/plugins/build-dicomweb && cd /tmp/orthanc/plugins/build-dicomweb && \
    cmake -DSTATIC_BUILD=ON -DCMAKE_BUILD_TYPE=Release ../OrthancDicomWeb-1.16 && \
    make -j && \
    make install

# Add configuration file
COPY orthanc/orthanc-config.json /etc/orthanc/orthanc.json

ENTRYPOINT ["Orthanc", "/etc/orthanc"]
