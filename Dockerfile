FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        python3.10 \
        python3.10-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3.10 -m pip install --no-cache-dir \
        torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt requirements-dev.txt /opt/eth/
RUN python3.10 -m pip install --no-cache-dir \
        -r /opt/eth/requirements.txt \
        -r /opt/eth/requirements-dev.txt

COPY eth /opt/eth/eth
ENV PYTHONPATH=/opt/eth/eth/runtime:/opt/eth
COPY external/apex /opt/eth/external/apex
RUN cd /opt/eth/external/apex \
    && python3.10 -m pip install --no-cache-dir --no-build-isolation -v . \
    && python3.10 -c "import apex; from apex.optimizers import FusedAdam; print(FusedAdam)"

WORKDIR /workspace/eth
