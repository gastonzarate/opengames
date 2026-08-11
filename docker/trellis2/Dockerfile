FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0" \
    PIP_NO_CACHE_DIR=1

# 8.6 = A10G (ml.g5) y RTX 3070 · 8.9 = L40S y RTX 4090 · 9.0 = H100.
# Compilar para las tres deja la imagen utilizable en gaston-pc y en AWS.

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build \
        python3.11 python3.11-dev python3-pip \
        libgl1 libglib2.0-0 libegl1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /opt
RUN git clone -b main --recursive https://github.com/microsoft/TRELLIS.2.git

WORKDIR /opt/TRELLIS.2
RUN python -m pip install --upgrade pip \
 && python -m pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124

# `docker build` no pasa el driver de la GPU al contenedor (a diferencia de
# `docker run --gpus`), así que nvidia-smi no existe durante el build.
# setup.sh solo usa "command -v nvidia-smi" para decidir PLATFORM=cuda vs
# hip; nunca lo ejecuta de verdad. Sin esto, todas las capas de extensiones
# fallan con "Error: No supported GPU found" antes de instalar nada. Un stub
# alcanza para pasar esa detección: la compilación real la hace nvcc contra
# TORCH_CUDA_ARCH_LIST, que no necesita una GPU visible en tiempo de build.
RUN printf '#!/bin/sh\nexit 0\n' > /usr/local/bin/nvidia-smi \
 && chmod +x /usr/local/bin/nvidia-smi

# /bin/sh en Ubuntu es dash: ". ./setup.sh --flag" en dash no le pasa los
# argumentos al script sourceado (eso es una extensión de bash). Sin esto,
# setup.sh corre sin flags, imprime su ayuda y sale con código 0: Docker cree
# que la capa instaló algo cuando en realidad no instaló nada.
SHELL ["/bin/bash", "-c"]

# Las extensiones se instalan de a una y en capas separadas a propósito: si una
# falla, la caché de Docker conserva las anteriores y el ciclo de arreglo no
# vuelve a compilar todo desde cero.
RUN . ./setup.sh --basic
RUN . ./setup.sh --flash-attn
RUN . ./setup.sh --nvdiffrast
RUN . ./setup.sh --nvdiffrec
RUN . ./setup.sh --cumesh
RUN . ./setup.sh --o-voxel
RUN . ./setup.sh --flexgemm

COPY smoke_imports.py /opt/smoke_imports.py
ENV PYTHONPATH=/opt/TRELLIS.2

# setup.sh no tiene "set -e": si un paso interno de un flag falla pero no es
# el último comando del bloque, la capa RUN igual sale con código 0. Este
# chequeo (en CPU, sin --gpus) es lo único que de verdad garantiza que las
# extensiones quedaron importables antes de dar la imagen por buena.
RUN python /opt/smoke_imports.py

CMD ["python", "/opt/smoke_imports.py"]
