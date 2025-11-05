# base image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# local and envs
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_NO_CACHE_DIR=false
ENV PIP_PREFER_BINARY=1
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /salt

# add some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git openssh-client ca-certificates h5utils wget vim nano build-essential jq \
    && rm -rf /var/lib/apt/lists/*

# update python pip
RUN python -m pip install --upgrade pip setuptools wheel

# --- deps layer (only files needed to compute metadata/deps) ---
# pyproject + init so setuptools can read __version__
COPY pyproject.toml README.md ./
COPY salt/__init__.py salt/__init__.py

# Build/Install flash-attn so it can import the already-installed torch
RUN python -m pip install --no-build-isolation --no-deps "flash-attn>=2.5.7"

# Install dependencies (make use of caching)
RUN python -m pip install ".[dev,muP]"

# Copy and install package (Source layer)
COPY . .

# Editable install without re-resolving deps
RUN python -m pip install -e . --no-deps
