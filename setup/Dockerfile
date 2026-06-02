# base image
FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel

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

# install uv for Python package management
RUN wget -qO- https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_NO_CACHE=1

# --- deps layer (only files needed to compute metadata/deps) ---
# pyproject + init so setuptools can read __version__
COPY pyproject.toml README.md ./
COPY salt/__init__.py salt/__init__.py

# Install flash-attn after torch so it can import the already-installed torch.
# The correct prebuilt wheel can be found here: https://flashattn.dev/#finder
RUN uv pip install --system --no-deps \
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17/flash_attn-2.8.3%2Bcu130torch2.12-cp312-cp312-linux_x86_64.whl"

# Install dependencies (make use of caching, skip flash because it's already installed)
RUN uv pip install --system --group dev ".[muP]"

# Copy and install package (Source layer)
COPY . .

# Editable install without re-resolving deps
RUN uv pip install --system -e . --no-deps
