# base image
# nvidia/cuda (NOT pytorch/pytorch): the pytorch images are pinned to Python
# 3.12, and we want Python 3.14. We bring our own Python (via uv) and install
# torch from the cu126 wheel index instead. CUDA 12.6 (not 13.x) because CERN's
# k8s-gpu CI runners ship driver 570.x / CUDA 12.8 — a cuda13 stack fails there
# with "driver too old, found 12080"; cu126 runs on the 12.8 driver via
# minor-version compatibility (and on any newer cluster driver too).
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

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
    git openssh-client ca-certificates h5utils wget curl vim nano build-essential jq krb5-user \
    && rm -rf /var/lib/apt/lists/*

# install uv for Python package + interpreter management
RUN wget -qO- https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
ENV UV_NO_CACHE=1

# Bring our own Python 3.14 (the base image has no Python) and put a venv on PATH.
# Everything below installs into /opt/venv, so plain `python`/`pip`/`salt` resolve
# to the 3.14 interpreter.
RUN uv python install 3.14
RUN uv venv --python 3.14 /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# --- deps layer (only files needed to compute metadata/deps) ---
# pyproject + init so setuptools can read __version__
COPY pyproject.toml README.md ./
COPY salt/__init__.py salt/__init__.py

# Install torch (+torchvision) from the CUDA 12.6 wheel index FIRST, so the later
# resolves see torch already satisfied (==2.12.0 matches the +cu126 local build)
# and flash-attn can import it. cp314 wheels are published for both.
RUN uv pip install --index-url https://download.pytorch.org/whl/cu126 \
    "torch==2.12.1" "torchvision==0.27.1"

# Install flash-attn after torch so it can import the already-installed torch.
# The correct prebuilt wheel can be found here: https://flashattn.dev/#finder
# (cp314 / cu126 / torch2.12 variant to match the Python + CUDA + torch above.)
RUN uv pip install --no-deps \
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17/flash_attn-2.8.3%2Bcu126torch2.12-cp314-cp314-linux_x86_64.whl"

# py-lap-solver 0.1.4: no cp314 wheel (max cp312), so a source build is needed.
# Its pyproject uses cmake.minimum-version, which scikit-build-core>=0.8 rejects
# ("Use cmake.version instead"). Mirror the proven recipe from
# container/salt-py314-cuda.def: pin scikit-build-core<0.8 + build deps and build
# with --no-build-isolation, so the later dep resolve sees it already satisfied.
# pyproject-metadata + pathspec: scikit-build-core 0.7.x runtime deps (required
# under --no-build-isolation). pybind11[global]: CMake config files on the
# global prefix for find_package(pybind11).
RUN uv pip install "scikit-build-core<0.8" cmake ninja pyproject-metadata pathspec "pybind11[global]" && \
    uv pip install --no-build-isolation --no-deps "py-lap-solver>=0.1.4"

# Install dependencies (torch already satisfied above; flash already installed)
RUN uv pip install --group dev ".[muP]"

# py-lap-solver's _batched_scipy_lap C++ extension lacks DT_NEEDED for
# libstdc++ ("undefined symbol: _ZTVN10__cxxabiv117__class_type_infoE" at
# import), so BatchedScipyOMP (salt's default LAP solver, no fallback) fails to
# register without preloading it. Same fix as container/salt-py314-cuda.def;
# Ubuntu 24.04 path, safe for torch/scipy/salt.
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Fail the image build (not the downstream test jobs) if the solver is missing.
RUN python -c "from py_lap_solver.solvers import Solvers; \
    assert 'BatchedScipyOMP' in Solvers.get_available_solvers(), 'BatchedScipyOMP missing — libstdc++ LD_PRELOAD fix not working'"

# Copy and install package (Source layer)
COPY . .

# Editable install without re-resolving deps
RUN uv pip install -e . --no-deps
