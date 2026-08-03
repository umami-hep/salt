# ONE Dockerfile, TWO images. The variants differ in exactly three values —
# base image, torch wheel index, flash-attn wheel — supplied as build args, so
# there is a single source of truth for everything else.
#
#   VARIANT  CUDA   archs (compiled into the torch wheel)  GPUs
#   -------  -----  ------------------------------------  ---------------------
#   V2H      12.6   sm_50 .. sm_90                         V100 T4 A100 H100 H200
#   T2B      13.0   sm_75(*) .. sm_120                     T4 A100 H100/H200 RTX 40xx/50xx
#
# V2H = Volta->Hopper, T2B = Turing->Blackwell. No single prebuilt torch wheel
# spans V100 and Blackwell: upstream dropped Volta from the same releases that
# added Blackwell. The ranges overlap widely, so only V100 owners (V2H) and
# Blackwell owners (T2B) are forced to a specific image. Coverage is gap-free —
# V2H starts at sm_50, so even if T2B's real floor turns out to be sm_80 there
# is no hole. See docs/setup.md and setup/valid-container-versions.
#
# (*) MEASURED for cu126 and cu129 (arch probes on real hardware). The cu130
#     floor is INFERRED, not measured — read it off a built T2B image with
#     `python -c "import torch; print(torch.cuda.get_arch_list())"`. The build
#     below prints the compiled arch flags into the CI log for exactly this.
#
# Defaults reproduce the V2H (cu126) image, so an argument-less `docker build`
# behaves as it always has.
#
# nvidia/cuda (NOT pytorch/pytorch): the pytorch images are pinned to Python
# 3.12, and we want Python 3.14. We bring our own Python (via uv) and install
# torch from the appropriate wheel index instead.
#
# DRIVER CAVEAT for the T2B variant, unresolved as of 2026-08-03: CUDA 13 needs
# a 13.x (580+) driver on the *host*. Every lxplus GPU node was measured on
# 2026-08-02 to run driver 13.0 or 13.3, so lxplus is fine. CERN's k8s-gpu CI
# fleet has NOT been re-checked — an older note here claimed 570.x/CUDA 12.8,
# which if still true means a T2B image cannot run the `gpu-smoke` CI job
# ("driver too old"). That is why `gpu-smoke` in .gitlab/.ci-test.yaml stays
# pinned to the V2H (cu126) image, which runs on any 12.8-or-newer driver via
# minor-version compatibility.
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

# Re-declared after FROM so they are visible to the RUN steps below. These two
# must agree with each other AND with CUDA_IMAGE's major version; the gate after
# the torch install enforces that rather than trusting it.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
ARG FLASH_WHEEL=https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17/flash_attn-2.8.3%2Bcu126torch2.12-cp314-cp314-linux_x86_64.whl

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

# Install torch (+torchvision) from the CUDA wheel index FIRST, so the later
# resolves see torch already satisfied and flash-attn can import it. The torch
# VERSION is deliberately identical across variants — only the wheel channel
# differs — so salt's pins are unaffected by the choice of image. cp314 wheels
# for 2.12.1/0.27.1 are published on both the cu126 and cu130 indices (verified
# 2026-08-03).
#
# The gate is not ceremony: if TORCH_INDEX is wrong or the channel stops
# publishing cp314, pip does NOT fail — it quietly serves a CPU wheel or a
# different +cuXXX build, and the image then dies on the user's first kernel
# launch. Assert the channel we asked for is the channel we got.
RUN uv pip install --index-url "${TORCH_INDEX}" \
    "torch==2.12.1" "torchvision==0.27.1" && \
    python -c "import torch; \
    want = '${TORCH_INDEX}'.rstrip('/').rsplit('/cu', 1)[-1]; \
    got = torch.version.cuda; \
    assert got is not None, 'not a CUDA build — TORCH_INDEX did not serve a +cuXXX wheel'; \
    assert got.replace('.', '') == want, f'wrong wheel channel: asked cu{want}, got CUDA {got}'; \
    print('torch', torch.__version__, '/ CUDA', got)"

# Record the compiled GPU architectures in the build log. Informational, NOT a
# gate: the build runner has no driver, and torch.cuda.get_arch_list() returns
# [] when is_available() is False, so an assert here would fail a perfectly good
# image. _cuda_getArchFlags reads the compiled-in flags and needs no device, but
# is not guaranteed to be available either. This line is the cheapest way to
# learn what a variant actually covers — see the (*) note in the header.
RUN python -c "import torch; \
    f = getattr(torch._C, '_cuda_getArchFlags', None); \
    print('compiled arch flags:', (f() if f else None) or '<unavailable in the driverless build sandbox>')" \
    || echo "compiled arch flags: <probe failed; informational only, not a gate>"

# Install flash-attn after torch so it can import the already-installed torch.
# Prebuilt wheel, never a source build: compiling flash-attn takes 1.5-2.5 h and
# is pure waste when mjun0812 publishes exactly the cp314 x torch2.12 x cuXXX
# matrix we need. Wheel finder: https://flashattn.dev/#finder
# NOTE this constrains which CUDA channels are usable at all — cu126 and cu130
# have cp314/torch2.12 wheels, cu129 does NOT. That is why T2B is cu130.
RUN uv pip install --no-deps "${FLASH_WHEEL}"

# py-lap-solver 0.1.4: no cp314 wheel (max cp312), so a source build is needed.
# Its pyproject uses cmake.minimum-version, which scikit-build-core>=0.8 rejects
# ("Use cmake.version instead"). Mirror the proven recipe from
# container/salt-py314-cuda.def: pin scikit-build-core<0.8 + build deps and build
# with --no-build-isolation, so the later dep resolve sees it already satisfied.
# pyproject-metadata + pathspec: scikit-build-core 0.7.x runtime deps (required
# under --no-build-isolation). pybind11[global]: CMake config files on the
# global prefix for find_package(pybind11).
#
# PORTABLE CODEGEN (gpu-smoke SIGILL/exit-132 fix): the 0.1.4 sdist's
# CMakeLists.txt hard-appends `-march=native -m64 -mfpmath=sse` (x86_64) via
# target_compile_options(), which lands AFTER CMAKE_CXX_FLAGS on the compile
# line — so CFLAGS/CXXFLAGS/CMAKE_ARGS flag overrides can NEVER win (the last
# -march wins for gcc/clang). Compiled on the docker build runner, the
# extension then carries ISA extensions (e.g. AVX-512) that the GPU CI node's
# CPU lacks -> SIGILL at `import salt` (via py_lap_solver). Fix: fetch the
# pinned sdist, patch -march=native to the portable x86-64-v2 baseline (and
# -mcpu=native -> generic for arm), PROVE the patch took (grep gates), then
# build from the patched tree. The pyproject's own CMAKE_*_FLAGS_RELEASE are
# already portable (-O3 -fPIC).
RUN uv pip install "scikit-build-core<0.8" cmake ninja pyproject-metadata pathspec "pybind11[global]" && \
    curl -sSL -o /tmp/py_lap_solver-0.1.4.tar.gz \
      "https://files.pythonhosted.org/packages/29/e2/7969402f59360378fa5bbd978daf8999155e78de823614546d8af80c2683/py_lap_solver-0.1.4.tar.gz" && \
    tar -xzf /tmp/py_lap_solver-0.1.4.tar.gz -C /tmp && \
    sed -i -e 's/-march=native/-march=x86-64-v2/g' -e 's/-mcpu=native/-mcpu=generic/g' \
      /tmp/py_lap_solver-0.1.4/CMakeLists.txt && \
    grep -q -- '-march=x86-64-v2' /tmp/py_lap_solver-0.1.4/CMakeLists.txt && \
    ! grep -qE -- '-(march|mcpu)=native' /tmp/py_lap_solver-0.1.4/CMakeLists.txt && \
    uv pip install --no-build-isolation --no-deps /tmp/py_lap_solver-0.1.4 && \
    rm -rf /tmp/py_lap_solver-0.1.4 /tmp/py_lap_solver-0.1.4.tar.gz

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
