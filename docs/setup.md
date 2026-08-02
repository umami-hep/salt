To use the framework, you can either use the prebuilt docker containers, or create your own uv or conda environment.
You should set up the package from a powerful machine with access to a GPU.

### Get the Code

Start by cloning the repo.
If you plan to contribute to the repo, you should work from a [fork](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html), **instead of** cloning the below link.

```bash
git clone https://gitlab.cern.ch/aft/algorithms/salt.git
cd salt
```

!!! info "You can skip this step if you install Salt directly from PyPI (see below)"


### Create Environment

You can install salt within a virtual environment or a docker image.
The recommended workflow is to create the Python environment and install Salt with `uv sync`.
Salt requires Python 3.10 to 3.14.

=== "uv"

    [uv](https://docs.astral.sh/uv/) is the recommended way to create the Python environment and install Salt.
    After cloning the repo, install uv if it is not already available:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    Then create and activate a fresh environment:

    ```bash
    uv venv --python 3.12
    source .venv/bin/activate
    ```

=== "conda"

    Conda/mamba remains useful on clusters with managed installations or when you need non-Python system packages.
    After cloning the repo, you will need to set up conda/mamba if you don't already have it installed.

    ??? info "Check for an existing conda installation"

        Your institute cluster may already have a managed conda/mamba installation present,
        please consult any local experts or documentation to find out whether this is the case.

        If already present you should skip the installation, and instead just create a new environment.

    You can either perform a manual installation by following the
    [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) documentation,
    or use the provided setup script, which can be run with

    ```bash
    source setup/setup_conda.sh
    ```

    Once you have mamba installed, you can instead create a fresh python environment using

    ```
    mamba create -n salt python=3.12
    ```

    To activate it, just run

    ```bash
    mamba activate salt
    ```

    If uv is not already available, install it into the environment with

    ```bash
    mamba install uv -c conda-forge
    ```

=== "apptainer"

    Prebuilt docker images are an easy way to use salt, but can also be a bit less flexible than other approaches.
    You can run the prebuilt docker images using [apptainer](https://apptainer.org/docs/user/latest/).

    ### Which image do I need?

    Salt ships **two** images, differing only in which GPU architectures the bundled
    PyTorch was compiled for. Rather than work it out from a table, ask the machine:

    ```bash
    ./setup/valid-container-versions
    ```

    It needs only `nvidia-smi` — no python, no torch, no container — so you can run it
    before pulling anything. It reports your GPU, its compute capability, which images
    will work, whether flash-attention is available, and the exact pull command:

    ```
    GPU: NVIDIA GeForce RTX 5060 Ti
      compute capability : 12.0  (sm_120)
      usable images      : T2B only
                           (V2H tops out at sm_90 — no Blackwell support)
      flash-attention    : supported
    ```

    For reference, the coverage it encodes:

    | Your GPU | Image |
    |---|---|
    | Tesla V100 / V100S | `V2H` **only** |
    | Tesla T4 | either |
    | A100 | either |
    | H100 / H200 / GH200 | either |
    | RTX 40xx / 50xx, B100 / B200 | `T2B` **only** |

    `V2H` covers **V**olta→**H**opper, `T2B` covers **T**uring→**B**lackwell. Most GPUs are
    served by both; only V100 and the newest consumer/datacenter cards are restricted to
    one. There is no single image covering both ends, because upstream PyTorch dropped
    Volta from the same wheels that added Blackwell.

    !!! warning "Using the wrong image fails silently at first"

        If the image does not cover your GPU, `torch.cuda.is_available()` still returns
        `True` — and then **every** kernel launch fails with
        `no kernel image is available for execution on the device`. The error appears at
        your first real operation, not at import, so it can look like a salt bug.

        Check before you train:

        ```bash
        python -c "import torch; print(torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())"
        ```

        Your device's capability must appear in the arch list. `(12, 0)` against a list
        ending at `sm_90` means you want the `T2B` image.

    A note on flash-attention: it requires Ampere (sm_80) or newer, so on V100 and T4 salt
    automatically uses its standard attention path in both images. This is expected and not
    an error.

    Next, decide how to obtain the image.
    You can either pull an image locally, or use the unpacked images hosted on CVMFS.
    The latter is faster, but requires a CVMFS connection.

    === "Use the image from CVMFS"

        You only need to read this if you aren't manually pulling the Salt image yourself.

        The Salt apptainer images are hosted on CVMFS.
        If you have a good connection to CVMFS, using this option can be faster than manually pulling the image.
        The images are located in
        `/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/aft/algorithms/`

        You can run an image using

        ```bash
        # V100 ... H100/H200
        apptainer shell -e --nv --bind $PWD \
            /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/aft/algorithms/salt:V2H/

        # T4 ... Blackwell (RTX 40xx/50xx, B100/B200)
        apptainer shell -e --nv --bind $PWD \
            /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/aft/algorithms/salt:T2B/
        ```

        See [the table above](#which-image-do-i-need) if you are unsure which one you want.

        The image comes with salt installed under `/salt/`, but if you want an editable install, you can follow the package install instructions [below](contributing.md#install-the-salt-package).

    === "Pull the image"

        You only need to read this if you want to pull the Salt image yourself, rather than using the unpacked image from CVMFS.
        This approach is slower than using the CVMFS image.

        The first step is to ensure that the `APPTAINER_CACHEDIR` environment variable is set to a directory with plenty of free space.
        You may want to add the following lin to your `~/.bashrc` to make sure the variable is consistently set when you log in.

        ```bash
        export APPTAINER_CACHEDIR=<some path>/.apptainer/
        ```

        Next, pull the image for your GPU — `V2H` or `T2B`, see
        [the table above](#which-image-do-i-need):

        ```bash
        TAG=V2H   # or T2B for RTX 40xx/50xx and B100/B200

        apptainer pull --docker-login \
            $APPTAINER_CACHEDIR/salt-${TAG}.simg \
            docker://gitlab-registry.cern.ch/aft/algorithms/salt:${TAG}
        ```

        You can then run the image

        ```bash
        apptainer shell -e --nv --bind $PWD \
            $APPTAINER_CACHEDIR/salt-${TAG}.simg
        ```

        Keeping the tag in the filename matters if you ever pull both — a bare `salt.simg`
        gives no way to tell which architectures it covers once it is on disk.

        The image comes with salt installed under `/salt/`, but if you want an editable install, you can follow the package install instructions [below](contributing.md#install-the-salt-package).


    --------------------------------------------------------

    ??? info "`apptainer shell` arguments"

        An explanation of the different arguments and flags is given [here](https://apptainer.org/docs/user/main/cli/apptainer_shell.html).

        In short, `--nv` is used for GPU support, `-e` ensures environment variables are not carried over to the image environment, and `--bind <path>` is used to mount a directory to the image.
        For convenience, you may wish to specify e.g. `--bind $PWD,/eos,/cvmfs`.

    Make sure you bind the directory in which you cloned the Salt repository and `cd` there after spinning up the image.
    This is required to to install the salt package, which is the next step.
    You may also wish to bind the directories containing your training files.

    Please note that if you want an editable install, you need to run the installation command below each time you open a new apptainer shell.

=== "lxplus (CERN)"

    On CERN's `lxplus`, the login nodes have no usable GPU — you train on the
    [HTCondor batch farm](https://batchdocs.web.cern.ch/gpu/index.html), the
    sanctioned route to a decent GPU at CERN. The helper `setup/salt-lxplus-gpu`
    submits GPU jobs for you; it runs salt two ways, **container-first**:

    - **Container (recommended).** A single Apptainer `.sif` is one loop-mounted
      file, so it avoids the per-job small-file penalty EOS FUSE imposes on a venv
      at *every* job's import time. This is the fast, reproducible default.
    - **Venv (fallback).** A `uv`-built virtualenv on shared storage — useful if you
      can't use a container or want an editable local install.

    #### Container path (recommended)

    **1. Get the image onto EOS (once).** Pull the published salt image:

    ```bash
    cd /eos/user/${USER:0:1}/$USER          # EOS home (batch-worker-visible)
    git clone https://gitlab.cern.ch/aft/algorithms/salt.git && cd salt
    # pull salt:latest to /eos/user/<i>/<user>/salt-containers/salt.sif
    setup/salt-lxplus-gpu pull
    ```

    (You can also point `SALT_LXPLUS_SIF` at any `.sif` you already have.)

    **2. Submit a GPU job:**

    ```bash
    export SALT_LXPLUS_SIF=/eos/user/${USER:0:1}/$USER/salt-containers/salt.sif
    setup/salt-lxplus-gpu submit config.yaml            # espresso (20 min)
    setup/salt-lxplus-gpu submit config.yaml longlunch  # 2 h walltime
    setup/salt-lxplus-gpu status                        # your condor jobs
    ```

    The job runs `apptainer exec --nv --bind /eos <sif> python -m salt.main
    fit --config …` on the GPU worker. To run a **local salt checkout** instead of
    the image's baked-in salt (developing a branch, or an image whose salt is an
    editable install), set `SALT_LXPLUS_SRC=/path/to/salt` — it is **bind-mounted
    over** the image's salt package (`SALT_LXPLUS_IMAGE_SRC`, default `/opt/salt-src`,
    is the image-side path). Bind-over is used rather than `PYTHONPATH` because
    `PYTHONPATH` does not reliably shadow an image's editable install. For extra
    importable packages *not* in the image (e.g. a custom reader), use
    `SALT_LXPLUS_PYPATH=/a,/b`; extra bind mounts, `SALT_LXPLUS_BIND=/a,/b`.

    !!! info "Where job files go (AFS) vs data (EOS)"

        Standard CERN batch schedds **reject `/eos` paths inside the submit file**
        (`Standard batch schedds cannot use /eos paths directly within the submit
        file` — this includes `arguments`, `environment`, `log`, and the job's
        working dir). So `salt-lxplus-gpu` puts the job **executable + `log`/`output`/
        `error` + config path + salt source on AFS** (`$HOME/.salt-lxplus/`, override
        with `SALT_LXPLUS_AFS`); the **SIF stays on `/eos`** and its path is passed to
        the job via a sourced AFS file (never a submit-file field). If you would rather
        keep everything on EOS, submit via the
        [EosSubmit schedds](https://batchdocs.web.cern.ch/local/eossubmit.html) instead.

    !!! warning "Reading datasets on EOS from inside the container"

        The SIF loop-mounts fine (apptainer reads it from *outside* the container), but
        **EOS's FUSE mount (`/eos/...` paths) is invisible *inside* apptainer** — a bind
        of `/eos` gives `Permission denied`. So:

        - **ROOT files** (uproot / ROOT readers): read them over **xrootd** with a
          `root://eosuser.cern.ch//eos/user/<i>/<user>/...` URL — this works inside the
          container without FUSE.
        - **Non-ROOT files** (HDF5, config YAML, checkpoints, MNIST idx): put them on
          **AFS** (readable in-container), or stage them into the job's node-local
          `$TMPDIR` *outside* the container with
          `xrdcp root://eosuser.cern.ch//eos/... "$TMPDIR"/` and bind `$TMPDIR`.

        `salt-lxplus-gpu` already binds AFS and runs your `SALT_LXPLUS_SRC` from there;
        keep configs + small datasets on AFS, or use `root://` URLs for big ROOT inputs.

    #### Venv path (fallback)

    Clone salt onto shared, batch-visible storage (the CUDA torch wheels total
    ~3–5 GB, too big for the 10 GB AFS home), then source the setup script:

    ```bash
    cd /eos/user/${USER:0:1}/$USER          # or an AFS workspace (see warning)
    git clone https://gitlab.cern.ch/aft/algorithms/salt.git && cd salt
    source setup/setup_lxplus.sh            # uv install, Python 3.14 venv, uv sync
    python -m salt.main --help         # verify the v2 entry point
    ```

    It auto-picks the install location: `$SALT_LXPLUS_DIR` (your override) → AFS
    workspace → EOS home → AFS home (only if ≥8 GB free); it never uses `/tmp`.
    On EOS it puts uv's cache on node-local `/tmp`, forces copy mode, and applies a
    `scikit-build-core<0.8` build constraint (a `py-lap-solver` cp314 build fix).
    Re-sourcing just re-activates the venv. With no `.sif` configured,
    `salt-lxplus-gpu submit` automatically uses this venv on the worker.

    !!! warning "Do not put the venv on AFS home or `/tmp`"

        AFS home is too small for the CUDA wheels. `/tmp` is **node-local** — the
        batch worker cannot see it, so the job will not find the venv. Use an AFS
        workspace (request one free at the CERN Resources Portal → AFS Workspaces —
        best latency) or EOS home. EOS works out of the box but is FUSE-mounted, so
        `uv sync` is slower there.

    !!! tip "Long install over SSH?"

        The first venv `uv sync` on EOS pulls several GB and can take a while. On a
        flaky SSH connection run it inside `tmux`/`screen` (or `nohup`) so it
        survives a dropped session — the venv lands on shared storage either way,
        so just re-`source setup/setup_lxplus.sh` afterwards to re-activate.

    #### Interactive vs batch, and flavours

    Use `salt-lxplus-gpu shell [flavour]` for a live GPU node (quick checks /
    debugging — you wait for a slot and lose it on logout); use `submit` for real
    training since it survives logout. Attach to a *running* batch job with
    `condor_ssh_to_job <jobid>`. The **flavour** is the walltime bucket —
    `espresso` (20 min) schedules fastest (ideal for smoke tests), stepping up to
    `longlunch` (2 h), `workday` (8 h), `tomorrow` (1 day). Both paths request one
    GPU with compute capability ≥ **7.0** and ≥ 12 GB memory via
    [`setup/lxplus_gpu.sub`](https://gitlab.cern.ch/aft/algorithms/salt/-/blob/main/setup/lxplus_gpu.sub);
    edit that file (or pass extra `-append` macros) to change the resource request.

    !!! tip "Capability floor vs. queue time"

        The default floor is `gpus_minimum_capability = 7.0`, which keeps the
        plentiful **V100 (7.0)** and **T4 (7.5)** slots in play. Requiring **8.0**
        restricts you to the **A100** pool, which is small and heavily contended —
        a job can sit idle for a long time even when hundreds of GPU slots are free.
        Raise the floor (`SALT_LXPLUS_GPU_CAPABILITY=8.0`) **only** if your model
        needs flash-attention (SM 80+); a plain torch-math training does not. Check
        current availability with
        `condor_status -compact -constraint 'TotalGpus > 0'`.


### Install the salt package

Once inside your container or virtual environment and in the top level directory of the cloned repo, you can install the `salt` package and its dependencies with `uv sync` using

=== "From source (recommended)"

    Cloning the repo and installing the package from source is the recommended way to install Salt.
    This allows you to easily modify configs and code and have the changes reflected in the package.

    ```bash
    uv sync
    ```

    To install special packages for development or special trainings (like flash attention), you need
    to adapt the command slightly:

    ```bash
    uv sync --group dev --extra muP --extra flash
    ```

    The `dev` dependency group and the `muP` and `flash` extras install additional packages, which can be needed for
    certain purposes.

    ??? failure "`The detected CUDA version mismatches the version that was used to compilePyTorch`"

        This failure is due to an issue with the installation of flash attention. To circumvent this,
        remove `flash` from the additional package installation and re-run it. See instructions below
        for how to install flash attention properly.

=== "From PyPI"

    Salt is [available on PyPI](https://pypi.org/project/salt-ml/),
    so you can also install with

    ```bash
    uv tool install salt-ml
    ```

To verify your installation, you can run the [test suite](contributing.md#test-suite).

??? failure "`ModuleNotFoundError` or `error: can't create or remove files in install directory` problems"

    If you get an `error: can't create or remove files in install directory` when installing
    or get `ModuleNotFoundError: No module named 'salt'` when trying to run the code,
    then you may need to install the package using the setup script, rather than directly using `uv sync`.

    ```bash
    source setup/install.sh
    ```

??? failure "`ERROR: Could not build wheels for jsonnet` during `uv sync`"

    If you see the following message when running `uv sync`:
    ```
    Failed to build jsonnet
    ERROR: Could not build wheels for jsonnet, which is required to install pyproject.toml-based projects
    ```
    You need to first install `jsonnet` via conda with
    ```
    conda install jsonnet
    ```
    and then re-run `uv sync`.


??? failure "`RuntimeError: The NVIDIA driver on your system is too old` when running salt"

    If you see the following error when running `salt fit`, then you need to install suitable pytorch version.
    You can read about available versions [here](https://pytorch.org/get-started/locally/).

    First, create a new conda environment and activate it.
    Assuming, you have chosen `pytorch-cuda=11.8`, run in the new conda environment:
    ```
    mamba install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

    and then re-run `uv sync`.


??? info "Installing `h5ls`"

    If you set up with conda/mamba, you can run

    ```bash
    mamba install h5utils
    ```

    to install the `h5ls` command.
    The `h5utils` is already present in the docker image.

### Install FlashAttention
FlashAttention is an attention algorithm which greatly reduces the computational overhead of the attention mechanism of
transformer models. To get FlashAttention installed, you need to find a prebuilt wheel version of it that fits your needs.
To do so, visit the [flash-attention pre-build wheels GitHub repo](https://github.com/mjun0812/flash-attention-prebuild-wheels).
You will have to go to another website where you can fill in information about your platform, the flash
attention version you want, and the versions of python, pytorch, and CUDA on your system. You can then copy
the URL link to the correct wheel and install it in the synced environment:

```bash
uv run --no-sync python -m pip install "<URL>"
```

This will install the correct FlashAttention version and you should not get any errors or warnings related to FlashAttention.


### Setup Logging

Salt has the potential to support any logging framework that is also supported by Lightning.
At the moment only comet is supported.

#### Comet

To use the [comet](https://www.comet.ml/) logger, you need to make an account with comet and [generate an API key](https://www.comet.ml/docs/quick-start/#getting-your-comet-api-key).
You also need to create a [workspace](https://www.comet.ml/docs/user-interface/#workspaces).
Next save the API key and the workspace name in environment variables called `COMET_API_KEY` and `COMET_WORKSPACE`.
These variables are automatically read by comet, see [here](https://www.comet.com/docs/v2/guides/tracking-ml-training/configuring-comet/#configure-comet-through-environment-variables) for more info.
Consider adding these variables to your [bashrc](https://www.journaldev.com/41479/bashrc-file-in-linux).

??? info "Add the environment variable to your bashrc"

    To ensure the environment variables are defined every time you log in,
    you can add the definitions to your bashrc.
    Simply add the lines

    ```bash
    export COMET_API_KEY="<Your API Key>"
    export COMET_WORKSPACE="<Your Workspace Name>"
    ```

    to your `~/.bashrc` file.
    If no such file exists, create one in your home directory.
