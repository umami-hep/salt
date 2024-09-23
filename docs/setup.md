To use the framework, you can either use the prebuilt docker containers, or create your own conda or venv environment.
You should set up the package from a powerful machine with access to a GPU.

### Get the Code

Start by cloning the repo.
If you plan to contribute to the repo, you should work from a [fork](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html), **instead of** cloning the below link.

```bash
git clone https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt.git
cd salt
```

!!! info "You can skip this step if you install Salt directly from PyPI (see below)"


### Create Environment

You can install salt within a virtual environment or a docker image.
The recommended workflow is to set the package up using conda.
Salt requires Python 3.9, 3.10 or 3.11.

=== "conda"

    After cloning the repo, you will need to set up conda/mamba if you don't already have it installed.

    ??? info "Check for an existing conda installation"

        Your institute cluster may already have a managed conda/mamba installation present,
        please consult any local experts or documentation to find out whether this is the case.

        If already present you should skip the installation, and instead just create a new environment.

    You can either perform a manual installation by following the
    [conda](https://docs.conda.io/)/[mamba](https://mamba.readthedocs.io/) documentation,
    or use the provided script is provided which will install mamba (a faster, drop-in replacement for conda).
    You can run the script with

    ```bash
    source setup/setup_conda.sh
    ```

    Once you have mamba installed, you can instead create a fresh python environment using

    ```
    mamba create -n salt python=3.11
    ```

    To activate it, just run

    ```bash
    conda activate salt
    ```

=== "venv"

    [venv](https://docs.python.org/3/library/venv.html) is a lightweight solution for creating virtual python environments, however it is not as fully featured as a fully fledged package manager such as conda.
    Create a fresh virtual environment and activate it using

    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

=== "singularity"

    Prebuilt docker images are an easy way to use salt, but can also be a bit less flexible than other approaches.
    You can run the prebuilt docker images using [singulartiy](https://sylabs.io/guides/latest/user-guide/).

    The first step is to decide which image you want to use.
    You can either pull an image locally, or use the unpacked images hosted on CVMFS.
    The latter is faster, but requires a CVMFS connection.

    === "Use the image from CVMFS"

        You only need to read this if you aren't manually pulling the Salt image yourself.

        The Salt singularity images are hosted on CVMFS.
        If you have a good connection to CVMFS, using this option can be faster than manually pulling the image.
        The images are located in
        `/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/`

        You can run the latest image using

        ```bash
        singularity shell -e --nv --bind $PWD \
            /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/salt:latest/
        ```

        The image comes with salt installed under `/salt/`, but if you want an editable install, you can follow the package install instructions [below](contributing.md#install-the-salt-package).

    === "Pull the image"

        You only need to read this if you want to pull the Salt image yourself, rather than using the unpacked image from CVMFS.
        This approach is slower than using the CVMFS image.

        The first step is to ensure that the `SINGULARITY_CACHEDIR` environment variable is set to a directory with plenty of free space.
        You may want to add the following lin to your `~/.bashrc` to make sure the variable is consistently set when you log in.

        ```bash
        export SINGULARITY_CACHEDIR=<some path>/.singularity/
        ```

        Next, pull the image:

        ```bash
        singularity pull --docker-login \
            $SINGULARITY_CACHEDIR/salt.simg \
            docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/salt:latest
        ```

        You can then run the image

        ```bash
        singularity shell -e --nv --bind $PWD \
            $SINGULARITY_CACHEDIR/salt.simg
        ```

        The image comes with salt installed under `/salt/`, but if you want an editable install, you can follow the package install instructions [below](contributing.md#install-the-salt-package).


    --------------------------------------------------------

    ??? info "`singularity shell` arguments"

        An explanation of the different arguments and flags is given [here](https://docs.sylabs.io/guides/latest/user-guide/cli/singularity_shell.html).

        In short, `--nv` is used for GPU support, `-e` ensures environment variables are not carried over to the image environment, and `--bind <path>` is used to mount a directory to the image.
        For convenience, you may wish to specify e.g. `--bind $PWD,/eos,/cvmfs`.

    Make sure you bind the directory in which you cloned the Salt repository and `cd` there after spinning up the image.
    This is required to to install the salt package, which is the next step.
    You may also wish to bind the directories containing your training files.

    Please note that if you want an editable install, you need to run the installation command below each time you open a new singularity shell.


### Install the salt package

Once inside your container or virtual environment and in the top level directory of the cloned repo, you can install the `salt` package and it's dependencies via `pip` using

=== "From source (recommended)"

    Cloning the repo and installing the package from source is the recommended way to install Salt.
    This allows you to easily modify configs and code and have the changes reflected in the package.

    ```bash
    python -m pip install -e .
    ```

=== "From PyPI"

    Salt is [available on PyPI](https://pypi.org/project/salt-ml/),
    so you can also install with

    ```bash
    python -m pip install salt-ml
    ```

To verify your installation, you can run the [test suite](contributing.md#test-suite).

??? failure "`ModuleNotFoundError` or `error: can't create or remove files in install directory` problems"

    If you get an `error: can't create or remove files in install directory` when installing
    or get `ModuleNotFoundError: No module named 'salt'` when trying to run the code,
    then you may need to install the package using the setup script, rather than directly using `pip`.

    ```bash
    source setup/install.sh
    ```

??? failure "`ERROR: Could not build wheels for jsonnet` during `pip install`"
    
    If you see the following message when running `pip install`:
    ```
    Failed to build jsonnet
    ERROR: Could not build wheels for jsonnet, which is required to install pyproject.toml-based projects
    ```
    You need to first install `jsonnet` via conda with
    ```
    conda install jsonnet
    ```
    and then re-run `pip install`.


??? failure "`RuntimeError: The NVIDIA driver on your system is too old` when running salt"
    
    If you see the following error when running `salt fit`, then you need to install suitable pytorch version.
    You can read about available versions [here](https://pytorch.org/get-started/locally/).

    First, create a new conda environment and activate it.
    Assuming, you have chosen `pytorch-cuda=11.8`, run in the new conda environment:
    ```
    mamba install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

    and then re-run `pip install`.


??? info "Installing `h5ls`"

    If you set up with conda/mamba, you can run

    ```bash
    mamba install h5utils
    ```

    to install the `h5ls` command.
    The `h5utils` is already present in the docker image.



### Setup Logging

Salt has the potential to supports any logging framework also supported by Lightning.
At the moment only comet is supported.

#### Comet

To use the [comet](https://www.comet.ml/) logger, you need to make an account with comet and [generate an API key](https://www.comet.ml/docs/quick-start/#getting-your-comet-api-key).
You also need to create a [workspace](https://www.comet.ml/docs/user-interface/#workspaces).
Next save the API key and the workspace name in environment variables called `COMET_API_KEY` and `COMET_WORKSPACE`.
These variables are automatically read by the comet, see [here](https://www.comet.com/docs/v2/guides/tracking-ml-training/configuring-comet/#configure-comet-through-environment-variables) for more info.
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
