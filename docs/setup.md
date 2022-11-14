To use the framework, you can either use the prebuilt docker containers, or create your own conda or venv environment.
You should set up the package from a powerful machine with access to a GPU.

### Get the Code

Start by cloning the repo. If you plan to contribute to the repo, you should work from a [fork](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html), rather than cloning the below link.

```bash
git clone https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt.git
cd salt
```

### Create Environment

You can install salt within a virtual environment or a docker image.
The recommended workflow is to set the package up using conda.

=== "conda"

    After cloning the repo, you will need to set up conda if you don't already have it installed.

    ??? info "Check for an existing conda installation"

        Your institute cluster may already have a managed conda installation present,
        please consult any local experts or documentation to find out whether this is the case.

        If already present you should skip the installation, and instead just create a new environment.

    You can either perform a manual installation by following the
    [conda](https://docs.conda.io/)/[mamba](https://mamba.readthedocs.io/) documentation,
    or use the provided script is provided which will install mamba (a faster, drop-in replacement for conda).
    You can run the script with

    ```bash
    source setup/setup_conda.sh
    ```

    Once you have conda installed, you can instead create a fresh python environment using

    ```
    conda create -n salt python=3.10
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

    Prebuilt docker images are an easy way to use salt, but can also be a bit more cumbersome than other approaches.
    You can run the prebuilt docker images using [singulartiy](https://sylabs.io/guides/latest/user-guide/).
    It is recommended to set `SINGULARITY_CACHEDIR` in your `~/.bashrc` to make sure images are pulled to a directory with enough free space.

    ```bash
    export SINGULARITY_CACHEDIR=<some path>/.singularity/
    ```

    Next, pull the image:

    ```bash
    singularity pull --docker-login \
        $SINGULARITY_CACHEDIR/salt.simg \
        docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/salt:latest
    ```

    Once you have the image locally, you can run it with

    ```bash
    singularity exec -ce --nv --bind $PWD \
        $SINGULARITY_CACHEDIR/salt.simg bash
    ```
    In order to mount a directory to the image when running `singularity exec`, use the `--bind <path>` argument


### Install the salt package

Once inside your container or virtual environment, you can install the `salt` package and it's dependencies via `pip` using

```bash
pip3 install -e .
```

To verify your installation, you can run the [test suite](contributing.md#test-suite).

??? info "Installation problems"

    If you get an `error: can't create or remove files in install directory` when installing
    or get `ModuleNotFoundError: No module named 'salt'` when trying to run the code,
    then you may need to install the package using the setup script, rather than directly using `pip`.

    ```bash
    source setup/install.sh
    ```

??? info "Installing `h5ls`"

    If you set up with conda, you can run

    ```bash
    mamba install h5utils
    ```

    to install the `h5ls` command.
    The `h5utils` is already present in the docker image.
