# Setup

To use the framework, you can either use the prebuilt docker containers, or create your own conda or venv environment.
You should set up the package from a powerful machine with access to a GPU.

Start by cloning the repo. If you plan to contribute to the repo, you should work from a [fork](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html), rather than cloning the below link.
```bash
git clone https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt.git
cd salt
```

### Create Environment

You can install salt within an environment or docker image.
The recommended workflow is to set the package up using conda.

=== "conda"

    After cloning the repo, you will need to set up conda if you don't already have it installed.
    A script is provided which will install [mamba](https://mamba.readthedocs.io/en/latest/index.html),
    and also create a fresh Python environment named `salt`.
    ```bash
    source ./setup/setup_conda.sh
    ```
    The script should activate the newly created environment for you.
    To activate it yourself, just run

    ```bash
    conda activate gnn
    ```

    If you already have conda installed, just run

    ```bash
    conda create -n salt python
    ```

    to create a fresh python environment.

=== "venv"

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

    Once you've pulled the image locally, you can run it with

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

To verify your installation, you can run the [test suite](contributing#test-suite).

???+ info "Installation problems"

    If you get an `error: can't create or remove files in install directory` when installing
    or get `ModuleNotFoundError: No module named 'salt'` when trying to run the code,
    then you may need to install the package using the setup script, rather than directly using `pip`.

    ```bash
    source setup/install.sh
    ```
