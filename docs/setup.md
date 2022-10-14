# Setup

To use the framework, you can either use the prebuilt docker containers, or create your own conda or venv environment.
You should set up the package from a powerful machine with access to a GPU.

Start by cloning the repo. If you plan to contribute to the repo, you should work from a [fork](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html), rather than cloning the below link.
```bash
git clone https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt.git
cd salt
```

### Prebuilt Image

Prebuilt docker images are the recommended way to work with the GNN. You can run the prebuilt docker images using [singulartiy](https://sylabs.io/guides/latest/user-guide/).
It is recommended to set `SINGULARITY_CACHEDIR` in your `~/.bashrc` to make sure images are pulled to a directory with enough free space.

```bash
export SINGULARITY_CACHEDIR=<some path>/.singularity/
```

Next, you can pull the image:

```bash
singularity pull --docker-login \
    $SINGULARITY_CACHEDIR/salt.simg \
    docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/salt:latest
```

As the repository is internal at the moment, you'll need to enter your CERN credentials when pulling.

Once you've pulled the docker image locally, you can run it with:

```bash
singularity exec -ce --nv --bind $PWD \
    --env-file ./setup/singularity_envs.sh \
    $SINGULARITY_CACHEDIR/salt.simg bash
```
In order to mount a directory to the image when running `singularity exec`, use the `--bind <path>` argument


### Local Environment

If you prefer to work outside of a container, you can setup up the code with conda or Python's venv.

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

=== "venv"

    Create a fresh virtual environment and activate it using

    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

## Install the salt package

Once inside your container or virtual environment, you can install the `salt` package and it's dependencies via `pip` using
```bash
pip3 install -e .
```

???+ info "Installation problems"

    If you get an `error: can't create or remove files in install directory` when installing
    or get `ModuleNotFoundError: No module named 'salt'` when trying to run the code,
    then you may need to install the package using the setup script, rather than directly using `pip`.

    ```bash
    source setup/install.sh
    ```
