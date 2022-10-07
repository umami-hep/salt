To use the framework, you can either use the prebuilt docker containers, or create your own venv or conda environment.
Requirements for training the GNN are a machine lots of CPUs, GPUs, and RAM.

Start by cloning the repo. If you plan to contribute to the repo, you should work from a [fork](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html).
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
    $SINGULARITY_CACHEDIR/gnn_image.simg \
    docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/salt:latest
```

As the repository is internal at the moment, you'll need to enter your CERN credentials when pulling.

Once you've pulled the docker image locally, you can run it with:

```bash
singularity exec -ce --nv --bind $PWD \
    --env-file ./setup/singularity_envs.sh \
    $SINGULARITY_CACHEDIR/gnn_image.simg bash
```
In order to mount a directory to the image when running `singularity exec`, use the `--bind <path>` argument

Within the image, install the salt package.

```bash
pip3 install -e .
```


### Conda

After cloning the repo, you can set up conda

Source the conda setup script (which will actually install [mamba](https://mamba.readthedocs.io/en/latest/index.html))
```bash
source ./setup/setup_conda.sh
```
This script will install mamba, and also create an empty python environment named `salt`.

The script should activate the newly created file for you, if you want to activate it yourself, just run

```bash
conda activate gnn
```

Finally, install the salt package.

```bash
pip3 install -e .
```


### Venv


Create a fresh virtual environment and activate it using

```bash
python3 -m venv env
source env/bin/activate
```

Finally, install the salt package.

```bash
pip3 install -e .
```
