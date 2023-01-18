Before getting started with this section, please make sure you have read the [preprocessing documentation](preprocessing.md) which describes how to produce training files for the framework.

### Setup Logging

Salt has the potential to supports any logging framework also supported by Lightning.
At the moment only comet is supported.

#### Comet

To use the [comet](https://www.comet.ml/) logger, you need to make an account with comet and [generate an API key](https://www.comet.ml/docs/quick-start/#getting-your-comet-api-key).
You also need to create a [workspace](https://www.comet.ml/docs/user-interface/#workspaces).
Next save the API key and the workspace name in environment variables called `PL_TRAINER__LOGGER__API_KEY` and `PL_TRAINER__LOGGER_WORKSPACE`.
These are named in such a way to be automatically read by the framework (it's possible to configure other aspects of the training using environment variables if you wish).
Consider adding these variables to your [bashrc](https://www.journaldev.com/41479/bashrc-file-in-linux).

??? info "Add the environment variable to your bashrc"

    To ensure the environment variables are defined every time you log in,
    you can add the definitions to your bashrc.
    Simply add the lines

    ```bash
    export PL_TRAINER__LOGGER__API_KEY="my_api_key"
    export PL_TRAINER__LOGGER_WORKSPACE="my_workspace_name"
    ```

    to your `~/.bashrc` file.
    If no such file exists, create one in your home directory.

### Training

Training is fully configured via a YAML config file and a CLI powered by [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/latest/cli/lightning_cli.html#lightning-cli).
This allows you control all aspects of the training from config or directly via command line arguments.

The configuration is split into two parts.
The [`base.yaml`]({{repo_url}}-/blob/main/salt/configs/base.yaml) config contains model-independent information like the input file paths and batch size.
This config is used by default for all trainings without you having to explicitly specify it.
Meanwhile the model configs, for example [`gnn.yaml`]({{repo_url}}-/blob/main/salt/configs/gnn.yaml) contain a full description of a specific model.
You can start a training for a given model by providing it as an argument to the `main.py` python script, which is also exposed through the command `main`.

```bash
main fit --config configs/GN1.yaml
```

The subcommand `fit` specifies you want to train the model, rather than [evaluate](evaluation.md) it.
It's possible to specify more than one configuration file, for example to override the values set in [`base.yaml`]({{repo_url}}-/blob/main/salt/configs/base.yaml).
The CLI will merge them [automatically](https://pytorch-lightning.readthedocs.io/en/latest/cli/lightning_cli_advanced.html#compose-yaml-files).

??? info "Running a test training"

    To test the training script you can run use the `--trainer.fast_dev_run` flag
    which will run over a small number of training and validation batches and then
    exit.

    ```bash
    main fit --config configs/GN1.yaml --trainer.fast_dev_run 2
    ```

    Logging and checkpoint are suppressed when using this flag.


#### Using the CLI

You can also configure the training directly through CLI arguments.
For a full list of available arguments run

```bash
main fit --help
```

#### Choosing GPUs

By default the config will try to use the first available GPU, but
you can specify which ones to use with the `--trainer.devices` flag.
Take a look [here](https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu_basic.html#train-on-multiple-gpus) to learn more about the different ways you can specify which GPUs to use.

??? warning "Check GPU usage before starting training."

    You should check with `nvidia-smi` that any GPUs you use are not in use by some other user before starting training.

#### Reproducibility

Trainings are fully reproducible thanks to the `--seed_everything` flag.
This flag is set in the [`base.yaml`]({{repo_url}}-/blob/main/salt/configs/base.yaml) config.

### Dataloading

There are two types of dataloading modes available, configured by the `data.batched_read` config flag.
When this flag is `False`, individual jets are loaded from the training file randomly, and pytorch handles the batching behind the scenes.
This is inefficient as h5 files benefit from block reads.
Setting `data.batched_read` to `True` will read a full batch at a time from the input file.
This is much more efficient, but only implements "weak shuffling" in that while the different batches are shuffled, the same batches of jets are used epoch after epoch.
In practice, this is unlikely to make much difference to the training.

#### Worker Counts

During training, data is loaded using worker processes.
The number of workers used is specified by `data.num_workers`.
Increasing the worker count will speed up training until you max out your GPUs processing capabilities.
Test different counts to find the optimal value, or just set this to the number of CPUs on your machine.

??? info "Maximum worker counts"

    You can find out the number of CPUs available on your machine by running

    ```bash
    cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1
    ```

    You should not use more workers than this.
    If you use too few or too many workers, you will see a warning at the start of training.

#### Fast Disk Access

Most HPC systems will have dedicated fast storage.
Loading training data from these drives can significantly improve training times.
To temporarily copy training files into a target directory before training, use the
`--data.move_files_temp=/temp/path/` flag.

If you have enough RAM, you can load the training data into shared memory before starting training by setting `move_files_temp` to a path under `/dev/shm/<username>`.

??? warning "Ensure temporary files are removed"

    The code will try to remove the temporary files when the training is complete, but if the training is interrupted this may not happen.
    You should double check whether you need to manually remove the temporary files to avoid clogging up your system's RAM.

### Slurm Batch

Those at institutions with Slurm managed GPU batch queues can submit training jobs using

```bash
sbatch submit_slurm.sh
```

The submit script only supports running from a conda environment for now.
There are several options in the script which need to be tailored to make sure to make a look inside.

??? info "Cleaning up after interruption"

    If training is interrupted, you can be left with floating worker processes on the node which can clog things up for other users.
    You should check for this after running training jobs which are cancelled or fail.
    To do so, `srun` into the affected node using

    ```bash
    srun --pty --cpus-per-task 2 -w compute-gpu-0-3 -p GPU bash
    ```

    and then kill any remaining running python processes using

    ```bash
    pkill -u <username> -f python -e
    ```

### Resuming Training

Model checkpoints are saved in timestamped directories under `logs/`.
These directories also get a copy of the fully merging training config (`config.yaml`), and a copy of the umami scale dict.
To resume a training, point to a previously saved config and a `.ckpt` checkpoint file by using the `--ckpt_path` argument.

```bash
main fit --config logs/run/config.yaml --ckpt_path path/to/checkpoint.ckpt
```

The full training state, including the state of the optimiser, is resumed.
The logs for the resumed training will be saved in a new directory, but the epoch count will continue from
where it left off.

### Switching Attention Mechanisms

By default the [`GN1.yaml`]({{repo_url}}-/blob/main/salt/configs/GN1.yaml) config uses the scaled dot product attention found in transformers.
To switch to the GATv2 attention, add the [`GATv2.yaml`]({{repo_url}}-/blob/main/salt/configs/GATv2.yaml) config fragment.

```bash
main fit --config configs/GN1.yaml --config configs/GATv2.yaml
```

Note that the  `num_heads` and `head_dim` arguments must match those in the `gnn.init_args` config block.

### Excluding Variables from training

It is possible to provide a dictionary of variables not to include in the training, by adding a dictionary to the model.yaml file as

```yaml
data:
     exclude: {'jet': ['GN120220509_pb', 'GN120220509_pc', 'GN120220509_pu'],
              'track': ['jet_GN120220509_pb', 'jet_GN120220509_pc', 'jet_GN120220509_pu']}
```

If one wants, for instance, to exclude only 'track' variables, it is possible:

```yaml
data:
     exclude: {'track': ['dphi']}
```

Modification of the input_size model parameter is handled automatically by the framework.
