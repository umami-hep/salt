

### Setup Logging

Salt has the potential to supports any logging framework also supported by PTL.
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

A simple config file is provided [here]({{repo_url}}-/blob/main/salt/configs/simple.yaml)
You can start a training using this config with the `train.py` script.

```bash
python train.py fit --config configs/simple.yaml
```

The first argument `fit` specifies you want to train the model, rather than `test` over some orthogonal dataset.
The `--config` argument specifies the config file to use.
It's possible to specify more than one configuration file, the CLI will merge them [automatically](https://pytorch-lightning.readthedocs.io/en/latest/cli/lightning_cli_advanced.html#compose-yaml-files).

??? info "Running a test training"

    To test the training script you can run use the `--trainer.fast_dev_run` flag
    which will run over a small number of training and validation batches and then
    exit.

    ```bash
    python train.py fit --config configs/simple.yaml --trainer.fast_dev_run 5
    ```

    Logging and checkpoint are suppressed when using this falg.

You can also configure the training directly through CLI arguments.
For a full list of available arguments run

```bash
python train.py fit --help
```

By default the config will try to use the first available GPU, but
you can specify which ones to use with the `--trainer.devices` flag.
Take a look [here](https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu_basic.html#train-on-multiple-gpus) to learn more about the different ways you can specify which GPUs to use.

??? warning "Check GPU usage before starting training."

    You should check with `nvidia-smi` that any GPUs you use are not in use by some other user before starting training.

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

Most HPC systems will have dedicated fast storage. Loading training data from these drives can significantly improve training times.

If you have enough RAM, you can load the training data into shared memory before starting training copying the training files to a path in `/dev/shm/`. Make sure to clean up your files in when you are done.


### Slurm Batch

Those at institutions with Slurm managed GPU batch queues can submit training jobs using

```bash
sbatch submit_slurm.sh
```

If training ends prematurely, you can be left with floating worker processes on the node which can clog things up for other users.
You should check for this after running training jobs which are cancelled or fail.
To do so, `srun` into the affected node using

```bash
srun --pty --cpus-per-task 2 -w compute-gpu-0-3 -p GPU bash
```

and then kill your running processes using

```bash
pkill -u <username> -f train.py -e
```

### Resuming Training

Model checkpoints are saved under `logs/` (need to work on the dir names...).
You can resume the full training state from a `.ckpt` checkpoint file by using the `--ckpt_path` argument.

```bash
python train.py fit --config path/to/config.yaml --ckpt_path path/to/checkpoint.ckpt
```
