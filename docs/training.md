

### Setup Logging

Salt has the potential to supports any logging framework also supported by PTL.
At the moment only comet is supported.

#### Comet

To use the [comet](https://www.comet.ml/) logger, you need to make an account with comet and [generate an API key](https://www.comet.ml/docs/quick-start/#getting-your-comet-api-key).
You also need to create a [workspace](https://www.comet.ml/docs/user-interface/#workspaces).
Next save the API key and the workspace name in environment variables called `PL_TRAINER__LOGGER__API_KEY` and `PL_TRAINER__LOGGER_WORKSPACE`.
These are named in such a way to be automatically read by the framework (it's possible to configure other aspects of the training using environment variables if you wish).
 (consider adding these definitions to your [bashrc](https://www.journaldev.com/41479/bashrc-file-in-linux)).

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

The first argument `fit` specifies you want to train the model, rather than run validation or inference.
The `--config` argument specifies the config file to use.
It's possible to specify more than one configuration file, the CLI will merge them [automatically](https://pytorch-lightning.readthedocs.io/en/latest/cli/lightning_cli_advanced.html#compose-yaml-files).

You can also configure the training directly through CLI arguments.
For a full list of available arguments run

```bash
python train.py fit --help
```

???+ warning "Check GPU usage before starting training."

    You should check with `nvidia-smi` that any GPUs you use are not in use by some other user before starting training.

Model checkpoints are saved under `logs/` (need to work on the dir names...).


### Training Tips

During training, data is loaded using worker processes.
The number of worker processes you use will be loosely related to number of CPUs available on your machine.
You can find out the number of CPUs available on your machine by running

```bash
cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1
```

Some other tips to make training as fast as possible are listed below.

**Worker counts**

If you have exclusive access to your machine, you should set `num_workers` equal to the number of CPUs on your machine.

**Moving data closer to the GPU**

- Most HPC systems will have dedicated fast storage. Loading training data from these drives can significantly improve training times. To automatically copy files to a drive, you can use the `move_files_temp` config flag. The files will be removed after training, but if the training job crashes is cancelled this may not happen. You should make sure to clean up your files if you using this setting.
- If you have enough RAM, you can load the training data into shared memory before starting training by setting `move_files_temp` to some path in `/dev/shm/`. Again, make sure to clean up your files in `/dev/shm/` after training as the script may fail to do this for you.


### Slurm GPU Clusters

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
