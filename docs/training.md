Before getting started with this section, please make sure you have read the documentation on [setup](setup.md) and [preprocessing](preprocessing.md).


### Training

Training is fully configured via a YAML config file and a CLI powered by [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/latest/cli/lightning_cli.html#lightning-cli).
This allows you control all aspects of the training from config or directly via command line arguments.

The configuration is split into two parts.
The [`base.yaml`]({{repo_url}}-/blob/main/salt/configs/base.yaml) config contains model-independent information like the input file paths and batch size.
This config is used by default for all trainings without you having to explicitly specify it.
Meanwhile the model configs, for example [`gnn.yaml`]({{repo_url}}-/blob/main/salt/configs/gnn.yaml) contain a full description of a specific model, including a list of input variables used.
You can start a training for a given model by providing it as an argument to the `main.py` python script, which is also exposed through the command `salt`.

```bash
salt fit --config configs/GN2.yaml
```

The subcommand `fit` specifies you want to train the model, rather than [evaluate](evaluation.md) it.
It's possible to specify more than one configuration file, for example to override the values set in [`base.yaml`]({{repo_url}}-/blob/main/salt/configs/base.yaml).
The CLI will merge them [automatically](https://pytorch-lightning.readthedocs.io/en/latest/cli/lightning_cli_advanced.html#compose-yaml-files).

??? info "Running a test training"

    To test the training script you can run use the `--trainer.fast_dev_run` flag
    which will run over a small number of training and validation batches and then
    exit.

    ```bash
    salt fit --config configs/GN2.yaml --trainer.fast_dev_run 2
    ```

    Logging and checkpoint are suppressed when using this flag.


#### Using the CLI

You can also configure the training directly through CLI arguments.
For a full list of available arguments run

```bash
salt fit --help
```


#### Choosing GPUs

By default the config will try to use the first available GPU, but
you can specify which ones to use with the `--trainer.devices` flag.
Take a look [here](https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu_basic.html#train-on-multiple-gpus) to learn more about the different ways you can specify which GPUs to use.

??? warning "Check GPU usage before starting training"

    You should check with `nvidia-smi` that any GPUs you use are not in use by some other user before starting training.

#### Resuming Training

Model checkpoints are saved in timestamped directories under `logs/`.
These directories also get a copy of the fully merging training config (`config.yaml`), and a copy of the umami scale dict.
To resume a training, point to a previously saved config and a `.ckpt` checkpoint file by using the `--ckpt_path` argument.

```bash
salt fit --config logs/run/config.yaml --ckpt_path path/to/checkpoint.ckpt
```

The full training state, including the state of the optimiser, is resumed.
The logs for the resumed training will be saved in a new directory, but the epoch count will continue from
where it left off.

Note that the [`OneCycleLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html) learning rate scheduler will set a pre-determined total number of steps in the cycle. In order to resume a training which exceeds the maximum number of epochs, using e.g. `--trainer.max_epochs <value>`, you need to re-set the `--model.lrs_config.last_epoch 0` as well.

### Reproducibility

Reproducibility of tagger trainings is very important.
The following features try to ensure reproducibility without too much overhead.

#### Commit Hashes & Tags

When training, the framework will complain if you have any uncommitted changes.
You can override this by setting the `--force` flag, but you should only do so if you are sure that preserving your training configuration is not necessary.
The commit hash used for the training is saved in the `metadata.yaml` for the training run.

Additionally, you can create and push a tag of the current state using the `--tag` argument.
This will push a tag to your personal fork of the repo.
The tag name is the same as the output directory name, which is generated automatically by the framework.

??? warning "Using `--force` will prevent a tag from being created and pushed"

    Try and avoid using `--force` if you can, as it hampers reproducibility.


#### Random Seeds

Training runs are reproducible thanks to the `--seed_everything` flag,
which is already set for you in the [`base.yaml`]({{repo_url}}-/blob/main/salt/configs/base.yaml) config.
The flat seeds all random number generators used in the training.
This means for example that weight initialisation and data shuffling happen in a deterministic way.

??? info "Stochastic operations can still lead to divergences between training runs"

    For more info take a look [here](https://pytorch.org/docs/stable/notes/randomness.html).



### Dataloading

Objects are loaded in weakly shuffled batches from the training file.
This is much more efficient than randomly accessing individual entries, which would be prohibitively slow.

Some other dataloading considerations are discussed below.

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



### Batch systems

#### HTCondor Batch

Those at institutions with HTCondor managed GPU batch queues can submit training jobs using

```bash
python submit/submit_htcondor.py --config configs/GN2.yaml
```

It is required to pass the path to the configuration file via the `--config` argument.
Further, it is possible to specify the environment in which the job will run with `-e` / `--environment`.
This depends how you installed salt and can either be a `local` environment, a `conda` environment
(which is assumed to be in the `salt/conda` directory) or a `singularity` container.

By default, all job jobmission files and batch logs will be stored in the `condor` directory
which is created upon job submission.

Running multiple trainings in parallel is possible by specifying different names for the jobs using the `-t` / `--tag` argument.

The job parameters such as memory requirements, number of GPUs and CPUs requested have to be modified in the file `submit/submit_htcondor.py`.

#### Slurm Batch

Those at institutions with Slurm managed GPU batch queues can submit training jobs using a very similar script.

All options described above for HTCondor and more (CPUs, GPUs, etc) are available as command line arguments. 

```bash
python submit/submit_slurm.py --config configs/GN2.yaml --tag test_salt --account MY-ACCOUNT --nodes 1 --gpus_per_node 2
```

The script submit/submit_slurm.py script itself can be modified if a required configuration is not supported in this way.

Where arguments need to agree between Slurm and Pytorch Lightning, such as ntasks-per-node for Slurm and trainer.devices for Lightning, this is handled by the script.

There is also an older submit/submit_slurm.sh bash script that is kept around for compatibility. Users are strongly encouraged to use the python script.

??? info "Cleaning up after interruption"

    If training is interrupted, you can be left with floating worker processes on the node which can clog things up for other users.
    You should check for this after running training jobs which are cancelled or fail.
    To do so, `srun` into the affected node using

    ```bash
    srun --pty --cpus-per-task 2 -w compute-gpu-0-3 -p GPU bash
    ```

    and then kill any remaining running python processes using

    ```bash
    pkill -u <username> -f salt -e
    ```


## Troubleshooting

If you encounter issues, as a first step you should try pulling the latest updates from `main` to see if your problem has been resolved.
If you need more help you can post on [mattermost](https://mattermost.web.cern.ch/aft-algs/channels/gnns).

### Slow Training

This section contains some suggestions for speeding up trainings.
Some external advice can be found [here](https://lightning.ai/docs/pytorch/stable/advanced/speed.html) and [here](https://lightning.ai/docs/pytorch/stable/levels/intermediate_level_13.html).

If you are not producing a "final" version of your model (i.e. with maximum possible performance), but instead are running some studies, you should consider the following:

- Limit the training statistics (e.g. 20M samples)
- Reduce the number of epochs you train for (e.g. 20 epochs)
- Remove any auxiliary tasks
- [Compile the model][compiled-models]

Other things you can always do:

- Use bfloat16 precision
- Use the maximum possible [batch size](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BatchSizeFinder.html)
- Increase your effective batch size by [accumulating gradients](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#accumulate-gradients)
- Ensure you have enough [workers for dataloading](worker-counts)
- Use newer GPUs if possible
- Use [multiple GPUs][choosing-gpus]
- Reduce the size of the model (in particular the number of layers)

### Confusing Errors

You might see confusing/cryptic errors when running on the GPU, for example

```
../aten/src/ATen/native/cuda/NLLLoss2d.cu:103: nll_loss2d_forward_kernel: block: [377,0,0], thread: [13,0,0] Assertion `t >= 0 && t < n_classes` failed.
```

Often, if you instead run on the CPU you will get a much more helpful error message.
Use `--trainer.accelerator=cpu` to run on the CPU instead of the GPU.


### NaNs

Salt will automatically check:

- That there are no `nan` inputs (see [salt.data.SaltDataset][salt.data.SaltDataset])
- That your normalisation paramters are finite in (see [salt.models.InputNorm][salt.models.InputNorm])

You may still encounter `nan` values in your outputs and losses.
Here are some mitigation strategies you can try:

- Make sure you have pulled the latest changes from `main`.
- Make doubly sure that your inputs are finite, even apply applying normalisation.
- Ensure you don't have unexpected non-finite labels.
- Try lowering your max learning rate in the `lrs_config`.
- If you apply very large loss weights in your task configs, these might contribute to large gradients, so you can try removing any loss weights provided to your [Tasks][salt.models.TaskBase].
- Check your training precision: if you have done the above and still have problems, you can try  `--trainer.precision=32` or `--trainer.precision=bf16-mixed`. See [here](https://lightning.ai/docs/pytorch/stable/common/trainer.html#precision) for more info. 
- Apply gradient clipping to negate the effects of exploding gradients. See [here] for more info.
- Auto detect gradient anomalies. See [here](https://lightning.ai/docs/pytorch/stable/debug/debugging_intermediate.html#detect-autograd-anomalies) for more info.
- If you are running on multiple GPUs, try running on a single GPU with `--trainer.devices=1`
