You should have already generated configs as discussed [here](https://ftag-gnn.docs.cern.ch/preprocessing/#generating-configs).

### Setup Logging

To use the [comet](https://www.comet.ml/) logger, you need to make an account with comet and [generate an API key](https://www.comet.ml/docs/quick-start/#getting-your-comet-api-key). Save this key in an environment variable called `COMET_API_KEY` in your shell, along with a `COMET_WORKSPACE` variable with the name of the [comet workspace](https://www.comet.ml/docs/user-interface/#workspaces) you create through the comet website (consider adding these definitions to your [bashrc](https://www.journaldev.com/41479/bashrc-file-in-linux)).

### Training modes

To load graphs during training, there are 3 dataloading approaches. 
The first, JIT, only needs the `.h5` files outputted by umami, which contain unstructured data about each jet which can be read and converted to graph format in the training loop.
The other two methods speed up the training by making use of prebuilt graphs. 
The steps to prebuild graphs are detailed in the preprocessing documentation [here](https://ftag-gnn.docs.cern.ch/preprocessing/#pre-building-graphs).

#### JIT Training
If you do not pre-build graphs, the JIT training mode builds graphs from the .h5 training dataset on the fly. This can mean lower memory use, but is also slower. 
To enable JIT training, set `graph_loading: jit` in the training config.

#### Prebuilt Graph Training
This training mode is used by setting `graph_loading: prebuilt` in the training config. In this mode, each worker available will load a single graph file at a time, only loading the next once it has iterated all graphs in the file. This mode can run significantly faster than the JIT mode, with moderatly high memory usage.

#### Preloaded Graph Training
This training mode is used by setting `graph_loading: preloaded` in the training config. This mode will load all required graph files into memory. This results in improved performance over the Prebuilt mode, at the cost of significantly larger memory usage.

### Start Training

You can start a training with the `train.py` script, for example

```bash
python train.py --config configs/classifier.yaml --gpus 0,1
```
In the first argument, specify the config file, training GPUs are specified in the second argument. In the example above, we train on the frist two GPUs accessible from the current machine. To run a quick test, use `--test_run`. Information about other arguments can be found by running `python train.py -h`. 

???+ warning "Check GPU usage before starting training."

    You should check with `nvidia-smi` that any GPUs you use are not in use by some other user before starting training.

Model checkpoints are saved under `saved_models/`. 


??? info "Restarting a previous training"

    It is possible to continue training from a model checkpoint. You should point the `--config` argument
    to the saved config file inside the previous training's output dir. You also need to specify which 
    checkpoint to restart the training from using the `--ckpt_path` argument.


### Training Tips 

During training, data is loaded using worker processes. 
The number of worker processes you use will be loosely related to number of CPUs available on your machine.
You can find out the number of CPUs available on your machine by running

```bash
cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1
```

Some other tips to make training as fast as possible are listed below.

**Worker counts**

- If you have exclusive access to your machine, and you run with Just-In-Time graphs, you should set `num_workers` equal to the number of CPUs on your machine.
- If using prebuilt graphs, it's better to choose `num_workers <= num_cpu/2`. This is due to each thread requiring another thread for memory pinning. 
- For prebuilt and preloaded graphs, you may have to experiment to find optimal number of workers. In some cases using too many results in a performance decrease. Optimal worker count appears to be in the region 6-9 workers per GPU.

**Prebuilding Graphs**

- If traning is too slow using Just-In-Time graphs, you should prebuild the graphs before starting training.

**Moving data closer to the GPU**

- Most HPC systems will have dedicated fast storage. Loading training data from these drives can significantly improve training times. To automatically copy files to a drive, you can use the `move_files_temp` config flag. The files will be removed after training, but if the training job crashes is cancelled this may not happen. You should make sure to clean up your files if you using this setting.
- If you have enough RAM, you can load the training data into shared memory before starting training by setting `move_files_temp` to some path in `/dev/shm/`. Again, make sure to clean up your files in `/dev/shm/` after training as the script may fail to do this for you.


### Slurm GPU Clusters

Those at UCL or other institutions with Slurm managed GPU batch queues can submit training jobs using

```bash
sbatch submit/submit_slurm.sh
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
