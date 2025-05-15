# Internal Tutorial

!!!warning "This tutorial is for CERN users. If you don't have a CERN account, you can following the [open tutorial](tutorial.md) instead."

## Introduction

In this tutorial, you will learn to setup and use the [Salt framework](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt/) in the context of $X \rightarrow bb$ tagging.
Salt is a high-level framework for training state-of-the-art flavour tagging algorithms.
In addition, plotting scripts are provided to plot the results of the evaluation using the [`puma`](https://github.com/umami-hep/puma) package.

In this tutorial, we cover the following functionalities of Salt:

1. Training of a subjet-based Xbb tagger
2. Training of a constituent-based Xbb tagger using tracks
3. Modification of high-level settings and network hyperparameters
4. Evaluation of results

!!!info "If you are not studying $X \rightarrow bb$ feel free to skip task 2"

The tutorial is meant to be followed in a self-guided manner. You will be prompted to do certain tasks by telling you what the desired outcome will be, without telling you how to do it. Using the [Salt documentation](index.md), you can find out how to achieve your goal. In case you are stuck, you can click on the "hint" toggle box to get a hint. If you tried for more than 10 min at a problem, feel free to toggle also the solution with a working example.

???+ question "What to do if you get stuck"
    
    In case you encounter some errors or you are completely stuck, you can reach out to the dedicated [FTAG tutorial mattermost channel](https://mattermost.web.cern.ch/aft-algs/channels/ftag-tutorials) (click [here](https://mattermost.web.cern.ch/signup_user_complete/?id=ektad7hj4fdf5nehdmh1js4zfy) to sign up).

This tutorial has been run a few times, click below for intro slides which give some context on the framework and this tutorial:

- [$X \rightarrow bb$ taskforce meeting](https://indico.cern.ch/event/1248303/).
- [FTAG Workshop 2023](https://indico.cern.ch/event/1311519/timetable/?view=standard#31-salt-tutorial).


## Prerequisites

For this tutorial, you need access to a shell on either CERN's `lxplus` or your local cluster with `/cvmfs` access to retrieve the `singularity` image needed. To set this up, please follow the instructions [here](setup.md) by selecting the "singularity" tab in the ["Create Environment"](setup.md#create-environment).

Efficient training is only possible on resources with GPU access.
It is highly encouraged to use an institute-managed GPU enabled machine if one is available.
Otherwise, CERN provides special lxplus nodes with GPU access for interactive computing.

You can log in to a CERN lxplus gpu node with:

```bash
ssh -Y <username>@lxplus-gpu.cern.ch
```

You can check that your node is configured with GPU access by running 

```bash
nvidia-smi
```

If you see a tabular output with information about one or more GPUs, then you are good to continue.

!!! warning "Check your machine is configured correctly"
    
    If you see `No devices were found` your node is badly configured, and you should log in again and hope for a new node.
    


### Training datasets

You should copy the training files before doing the tutorial.
If you don't the training will be much slower but you can compensate for that by reducing the number of training and validation jets as hinted in the tasks below.
The train/val/test samples for the tutorial each have 2M jets and are stored on EOS in the following directory

- `/eos/user/u/umami/tutorials/salt/2023/inputs/`

=== "Copy to user EOS space"
    If you are running on lxplus, copying the training files to your private storage on `/eos/user/${USER:0:1}/$USER/` is recommended to avoid overly high concurrent access:
    ```
    rsync -vaP /eos/user/u/umami/tutorials/salt/2023/inputs/ /eos/user/${USER:0:1}/$USER/training-samples
    ```

=== "Local Cluster"

    If you are running on your local cluster, you can copy the files to a directory with fast access:

    ```
    rsync -vaP <cern username>@lxplus.cern.ch:/eos/user/u/umami/tutorials/salt/2023/inputs/ /fast/disk/training-samples/
    ```

??? warning "Access to EOS is slow, copying files before the tutorial is highly recommended!"

    The training files are stored on EOS, which is a distributed file system. Accessing files on EOS is slow, so it is recommended to copy the files to a local directory before starting the tutorial. If you attempt to run the tutorial directly from EOS, you will experience very slow training times.


??? error "What to do if you don't have access to the EOS folder"

    The training files stored on EOS are only shared with user subscribed to the egroups/mailing lists
    
    - `atlas-cp-flavtag-btagging-algorithms`
    - `atlas-cp-flavtag-jetetmiss-BoostedXbbTagging`

    If you are not yet subscribed, please consider doing so to get access to the training files.
    You can subscribe using the [CERN egroups webpage](https://e-groups.cern.ch/e-groups/EgroupsSearch.do).

    If you already are subscribed and try to copy from inside a singularity container, it might fail. In that case, copy the files without using the singularity container.

When training a model, [it is possible to specify a local directory](training.md#fast-disk-access) with fast access, e.g. `/tmp` to which the files will be copied.
This will speed up the training on e.g. `lxplus` significantly (though you will still incur the initial cost of copying the files).

The total size of the training, validation and test files is 17GB, make sure you have sufficient free space.
Alongside the input h5 are the `norm_dict.yaml` and `class_dict.yaml` which are also used for training.


### Singularity image

The FTAG group provides salt-ready singularity images via `/cvmfs/unpacked.cern.ch` on lxplus (or any cluster which has `/cvmfs` mounted). On the node, you can use `singularity` to launch the container from the image on `/cvmfs/unpacked.cern.ch` with the already prepared `salt` framework.
We'll use the tagged image for version `0.3` of the code.

=== "lxplus (eos access)"

    If you run on lxplus, it is advantageous to also mount the `/afs`, `/eos`, `/tmp` and `/cvmfs` directories:

    ```bash
    singularity shell -e --env KRB5CCNAME=$KRB5CCNAME --nv --bind $PWD,/afs,/eos,/tmp,/cvmfs,/run/user \
        /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/salt:0-3
    ```

=== "other (cvmfs only)"

    ```
    singularity shell -e --env KRB5CCNAME=$KRB5CCNAME --nv --bind $PWD,/cvmfs,/run/user \
        /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/salt:0-3
    ```

If you have issues accessing bound paths, ensure your Kerberos credentials are set with `export KRB5CCNAME=FILE:/run/user/${UID}/krb5cc`

After running the [`singularity shell`](https://docs.sylabs.io/guides/latest/user-guide/cli/singularity_shell.html#singularity-shell) command, you can re-source your `.bashrc` to get some of the features of your normal terminal back by running 

```bash
source ~/.bashrc
```


## Tutorial tasks

### 1. Fork, clone and install Salt

Although the singularity images come with salt pre-installed, they do not allow for an editable version of the package.
It's therefore highly recommended to re-install the package from source to give you full control.
To do so, you need to do the following steps:

1. Create a personal fork of Salt in Gitlab.
2. Clone the forked repository to your machine using `git`.
3. Switch to the `0.3` tag which is used for the tutorial.
4. (Optional) Run the setup to switch to development mode.
5. Run the test suite

Go to the GitLab project page of Salt to begin with the task: <https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt>

??? info "Hint: Create a personal fork of Salt in Gitlab"

    In case you are stuck how to create your personal fork of the project, you can find some general information on git and the forking concept [here in the GitLab documentation](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html).

??? info "Hint: Clone the forked repository to your machine using `git`"

    The command `git clone` is the one you need. You can look up the use [here](setup.md). You can use the `--branch` argument to checkout a specific branch, e.g. `--branch 0.3` to checkout the `0.3` tag.


??? info "Hint: installing the package in development mode"

    By default, the singularity image comes with salt preinstalled, but this not an editable installation. If you want to make code changes, you can install salt in development mode using `pip` with the `-e` flag.


??? info "Hint: Run the test suite"

    You can run the suite of unit tests as outlined in the [salt documentation on ](contributing.md#test-suite). Make sure that you enter the `salt` source code directory before you execute the test suite!

    ```bash
    cd salt/
    pytest --cov=salt --show-capture=stdout
    ```

    Note that, depending on your machine, the test suite may take a while to run. To just run a single test, you can instead use
    
    ```bash
    pytest --cov=salt --show-capture=stdout tests/test_pipeline.py::TestModels::test_GN1
    ```

??? warning "Solution"

    Open the website <https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt> in a browser. You may need to authenticate with your CERN login credentials. In the top right corner of the Salt project you see three buttons which show a bell (notifications), a star (to favourite the project) next to a number, and a forking graph (to fork the project) with the text "Fork" next to a number. Click on the word "Fork" to open a new website, allowing you to specify the namespace of your fork. Click on "Select a namespace", choose your CERN username, and create the fork by clicking on "Fork project".

    Next, you need to clone the project using `git`. Open a fresh terminal on the cluster your are working on, create a new folder and proceed with the cloning. To do so, open your forked project in a browser. The address typically is `https://gitlab.cern.ch/<your CERN username>/salt`. When clicking on the blue "Clone" button at the right hand-side of the page, a drop-down mini-page appears with the ssh path to the forked git project. Let's check out your personal fork and add the original project as upstream:

    ```bash
    git clone ssh://git@gitlab.cern.ch:7999/<your CERN username>/salt.git
    cd salt
    git remote add upstream ssh://git@gitlab.cern.ch:7999/atlas-flavor-tagging-tools/algorithms/salt.git
    git checkout 0.3
    ```

    You now forked and cloned Salt and should be ready to go!

    Launch the salt singularity container (make sure to bind the directory containing the cloned git project) and change the directory to the top level directory of the project.


    ```bash
    singularity shell -e --nv --bind $PWD,/afs,/eos,/tmp,/cvmfs \
    /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/salt:0-3
    ```

    If you want to modify the salt code and contribute to development, you need to install the salt package to switch to development mode:

    ```bash
    python -m pip install -e .
    ```

    Finally, you can run a test to check if everything works fine:
    
    ```bash
    pytest --cov=salt --show-capture=stdout tests/test_pipeline.py::test_GN1
    ```

    Make sure you are in the directory containing `tests/` when running the test suite.
    The full test suite would likely take some time to run, but can be invoked with

    ```bash
    cd salt/
    pytest --cov=salt --show-capture=stdout
    ```


### 2. (Optional) Set up logging

[Comet.ml](https://www.comet.com/) is an online ML logging service. It's free for academic use. If you get stuck, consult the [comet documentation](https://www.comet.com/docs/v2/guides/getting-started/quickstart/) and the hints below.

1. Create an account and a project.
2. Generate an API key.
3. Save the API key and project name in the relevant environment variables, or add them to your `~/bashrc` file.

??? info "Hint: Creating an account and a project"

    To use it, [create an account](https://www.comet.com/signup), and then create a project using the blue `+ New project` button in the GUI interface.

??? info "Hint: Generating an API key"

    You then need to create and save an API key for your project, which you can use to log your training runs. You can find the API key in the [account settings page](https://www.comet.com/account-settings/apiKeys).

??? info "Hint: Saving info to environment variables"

    See the Salt [logging docs](setup.md#setup-logging) for info on which environment variables to use.

??? danger "Warning: If you don't set up logging, you may need to disable it in the training config file"

    Open the `base.yaml` config file and set `logger: False` under the `trainer:` block. Remove the existing sub-blocks under `logger:`. You also need to remove the 
    ```
    - class_path: pytorch_lightning.callbacks.LearningRateMonitor
    ``` 
    line under the `trainer: callbacks:` block, since this feature requires a logger to work.

### 3. Train subjet-based tagger

In this task, you will train an algorithm based on inputs from variable-radius track sub-jets of the large-radius jet. For the purposes of the tutorial, the training is configured to use the first 5M jets from the training dataset, and to run for 10 epochs.

You can take a look inside the `SubjetXbb.yaml` or `norm_dict.yaml` file to see which variables will be used for training the subjet based model. The r22 GN2 PFlow scores have been included, along with some other kinematic information about the subjet.

1. Modify the `SubjetXbb.yaml` model config file to use the correct paths to your locally downloaded training files.
2. Run the training for 10 epochs.

??? info "Hint: Modifying the `SubjetXbb.yaml` config file"

    You'll need to specify the `SubjetXbb.yaml` model config to use. This config file needs to be edited with the correct paths to your locally downloaded training files. You'll need to modify the `train_file`, `val_file` and `scale_dict` keys under the `data:` block.
    To change the number of epochs the training runs for, you can modify the `max_epochs` key under the `trainer:` block.

??? info "Hint: Warning about the number of workers used in the dataloaders"

    By default the `--data.num_workers` flag is set to 10. On your machine, you might see a warning if this is too many, and the suggested number to use instead. Including the `--data.num_workers` flag in the training command will override the default value.

??? info "Hint: Running the training"

    The command to run a training is described in the Salt documentation [here](training.md#training). Make sure you specify `--config configs/SubjetXbb.yaml` to use the correct config file.

??? info "Hint: Speeding up the training"

    Take a look at the [dataloading](training.md#dataloading) section of the training documentation. You can try increasing the worker count, moving the files to fast storage or RAM, or reducing the overall number of training jets using the `--data.num_jets_train` and `--data.num_jets_val` flags. Finally, you could decrease the number of training epochs with `--trainer.max_epochs` flag.

    Make sure that you really have copied the training files from the `/eos` location to a local path, otherwise reading the input via `/eos` can also slow down the training. You could also experiment with the [`--data.move_files_temp`](training.md#fast-disk-access) flag to transfer the data to a high-speed file reading resource before training. On lxplus, you could do so by adding `--data.move_files_temp /tmp` to the training command.


??? warning "Solution"

    After modifying the `SubjetXbb.yaml` config file as described in the first hint, you can run a test training with 
    ```bash
    salt fit --config configs/SubjetXbb.yaml --trainer.fast_dev_run 2
    ```
    Assuming this completes without any errors, you can run a full training by omitting the `--trainer.fast_dev_run` flag.
    By default, the training uses the first GPU on system. If you want to use a different GPU, you can specify it with the `--trainer.devices` flag as described in the [documentation](training.md#choosing-gpus). To run on the CPU, you can use `--trainer.accelerator cpu`.

    For more training options, including tips for how to speed up the training, take a look at the [documentation](training.md).


### 4. Train track-based tagger

In this task, you will train an algorithm based directly on the tracks associated with the large-radius jet as inputs. Again, take a look inside the variable config to get an idea of which variables are being used in the track-based case.

1. Modify the `GN2X.yaml` model config file to use the correct paths to your locally downloaded training files.
2. Run the training for 10 epochs.
3. Compare the losses of the subjet-based model and the track-based model.

Note that you may ecounter _carefully planned_ errors as part of this task, please use the hints below to try and resolve them.

??? info "Hint: See hints for the previous task"

    This task is very similar to Task 2, for a different model config: `GN2X.yaml`.

??? info "Hint: What to do about a `MisconfigurationException`" 
    
    You might see the following error if you run on a machine with only one accessible GPU.

    ```
    lightning.fabric.utilities.exceptions.MisconfigurationException: You requested gpu: [0, 1]
     But your machine only has: [0]
    ```
    
    This is because the default `GN2X.yaml` config asks for 2 GPUs to speed up training.
    This is a good opportunity to learn about requesting GPUs. 
    You can read [here](training.md#choosing-gpus) for hints about what to do.

??? info "Hint: What to do about `ValueError: Variables {...} were not found in dataset`" 
    
    This kind of error is quite common if the inputs become out of sync with the config.
    In our case, the config has been updated to use the new truth label names, but the samples are somewhat older and have not been updated.

    You need to modify the `GN2X.yaml` config to revert to the old label names which do not have the `ftag` prefex, e.g. `ftagTruthOriginLabel` -> `truthOriginLabel`.
    This needs to be done in the task config, i.e. around L150 and L191.


??? warning "Solution"

    The training should run in the same way as the previous task, but will take longer to complete, since we are processing up to 100 tracks per jet, rather than just the info about 3 subjets.

    You should take note of the initial and final values of the losses for the two models so that you can compare them. Which loss decreases faster from its initial value? Which is lower after the training has been completed? Why do you think this is? (Remember the default choice of 2M training jets is a small fraction of the total number of jets used to train the GN2 tagger which is used to produce the subjet scores.)

    In order to fix the `MisconfigurationException`, just add `--trainer.devices=1` as a command line flag to your `salt fit` call.

### 5. Modify network parameters and retrain

In this task, you will modify parameters of the models trained in the previous tasks and retrain the networks. You should consider what effect the changes have on the evolution of the loss, the size of the model, and the training speed. This task is open-ended and you are encouraged to experiment with modifying the different config files.

??? info "Hint: Changing the number of training jets"

    Inside the model config file you wish to change, look at the `num_jets_train` config key inside the `data:` block. You can take a look at how the number of jets affects the final performance of the models. You can also configure this from the CLI using `--data.num_jets_train <num>`.

??? info "Hint: Changing the model architecture"

    Inside the model config file you wish to change, look at the `model:` block. The core of the model is the Graph Network/Transformer configured in the `gnn:` block. You can modify the number of layers, the number of attention heads, or the embedding dimension and see what effect his has on the training.

??? info "Hint: Removing auxiliary tasks"

    The `GN2X.yaml` includes the auxiliary track classification task. To remove it, look in the `tasks:` block for the list item which has `name: track_classification`. Removing the associated block will disable that part of the model and remove the associated track classification loss from the overall training loss function when training.

??? warning "Solution"

    This task is really just about tinkering, but you may notice the following things:
    
    - Reducing the number of jets is detrimental to the performance of the model. If you study how the lowest value of the loss changes as a function of the number of training jets, you should come to the conclusion that larger training samples would be beneficial to improving performance.
    - Increasing the model size (number of layers, embedding dimension) leads to slower training times, and more overtraining (especially visible with such a small number of training jets)
    - The loss for the subjet-based model initially drops much more quickly than for the track-based model. This is because it has outputs from an already trained model as inputs.
    - Given enough time and enough input jets, the loss for the track-based model will drop below that of the subjet-based model. This reflects the fact constituent-based tagging approach is more powerful than the subjet-based approach in the long run.


### 6. Evaluate the models on the test set

After training, the model is evaluated on an independent set of testing jets. The results are used to produce performance plots. The test file you will use for the tutorial is called `pp_output_test.h5`, and contains a mixture of the different jet classes used for training. The jet labels are specified by the `R10TruthLabel_R22v1` jet variable. The classes are specified in the `enum` [here](https://gitlab.cern.ch/atlas/athena/-/blob/master/PhysicsAnalysis/AnalysisCommon/ParticleJetTools/ParticleJetTools/LargeRJetLabelEnum.h#L11).

1. Choose which models you want to evaluate and 
2. Run the model evaluation command

??? info "Hint: Comparing to a provided pre-trained model"

    If you had problems with the training, or you just want to compare to a benchmark, you can use one of the provided model checkpoints to evaluate.
    These models are found in the `/eos/user/u/umami/tutorials/salt/2023/trained-models/` directory.
    Note that they are not claimed to be especially well performing models (they were only trained for 5 epochs) - you may well find a configuration that outperforms them!

??? info "Hint: Running the evaluation command"

    Take a look at the [relevant page](evaluation.md) in the salt documentation.
    You might want to choose to evaluate on e.g. 1e5 jets, rather than the full 2M.

??? warning "Solution"

    Find the path to the saved config file of the model you want to evaluate.
    This will be located in a timestamped directory under `logs/`.
    Also have handy the path to your `/eos/home-u/umami/tutorials/salt/2023/inputs/pp_output_test.h5` (or run directly on this file).
    To run the evaluation, use

    ```bash
    salt test --config logs/<timestamp>/config.yaml --data.test_file path/to/pp_output_test.h5
    ```
    
    If you want to evaluate the pre-trained model on EOS, this command will be for example
    ```bash
    salt test \ 
      --config /eos/home-u/umami/tutorials/salt/2023/trained-models/SubjetXbb_20230920-T192350/config.yaml \
      --data.test_file /eos/home-u/umami/tutorials/salt/2023/inputs/pp_output_test.h5
    ```

    Salt automatically evaluates the checkpoint with the lowest associated validation loss for evaluation, but you can use `--ckpt_path` to specify this manually.
    The resulting evaluation file will be saved in `ckpts/` in the training output directory, alongside the checkpoint that was used to run the evaluation.
    Read the [salt docs](evaluation.md#running-the-test-loop) for more info.


### 7. Create plots which quantity the trained algorithms performance

In this task, you will create plots of performance metrics using the [`puma`](https://github.com/umami-hep/puma/) python package.
You can find more information on how to use `puma` for plotting in the [corresponding plotting tutorial](https://ftag-docs.docs.cern.ch/software/tutorials/tutorial-plotting/).

1. Produce a histogram of the jet scores for each class.
2. Produce ROC curves as a function of signal efficiency.

??? info "Hint: Installing puma"

    Your Salt installation will install puma as a dependency. 
    You can also follow the quickstart guide in the [puma docs](https://umami-hep.github.io/puma/main/index.html#) to learn how to install it yourself.

??? info "Hint: Plotting histograms"

    Take a look at the [relevant page](https://umami-hep.github.io/puma/main/examples/histograms.html) in the puma docs.

??? info "Hint: Plotting ROCs"

    Take a look at the [relevant page](https://umami-hep.github.io/puma/main/examples/rocs.html) in the puma docs.

??? info "Hint: What to use as a discriminant?"

    Since we have four classes, calculating a discriminant is more complicated than in the single b-tagging case. 
    One option is to use the score for the signal class directly as the discriminant, but please note, this may lead to a suboptimal trade off between the different background rejections.

??? info "Hint: What are the truth labels"

    Take a look at their definition [here](https://gitlab.cern.ch/atlas/athena/-/blob/master/PhysicsAnalysis/AnalysisCommon/ParticleJetTools/ParticleJetTools/LargeRJetLabelEnum.h#L34). Hbb is `11`, Hcc is `12`, and top is `1` and QCD is `10`.

??? warning "Solution"

    The prepared script `make_plots.py` provides an implementation example to plot tagger discriminant distributions and ROC curves. 
    It is located here: `/eos/home-u/umami/tutorials/salt/2023/make_plots.py`.
    
    At the beginning of the script, you are invited to fill the `networks` dictionary with one or more trained model paths and dedicated keys. In the current version, the pretrained model paths have been implemented. If you want, you can modify them with your own trained models. 
    In addition, a `reference` is also requested for the model comparisons. It should correspond to one of the model keys added to `networks`.
    Finally, `test_path` has to be completed with the path to your `pp_output_test.h5` sample. 
    
    Two different tagger discriminants are defined by `Hbb` and `Hcc` signal class probabilities.
    
    In a first step, the script extracts the different jet tagging probabilities as well as the needed kinematic information to define the jet selection to be applied. Implemented as a boolean `mask`, the jet selection can be easily modified. Efficiencies and rejections are also computed.
    
    For a given tagger discriminant, the distributions corresponding to the different jet flavours and trained models are then plotted on the same figure in order to perform a complete comparison. The Puma's `HistogramPlot` and `Histogram` objects offer a lot of configuration variables which can be modified according to cosmetic tastes and needs. Finally, the plotting of the corresponding ROCs follows similarly in another set of figures. 

    Below, the content of `make_plots.py` is shown:


    ```python

    import h5py
    import numpy as np
    from puma import Histogram, HistogramPlot, Roc, RocPlot
    from puma.metrics import calc_rej
    from puma.utils import get_good_colours, get_good_linestyles, logger

    networks = {
        "SubjetXbb" : "/eos/home-u/umami/tutorials/salt/2023/trained-models/SubjetXbb_20230920-T192350/ckpts/epoch=004-val_loss=0.59234__test_pp_output_test.h5",
        "GN2X" : "/eos/home-u/umami/tutorials/salt/2023/trained-models/GN2X_20230920-T193158/ckpts/epoch=004-val_loss=1.15303__test_pp_output_test.h5"
    }

    reference = "SubjetXbb"
    test_path = '/eos/home-u/umami/tutorials/salt/2023/inputs/pp_output_test.h5'
    num_jets = 100_000

    # load test data
    logger.info("Load data")
    with h5py.File(test_path, 'r') as test_f:
        jets = test_f['jets'][:num_jets]
        jet_pt = jets['pt'] / 1000
        jet_mass = jets['mass'] / 1000
        jet_eta = np.abs(jets['eta'])
        flav = jets['R10TruthLabel_R22v1']
        mask = (jet_pt < 1000) & (jet_pt > 250) & (jet_mass > 50) & (jet_mass < 300)
        is_QCD = flav == 10
        is_Hcc = flav == 12
        is_Hbb = flav == 11
        is_Top = flav == 1
        n_jets_QCD = np.sum(is_QCD & mask)
        n_jets_Top = np.sum(is_Top & mask)

    results = {}
    logger.info("Calculate rejections")
    for key, val in networks.items():
        with h5py.File(val, 'r') as f:
            jets = f['jets'][:num_jets]
            pHbb = jets[f'{key}_phbb']
            pHcc = jets[f'{key}_phcc']
            pQCD = jets[f'{key}_pqcd']
            pTop = jets[f'{key}_ptop']
            disc_Hbb = pHbb
            disc_Hcc = pHcc

            sig_eff = np.linspace(0.4, 1, 100)
            Hbb_rej_QCD = calc_rej(disc_Hbb[is_Hbb & mask], disc_Hbb[is_QCD & mask], sig_eff)
            Hbb_rej_Top = calc_rej(disc_Hbb[is_Hbb & mask], disc_Hbb[is_Top & mask], sig_eff)
            Hcc_rej_QCD = calc_rej(disc_Hcc[is_Hcc & mask], disc_Hcc[is_QCD & mask], sig_eff)
            Hcc_rej_Top = calc_rej(disc_Hcc[is_Hcc & mask], disc_Hcc[is_Top & mask], sig_eff)
            results[key] = {
                'sig_eff' : sig_eff,
                'disc_Hbb' : disc_Hbb,
                'disc_Hcc' : disc_Hcc,
                'Hbb_rej_QCD' : Hbb_rej_QCD,
                'Hbb_rej_Top' : Hbb_rej_Top,
                'Hcc_rej_QCD' : Hcc_rej_QCD,
                'Hcc_rej_Top' : Hcc_rej_Top
            }

    logger.info("Plotting Discriminants.")
    plot_histo = {
        key : HistogramPlot(
            n_ratio_panels=1,
            ylabel="Normalised number of jets",
            xlabel=f"{key}-jet discriminant",
            logy=True,
            leg_ncol=1,
            figsize=(6.5, 4.5),
            bins=np.linspace(0, 1, 50),
            y_scale=1.5,
            atlas_second_tag="$\\sqrt{s}=13$ TeV, Xbb jets",
        ) for key in ['Hbb', 'Hcc']}
    linestyles = get_good_linestyles()[:len(networks.keys())]
    colours = get_good_colours()[:3]
    for key, value in plot_histo.items():
        for network, linestyle in zip(networks.keys(), linestyles):
            value.add(
                Histogram(
                    results[network][f'disc_{key}'][is_QCD],
                    label="QCD jets" if network == reference else None,
                    ratio_group="QCD",
                    colour=colours[0],
                    linestyle=linestyle,
                ),
                reference=(network == reference),
                )
            value.add(
                Histogram(
                    results[network][f'disc_{key}'][is_Top],
                    label="Top jets" if network == reference else None,
                    ratio_group="Top",
                    colour=colours[1],
                    linestyle=linestyle,
                ),
                reference=(network == reference),
                )
            value.add(
                Histogram(
                    results[network][f'disc_{key}'][is_Hbb if key == 'Hbb' else is_Hcc],
                    label=f"{key} jets" if network == reference else None,
                    ratio_group=f"{key}",
                    colour=colours[2],
                    linestyle=linestyle,
                ),
                reference=(network == reference),
                )
        value.draw()
        # The lines below create a legend for the linestyles
        value.make_linestyle_legend(
            linestyles=linestyles, labels=networks.keys(), bbox_to_anchor=(0.5, 1)
        )
        value.savefig(f"disc_{key}.png", transparent=False)

    # here the plotting of the roc starts
    logger.info("Plotting ROC curves.")
    plot_roc = {
        key : RocPlot(
            n_ratio_panels=2,
            ylabel="Background rejection",
            xlabel=f"{key}-jet efficiency",
            atlas_second_tag="$\\sqrt{s}=13$ TeV, Xbb jets",
            figsize=(6.5, 6),
            y_scale=1.4,
        ) for key in ['Hbb', 'Hcc']}

    for key, value in plot_roc.items():
        for network in networks.keys():
            value.add_roc(
                Roc(
                    sig_eff,
                    results[network][f'{key}_rej_QCD'],
                    n_test=n_jets_QCD,
                    rej_class="qcd",
                    signal_class=f"{key}",
                    label=f"{network}",
                ),
                reference=(reference == network),
            )
            value.add_roc(
                Roc(
                    sig_eff,
                    results[network][f'{key}_rej_Top'],
                    n_test=n_jets_Top,
                    rej_class="top",
                    signal_class=f"{key}",
                    label=f"{network}",
                ),
                reference=(reference == network),
            )
        # setting which flavour rejection ratio is drawn in which ratio panel
        value.set_ratio_class(1, "qcd")
        value.set_ratio_class(2, "top")
        value.draw()
        value.savefig(f"roc_{key}.png", transparent=False)
    ```
