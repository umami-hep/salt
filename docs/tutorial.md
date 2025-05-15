# Open Data Tutorial

In this tutorial, we will walk through the steps to use the salt software package. 
This tutorial has been given a few times, in
[2023](https://indico.cern.ch/event/1311519/contributions/5577998/attachments/2721103/4727424/salt_tutorial.pdf), 
[2024](https://indico.cern.ch/event/1352459/contributions/5860483/attachments/2854884/4992618/Salt_tutorial_GN2_ftag.pdf), and
[2025](https://indico.cern.ch/event/1488957/contributions/6403390/).

We will be using the `tutorial.yaml` configuration file for steering the training and an open dataset as input for the training.
In a production context you will create the pre-processed data for training and evaluation yourself using the [umami-preprocessing](https://github.com/umami-hep/umami-preprocessing) (UPP) software. For the context of this tutorial, we will use already preprocessed data which are separated into training, validation and test datasets.

After you have completed the tutorial, you will be familiar with the following concepts:

- Working with an open dataset which already underwent preprocessing for training
- Steering the training with a configuration file
- Running a nominal training 
- Evaluating training performance
- Modify the configuration file to change the model definition
- Compare the performance in simulated events for different model architectures

We strongly recommend to follow the tutorial on a computer with a GPU to benefit from faster turnaround time for the training.

## Step 1: Installation

Before we begin, make sure you have the software package installed on your system. If not, please make sure you have read the documentation on [setup](setup.md) and followed it to install `salt`.


## Step 2: Obtaining the training dataset

The basis of this tutorial is the "Top quark pair events for heavy flavour tagging and vertexing at the LHC" dataset. It contains jets from top anti-top decays. The event and parton shower are simulated in Pythia 8 with a centre of mass energy of 13 TeV, with detector response modelled in the Delphes framework. The detector response is modelled on the ATLAS detector, and a mean pileup of 50 was used. The dataset with additional information is available here: https://zenodo.org/records/10371998

Create a directory at a location with sufficient free disk space. The unpacked dataset corresponds to 28 GB disk space. The compressed file which is downloaded corresponds to a size of 14 GB.

Execute the following commands to download all files to a directory which you will need to define (replace `<path to directory>` with the path to the directory of your choice).

```bash
export TUTORIAL_DATA=<path to directory>
mkdir -p $TUTORIAL_DATA
cd $TUTORIAL_DATA
curl -o $TUTORIAL_DATA/tutorialdata.zip "https://zenodo.org/api/records/10371998/files-archive"
unzip $TUTORIAL_DATA/tutorialdata.zip -d $TUTORIAL_DATA
rm $TUTORIAL_DATA/tutorialdata.zip
```

After you completed the download, you should have the following files downloaded to your directory:

- `pp_output_train.h5`: 13.5 million training jets, consisting of 4.5 million b-jets, c-jets, and light-flavoured jets. Resampling is applied over the jet pT and eta, to ensure equivalent kinematic distributions.
- `pp_output_val.h5`: 1.35 million jets for validation, consisting of 450,000 of each jet flavour. Kinematics are resampled in the same way as the training file.
- `pp_output_test_ttbar.h5`: 1.35 million jets for evaluation, consisting of 450,000 of each jet flavour, with no kinematic resampling applied.
- `class_dict.yaml`: Details the relative weights for classification labels, based on the frequency of occurrence for a given entry. 
- `norm_dict.yaml`: Contains the means and standard deviations of variables that can be used for training, allowing for scaling.

The `h5` files contain (among others) the following datasets which are relevant for this tutorial:

- `jets`: jets, including variables such as jet kinematics, flavours, and summary statistics on the number of hadrons and constituents in the jet.
- `consts`: up to 50 charged constituents (tracks) per jet. Includes details on constituent kinematics and identification. A variable 'valid' is True for tracks in the jet, and False for all other tracks. The additional variable 'truth_origin_label' details the origin of the track and the variable `truth_vertex_idx` indicates groups of tracks which originate from a common vertex (indiced with an integer).


## Step 3: Running the Software

Trainings are run using configuration files. A configuration file has been prepared for this tutorial which defines the model, the training and validation datasets, as well as the corresponding `class_dict.yaml` and `norm_dict.yaml` files. Further, the hyperparameters of the model and configuration for steering the training, such as batch size and number of workers and CPU/GPU usage are defined.

You need to modify the file and replace these sections

```
  train_file: <path to directory>/pp_output_train.h5
  val_file: <path to directory>/pp_output_val.h5
  norm_dict: <path to directory>/norm_dict.yaml
  class_dict: <path to directory>/class_dict.yaml
```

with the path you downloaded the files in the previous step.

After you modified the file, you need to commit the changes. We suggest to open a new branch for this.

```bash
git checkout -b tutorial
git add .
git commit -m "update tutorial.yaml"
```

Read the documentation for [training](training.md) with salt.

Salt provides the option to use [Comet.ml](https://www.comet.com/) as an online ML logging tool. To set it up you should take a look at [(Optional) Set up logging](tutorial-Xbb.md#2-optional-set-up-logging).

First, you should run a test training which will run over a small number of training and validation batches and then exit without writing any output. This will show you if you have everything set up correctly.
Assuming you start in the `salt` main directory, you first need to navigate to the `salt` subdirectory which hosts the configuration file directory `configs`.

```bash
cd salt
salt fit --config configs/tutorial.yaml --trainer.fast_dev_run 2
```

If this did run successfully, you can run the full training.

```bash
salt fit --config configs/tutorial.yaml
```

!!! tip "What should you do if you get the error `GitError`?"

    If your training fails with the raised error
    ```
    GitError(
        ftag.git_check.GitError: Uncommitted changes detected. Please commit them before running, or use --force.
    )
    ```

    You should commit your changes with `git` before running the training.

The process of training might take some time. Typically, after 10 minutes the 20 epochs should have finished and you will see a directory created in `logs/GN2_<timestamp>`. It contains the checkpoints for each epoch in a subdirectory `logs/GN2_<timestamp>/ckpts` with the filenames encoding the epoch and corresponding loss evaluated on the validation sample. Ideally, you should see the validation loss decreasing with increasing epoch.

## Step 4: Evaluating the result

For evaluating the performance of the trained algorithm, you need to process the test dataset with a trained model configuration that is stored as one of the checkpoints.

Read the documentation for [evaluation](evaluation.md) with salt.

You can run the test loop on the `pp_` dataset by executing the following command. Remember to replace the <timestamp> with the appropriate string corresponding to the directory name and `<path to directory>` with the path to the directory where you downloaded the dataset.

```bash
salt test --config logs/GN2_<timestamp>/config.yaml --data.test_file <path to directory>/pp_output_test_ttbar.h5
```

As a result, you will find in the directory `logs/GN2_<timestamp>/ckpts/` a `.h5` file which is the test dataset file with the output scores of the trained model appended to it.

You can use the [puma](https://github.com/umami-hep/puma) plotting software and helper tools in [atlas-ftag-tools](https://github.com/umami-hep/atlas-ftag-tools/) to rapidly create plots to quantify the performance of the trained network.

An example script is provided here. You should have installed the dependencies `numpy`, `pandas`, `atlas-ftag-tools`, and `puma-hep` as part of the `salt` installation.

??? info "Example plotting script to obtain ROC curve `plot_roc.py` "

    ```python
    import numpy as np
    import pandas as pd
    from ftag.hdf5 import H5Reader
    from puma import Roc, RocPlot
    from puma.metrics import calc_rej

    fname = "logs/GN2_<timestamp>/ckpts/epoch<best epoch>.h5"
    reader = H5Reader(fname, batch_size=1_000)
    df = pd.DataFrame(reader.load({"jets": ["pt", "eta", "flavour", "GN2_pu", "GN2_pc", "GN2_pb"]}, num_jets=10_000)['jets'])

    def disc_fct(arr: np.ndarray, f_c: float = 0.018) -> np.ndarray:
        return np.log(arr[2] / (f_c * arr[1] + (1 - f_c) * arr[0]))

    discs_gn2 = np.apply_along_axis(
        disc_fct, 1, df[["GN2_pu", "GN2_pc", "GN2_pb"]].values
    )

    sig_eff = np.linspace(0.49, 1, 20)
    is_light = df["flavour"] == 0
    is_c = df["flavour"] == 4
    is_b = df["flavour"] == 5

    n_jets_light = sum(is_light)
    n_jets_c = sum(is_c)

    gn2_ujets_rej = calc_rej(discs_gn2[is_b], discs_gn2[is_light], sig_eff)
    gn2_cjets_rej = calc_rej(discs_gn2[is_b], discs_gn2[is_c], sig_eff)

    plot_roc = RocPlot(
        n_ratio_panels=2,
        ylabel="Background rejection",
        xlabel="$b$-jet efficiency",
        atlas_second_tag="$\\sqrt{s}=13$ TeV, ttbar jets \ntutorial sample, $f_{c}=0.018$",
        figsize=(6.5, 6),
        y_scale=1.4,
    )
    plot_roc.add_roc(
        Roc(
            sig_eff,
            gn2_ujets_rej,
            n_test=n_jets_light,
            rej_class="ujets",
            signal_class="bjets",
            label="GN2",
        ),
        reference=True,
    )
    plot_roc.add_roc(
        Roc(
            sig_eff,
            gn2_cjets_rej,
            n_test=n_jets_c,
            rej_class="cjets",
            signal_class="bjets",
            label="GN2",
        ),
        reference=True,
    )

    plot_roc.set_ratio_class(1, "ujets")
    plot_roc.set_ratio_class(2, "cjets")
    plot_roc.draw()
    plot_roc.savefig("roc.png", transparent=False)
    ```


## Step 5: Modification of the model via the configuration file

In this step you will modify the model by editing the configuration file. Note that in salt everything required for the training and the model definition is defined in the configuration file.

Look for the following lines and comment them out or remove them:

```yaml
- class_path: salt.models.ClassificationTask
              init_args:
                name: const_origin
                input_name: consts
                label: truth_origin_label
                class_names: [pileup, primary, fromBC, fromB, fromC, fromS, fromTau*, secondary]
                weight: 0.5
                loss:
                  class_path: torch.nn.CrossEntropyLoss
                  init_args:
                    weight: [1.0, 25.47, 8.27, 13.17, 8.84, 4.9, 11985.94, 1.25]
                dense_config:
                  <<: *task_dense_config
                  output_size: 8
                  context_size: *out_dim

            - class_path: salt.models.VertexingTask
              init_args:
                name: const_vertexing
                input_name: consts
                label: truth_vertex_idx
                weight: 1.5
                loss:
                  class_path: torch.nn.BCEWithLogitsLoss
                  init_args: { reduction: none }
                dense_config:
                  <<: *task_dense_config
                  input_size: 256
                  output_size: 1
                  context_size: *out_dim
```

This will remove the auxiliary tasks operating on the tracks associated to the jets. As a consequence, you should observe inferior performance.

Commit your changes with git 

```bash
git add .
git commit -m "deactivate auxiliary tasks"
```

and train the model again.

```bash
salt fit --config configs/tutorial.yaml
```

After the training completed, run again the evaluation. Remember to modify the timestamp so that the modified model is picked up!

```bash
salt test --config logs/GN2_<new timestamp>/config.yaml --data.test_file <path to directory>/pp_output_test_ttbar.h5
```

Again, you will find the output `h5` file in the corresponding `ckpts` directory.
Now you can compare the performance between the two models with the following script. The type of comparison is called "Ablation study" because it removes some parts of the model to study their significance for the model.

??? info "Example plotting script to compare ROC curves `plot_roc_ablation.py` "

    ```python
    import numpy as np
    import pandas as pd
    from ftag.hdf5 import H5Reader
    from puma import Roc, RocPlot
    from puma.metrics import calc_rej

    fname_default = "logs/GN2_<timestamp>/ckpts/epoch<best epoch>.h5"
    reader_default = H5Reader(fname_default, batch_size=1_000)
    df_default = pd.DataFrame(reader_default.load({"jets": ["pt", "eta", "flavour", "GN2_pu", "GN2_pc", "GN2_pb"]}, num_jets=10_000)['jets'])

    fname_ablation = "logs/GN2_<new timestamp>/ckpts/epoch<best epoch>.h5"
    reader_ablation = H5Reader(fname_ablation, batch_size=1_000)
    df_ablation = pd.DataFrame(reader_ablation.load({"jets": ["pt", "eta", "flavour", "GN2_pu", "GN2_pc", "GN2_pb"]}, num_jets=10_000)['jets'])

    def disc_fct(arr: np.ndarray, f_c: float = 0.018) -> np.ndarray:
        return np.log(arr[2] / (f_c * arr[1] + (1 - f_c) * arr[0]))

    discs_gn2_default = np.apply_along_axis(
        disc_fct, 1, df_default[["GN2_pu", "GN2_pc", "GN2_pb"]].values
    )

    discs_gn2_ablation = np.apply_along_axis(
        disc_fct, 1, df_ablation[["GN2_pu", "GN2_pc", "GN2_pb"]].values
    )

    sig_eff = np.linspace(0.49, 1, 20)
    is_light = df_default["flavour"] == 0
    is_c = df_default["flavour"] == 4
    is_b = df_default["flavour"] == 5

    n_jets_light = sum(is_light)
    n_jets_c = sum(is_c)

    gn2_default_ujets_rej = calc_rej(discs_gn2_default[is_b], discs_gn2_default[is_light], sig_eff)
    gn2_default_cjets_rej = calc_rej(discs_gn2_default[is_b], discs_gn2_default[is_c], sig_eff)

    gn2_ablation_ujets_rej = calc_rej(discs_gn2_ablation[is_b], discs_gn2_ablation[is_light], sig_eff)
    gn2_ablation_cjets_rej = calc_rej(discs_gn2_ablation[is_b], discs_gn2_ablation[is_c], sig_eff)


    plot_roc = RocPlot(
        n_ratio_panels=2,
        ylabel="Background rejection",
        xlabel="$b$-jet efficiency",
        atlas_second_tag="$\\sqrt{s}=13$ TeV, ttbar jets \ntutorial sample, $f_{c}=0.018$",
        figsize=(6.5, 6),
        y_scale=1.4,
    )
    plot_roc.add_roc(
        Roc(
            sig_eff,
            gn2_default_ujets_rej,
            n_test=n_jets_light,
            rej_class="ujets",
            signal_class="bjets",
            label="GN2",
        ),
        reference=True,
    )
    plot_roc.add_roc(
        Roc(
            sig_eff,
            gn2_default_cjets_rej,
            n_test=n_jets_c,
            rej_class="cjets",
            signal_class="bjets",
            label="GN2",
        ),
        reference=True,
    )

    plot_roc.add_roc(
        Roc(
            sig_eff,
            gn2_ablation_ujets_rej,
            n_test=n_jets_light,
            rej_class="ujets",
            signal_class="bjets",
            label="GN2 (no aux)",
        ),
    )
    plot_roc.add_roc(
        Roc(
            sig_eff,
            gn2_ablation_cjets_rej,
            n_test=n_jets_c,
            rej_class="cjets",
            signal_class="bjets",
            label="GN2 (no aux)",
        ),
    )

    plot_roc.set_ratio_class(1, "ujets")
    plot_roc.set_ratio_class(2, "cjets")
    plot_roc.draw()
    plot_roc.savefig("roc_ablation.png", transparent=False)
    ```

## Step 6: Exporting model to ONNX format

Once you have selected your ideal trained model, the model has to be converted into ONNX format before implementing it into Athena for inference.

ONNX (Open Neural Network Exchange) is a common open-source format for AI models, which enables them to be used across various frameworks and hardware accelerators.
Read the documentation for [onnx export](export.md) with salt.

To do this, run
```
to_onnx --ckpt_path logs/GN2_<timestamp>/<best_checkpoint>.ckpt --name MyModel
```

Once completed, your ONNX model will be saved in `logs/GN2_<timestamp>/network.onnx`. 
To obtain the metadata of your onnx model, do
```
get_onnx_metadata logs/GN2_<timestamp>/network.onnx
```
The script already includes a quick validation test between the scores produced by the PyTorch model and the ONNX model. 
We also recommend you to verify your ONNX model in Athena. More information about this can be found in [Athena Validation](export.md#athena-validation)
