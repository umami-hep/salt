# Advanced Configuration


### Dataloading

#### Using Multiple Small H5 Files (Wildcards)

Training files for can become quite large these days. To still have the possibility to store them properly, the big training
files are broken up into smaller training files. To use them similar to a big file, you can use so-called wildcards. Let's
assume you have your smaller files, which are named `pp_output_train_split_000.h5`, `pp_output_train_split_001.h5` and so on.
When all of them are stored in the same folder, the path used to define the training h5 file in the config can be given as:

```yaml
data:
  train_file: /path/to/somewhere/pp_output_train_split_*.h5
```

This will automatically trigger the Virtual Dataset (VDS) creation of Salt, using the VDS
capabilities of the [`atlas-ftag-tools`](https://github.com/umami-hep/atlas-ftag-tools). The VDS is
something like a symlink to the actual files and allows Salt to correctly read in the h5 files. The
VDS will be created by default in the same folder where also the wildcard points to. In the example
given above, the VDS file path would be `/path/to/somewhere/pp_output_train_split_vds/vds.h5`. If
you want a specific path, you can define this like this:

```yaml
data:
  train_file: /path/to/somewhere/pp_output_train_split_*.h5
  train_vds_file: /path/to/something/completely/else/my_train_vds_file.h5
```

All the aformentioned settings are also usable for the validation and test file(s):

```yaml
data:
  train_file: /path/to/somewhere/pp_output_train_split_*.h5
  train_vds_file: /path/to/something/completely/else/my_train_vds_file.h5
  val_file: /path/to/somewhere/pp_output_val_split_*.h5
  val_vds_file: /path/to/something/completely/else/my_val_vds_file.h5
  test_file: /path/to/somewhere/pp_output_test_split_*.h5
  test_vds_file: /path/to/something/completely/else/my_test_vds_file.h5
```

#### Selecting Training Variables

Training files are structured arrays, so it is easy to specify which variables you want to include in the training by name.

In your `data` config there is a `variables` key which specifies which variables to include in the training for each input type.
These are defined in the model files, rather than the `base.yaml` config.

??? warning "Make sure you do not train on truth information!"

    The variables listed under `data.variables` are the inputs to the training.
    You should _not_ include any truth information (unless you are testing this explicitly),
    but rather specify truth labels for each task in your model config.

For example, in [`GN2.yaml`]({{repo_url}}-/blob/main/salt/configs/GN2/GN2.yaml) you will find the following variables:

```yaml
data:
  variables:
    jets:
      - pt_btagJes
      - eta_btagJes
    tracks:
      - d0
      - z0SinTheta
      - dphi
      - deta
      ...
```

The number of variables specified here will be used to automatically set the `input_size` of your [`salt.models.InitNet`][salt.models.InitNet] modules.

Training with multiple types of inputs beyond jets and tracks is supported to create a heterogeneous model. An example of this can be found in [`GN2emu.yaml`]({{repo_url}}-/blob/main/salt/configs/GN2/GN2emu.yaml) which includes a separate electrons input type.

#### Mapping Input Dataset Names

By default, the input names are used to directly retrieve dataset in the input h5 files.
If you want to use a different name to retrieve the h5 datasets, you can specify a mapping in the `data` config, using the `input_map` key.

```yaml
data:
  input_map:
    internal_name: h5_dataset_name
```

In this example, `h5_dataset_name` will be used to retrieve the datasets from the input h5 files, while internally (and elsewhere in the configuration) this input type will be referred to as `internal_name`.


#### Truncating Inputs

You can truncate the number of tracks used for training.
This can be useful to speed up training.
To do so, pass `--data.num_inputs.tracks=10` to limit the training to the first ten tracks.
This can also be configured in yaml via

```yaml
data:
  num_inputs:
    tracks: 10
```

#### Remapping Labels

This section is about remapping labels on the fly, which is useful in case they are not already mapped to `0, 1, 2...`.
In the config block for each task, you can specify a `label_map` which maps the label values as stored in the input file to the ones you want to use in the task loss calculation.
For example, instead of using the pre-mapped `flavour_label`, you could directly train on `HadronConeExclTruthLabelID` using,

```yaml
class_path: salt.models.ClassificationTask
init_args:
    input_name: jets
    name: jet_classification
    label: HadronConeExclTruthLabelID
    label_map: { 0: 0, 4: 1, 5: 2 }
    class_names: [ujets, cjets, bjets]
    ...
```

Note, when using `label_map` you also need to provide `class_names`.
When using `flavour_label` as the target, the class names are automatically determined for you from the training file.

#### Relabelling on the fly

Similarly, it is possible to modify the labels available from the dataloading to use different ones during training. If a set of flavour labels has been used during preprocessing, these can be changed by relabelling on-the-fly to a new set of classes. These will typically be a more granular breakdown of the inital classes.
For example, if in preprocessing the classes `hbb`, `hcc`, `top`, `qcd`, have been used, one may want to breakdown the qcd class into its subclasses `qcdbb`, `qcdbx`, `qcdcx`, `qcdll`.
Another use case occurs when evaluating a test file which does not contain the `flavour_label` field; this typically happens when the test file has not been processed via UPP. In this case, the classification task (which would otherwise fail with an error on the missing flavour label), runs and the `flavour_label` is generated on-the-fly.
To apply the relabelling on-the-fly, the following config can be used:

```yaml
data:

  labeller_config:
    use_labeller: True
    class_names: ['hbb', 'hcc', 'top', 'qcdbb', 'qcdbx', 'qcdcx', 'qcdll']
    require_labels: False
    ...
```
The option `class_names` will contain the list of new target classes to apply the relabelling scheme; these will be different from the ones available in the preprocessing output.
The option `require_labels` controls whether all the jets will be relabelled to the new target classes. If set to `True` (default), the job will fail if some jets do not verify the selection criteria of any of the new target classes; if set to `False`, a warning is thrown and some jets will not be relabelled.
Note, when using `use_labeller: True` you also need to provide `class_names`. The model output size for the ClassificationTask will also need to be modified accordingly to match the new number of output classes.
The labeller is disabled either by removing the `labeller_config` block entirely, or by setting `use_labeller: False`.
At present, this feature is only available for jets and for the `flavour_label` label in the ClassificationTask.

#### Input Augmentation

Different transformations ("transforms") can be applied to input data after being loaded but before training. You can choose from the classes defined in `salt.data.transforms` and they are applied in the same order that they are defined in the configuration file. Transforms are specified under the `data` configuration block.

The `GaussianNoise` class is available for applying noise to the input features of your choice. The input type (usually `jets` or `tracks`), variable name, and mean and standard deviation of the desired noise are specified. The mean and standard deviation are fractions of the input values. An example config for this is shown below.

```yaml
transforms:
  - class_path: salt.data.transforms.GaussianNoise
    init_args:
      noise_params:
        - input_type: jets
          variable: pt_btagJes
          mean: 0.0
          std: 0.1
        - input_type: tracks
          variable: d0
          mean: 0.1
          std: 0.05
```

This will add noise with mean 0 and standard deviation 0.1 to the `pt_btagJes` jet feature and separately add noise with mean 0.1 and standard deviation 0.05 to the `d0` track feature.

#### Data From S3

To use S3 as an ATLAS user, some upstream setting up must be done with the [CERN OpenStack project](https://clouddocs.web.cern.ch/index.html). In particular, you must have access to a bucket and initialised your own public and secret keys. Once you have this information, you can access your bucket from anywhere using your credentials. To set up the credentials for salt, please incluse the following configuration in the `base_config.yaml` or your model config, under the `data` option:
```yaml
config_s3:
  use_S3: False  # Set to true to setup S3 (needed for storing results)
  download_S3: False # Set to true to download files in download_files from S3
  pubKey: # public key
  secKey: # private key
  url: https://s3.cern.ch # url, for OpenStack at cern used this.
  bucket: # bucket name
  download_path: # local path to download the files to
  download_files: # files key to download, matching an entry in the config.data
    - train_file
    - val_file
    - norm_dict
    - class_dict
```
Note that you can setup salt to use S3 to download your data locally with the `download_S3` key set to True and the files key (matching entries in the config `data` part of the yaml) being download locally to the `download_path`. Note that you can run a salt training directly on data located on S3 and downloading it locally: the download S3 scripts will update the paths to point locally automatically. You can also choose to first download the script with the salt-installed `download_S3` as such: 

```bash
download_S3 --config configs/GN2/GN2.yaml
```

This will run the downloading script without starting the salt CLI. 

Importantly, if your aim is to use S3 to store training data (configs, checkpoints of model, performance, ...), you must modify some entries in the callbacks in the base config. 
```yaml
trainer:
  ...
  default_root_dir: s3://BUCKET/FOLDER
  ...
  logger:
    class_path:  lightning.pytorch.loggers.TensorBoardLogger
```
As highlighted above, the `default_root_dir` should be a valid url to an S3 folder under your bucket. The default CometLogger will not work with S3 and you must instead use the TensorBoardLogger (please take care to not keep the instantiate arguments of commet by commenting `init_args: { project_name: salt, display_summary_level: 0 }`). 


### Model Architecture


#### Global Object Features

By default, inputs from the global object are concatenated with each of the input constituents at the beginning of the model
in the [`salt.models.InitNet`][salt.models.InitNet].
You can instead choose to concatenate global inputs with the pooled representation after the encoder step.
In order to this you should add a `global` key under `data.variables` and specify which global-level variables do you want to use.

??? warning "Don't forget to change pooling and task input size accordingly"

    If you concatenate 2 global variables you should increase the `input_size` by 2 for all tasks (except vertexing, here you should increase by 4).

For example you can concatenate jet features after the gnn model with:

`variables` section
```yaml
data:
    variables:
        global:
        - pt_btagJes
        - eta_btagJes
        ...
```

You can find a complete example of adding jet-level SMT variables in the [`GN2emu.yaml`]({{repo_url}}-/blob/main/salt/configs/GN2Cat.yaml) config.


#### Edge Features

It is possible to include edge features as network input, representing relational information between constituent tracks. These have to be implemented on an individual basis since they are not stored in the input files, but rather calculated on the fly within Salt. As such, the information is drawn from the track container (or equivalent) and requires the presence of track variables relevant to the calculation of each edge feature. Currently implemented features are:

- `dR` = $log(\sqrt{d\eta^2 + d\phi^2})$ (requires `phi`, `eta`)
- `kt` = $log(min(p_T)\sqrt{d\eta^2 + d\phi^2})$ (requires `pt`)
- `z` = $log(\frac{min(p_T)}{\Sigma(p_T)})$ (requires `pt`, `phi`, `eta`)
- `isSelfLoop` = 1 if edge represents self-connection, 0 if not
- `subjetIndex` = 1 if tracks are part of same subjet, 0 if not (requires `subjetIndex`)


#### Heterogeneous Models
If multiple input types are provided, separate initialiser networks should be provided for each input type.
An example using both track and electron input types is provided below:

```yaml
init_nets:
  - input_name: tracks
    dense_config: &init
    output_size: &embed_dim 192
    hidden_layers: [256]
    activation: &activation SiLU
  - input_name: electrons
    dense_config:
    <<: *init
```

The separate input types are by default combined and treated homogeneously within the GNN layers. 

#### Parameterisation

It is possible to condition the network output on particular variables (such as an exotic particles mass) to create a so-called *parameterised  neural network*. Parameters are treated as additional input variables during training and the user can chose which values to use when evaluating the model.

A parameterised network can be configured in the following way:

- `variables` section: Add your parameters to lists of variables used in training.
    ```yaml
    data:
        variables:
            ...
            parameters:
            - mass
            ...
    ```
- `parameters` section: In a dedicated section, for each parameter, specify the values of the parameter that appear in your training set (as a list in `train`) and the value you wish to evaluate the model at (`test`). Optionally you can include a list of probabilties for each parameter corresponding to the probabilities of assigning background jets one of the parameter values given in `train`. These probabilties should reflect how each parameter value is represented within the training data set. If probabilities are not given, values will be assigned to background jets with equal probability. Ensure parameters appear in the same order in `parameters` as in `variables`.
    ```yaml
    parameters:
        mass:
            train: [5, 16, 55]
            test: 40
            prob: [0.2, 0.3, 0.5]
    ```

The implementation above produces the default parameterisation mechanism in which parameters are concatenated to the inputs of the model, as described by [Baldi et al](https://arxiv.org/pdf/1601.07913.pdf). Alternatively, one may instead apply feature wise transformations, as described
by [Dumoulin et al](https://distill.pub/2018/feature-wise-transformations/). Here, parameters are passed as inputs to seperate networks whose outputs can be used to scale or bias the features of a layer. To use feature transformations you must add `featurewise_nets` to your models configuration as follows:


```yaml
  model:
    class_path: salt.models.SaltModel
    init_args:
    ...     
      featurewise_nets:
        - layer: input
          dense_config_scale:
            hidden_layers: [4]
            output_size: 17
          dense_config_bias:
            hidden_layers: [4]
            output_size: 17
        - layer: encoder
          apply_norm: True
          dense_config_scale:
            hidden_layers: [128]
            output_size: 256
          dense_config_bias:
            hidden_layers: [128]
            output_size: 256
        - layer: global
          dense_config_scale:
            output_size: 128
            hidden_layers: [64]
            final_activation: Sigmoid
    ...
```

Here, two instances of featurewise transformations have been added to the model. For each, you must specify the layer whose features you would
like to transform (this can currently be either `input`, which applies the transformations to the features before they are passed into the initialisation network, `encoder`, which applies the transformations to the inputs of each layer to the encoder using separate networks, or `global`, which applies them to the global track representations outputted by the encoder). For each instance, you can specify either one or both of `dense_config_scale` or `dense_config_bias`, which configure dense networks whose output scales and biases the features of the chosen layer, respectively. It is important to ensure the `output_size` of these networks matches the number of features in the layer you are transforming. In this case, the transformations are applied to a model with 17 inputs per track, the layers of an encoder with 256 features, and the output of the encoder, which has 128 features for each track representation. You can optionally apply a layer normalisation after applying the transformations by setting `apply_norm: True` for a given network, as shown above.


### Training

#### Compiled Models

Pytorch 2.0 introduced compiled models via `torch.compile()` which improves execution times.
In tests, compilation can increase the execution speed of the model by up to 1.5x.
You can enable compilation by passing the `--compile` flag to the CLI.
You may see some warnings printed at the start of training, and the first step will take a while as the model is JIT compiled.

??? failure "If you see `g++` compile errors, you may need to update your compiler"

    You can check your `g++`/`gcc` version with `g++ --version`.
    To use `torch.compile()`, you'll need `gcc` version 10 or later.

    You can install a more recent version with
    ```bash
    conda install -c conda-forge cxx-compiler
    ```

??? abstract "`torch.compile()` results"

    The following results were obtained on a single A100 GPU
    with a batch size of 5,000 and 40 workers.

    | Model      | Eager    | `torch.compile()` | Speedup |
    | ---------- | -------- | ----------------- | ------- |
    | GN3 No Aux | 8.9 it/s | 15.4 it/s         | 1.73x   |
    | GN3        | 6.4 it/s | 9.0  it/s         | 1.41x   |

    Memory usage should be unaffected by compiling the model.
    Please report any issues you may have

!!! warning "`torch.compile()` has not been tested with mutli GPU training"


### Hyperparameter Optimisation

#### Katib

In order to train salt on Katib, the performance must be printed to the output stream. The `PerformanceWriter` callback is available for that very purpose. It also stores the printed metrics in a json file stored at a writable local path `dir_path` (by default `trainer.log_dir`). For katib, it is important to set the stdout value to True and pointing the Katib metric collector to std_out. 

An example configuration to be added to the `base.yaml` config file is: 

```yaml
callbacks:
  - class_path: salt.callbacks.PerformanceWriter
    init_args:
      dir_path: /mylocal/path #any local path that is writable
      add_metrics: # a list of string of potential additional metrics - included by default: train_loss, val_loss, val_accuracy_loss
        - a_fancy_new_metric
        - another_fancy_new_metric 
      std_out: True # whether to print to std_out 
```



#### muTransfer

Salt is compatible with the muTransfer technique outline in the paper [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466).

##### Setup

To setup mup, the model configuration (e.g., `GN2.yaml`) has to include the following extra-configuration setup to be placed under the `config.model` (e.g., after `model.lrs_config` and before `model.model`):

```yaml
mup_config:
    shape_path: my_path_to_a_folder_for_shape
    embed_dim:
      apply_to: [init_nets, encoder]
      parameter_name: [output_size, embed_dim]
      parameter_base: 128 
      parameter_delta: 4
```

Such that the `base` (`delta`) models are instantiated with the parameters highlighted in `parameter_name`, respectively corresponding to the module `apply_to`, taking the value `parameter_base` (`parameter_delta`). The `storeshapes` file will be placed at the path `shape_path` or, if this parameter is not set, at `./temp_mup/` with the `base` and `delta` models as well as their configuration (useful to debug they were correctly setup).

To run a GN2 training with mup, you also need to specify in  `encoder` (and the `init_nets` if it is affected) config that it should be in `mup` configuration with the following boolean parameters: 

- for `init_nets` (only if changing embedding dim):

```yaml
init_nets:
    - input_name: tracks
        dense_config:
            ...
            mup: True
```

- for `encoder`:

```yaml
encoder:
    class_path: salt.models.TransformerEncoder
    init_args:
        ...
        mup: True
```


##### Run

To run mup, you must instantiate a GN2 model into the Maximal Update Parametrisation (mup). To do this, you must follow the following steps, which are further detailed next. 

- step 1: create `storeshapes` file using a model config file with mup configuration: 

```bash
setup_mup -config GN2/GN2.yaml
```

- step 2: run a mup training normally with the model config with mup configuration:

```bash
salt fit --config GN2/GN2.yaml
```

The config file `GN2_mup.yaml` gives an example of a valid configuration file for mup.

A gentle introduction to mup is available in this [talk](https://indico.cern.ch/event/1339085/#3-mup-for-gn2-hyperparameter-o).

Important note: mup has been implemented to scale the transformer encoder (and init_nets if the embedding is changed). The last layer in the scaling __must__ be the out-projecting of the encoder (controlled with `out_dim`), which in particular must be set!


**Step 1:**

To leverage the existing [mup library](https://github.com/microsoft/mup), a `base` and `delta` models have to be instantiated using the `main_mup` script to generate a `storeshapes` file to be passed to the mup library. Note that you __must__ vary a parameter between the `base` and `delta` models, as this will define the dimension to muTransfer along (embedding dimension and num_heads are supported). This script is installed with salt and callable under the name `setup_mup`. For example, run: 

```bash
setup_mup -c GN2/GN2.yaml
```

Where the `GN2.yaml` is your usual model configuration file, endowed with the following extra-configuration setup to be placed under the `config.model` (e.g., after `model.lrs_config` and before `model.model`):

```yaml
mup_config:
    shape_path: my_path_to_a_folder_for_shape
    embed_dim:
      apply_to: [init_nets, encoder]
      parameter_name: [output_size, embed_dim]
      parameter_base: 128 
      parameter_delta: 4
```

The `setup_mup` script will instantiate a `base` (`delta`) model with the parameters highlighted in `parameter_name`, respectively corresponding to the module `apply_to`, taking the value `parameter_base` (`parameter_delta`). The `storeshapes` file will be placed at the path `shape_path` or, if this parameter is not set, at `./temp_mup/` with the `base` and `delta` models as well as their configuration (useful to debug they were correctly setup). Note: currently supporting the num_heads & embedding size of the transformer `encoder`, with the latter being also relevant to `init_nets`. Both the base and delta value have to be divided by your chosen `num_heads`!



**Step 2:**

With step 1 creating a `storeshapes` under the path `shape_path` or the default `./temp_mup`, you can now turn to training a GN2 models with your desired widths. The model will have to load the `storeshapes` in the initialiser of `ModelWrapper`, and you must make sure the model has the mup_config passed to it with, in particular, the right path to the `storeshapes` (easiest is to not change the config w.r.t. base and delta model initialisation). 

To run a GN2 training with mup, you also need to specify in  `encoder` (and the `init_nets` if it is affected) config that it should be in `mup` configuration with the following boolean parameters: 
- for `init_nets` (only if changing embedding dim):
```yaml
init_nets:
    - input_name: tracks
        dense_config:
            ...
            mup: True
```
- for `encoder`:
```yaml
encoder:
    class_path: salt.models.Transformer
    init_args:
        ...
        mup: True
```

If correctly setup, you can just run a salt training in the usual way: 
```bash
salt fit --config GN2/GN2.yaml
```

You are now training a mup-GN2!
