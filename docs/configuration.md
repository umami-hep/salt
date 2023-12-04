# Advanced Configuration


### Dataloading

#### Selecting Training Variables

Training files are structured arrays, so it is easy to specify which variables you want to include in the training by name.

In your `data` config there is a `variables` key which specifies which variables to include in the training for each input type.
These are defined in the model files, rather than the `base.yaml` config.

??? warning "Make sure you do not train on truth information!"

    The variables listed under `data.variables` are the inputs to the training.
    You should include any truth information, but rather specify truth labels for each task in your model config.

For example, in [`GN1.yaml`]({{repo_url}}-/blob/main/salt/configs/GN1.yaml) you will find the following variables:

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

Training with multiple types of inputs beyond jets and tracks is supported to create a heterogeneous model. An example of this can be found in [`GN2emu.yaml`]({{repo_url}}-/blob/main/salt/configs/GN2emu.yaml) which includes a separate electrons input type.

#### Mapping Input Dataset Names

By default, the input names are used to directly retrieve dataset in the input h5 files.
If you want to use a different name to retrieve the h5 datasets, you can specify a mapping in the `data` config, using the `input_map` key.

```yaml
data:
  input_map:
    internal_name: h5_dataset_name
```

In this example, `h5_dataset_name` will be used to retrieve the datasets from the input h5 files, while internally (and elsewhere in the configuration) this input type will be referred to as `internal_name`.


#### Global Object Features

By default, inputs from the global object are concatenated with each of the input constituents at the beginning of the model
in the [`salt.models.InitNet`][salt.models.InitNet].
You can instead choose to concatenate global inputs with the pooled representation after the encoder step.
In order to this you should add a `GLOBAL` key under `data.variables` and specify which global-level variables do you want to use.

??? warning "Don't forget to change pooling and task input size accordingly"

    If you concatenate 2 global variables you should increase the `input_size` by 2 for `pool_net` and all tasks (except vertexing, here you should increase by 4).

For example you can concatenate jet features after the gnn model with:

`variables` section
```yaml
data:
    variables:
        GOBAL:
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


#### Parameterised GNN

It is possible to condition the network output on particular variables (such as an exotic particles mass) to create a so-called *parameterised  neural network*, as described by [Baldi et al](https://arxiv.org/pdf/1601.07913.pdf). Parameters are treated as additional input variables during training and the user can chose which values to use when evaluating the model.

A parameterised network can be configured in the following way:

- `inputs` section: Specify the collection containing the parameters.
    ```yaml
    input_names:
        ...
        PARAMETERS: jets
        ...
    ```
- `variables` section: Add your parameters to lists of variables used in training.
    ```yaml
    data:
        variables:
            ...
            PARAMETERS:
            - mass
            ...
    ```
- `PARAMETERS` section: In a dedicated section, for each parameter, specify the values of the parameter that appear in your training set (as a list in `train`) and the value you wish to evaluate the model at (`test`). Optionally you can include a list of probabilties for each parameter corresponding to the probabilities of assigning background jets one of the parameter values given in `train`. These probabilties should reflect how each parameter value is represented within the training data set. If probabilities are not given, values will be assigned to background jets with equal probability. Ensure parameters appear in the same order in `PARAMETERS` as in `variables`.
    ```yaml
    PARAMETERS:
        mass:
            train: [5, 16, 55]
            test: 40
            prob: [0.2, 0.3, 0.5]
    ```

??? warning "Ensure `concat_jet_tracks` is set to `true` when using parameters"

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


### Model Architecture

#### Switching Attention Mechanisms

By default the [`GN1.yaml`]({{repo_url}}-/blob/main/salt/configs/GN1.yaml) config uses the scaled dot product attention found in transformers.
To switch to the GATv2 attention, add the [`GATv2.yaml`]({{repo_url}}-/blob/main/salt/configs/GATv2.yaml) config fragment.

```bash
salt fit --config configs/GN1.yaml --config configs/GATv2.yaml
```

Note that the  `num_heads` and `head_dim` arguments must match those in the `gnn.init_args` config block.


#### Switching Pooling Mechanisms

The [`GN2.yaml`]({{repo_url}}-/blob/main/salt/configs/GN2.yaml) config by default uses Global Attention Pooling. This can be switched out for Cross Attention Pooling where a learnable class token is attended to by the learned track representation. The same arguments used for the Transformer Encoder apply to the Cross Attention layers. An example Cross Attention block is shown below:

```yaml
pool_net:
class_path: salt.models.CrossAttentionPooling
init_args:
    input_size: 128
    num_layers: 4
    mha_config:
        num_heads: 4
        attention:
            class_path: salt.models.ScaledDotProductAttention
        out_proj: False
    dense_config:
        norm_layer: *norm_layer
        activation: *activation
        hidden_layers: [128]
        dropout: 0.1

```


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
    norm_layer: &norm_layer LayerNorm
  - input_name: electrons
    dense_config:
    <<: *init
```

The separate input types are by default combined and treated homogeneously within the GNN layers. Alternatively, each input type can be updated with separate self-attention blocks and cross-attention blocks between each input type:

```yaml
    encoder:
    class_path: salt.models.TransformerCrossAttentionEncoder
    init_args:
        input_names: [tracks, electrons]
        ca_every_layer: true
        embed_dim: 192
        num_layers: 6
        out_dim: &out_dim 128
        mha_config:
        num_heads: 4
        attention:
            class_path: salt.models.ScaledDotProductAttention
        out_proj: False
        sa_dense_config:
        norm_layer: *norm_layer
        activation: *activation
        hidden_layers: [256]
        dropout: &dropout 0.1
```

#### Compiled Models

Pytorch 2.0 introduced compiled models via `torch.compile()` for improved execution times. 
You can enabled this by passing the `--compile` flag to the CLI.
When enabled, you may see some warnings printed at the start of training, and the first step will take a while as the model is JIT compiled.

As of pytorch 2.0.1 there are some known limitations when compiling models:

- Compiled models have fixed shapes, which precludes running the vertexing aux task (which uses tensors of variable size depending on the number of valid tracks in the batch).
- There may be issues exporting a compiled model with ONNX.

With these limitations in mind, you may find compiling your model useful for speeding up studies in cases where you don't require the vertexing task or ONNX export.

You can also try to decorate functions with `@torch.compile`, for example the `forward()` methods of various submodules.
Passing `mode="reduce-overhead"` may also improve further performance.
Note that this will break ONNX export.

### Katib

In order to train salt on Katib, the performance must be printed to the output stream. The `PerformanceWriter` callback is available for that very purpose. It also stores the printed metrics in a json file stored at a writable local path `dir_path` (by default `trainer.log_dir`). For katib, it is important to set the stdout value to True and pointing the Katib metric collector to stdOut. 

An example configuration to be added to the `base.yaml` config file is: 

```yaml
callbacks:
  - class_path: salt.callbacks.PerformanceWriter
    init_args:
      dir_path: /mylocal/path #any local path that is writable
      add_metrics: # a list of string of potential additional metrics - included by default: train_loss, val_loss, val_accuracy_loss
        - a_fancy_new_metric
        - another_fancy_new_metric 
      stdOut: True # whether to print to stdOut 
```

### muP

More detail on muP is given in the training docs, but the relevant configuration setups are sumamrised here. The model configuration (e.g., `GN2.yaml`) has to include the following extra-configuration setup to be placed under the `config.model` (e.g., after `model.lrs_config` and before `model.model`):

```yaml
muP_config:
    shape_path: my_path_to_a_folder_for_shape
    embed_dim:
      apply_to: [init_nets, encoder]
      parameter_name: [output_size, embed_dim]
      parameter_base: 128 
      parameter_delta: 4
```

Such that the `base` (`delta`) models are instantiated with the parameters highlighted in `parameter_name`, respectively corresponding to the module `apply_to`, taking the value `parameter_base` (`parameter_delta`). The `storeshapes` file will be placed at the path `shape_path` or, if this parameter is not set, at `./temp_muP/` with the `base` and `delta` models as well as their configuration (useful to debug they were correctly setup).

To run a GN2 training with muP, you also need to specify in  `encoder` (and the `init_nets` if it is affected) config that it should be in `muP` configuration with the following boolean parameters: 
- for `init_nets` (only if changing embedding dim):
```yaml
init_nets:
    - input_name: tracks
        dense_config:
            ...
            muP: True
```
- for `encoder`:
```yaml
encoder:
    class_path: salt.models.TransformerEncoder
    init_args:
        ...
        muP: True
```

### S3:

To use S3 as an ATLAS user, some upstream setting up must be done with the [CERN OpenStack project](https://clouddocs.web.cern.ch/index.html). In particular, you must have access to a bucket and initialised your own public and secret keys. Once you have this information, you can access your bucket from anywhere using your credentials. To set up the credentials for salt, please incluse the following configuration in the `base_config.yaml` or your model config, under the `data` option:
```yaml
config_S3:
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
download_S3 --config configs/GN2.yaml
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
