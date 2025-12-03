You can evaluate models train using salt over a test set.
Test samples are loaded from structured numpy arrays stored in h5 files, as for training.
After producing the evaluation file, you can make performance plots using [puma](https://github.com/umami-hep/puma).

### Running the Test Loop

To evaluate a trained model on a test file, use the `salt test` command.

```bash
salt test --config logs/<timestamp>/config.yaml --data.test_file path/to/test.h5
```

As in the above example, you need to specify the saved config from the training run.
By default, the checkpoint with the lowest validation loss is used for training.
You can specify a different checkpoint with the `--ckpt_path` argument.

??? warning "When evaluating a model from a [resumed training][resuming-training], you need to explicitly specify `--ckpt_path`."

    When you [resume training][resuming-training], you specify a `--ckpt_path` and this is saved with the model config.
    If you then run `salt test` on the resulting config without specifying a new `--ckpt_path`, this same checkpoint will
    we be evaluated. To instead evaluate on the desired checkpoint from the resumed training job, you should explicitly
    specify `--ckpt_path` again to overwrite the one that is already saved in the config.

    If you still want to choose the best epoch automatically, use `--ckpt_path null`.

You also need to specify a path to the test file using `--data.test_file`.
This should be a prepared umami test file, and the framework should extract
the sample name and append this to the checkpint file basename.
The result is saved as an h5 file in the `ckpts/` dir.

You can use `--data.num_test` to set the number of samples to test on if you want to
override the default value from the training config.

??? info "Only one GPU is supported for the test loop."

    When testing, only a single GPU is supported.
    This is enforced by the framework, so if you try to use more than one device you will see a message
    `Setting --trainer.devices=1`


??? warning "Output files are overwritten by default."

    You can use `--data.test_suff` to append an additional suffix to the evaluation output file name.

### Extra Evaluation Variables
When evaluating a model, the jet and track variables included in the output file can be configured.
The variables can be configured as follows within the `PredictionWriter` callback configuration in the base configuration file.

```yaml
callbacks:
    - class_path: salt.callbacks.Checkpoint
      init_args:
        monitor_loss: val/jet_classification_loss
    - class_path: salt.callbacks.PredictionWriter
      init_args:
        write_tracks: False
        extra_vars:
          jets:
            - pt_btagJes
            - eta_btagJes
            - HadronConeExclTruthLabelID
            - n_tracks
            - n_truth_promptLepton
            tracks:
            - truthOriginLabel
            - truthVertexIndex
```

By default, only the jet quantities are evaluated to save time and space.
If you want to study the track aux task performance, you need to specify `write_tracks: True` in the `PredictionWriter` callback configuration.

The full API for the `PredictionWriter` callback is found below.

### ::: salt.callbacks.PredictionWriter

### Integrated Gradients 

Integrated gradients is a method for attributing contributions from each input feature to model outputs. Further details
can be found [here](https://indico.cern.ch/event/1526345/contributions/6446806/attachments/3044515/5379104/IG_PUB-1.pdf).
A callback can be added to the config after training has been completed, before evaluation is run. 
An example can be found below:


```yaml
callbacks:
  - class_path: salt.callbacks.IntegratedGradientWriter
    init_args:
      add_softmax: true
      n_baselines: 5
      min_allowed_track_sizes: 5
      max_allowed_track_sizes: 25
      n_steps: 50
      n_jets: 100_000
      internal_batch_size: 10_000
      input_keys:
        inputs: 
          - jets
          - tracks
        pad_masks:
          - tracks
      output_keys: [jets, jets_classification]
      overwrite: true
```

Descriptions of the parameters can be found below:

### ::: salt.callbacks.IntegratedGradientWriter

### Confusion Matrix

A callback to log the confusion matrix during training, at the end of each epoch. The confusion matrix is calculated on the validation dataset.
An example can be found below:

```yaml
callbacks:
  - class_path: salt.callbacks.ConfusionMatrixCallback
    init_args:
      task_name: "jets_classification"
      class_names_override: ["b-jets", "c-jets", "u-jets"]
```

The `class_names_override` can also be a mapping between the existing class names and the new ones.
This is particularly useful when the user wants to override only some of the class names.
Class names that don't appear as keys in the mapping are left unchanged.

Descriptions of the parameters can be found below:

### ::: salt.callbacks.ConfusionMatrixCallback
