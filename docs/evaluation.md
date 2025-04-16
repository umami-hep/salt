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
