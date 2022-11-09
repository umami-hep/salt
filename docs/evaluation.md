You can evaluate models tried using salt over a test set.
Test samples are loaded from structured numpy arrays stored in h5 files in the same format as those produced by the TDD,
or the "preparation" stage of umami.
After producing the evaluation file, you can make performance plots using [puma](https://github.com/umami-hep/puma).

### Running the Test Loop

To evaluated a trained model on a test file, the `test` subcommand is used.

```bash
main test --config logs/<timestamp>/config.yaml --data.test_file path/to/test.h5
```

As in the above example, you need to specify the saved config from the training run.
By default, the checkpoint with the lowest validation loss is used for training.
You can specify a different checkpoint with the `--ckpt_path` argument.

You also need to specify a path to the test file using `--data.test_file`.
This should be a prepared umami test file, and the framework should extract
the sample name and append this to the checkpint file basename.
The result is saved as an h5 file in the `ckpts/` dir.

You can use `--data.num_jets_test` to set the number of training jets if you want to
override the default in the training config.

??? warning "Only one GPU is supported for the test loop."

    When testing, only a single GPU is supported.
    This is enforced by the framework, so if you try to use more than one device you will see a message
    `Setting --trainer.devices=1`


??? warning "Output files are overwritten by default."

    Get in touch if this is a problem.


### Track evaluation

By default, only the jet quantities are evaluated to save time and space.
If you want to study the track aux task performance, you need to specify `write_tracks: True` in the `PredictionWriter` callback configuration in the base configuration file.
