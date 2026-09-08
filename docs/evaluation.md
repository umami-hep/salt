# Evaluation

You can evaluate a trained salt model over a test set with `salt test`. Test
samples are loaded from structured HDF5 files, the same format used for
training. Each test file requires a separate `salt test` call; there is no
multi-test-file mode. After producing the evaluation file, you can make
performance plots using [puma](https://github.com/umami-hep/puma).

## Running the test loop

`salt test` evaluates a trained model on a test file:

```bash
salt test --config logs/<timestamp>/config.yaml --data.test_file path/to/test.h5
```

See [`salt test`](cli.md#salt-test) for the full flag reference, including
`--ckpt_path`, `--data.num_test` and the checkpoint-resolution rules used
when you evaluate a model whose training run was itself
[resumed](training.md#checkpoints-and-resume). The evaluation file is
written next to the checkpoint that was read; see
[what `salt test` writes](outputs.md#what-salt-test-writes) for the exact
path and naming.

??? info "Only one GPU is supported for the test loop"

    Multi-device test writing is out of scope: the framework forces
    `--trainer.devices=1` and logs `salt test: forcing --trainer.devices=1
    (single-device eval)` if you ask for more than one device.

??? warning "Output files are overwritten by default"

    Use `--data.test_suff` to append an additional suffix to the evaluation
    output file name, so repeated `salt test` calls on the same checkpoint
    don't clobber each other's output.

??? info "Where the `{sample}` in the output filename comes from"

    The output filename is `{checkpoint_stem}__test_{sample}.h5`. `{sample}`
    is the fourth underscore-separated field of the test file's stem if it
    has exactly four (the umami naming convention), otherwise the whole
    stem, with `--data.test_suff` appended if you set one.

## Extra evaluation variables

Which jet and track variables land in the evaluation file is controlled by
`InputCopyWriter`'s `variables:` mapping inside your config's `outputs:`
section, not a callback. See
[Choose which input variables are copied](outputs.md#choose-which-input-variables-are-copied)
for the exact YAML shape and defaults.

## Confusion matrix

A callback to log the confusion matrix during training, at the end of each
epoch. The confusion matrix is calculated on the validation dataset. An
example:

```yaml
callbacks:
  confusion_matrix:
    class_path: salt.callbacks.ConfusionMatrix
    init_args:
      task_name: "jets_classification"
      class_names_override: ["b-jets", "c-jets", "u-jets"]
```

`callbacks:` is a name-keyed dict that deep-merges across your config stack, so an
overlay can replace one callback's `init_args` or drop it entirely with
`<name>: null`. `task_name` is the `model.modules` dict key of the classification
task this callback watches.

`class_names_override` can also be a mapping between the existing class
names and the new ones. This is particularly useful when you want to
override only some of the class names; class names that don't appear as
keys in the mapping are left unchanged.

## What the eval file contains

The evaluation file's groups mirror your reader's streams (a `jets` group, a
`tracks` group, and so on), each a structured array with one row per jet or
event. Within a group, columns come in a fixed order: input copies, task
outputs, target labels, then the pad mask. See
[what `salt test` writes](outputs.md#what-salt-test-writes) for the full
column model and where each block comes from, and
[Declaring outputs](outputs.md#declaring-outputs-the-outputs-section) for
how to change what gets written.

To read the file back into Python (for example to check an ONNX export
against the scores `salt test` produced), see
[Check it against salt](deployment/python.md#4-check-it-against-salt).

!!! note "`hdf5plugin` is not optional"

    salt writes its eval H5 with a blosc filter. Without `import hdf5plugin`
    the read fails with an unhelpful `can't open directory (.../plugin)`
    error rather than anything about compression.
