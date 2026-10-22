In order to use your trained model in Athena you need to export it to [ONNX](https://onnxruntime.ai/).


### Model Export

The `onnx/to_onnx.py` python script handles the ONNX conversion process for you.
The script has several arguments, you can learn about them by running

```bash
to_onnx --help
```

At a minimum, you should specify the path to a checkpoint to convert and a name for the exported model.
For example

```bash
to_onnx \
    --ckpt_path logs/<timestamp>/ckpts/checkpoint.ckpt \
    --name GN2vXX
```

If you don't specify a config path using `--config`, the script will look for a `config.yaml` in the run directory, i.e. one level above the `ckpts/` dir containing the specified `--ckpt_path`.

??? warning "The `r22default` track selection is used by default"

    The track selection you specify must correspond to one of the options defined in `trk_select_regexes` variable in
    [`DataPrepUtilities.cxx`](https://gitlab.cern.ch/atlas/athena/-/blob/master/PhysicsAnalysis/JetTagging/FlavorTagDiscriminants/Root/DataPrepUtilities.cxx).

    The selection you use must also match the selection applied in your training samples.
    Track selection is applied when dumping using the TDD.
    The current default FTAG selection is called `r22default`, but you should take note of the changes described in
    [training-dataset-dumper!427](https://gitlab.cern.ch/aft/algorithms/training-dataset-dumper/-/merge_requests/427)
    to make sure you are using the correct selection.

You can also optionally specify a model name with `--name` (by default this is taken from the `name` in the training config).
The model name is used to construct the output probability variable names in Athena.

??? tip "Exporting a model trained with `torch.compile()`."

    If you trained your model with `torch.compile()`, you need to repair your checkpoint before exporting.
    You can do this by running the [`repair_ckpt.py`](https://gitlab.cern.ch/aft/algorithms/salt/-/blob/main/salt/utils/repair_ckpt.py) 
    script:

    ```bash
    repair_ckpt <path_to_checkpoint>
    ```

### Batched standalone export

By default `to_onnx` produces an Athena-style model that runs a **single event per inference call**: sequence inputs (tracks, flows, ...) are given without an event batch dimension and padding masks are not part of the interface.

If instead you want to run the model in a **standalone ONNX Runtime** environment and evaluate **many events per inference call**, pass `--batched-standalone`:

```bash
to_onnx \
    --ckpt_path logs/<timestamp>/ckpts/checkpoint.ckpt \
    --name eventmodel \
    --batched-standalone
```

This changes the exported interface as follows:

- The global input keeps shape `[batch, features]`.
- Each sequence input keeps shape `[batch, n_objects, features]` (an explicit event batch dimension is added).
- Each sequence input gets an accompanying **boolean mask input** named `<sequence>_features_mask` with shape `[batch, n_objects]`, where `True` marks a **padded** object and `False` a real one.
- The `batch` axis (and the per-sequence object axis) are marked as dynamic, so a single exported model handles any number of events and objects.

!!! warning "Not Athena-compatible"

    `--batched-standalone` exports are intended for standalone ONNX Runtime inference only and **will not work in Athena**. The script emits a `UserWarning` to remind you of this. For Athena deployment, export without this flag.

!!! info "Global (event-level) outputs only"

    Batched standalone export currently emits **global/event-level task outputs only**. Auxiliary per-object outputs (`track_origin`, `track_vertexing`, `track_type`) and MaskFormer object outputs (`--object_name`) are **not yet implemented** for this mode (their validation needs batched post-processing that is not in place — a SALT limitation, not an ONNX one) and will raise an error. Use the default export for those.

As with the default path, the script validates the exported model against PyTorch (see [Athena Validation](#athena-validation)) before saving, this time sweeping a range of sequence lengths over a multi-event batch with padded objects.

### Combining outputs
For compatability, or just ease, it might be required to combine outputs. This can be done via the 'combine_outputs' argument, which takes the form

```
--combine_outputs plight,pquark,pgluon pquark,0.5*pud,1.5*pgluon
```

In the above, we define two combinations:
- The first, we define `plight = 1.0 * pquark + 1.0 * pgluon`
- The second, we define `pquark = 0.5*pud + 1.5*pgluon`

### Renaming outputs
Renaming of global task outputs can be performed via the `--rename` argument:

```
--rename poldnamebjet:pb poldnamecjet:pc
```

Which will rename `{name}_poldnamebjet` to `{name}_pb`, and `{name}_poldnamecjet` to `{name}_pc`

### Athena Validation

You may see some warnings during export, but the `to_onnx` script will verify that there is a good level of compatability between the pytorch and ONNX model outputs, and that there are no `nan` or `0` values in the output.
However, as a final check, you should verify the performance of your pytorch model against a version running from the TDD.

First, follow the instructions [here](https://training-dataset-dumper.docs.cern.ch/configuration/#dl2-config) to dump the scores of your export model.
Please take note of the following considerations when comparing Athena and Python evaluated models:

- Models in Athena are evaluated with full precision inputs. Make sure to dump using the TDD at full precision (use the provided flag `--force-full-precision`).
- Models evaluated in Python are limited to 40 input tracks, whereas models evaluated in Athena have no such limit.
- The exported model has every Softmax written out as max, exp, sum and divide, because the onnxruntime Softmax gives slightly different results on different CPUs. The outputs agree with PyTorch at the same level as before.


Once you have evaluated your model using the TDD, you should use the resulting h5 file to run `salt test`.
Be sure to run with `--trainer.precision 32`.

Finally, you can then the `compare_models` command to compare the scores of the two models.

```bash
compare_models \
    --file_a tdd/output.h5 \
    --tagger_a name \
    --file_b salt/eval.h5 \
    --tagger_b name
```

See `compare_models.py -h` for more information.

??? info "What level of discrepancy is expected?"

    We usually ask for agreement within `1e-6` for the output probabilities, which is approximately floating point precision error.
    If you see one or two jets with a discrepancy of `1e-5`, this is probably fine.
    Common causes of more significant discrepancies are:

    - Not dumping at full precision using the TDD (see above)
    - Not running `salt test` with `--trainer.precision 32` if you trained at lower precision.
    - Not writing out salt evaluation scores at full precision (see the `PredictionWriter` callback)
    - Enabling some runtime optimisaiton in pytorch (e.g. [here](https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision))



### Deploying FTAG Models in Athena

Please see [this page](https://ftag.docs.cern.ch/reco_algs/taggers/deploy/) in the central FTAG documentation.


### Viewing ONNX Model Metadata

To view the metadata stored in an ONNX file, you can use

```bash
get_onnx_metadata path/to/model.onnx
```

Inside are the list of input features including normalisation values, and also the list of outputs and the model name.

??? info "A command with the same name is also available in Athena"

    After setting up Athena, you can also run a different [`get_onnx_metadata`](https://gitlab.cern.ch/atlas/athena/-/blob/master/PhysicsAnalysis/JetTagging/FlavorTagDiscriminants/util/get-onnx-metadata.cxx) command which has the same function.
