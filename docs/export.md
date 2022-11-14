In order to use your trained model in Athena you need to export it to [ONNX](https://onnxruntime.ai/).


### Model Conversion

The `to_onnx.py` python script handles the ONNX conversion process for you.
The script has several arguments, you can learn about them by running

```bash
to_onnx --help
```

At a minimum, you need to specify the path to a training config, a checkpoint to convert, and a track selection.
For example

```bash
to_onnx \
    --config logs/timestamp/config.yaml \
    --ckpt_path logs/timestamp/ckpts/checkpoint.ckpt \
    --track_selection r22default
```

??? warning "Track selection"

    The track selection you specify must correspond to one of the options defined in `trk_select_regexes` variable in
    [`DataPrepUtilities.cxx`](https://gitlab.cern.ch/atlas/athena/-/blob/master/PhysicsAnalysis/JetTagging/FlavorTagDiscriminants/Root/DataPrepUtilities.cxx).

    The selection you use must also match the selection applied in your training samples.
    Track selection is applied when dumping using the TDD.
    The current default FTAG selection is called `r22default`, but you should take note of the changes described in
    [training-dataset-dumper!427](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/-/merge_requests/427)
    to make sure you are using the correct selection.

You can also optionally specify a different scale dict to the one in the training config, and a model name (by default this is `salt`).
The model name is used to construct the output probability variable names in Athena.


### Final Validation

You may see some warnings during export, but the `to_onnx` script will verify that there is a good level of compatability between the pytorch and ONNX model outputs, and that there are no `nan` or `0` values in the ouput.

However, as a final check, you should verify the performance of your pytorch model against a version running from the TDD by following the instructions [here](https://training-dataset-dumper.docs.cern.ch/configuration/#dl2-config) to dump the scores of your converted model.
Make sure to dump at full precision (use the provided flag) so you can then rerun the dump through your pytorch model to fully close the loop.


### Viewing ONNX Model Metadata

To view the metadata stored in an ONNX file, you can use

```bash
get_onnx_metadata path/to/model.onnx
```

Inside are the list of input features including normalisation values, and also the list of outputs and the model name.


??? info "A command with the same name is also available in Athena"

    After setting up Athena, you can also run a different [`get_onnx_metadata`](https://gitlab.cern.ch/atlas/athena/-/blob/master/PhysicsAnalysis/JetTagging/FlavorTagDiscriminants/util/get-onnx-metadata.cxx) command which has the same function.
