A series of preprocessing steps are required to extract information from xAOD files into a format conducive for training.
If you want to get started running the code without producing your own samples, some samples are available on EOS

- `/eos/user/u/umami/training-samples/gnn/test/` - 10M jets for single b-tagging
- `/eos/user/u/umami/training-samples/gnn/xbb/` - 12M jets for Xbb tagging

### Dumping Training Samples

Training samples are created using the [training dataset dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/).
The default config file [`EMPFlow.json`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/-/blob/r22/configs/single-b-tag/EMPFlowGNN.json) has all the information required to train models with salt.

Predumped h5 samples are available [here](https://ftag.docs.cern.ch/software/samples/).


### Preprocessing with Umami

The h5 files produced by the TDD are processed by the [umami framework](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tree/master/umami) to produce training files.
The umami framework handles jet selection, kinematic resampling, normalisation and shuffling.
The [default preprocessing config](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/preprocessing/PFlow-Preprocessing.yaml) and [this variable config](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/configs/GNN_Variables.yaml) are good places to start for the creation of train samples for salt.

For more information on how to configure the preprocessing, take a look at the umami [docs](https://umami-docs.web.cern.ch/preprocessing/ntuple_preparation/#config-file).

#### Preprocessing Requirements

1. Please ensure you run preprocessing with a recent version of umami that includes [!648](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/648) and [!665](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/665) (i.e. versions >=0.17).

2. It is also recommend to set `concat_jet_tracks: True` in your preprocessing config. If you want to concatenate only a subset of jet variables to each track, just provide the variable names as a list. See the [here](https://umami-docs.web.cern.ch/preprocessing/write_train_sample/#config-file) for more info.

3. Finally, it is recommended to produce the training samples with 16-bit floating point precision. To do this set `precision: float16` in your preprocessing config. Reducing the precision leads to significantly smaller filesizes and improved dataloading speeds while maintaining the same level of performance.

#### Creating the Validation Sample

Umami can create a resampled validation file for you.
See [here](https://umami-docs.web.cern.ch/preprocessing/resampling/#create-the-resampled-hybrid-validation-sample) and [here](https://umami-docs.web.cern.ch/preprocessing/write_train_sample/#writing-validation-samples).

#### Directory Structure

Training files are suggested to follow a certain directory structure, which is based on the output structure of umami preprocessing jobs.

```bash
- base_dir/
    - train_sample_1/
        # umami configuration
        - PFlow-Preprocessing.yaml
        - PFlow-scale_dict.json
        - GNN_Variables.yaml

        # tdd output datasets
        - source/
            - tdd_output_ttbar/
            - tdd_output_zprime/

        # umami hybrid samples
        - prepared/
            - MC16d-inclusive_testing_ttbar_PFlow.h5
            - MC16d-inclusive_testing_zprime_PFlow.h5

        # umami preprocessed samples
        - preprocessed/
            - PFlow-hybrid-resampled_scaled_shuffled.h5
            - PFlow-hybrid-validation-resampled_scaled_shuffled.h5
```
