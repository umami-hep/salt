A series of preprocessing steps are required to extract information from xAOD files into a format conducive for training.
If you want to get started running the code without producing your own samples, some samples are available on EOS:

| Sample | Num Jets | Location |
|--------|----------|----------|
| Single b-tagging | 30M | `/eos/atlas/atlascerngroupdisk/perf-flavtag/training/training_gn2_20230915_mc20mc23_combined_30Mjets` |
| Single b-tagging | 300M | `/eos/atlas/atlascerngroupdisk/perf-flavtag/training/training_gn2_20230915_mc20mc23_combined_300Mjets` |
| Xbb tagging | 5M | `/eos/user/u/umami/training-samples/gnn/xbb/` |
| Xbb tagging (new format) | 6M | `/eos/user/u/umami/training-samples/gnn/xbb_3d/` |



### Dumping Training Samples

Training samples are created using the [training dataset dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/).
The default config file [`EMPFlow.json`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/-/blob/r22/configs/single-b-tag/EMPFlowGNN.json) has all the information required to train models with salt.

Pre-dumped h5 samples are available [here](https://ftag.docs.cern.ch/software/samples/).

### Pre-processing 

The h5 files produced by the TDD are processed by UPP or umami to produce training, validation and testing files.
The preprocessing framework handles jet selection, kinematic resampling, normalisation and shuffling.
UPP is a newer preprocessing package that is faster and more flexible than the older umami.

=== "UPP" 
    
    UPP is a new package for preprocessing.
    The repo can be found [here](https://github.com/umami-hep/umami-preprocessing).
    Documentation is also available [here](https://umami-hep.github.io/umami-preprocessing/).

=== "Umami"

    ### Preprocessing with Umami

    The umami repo can be found [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tree/master/umami).

    The [default preprocessing config](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/examples/preprocessing/PFlow-Preprocessing.yaml) and [this variable config](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/configs/GNN_Variables.yaml) are good places to start for the creation of train samples for salt.

    For more information on how to configure the preprocessing, take a look at the umami [docs](https://umami-docs.web.cern.ch/preprocessing/ntuple_preparation/#config-file).

    ???+ warning "Updates to the training file format"

        In [!87](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt/-/merge_requests/87), Salt switched to a training file format which is the same as the TDD output format.
        This is both easier to use, more flexible, and faster.
        If you are using umami, instead of the `*-hybrid-resampled_scaled_shuffled.h5`, you should use the resampled (but not scaled or shuffled) file, e.g. `*-hybrid-resampled.h5`.

        If you have [umami/!713](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/merge_requests/713), then the resampled file is ready to go out of the box.
        If not, the jet flavour labels are stored in their own `labels/` dataset.
        To access them, you will need to set `input_name: /` and `label: labels` in your jet classification task config.
        Alternatively, you can use an on the fly label mapping using the task's `label_map` option, see [here](training.md#remapping-labels)


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
