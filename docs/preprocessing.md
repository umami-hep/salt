At the moment, Salt accepts data in HDF5 format.
The specific format of the H5 files is described [below][salt-input-format].

A series of steps are therefore to extract data from xAODs into a format that can be read by salt.
If you want to get started running the code without producing your own samples, some samples are available on EOS:

??? example "Single b-tagging"

    Several 30M jet slices are available below in the following folder:

    ```
    /eos/atlas/atlascerngroupdisk/perf-flavtag/training/training_gn2_fold3_20231205_mc20mc23_combined_270M
    ```

    The samples contain only jets from the 3rd fold in a 4-fold setup (see the preprocessing configs for more info).
    
    If you want to train on more than 30M jets, you can easily combine the files using
    [`atlas-ftag-tools`](https://github.com/umami-hep/atlas-ftag-tools/#create-virtual-file)

??? example "Xbb tagging"
    
    See also the [xbb docs](https://xbb-docs.docs.cern.ch/Samples/Training/)

    | Sample | Num Jets | Location |
    |--------|----------|----------|
    | Xbb tagging | 5M | `/eos/user/u/umami/training-samples/gnn/xbb/` |
    | Xbb tagging (new format) | 6M | `/eos/user/u/umami/training-samples/gnn/xbb_3d/` |



## xAOD to H5 Dumping

Training samples are created using the [training dataset dumper](https://gitlab.cern.ch/aft/algorithms/training-dataset-dumper/) (TDD).
The default config file [`EMPFlow.json`](https://gitlab.cern.ch/aft/algorithms/training-dataset-dumper/-/blob/r22/configs/single-b-tag/EMPFlowGNN.json) has all the information required to train models with salt.

Pre-dumped h5 samples are available [here](https://ftag.docs.cern.ch/samples/h5_sample_list/).


## Python pre-processing

The H5 files produced by the TDD are processed by [UPP](https://github.com/umami-hep/umami-preprocessing) to produce training, validation and testing files.
UPP is a flexible and easy to use package which handles object selection, kinematic resampling, normalisation, shuffling and train/val/test splitting.

Training files are suggested to follow a certain directory structure, which is based on the output structure of UPP preprocessing jobs.

???info "Recommended directory structure"

    ```bash
    - base_dir/
        - sample_1/

            # tdd output datasets
            - ntuples/
                - dsid_1/
                - dsid_2/
                ...

            # UPP output files, used for training
            - output/
                - norm_dict.yaml
                - class_dict.yaml
                - pp_output_train.h5
                - pp_output_val.h5
                - pp_output_test_ttbar.h5
                - pp_output_test_zprime.h5

            # some other intermediate UPP outputs

        - sample_2/
            # as above
    ```


## Salt input format

To see how data is structured when inputted to a salt model,
take a look at the signature of the `forward()` function of the
[`SaltModel` class][salt.models.SaltModel].
