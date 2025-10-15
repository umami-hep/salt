import h5py
import numpy as np
import pytest
import torch
from ftag import get_mock_file, Flavours

from salt.data import SaltDataset
from salt.data.datasets import malformed_truthorigin_check
from salt.utils.configs import LabellerConfig


def test_salt_dataset():
    f = get_mock_file()[0]
    norm_dict = {}
    variables = {"jets": ["pt", "eta"], "tracks": ["d0"]}
    dataset = SaltDataset(f, norm_dict, variables, "train")
    for i in range(0, len(dataset), 10):
        dataset[i : i + 10]


def test_track_selector():
    f = get_mock_file()[0]
    norm_dict = {}
    variables = {"jets": ["pt", "eta"], "tracks": ["d0"]}
    selections = {"tracks": ["d0 < 3.5"]}
    no_sel = SaltDataset(f, norm_dict, variables, "train")
    sel = SaltDataset(f, norm_dict, variables, "train", selections=selections)
    for i in range(0, len(no_sel), 10):
        inputs, masks, _ = no_sel[i : i + 10]
        assert torch.any(inputs["tracks"][~masks["tracks"]] > 3.5)
        inputs, masks, _ = sel[i : i + 10]
        assert not torch.any(inputs["tracks"][~masks["tracks"]] > 3.5)


def test_input_batch():
    f = get_mock_file()[0]
    norm_dict = {}
    variables = {
        "jets": [
            "pt",
            "eta",
            "R10TruthLabel_R22v1",
            "GhostBHadronsFinalCount",
            "GhostCHadronsFinalCount",
        ],
        "tracks": ["d0"],
    }
    labels = {"jets": ["flavour_label"], "tracks": ["numberOfSCTSharedHits"]}
    cn = list(Flavours.by_category("xbb").labels.keys())
    labellerConf = LabellerConfig(use_labeller=True, class_names=cn, require_labels=False)
    dataset = SaltDataset(
        f, norm_dict, variables, "train", labels=labels, labeller_config=labellerConf
    )
    input_map = {k: k for k in variables}
    for i in range(0, len(dataset), 10):
        ds = dataset[i : i + 10]
        labels = ds[2]
        for input_name in input_map:
            batch = dataset.arrays[input_name]
            labels[input_name] = {}
        assert batch.shape == (10, 40)
        assert input_map == {"jets": "jets", "tracks": "tracks"}


def test_process_labels():
    f = get_mock_file()[0]
    norm_dict = {}
    variables = {
        "jets": [
            "pt",
            "eta",
            "R10TruthLabel_R22v1",
            "GhostBHadronsFinalCount",
            "GhostCHadronsFinalCount",
        ]
    }
    labels = {"jets": ["flavour_label"]}
    cn = list(Flavours.by_category("xbb").labels.keys())
    labellerConf = LabellerConfig(use_labeller=True, class_names=cn, require_labels=False)
    dataset = SaltDataset(
        f, norm_dict, variables, "train", labels=labels, labeller_config=labellerConf
    )
    input_map = {k: k for k in variables}
    for i in range(0, len(dataset), 10):
        ds2 = dataset[i : i + 10]
        labels = ds2[2]
        for input_name in input_map:
            batch = dataset.arrays[input_name]
            labels[input_name] = {}
            output = dataset.process_labels(labels, batch, input_name)
            assert isinstance(*output.values(), torch.Tensor) is True


def test_file_vars_for_cuts():
    f = get_mock_file()[0]
    norm_dict = {}
    file_open = h5py.File(f, "r")
    input_variables = {"jets": ["pt", "eta", "R10TruthLabel_R22v1", "GhostCHadronsFinalCount"]}
    all_file_vars = list(file_open["jets"].dtype.fields.keys())
    labels = {"jets": ["flavour_label"]}
    cn = list(Flavours.by_category("xbb").labels.keys())
    labellerConf = LabellerConfig(use_labeller=True, class_names=cn, require_labels=False)
    dataset = SaltDataset(
        f, norm_dict, input_variables, "train", labels=labels, labeller_config=labellerConf
    )
    all_unique_cuts = list(
        set(sum((label.cuts.variables for label in dataset.labeller.labels), []))
    )
    assert not (all(x in input_variables["jets"] for x in all_unique_cuts))
    assert all(x in all_file_vars for x in all_unique_cuts)


def test_process_labels_size_check():
    f = get_mock_file()[0]
    norm_dict = {}
    variables = {
        "jets": [
            "pt",
            "eta",
            "R10TruthLabel_R22v1",
            "GhostBHadronsFinalCount",
            "GhostCHadronsFinalCount",
        ]
    }
    labels = {"jets": ["flavour_label"]}
    class_names = list(Flavours.by_category("xbb").labels.keys())
    labellerConf_true = LabellerConfig(
        use_labeller=True, class_names=class_names, require_labels=True
    )
    ds_labeller = SaltDataset(
        f, norm_dict, variables, "train", labels=labels, labeller_config=labellerConf_true
    )
    labellerConf_false = LabellerConfig(use_labeller=False)
    ds_batch = SaltDataset(
        f, norm_dict, variables, "train", labels=labels, labeller_config=labellerConf_false
    )
    input_map = {k: k for k in variables}
    for i in range(0, len(ds_labeller), 10):
        ds_labeller[i : i + 10]
        ds_batch[i : i + 10]
        labels_otf = ds_labeller[i : i + 10][2]
        labels_batch = ds_batch[i : i + 10][2]
        for input_name in input_map:
            batch_otf = ds_labeller.arrays[input_name]
            labels_otf[input_name] = {}
            on_the_fly_output = ds_labeller.process_labels(labels_otf, batch_otf, input_name)
            batch_batch = ds_batch.arrays[input_name]
            labels_batch[input_name] = {}
            batch_output = ds_batch.process_labels(labels_batch, batch_batch, input_name)
            assert batch_output["flavour_label"].size() == on_the_fly_output["flavour_label"].size()


def test_process_labels_some_unlabelled():
    f = get_mock_file()[0]
    norm_dict = {}
    variables = {
        "jets": [
            "pt",
            "eta",
            "R10TruthLabel_R22v1",
            "GhostBHadronsFinalCount",
            "GhostCHadronsFinalCount",
        ]
    }
    labels = {"jets": ["flavour_label"]}
    class_names = ["hbb", "top", "qcdbb", "qcdbx", "qcdll"]
    labellerConf_true = LabellerConfig(
        use_labeller=True, class_names=class_names, require_labels=True
    )
    ds_labeller_requireTrue = SaltDataset(
        f, norm_dict, variables, "train", labels=labels, labeller_config=labellerConf_true
    )
    labellerConf_false = LabellerConfig(use_labeller=False)
    ds_labeller_requireFalse = SaltDataset(
        f, norm_dict, variables, "train", labels=labels, labeller_config=labellerConf_false
    )
    ds_labeller_requireFalse[0:10]
    with pytest.raises(ValueError, match="Some objects were not labelled"):
        ds_labeller_requireTrue[0:10]


def test_malformed_check_pass():
    f = get_mock_file()[0]
    norm_dict = {}
    variables = {"jets": ["pt", "eta"], "tracks": ["d0"]}
    labels = {"tracks": ["ftagTruthOriginLabel"]}
    ds = SaltDataset(f, norm_dict, variables, "train", labels=labels, recover_malformed=True)
    malformed_truthOrigin = np.array([-457384, -3, -2, -1, 0, 1, 5, 7, 8, 7892])
    recovered_truthOrigin = malformed_truthorigin_check(ds, malformed_truthOrigin)
    expected = np.array([-1, -1, -1, -1, 0, 1, 5, 7, -1, -1])
    assert np.array_equal(recovered_truthOrigin, expected)
    counts = np.unique(recovered_truthOrigin, return_counts=True)
    expected_counts = (np.array([-1, 0, 1, 5, 7]), np.array([6, 1, 1, 1, 1]))
    assert np.array_equal(counts, expected_counts)


def test_malformed_check_fail():
    f = get_mock_file()[0]
    norm_dict = {}
    variables = {"jets": ["pt", "eta"], "tracks": ["d0"]}
    labels = {"tracks": ["ftagTruthOriginLabel"]}
    ds = SaltDataset(f, norm_dict, variables, "train", labels=labels, recover_malformed=False)
    malformed_truthOrigin = np.array([-457384, -3, -2, -1, 0, 1, 5, 7, 8, 7892])
    with pytest.raises(ValueError, match="Recover flag is off, failing"):
        malformed_truthorigin_check(ds, malformed_truthOrigin)
