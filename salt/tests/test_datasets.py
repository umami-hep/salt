import h5py
import numpy as np
import pytest
import torch
from ftag import get_mock_file, Flavours
from tempfile import NamedTemporaryFile, mkdtemp
from pathlib import Path

from salt.data import SaltDataset
from salt.data.datasets import malformed_truthorigin_check
from salt.utils.configs import LabellerConfig, MaskformerConfig, MaskformerObjectConfig


def test_salt_dataset():
    f = get_mock_file()[0]
    norm_dict = {}
    variables = {"jets": ["pt", "eta"], "tracks": ["d0"]}
    dataset = SaltDataset(f, norm_dict, variables, "train")
    for i in range(0, len(dataset), 10):
        dataset[i : i + 10]


def test_salt_dataset_wildcard():
    # Get a temp directory
    temp_dir = mkdtemp()

    # Create two mock files in the same folder
    f = get_mock_file(
        fname=NamedTemporaryFile(
            prefix="train_1_",
            suffix=".h5",
            dir=temp_dir,
        ).name
    )[0]
    g = get_mock_file(
        fname=NamedTemporaryFile(
            prefix="train_2_",
            suffix=".h5",
            dir=temp_dir,
        ).name
    )[0]

    # Define a wildcard
    h = str(Path(temp_dir) / "train_*.h5")

    # Define default norm and variable dicts
    norm_dict = {}
    variables = {"jets": ["pt", "eta"], "tracks": ["d0"]}
    dataset = SaltDataset(h, norm_dict, variables, "train")
    for i in range(0, len(dataset), 10):
        dataset[i : i + 10]


def test_salt_dataset_no_matching_wildcard():
    # Get a temp directory
    temp_dir = mkdtemp()

    # Define a wildcard
    h = str(Path(temp_dir) / "val*.h5")

    # Define default norm and variable dicts
    norm_dict = {}
    variables = {"jets": ["pt", "eta"], "tracks": ["d0"]}

    with pytest.raises(FileNotFoundError, match="No files match wildcard:"):
        SaltDataset(h, norm_dict, variables, "train")


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


def test_alt_target():
    f = get_mock_file()[0]
    norm_dict = {}
    variables = {"jets": ["pt", "pt_btagJes", "HadronConeExclTruthLabelID"]}
    labels = {"jets": ["HadronConeExclTruthLabelID", "pt", "pt_btagJes"]}
    multi_target = [
        {
            "input_name": "jets",
            "sel_label": "HadronConeExclTruthLabelID",
            "opp": "==",
            "value": 15,
            "target": "pt",
            "source": "pt_btagJes",
        }
    ]
    ori = SaltDataset(f, norm_dict, variables, "train", labels=labels)
    replace = SaltDataset(
        f, norm_dict, variables, "train", labels=labels, multi_target=multi_target
    )
    _, _, labels_ori = ori[0 : len(ori)]
    _, _, labels_replace = replace[0 : len(replace)]
    mask = labels_ori["jets"]["pt"] != labels_replace["jets"]["pt"]
    assert torch.all(labels_ori["jets"]["pt_btagJes"][mask] == labels_replace["jets"]["pt"][mask])
    assert torch.all(labels_ori["jets"]["HadronConeExclTruthLabelID"][mask] == 15)
    # source label should still be available for other tasks
    assert "pt_btagJes" in labels_replace["jets"]


def test_new_target():
    f = get_mock_file()[0]
    norm_dict = {}
    variables = {"jets": ["pt", "pt_btagJes", "HadronConeExclTruthLabelID"]}
    labels = {"jets": ["HadronConeExclTruthLabelID", "pt_label_handle", "pt_btagJes"]}
    multi_target = [
        {
            "input_name": "jets",
            "sel_label": "HadronConeExclTruthLabelID",
            "opp": "==",
            "value": 15,
            "custom_target": "pt_label_handle",
            "source": "pt_btagJes",
        }
    ]
    ori = SaltDataset(f, norm_dict, variables, "train", labels=labels, multi_target=multi_target)
    _, _, labels = ori[0 : len(ori)]
    mask = labels["jets"]["pt_label_handle"].isnan()
    assert torch.all(labels["jets"]["HadronConeExclTruthLabelID"][~mask] == 15)


def test_invalid_operator():
    f = get_mock_file()[0]
    norm_dict = {}
    variables = {"jets": ["pt", "pt_btagJes", "HadronConeExclTruthLabelID"]}
    labels = {"jets": ["HadronConeExclTruthLabelID", "pt", "pt_btagJes"]}
    multi_target = [
        {
            "input_name": "jets",
            "sel_label": "HadronConeExclTruthLabelID",
            "opp": "invalid_op",
            "value": 15,
            "target": "pt",
            "source": "pt_btagJes",
        }
    ]
    dataset = SaltDataset(
        f, norm_dict, variables, "train", labels=labels, multi_target=multi_target
    )
    with pytest.raises(KeyError, match="Invalid operator: invalid_op"):
        dataset[0:10]


def test_missing_target():
    f = get_mock_file()[0]
    norm_dict = {}
    variables = {"jets": ["pt", "pt_btagJes", "HadronConeExclTruthLabelID"]}
    labels = {"jets": ["HadronConeExclTruthLabelID", "pt_label_handle", "pt_btagJes"]}
    multi_target = [
        {
            "input_name": "jets",
            "sel_label": "HadronConeExclTruthLabelID",
            "opp": "==",
            "value": 15,
            "custom_target": "pt_label_handle_wrong",
            "source": "pt_btagJes",
        }
    ]
    dataset = SaltDataset(
        f, norm_dict, variables, "train", labels=labels, multi_target=multi_target
    )
    with pytest.raises(ValueError, match="no field of name pt_label_handle"):
        dataset[0:10]


def test_missing_source():
    f = get_mock_file()[0]
    norm_dict = {}
    variables = {"jets": ["pt", "HadronConeExclTruthLabelID"]}
    labels = {"jets": ["HadronConeExclTruthLabelID", "pt_label_handle"]}
    multi_target = [
        {
            "input_name": "jets",
            "sel_label": "HadronConeExclTruthLabelID",
            "opp": "==",
            "value": 15,
            "custom_target": "pt_label_handle",
            "source": "boo",
        }
    ]
    dataset = SaltDataset(
        f, norm_dict, variables, "train", labels=labels, multi_target=multi_target
    )
    with pytest.raises(
        KeyError, match="Source field 'boo' not found in batch for custom_target 'pt_label_handle'"
    ):
        dataset[0:10]


def test_target_and_custom_target_both_defined():
    f = get_mock_file()[0]
    norm_dict = {}
    variables = {"jets": ["pt", "HadronConeExclTruthLabelID"]}
    labels = {"jets": ["HadronConeExclTruthLabelID", "pt"]}
    multi_target = [
        {
            "input_name": "jets",
            "sel_label": "HadronConeExclTruthLabelID",
            "opp": "==",
            "value": 15,
            "target": "pt",
            "custom_target": "pt_label_handle",
            "source": "pt_btagJes",
        }
    ]
    with pytest.raises(ValueError, match="cannot define both 'target' and 'custom_target'"):
        SaltDataset(f, norm_dict, variables, "train", labels=labels, multi_target=multi_target)


def test_target_and_custom_target_neither_defined():
    f = get_mock_file()[0]
    norm_dict = {}
    variables = {"jets": ["pt", "HadronConeExclTruthLabelID"]}
    labels = {"jets": ["HadronConeExclTruthLabelID", "pt"]}
    multi_target = [
        {
            "input_name": "jets",
            "sel_label": "HadronConeExclTruthLabelID",
            "opp": "==",
            "value": 15,
            "source": "pt_btagJes",
        }
    ]
    with pytest.raises(ValueError, match="must define either 'target' or 'custom_target'"):
        SaltDataset(f, norm_dict, variables, "train", labels=labels, multi_target=multi_target)


def _make_mf_h5(tmp_path, n_jets=20, n_obj=5):
    """Create a minimal HDF5 file with jets and truth_hadrons groups for MF tests."""
    f = tmp_path / "mf_test.h5"
    rng = np.random.default_rng(42)
    # jets group
    jets_dtype = np.dtype([("pt", "f4"), ("eta", "f4"), ("barcode", "i4")])
    jets_data = np.zeros(n_jets, dtype=jets_dtype)
    jets_data["pt"] = rng.uniform(20, 500, n_jets)
    # truth_hadrons group (objects) — class_label = flavour, lxy = ftagTPDecayPVDistance
    obj_dtype = np.dtype([
        ("barcode", "i4"),
        ("flavour", "i4"),
        ("ftagTPDecayPVDistance", "f4"),
    ])
    obj_data = np.zeros((n_jets, n_obj), dtype=obj_dtype)
    # raw flavour labels: 5=b (→0), 4=c (→1), -1=null (→2)
    # slot layout: 0=b near, 1=b far, 2=b far, 3=null NaN, 4=c near
    raw_labels = np.array([5, 5, 5, -1, 4], dtype="i4")
    obj_data["flavour"] = raw_labels[np.newaxis, :]  # same for all jets
    # Lxy values: 0=near(50), 1=just outside(250), 2=far(1000), 3=NaN(null slot), 4=near(100)
    lxy_values = np.array([50.0, 250.0, 1000.0, np.nan, 100.0], dtype="f4")
    obj_data["ftagTPDecayPVDistance"] = lxy_values[np.newaxis, :]
    obj_data["barcode"] = rng.integers(1, 9999, (n_jets, n_obj))
    # tracks group (needed by MaskformerConfig constituent)
    trk_dtype = np.dtype([("ftagTruthParentBarcode", "i4"), ("d0", "f4")])
    trk_data = np.zeros((n_jets, 20), dtype=trk_dtype)
    with h5py.File(f, "w") as hf:
        hf.create_dataset("jets", data=jets_data)
        hf.create_dataset("truth_hadrons", data=obj_data)
        hf.create_dataset("tracks", data=trk_data)
    return str(f)


def test_lxy_mask_relabels_far_vertices(tmp_path):
    """Vertices with |Lxy| > max_lxy_mm must be re-labelled to the null class index."""
    f = _make_mf_h5(tmp_path)
    norm_dict = {}
    variables = {
        "jets": ["pt", "eta"],
        "objects": ["barcode", "flavour", "ftagTPDecayPVDistance"],
        "tracks": ["d0"],
    }
    lbl = {"objects": ["barcode", "flavour"], "tracks": ["ftagTruthParentBarcode"]}
    mf_config = MaskformerConfig(
        object=MaskformerObjectConfig(
            name="truth_hadrons",
            id_label="barcode",
            class_label="flavour",
            object_classes={
                "b":    {"raw": 5,  "mapped": 0},
                "c":    {"raw": 4,  "mapped": 1},
                "null": {"raw": -1, "mapped": 2},
            },
            max_lxy_mm=200.0,
            lxy_field="ftagTPDecayPVDistance",
        ),
        constituent=MaskformerObjectConfig(name="tracks", id_label="ftagTruthParentBarcode"),
    )
    # ignore_finite_checks=True: NaN Lxy on null slots is expected in real data
    ds = SaltDataset(
        f, norm_dict, variables, "train", labels=lbl, mf_config=mf_config,
        ignore_finite_checks=True,
    )
    # null class index = 2 (len(object_classes) - 1)
    null_idx = mf_config.object.null_index  # = 2
    assert null_idx == 2

    # Check all jets in the dataset
    n_total = len(ds)
    for start in range(0, n_total, 10):
        batch_labels = ds[start : start + 10][2]
        oc = batch_labels["objects"]["object_class"]  # shape: (batch, n_obj)
        # slot 0: raw=b(5), Lxy=50 mm  → near → mapped to 0 (NOT null)
        assert (oc[:, 0] == 0).all(), "Slot 0 (b, Lxy=50) should be class b (0)"
        # slot 1: raw=b(5), Lxy=250 mm → far → re-labelled to null
        assert (oc[:, 1] == null_idx).all(), "Slot 1 (b, Lxy=250) should be null after mask"
        # slot 2: raw=b(5), Lxy=1000 mm → far → re-labelled to null
        assert (oc[:, 2] == null_idx).all(), "Slot 2 (b, Lxy=1000) should be null after mask"
        # slot 3: raw=null(-1), Lxy=NaN → NaN comparison is False, not flipped; stays null
        assert (oc[:, 3] == null_idx).all(), "Slot 3 (null, NaN Lxy) should stay null"
        # slot 4: raw=c(4), Lxy=100 mm → near → mapped to 1 (NOT null)
        assert (oc[:, 4] == 1).all(), "Slot 4 (c, Lxy=100) should be class c (1)"


def test_lxy_mask_disabled_when_none(tmp_path):
    """With max_lxy_mm=None (default), no Lxy cut is applied."""
    f = _make_mf_h5(tmp_path)
    norm_dict = {}
    variables = {
        "jets": ["pt", "eta"],
        "objects": ["barcode", "flavour", "ftagTPDecayPVDistance"],
        "tracks": ["d0"],
    }
    lbl = {"objects": ["barcode", "flavour"], "tracks": ["ftagTruthParentBarcode"]}
    mf_config = MaskformerConfig(
        object=MaskformerObjectConfig(
            name="truth_hadrons",
            id_label="barcode",
            class_label="flavour",
            object_classes={
                "b":    {"raw": 5,  "mapped": 0},
                "c":    {"raw": 4,  "mapped": 1},
                "null": {"raw": -1, "mapped": 2},
            },
            max_lxy_mm=None,  # disabled
            lxy_field="ftagTPDecayPVDistance",
        ),
        constituent=MaskformerObjectConfig(name="tracks", id_label="ftagTruthParentBarcode"),
    )
    ds = SaltDataset(
        f, norm_dict, variables, "train", labels=lbl, mf_config=mf_config,
        ignore_finite_checks=True,
    )
    null_idx = mf_config.object.null_index  # = 2
    batch_labels = ds[0:10][2]
    oc = batch_labels["objects"]["object_class"]
    # slot 1 Lxy=250 should NOT be null (no cut applied)
    assert (oc[:, 1] != null_idx).all(), "With max_lxy_mm=None, slot 1 (Lxy=250) should NOT be null"
