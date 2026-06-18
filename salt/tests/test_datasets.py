from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp

import h5py
import numpy as np
import pytest
import torch
from ftag import Flavours, get_mock_file

from salt.data import SaltDataset
from salt.data.datasets import _select_objects, malformed_truthorigin_check
from salt.utils.configs import (
    LabellerConfig,
    MaskformerConfig,
    MaskformerObjectConfig,
    ObjectCut,
)
from salt.utils.mask_utils import build_target_masks


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
            assert all(isinstance(v, torch.Tensor) for v in output.values())


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
    # truth_hadrons group (objects) — class_label = flavour, lxy = Lxy
    obj_dtype = np.dtype([
        ("barcode", "i4"),
        ("flavour", "i4"),
        ("Lxy", "f4"),
    ])
    obj_data = np.zeros((n_jets, n_obj), dtype=obj_dtype)
    # raw flavour labels: 5=b (→0), 4=c (→1), -1=null (→2)
    # slot layout: 0=b near, 1=b far, 2=b far, 3=null NaN, 4=c near
    raw_labels = np.array([5, 5, 5, -1, 4], dtype="i4")
    obj_data["flavour"] = raw_labels[np.newaxis, :]  # same for all jets
    # Lxy values: 0=near(50), 1=just outside(250), 2=far(1000), 3=NaN(null slot), 4=near(100)
    lxy_values = np.array([50.0, 250.0, 1000.0, np.nan, 100.0], dtype="f4")
    obj_data["Lxy"] = lxy_values[np.newaxis, :]
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
        "objects": ["barcode", "flavour", "Lxy"],
        "tracks": ["d0"],
    }
    lbl = {"objects": ["barcode", "flavour"], "tracks": ["ftagTruthParentBarcode"]}
    mf_config = MaskformerConfig(
        object=MaskformerObjectConfig(
            name="truth_hadrons",
            id_label="barcode",
            class_label="flavour",
            object_classes={
                "b": {"raw": 5, "mapped": 0},
                "c": {"raw": 4, "mapped": 1},
                "null": {"raw": -1, "mapped": 2},
            },
            max_lxy_mm=200.0,
            lxy_field="Lxy",
        ),
        constituent=MaskformerObjectConfig(name="tracks", id_label="ftagTruthParentBarcode"),
    )
    # ignore_finite_checks=True: NaN Lxy on null slots is expected in real data
    ds = SaltDataset(
        f,
        norm_dict,
        variables,
        "train",
        labels=lbl,
        mf_config=mf_config,
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
        "objects": ["barcode", "flavour", "Lxy"],
        "tracks": ["d0"],
    }
    lbl = {"objects": ["barcode", "flavour"], "tracks": ["ftagTruthParentBarcode"]}
    mf_config = MaskformerConfig(
        object=MaskformerObjectConfig(
            name="truth_hadrons",
            id_label="barcode",
            class_label="flavour",
            object_classes={
                "b": {"raw": 5, "mapped": 0},
                "c": {"raw": 4, "mapped": 1},
                "null": {"raw": -1, "mapped": 2},
            },
            max_lxy_mm=None,  # disabled
            lxy_field="Lxy",
        ),
        constituent=MaskformerObjectConfig(name="tracks", id_label="ftagTruthParentBarcode"),
    )
    ds = SaltDataset(
        f,
        norm_dict,
        variables,
        "train",
        labels=lbl,
        mf_config=mf_config,
        ignore_finite_checks=True,
    )
    null_idx = mf_config.object.null_index  # = 2
    batch_labels = ds[0:10][2]
    oc = batch_labels["objects"]["object_class"]
    # slot 1 Lxy=250 should NOT be null (no cut applied)
    assert (oc[:, 1] != null_idx).all(), "With max_lxy_mm=None, slot 1 (Lxy=250) should NOT be null"


# ---------------------------------------------------------------------------
# Helpers and tests for the generic per-jet object selection
# (cuts / sort / pv-pin / max_objects truncation)
# ---------------------------------------------------------------------------


# Class layout used by the generic selection tests:
#   pv:   raw=0,  mapped=0  (PV — pinned at slot 0, exempt from cuts/sort)
#   b:    raw=5,  mapped=1
#   null: raw=-1, mapped=2
#
# Slot layout (same for all jets, barcode == slot index):
#   0: pv,    pt=10,  Lxy=5,    valid=1  → PV (always kept at output slot 0)
#   1: b,     pt=80,  Lxy=50,   valid=1  → near, medium-high pt
#   2: b,     pt=60,  Lxy=80,   valid=1  → near, medium pt
#   3: b,     pt=40,  Lxy=300,  valid=1  → FAR (excluded by Lxy<=200 cut)
#   4: b,     pt=90,  Lxy=30,   valid=1  → near, highest pt of non-PV
#   5: null,  pt=0,   Lxy=0,    valid=0  → padding slot (explicitly invalid)


def _make_mf_h5_selection(tmp_path, n_jets=10, n_obj=6, *, nan_in_lxy=False):
    """HDF5 fixture with a 'valid' field and deterministic per-slot values."""
    f = tmp_path / "mf_sel_test.h5"
    obj_dtype = np.dtype([
        ("barcode", "i4"),
        ("flavour", "i4"),
        ("pt", "f4"),
        ("Lxy", "f4"),
        ("valid", "u1"),
    ])
    flavours = np.array([0, 5, 5, 5, 5, -1], dtype="i4")  # 0=pv, 5=b, -1=null
    pts = np.array([10.0, 80.0, 60.0, 40.0, 90.0, 0.0], dtype="f4")
    lxys = np.array([5.0, 50.0, 80.0, 300.0, 30.0, 0.0], dtype="f4")
    valids = np.array([1, 1, 1, 1, 1, 0], dtype="u1")
    if nan_in_lxy:
        # Inject NaN Lxy on slot 2 so a Lxy cut must drop it (strict NaN-fails).
        lxys = lxys.copy()
        lxys[2] = np.nan

    obj_data = np.zeros((n_jets, n_obj), dtype=obj_dtype)
    for slot in range(n_obj):
        obj_data[:, slot]["barcode"] = slot  # barcode == slot index
        obj_data[:, slot]["flavour"] = flavours[slot]
        obj_data[:, slot]["pt"] = pts[slot]
        obj_data[:, slot]["Lxy"] = lxys[slot]
        obj_data[:, slot]["valid"] = valids[slot]
    jets_dtype = np.dtype([("pt", "f4"), ("eta", "f4")])
    jets_data = np.zeros(n_jets, dtype=jets_dtype)
    trk_dtype = np.dtype([("ftagTruthParentBarcode", "i4"), ("d0", "f4")])
    trk_data = np.zeros((n_jets, 10), dtype=trk_dtype)
    with h5py.File(f, "w") as hf:
        hf.create_dataset("jets", data=jets_data)
        hf.create_dataset("truth_hadrons", data=obj_data)
        hf.create_dataset("tracks", data=trk_data)
    return str(f)


def _mf_config_selection(**kwargs):
    """Return a MaskformerConfig with PV/b/null classes for selection tests."""
    return MaskformerConfig(
        object=MaskformerObjectConfig(
            name="truth_hadrons",
            id_label="barcode",
            class_label="flavour",
            object_classes={
                "pv": {"raw": 0, "mapped": 0},
                "b": {"raw": 5, "mapped": 1},
                "null": {"raw": -1, "mapped": 2},
            },
            **kwargs,
        ),
        constituent=MaskformerObjectConfig(name="tracks", id_label="ftagTruthParentBarcode"),
    )


_VARS_SEL = {
    "jets": ["pt", "eta"],
    "objects": ["barcode", "flavour", "Lxy", "pt"],
    "tracks": ["d0"],
}
_LBL_SEL = {"objects": ["barcode", "flavour"], "tracks": ["ftagTruthParentBarcode"]}


# A. Cuts drop, PV preserved
def test_object_selection_cuts_drop_with_pv_preserved(tmp_path):
    """Lxy<=200 cut: PV (Lxy=5) survives, slot 3 (Lxy=300) is dropped."""
    f = _make_mf_h5_selection(tmp_path)
    ds = SaltDataset(
        f,
        {},
        _VARS_SEL,
        "train",
        labels=_LBL_SEL,
        mf_config=_mf_config_selection(
            cuts=[ObjectCut(field="Lxy", max=200.0)],
        ),
        ignore_finite_checks=True,
    )
    _, _, labels = ds[0:5]
    barcodes = labels["objects"]["barcode"]  # (B, 6) — no truncation
    # PV (barcode=0) must remain at slot 0
    assert (barcodes[:, 0] == 0).all(), "PV (barcode=0) must remain at slot 0 after cuts"
    # Slot 3 (Lxy=300) must not appear anywhere in the output
    assert (barcodes != 3).all(), "Far vertex (barcode=3, Lxy=300) must be dropped by Lxy<=200 cut"


# B. Sort by pt desc, PV pinned
def test_object_selection_sort_by_pt_pv_pinned(tmp_path):
    """sort_by=pt desc: output is [PV, slot4(pt=90), slot1(pt=80), slot2(pt=60), slot3(pt=40), pad]."""
    f = _make_mf_h5_selection(tmp_path)
    ds = SaltDataset(
        f,
        {},
        _VARS_SEL,
        "train",
        labels=_LBL_SEL,
        mf_config=_mf_config_selection(sort_by="pt"),
        ignore_finite_checks=True,
    )
    _, _, labels = ds[0:5]
    barcodes = labels["objects"]["barcode"].numpy()
    # PV at slot 0
    assert (barcodes[:, 0] == 0).all(), "PV must be at slot 0"
    # Then: slot 4 (pt=90), 1 (pt=80), 2 (pt=60), 3 (pt=40). Slot 5 is invalid → pad.
    expected_active = [0, 4, 1, 2, 3]
    for b in range(barcodes.shape[0]):
        assert list(barcodes[b, : len(expected_active)]) == expected_active, (
            f"Unexpected order {list(barcodes[b])}, expected prefix {expected_active}"
        )


# C. Truncation
def test_object_selection_max_objects_truncates(tmp_path):
    """max_objects=3 → out shape (B, 3); PV at slot 0."""
    f = _make_mf_h5_selection(tmp_path)
    ds = SaltDataset(
        f,
        {},
        _VARS_SEL,
        "train",
        labels=_LBL_SEL,
        mf_config=_mf_config_selection(max_objects=3, sort_by="pt"),
        ignore_finite_checks=True,
    )
    inputs, _, labels = ds[0:5]
    assert inputs["objects"].shape[1] == 3
    barcodes = labels["objects"]["barcode"].numpy()
    # [PV(0), slot4(pt=90), slot1(pt=80)]
    expected = [0, 4, 1]
    for b in range(barcodes.shape[0]):
        assert list(barcodes[b]) == expected, (
            f"Unexpected truncated order {list(barcodes[b])}, expected {expected}"
        )


# D. Combined cuts + sort + truncate
def test_object_selection_combined_cuts_sort_truncate(tmp_path):
    """Lxy<=200 cut + sort_by=pt desc + max_objects=4 → leading 4 survivors."""
    f = _make_mf_h5_selection(tmp_path)
    ds = SaltDataset(
        f,
        {},
        _VARS_SEL,
        "train",
        labels=_LBL_SEL,
        mf_config=_mf_config_selection(
            cuts=[ObjectCut(field="Lxy", max=200.0)],
            sort_by="pt",
            max_objects=4,
        ),
        ignore_finite_checks=True,
    )
    inputs, _, labels = ds[0:5]
    assert inputs["objects"].shape[1] == 4
    barcodes = labels["objects"]["barcode"].numpy()
    # Survivors after Lxy<=200: PV(0), slot1(pt=80,Lxy=50), slot2(pt=60,Lxy=80), slot4(pt=90,Lxy=30).
    # Slot 3 dropped (Lxy=300). Slot 5 invalid. Sort non-PV by pt desc:
    # → [PV, slot4(90), slot1(80), slot2(60)]
    expected = [0, 4, 1, 2]
    for b in range(barcodes.shape[0]):
        assert list(barcodes[b]) == expected, (
            f"Combined order {list(barcodes[b])} != expected {expected}"
        )


# E. Pad hygiene
def test_object_selection_pad_hygiene(tmp_path):
    """Pad slots: valid=False, id_label=-1, class_label=null_raw."""
    f = _make_mf_h5_selection(tmp_path)
    ds = SaltDataset(
        f,
        {},
        _VARS_SEL,
        "train",
        labels=_LBL_SEL,
        mf_config=_mf_config_selection(
            cuts=[ObjectCut(field="Lxy", max=200.0)],
            sort_by="pt",
            max_objects=6,  # allow pads to exist
        ),
        ignore_finite_checks=True,
    )
    # Reach into the dataset to validate the structured array directly
    # (so we can check 'valid', class_label, and id_label sentinels).
    ds._setup()
    obj_dtype = ds.arrays["objects"].dtype
    n_jets = 5
    batch = np.zeros((n_jets, ds.dss["objects"].shape[1]), dtype=obj_dtype)
    ds.dss["objects"].read_direct(batch, np.s_[0:n_jets])
    out = _select_objects(batch, ds.mf_config.object)
    # 4 survivors per jet (PV + 3 b-slots passing Lxy cut), pad slots 4–5.
    n_survivors = 4
    pad_valid = out["valid"][:, n_survivors:]
    pad_barcode = out["barcode"][:, n_survivors:]
    pad_flavour = out["flavour"][:, n_survivors:]
    assert (pad_valid == 0).all(), "Pad slots must have valid=0/False"
    assert (pad_barcode == -1).all(), "Pad slots must have id_label=-1 sentinel"
    null_raw = ds.mf_config.object.null_raw_value
    assert (pad_flavour == null_raw).all(), (
        f"Pad class_label must be null_raw ({null_raw}), got {pad_flavour}"
    )


# F. No-PV jet
def test_object_selection_no_pv_jet(tmp_path):
    """If no slot matches pv_class, slot 0 is the highest-sort_by survivor."""
    f = _make_mf_h5_selection(tmp_path)
    # Use pv_class=None so PV is NOT pinned. Slot 0 is now just another vertex
    # but its raw flavour=0 is no longer in the class_map → it would map to
    # null_index. Use a config that never references the PV class instead:
    # set pv_class=None and use a class_map without 'pv' would change the
    # null index. Simpler: keep the pv class in the map so the dtype remains
    # the same, but disable pv pinning by setting pv_class=None.
    cfg = MaskformerConfig(
        object=MaskformerObjectConfig(
            name="truth_hadrons",
            id_label="barcode",
            class_label="flavour",
            object_classes={
                "pv": {"raw": 0, "mapped": 0},
                "b": {"raw": 5, "mapped": 1},
                "null": {"raw": -1, "mapped": 2},
            },
            sort_by="pt",
            pv_class=None,  # disable PV pinning
        ),
        constituent=MaskformerObjectConfig(name="tracks", id_label="ftagTruthParentBarcode"),
    )
    ds = SaltDataset(
        f,
        {},
        _VARS_SEL,
        "train",
        labels=_LBL_SEL,
        mf_config=cfg,
        ignore_finite_checks=True,
    )
    _, _, labels = ds[0:5]
    barcodes = labels["objects"]["barcode"].numpy()
    # With no PV pinning and sort_by=pt desc, slot 0 should be the highest pt
    # valid vertex: slot 4 (pt=90).
    assert (barcodes[:, 0] == 4).all(), (
        f"With pv_class=None, slot 0 should be top-pt (barcode=4), got {barcodes[:, 0]}"
    )


# G. Missing field error
def test_object_selection_missing_field_raises(tmp_path):
    """A cut on a field not in the loaded dtype must raise KeyError."""
    f = _make_mf_h5_selection(tmp_path)
    ds = SaltDataset(
        f,
        {},
        _VARS_SEL,
        "train",
        labels=_LBL_SEL,
        mf_config=_mf_config_selection(
            cuts=[ObjectCut(field="not_a_real_field", min=0.0)],
        ),
        ignore_finite_checks=True,
    )
    with pytest.raises(KeyError, match="not_a_real_field"):
        ds[0:5]


# H. Backward compat: no new fields → identical behaviour to legacy path
def test_object_selection_backward_compat_no_op(tmp_path):
    """cuts=None, sort_by=None, max_objects=None → batch identical to direct read."""
    f = _make_mf_h5_selection(tmp_path)
    # Config without any new selection knob — just the legacy max_lxy_mm=None.
    ds = SaltDataset(
        f,
        {},
        _VARS_SEL,
        "train",
        labels=_LBL_SEL,
        mf_config=_mf_config_selection(),  # all defaults except pv_class=0
        ignore_finite_checks=True,
    )
    # _needs_object_selection should be False → no _select_objects call
    assert not ds._needs_object_selection(), (
        "Legacy config (no cuts/sort_by/max_objects) must skip _select_objects"
    )
    _, _, labels = ds[0:5]
    barcodes = labels["objects"]["barcode"].numpy()
    # Original slot order preserved.
    expected = [0, 1, 2, 3, 4, 5]
    for b in range(barcodes.shape[0]):
        assert list(barcodes[b]) == expected, (
            f"Backward-compat order changed: {list(barcodes[b])} != {expected}"
        )


# I. build_target_masks invariant after permutation
def test_object_selection_build_target_masks_invariant(tmp_path):
    """After sort+truncate, build_target_masks(out_ids, track_ids) reflects the new order.

    This is the load-bearing test that protects the design assumption that no
    separate per-vertex mask permutation is required: build_target_masks rebuilds
    the truth mask from id_label after permutation, so as long as id_label is
    permuted (and pad slots are -1), the masks line up automatically.
    """
    f = _make_mf_h5_selection(tmp_path)
    cfg = _mf_config_selection(sort_by="pt", max_objects=3)
    ds = SaltDataset(
        f, {}, _VARS_SEL, "train", labels=_LBL_SEL, mf_config=cfg, ignore_finite_checks=True
    )
    ds._setup()

    # Read raw batch directly
    n_jets = 5
    obj_dtype = ds.arrays["objects"].dtype
    raw = np.zeros((n_jets, ds.dss["objects"].shape[1]), dtype=obj_dtype)
    ds.dss["objects"].read_direct(raw, np.s_[0:n_jets])

    # Synthesise per-track parent barcodes that point at non-PV vertices 1, 2, 4.
    # We don't care about exact track-vertex association beyond consistency.
    n_trk = 12
    parent_barcodes = np.zeros((n_jets, n_trk), dtype="i4")
    # 4 tracks each pointing at vertices 1, 2, 4
    parent_barcodes[:, 0:4] = 1
    parent_barcodes[:, 4:8] = 2
    parent_barcodes[:, 8:12] = 4

    out = _select_objects(raw, cfg.object)
    out_ids = torch.as_tensor(out["barcode"].astype("int64"))  # (B, 3)
    parent_ids = torch.as_tensor(parent_barcodes.astype("int64"))  # (B, 12)

    # Expected output barcodes after sort=pt-desc + max_objects=3: [PV(0), 4, 1]
    # → masks: row0=PV (no track matches), row1=barcode=4 (tracks 8–11),
    #          row2=barcode=1 (tracks 0–3). Vertex 2 (4 tracks) is *not in
    #          the truncated output*, so its tracks have no matching row.
    # Use a copy of out_ids because build_target_masks mutates -1 → -999 in place.
    masks = build_target_masks(out_ids.clone(), parent_ids)
    assert masks.shape == (n_jets, 3, n_trk)
    # PV row: no track has parent barcode 0 → all False
    assert not masks[:, 0, :].any(), "PV row must have no track matches"
    # Slot 1: barcode 4 → tracks 8–11 only
    assert masks[:, 1, 8:12].all(), "Slot 1 (barcode=4) must match its tracks"
    assert not masks[:, 1, :8].any(), "Slot 1 must not match other tracks"
    # Slot 2: barcode 1 → tracks 0–3 only
    assert masks[:, 2, 0:4].all(), "Slot 2 (barcode=1) must match its tracks"
    assert not masks[:, 2, 4:].any(), "Slot 2 must not match other tracks"


# J. NaN in cut field fails (vertex dropped)
def test_object_selection_nan_fails_cut(tmp_path):
    """NaN in a cut field must drop the vertex (strict NaN-fails semantics)."""
    f = _make_mf_h5_selection(tmp_path, nan_in_lxy=True)  # slot 2 Lxy=NaN
    ds = SaltDataset(
        f,
        {},
        _VARS_SEL,
        "train",
        labels=_LBL_SEL,
        mf_config=_mf_config_selection(
            cuts=[ObjectCut(field="Lxy", min=-1e9, max=1e9)],  # finite → NaN fails
        ),
        ignore_finite_checks=True,
    )
    _, _, labels = ds[0:5]
    barcodes = labels["objects"]["barcode"].numpy()
    # Slot 2 (NaN Lxy) must not appear anywhere — strict NaN fail.
    assert (barcodes != 2).all(), (
        f"NaN Lxy slot must be dropped, but barcode=2 appears in {barcodes.tolist()}"
    )
    # PV (barcode=0) still pinned at slot 0
    assert (barcodes[:, 0] == 0).all()


def test_object_cut_requires_min_or_max():
    """ObjectCut with no min and no max must raise ValueError on construction."""
    with pytest.raises(ValueError, match="must set at least one of min/max"):
        ObjectCut(field="pt")


def test_object_cut_max_objects_legacy_alias():
    """num_objects (legacy) is bridged to max_objects in __post_init__."""
    cfg = MaskformerObjectConfig(
        name="truth_hadrons",
        id_label="barcode",
        class_label="flavour",
        object_classes={
            "pv": {"raw": 0, "mapped": 0},
            "b": {"raw": 5, "mapped": 1},
            "null": {"raw": -1, "mapped": 2},
        },
        num_objects=7,
    )
    assert cfg.max_objects == 7
    assert cfg.num_objects == 7


def test_object_pv_class_validation():
    """pv_class out of range must raise ValueError."""
    with pytest.raises(ValueError, match="pv_class"):
        MaskformerObjectConfig(
            name="truth_hadrons",
            id_label="barcode",
            class_label="flavour",
            object_classes={
                "pv": {"raw": 0, "mapped": 0},
                "b": {"raw": 5, "mapped": 1},
                "null": {"raw": -1, "mapped": 2},
            },
            pv_class=99,
        )


# G. pad_masks must NOT include "objects" even when data.variables.objects is set
# (regression test for exp 22 attention shape mismatch crash)
def test_objects_excluded_from_pad_masks(tmp_path):
    """When ``data.variables.objects`` is non-empty (required for cuts/sort), the
    standard input-loader path runs for ``input_name == "objects"`` because
    ``self.input_variables.get("objects")`` is truthy. Before the fix, this
    populated ``pad_masks["objects"]`` (because the truth-vertices array carries
    a ``valid`` field), and the encoder later concatenated it into its
    sequence-axis mask alongside tracks/flows. Since ``"objects"`` has no init
    network, the encoder's ``x`` (tracks+flows+REGISTERS) and its concatenated
    ``mask`` (tracks+flows+objects+REGISTERS) had different lengths,
    crashing ``scaled_dot_product_attention`` with a dim-3 mismatch.

    Objects metadata is consumed by ``_select_objects`` /
    ``build_target_masks``, never by the encoder, so it must be excluded from
    ``pad_masks`` (alongside ``EDGE``, ``parameters``, ``global``, and the
    global object).
    """
    f = _make_mf_h5_selection(tmp_path)
    ds = SaltDataset(
        f,
        {},
        _VARS_SEL,  # includes "objects": ["barcode", "flavour", "Lxy", "pt"]
        "train",
        labels=_LBL_SEL,
        mf_config=_mf_config_selection(
            cuts=[ObjectCut(field="Lxy", max=200.0)],
            sort_by="pt",
            max_objects=4,
        ),
        ignore_finite_checks=True,
    )
    inputs, pad_masks, _ = ds[0:5]
    # Objects should still load as inputs (downstream code may inspect dtype etc.)
    assert "objects" in inputs
    # But pad_masks must NOT carry "objects" — that would leak into the encoder
    # mask concatenation and break attention shapes.
    assert "objects" not in pad_masks, (
        "pad_masks must NOT include 'objects'; it would mismatch the encoder "
        "input shape and crash attention. Excluded set in datasets.py must "
        "include 'objects'."
    )
    # Sanity: tracks (which have no 'valid' field in this fixture) carry no
    # pad_mask either — we just need to confirm 'objects' is excluded.
    assert "objects" not in pad_masks
