"""Tests for `salt.core.callbacks.ConfusionMatrix` (split from test_callbacks.py)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from salt.callbacks.confusion_matrix import ConfusionMatrixCallback as V1ConfusionMatrix
from salt.core.callbacks import ConfusionMatrix
from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.saltmodule import bundle_as_v1_outputs


ORIGIN_NAMES = tuple(f"c{i}" for i in range(8))


def make_bundle(seed: int, batch: int = 50, n_tracks: int = 10) -> Bundle:
    """One VAL-style step bundle: raw logits + labels for a jet and a track task."""
    gen = torch.Generator().manual_seed(seed)
    bundle = Bundle()
    bundle.set("preds.jets.jets_classification", torch.randn(batch, 3, generator=gen))
    bundle.set("labels.jets.flavour_label", torch.randint(0, 3, (batch,), generator=gen))
    bundle.set("preds.tracks.track_origin", torch.randn(batch, n_tracks, 8, generator=gen))
    bundle.set(
        "labels.tracks.ftagTruthOriginLabel",
        torch.randint(-1, 8, (batch, n_tracks), generator=gen),  # -1 = padding
    )
    return bundle


def stub_pl_module() -> SimpleNamespace:
    """A SaltModule-shaped stub: the duck-typed classification-task surface."""
    return SimpleNamespace(
        _graph_modules={
            "jets_classification": SimpleNamespace(
                stream="jets", label="flavour_label", class_names=("bjets", "cjets", "ujets")
            ),
            "track_origin": SimpleNamespace(
                stream="tracks", label="ftagTruthOriginLabel", class_names=ORIGIN_NAMES
            ),
            "encoder": SimpleNamespace(),  # not a task — must not be a candidate
        }
    )


def run_v2(task_name: str, bundles: list[Bundle], **kwargs) -> ConfusionMatrix:
    """Drive the v2 callback over `bundles` through the Lightning hook surface."""
    callback = ConfusionMatrix(task_name=task_name, **kwargs)
    pl_module = stub_pl_module()
    callback.setup(None, pl_module, stage="fit")
    for i, bundle in enumerate(bundles):
        callback.on_validation_batch_end(None, pl_module, {"bundle": bundle}, None, i)
    callback.on_validation_epoch_end(SimpleNamespace(logger=None, current_epoch=0), pl_module)
    return callback


def run_v1(task_name: str, stream: str, label: str, class_names, bundles) -> V1ConfusionMatrix:
    """Drive the v1 callback with shimmed outputs (`bundle_as_v1_outputs`)."""
    callback = V1ConfusionMatrix(task_name=task_name)
    callback.truth_labels = []
    callback.pred_labels = []
    callback.task_input_name = stream
    callback.task_label_name = label
    callback.task_class_names = list(class_names)
    for i, bundle in enumerate(bundles):
        callback.on_validation_batch_end(
            None, None, {"outputs": bundle_as_v1_outputs(bundle)}, None, i
        )
    return callback


class TestConfusionMatrixValues:
    """W5 in miniature: v2 values == v1 values on identical eval batches."""

    @pytest.mark.parametrize(
        ("task_name", "stream", "label", "class_names"),
        [
            ("jets_classification", "jets", "flavour_label", ("bjets", "cjets", "ujets")),
            ("track_origin", "tracks", "ftagTruthOriginLabel", ORIGIN_NAMES),
        ],
    )
    def test_matches_v1_callback(self, task_name, stream, label, class_names):
        bundles = [make_bundle(seed) for seed in (1, 2, 3)]
        v2 = run_v2(task_name, bundles)
        v1 = run_v1(task_name, stream, label, class_names, bundles)
        # identical accumulated values, element-wise (v1 keeps its lists —
        # it only resets at epoch end; v2 stashes them at epoch end)
        assert len(v2.last_truth_labels) == len(v1.truth_labels) > 0
        for ours, theirs in zip(v2.last_truth_labels, v1.truth_labels, strict=True):
            assert torch.equal(ours, theirs)
        for ours, theirs in zip(v2.last_pred_labels, v1.pred_labels, strict=True):
            assert torch.equal(ours, theirs)
        # identical matrix under the same transparent reduction
        v1_matrix, v1_ignored = ConfusionMatrix.confusion_counts(
            v1.truth_labels, v1.pred_labels, len(class_names)
        )
        assert torch.equal(v2.last_matrix, v1_matrix)
        assert v2.last_ignored == v1_ignored

    def test_counts_matrix_hand_example(self):
        truth = [torch.tensor([0, 1, 2, -1, 1])]
        preds = [torch.tensor([0, 2, 2, 1, 1])]
        matrix, ignored = ConfusionMatrix.confusion_counts(truth, preds, 3)
        expected = torch.tensor([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
        assert torch.equal(matrix, expected)
        assert ignored == 1  # the -1 padding entry

    def test_track_padding_is_counted_as_ignored(self):
        bundles = [make_bundle(7)]
        v2 = run_v2("track_origin", bundles)
        n_padded = int((bundles[0].get("labels.tracks.ftagTruthOriginLabel") == -1).sum())
        assert v2.last_ignored == n_padded
        assert (
            int(v2.last_matrix.sum()) + n_padded
            == bundles[0].get("labels.tracks.ftagTruthOriginLabel").numel()
        )


class TestConfusionMatrixSurface:
    def test_requires_declared_after_setup(self):
        callback = ConfusionMatrix(task_name="jets_classification")
        callback.setup(None, stub_pl_module(), stage="fit")
        assert callback.requires == (
            "preds.jets.jets_classification",
            "labels.jets.flavour_label",
        )

    def test_epoch_end_resets_accumulators_but_keeps_stash(self):
        v2 = run_v2("jets_classification", [make_bundle(1)])
        assert v2.truth_labels == [] and v2.pred_labels == []
        assert v2.last_matrix is not None and len(v2.last_truth_labels) == 50

    def test_class_names_override_list_and_dict(self):
        callback = ConfusionMatrix(
            task_name="jets_classification", class_names_override=["b", "c", "u"]
        )
        callback.setup(None, stub_pl_module(), stage="fit")
        assert callback.task_class_names == ["b", "c", "u"]
        callback = ConfusionMatrix(
            task_name="jets_classification", class_names_override={"bjets": "b"}
        )
        callback.setup(None, stub_pl_module(), stage="fit")
        assert callback.task_class_names == ["b", "cjets", "ujets"]

    def test_non_fit_setup_is_a_noop(self):
        callback = ConfusionMatrix(task_name="nope")
        callback.setup(None, stub_pl_module(), stage="test")  # must not raise
        assert callback.requires == ()

    def test_unknown_task_is_config_error_with_candidates(self):
        callback = ConfusionMatrix(task_name="jets_classificaton")  # typo
        with pytest.raises(ConfigError, match="jets_classification"):
            callback.setup(None, stub_pl_module(), stage="fit")

    def test_non_task_module_is_config_error(self):
        callback = ConfusionMatrix(task_name="encoder")
        with pytest.raises(ConfigError, match="classification task"):
            callback.setup(None, stub_pl_module(), stage="fit")

    def test_non_saltmodule_is_config_error(self):
        callback = ConfusionMatrix(task_name="jets_classification")
        with pytest.raises(ConfigError, match="graph-module dict"):
            callback.setup(None, SimpleNamespace(), stage="fit")

    def test_fit_val_demand_static_without_setup(self):
        # the FIT/VAL-sink declaration (design §3.1, §3.4): resolves from the
        # module dict alone, NO setup needed (the static-tooling path)
        callback = ConfusionMatrix(task_name="jets_classification")
        modules = stub_pl_module()._graph_modules
        assert callback.fit_val_demand(modules) == (
            "preds.jets.jets_classification",
            "labels.jets.flavour_label",
        )
        assert callback.requires == ()  # setup never ran; static surface only

    def test_fit_val_demand_unknown_task_raises_with_candidates(self):
        callback = ConfusionMatrix(task_name="nope")
        with pytest.raises(ConfigError, match="classification task") as excinfo:
            callback.fit_val_demand(stub_pl_module()._graph_modules)
        assert "jets_classification" in str(excinfo.value)  # candidate listed
