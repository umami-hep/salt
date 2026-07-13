"""Tests for `salt.core.callbacks.ConfusionMatrix` (split from test_callbacks.py)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from salt.core.callbacks import ConfusionMatrix
from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError


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


class TestConfusionMatrixValues:
    """Accumulation + reduction values on deterministic eval batches."""

    # DEL-1: run_v1 + test_matches_v1_callback (v1-vs-v2 value parity) retired
    # with the v1 tree (parity-closure doctrine: git checkout 29c67a1). The
    # accumulation-across-batches surface is pinned below.

    @pytest.mark.parametrize(
        ("task_name", "label_key", "n_classes"),
        [
            ("jets_classification", "labels.jets.flavour_label", 3),
            ("track_origin", "labels.tracks.ftagTruthOriginLabel", 8),
        ],
    )
    def test_accumulates_across_batches(self, task_name, label_key, n_classes):
        bundles = [make_bundle(seed) for seed in (1, 2, 3)]
        v2 = run_v2(task_name, bundles)
        # every label element from every batch is accumulated exactly once
        n_labels = sum(int(b.get(label_key).numel()) for b in bundles)
        assert len(v2.last_truth_labels) > 0
        assert sum(int(t.numel()) for t in v2.last_truth_labels) == n_labels
        assert sum(int(t.numel()) for t in v2.last_pred_labels) == n_labels
        # the epoch matrix counts every non-padded label exactly once
        n_padded = sum(int((b.get(label_key) == -1).sum()) for b in bundles)
        assert int(v2.last_matrix.sum()) + n_padded == n_labels
        assert v2.last_matrix.shape == (n_classes, n_classes)

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
