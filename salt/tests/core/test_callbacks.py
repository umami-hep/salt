"""Tests for the v2 callbacks (`salt.core.callbacks`): ConfusionMatrix + GraphArtifacts.

The ConfusionMatrix value bar is W5 (plan 06): fed the same eval batches, the
v2 bundle-native callback must accumulate exactly the values the v1
``ConfusionMatrixCallback`` accumulates through the `bundle_as_v1_outputs`
shim — compared here logger-free via the stashed lists/matrix.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from salt.callbacks.confusion_matrix import ConfusionMatrixCallback as V1ConfusionMatrix
from salt.core.callbacks import ConfusionMatrix, GraphArtifacts
from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import Mode, flatten_spec
from salt.core.saltmodule import SaltModule, bundle_as_v1_outputs
from salt.tests.core.gn2_fixture import write_parity_norm_dict
from salt.tests.core.gn2v2_fixture import build_gn2v2_modules, gn2v2_sources

LRS = {"initial": 1e-4, "max": 1e-3, "end": 1e-5, "pct_start": 0.1}
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


# ---------------------------------------------------------------------------
# GraphArtifacts (design §4.4 run-dir artifacts)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_model(tmp_path_factory) -> SaltModule:
    """A SaltModule with compiled FIT/VAL/TEST plans (no trainer needed)."""
    base = tmp_path_factory.mktemp("artifacts_model")
    nd, cd = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd, cd)
    model = SaltModule(build_gn2v2_modules(nd), lrs_config=LRS)
    boundary = flatten_spec(gn2v2_sources())
    model.compile_mode(Mode.FIT, boundary)
    model.compile_mode(Mode.VAL, boundary)
    model.compile_mode(Mode.TEST, boundary)
    return model


def stub_trainer(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        is_global_zero=True,
        log_dir=str(tmp_path),
        default_root_dir=str(tmp_path),
        datamodule=None,
    )


class TestGraphArtifacts:
    def test_fit_artifacts_written(self, fitted_model, tmp_path):
        GraphArtifacts().on_fit_start(stub_trainer(tmp_path), fitted_model)
        plan_fit = (tmp_path / "plan_fit.txt").read_text()
        assert "plan [mode=FIT]" in plan_fit
        assert "# model plan" in plan_fit
        assert "plan [mode=VAL]" in (tmp_path / "plan_val.txt").read_text()
        assert "digraph salt_core_fit" in (tmp_path / "graph_fit.dot").read_text()
        assert (tmp_path / "graph_fit.svg").stat().st_size > 0

    def test_test_artifacts_written(self, fitted_model, tmp_path):
        GraphArtifacts().on_test_start(stub_trainer(tmp_path), fitted_model)
        assert "plan [mode=TEST]" in (tmp_path / "plan_test.txt").read_text()
        assert (tmp_path / "graph_test.dot").exists()
        assert (tmp_path / "graph_test.svg").stat().st_size > 0

    def test_test_artifacts_default_next_to_checkpoint(self, fitted_model, tmp_path):
        # M3-review HIGH fix: with a known ckpt_path, the test-path default
        # is the checkpoint dir (where the eval H5 goes), not the log dir
        trainer = stub_trainer(tmp_path)
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        trainer.ckpt_path = str(ckpt_dir / "epoch=000-loss=1.00000.ckpt")
        GraphArtifacts().on_test_start(trainer, fitted_model)
        assert (ckpt_dir / "plan_test.txt").exists()
        assert not (tmp_path / "plan_test.txt").exists()
        # the fit path is unchanged (log dir), ckpt_path or not
        GraphArtifacts().on_fit_start(trainer, fitted_model)
        assert (tmp_path / "plan_fit.txt").exists()

    def test_writer_sinks_table_in_plan_test(self, fitted_model, tmp_path):
        # M3-review fix: plan_test.txt answers "which writer consumes
        # preds.X" via the per-writer demand table
        from salt.core.data import H5StructuredReader
        from salt.core.writers import TaskWriter, WriterCallback

        trainer = stub_trainer(tmp_path)
        trainer.callbacks = [WriterCallback(modules={"tasks": TaskWriter()})]
        trainer.datamodule = SimpleNamespace(
            reader=H5StructuredReader(
                groups={"jets": {"vector": True}, "tracks": {"vector": False}}
            )
        )
        GraphArtifacts().on_test_start(trainer, fitted_model)
        text = (tmp_path / "plan_test.txt").read_text()
        assert "# writer sinks (design §8)" in text
        assert "tasks (TaskWriter): preds.jets.jets_classification" in text
        assert "preds.tracks.track_vertexing" in text

    def test_resolved_io_yaml_written(self, fitted_model, tmp_path):
        # design §4.4: the machine-readable artifact (M3-review fix — it was
        # silently missing)
        import yaml

        GraphArtifacts().on_fit_start(stub_trainer(tmp_path), fitted_model)
        payload = yaml.safe_load((tmp_path / "resolved_io.yaml").read_text())
        assert set(payload) == {"fit", "val"}
        assert payload["fit"]["plan_hash"]
        encoder = payload["fit"]["modules"]["encoder"]
        assert encoder["class"] == "TransformerEncoder"
        assert "requires" in encoder and "produces" in encoder
        # specs carry resolved shape/dtype/kind (and fields where declared)
        jets_in = payload["fit"]["sources"]["inputs.jets"]
        assert jets_in["kind"] == "data"
        assert jets_in["fields"]

    def test_image_format_and_output_dir_options(self, fitted_model, tmp_path):
        out = tmp_path / "sub"
        GraphArtifacts(output_dir=str(out), image_format="png").on_test_start(
            stub_trainer(tmp_path), fitted_model
        )
        assert (out / "graph_test.png").stat().st_size > 0

    def test_non_rank_zero_writes_nothing(self, fitted_model, tmp_path):
        trainer = stub_trainer(tmp_path)
        trainer.is_global_zero = False
        GraphArtifacts().on_fit_start(trainer, fitted_model)
        assert list(tmp_path.iterdir()) == []

    def test_module_without_plans_warns_and_skips(self, tmp_path):
        with pytest.warns(UserWarning, match="no compiled plans"):
            GraphArtifacts().on_fit_start(stub_trainer(tmp_path), SimpleNamespace(plans={}))
        assert list(tmp_path.iterdir()) == []

    def test_registered_by_default_in_base2(self, tmp_path):
        # base2.yaml ships the artifacts entry; the CLI assembles it into
        # trainer.callbacks (run=False instantiation — no fit needed)
        from salt.core.schema import dump_schema, save_schema
        from salt.tests.core.test_salt2_cli import make_cli
        from salt.utils.inputs import write_dummy_file

        nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
        write_parity_norm_dict(nd, cd)
        h5 = tmp_path / "pp_output_train.h5"
        write_dummy_file(h5, nd)
        schema = tmp_path / "schema.yaml"
        save_schema(dump_schema(h5), schema)
        cli = make_cli({"dir": tmp_path, "h5": h5, "nd": nd, "schema": schema})
        assert any(isinstance(cb, GraphArtifacts) for cb in cli.trainer.callbacks)
