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
from salt.core.callbacks import Checkpoint, ConfusionMatrix, GraphArtifacts, ProgressBar
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


# ---------------------------------------------------------------------------
# MaskformerMetrics (FD 1200-1202) — matched.objects.* FIT/VAL sink + metrics
# ---------------------------------------------------------------------------


def make_matched_bundle(seed: int = 5, batch: int = 6, m: int = 5, n_cls: int = 3, t: int = 10):
    """A VAL step bundle carrying the matcher-permuted ``matched.objects.*`` keys."""
    from salt.tests.core.regression_fixture import MASKFORMER_WRITER_REG_TARGETS

    gen = torch.Generator().manual_seed(seed)
    bundle = Bundle()
    bundle.set("matched.objects.class_logits", torch.randn(batch, m, n_cls, generator=gen))
    bundle.set("matched.objects.object_class", torch.randint(0, n_cls, (batch, m), generator=gen))
    bundle.set("matched.objects.masks", torch.randn(batch, m, t, generator=gen))
    bundle.set("matched.objects.target_masks", torch.rand(batch, m, t, generator=gen) > 0.5)
    bundle.set(
        "matched.objects.regression",
        torch.randn(batch, m, len(MASKFORMER_WRITER_REG_TARGETS), generator=gen),
    )
    bundle.set(
        "matched.objects.target_regression",
        torch.randn(batch, m, len(MASKFORMER_WRITER_REG_TARGETS), generator=gen),
    )
    bundle.set("masks.tracks", torch.zeros(batch, t, dtype=torch.bool))
    return bundle


class TestMaskformerMetrics:
    def test_fit_val_demand_declares_matched_keys(self):
        # the DP2 declaration: the matcher-permuted class/object-class/mask keys
        # the callback reads each VAL epoch (config-only, no setup needed)
        from salt.core.callbacks import MaskformerMetrics

        callback = MaskformerMetrics()
        assert callback.fit_val_demand({}) == (
            "matched.objects.class_logits",
            "matched.objects.object_class",
            "matched.objects.masks",
            "matched.objects.target_masks",
        )

    def test_enters_fit_val_sinks_and_keeps_matched_loss_alive(self):
        # the criterion: the REAL MaskformerMetrics callback, attached to a
        # SaltModule carrying a MaskFormerMatchedLoss, makes the matched.* keys
        # FIT/VAL plan sinks (DP2) AND keeps the matched loss alive under pruning
        from salt.core.callbacks import MaskformerMetrics
        from salt.core.graph.planner import compile_plan
        from salt.core.nn import LossSum
        from salt.tests.core.regression_fixture import (
            build_matched_loss_module,
        )

        loss_module = build_matched_loss_module()
        model = SaltModule({"mf_matched_loss": loss_module, "loss": LossSum()}, lrs=LRS)
        model._trainer = SimpleNamespace(  # noqa: SLF001 - duck-typed attach
            callbacks=[MaskformerMetrics()], datamodule=SimpleNamespace(reader=None)
        )
        for mode in (Mode.FIT, Mode.VAL):
            sinks = model._model_sinks(mode)  # noqa: SLF001
            assert "matched.objects.class_logits" in sinks
            assert "matched.objects.object_class" in sinks
        # the matched loss survives pruning to a FIT plan with those sinks: the
        # matched.* products keep its module alive (it has no loss anchoring them
        # beyond losses.* — the callback sink is what keeps matched.* reachable)
        m = 5
        n_cls = 3
        from salt.core.graph.spec import TensorSpec, unflatten_spec

        f = Mode.FIT
        sources = unflatten_spec({
            "objects.class_logits": TensorSpec(shape=("B", m, n_cls), dtype="float32", modes=f),
            "objects.class_probs": TensorSpec(shape=("B", m, n_cls), dtype="float32", modes=f),
            "objects.embed": TensorSpec(shape=("B", m, 16), dtype="float32", modes=f),
            "objects.masks": TensorSpec(shape=("B", m, "T:tracks"), dtype="float32", modes=f),
            "preds.objects.regression": TensorSpec(shape=("B", m, 3), dtype="float32", modes=f),
            "targets.objects.regression": TensorSpec(shape=("B", m, 3), dtype="float32", modes=f),
            "labels.objects.object_class": TensorSpec(
                shape=("B", m), dtype="int64", kind="label", modes=f
            ),
            "labels.objects.masks": TensorSpec(
                shape=("B", m, "T:tracks"), dtype="bool", kind="label", modes=f
            ),
        })
        plan = compile_plan(
            {"mf_matched_loss": loss_module},
            Mode.FIT,
            sources=sources,
            sinks=[
                "matched.objects.class_logits",
                "matched.objects.object_class",
                "matched.objects.masks",
                "matched.objects.target_masks",
            ],
        )
        assert "mf_matched_loss" in plan.module_names

    def test_compute_metrics_from_matched_bundle(self):
        # the metrics are computed from matched.objects.* (bundle-native) and stashed
        from salt.core.callbacks import MaskformerMetrics

        callback = MaskformerMetrics()
        trainer = SimpleNamespace(fast_dev_run=False)
        logged: dict[str, float] = {}
        pl_module = SimpleNamespace(log=lambda name, value: logged.__setitem__(name, float(value)))
        callback.on_validation_batch_end(
            trainer, pl_module, {"bundle": make_matched_bundle()}, None, 0
        )
        # the v1 metric set is present, logged under the val/ prefix and stashed
        assert callback.last_metrics
        assert "class_exact_match" in callback.last_metrics
        assert "notnull_eff" in callback.last_metrics and "notnull_pur" in callback.last_metrics
        assert "query_perfect_match_eff" in callback.last_metrics
        assert "query_regression_mae" in callback.last_metrics
        assert any(k.startswith("val/") for k in logged)
        # efficiencies / purities are valid fractions in [0, 1]
        assert 0.0 <= callback.last_metrics["notnull_eff"] <= 1.0
        assert 0.0 <= callback.last_metrics["class_exact_match"] <= 1.0

    def test_only_val_skips_train_logging(self):
        from salt.core.callbacks import MaskformerMetrics

        callback = MaskformerMetrics(only_val=True)
        logged: dict[str, float] = {}
        pl_module = SimpleNamespace(log=lambda n, v: logged.__setitem__(n, v))
        callback.on_train_batch_end(
            SimpleNamespace(fast_dev_run=False),
            pl_module,
            {"bundle": make_matched_bundle()},
            None,
            0,
        )
        assert logged == {}  # only_val=True -> no train logging


# ---------------------------------------------------------------------------
# GraphArtifacts (design §4.4 run-dir artifacts)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_model(tmp_path_factory) -> SaltModule:
    """A SaltModule with compiled FIT/VAL/TEST plans (no trainer needed)."""
    base = tmp_path_factory.mktemp("artifacts_model")
    nd, cd = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd, cd)
    model = SaltModule(build_gn2v2_modules(nd), lrs=LRS)
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
                groups={"jets": {"global_object": True}, "tracks": {"global_object": False}}
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


# ---------------------------------------------------------------------------
# Checkpoint (the v1 salt.callbacks.Checkpoint port — D2 slice) + ProgressBar
# ---------------------------------------------------------------------------


def _ckpt_trainer(log_dir: str, *, fast_dev_run: bool = False) -> SimpleNamespace:
    """A minimal trainer satisfying the real `ModelCheckpoint.setup`."""
    return SimpleNamespace(
        fast_dev_run=fast_dev_run,
        log_dir=log_dir,
        default_root_dir=log_dir,
        is_global_zero=True,
        loggers=[],
        strategy=SimpleNamespace(broadcast=lambda x: x),
    )


class TestCheckpoint:
    def test_filename_monitor_contract(self):
        # the v1 ctor string (checkpoint.py:26): epoch=NNN-loss={monitor:.5f}
        cb = Checkpoint(monitor_loss="val/jets_classification_loss")
        assert cb.filename == "epoch={epoch:03d}-loss={val/jets_classification_loss:.5f}"
        assert cb.save_top_k == -1  # keep every epoch (v1 :27)
        assert cb.monitor == "val/jets_classification_loss"
        assert cb.auto_insert_metric_name is False

    def test_formatted_name_keeps_loss_tag(self):
        # the per-task metric (carrying a '/') renders into the loss= stem the
        # salt2-test best-epoch glob keys on
        cb = Checkpoint(monitor_loss="val/loss")
        name = cb.format_checkpoint_name({"epoch": 9, "val/loss": 0.64624})
        assert name == "epoch=009-loss=0.64624.ckpt"

    def test_fname_string_override(self):
        cb = Checkpoint(monitor_loss="val/loss", fname_string="val_loss")
        assert cb.filename == "epoch={epoch:03d}-val_loss={val/loss:.5f}"

    def test_setup_fit_forces_ckpts_dir(self, tmp_path):
        # the real setup forces dirpath to <log_dir>/ckpts (v1 :44-46), driven
        # through ModelCheckpoint.setup's pre-set-dirpath short-circuit
        cb = Checkpoint(monitor_loss="val/loss")
        cb.setup(_ckpt_trainer(str(tmp_path)), SimpleNamespace(), stage="fit")
        assert str(cb.dirpath) == str(tmp_path / "ckpts")

    def test_setup_non_fit_does_not_force_ckpts(self, tmp_path):
        # non-fit setup is a no-op for our branch (v1 :30-32): super resolves to
        # the Lightning default 'checkpoints/', not the v1 'ckpts/'
        cb = Checkpoint()
        cb.setup(_ckpt_trainer(str(tmp_path)), SimpleNamespace(), stage="test")
        assert not str(cb.dirpath).endswith("ckpts")

    def test_setup_fast_dev_run_does_not_force_ckpts(self, tmp_path):
        cb = Checkpoint()
        cb.setup(_ckpt_trainer(str(tmp_path), fast_dev_run=True), SimpleNamespace(), stage="fit")
        assert not str(cb.dirpath).endswith("ckpts")

    def test_setup_s3_log_dir_raises(self):
        # the v1 s3 branch (checkpoint.py:34-43) is deferred to M6 — loud error
        cb = Checkpoint()
        with pytest.raises(ConfigError, match="s3://"):
            cb.setup(_ckpt_trainer("s3://bucket/run"), SimpleNamespace(), stage="fit")

    def test_best_checkpoint_resolves_what_setup_writes(self, tmp_path):
        # the salt2-test run-dir contract end-to-end: a checkpoint named by the
        # callback under its setup-forced ckpts/ dir is discovered by
        # salt.core.main._best_checkpoint (the no-ckpt_path fallback)
        from salt.core.main import _best_checkpoint

        cb = Checkpoint(monitor_loss="val/loss")
        cb.setup(_ckpt_trainer(str(tmp_path)), SimpleNamespace(), stage="fit")
        ckpt_dir = Path(cb.dirpath)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for epoch, loss in ((8, 0.70123), (9, 0.64624)):
            name = cb.format_checkpoint_name({"epoch": epoch, "val/loss": loss})
            (ckpt_dir / name).write_text("ckpt")
        (tmp_path / "config.yaml").write_text("class_path: salt.core.SaltModule\n")
        best = _best_checkpoint(tmp_path / "config.yaml")
        assert Path(best).name == "epoch=009-loss=0.64624.ckpt"  # lowest loss


class TestProgressBar:
    def test_is_stock_tqdm_under_salt_name(self):
        from lightning.pytorch.callbacks import TQDMProgressBar

        bar = ProgressBar(refresh_rate=50)
        assert isinstance(bar, TQDMProgressBar)  # the stock bar (v1 base.yaml:38-39)
        assert bar.refresh_rate == 50  # init args pass straight through

    def test_registered_by_default_in_base2(self, tmp_path):
        # base2.yaml ships checkpoint (Checkpoint) + progress (ProgressBar); the
        # CLI assembles both into trainer.callbacks (run=False, no fit)
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
        assert any(isinstance(cb, Checkpoint) for cb in cli.trainer.callbacks)
        assert any(isinstance(cb, ProgressBar) for cb in cli.trainer.callbacks)
