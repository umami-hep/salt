"""Cross-cutting task-module tests: full plan execution, expose opt-out, no-IO guard."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from salt.core.graph import (
    Bundle,
    ConfigError,
    Executor,
    Mode,
    compile_plan,
    flatten_spec,
)
from salt.core.nn import (
    LossSum,
    bind_all,
    materialise_all,
    resolve_bind_schema,
)
from salt.core.nn.tasks import (
    ClassificationTaskModule,
    RegressionTaskModule,
    VertexingTaskModule,
)
from salt.tests._fixtures.gn2v2_fixture import (
    make_gn2_batch,
)
from salt.tests._fixtures.gn2v2_fixture import (
    ORIGIN_CLASSES,
    build_gn2v2_modules,
    compile_gn2v2,
    gn2v2_sources,
)
from salt.tests.unit.nn.conftest import B, T, fit_bundle

# full plan execution (FIT + TEST), debug mode on


class TestGn2V2Execution:
    def test_fit_plan_runs_in_debug_mode(self, gn2v2):
        """Debug execution: read-tracking, write-once, and mutation checks all pass."""
        modules, plan, _ = gn2v2
        b = Executor(plan).run(fit_bundle(), debug=True)
        total = b.get("loss.total")
        assert total.shape == ()
        assert torch.isfinite(total)
        expected = (
            b.get("losses.jets_classification")
            + b.get("losses.track_origin")
            + b.get("losses.track_vertexing")
        )
        assert torch.equal(total, expected)

    def test_fit_preds_are_raw_logits(self, gn2v2):
        modules, plan, _ = gn2v2
        b = Executor(plan).run(fit_bundle())
        logits = b.get("preds.jets.jets_classification")
        assert logits.shape == (B, 3)
        assert not torch.allclose(logits.sum(-1), torch.ones(B))  # not softmaxed

    def test_test_plan_classification_preds_are_raw_logits(self, gn2v2):
        """TEST classification + vertexing preds are RAW since the flips (design §2)."""
        modules, _, _ = gn2v2
        plan = compile_gn2v2(modules, Mode.TEST)
        assert "loss" not in plan.module_names  # LossSum inactive outside TRAINING
        b = Bundle()
        inputs, masks = make_gn2_batch(B, T)
        for stream, x in inputs.items():
            b.set(f"inputs.{stream}", x)
        b.set("masks.tracks", masks["tracks"])
        with torch.no_grad():
            b = Executor(plan).run(b, debug=True)
        # classification heads now publish RAW logits in TEST (the flip): the
        # rows do NOT sum to 1 (not softmaxed)
        logits = b.get("preds.jets.jets_classification")
        assert logits.shape == (B, 3)
        assert not torch.allclose(logits.sum(-1), torch.ones(B), atol=1e-3)
        track_logits = b.get("preds.tracks.track_origin")
        valid = ~masks["tracks"]
        assert not torch.allclose(
            track_logits[valid].sum(-1), torch.ones(int(valid.sum())), atol=1e-3
        )
        # vertexing (W34.3 flipped) TEST output: RAW [E, 1] edge scores (the
        # union-find moved off forward to get_output), NOT the [B, T, 1]
        # per-node assignments the forward used to publish.
        vtx = b.get("preds.tracks.track_vertexing")
        assert vtx.ndim == 2 and vtx.shape[1] == 1  # [E, 1] raw edge scores

    def test_onnx_plan_keeps_raw_vertexing_scores(self, gn2v2):
        """ONNX vertexing publishes raw edge scores for the export reduce (§3.3)."""
        modules, _, _ = gn2v2
        plan = compile_gn2v2(modules, Mode.ONNX)
        b = Bundle()
        inputs, masks = make_gn2_batch(B, T)
        for stream, x in inputs.items():
            b.set(f"inputs.{stream}", x)
        b.set("masks.tracks", masks["tracks"])
        with torch.no_grad():
            b = Executor(plan).run(b)
        scores = b.get("preds.tracks.track_vertexing")
        assert scores.ndim == 2
        assert scores.shape[1] == 1  # [E, 1] raw edge scores

    def test_all_modules_are_nn_modules(self, gn2v2):
        """The module dict must be nn.ModuleDict-compatible (SaltModule, design §3.4)."""
        modules, _, _ = gn2v2
        assert all(isinstance(m, nn.Module) for m in modules.values())
        nn.ModuleDict(modules)  # must not raise


class TestExposeOptOut:
    """Per-task ``expose: [fit, val]`` opt-out (design §4.2, M5 sub-wave D)."""

    def test_parse_default_is_all_modes(self):
        task = ClassificationTaskModule(
            stream="jets", label="f", class_names=["a", "b"], input="pooled.global"
        )
        assert task.expose_modes == Mode.ALL

    def test_parse_fit_val_gates_pred_port(self):
        task = ClassificationTaskModule(
            stream="tracks", label="o", class_names=["a", "b"],
            context="pooled.global", expose=["fit", "val"],
        )
        task.name = "aux"
        assert task.expose_modes == (Mode.FIT | Mode.VAL)
        produced_fit = flatten_spec(task.declare_io(Mode.FIT).produces)
        # the pred port is active in FIT/VAL but NOT in TEST/ONNX
        pred = produced_fit[task.pred_key]
        assert pred.active_in(Mode.FIT) and pred.active_in(Mode.VAL)
        assert not pred.active_in(Mode.TEST)
        assert not pred.active_in(Mode.ONNX)
        # the loss is FIT|VAL regardless (the task still trains)
        assert produced_fit[task.loss_key].active_in(Mode.FIT)

    def test_parse_case_insensitive(self):
        task = ClassificationTaskModule(
            stream="jets", label="f", class_names=["a", "b"], input="pooled.global",
            expose=["FIT", "Val"],
        )
        assert task.expose_modes == (Mode.FIT | Mode.VAL)

    def test_empty_list_rejected(self):
        with pytest.raises(ConfigError, match="empty list"):
            ClassificationTaskModule(
                stream="jets", label="f", class_names=["a"], input="pooled.global", expose=[]
            )

    def test_string_value_rejected(self):
        with pytest.raises(ConfigError, match="list of mode names"):
            ClassificationTaskModule(
                stream="jets", label="f", class_names=["a"], input="pooled.global", expose="fit"
            )

    def test_unknown_mode_rejected(self):
        with pytest.raises(ConfigError, match="unknown expose mode"):
            ClassificationTaskModule(
                stream="jets", label="f", class_names=["a"], input="pooled.global",
                expose=["fit", "predict"],
            )

    def test_expose_on_regression_and_vertexing(self):
        # the opt-out lives on the shared base, so all task families carry it:
        # the pred port is declared but mode-inactive in TEST (planner-pruned)
        reg = RegressionTaskModule(stream="jets", targets="x", input="pooled.global",
                                   expose=["fit", "val"])
        reg.name = "reg"
        reg_pred = flatten_spec(reg.declare_io(Mode.TEST).produces)[reg.pred_key]
        assert reg_pred.active_in(Mode.FIT) and not reg_pred.active_in(Mode.TEST)
        vtx = VertexingTaskModule(
            stream="tracks", label="ftagTruthVertexIndex", origin_label="o",
            expose=["fit", "val"],
        )
        vtx.name = "vtx"
        vtx_pred = flatten_spec(vtx.declare_io(Mode.TEST).produces)[vtx.pred_key]
        assert vtx_pred.active_in(Mode.FIT) and not vtx_pred.active_in(Mode.TEST)

    def _modules_with_exposed_aux(self, norm_dict):
        modules = build_gn2v2_modules(norm_dict)
        # rebuild track_origin with expose: [fit, val] (a train-only aux task)
        modules["track_origin"] = ClassificationTaskModule(
            stream="tracks", label="ftagTruthOriginLabel",
            class_names=list(ORIGIN_CLASSES), context="pooled.global", weight=0.5,
            dense={"hidden_layers": [16], "activation": "ReLU"}, expose=["fit", "val"],
        )
        modules["track_origin"].name = "track_origin"
        # fresh LossSum re-narrowed over the rebuilt module set (the prior one
        # has already fixed its keys and cannot re-narrow)
        loss = LossSum()
        loss.name = "loss"
        loss.narrow(LossSum.collect_loss_keys(modules))
        modules["loss"] = loss
        return modules

    def test_exposed_task_kept_in_fit_plan(self, norm_paths):
        modules = self._modules_with_exposed_aux(norm_paths[0])
        plan = compile_plan(modules, Mode.FIT, sources=gn2v2_sources(), sinks=["loss.total"])
        assert "track_origin" in plan.module_names

    def test_exposed_task_pruned_from_test_plan(self, norm_paths):
        modules = self._modules_with_exposed_aux(norm_paths[0])
        # the exposed aux produces nothing in TEST → the planner prunes it from
        # the TEST plan entirely (compile with its full preds set as sinks: the
        # exposed pred is mode-inactive, so it cannot be demanded)
        plan = compile_plan(
            modules,
            Mode.TEST,
            sources=gn2v2_sources(),
            sinks=["preds.jets.jets_classification", "preds.tracks.track_vertexing"],
        )
        assert "track_origin" not in plan.module_names
        assert "jets_classification" in plan.module_names

    def test_default_aux_alive_in_test_when_demanded(self, norm_paths):
        # negative control: WITHOUT expose, the same task IS active in TEST — its
        # pred is a legitimate TEST sink the planner keeps (the column expose
        # would have removed). Proves the exposed/non-exposed difference is the
        # pred port's TEST activity, not some other pruning.
        modules = build_gn2v2_modules(norm_paths[0])
        plan = compile_plan(
            modules,
            Mode.TEST,
            sources=gn2v2_sources(),
            sinks=[
                "preds.jets.jets_classification",
                "preds.tracks.track_origin",
                "preds.tracks.track_vertexing",
            ],
        )
        assert "track_origin" in plan.module_names

    def test_exposed_pred_cannot_be_a_test_sink(self, norm_paths):
        # the exposed pred is mode-inactive in TEST, so demanding it as a TEST
        # sink is a no-producer error naming the FIT/VAL modes where it lives
        from salt.core.graph.errors import GraphError

        modules = self._modules_with_exposed_aux(norm_paths[0])
        with pytest.raises(GraphError, match="FIT/VAL"):
            compile_plan(
                modules,
                Mode.TEST,
                sources=gn2v2_sources(),
                sinks=["preds.tracks.track_origin"],
            )


class TestNoIOGuard:
    """Design §2.3 CI guard: declare_io and bind run under a no-I/O trap."""

    def test_declare_and_bind_are_file_free(self, monkeypatch):
        # construction is config capture only — safe to build under the trap
        def _forbid(*args, **kwargs):
            raise AssertionError(f"file I/O during declare_io/bind (design §2.3): open({args!r})")

        monkeypatch.setattr("builtins.open", _forbid)
        monkeypatch.setattr("h5py.File", _forbid)
        monkeypatch.setattr("pathlib.Path.open", _forbid)
        modules = build_gn2v2_modules("/nonexistent/norm_dict.yaml")
        for mode in (Mode.FIT, Mode.VAL, Mode.TEST, Mode.ONNX):
            for module in modules.values():
                module.declare_io(mode)
        plans = [compile_gn2v2(modules, mode) for mode in (Mode.FIT, Mode.TEST)]
        bind_all(modules, resolve_bind_schema(plans))

        # release the trap: materialise IS the sanctioned file hook and must
        # be the first thing that touches the (nonexistent) norm dict
        monkeypatch.undo()
        with pytest.raises(FileNotFoundError):
            materialise_all(modules)
