"""``salt graph`` (validate/plan/execute/plot/why) against the toy trainer fixture configs
(``salt/tests/_fixtures/configs/toy_trainer*.yaml``) — the graph kernel's own CLI surface.
"""

from pathlib import Path

import pytest
import torch

from salt.cli import load_config
from salt.cli import main as cli_main
from salt.data import MODEL_VISIBLE_NAMESPACES
from salt.data.base import SaltDatasetModule
from salt.graph.bundle import Bundle
from salt.graph.errors import AllModesDeadError
from salt.graph.executor import Executor
from salt.graph.planner import SINKS, Edge, compile_plan, deadcode
from salt.graph.spec import PRIMARY_MODES, Mode, unflatten_spec
from salt.model.bind import resolve_bind_schema
from salt.tests._fixtures.toys import ToyDead

CONFIG_DIR = Path(__file__).parent.parent.parent / "_fixtures" / "configs"
TOY_CFG = str(CONFIG_DIR / "toy_trainer.yaml")
BROKEN_CFG = str(CONFIG_DIR / "toy_trainer_broken.yaml")


def _plan(cfg, mode):
    return compile_plan(cfg.modules, mode, cfg.sources, schema=cfg.schema, sinks=cfg.sinks)


def _deadcode(cfg, mode):
    return deadcode(cfg.modules, mode, cfg.sources, cfg.schema, cfg.sinks)


def _model_plan(cfg, mode):
    """The model-side plan `SaltModule.compile_mode` builds: the model modules (+ the
    folded sink) over the dataset boundary — the data steps' model-visible produces.
    """
    data_side = {name for name, m in cfg.modules.items() if isinstance(m, SaltDatasetModule)}
    boundary = {
        key: spec
        for step in _plan(cfg, mode).steps
        if step.name in data_side
        for key, spec in step.produces.items()
        if key.split(".", 1)[0] in MODEL_VISIBLE_NAMESPACES
    }
    model_side = {name: m for name, m in cfg.modules.items() if name not in data_side}
    return compile_plan(model_side, mode, unflatten_spec(boundary), sinks=cfg.sinks)


def _boundary_bundle(plan, n=5):
    """A seeded bundle carrying every source leaf of `plan` (batch size `n`)."""
    gen = torch.Generator().manual_seed(0)
    nested: dict = {}
    for key, spec in plan.sources.items():
        if spec.kind == "pad_mask":
            value = torch.zeros(n, dtype=torch.bool)
        elif spec.kind == "label":
            value = torch.randint(3, (n,), generator=gen)
        elif spec.kind == "meta":
            value = torch.tensor([0, n])
        else:  # inputs.x: one column per declared field
            value = torch.randn(n, len(spec.fields), generator=gen)
        *parents, leaf = key.split(".")
        node = nested
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return Bundle(nested)


# validate


class TestValidate:
    @pytest.mark.parametrize("mode", ["fit", "test", "onnx"])
    def test_validate_single_mode(self, mode, capsys):
        assert cli_main(["graph", "validate", "-c", TOY_CFG, "--mode", mode]) == 0
        assert f"OK [mode={mode.upper()}]" in capsys.readouterr().out

    def test_validate_all_modes(self, capsys):
        assert cli_main(["graph", "validate", "-c", TOY_CFG]) == 0
        captured = capsys.readouterr()
        for mode in ("FIT", "VAL", "TEST", "ONNX"):
            assert f"OK [mode={mode}]" in captured.out
        # the reader carries a schema artifact, so field spellings are checked
        assert "cannot be checked statically" not in captured.err


# plan compilation: the loss path is fit/val-only


class TestPlans:
    def test_fit_plan_contains_loss_path(self):
        cfg = load_config(TOY_CFG)
        plan = _plan(cfg, Mode.FIT)
        # deterministic topo order: Kahn, ties broken by declaration order —
        # data.modules (reader, features, labels) then model.modules
        assert plan.module_names == ("reader", "features", "labels", "embed", "head", "loss")
        assert "losses.x" in plan.step("head").produces
        assert "labels.x.y" in plan.step("labels").produces  # narrowed from labels.**
        assert Edge("labels", "labels.x.y", "head") in plan.edges
        assert Edge("head", "losses.x", "loss") in plan.edges
        assert Edge("loss", "loss.total", SINKS) in plan.edges
        assert Edge("reader", "raw.x", "labels") in plan.edges
        assert not plan.sources  # the reader is the source node

    @pytest.mark.parametrize("mode", [Mode.TEST, Mode.ONNX])
    def test_test_and_onnx_plans_have_no_loss_path(self, mode):
        cfg = load_config(TOY_CFG)
        plan = _plan(cfg, mode)
        # Labels narrows to nothing: kept as a no-op terminal consumer of raw.x
        assert not plan.step("labels").produces
        produced = {key for step in plan.steps for key in step.produces}
        assert not any(key.startswith(("losses.", "loss.", "labels.")) for key in produced)
        assert "preds.x" in produced
        # TEST anchors on the writer sink, ONNX on every preds.* key
        assert Edge("head", "preds.x", "writer" if mode is Mode.TEST else SINKS) in plan.edges

    def test_writer_is_test_only(self):
        cfg = load_config(TOY_CFG)
        assert "writer" in _plan(cfg, Mode.TEST).module_names
        assert "writer" not in _plan(cfg, Mode.ONNX).module_names
        assert "writer" not in _plan(cfg, Mode.FIT).module_names

    def test_plan_hash_stable_across_loads(self):
        first, second = load_config(TOY_CFG), load_config(TOY_CFG)
        assert _plan(first, Mode.FIT).plan_hash == _plan(second, Mode.FIT).plan_hash
        assert _plan(first, Mode.FIT).plan_hash != _plan(first, Mode.TEST).plan_hash


# execution of the model-side plan — debug read tracking on


class TestExecution:
    def test_fit_plan_executes_loss_path(self):
        cfg = load_config(TOY_CFG)
        plan = _model_plan(cfg, Mode.FIT)
        bundle = _boundary_bundle(plan)
        out = Executor(plan).run(bundle, debug=True)
        assert out is bundle  # run returns the same bundle
        assert set(out.keys()) == {
            "inputs.x",
            "masks.x",
            "labels.x.y",
            "embed.x",
            "preds.x",
            "losses.x",
            "loss.total",
        }
        preds = out.get("preds.x")
        assert preds.shape == (5, 3)
        assert torch.allclose(preds.sum(dim=-1), torch.ones(5))  # softmax rows
        loss = out.get("loss.total")
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0  # cross entropy

    def test_test_plan_executes_and_writer_collects(self):
        cfg = load_config(TOY_CFG)
        plan = _model_plan(cfg, Mode.TEST)
        out = Executor(plan).run(_boundary_bundle(plan), debug=True)
        assert "losses.x" not in out
        assert "labels.x.y" not in out
        writer = cfg.modules["writer"]
        assert writer.collected == []  # a sink node: the executor never calls it
        writer.consume(out)  # the driver hands it each executed bundle
        assert len(writer.collected) == 1
        assert writer.collected[0].shape == (5, 3)

    def test_missing_source_raises_keyerror(self):
        cfg = load_config(TOY_CFG)
        plan = _model_plan(cfg, Mode.FIT)
        with pytest.raises(KeyError, match=r"inputs\.x"):
            Executor(plan).run(Bundle())


# deadcode and the all-modes-dead error


class TestDeadcode:
    def test_fit_deadcode_reports_unconsumed_preds(self):
        cfg = load_config(TOY_CFG)
        findings = _deadcode(cfg, Mode.FIT)
        assert [(f.module, f.key) for f in findings] == [("head", "preds.x")]
        assert "never consumed" in findings[0].reason

    @pytest.mark.parametrize("mode", [Mode.TEST, Mode.ONNX])
    def test_test_and_onnx_deadcode_clean(self, mode):
        cfg = load_config(TOY_CFG)
        assert _deadcode(cfg, mode) == []

    def test_deadcode_cli_report(self, capsys):
        assert cli_main(["graph", "deadcode", "-c", TOY_CFG]) == 0
        out = capsys.readouterr().out
        assert "'preds.x' (head)" in out
        assert "[deadcode] mode=TEST: OK" in out
        assert "[deadcode] mode=ONNX: OK" in out

    def test_toy_dead_raises_all_modes_dead(self):
        cfg = load_config(TOY_CFG)
        dead = ToyDead()
        dead.name = "deadend"
        modules = {**cfg.modules, "deadend": dead}
        with pytest.raises(AllModesDeadError, match="deadend"):
            compile_plan(modules, Mode.FIT, cfg.sources, schema=cfg.schema, sinks=cfg.sinks)
        # deadcode reports instead of raising
        findings = deadcode(modules, Mode.FIT, cfg.sources, cfg.schema, cfg.sinks)
        assert ("deadend", "*") in [(f.module, f.key) for f in findings]


# plot


class TestPlot:
    def test_fit_dot_contains_expected_edges(self, tmp_path, capsys):
        # .dot, not .svg/.png: this test asserts DOT text only, and a
        # non-.dot suffix triggers _render_with_dot's shell-out to the
        # Graphviz `dot` binary, which the CI image does not install
        # (pipeline #15650554 diagnosis item 7).
        out_path = tmp_path / "graph.dot"
        assert cli_main(["graph", "plot", "-c", TOY_CFG, "--mode", "fit", "-o", str(out_path)]) == 0
        dot = (tmp_path / "graph.dot").read_text()
        assert "digraph salt_core_fit {" in dot
        # port-card nodes: each consumed/produced key is a card row with its shape
        assert '"reader" [label=<' in dot
        assert '"head" [label=<' in dot
        assert "raw.x" in dot
        assert "inputs.x" in dot
        assert "embed.x" in dot
        assert "(B, 16)" in dot
        assert "masks.x" in dot
        assert "(B)" in dot
        assert "labels.x.y" in dot
        assert "losses.x" in dot
        assert "loss.total" in dot
        # deduped node -> node edges (no per-key edge labels)
        assert '"reader" -> "features";' in dot
        assert '"reader" -> "labels";' in dot
        assert '"features" -> "embed";' in dot
        assert '"embed" -> "head";' in dot
        assert '"labels" -> "head";' in dot
        assert '"head" -> "loss";' in dot
        assert '"loss" -> "<sinks>";' in dot
        # kind-coloured rows: labels orange, losses red, preds blue
        assert "#b5651d" in dot  # label kind
        assert "#c0392b" in dot  # loss kind
        assert "#1f6fb2" in dot  # preds kind
        # writer is TEST-gated: absent from the FIT graph entirely
        assert '"writer"' not in dot
        # inputs.x carries Features' fields but no shape (rows render the producer's
        # spec), so its 8-wide feature contract is checked on the static resolution
        cfg = load_config(TOY_CFG)
        widths = resolve_bind_schema([_plan(cfg, mode) for mode in PRIMARY_MODES]).widths
        assert widths["inputs.x"] == 8


# why (the debugging story)


class TestWhy:
    def test_fit_only_loss_explained_in_test_mode(self, capsys):
        rc = cli_main(["graph", "why", "-c", TOY_CFG, "--mode", "test", "--key", "losses.x"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "not in the plan" in out
        assert "mode-gated" in out
        assert "FIT/VAL" in out

    def test_fit_only_wildcard_label_explained_in_test_mode(self, capsys):
        rc = cli_main(["graph", "why", "-c", TOY_CFG, "--mode", "test", "--key", "labels.x.y"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "not in the plan" in out
        assert "matches wildcard 'labels.**' of 'labels'" in out
        assert "nothing demands" in out  # Labels is demand-gated, active in every mode

    def test_narrowed_label_present_in_fit_mode(self, capsys):
        rc = cli_main(["graph", "why", "-c", TOY_CFG, "--mode", "fit", "--key", "labels.x.y"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "producer:  labels (Labels)" in out
        assert "kind=label" in out
        assert "head" in out  # consumer


# broken config: quality error


class TestBrokenConfig:
    def test_typoed_key_fails_with_quality_error(self, capsys):
        assert cli_main(["graph", "validate", "-c", BROKEN_CFG]) == 1
        err = capsys.readouterr().err
        assert "ConnectivityError" in err
        assert "'head'" in err  # names the consumer
        assert "'embed.y'" in err  # names the missing key
        assert "did you mean" in err
        assert "embed.x" in err  # nearest-key suggestion
        # quality bar: suggestion AND availability AND a concrete fix
        assert "available keys:" in err
        assert "fix: correct the require in module 'head'" in err
