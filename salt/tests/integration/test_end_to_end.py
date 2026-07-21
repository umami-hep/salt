"""Toy end-to-end integration test for the graph kernel (validate/plan/execute/plot)."""

from pathlib import Path

import pytest
import torch

from salt.cli import load_config
from salt.cli import main as cli_main
from salt.graph.bundle import Bundle
from salt.graph.errors import AllModesDeadError
from salt.graph.executor import Executor
from salt.graph.planner import SINKS, SOURCES, Edge, compile_plan, deadcode
from salt.graph.spec import Mode
from salt.tests._fixtures.toys import ToyDead

CONFIG_DIR = Path(__file__).parent.parent / "_fixtures" / "configs"
TOY_CFG = str(CONFIG_DIR / "toy.yaml")
BROKEN_CFG = str(CONFIG_DIR / "toy_broken.yaml")


def _plan(cfg, mode):
    return compile_plan(cfg.modules, mode, cfg.sources, schema=cfg.schema, sinks=cfg.sinks)


def _deadcode(cfg, mode):
    return deadcode(cfg.modules, mode, cfg.sources, cfg.schema, cfg.sinks)


# validate (design §4.1)


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
        # schema is configured, so the only warnings are the known fit/val dead preds
        assert "cannot be checked statically" not in captured.err


# plan compilation (design §3.1): the loss path is fit/val-only


class TestPlans:
    def test_fit_plan_contains_loss_path(self):
        cfg = load_config(TOY_CFG)
        plan = _plan(cfg, Mode.FIT)
        # deterministic topo order: Kahn, ties broken by config declaration
        # order (§3.1) — toy.yaml declares source, embed, labels, head
        assert plan.module_names == ("source", "embed", "labels", "head")
        assert "losses.total" in plan.step("head").produces
        assert "labels.x" in plan.step("labels").produces  # narrowed from labels.*
        assert Edge("labels", "labels.x", "head") in plan.edges
        assert Edge("head", "losses.total", SINKS) in plan.edges
        assert Edge(SOURCES, "raw.x", "labels") in plan.edges
        assert plan.sources.keys() == {"raw.x"}

    @pytest.mark.parametrize("mode", [Mode.TEST, Mode.ONNX])
    def test_test_and_onnx_plans_have_no_loss_path(self, mode):
        cfg = load_config(TOY_CFG)
        plan = _plan(cfg, mode)
        assert "labels" not in plan.module_names
        produced = {key for step in plan.steps for key in step.produces}
        assert not any(key.startswith(("losses.", "labels.")) for key in produced)
        assert "preds.x" in produced
        assert Edge("head", "preds.x", SINKS) in plan.edges

    def test_writer_is_test_only(self):
        cfg = load_config(TOY_CFG)
        assert "writer" in _plan(cfg, Mode.TEST).module_names
        assert "writer" not in _plan(cfg, Mode.ONNX).module_names
        assert "writer" not in _plan(cfg, Mode.FIT).module_names

    def test_plan_hash_stable_across_loads(self):
        first, second = load_config(TOY_CFG), load_config(TOY_CFG)
        assert _plan(first, Mode.FIT).plan_hash == _plan(second, Mode.FIT).plan_hash
        assert _plan(first, Mode.FIT).plan_hash != _plan(first, Mode.TEST).plan_hash


# execution (design §3.2) — debug read tracking on


class TestExecution:
    def test_fit_plan_executes_loss_path(self):
        cfg = load_config(TOY_CFG)
        plan = _plan(cfg, Mode.FIT)
        bundle = Bundle({"raw": {"x": torch.randn(5, 8)}})
        out = Executor(plan).run(bundle, debug=True)
        assert out is bundle  # design §3.2: run returns the same bundle
        assert set(out.keys()) == {
            "raw.x",
            "inputs.x",
            "masks.x",
            "labels.x",
            "embed.x",
            "preds.x",
            "losses.total",
        }
        preds = out.get("preds.x")
        assert preds.shape == (5, 3)
        assert torch.allclose(preds.sum(dim=-1), torch.ones(5))  # softmax rows
        loss = out.get("losses.total")
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0  # cross entropy

    def test_test_plan_executes_and_writer_collects(self):
        cfg = load_config(TOY_CFG)
        plan = _plan(cfg, Mode.TEST)
        out = Executor(plan).run(Bundle({"raw": {"x": torch.randn(5, 8)}}), debug=True)
        assert "losses.total" not in out
        assert "labels.x" not in out
        writer = cfg.modules["writer"]
        assert len(writer.collected) == 1
        assert writer.collected[0].shape == (5, 3)

    def test_missing_source_raises_keyerror(self):
        cfg = load_config(TOY_CFG)
        plan = _plan(cfg, Mode.FIT)
        with pytest.raises(KeyError, match=r"raw\.x"):
            Executor(plan).run(Bundle())


# deadcode (design §4.2) and the all-modes-dead error (design §3.1 principle 10)


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
        # deadcode reports instead of raising (design §4.2)
        findings = deadcode(modules, Mode.FIT, cfg.sources, cfg.schema, cfg.sinks)
        assert ("deadend", "*") in [(f.module, f.key) for f in findings]


# plot (design §4.3)


class TestPlot:
    def test_fit_dot_contains_expected_edges(self, tmp_path, capsys):
        out_path = tmp_path / "graph.svg"
        assert cli_main(["graph", "plot", "-c", TOY_CFG, "--mode", "fit", "-o", str(out_path)]) == 0
        dot = (tmp_path / "graph.dot").read_text()
        assert "digraph salt_core_fit {" in dot
        # port-card nodes: each consumed/produced key is a card row with its shape
        assert '"source" [label=<' in dot
        assert '"head" [label=<' in dot
        assert "raw.x" in dot
        assert "(B, 8)" in dot
        assert "inputs.x" in dot
        assert "embed.x" in dot
        assert "(B, 16)" in dot
        assert "masks.x" in dot
        assert "(B)" in dot
        assert "labels.x" in dot
        assert "losses.total" in dot
        # deduped node -> node edges (no per-key edge labels)
        assert '"<sources>" -> "source";' in dot
        assert '"<sources>" -> "labels";' in dot
        assert '"source" -> "embed";' in dot
        assert '"embed" -> "head";' in dot
        assert '"labels" -> "head";' in dot
        assert '"head" -> "<sinks>";' in dot
        # kind-coloured rows: labels orange, losses red, preds blue
        assert "#b5651d" in dot  # label kind
        assert "#c0392b" in dot  # loss kind
        assert "#1f6fb2" in dot  # preds kind
        # writer is TEST-gated: absent from the FIT graph entirely
        assert '"writer"' not in dot


# why (design §3.1 debugging story)


class TestWhy:
    def test_fit_only_loss_explained_in_test_mode(self, capsys):
        rc = cli_main(["graph", "why", "-c", TOY_CFG, "--mode", "test", "--key", "losses.total"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "not in the plan" in out
        assert "mode-gated" in out
        assert "FIT/VAL" in out

    def test_fit_only_wildcard_label_explained_in_test_mode(self, capsys):
        rc = cli_main(["graph", "why", "-c", TOY_CFG, "--mode", "test", "--key", "labels.x"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "not in the plan" in out
        assert "matches wildcard 'labels.*' of 'labels'" in out
        assert "mode-gated" in out

    def test_narrowed_label_present_in_fit_mode(self, capsys):
        rc = cli_main(["graph", "why", "-c", TOY_CFG, "--mode", "fit", "--key", "labels.x"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "producer:  labels (ToyWildcardLabels)" in out
        assert "kind=label" in out
        assert "head" in out  # consumer


# broken config: §4.1-quality error


class TestBrokenConfig:
    def test_typoed_key_fails_with_quality_error(self, capsys):
        assert cli_main(["graph", "validate", "-c", BROKEN_CFG]) == 1
        err = capsys.readouterr().err
        assert "ConnectivityError" in err
        assert "'head'" in err  # names the consumer
        assert "'embed.y'" in err  # names the missing key
        assert "did you mean" in err
        assert "embed.x" in err  # nearest-key suggestion
        # §4.1 quality bar: suggestion AND availability AND a concrete fix
        assert "available keys:" in err
        assert "fix: correct the require in module 'head'" in err
