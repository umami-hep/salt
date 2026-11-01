"""Tests for `salt.callbacks.GraphArtifacts` (split from test_callbacks.py)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from salt.callbacks import GraphArtifacts
from salt.graph.render import graph_dot, pruned_modules
from salt.graph.spec import Mode, flatten_spec
from salt.model.saltmodule import SaltModule
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict
from salt.tests._fixtures.gn2v2_fixture import build_gn2v2_modules, gn2v2_sources
from salt.tests.unit.callbacks.conftest import LRS


_HAS_DOT = shutil.which("dot") is not None
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def assert_png_pdf(base: Path) -> None:
    """``<base>.png`` and ``<base>.pdf`` exist with real PNG / PDF headers."""
    assert Path(f"{base}.png").read_bytes()[:8] == _PNG_MAGIC
    assert Path(f"{base}.pdf").read_bytes()[:5] == b"%PDF-"


# GraphArtifacts (run-dir artifacts)


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
        if _HAS_DOT:  # the images are rendered by the dot binary (DOT sidecar always)
            assert_png_pdf(tmp_path / "graph_fit")

    def test_test_artifacts_written(self, fitted_model, tmp_path):
        GraphArtifacts().on_test_start(stub_trainer(tmp_path), fitted_model)
        assert "plan [mode=TEST]" in (tmp_path / "plan_test.txt").read_text()
        assert (tmp_path / "graph_test.dot").exists()
        if _HAS_DOT:
            assert_png_pdf(tmp_path / "graph_test")

    def test_test_artifacts_default_next_to_checkpoint(self, fitted_model, tmp_path):
        # with a known ckpt_path, the test-path default is the checkpoint dir
        # (where the eval H5 goes), not the log dir
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

    def test_resolved_io_yaml_written(self, fitted_model, tmp_path):
        # the machine-readable artifact
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

    def test_output_dir_option(self, fitted_model, tmp_path):
        out = tmp_path / "sub"
        GraphArtifacts(output_dir=str(out)).on_test_start(stub_trainer(tmp_path), fitted_model)
        assert (out / "plan_test.txt").exists()
        assert (out / "graph_test.dot").exists()
        assert not (tmp_path / "plan_test.txt").exists()
        if _HAS_DOT:
            assert_png_pdf(out / "graph_test")

    def test_dot_carries_widths(self, fitted_model, tmp_path):
        # the callback draws the statically resolved feature widths, like `salt graph plot`
        GraphArtifacts().on_fit_start(stub_trainer(tmp_path), fitted_model)
        dot = (tmp_path / "graph_fit.dot").read_text()
        assert "(B, T:tracks, 16)" in dot  # the 16-wide track embedding; T:tracks symbolic
        plan = fitted_model.plans[Mode.FIT]
        modules = fitted_model._graph_modules
        assert dot != graph_dot(plan, modules, pruned=pruned_modules(modules, plan))

    def test_missing_dot_warns_not_raises(self, fitted_model, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(shutil, "which", lambda *_args, **_kwargs: None)
        with caplog.at_level(logging.WARNING, logger="salt"):
            GraphArtifacts().on_fit_start(stub_trainer(tmp_path), fitted_model)
        assert (tmp_path / "graph_fit.dot").exists()
        assert not (tmp_path / "graph_fit.png").exists()
        assert not (tmp_path / "graph_fit.pdf").exists()
        messages = [record.getMessage() for record in caplog.records]
        assert any("`dot` not on PATH" in m and "dot -Tpng" in m for m in messages)

    def test_pruned_matches_plan_module_names(self, fitted_model, tmp_path):
        # a module missing from the TEST plan is drawn as a dashed "(pruned)" card
        # exactly when it is demand-pruned (has a TEST-active port); live steps never are
        GraphArtifacts().on_test_start(stub_trainer(tmp_path), fitted_model)
        dot = (tmp_path / "graph_test.dot").read_text()
        plan = fitted_model.plans[Mode.TEST]
        modules = fitted_model._graph_modules
        pruned = pruned_modules(modules, plan)
        assert set(pruned) <= set(modules) - set(plan.module_names)
        for name in modules:
            assert (f"<B>{name} (pruned)</B>" in dot) == (name in pruned)

    def test_non_rank_zero_writes_nothing(self, fitted_model, tmp_path):
        trainer = stub_trainer(tmp_path)
        trainer.is_global_zero = False
        GraphArtifacts().on_fit_start(trainer, fitted_model)
        assert list(tmp_path.iterdir()) == []

    def test_module_without_plans_warns_and_skips(self, tmp_path):
        with pytest.warns(UserWarning, match="no compiled plans"):
            GraphArtifacts().on_fit_start(stub_trainer(tmp_path), SimpleNamespace(plans={}))
        assert list(tmp_path.iterdir()) == []

    def test_registered_by_default_in_base(self, tmp_path):
        # base.yaml ships the artifacts entry; the CLI assembles it into
        # trainer.callbacks (run=False instantiation — no fit needed)
        from salt.schema import dump_schema, save_schema
        from salt.tests._fixtures.dummy_h5 import dummy_h5
        from salt.tests.unit.test_main import make_cli

        nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
        write_parity_norm_dict(nd, cd)
        h5, _, _ = dummy_h5(tmp_path)
        schema = tmp_path / "schema.yaml"
        save_schema(dump_schema(h5), schema)
        cli = make_cli({"dir": tmp_path, "h5": h5, "nd": nd, "schema": schema})
        assert any(isinstance(cb, GraphArtifacts) for cb in cli.trainer.callbacks)
