"""Tests for `salt.core.callbacks.GraphArtifacts` (split from test_callbacks.py)."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from salt.core.callbacks import GraphArtifacts
from salt.core.graph.spec import Mode, flatten_spec
from salt.core.saltmodule import SaltModule
from salt.tests._fixtures.gn2_fixture import write_parity_norm_dict
from salt.tests._fixtures.gn2v2_fixture import build_gn2v2_modules, gn2v2_sources
from salt.tests.unit.callbacks.conftest import LRS


_HAS_DOT = shutil.which("dot") is not None


# GraphArtifacts (design §4.4 run-dir artifacts)


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
        if _HAS_DOT:  # the image is rendered by the dot binary (DOT sidecar always)
            assert (tmp_path / "graph_fit.svg").stat().st_size > 0

    def test_test_artifacts_written(self, fitted_model, tmp_path):
        GraphArtifacts().on_test_start(stub_trainer(tmp_path), fitted_model)
        assert "plan [mode=TEST]" in (tmp_path / "plan_test.txt").read_text()
        assert (tmp_path / "graph_test.dot").exists()
        if _HAS_DOT:
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

    # test_writer_sinks_table_in_plan_test removed in W6c:
    # WriterCallback/TaskWriter were deleted with callback.py/modules.py.

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

    @pytest.mark.skipif(not _HAS_DOT, reason="graphviz `dot` binary not on PATH")
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
        from salt.tests.unit.test_main import make_cli
        from salt.utils.inputs import write_dummy_file

        nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
        write_parity_norm_dict(nd, cd)
        h5 = tmp_path / "pp_output_train.h5"
        write_dummy_file(h5, nd)
        schema = tmp_path / "schema.yaml"
        save_schema(dump_schema(h5), schema)
        cli = make_cli({"dir": tmp_path, "h5": h5, "nd": nd, "schema": schema})
        assert any(isinstance(cb, GraphArtifacts) for cb in cli.trainer.callbacks)
