"""Tests for the data-free norm-dict preflight."""

from __future__ import annotations

import pytest
import yaml

from salt.graph.errors import ConfigError
from salt.graph.spec import Mode, flatten_spec
from salt.model.modules.norm import Normaliser
from salt.model.saltmodule import SaltModule
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict
from salt.tests._fixtures.gn2v2_fixture import build_gn2v2_modules, gn2v2_sources

LRS = {"initial": 1e-4, "max": 1e-3, "end": 1e-5, "pct_start": 0.1}


@pytest.fixture
def norm_dict(tmp_path):
    nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
    write_parity_norm_dict(nd, cd)
    return nd


def make_normaliser(norm_dict, streams=("jets", "tracks")) -> Normaliser:
    norm = Normaliser(norm_dict=norm_dict, streams=list(streams), global_object="jets")
    norm.name = "norm"
    return norm


class TestNormaliserPreflight:
    def test_good_dict_passes_unbound(self, norm_dict):
        make_normaliser(norm_dict).preflight()  # must not raise

    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            make_normaliser(tmp_path / "nope.yaml").preflight()

    def test_invalid_yaml(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("{::not yaml::")
        with pytest.raises(ConfigError, match="not valid YAML"):
            make_normaliser(path).preflight()

    def test_non_mapping(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n")
        with pytest.raises(ConfigError, match="must be a mapping"):
            make_normaliser(path).preflight()

    def test_missing_stream(self, norm_dict):
        with pytest.raises(ConfigError, match="missing input type 'muons'"):
            make_normaliser(norm_dict, streams=("jets", "muons")).preflight()

    def test_bound_missing_variable(self, tmp_path, norm_dict):
        # bind via a SaltModule compile so _fields carries the feature list,
        # then strip one variable from the dict on disk
        modules = build_gn2v2_modules(norm_dict)
        model = SaltModule(modules, lrs=LRS)
        model.compile_mode(Mode.FIT, flatten_spec(gn2v2_sources()))
        model._ensure_bound()  # noqa: SLF001 - exercising the setup path piecewise
        payload = yaml.safe_load(norm_dict.read_text())
        dropped = next(iter(payload["tracks"]))
        del payload["tracks"][dropped]
        norm_dict.write_text(yaml.safe_dump(payload))
        with pytest.raises(ConfigError, match=dropped):
            modules["norm"].preflight()

    def test_bound_zero_std(self, norm_dict):
        payload = yaml.safe_load(norm_dict.read_text())
        variable = next(iter(payload["jets"]))
        payload["jets"][variable]["std"] = 0.0
        norm_dict.write_text(yaml.safe_dump(payload))
        modules = build_gn2v2_modules(norm_dict)
        model = SaltModule(modules, lrs=LRS)
        model.compile_mode(Mode.FIT, flatten_spec(gn2v2_sources()))
        model._ensure_bound()  # noqa: SLF001 - exercising the setup path piecewise
        with pytest.raises(ConfigError, match="zero standard deviation"):
            modules["norm"].preflight()


class TestSaltModulePreflights:
    @pytest.mark.integration  # runs a real Lightning .fit() (imports integration helpers)
    def test_fresh_fit_fails_fast_on_bad_norm_dict(self, tmp_path):
        # the real trainer path: fresh fit + nonexistent norm dict ->
        # ConfigError from the preflight at fit start (NOT a bare
        # FileNotFoundError from inside materialise)
        from salt.schema import dump_schema, save_schema
        from salt.tests.integration.test_saltmodule import build_datamodule, build_model, make_trainer
        from salt.testing.inputs import write_dummy_file

        nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
        write_parity_norm_dict(nd, cd)
        h5 = tmp_path / "pp_output_train.h5"
        write_dummy_file(h5, nd)
        schema = tmp_path / "schema.yaml"
        save_schema(dump_schema(h5), schema)
        data = {"dir": tmp_path, "h5": h5, "nd": nd, "schema": schema}
        model = build_model(data, norm_dict=tmp_path / "does_not_exist.yaml")
        with pytest.raises(ConfigError, match="preflight.*not found"):
            make_trainer(max_epochs=1).fit(model, build_datamodule(data))

    def test_run_preflights_propagates(self, tmp_path, norm_dict):
        modules = build_gn2v2_modules(norm_dict)
        model = SaltModule(modules, lrs=LRS)
        model.compile_mode(Mode.FIT, flatten_spec(gn2v2_sources()))
        model._ensure_bound()  # noqa: SLF001 - exercising the setup path piecewise
        model._run_preflights()  # noqa: SLF001 - good dict passes
        norm_dict.unlink()
        with pytest.raises(ConfigError, match="not found"):
            model._run_preflights()  # noqa: SLF001

    def test_validate_reports_preflight_as_warning(self, tmp_path, capsys):
        # the documented data-free flow: validate with a placeholder norm
        # dict must stay exit 0 (warning only), --strict promotes it
        from salt.cli import main as cli_main
        from salt.main import CONFIG_DIR
        from salt.schema import dump_schema, save_schema
        from salt.testing.inputs import write_dummy_file

        nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
        write_parity_norm_dict(nd, cd)
        h5 = tmp_path / "pp_output_train.h5"
        write_dummy_file(h5, nd)
        schema = tmp_path / "schema.yaml"
        save_schema(dump_schema(h5), schema)
        args = [
            "graph",
            "validate",
            "-c",
            str(CONFIG_DIR / "GN2/gn2v2-dummy.yaml"),
            "--mode",
            "fit",
            "--set",
            f"data.modules.reader.init_args.schema={schema}",
            "--set",
            "model.modules.norm.init_args.norm_dict=unused.yaml",
        ]
        assert cli_main(args) == 0
        err = capsys.readouterr().err
        assert "preflight of module 'norm'" in err
        assert "not found" in err
        assert cli_main([*args, "--strict"]) == 1
