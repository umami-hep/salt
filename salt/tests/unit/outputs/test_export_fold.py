"""The ONNX export contract, assembled on `OnnxExportSink` — its sole config home."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from salt.cli import load_config
from salt.graph.errors import ConfigError
from salt.graph.spec import Mode
from salt.outputs.sinks.onnx.config import (
    ExportCombine,
    ExportConfig,
    ExportInput,
    resolve_export_config,
)
from salt.outputs.sinks.onnx.export import _resolve_export_contract
from salt.outputs import OnnxExportSink
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict
from salt.tests._fixtures.gn2v2_test_config import small_config

pytestmark = pytest.mark.cpu_always

_CONFIGS = Path(__file__).parents[3] / "configs"
_DUMMY = small_config()
_MASKFORMER = _CONFIGS / "MaskFormer.yaml"
_GN2V2_OPENDATA = _CONFIGS / "gn2v2-opendata.yaml"

_INPUTS = [
    {"port": "inputs.jets", "name": "jet_features"},
    {"port": "inputs.tracks", "name": "track_features", "sequence": True, "dyn_axis": "n_tracks"},
]


class _StubCLI:
    """The config read `_resolve_export_contract` makes off a parsed run config: its ``name``."""

    def __init__(self, name="salt"):
        self.config_init = {"name": name}

    def _get(self, config, key):
        """Mirror `SaltCLI._get` over a plain dict namespace."""
        return config.get(key)


# ---------------------------------------------------------------------------
# the sink IS the export contract
# ---------------------------------------------------------------------------


def test_sink_assembles_an_equivalent_export_config():
    """The sink's `export_config` equals resolving the same keys as an `ExportConfig`."""
    sink = OnnxExportSink(
        model_name="GN2v2",
        inputs=_INPUTS,
        track_selection="r22loose",
        rename={"pu": "plight"},
        combine=[{"name": "pbc", "inputs": {"pb": 1.0, "pc": 1.0}}],
    )
    expected = resolve_export_config(
        ExportConfig(
            model_name="GN2v2",
            track_selection="r22loose",
            inputs=[ExportInput(**entry) for entry in _INPUTS],
            rename={"pu": "plight"},
            combine=[ExportCombine(name="pbc", inputs={"pb": 1.0, "pc": 1.0})],
        ),
        "some_run",
    )
    assert sink.export_config("some_run") == expected


def test_sink_coerces_dataclasses_and_mappings_alike():
    """`inputs:`/`combine:` accept dataclasses as well as plain config mappings."""
    from_dataclasses = OnnxExportSink(
        inputs=[ExportInput(port="inputs.jets", name="jet_features")],
        combine=[ExportCombine(name="pbc", inputs={"pb": 1.0})],
    )
    from_mappings = OnnxExportSink(
        inputs=[{"port": "inputs.jets", "name": "jet_features"}],
        combine=[{"name": "pbc", "inputs": {"pb": 1.0}}],
    )
    assert from_dataclasses.inputs == from_mappings.inputs
    assert from_dataclasses.combine == from_mappings.combine


def test_model_name_defaults_to_the_sanitised_run_name():
    """An unset `model_name:` resolves to the run name with ``_``/``-`` stripped."""
    assert OnnxExportSink(inputs=_INPUTS).export_config("GN2v2_dummy").model_name == "GN2v2dummy"


def test_no_input_signature_names_the_sink_home():
    """A contract with no input signature points at the sink's ``inputs:``."""
    with pytest.raises(ConfigError) as excinfo:
        OnnxExportSink(model_name="GN2v2").export_config("run")
    message = str(excinfo.value)
    assert "OnnxExportSink" in message
    assert "init_args.inputs" in message


def test_resolve_export_contract_reads_the_sink_and_the_run_name():
    """`_resolve_export_contract` resolves off the sink's own fields plus the run name."""
    sink = OnnxExportSink(model_name="GN2v2", inputs=_INPUTS)
    resolved = _resolve_export_contract(_StubCLI(name="run"), sink)
    assert resolved.model_name == "GN2v2"
    assert [entry.port for entry in resolved.inputs] == ["inputs.jets", "inputs.tracks"]


def test_cli_name_override_wins_over_the_sink():
    """``salt export -n NAME`` overrides the sink's own `model_name`."""
    sink = OnnxExportSink(inputs=_INPUTS, model_name="FromSink")
    resolved = _resolve_export_contract(_StubCLI(name="run"), sink, "OverrideName")
    assert resolved.model_name == "OverrideName"


# ---------------------------------------------------------------------------
# the shipped configs: the sink is the only home
# ---------------------------------------------------------------------------


def test_maskformer_carries_the_contract_on_the_sink():
    """`MaskFormer.yaml` names the export contract on the sink — no top-level block."""
    raw = yaml.safe_load(_MASKFORMER.read_text())
    assert "export" not in raw
    init_args = raw["outputs"]["onnx_export"]["init_args"]
    assert init_args["model_name"] == "MFv2"
    assert [entry["port"] for entry in init_args["inputs"]] == ["inputs.jets", "inputs.tracks"]


def test_gn2v2_opendata_carries_the_contract_on_the_sink():
    """`gn2v2-opendata.yaml` also declares the sink directly — no top-level block."""
    raw = yaml.safe_load(_GN2V2_OPENDATA.read_text())
    assert "export" not in raw
    init_args = raw["outputs"]["onnx_export"]["init_args"]
    assert init_args["model_name"] == "GN2v2opendata"
    assert [entry["port"] for entry in init_args["inputs"]] == ["inputs.jets", "inputs.tracks"]


def test_gn2v2_dummy_carries_the_contract_on_the_sink():
    """The derived GN2v2-dummy test fixture inherits the sink form from the shipped config."""
    raw = yaml.safe_load(_DUMMY.read_text())
    assert "export" not in raw
    init_args = raw["outputs"]["onnx_export"]["init_args"]
    assert init_args["model_name"] == "GN2v2dummy"
    assert [entry["port"] for entry in init_args["inputs"]] == ["inputs.jets", "inputs.tracks"]


def test_gn2v2_dummy_resolves_its_contract_from_the_declared_sink():
    """The section-declared sink resolves directly: config -> sink -> resolved contract."""
    from salt.main import SaltCLI
    from salt.utils.config_utils import disable_logger_in_config

    cli = SaltCLI(
        args=[
            "--config",
            disable_logger_in_config(str(_DUMMY)),
            "--model.modules.norm.init_args.norm_dict=unused.yaml",
        ],
        run=False,
    )
    sink = cli._get(cli.config_init, "outputs")["onnx_export"]  # noqa: SLF001 - main.py precedent
    assert isinstance(sink, OnnxExportSink)
    assert sink.model_name == "GN2v2dummy"
    resolved = _resolve_export_contract(cli, sink)
    assert resolved.model_name == "GN2v2dummy"
    assert [entry.port for entry in resolved.inputs] == ["inputs.jets", "inputs.tracks"]
    assert [entry.dyn_axis for entry in resolved.inputs] == [None, "n_tracks"]


@pytest.fixture(scope="module")
def maskformer_cfg(tmp_path_factory):
    """`MaskFormer.yaml` through the real CLI surface (no data touched)."""
    base = tmp_path_factory.mktemp("mf_export_fold")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    with open(nd_path) as fh:
        nd = yaml.safe_load(fh)
    # the one jets variable this config declares beyond the shared parity fixture
    jets = nd.setdefault("jets", {})
    if "mass" not in jets:
        idx = len(jets)
        jets["mass"] = {"mean": round(0.1 * (idx + 1), 6), "std": round(1.0 + 0.05 * (idx + 1), 6)}
    with open(nd_path, "w") as fh:
        yaml.dump(nd, fh, sort_keys=False)
    return load_config(_MASKFORMER, [f"model.modules.norm.init_args.norm_dict={nd_path}"])


def test_maskformer_resolves_its_contract_from_the_sink(maskformer_cfg):
    """`MaskFormer.yaml`'s ONNX contract resolves off the section-declared sink alone."""
    assert maskformer_cfg.mode_errors.get(Mode.ONNX) is None
    sink = maskformer_cfg.modules["onnx_export"]
    assert isinstance(sink, OnnxExportSink)
    resolved = sink.export_config("MaskFormer")
    assert resolved.model_name == "MFv2"
    assert [entry.port for entry in resolved.inputs] == ["inputs.jets", "inputs.tracks"]
    assert [entry.name for entry in resolved.inputs] == ["jet_features", "track_features"]
    assert [entry.dyn_axis for entry in resolved.inputs] == [None, "n_tracks"]
