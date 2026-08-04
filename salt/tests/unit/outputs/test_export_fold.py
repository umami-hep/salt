"""The ONNX export contract folded onto `OnnxExportSink`, and its top-level alias."""

from __future__ import annotations

import warnings
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
    ExportOutput,
    resolve_export_config,
)
from salt.outputs.sinks.onnx.export import _merge_export_alias, _resolve_export_contract
from salt.outputs import OnnxExportSink
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict

pytestmark = pytest.mark.cpu_always

_CONFIGS = Path(__file__).parents[3] / "configs"
_DUMMY = _CONFIGS / "GN2/gn2v2-dummy.yaml"
_MASKFORMER = _CONFIGS / "MaskFormer.yaml"

_INPUTS = [
    {"port": "inputs.jets", "name": "jet_features"},
    {"port": "inputs.tracks", "name": "track_features", "sequence": True, "dyn_axis": "n_tracks"},
]


class _StubCLI:
    """The two config reads `_resolve_export_contract` makes off a parsed run config."""

    def __init__(self, export=None, name="salt"):
        self.config_init = {"export": export, "name": name}

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


# ---------------------------------------------------------------------------
# the deprecated top-level export: block
# ---------------------------------------------------------------------------


def test_top_level_block_still_fills_the_sink_and_warns():
    """The alias fills every key the sink left unset — and says it is deprecated."""
    sink = OnnxExportSink()
    block = ExportConfig(
        model_name="GN2v2",
        inputs=[ExportInput(port="inputs.jets", name="jet_features")],
        track_selection="r22loose",
        rename={"pu": "plight"},
        combine=[ExportCombine(name="pbc", inputs={"pb": 1.0})],
    )
    with pytest.warns(DeprecationWarning, match="top-level `export:` block is deprecated"):
        resolved = _resolve_export_contract(_StubCLI(block, name="run"), sink)
    assert resolved.model_name == "GN2v2"
    assert [entry.port for entry in resolved.inputs] == ["inputs.jets"]
    assert resolved.track_selection == "r22loose"
    assert resolved.rename == {"pu": "plight"}
    assert [entry.name for entry in resolved.combine] == ["pbc"]


def test_no_block_resolves_the_sink_alone_without_warning():
    """A config without the block resolves the sink's own contract, warning-free."""
    sink = OnnxExportSink(model_name="GN2v2", inputs=_INPUTS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = _resolve_export_contract(_StubCLI(None, name="run"), sink)
    assert resolved.model_name == "GN2v2"
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


def test_cli_name_override_wins_over_both_homes():
    """``salt export -n NAME`` overrides the sink and the alias block alike."""
    sink = OnnxExportSink(inputs=_INPUTS, model_name="FromSink")
    resolved = _resolve_export_contract(_StubCLI(None, name="run"), sink, "OverrideName")
    assert resolved.model_name == "OverrideName"


@pytest.mark.parametrize(
    ("key", "block_kwargs", "sink_kwargs"),
    [
        ("model_name", {"model_name": "FromBlock"}, {"model_name": "FromSink"}),
        (
            "inputs",
            {"inputs": [ExportInput(port="inputs.jets", name="a")]},
            {"inputs": _INPUTS},
        ),
        ("track_selection", {"track_selection": "r22loose"}, {"track_selection": "ip3d"}),
        ("rename", {"rename": {"pu": "a"}}, {"rename": {"pu": "b"}}),
        (
            "combine",
            {"combine": [ExportCombine(name="x", inputs={"pb": 1.0})]},
            {"combine": [{"name": "y", "inputs": {"pb": 1.0}}]},
        ),
    ],
)
def test_key_in_both_homes_is_refused(key, block_kwargs, sink_kwargs):
    """A key carried by BOTH homes errors, naming the key and both homes."""
    sink = OnnxExportSink(**sink_kwargs)
    with pytest.warns(DeprecationWarning), pytest.raises(ConfigError) as excinfo:
        _merge_export_alias(_StubCLI(ExportConfig(**block_kwargs), name="run"), sink)
    message = str(excinfo.value)
    assert f"export.{key}" in message
    assert "OnnxExportSink" in message
    assert "top-level" in message


def test_declared_export_outputs_stays_a_hard_error():
    """The retired ``export.outputs`` carrier is refused through the alias too."""
    block = ExportConfig(outputs=[ExportOutput(port="preds.jets.a", names=["pb"])])
    with pytest.raises(ConfigError, match="export.outputs was REMOVED"):
        _merge_export_alias(_StubCLI(block, name="run"), OnnxExportSink())


# ---------------------------------------------------------------------------
# the shipped configs: MaskFormer migrated, gn2v2-dummy on the alias
# ---------------------------------------------------------------------------


def test_maskformer_carries_the_contract_on_the_sink():
    """`MaskFormer.yaml` is the worked proof of the new home: no top-level block."""
    raw = yaml.safe_load(_MASKFORMER.read_text())
    assert "export" not in raw
    init_args = raw["outputs"]["onnx_export"]["init_args"]
    assert init_args["model_name"] == "MFv2"
    assert [entry["port"] for entry in init_args["inputs"]] == ["inputs.jets", "inputs.tracks"]


def test_gn2v2_dummy_keeps_the_deprecated_block():
    """`gn2v2-dummy.yaml` stays on the alias — the live proof the window is open."""
    raw = yaml.safe_load(_DUMMY.read_text())
    assert raw["export"]["model_name"] == "GN2v2dummy"
    assert "onnx_export" not in raw["outputs"]  # the sink is INJECTED, not declared


def test_gn2v2_dummy_resolves_its_contract_through_the_alias():
    """The alias reaches the INJECTED sink: block -> sink -> resolved contract.

    The regression test for the deprecation window — the sink `salt export`
    finds is the same object `_inject_command_sinks` registered, so the
    top-level block still lands on it.
    """
    from salt.cli import _static_onnx_export_sink
    from salt.config_utils import disable_logger_in_config
    from salt.main import SaltCLI

    cli = SaltCLI(
        args=[
            "--config",
            disable_logger_in_config(str(_DUMMY)),
            "--model.modules.norm.init_args.norm_dict=unused.yaml",
        ],
        run=False,
    )
    sink = _static_onnx_export_sink(cli)
    assert isinstance(sink, OnnxExportSink)  # injected by _inject_command_sinks
    assert sink.inputs == []  # the contract is NOT on the sink
    assert sink.model_name is None
    with pytest.warns(DeprecationWarning, match="top-level `export:` block is deprecated"):
        resolved = _resolve_export_contract(cli, sink)
    assert resolved.model_name == "GN2v2dummy"
    assert [entry.port for entry in resolved.inputs] == ["inputs.jets", "inputs.tracks"]
    assert [entry.dyn_axis for entry in resolved.inputs] == [None, "n_tracks"]
    # the merge landed on the sink itself, not on a copy
    assert sink.model_name == "GN2v2dummy"


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
