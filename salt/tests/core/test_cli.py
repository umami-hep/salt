"""Tests for the salt2 CLI (salt.core.cli — design §4.1-§4.4, §2.6).

Calls ``main([...])`` directly (no subprocess, no installed script) with toy
modules defined in this module and tmp YAML configs. The toys are referenced
by ``class_path: salt.tests.core.test_cli.<Class>`` — the M1 loader imports
them via importlib.
"""

import textwrap
from functools import reduce
from operator import or_
from pathlib import Path

import h5py
import numpy as np
import pytest

from salt.core.cli import main
from salt.core.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.core.schema import load_schema

# the in-repo test-scale GN2v2 trainer config (16-dim, no machine paths) — the
# synthetic --probe path needs a real trainer config but no data file
_DUMMY_CFG = str(Path(__file__).parent.parent.parent / "core" / "configs" / "gn2v2-dummy.yaml")

# ---------------------------------------------------------------------------
# toy modules (no physics — M1 scope); instance names assigned by the CLI
# ---------------------------------------------------------------------------


def _spec(cfg):
    """Build a TensorSpec from a plain YAML-style mapping."""
    kwargs = dict(cfg or {})
    if "modes" in kwargs:
        names = kwargs["modes"]
        if isinstance(names, str):
            names = [names]
        kwargs["modes"] = reduce(or_, (Mode[name.upper()] for name in names))
    if "shape" in kwargs:
        kwargs["shape"] = tuple(kwargs["shape"])
    return TensorSpec(**kwargs)


class Toy:
    """Minimal GraphModule with ports given as plain {key: spec-kwargs} dicts."""

    def __init__(self, requires=None, produces=None):
        self.name = "unnamed"  # overwritten by the CLI loader from the config key
        self._io = IO(
            requires=unflatten_spec({k: _spec(v) for k, v in (requires or {}).items()}),
            produces=unflatten_spec({k: _spec(v) for k, v in (produces or {}).items()}),
        )

    def declare_io(self, mode):
        return self._io


class WildToy(Toy):
    """Framework-style wildcard producer (design §2.2)."""

    allow_wildcards = True


# ---------------------------------------------------------------------------
# config fixtures
# ---------------------------------------------------------------------------

TOY = "salt.tests.core.test_cli.Toy"

GOOD_CFG = f"""
modules:
  embed:
    class_path: {TOY}
    init_args:
      requires: {{inputs.x: {{}}}}
      produces: {{embed.x: {{}}}}
  pred:
    class_path: {TOY}
    init_args:
      requires: {{embed.x: {{}}}}
      produces: {{preds.x: {{}}}}
  labeller:
    class_path: {TOY}
    init_args:
      produces: {{labels.x: {{kind: label, modes: [fit, val]}}}}
  loss:
    class_path: {TOY}
    init_args:
      requires:
        preds.x: {{modes: [fit, val]}}
        labels.x: {{kind: label, modes: [fit, val]}}
      produces: {{losses.total: {{kind: loss, modes: [fit, val]}}}}
sources:
  inputs.x: {{}}
sinks:
  fit: [losses.total]
  val: [losses.total]
  test: [preds.x]
  onnx: [preds.x]
"""

# 'aux' reaches a sink only in FIT -> demand-pruned in VAL/TEST/ONNX
PRUNED_CFG = f"""
modules:
  embed:
    class_path: {TOY}
    init_args:
      requires: {{inputs.x: {{}}}}
      produces: {{embed.x: {{}}}}
  pred:
    class_path: {TOY}
    init_args:
      requires: {{embed.x: {{}}}}
      produces: {{preds.x: {{}}}}
  aux:
    class_path: {TOY}
    init_args:
      requires: {{embed.x: {{}}}}
      produces: {{aux.x: {{}}}}
sources:
  inputs.x: {{}}
sinks:
  fit: [preds.x, aux.x]
  val: [preds.x]
  test: [preds.x]
  onnx: [preds.x]
"""

BAD_CFG = f"""
modules:
  embed:
    class_path: {TOY}
    init_args:
      requires: {{inputs.x: {{}}}}
      produces: {{embed.x: {{}}}}
  pred:
    class_path: {TOY}
    init_args:
      requires: {{embed.y: {{}}}}
      produces: {{preds.x: {{}}}}
sources:
  inputs.x: {{}}
sinks: [preds.x]
"""

# 'pred' survives TEST (aux.keep reaches the sink) but its preds.x port is
# unconsumed there -> error-level deadcode finding (design §4.2)
DEAD_PREDS_CFG = f"""
modules:
  pred:
    class_path: {TOY}
    init_args:
      requires: {{inputs.x: {{}}}}
      produces: {{preds.x: {{}}, aux.keep: {{}}}}
sources:
  inputs.x: {{}}
sinks:
  fit: [preds.x, aux.keep]
  val: [preds.x, aux.keep]
  test: [aux.keep]
  onnx: [preds.x, aux.keep]
"""


@pytest.fixture
def cfg(tmp_path):
    def write(text, name="cfg.yaml"):
        path = tmp_path / name
        path.write_text(textwrap.dedent(text))
        return str(path)

    return write


# ---------------------------------------------------------------------------
# graph validate (design §4.1)
# ---------------------------------------------------------------------------


class TestValidate:
    def test_happy_path_all_modes(self, cfg, capsys):
        assert main(["graph", "validate", "-c", cfg(GOOD_CFG)]) == 0
        out = capsys.readouterr().out
        for mode in ("FIT", "VAL", "TEST", "ONNX"):
            assert f"OK [mode={mode}]" in out
        assert "plan_hash=" in out

    def test_single_mode(self, cfg, capsys):
        assert main(["graph", "validate", "-c", cfg(GOOD_CFG), "--mode", "test"]) == 0
        out = capsys.readouterr().out
        assert "OK [mode=TEST]" in out
        assert "OK [mode=FIT]" not in out

    def test_missing_producer_fails_with_quality_message(self, cfg, capsys):
        assert main(["graph", "validate", "-c", cfg(BAD_CFG)]) == 1
        err = capsys.readouterr().err
        assert "ConnectivityError" in err
        assert "'pred'" in err  # names the consumer
        assert "embed.y" in err  # names the missing key
        assert "embed.x" in err  # did-you-mean suggestion

    def test_no_schema_warns_but_passes(self, cfg, capsys):
        assert main(["graph", "validate", "-c", cfg(GOOD_CFG)]) == 0
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "cannot be checked statically" in err

    def test_strict_promotes_warnings_to_errors(self, cfg, capsys):
        assert main(["graph", "validate", "-c", cfg(GOOD_CFG), "--strict"]) == 1
        err = capsys.readouterr().err
        assert "promoted to errors" in err

    def test_strict_passes_with_schema_and_no_dead_outputs(self, cfg, capsys):
        text = GOOD_CFG + "schema: [inputs.x]\n"
        assert main(["graph", "validate", "-c", cfg(text), "--strict"]) == 0

    def test_missing_config_file(self, capsys):
        assert main(["graph", "validate", "-c", "/nonexistent/cfg.yaml"]) == 1
        assert "not found" in capsys.readouterr().err

    def test_bad_class_path(self, cfg, capsys):
        text = """
        modules:
          embed:
            class_path: not_a_real_pkg.Embed
        sources: [inputs.x]
        """
        assert main(["graph", "validate", "-c", cfg(text)]) == 1
        assert "cannot import" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# graph deadcode (design §4.2)
# ---------------------------------------------------------------------------


class TestDeadcode:
    def test_clean_mode_reports_ok(self, cfg, capsys):
        assert main(["graph", "deadcode", "-c", cfg(PRUNED_CFG), "--mode", "fit"]) == 0
        out = capsys.readouterr().out
        assert "[deadcode] mode=FIT: OK" in out

    def test_pruned_module_reported(self, cfg, capsys):
        assert main(["graph", "deadcode", "-c", cfg(PRUNED_CFG), "--mode", "test"]) == 0
        out = capsys.readouterr().out
        assert "[deadcode] mode=TEST:" in out
        assert "'aux'" in out
        assert "pruned" in out

    def test_all_modes_by_default(self, cfg, capsys):
        assert main(["graph", "deadcode", "-c", cfg(PRUNED_CFG)]) == 0
        out = capsys.readouterr().out
        for mode in ("FIT", "VAL", "TEST", "ONNX"):
            assert f"mode={mode}" in out

    def test_unconsumed_preds_in_test_is_error_and_exits_nonzero(self, cfg, capsys):
        # design §4.2: unconsumed preds.* in TEST is an error by default
        assert main(["graph", "deadcode", "-c", cfg(DEAD_PREDS_CFG), "--mode", "test"]) == 1
        out = capsys.readouterr().out
        assert "ERROR 'preds.x' (pred)" in out
        assert "never persisted" in out

    def test_same_port_is_warning_exit_zero_outside_test(self, cfg, capsys):
        assert main(["graph", "deadcode", "-c", cfg(DEAD_PREDS_CFG), "--mode", "fit"]) == 0
        assert "[deadcode] mode=FIT: OK" in capsys.readouterr().out

    def test_validate_fails_on_error_level_deadcode(self, cfg, capsys):
        assert main(["graph", "validate", "-c", cfg(DEAD_PREDS_CFG), "--mode", "test"]) == 1
        err = capsys.readouterr().err
        assert "ERROR:" in err
        assert "preds.x" in err
        assert "error-level deadcode" in err


# ---------------------------------------------------------------------------
# graph plan (design §4.4 table)
# ---------------------------------------------------------------------------


class TestPlan:
    def test_table_order_and_binding_constraints(self, cfg, capsys):
        assert main(["graph", "plan", "-c", cfg(GOOD_CFG), "--mode", "fit"]) == 0
        out = capsys.readouterr().out
        assert "plan [mode=FIT]" in out
        assert "plan_hash=" in out
        lines = [line for line in out.splitlines() if ". " in line]
        names = [line.split(". ", 1)[1].split()[0] for line in lines]
        # topo ties broken by config declaration order (§3.1): pred is declared
        # before labeller in GOOD_CFG, so it runs first
        assert names == ["embed", "pred", "labeller", "loss"]
        pred_line = lines[names.index("pred")]
        assert "after embed" in pred_line
        assert "needs embed.x" in pred_line
        assert "sources: inputs.x" in out

    def test_sourced_step_says_after_sources(self, cfg, capsys):
        assert main(["graph", "plan", "-c", cfg(GOOD_CFG), "--mode", "fit"]) == 0
        out = capsys.readouterr().out
        embed_line = next(line for line in out.splitlines() if "embed" in line and "1." in line)
        assert "<sources>" in embed_line

    def test_narrowed_wildcards_printed(self, cfg, tmp_path, capsys):
        text = f"""
        modules:
          reader:
            class_path: salt.tests.core.test_cli.WildToy
            init_args:
              produces: {{"inputs.*": {{}}}}
          use:
            class_path: {TOY}
            init_args:
              requires: {{inputs.x: {{}}}}
              produces: {{preds.x: {{}}}}
        schema: [inputs.x]
        """
        assert main(["graph", "plan", "-c", cfg(text), "--mode", "fit"]) == 0
        out = capsys.readouterr().out
        assert "narrowed wildcards [mode=FIT]:" in out
        assert "reader: inputs.x" in out


# ---------------------------------------------------------------------------
# graph plot (design §4.3)
# ---------------------------------------------------------------------------


class TestPlot:
    def test_dot_always_emitted(self, cfg, tmp_path, capsys):
        out_path = tmp_path / "graph.svg"
        rc = main(["graph", "plot", "-c", cfg(GOOD_CFG), "--mode", "fit", "-o", str(out_path)])
        assert rc == 0
        dot_path = tmp_path / "graph.dot"
        assert dot_path.exists()
        dot = dot_path.read_text()
        # port-card nodes carry the keys; edges are deduped node -> node
        assert '"embed" [label=<' in dot
        assert "embed.x" in dot
        assert '"embed" -> "pred";' in dot
        assert '"<sources>" -> "embed";' in dot
        assert "#b5651d" in dot  # labels.x styled as label kind (orange)
        assert "#c0392b" in dot  # losses.total styled as loss kind (red)
        captured = capsys.readouterr().out
        assert out_path.exists() or "graphviz" in captured

    def test_pruned_module_rendered_dashed(self, cfg, tmp_path, capsys):
        out_path = tmp_path / "graph.dot"
        rc = main(["graph", "plot", "-c", cfg(PRUNED_CFG), "--mode", "test", "-o", str(out_path)])
        assert rc == 0
        dot = out_path.read_text()
        assert "(pruned)" in dot
        assert "style=dashed" in dot
        assert '"aux"' in dot

    def test_matplotlib_is_the_primary_renderer(self, cfg, tmp_path, capsys, monkeypatch):
        # graphviz absence is irrelevant: matplotlib renders the image
        import salt.core.cli as cli_mod

        monkeypatch.setattr(cli_mod, "graphviz", None)
        out_path = tmp_path / "graph.svg"
        rc = main(["graph", "plot", "-c", cfg(GOOD_CFG), "--mode", "fit", "-o", str(out_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "matplotlib" in out
        assert out_path.exists()
        assert (tmp_path / "graph.dot").exists()

    def test_says_so_when_no_renderer_available(self, cfg, tmp_path, capsys, monkeypatch):
        # matplotlib AND graphviz unavailable -> DOT stays, with a render hint
        import salt.core.cli as cli_mod

        def _no_mpl(*args, **kwargs):
            raise ImportError("matplotlib disabled for the test")

        monkeypatch.setattr(cli_mod, "render_graph", _no_mpl)
        monkeypatch.setattr(cli_mod, "graphviz", None)
        out_path = tmp_path / "graph.svg"
        rc = main(["graph", "plot", "-c", cfg(GOOD_CFG), "--mode", "fit", "-o", str(out_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "not importable" in out
        assert "graph.dot" in out
        assert not out_path.exists()

    def test_synthetic_probe_writes_concrete_shapes(self, tmp_path, capsys):
        # --probe alone (no data file) synthesises a batch from the reader schema
        # and annotates the DOT with concrete numeric shapes (design §4.3). Uses
        # the in-repo test-scale GN2v2 config; the synthetic norm_dict is injected
        # by the probe (the placeholder --set only satisfies the config parse).
        out_path = tmp_path / "graph.dot"
        rc = main([
            "graph", "plot",
            "-c", _DUMMY_CFG,
            "--mode", "fit",
            "--probe",
            "--set", "model.modules.norm.init_args.norm_dict=unused.yaml",
            "-o", str(out_path),
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "probed" in out and "synthetic batch" in out
        dot = out_path.read_text()
        # inputs/embed/labels now carry CONCRETE numeric shapes (B==16)
        assert "(16, 40, 19)" in dot  # inputs.tracks: 19 features, 40 tokens
        assert "inputs.tracks" in dot
        assert "labels.jets.flavour_label" in dot
        # at least one embed.* row shows a concrete [B, T, dim] triple
        assert "(16, 40, 16)" in dot  # embed.tracks at the 16-dim test scale


# ---------------------------------------------------------------------------
# graph why (design §3.1 debugging story)
# ---------------------------------------------------------------------------


class TestWhy:
    def test_present_key(self, cfg, capsys):
        rc = main(["graph", "why", "-c", cfg(GOOD_CFG), "--mode", "fit", "--key", "embed.x"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "producer:  embed" in out
        assert "pred" in out  # consumer
        assert "kind=data" in out

    def test_source_key(self, cfg, capsys):
        rc = main(["graph", "why", "-c", cfg(GOOD_CFG), "--mode", "fit", "--key", "inputs.x"])
        assert rc == 0
        assert "<sources>" in capsys.readouterr().out

    def test_pruned_key_explained(self, cfg, capsys):
        rc = main(["graph", "why", "-c", cfg(PRUNED_CFG), "--mode", "test", "--key", "aux.x"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "not in the plan" in out
        assert "demand-pruned" in out
        assert "'aux'" in out
        assert "fix:" in out

    def test_mode_gated_key_explained(self, cfg, capsys):
        rc = main(["graph", "why", "-c", cfg(GOOD_CFG), "--mode", "test", "--key", "labels.x"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "mode-gated" in out
        assert "FIT" in out

    def test_nonexistent_key_fails_with_suggestion(self, cfg, capsys):
        rc = main(["graph", "why", "-c", cfg(GOOD_CFG), "--mode", "fit", "--key", "embed.nope"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "no module or source produces it" in err
        assert "embed.x" in err  # did-you-mean


# ---------------------------------------------------------------------------
# schema dump (design §2.6)
# ---------------------------------------------------------------------------


class TestSchemaDump:
    def test_dump_round_trip(self, tmp_path, capsys):
        h5_path = tmp_path / "train.h5"
        jets = np.zeros(5, dtype=np.dtype([("pt", "f4"), ("flavour_label", "i4")]))
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("jets", data=jets)
            f["jets"].attrs["flavour_label"] = ["bjets", "cjets", "ujets"]
        out = tmp_path / "schema.yaml"
        assert main(["schema", "dump", str(h5_path), "-o", str(out)]) == 0
        assert "wrote schema for 1 group(s)" in capsys.readouterr().out
        schema = load_schema(out)
        assert schema.groups["jets"].fields == {"pt": "float32", "flavour_label": "int32"}
        assert schema.groups["jets"].attrs["flavour_label"] == ["bjets", "cjets", "ujets"]

    def test_missing_input_file(self, tmp_path, capsys):
        out = tmp_path / "schema.yaml"
        assert main(["schema", "dump", str(tmp_path / "nope.h5"), "-o", str(out)]) == 1
        assert "not found" in capsys.readouterr().err

    def test_dumped_schema_usable_in_validate(self, cfg, tmp_path, capsys):
        # end-to-end: dump a schema, reference it from a config, validate clean
        h5_path = tmp_path / "train.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("inputs", data=np.zeros(5, dtype=np.dtype([("x", "f4")])))
        assert main(["schema", "dump", str(h5_path), "-o", str(tmp_path / "schema.yaml")]) == 0
        capsys.readouterr()
        text = GOOD_CFG + "schema: schema.yaml\n"
        assert main(["graph", "validate", "-c", cfg(text), "--strict"]) == 0
