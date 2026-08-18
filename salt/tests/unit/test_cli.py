"""Tests for the salt CLI (salt.cli)."""

import shutil
import textwrap
from functools import reduce
from operator import or_

import h5py
import numpy as np
import pytest

from salt.cli import main
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.schema import load_schema
from salt.tests._fixtures.gn2v2_test_config import small_config

# the in-repo test-scale GN2v2 trainer config (16-dim, no machine paths) — the
# static width-resolution plot path needs a real trainer config but no data file
_DUMMY_CFG = str(small_config())

# toy modules (no physics); instance names assigned by the CLI


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
    """Framework-style wildcard producer."""

    allow_wildcards = True


# config fixtures

TOY = "salt.tests.unit.test_cli.Toy"

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
# unconsumed there -> error-level deadcode finding
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


# graph validate


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


# graph deadcode


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
        # unconsumed preds.* in TEST is an error by default
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


# graph plan


class TestPlan:
    def test_table_order_and_binding_constraints(self, cfg, capsys):
        assert main(["graph", "plan", "-c", cfg(GOOD_CFG), "--mode", "fit"]) == 0
        out = capsys.readouterr().out
        assert "plan [mode=FIT]" in out
        assert "plan_hash=" in out
        lines = [line for line in out.splitlines() if ". " in line]
        names = [line.split(". ", 1)[1].split()[0] for line in lines]
        # topo ties broken by config declaration order: pred is declared
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
            class_path: salt.tests.unit.test_cli.WildToy
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


# graph plot


class TestPlot:
    def test_dot_always_emitted(self, cfg, tmp_path, capsys):
        # a .dot target short-circuits before any image render: the DOT source
        # is the output, so this holds with or without the dot binary
        out_path = tmp_path / "graph.dot"
        rc = main(["graph", "plot", "-c", cfg(GOOD_CFG), "--mode", "fit", "-o", str(out_path)])
        assert rc == 0
        assert out_path.exists()
        dot = out_path.read_text()
        # port-card nodes carry the keys; edges are deduped node -> node
        assert '"embed" [label=<' in dot
        assert "embed.x" in dot
        assert '"embed" -> "pred";' in dot
        assert '"<sources>" -> "embed";' in dot
        assert "#b5651d" in dot  # labels.x styled as label kind (orange)
        assert "#c0392b" in dot  # losses.total styled as loss kind (red)

    def test_pruned_module_rendered_dashed(self, cfg, tmp_path, capsys):
        out_path = tmp_path / "graph.dot"
        rc = main(["graph", "plot", "-c", cfg(PRUNED_CFG), "--mode", "test", "-o", str(out_path)])
        assert rc == 0
        dot = out_path.read_text()
        assert "(pruned)" in dot
        assert "style=dashed" in dot
        assert '"aux"' in dot

    @pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz `dot` not on PATH")
    def test_dot_binary_renders_png_and_pdf(self, cfg, tmp_path, capsys):
        # the dot binary is the sole renderer: a PNG -o emits PNG + a sibling
        # PDF + the .dot sidecar, all via `dot`
        out_path = tmp_path / "graph.png"
        rc = main(["graph", "plot", "-c", cfg(GOOD_CFG), "--mode", "fit", "-o", str(out_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "graphviz/dot" in out
        assert out_path.stat().st_size > 0
        assert out_path.with_suffix(".pdf").stat().st_size > 0
        assert out_path.with_suffix(".dot").exists()
        assert out_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_missing_dot_raises_actionable_error(self, cfg, tmp_path, capsys, monkeypatch):
        # `dot` absent -> a clear actionable error (no silent matplotlib fallback);
        # the DOT sidecar is still written for manual rendering
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        out_path = tmp_path / "graph.png"
        rc = main(["graph", "plot", "-c", cfg(GOOD_CFG), "--mode", "fit", "-o", str(out_path)])
        assert rc == 1
        captured = capsys.readouterr()
        out = captured.out + captured.err
        assert "dot" in out and "salt container" in out
        assert (tmp_path / "graph.dot").exists()
        assert not out_path.exists()

    def test_static_widths_write_concrete_feature_dims(self, tmp_path):
        # the plot path resolves the bind schema STATICALLY (no data file, no
        # batch run, no --probe) and annotates each port-card row with its
        # concrete FEATURE width while the data-dependent batch/sequence dims
        # stay symbolic. The placeholder --set only satisfies
        # the config parse — no norm_dict values are ever read.
        out_path = tmp_path / "graph.dot"
        rc = main([
            "graph", "plot",
            "-c", _DUMMY_CFG,
            "--mode", "fit",
            "--set", "model.modules.norm.init_args.norm_dict=unused.yaml",
            "-o", str(out_path),
        ])
        assert rc == 0
        dot = out_path.read_text()
        # concrete feature dims from the static resolution (the widths that build
        # the nn.Linear layers); B and the sequence dim T:tracks stay symbolic
        assert "normed.tracks" in dot
        assert "(B, T:tracks, 19)" in dot   # 19 input features, T:tracks symbolic
        assert "(B, T:tracks, 16)" in dot   # encoded.tracks: 16 embed width
        # NO symbolic feature dim leaked, and NO probe ran (no labeller line)
        assert "F:norm" not in dot
        assert "not labelled" not in dot


# graph why


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


# schema dump


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
