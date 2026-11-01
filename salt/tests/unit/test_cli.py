"""Tests for the salt CLI (salt.cli)."""

import re
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

import salt.cli
from salt.cli import _parse_trainer_cli, main
from salt.graph.errors import ConfigError
from salt.outputs.sinks.onnx.export import _run_free_cli
from salt.schema import load_schema
from salt.tests._fixtures.gn2v2_test_config import small_config

# the in-repo test-scale GN2v2 trainer config (16-dim, no machine paths) — the
# static width-resolution plot path needs a real trainer config but no data file
_DUMMY_CFG = str(small_config())

# the toy trainer config (no physics, no data file) and its typo'd twin
_CONFIGS = Path(__file__).parent.parent / "_fixtures" / "configs"
TOY_CFG = str(_CONFIGS / "toy_trainer.yaml")
BROKEN_CFG = str(_CONFIGS / "toy_trainer_broken.yaml")
TOYS = "salt.tests._fixtures.toys"


def _model_modules(raw):
    return raw["model"]["init_args"]["modules"]


def _pruned(raw):
    """Add 'aux' (ToyAuxLoss): LossSum takes losses.aux in FIT/VAL only -> pruned in TEST/ONNX."""
    _model_modules(raw)["aux"] = {"class_path": f"{TOYS}.ToyAuxLoss"}


def _dead_preds(raw):
    """Give head a second port preds.y that nothing consumes (the writer takes preds.x)."""
    _model_modules(raw)["head"]["init_args"]["extra_preds"] = "y"


def _no_schema(raw):
    """A reader without a schema artifact."""
    raw["data"]["modules"]["reader"].setdefault("init_args", {})["schema"] = False


@pytest.fixture
def variant(tmp_path):
    """Write a copy of the toy trainer config edited in place by `edit`; returns its path."""

    def write(edit, name="cfg.yaml"):
        raw = yaml.safe_load(Path(TOY_CFG).read_text())
        edit(raw)
        path = tmp_path / name
        path.write_text(yaml.safe_dump(raw, sort_keys=False))
        return str(path)

    return write


# graph validate


class TestValidate:
    def test_happy_path_all_modes(self, capsys):
        assert main(["graph", "validate", "-c", TOY_CFG]) == 0
        out = capsys.readouterr().out
        for mode in ("FIT", "VAL", "TEST", "ONNX"):
            assert f"OK [mode={mode}]" in out
        assert "plan_hash=" in out

    def test_single_mode(self, capsys):
        assert main(["graph", "validate", "-c", TOY_CFG, "--mode", "test"]) == 0
        out = capsys.readouterr().out
        assert "OK [mode=TEST]" in out
        assert "OK [mode=FIT]" not in out

    def test_missing_producer_fails_with_quality_message(self, capsys):
        assert main(["graph", "validate", "-c", BROKEN_CFG]) == 1
        err = capsys.readouterr().err
        assert "ConnectivityError" in err
        assert "'head'" in err  # names the consumer
        assert "embed.y" in err  # names the missing key
        assert "embed.x" in err  # did-you-mean suggestion

    def test_no_schema_warns_but_passes(self, variant, capsys):
        assert main(["graph", "validate", "-c", variant(_no_schema)]) == 0
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "cannot be checked statically" in err

    def test_strict_promotes_warnings_to_errors(self, variant, capsys):
        # FIT only: the schema warning is the one promoted (ONNX adds its own)
        cfg = variant(_no_schema)
        assert main(["graph", "validate", "-c", cfg, "--strict", "--mode", "fit"]) == 1
        err = capsys.readouterr().err
        assert "cannot be checked statically" in err
        assert "promoted to errors" in err

    def test_strict_passes_with_schema_and_no_dead_outputs(self, capsys):
        # FIT: the unconsumed preds.x is info-level, never promoted (ONNX always
        # warns that a config without an OnnxExportSink leaves its contract unchecked)
        assert main(["graph", "validate", "-c", TOY_CFG, "--strict", "--mode", "fit"]) == 0

    def test_missing_config_file(self, capsys):
        assert main(["graph", "validate", "-c", "/nonexistent/cfg.yaml"]) == 1
        assert "not found" in capsys.readouterr().err

    def test_bad_class_path(self, variant, capsys):
        def bad(raw):
            _model_modules(raw)["embed"]["class_path"] = "not_a_real_pkg.Embed"

        assert main(["graph", "validate", "-c", variant(bad)]) == 1
        err = capsys.readouterr().err
        assert re.search(r"failed to (parse|instantiate) through the salt surface", err)

    def test_toy_format_config_rejected(self, tmp_path, capsys):
        path = tmp_path / "toy.yaml"
        path.write_text("modules:\n  embed:\n    class_path: pkg.Embed\n")
        assert main(["graph", "validate", "-c", str(path)]) == 1
        err = capsys.readouterr().err
        assert "is not a salt trainer config" in err
        assert "format was removed" in err


# graph deadcode


class TestDeadcode:
    def test_clean_mode_reports_ok(self, capsys):
        assert main(["graph", "deadcode", "-c", TOY_CFG, "--mode", "test"]) == 0
        out = capsys.readouterr().out
        assert "[deadcode] mode=TEST: OK" in out

    def test_pruned_module_reported(self, variant, capsys):
        assert main(["graph", "deadcode", "-c", variant(_pruned), "--mode", "test"]) == 0
        out = capsys.readouterr().out
        assert "[deadcode] mode=TEST:" in out
        assert "'aux'" in out
        assert "pruned" in out

    def test_all_modes_by_default(self, variant, capsys):
        assert main(["graph", "deadcode", "-c", variant(_pruned)]) == 0
        out = capsys.readouterr().out
        for mode in ("FIT", "VAL", "TEST", "ONNX"):
            assert f"mode={mode}" in out

    def test_unconsumed_preds_in_test_is_error_and_exits_nonzero(self, variant, capsys):
        # unconsumed preds.* in TEST is an error by default
        assert main(["graph", "deadcode", "-c", variant(_dead_preds), "--mode", "test"]) == 1
        out = capsys.readouterr().out
        assert "ERROR 'preds.y' (head)" in out
        assert "never persisted" in out

    def test_same_port_is_info_exit_zero_outside_test(self, variant, capsys):
        assert main(["graph", "deadcode", "-c", variant(_dead_preds), "--mode", "fit"]) == 0
        assert "INFO 'preds.y' (head)" in capsys.readouterr().out

    def test_validate_fails_on_error_level_deadcode(self, variant, capsys):
        assert main(["graph", "validate", "-c", variant(_dead_preds), "--mode", "test"]) == 1
        err = capsys.readouterr().err
        assert "ERROR:" in err
        assert "preds.y" in err
        assert "error-level deadcode" in err


# graph plan


def _step_lines(out):
    return [line for line in out.splitlines() if re.match(r"\s*\d+\. ", line)]


class TestPlan:
    def test_table_order_and_binding_constraints(self, capsys):
        assert main(["graph", "plan", "-c", TOY_CFG, "--mode", "fit"]) == 0
        out = capsys.readouterr().out
        assert "plan [mode=FIT]" in out
        assert "plan_hash=" in out
        lines = _step_lines(out)
        names = [line.split(". ", 1)[1].split()[0] for line in lines]
        # topo ties broken by declaration order: data.modules, then model.modules
        assert names == ["reader", "features", "labels", "embed", "head", "loss"]
        head_line = lines[names.index("head")]
        assert "after embed" in head_line
        assert "needs embed.x" in head_line
        assert "sources: (none)" in out  # the reader is the source node

    def test_reader_step_has_no_inputs(self, capsys):
        assert main(["graph", "plan", "-c", TOY_CFG, "--mode", "fit"]) == 0
        first = _step_lines(capsys.readouterr().out)[0]
        assert "reader" in first
        assert "after (no inputs)" in first

    def test_narrowed_wildcards_printed(self, capsys):
        assert main(["graph", "plan", "-c", TOY_CFG, "--mode", "fit"]) == 0
        out = capsys.readouterr().out
        assert "narrowed wildcards [mode=FIT]:" in out
        assert "labels: labels.x.y" in out


# graph plot


class TestPlot:
    def test_dot_always_emitted(self, tmp_path, capsys):
        # a .dot target short-circuits before any image render: the DOT source
        # is the output, so this holds with or without the dot binary
        out_path = tmp_path / "graph.dot"
        rc = main(["graph", "plot", "-c", TOY_CFG, "--mode", "fit", "-o", str(out_path)])
        assert rc == 0
        assert out_path.exists()
        dot = out_path.read_text()
        # port-card nodes carry the keys; edges are deduped node -> node
        assert '"embed" [label=<' in dot
        assert "embed.x" in dot
        assert '"embed" -> "head";' in dot
        assert '"features" -> "embed";' in dot
        assert "#b5651d" in dot  # labels.x.y styled as label kind (orange)
        assert "#c0392b" in dot  # losses.x / loss.total styled as loss kind (red)

    def test_pruned_module_rendered_dashed(self, variant, tmp_path, capsys):
        out_path = tmp_path / "graph.dot"
        rc = main(["graph", "plot", "-c", variant(_pruned), "--mode", "test", "-o", str(out_path)])
        assert rc == 0
        dot = out_path.read_text()
        assert "(pruned)" in dot
        assert "style=dashed" in dot
        assert '"aux"' in dot

    @pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz `dot` not on PATH")
    def test_dot_binary_renders_png_and_pdf(self, tmp_path, capsys):
        # the dot binary is the sole renderer: a PNG -o emits PNG + a sibling
        # PDF + the .dot sidecar, all via `dot`
        out_path = tmp_path / "graph.png"
        rc = main(["graph", "plot", "-c", TOY_CFG, "--mode", "fit", "-o", str(out_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "graphviz/dot" in out
        assert out_path.stat().st_size > 0
        assert out_path.with_suffix(".pdf").stat().st_size > 0
        assert out_path.with_suffix(".dot").exists()
        assert out_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_missing_dot_raises_actionable_error(self, tmp_path, capsys, monkeypatch):
        # `dot` absent -> a clear actionable error (no silent matplotlib fallback);
        # the DOT sidecar is still written for manual rendering
        monkeypatch.setattr(shutil, "which", lambda *_args, **_kwargs: None)
        out_path = tmp_path / "graph.png"
        rc = main(["graph", "plot", "-c", TOY_CFG, "--mode", "fit", "-o", str(out_path)])
        assert rc == 1
        captured = capsys.readouterr()
        out = captured.out + captured.err
        assert "dot" in out and "salt container" in out
        assert (tmp_path / "graph.dot").exists()
        assert not out_path.exists()

    def test_other_suffix_is_config_error(self, tmp_path, capsys):
        # -o is a stem: only .dot/.png/.pdf/no suffix; .svg would get PNG bytes, so it is refused
        out_path = tmp_path / "graph.svg"
        rc = main(["graph", "plot", "-c", TOY_CFG, "--mode", "fit", "-o", str(out_path)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "ConfigError" in err
        assert "'.svg'" in err
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz `dot` not on PATH")
    def test_no_suffix_renders_stem(self, tmp_path, capsys):
        stem = tmp_path / "graph"
        rc = main(["graph", "plot", "-c", TOY_CFG, "--mode", "fit", "-o", str(stem)])
        assert rc == 0
        assert "digraph" in (tmp_path / "graph.dot").read_text()
        assert (tmp_path / "graph.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert (tmp_path / "graph.pdf").read_bytes()[:5] == b"%PDF-"
        assert not stem.exists()

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
    def test_present_key(self, capsys):
        rc = main(["graph", "why", "-c", TOY_CFG, "--mode", "fit", "--key", "embed.x"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "producer:  embed" in out
        assert "head" in out  # consumer
        assert "kind=data" in out

    def test_reader_key(self, capsys):
        rc = main(["graph", "why", "-c", TOY_CFG, "--mode", "fit", "--key", "raw.x"])
        assert rc == 0
        assert "producer:  reader (ToyReader)" in capsys.readouterr().out

    def test_pruned_key_explained(self, variant, capsys):
        cfg = variant(_pruned)
        rc = main(["graph", "why", "-c", cfg, "--mode", "test", "--key", "losses.aux"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "not in the plan" in out
        assert "demand-pruned" in out
        assert "'aux'" in out
        assert "fix:" in out

    def test_mode_gated_key_explained(self, capsys):
        rc = main(["graph", "why", "-c", TOY_CFG, "--mode", "test", "--key", "loss.total"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "mode-gated" in out
        assert "FIT" in out

    def test_nonexistent_key_fails_with_suggestion(self, capsys):
        rc = main(["graph", "why", "-c", TOY_CFG, "--mode", "fit", "--key", "embed.nope"])
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

    def test_dumped_schema_usable_in_validate(self, variant, tmp_path, capsys):
        # end-to-end: dump a schema, hand it to a real H5StructuredReader, validate clean
        h5_path = tmp_path / "train.h5"
        fields = [(f"f{i}", "f4") for i in range(8)] + [("y", "i4")]
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("x", data=np.zeros(5, dtype=np.dtype(fields)))
        schema_path = tmp_path / "schema.yaml"
        assert main(["schema", "dump", str(h5_path), "-o", str(schema_path)]) == 0
        capsys.readouterr()

        def h5_reader(raw):
            raw["data"]["modules"]["reader"] = {
                "class_path": "salt.data.H5StructuredReader",
                "init_args": {"groups": {"x": {"global_object": True}}, "schema": str(schema_path)},
            }

        cfg = variant(h5_reader)
        assert main(["graph", "validate", "-c", cfg, "--strict", "--mode", "fit"]) == 0


_BUILDERS = [("cli", _parse_trainer_cli), ("export", _run_free_cli)]


class TestRunFreeCliBuilders:
    """Both run-free builders share one core; each keeps its own error text."""

    @pytest.mark.parametrize(("name", "build"), _BUILDERS, ids=[n for n, _ in _BUILDERS])
    def test_bad_set_entry_rejected_before_parsing(self, name, build):
        with pytest.raises(ConfigError, match=r"--set entries must be KEY=VALUE, got 'novalue'"):
            build([small_config()], ["model.modules.norm.init_args.norm_dict=unused.yaml", "novalue"])

    @pytest.mark.parametrize(("name", "build"), _BUILDERS, ids=[n for n, _ in _BUILDERS])
    def test_parser_exit_translates_to_config_error(self, name, build, monkeypatch):
        def boom(*_a, **_k):
            raise SystemExit(2)

        monkeypatch.setattr(salt.cli, "_build_run_free_cli", boom)
        expected = {
            "cli": r"trainer config .* failed to parse through the salt surface \(parser exit 2;.*Required init_args left as overrides",
            "export": r"run config\(s\) .* failed to parse through the salt surface \(parser exit 2;.*Supply required init_args data-free via --set if needed",
        }[name]
        with pytest.raises(ConfigError, match=expected):
            build([small_config()], [])

    def test_value_error_translation_is_asymmetric(self, monkeypatch):
        def boom(*_a, **_k):
            raise ValueError("boom")

        monkeypatch.setattr(salt.cli, "_build_run_free_cli", boom)
        with pytest.raises(ConfigError, match=r"failed to instantiate through the salt surface:\nboom"):
            _parse_trainer_cli([small_config()], [])
        with pytest.raises(ValueError, match="boom"):
            _run_free_cli([small_config()], [])
