"""Tests for the ``salt2`` config surface / ``salt.core.main`` (design §5, §5.3).

Everything runs through the REAL CLI classes (`Salt2CLI` / `main`), not toy
parsers: the shipped ``gn2v2-dummy.yaml`` is parsed with its documented
required path overrides over a tmp dummy file, ``--print_config``
round-trips, config-file deep-merge adds a fourth task with siblings
surviving, ``--model.modules.X=null`` deletes, dotted overrides reach
``init_args``, `instantiate_classes` produces the `SaltModule`, the
callbacks-dict assembly lands in ``trainer.callbacks``, and a 2-step
``salt2 fit`` smoke runs on CPU (gate G2 surface). The ``graph``/``schema``
subcommands keep dispatching to the M1 tooling.

Parse/instantiate assertions use ``run=False`` (same parser surface, no
trainer run); the fit smoke uses ``main([...])`` with ``run=True``.

This file merges two former test modules that both exercise ``salt.core.main``:

1. the ``salt2`` CLI config surface (formerly ``test_salt2_cli.py``);
2. the Wave-1 ``--class_dict`` fan-out (formerly ``test_wave1_fanout.py``) —
   see the "Wave-1 class_dict fan-out" section below; its helpers/fixtures are
   ``wave1_*``-prefixed to avoid colliding with the CLI-surface ones above.
   ``norm_dict`` is NOT fanned out: it is the `Normaliser` module's own config,
   set on ``model.modules.norm.init_args.norm_dict`` directly — covered by the
   ``TestNormDictOnModule`` section.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary

from salt.core.callbacks import Checkpoint, ProgressBar
from salt.core.config_utils import disable_logger_in_config
from salt.core.data import GraphDataModule
from salt.core.main import CONFIG_DIR, Salt2CLI, main
from salt.core.nn.tasks import ClassificationTaskModule
from salt.core.saltmodule import SaltModule
from salt.core.schema import dump_schema, save_schema
from salt.tests._fixtures.gn2_fixture import write_parity_norm_dict
from salt.utils.inputs import write_dummy_file

DUMMY_CFG = CONFIG_DIR / "gn2v2-dummy.yaml"
OPENDATA_CFG = CONFIG_DIR / "gn2v2-opendata.yaml"
TOY_GRAPH_CFG = Path(__file__).parent.parent / "_fixtures" / "configs" / "toy.yaml"
GN2V2_MODULES = {
    "norm",
    "track_embed",
    "concat",
    "encoder",
    "split",
    "pool",
    "jets_classification",
    "track_origin",
    "track_vertexing",
    "loss",
    # W6c: jet_probs (ClassProbs) is the shared ONNX+TEST conversion producer;
    # track_origin_probs (SeqClassProbs) is the TEST-eval probability producer.
    # track_origin_index (SeqClassIndex) and track_vertex_index (VertexUnionFind)
    # are retired from model.modules in this wave.
    "jet_probs",
    "track_origin_probs",
}

FOURTH_TASK_YAML = """
model:
  init_args:
    modules:
      track_type:
        class_path: salt.core.nn.tasks.ClassificationTaskModule
        init_args:
          stream: tracks
          context: pooled.global
          label: ftagTruthOriginLabel
          class_names: [a, b, c]
          dense: {hidden_layers: [16], activation: ReLU}
"""

NULL_VERTEXING_YAML = """
model:
  init_args:
    modules:
      track_vertexing: null
"""


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    # tmp dummy H5 + parity norm dict + schema artifact (NO machine paths in
    # the committed YAML — the documented required overrides, design §5)
    base = tmp_path_factory.mktemp("salt2_cli")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    h5_path = base / "pp_output_train.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


def required_overrides(data) -> list[str]:
    """The gn2v2-dummy.yaml documented required overrides (its header)."""
    return [
        f"--data.train_file={data['h5']}",
        f"--data.val_file={data['h5']}",
        f"--data.modules.reader.init_args.schema={data['schema']}",
        f"--model.modules.norm.init_args.norm_dict={data['nd']}",
    ]


def make_cli(data, extra: list[str] | None = None, config: Path = DUMMY_CFG) -> Salt2CLI:
    """Parse + instantiate (run=False) through the real CLI surface.

    The config is modified to disable trainer.logger (keyless test envs fail
    during instantiate_classes with "Comet.ml requires an API key").
    """
    cfg = disable_logger_in_config(str(config))
    return Salt2CLI(
        args=["--config", cfg, *required_overrides(data), *(extra or [])],
        run=False,
    )


def write_yaml(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


# ---------------------------------------------------------------------------
# parse + instantiate (gn2v2-dummy.yaml is the shipped worked config, §5.1)
# ---------------------------------------------------------------------------


class TestParseAndInstantiate:
    def test_dummy_config_instantiates_saltmodule(self, data):
        cli = make_cli(data)
        assert isinstance(cli.model, SaltModule)
        assert isinstance(cli.datamodule, GraphDataModule)
        assert set(cli.model.net.keys()) == GN2V2_MODULES
        # instance names were assigned from the config dict keys (design §2.2)
        assert cli.model.net["encoder"].name == "encoder"
        # data modules in YAML order: reader -> features -> labels
        assert str(cli.datamodule.train_file) == str(data["h5"])

    def test_name_linked_into_model(self, data):
        # the single surviving link of the v1 CLI glue (design §5)
        cli = make_cli(data)
        assert cli.config.name == "GN2v2_dummy"
        assert cli.model.name == "GN2v2_dummy"

    def test_opendata_config_parses(self, data):
        # same model at open-data scale; file paths are required overrides
        # (real-data runs are the gates experiment's job — no /data here)
        cli = make_cli(data, config=OPENDATA_CFG)
        assert isinstance(cli.model, SaltModule)
        # plan 34 W34.4b: the open-data config is RELOCATED onto the outputs:
        # section + dumb sinks — the plan-29/31 conversion PRODUCERS (jet_probs/
        # track_origin_index/track_vertex_index) are RETIRED from model.modules
        # (the get_output fold replaces them). model.net carries the model modules
        # (nets + tasks + loss) PLUS the composed section GRAPH-node writers
        # (run_tasks + pad_mask; inputs_copy is manifest-only, not an nn.Module so
        # not in net). No conversion producers remain.
        non_producer = GN2V2_MODULES - {"jet_probs", "track_origin_index", "track_vertex_index", "track_origin_probs"}
        assert set(cli.model.net.keys()) == non_producer | {"run_tasks", "pad_mask"}
        jets_task = cli.model.net["jets_classification"]
        assert isinstance(jets_task, ClassificationTaskModule)
        assert list(jets_task.class_names) == ["bjets", "cjets", "ujets", "taujets"]
        # the plan-34 outputs: section is composed onto the model (W34.4b)
        section = cli.model._output_section  # noqa: SLF001
        assert set(section) == {"inputs_copy", "run_tasks", "pad_mask"}
        assert section["run_tasks"].is_run_task_output()


# ---------------------------------------------------------------------------
# --print_config round-trip (spike capability 2 at the real surface)
# ---------------------------------------------------------------------------


class TestPrintConfig:
    def test_round_trip(self, data, tmp_path, capsys):
        with pytest.raises(SystemExit) as excinfo:
            make_cli(data, extra=["--print_config"])
        assert excinfo.value.code == 0
        printed = capsys.readouterr().out
        # full class_path/init_args blocks with defaults made explicit
        assert "class_path: salt.core.SaltModule" in printed
        assert "class_path: salt.core.nn.TransformerEncoder" in printed
        assert "torch-math" in printed

        # feeding the printed config back yields the same parsed namespaces
        cli_orig = make_cli(data)
        printed_path = write_yaml(tmp_path, "printed.yaml", printed)
        cli_again = Salt2CLI(args=["--config", printed_path], run=False)
        assert cli_again.config.model == cli_orig.config.model
        assert cli_again.config.data == cli_orig.config.data
        assert cli_again.config.callbacks == cli_orig.config.callbacks


# ---------------------------------------------------------------------------
# deep-merge across config files (design §5.3, DeepMergeParser)
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_override_file_adds_fourth_task(self, data, tmp_path):
        override = write_yaml(tmp_path, "add_task.yaml", FOURTH_TASK_YAML)
        cli = make_cli(data, extra=["--config", override])
        # the new entry merged in; ALL siblings survive (the §5.3 semantics
        # that native jsonargparse loses — dict leaves replace wholesale)
        assert set(cli.model.net.keys()) == GN2V2_MODULES | {"track_type"}
        task = cli.model.net["track_type"]
        assert isinstance(task, ClassificationTaskModule)
        # sibling init_args of the model itself survive the partial restate
        assert cli.model.lrs["max"] == pytest.approx(1.0e-3)
        # the un-restated sibling modules keep their config
        assert cli.model.net["track_origin"].weight == pytest.approx(0.5)

    def test_override_file_updates_one_init_arg(self, data, tmp_path):
        override = write_yaml(
            tmp_path,
            "update_enc.yaml",
            "model:\n  init_args:\n    modules:\n      encoder:\n        init_args:\n"
            "          num_layers: 3\n",
        )
        cli = make_cli(data, extra=["--config", override])
        enc_cfg = cli.config.model.init_args.modules["encoder"]
        # class_path inherited, other init_args survive, the one key updated
        assert enc_cfg["class_path"] == "salt.core.nn.TransformerEncoder"
        assert enc_cfg["init_args"]["num_layers"] == 3
        assert enc_cfg["init_args"]["dim"] == 16
        assert set(cli.model.net.keys()) == GN2V2_MODULES  # siblings survive


# ---------------------------------------------------------------------------
# null-deletion (design §5.3: parse-to-None + assembly-time filtering)
# ---------------------------------------------------------------------------


class TestNullDeletion:
    def test_cli_dotted_null_deletes_module(self, data):
        cli = make_cli(data, extra=["--model.modules.track_vertexing=null"])
        assert "track_vertexing" not in cli.model.net
        assert set(cli.model.net.keys()) == GN2V2_MODULES - {"track_vertexing"}

    def test_config_file_null_deletes_module(self, data, tmp_path):
        override = write_yaml(tmp_path, "null_task.yaml", NULL_VERTEXING_YAML)
        cli = make_cli(data, extra=["--config", override])
        assert "track_vertexing" not in cli.model.net
        assert set(cli.model.net.keys()) == GN2V2_MODULES - {"track_vertexing"}

    def test_data_module_null_deletes(self, data):
        # labels can be dropped in TEST-style configs (design §5.3 symmetry);
        # parse-level check only — a labels-less FIT would fail at compile.
        # The deprecated --data.train_file alias synthesises an implicit
        # InputSamples (plan-25 W3.A / O-ALIAS-WINDOW), so it joins _modules;
        # a VDS is then auto-injected alongside it (plan-25 W3.B), so it too
        # joins _modules.
        cli = make_cli(data, extra=["--data.modules.labels=null"])
        dm_modules = cli.datamodule._modules  # noqa: SLF001 - assembly result
        assert set(dm_modules) == {"reader", "features", "input_samples", "vds"}


# ---------------------------------------------------------------------------
# dotted CLI overrides into init_args (spike capability 4 at the real surface)
# ---------------------------------------------------------------------------


class TestDottedOverrides:
    def test_short_form_into_module_init_args(self, data):
        cli = make_cli(data, extra=["--model.modules.encoder.init_args.num_layers=3"])
        enc_cfg = cli.config.model.init_args.modules["encoder"]
        assert enc_cfg["init_args"]["num_layers"] == 3
        assert enc_cfg["init_args"]["dim"] == 16  # untouched init_args survive
        # the instantiated module composed a 3-layer v1 Transformer
        assert len(cli.model.net["encoder"].encoder.layers) == 3

    def test_long_form_into_module_init_args(self, data):
        cli = make_cli(data, extra=["--model.init_args.modules.track_origin.init_args.weight=0.25"])
        assert cli.model.net["track_origin"].weight == pytest.approx(0.25)

    def test_data_module_init_args_override(self, data):
        cli = make_cli(data, extra=["--data.modules.reader.init_args.num=200"])
        assert cli.datamodule._reader_proto.num == 200  # noqa: SLF001 - prototype config


# ---------------------------------------------------------------------------
# callbacks: dict mechanics + assembly into trainer.callbacks (design §5.3)
# ---------------------------------------------------------------------------


class TestCallbacksDict:
    def test_base2_defaults_assembled(self, data):
        # base2.yaml ships the salt.core.callbacks.Checkpoint port (a
        # ModelCheckpoint subclass) + ProgressBar + ModelSummary (D2)
        cli = make_cli(data)
        assert any(isinstance(cb, Checkpoint) for cb in cli.trainer.callbacks)
        assert any(isinstance(cb, ProgressBar) for cb in cli.trainer.callbacks)
        assert any(isinstance(cb, ModelSummary) for cb in cli.trainer.callbacks)
        ckpt = next(cb for cb in cli.trainer.callbacks if isinstance(cb, ModelCheckpoint))
        assert ckpt.monitor == "val/loss"  # base2.yaml monitor_loss default
        assert ckpt.save_top_k == -1  # the v1 Checkpoint keeps every epoch

    def test_one_key_override(self, data):
        # monitor_loss is the new D2 Checkpoint arg (it sets BOTH the monitor
        # and the filename loss tag)
        cli = make_cli(data, extra=["--callbacks.checkpoint.init_args.monitor_loss=train/loss"])
        ckpt = next(cb for cb in cli.trainer.callbacks if isinstance(cb, ModelCheckpoint))
        assert ckpt.monitor == "train/loss"
        assert ckpt.filename == "epoch={epoch:03d}-loss={train/loss:.5f}"
        # the sibling dict entry survives the one-key override
        assert any(isinstance(cb, ModelSummary) for cb in cli.trainer.callbacks)

    def test_null_deletes_callback(self, data):
        cli = make_cli(data, extra=["--callbacks.checkpoint=null"])
        # no configured checkpoint left (Lightning may add its bare default,
        # which has no monitor)
        assert not any(getattr(cb, "monitor", None) == "val/loss" for cb in cli.trainer.callbacks)
        assert any(isinstance(cb, ModelSummary) for cb in cli.trainer.callbacks)



# ---------------------------------------------------------------------------
# FIT/VAL callback-declared sinks via the STATIC graph tooling (M5 D-prereq;
# design §3.1 454-456, §3.4 667-671) — the runtime path is covered in
# test_saltmodule.py::TestCallbackSinks; here the static `salt2 graph` path
# (load_config / `salt2 graph validate`) must see the SAME FIT/VAL sinks.
# ---------------------------------------------------------------------------

CONFMAT_CALLBACK_YAML = """
callbacks:
  confmat:
    class_path: salt.core.callbacks.ConfusionMatrix
    init_args:
      task_name: jets_classification
"""


class TestStaticFitValCallbackSinks:
    def test_validate_fit_with_callback_passes(self, data, tmp_path):
        # a configured ConfusionMatrix must not break `salt2 graph validate
        # --mode fit` (it declares preds/labels the task already keeps alive)
        from salt.core.main import main as graph_main

        override = write_yaml(tmp_path, "confmat.yaml", CONFMAT_CALLBACK_YAML)
        rc = graph_main([
            "graph",
            "validate",
            "-c",
            str(DUMMY_CFG),
            "-c",
            override,
            "--mode",
            "fit",
            "--set",
            f"data.train_file={data['h5']}",
            "--set",
            f"data.val_file={data['h5']}",
            "--set",
            f"data.modules.reader.init_args.schema={data['schema']}",
            "--set",
            f"model.modules.norm.init_args.norm_dict={data['nd']}",
        ])
        assert rc == 0

    def test_load_config_fit_sinks_include_callback_demand(self, data, tmp_path):
        # the static adapter sees the callback FIT/VAL sinks + their origins,
        # exactly as the runtime SaltModule does
        from salt.core.cli import load_config
        from salt.core.graph.spec import Mode

        override = write_yaml(tmp_path, "confmat.yaml", CONFMAT_CALLBACK_YAML)
        cfg = load_config(
            [DUMMY_CFG, Path(override)],
            set_overrides=[
                f"data.train_file={data['h5']}",
                f"data.val_file={data['h5']}",
                f"data.modules.reader.init_args.schema={data['schema']}",
                f"model.modules.norm.init_args.norm_dict={data['nd']}",
            ],
        )
        for mode in (Mode.FIT, Mode.VAL):
            assert "loss.total" in cfg.sinks[mode]
            assert "preds.jets.jets_classification" in cfg.sinks[mode]
        origins = cfg.sink_origins[Mode.FIT]
        assert "ConfusionMatrix" in origins["preds.jets.jets_classification"]


# ---------------------------------------------------------------------------
# 2-step fit smoke through the REAL CLI (run=True path; gate G2 surface)
# ---------------------------------------------------------------------------


class TestFitSmoke:
    def test_two_step_fit(self, data, tmp_path):
        rc = main([
            "fit",
            "--config",
            str(DUMMY_CFG),
            *required_overrides(data),
            f"--trainer.default_root_dir={tmp_path}",
            "--trainer.accelerator=cpu",
            # base2 ships a default-ON CometLogger (plan-24 Wave 0); turn it off so
            # the smoke run emits no offline Comet archive (and lr_monitor drops)
            "--trainer.logger=false",
            "--trainer.max_epochs=1",
            "--trainer.limit_train_batches=2",
            "--trainer.limit_val_batches=2",
            "--trainer.num_sanity_val_steps=0",
            "--trainer.log_every_n_steps=1",
            # null-delete the base2 ProgressBar (can't combine with the stock
            # enable_progress_bar=false; the D2 ProgressBar is default-on now)
            "--callbacks.progress=null",
        ])
        assert rc == 0
        # base2's Checkpoint (D2) wrote a checkpoint under the run dir's ckpts/,
        # with the 'loss=' stem the salt2-test fallback globs (M3-review fix)
        ckpts = list(tmp_path.rglob("*.ckpt"))
        assert ckpts, f"no checkpoint written under {tmp_path}"
        assert all("loss=" in ckpt.name for ckpt in ckpts), [c.name for c in ckpts]
        # the D2 Checkpoint forces the v1 'ckpts/' run-dir layout
        assert any(ckpt.parent.name == "ckpts" for ckpt in ckpts), [str(c) for c in ckpts]
        # the resolved config was persisted (SaveConfigCallback, design §5)
        configs = list(tmp_path.rglob("config.yaml"))
        assert configs, f"no config.yaml written under {tmp_path}"
        assert "class_path: salt.core.SaltModule" in configs[0].read_text()
        # M3-review fix: the SAVED run config (which carries ckpt_path: null)
        # round-trips into the salt2 graph tooling
        assert "ckpt_path" in configs[0].read_text()
        assert main(["graph", "validate", "-c", str(configs[0]), "--mode", "fit"]) == 0
        assert main(["graph", "plan", "-c", str(configs[0]), "--mode", "test"]) == 0


# ---------------------------------------------------------------------------
# graph/schema dispatch (the M1 tooling keeps working through salt2)
# ---------------------------------------------------------------------------


class TestGraphDispatch:
    def test_graph_validate(self, capsys):
        assert main(["graph", "validate", "-c", str(TOY_GRAPH_CFG)]) == 0
        assert "OK [mode=FIT]" in capsys.readouterr().out

    def test_schema_dump(self, data, tmp_path, capsys):
        out = tmp_path / "schema.yaml"
        assert main(["schema", "dump", str(data["h5"]), "-o", str(out)]) == 0
        assert out.is_file()


class TestFitRetryLoop:
    def test_second_fit_overwrites_stale_config(self, data, tmp_path, capsys):
        # stage-E ergonomics fix: a leftover config.yaml from a previous run
        # must not abort the retry loop (SaveConfigCallback overwrite=True),
        # and the artifact locations are announced at end of fit
        args = [
            "fit",
            "--config",
            str(DUMMY_CFG),
            *required_overrides(data),
            f"--trainer.default_root_dir={tmp_path}",
            "--trainer.accelerator=cpu",
            "--trainer.logger=false",  # opt out of the default CometLogger (plan-24 W0)
            "--trainer.max_epochs=1",
            "--trainer.limit_train_batches=2",
            "--trainer.limit_val_batches=2",
            "--trainer.num_sanity_val_steps=0",
            "--trainer.log_every_n_steps=1",
            "--callbacks.progress=null",  # see test_two_step_fit
        ]
        assert main(list(args)) == 0
        assert "salt2 fit artifacts" in capsys.readouterr().out
        assert main(list(args)) == 0  # used to raise: "expected ... to NOT exist"


class TestGraphFitConfigAdapter:
    """``salt2 graph`` over the REAL §5.1 trainer configs (stage-E HIGH fix)."""

    @staticmethod
    def set_flags(data) -> list[str]:
        return [
            "--set",
            f"model.modules.norm.init_args.norm_dict={data['nd']}",
            "--set",
            f"data.modules.reader.init_args.schema={data['schema']}",
        ]

    def test_validate_fit_config_all_modes(self, data, capsys):
        rc = main(["graph", "validate", "-c", str(DUMMY_CFG), *self.set_flags(data)])
        out = capsys.readouterr().out
        assert rc == 0
        for mode in ("FIT", "VAL", "TEST", "ONNX"):
            assert f"OK [mode={mode}]" in out
        # the §2.6 class-names cross-check ran (jets flavour_label attr)
        assert "class_names" in out

    def test_validate_is_data_free(self, capsys):
        # no schema, no norm dict on disk: bind is file-free, so any --set
        # value satisfies the parser without touching a file
        rc = main([
            "graph",
            "validate",
            "-c",
            str(DUMMY_CFG),
            "--mode",
            "fit",
            "--set",
            "model.modules.norm.init_args.norm_dict=unused.yaml",
        ])
        assert rc == 0
        # without a schema artifact the §2.6 opt-out warning is emitted
        assert "field spellings" in capsys.readouterr().err

    def test_plan_shows_full_pipeline(self, data, capsys):
        rc = main(["graph", "plan", "-c", str(DUMMY_CFG), "--mode", "fit", *self.set_flags(data)])
        out = capsys.readouterr().out
        assert rc == 0
        # dataset modules AND model modules in one ordered plan
        for name in ("reader", "features", "labels", "encoder", "jets_classification", "loss"):
            assert name in out

    def test_why_fit_config(self, data, capsys):
        rc = main([
            "graph",
            "why",
            "-c",
            str(DUMMY_CFG),
            "--mode",
            "fit",
            "--key",
            "encoded.seq",
            *self.set_flags(data),
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "producer" in out
        assert "encoder" in out

    def test_validate_label_typo_names_task_module(self, data, capsys):
        rc = main([
            "graph",
            "validate",
            "-c",
            str(DUMMY_CFG),
            "--mode",
            "fit",
            *self.set_flags(data),
            "--set",
            "model.modules.jets_classification.init_args.label=flavor_label",
        ])
        assert rc == 1
        err = capsys.readouterr().err
        assert "jets_classification" in err  # the demander, not '<sinks>'
        assert "flavour_label" in err  # nearest-key suggestion
        assert "<sinks>" not in err

    def test_validate_reordered_class_names_fails(self, data, capsys):
        rc = main([
            "graph",
            "validate",
            "-c",
            str(DUMMY_CFG),
            *self.set_flags(data),
            "--set",
            'model.modules.jets_classification.init_args.class_names=["bjets","ujets","cjets"]',
        ])
        assert rc == 1
        err = capsys.readouterr().err
        assert "DIFFERENT ORDER" in err
        assert "model.modules.jets_classification" in err

    def test_set_rejected_for_toy_configs(self, capsys):
        rc = main([
            "graph",
            "validate",
            "-c",
            str(TOY_GRAPH_CFG),
            "--set",
            "model.modules.x.init_args.y=1",
        ])
        assert rc == 1
        assert "trainer configs only" in capsys.readouterr().err

    def test_validate_onnx_legacy_outputs_fail_with_the_migration_error(self, tmp_path, capsys):
        # M4.5: export.outputs was removed — a pre-amendment config carrying
        # the section must fail `validate --mode onnx` with the §4.1-bar
        # migration error pointing at the writers (NOT silently validate
        # green against a stale hand-typed manifest)
        import yaml

        config = yaml.safe_load(DUMMY_CFG.read_text())
        config["export"]["outputs"] = [
            {"port": "preds.jets.jets_classification", "names": ["pb", "pc", "pu"]}
        ]
        bad = tmp_path / "legacy_outputs.yaml"
        bad.write_text(yaml.dump(config, sort_keys=False))
        rc = main([
            "graph",
            "validate",
            "-c",
            str(bad),
            "--mode",
            "onnx",
            "--set",
            "model.modules.norm.init_args.norm_dict=unused.yaml",
        ])
        assert rc == 1
        err = capsys.readouterr().err
        assert "REMOVED by the M4.5" in err
        assert "writers" in err

    def test_validate_onnx_sinks_derive_from_writers(self, capsys):
        # the unified-manifest happy path: ONNX validates green with sinks
        # from the writers (the shipped config carries NO export.outputs)
        rc = main([
            "graph",
            "validate",
            "-c",
            str(DUMMY_CFG),
            "--mode",
            "onnx",
            "--set",
            "model.modules.norm.init_args.norm_dict=unused.yaml",
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "OK [mode=ONNX]" in out

    def test_validate_onnx_underscore_model_name_fails(self, tmp_path, capsys):
        # the §4.1 'export.model_name contains no _/-' validate check —
        # used to pass even with --strict (M4-review HIGH fix)
        import yaml

        config = yaml.safe_load(DUMMY_CFG.read_text())
        config["export"]["model_name"] = "GN2_v2_dummy"
        bad = tmp_path / "bad_name.yaml"
        bad.write_text(yaml.dump(config, sort_keys=False))
        rc = main([
            "graph",
            "validate",
            "-c",
            str(bad),
            "--mode",
            "onnx",
            "--set",
            "model.modules.norm.init_args.norm_dict=unused.yaml",
        ])
        assert rc == 1
        err = capsys.readouterr().err
        assert "GN2_v2_dummy" in err
        assert "underscores or dashes" in err

    def test_validate_onnx_export_less_config_warns(self, tmp_path, capsys):
        # a trainer config without an export: block keeps the all-preds
        # fallback but says so — and --strict promotes it (the §9.3
        # converted-config CI gate expects the block to exist)
        import yaml

        config = yaml.safe_load(DUMMY_CFG.read_text())
        config.pop("export")
        no_export = tmp_path / "no_export.yaml"
        no_export.write_text(yaml.dump(config, sort_keys=False))
        flags = ["--set", "model.modules.norm.init_args.norm_dict=unused.yaml"]
        rc = main(["graph", "validate", "-c", str(no_export), "--mode", "onnx", *flags])
        out, err = capsys.readouterr()
        assert rc == 0  # non-strict: warning only
        assert "OK [mode=ONNX]" in out
        assert "no export: block" in err
        assert (
            main(["graph", "validate", "-c", str(no_export), "--mode", "onnx", "--strict", *flags])
            == 1
        )

    def test_plan_onnx_uses_export_sinks_and_prints_caveat(self, data, capsys):
        rc = main(["graph", "plan", "-c", str(DUMMY_CFG), "--mode", "onnx", *self.set_flags(data)])
        out = capsys.readouterr().out
        assert rc == 0
        # the static ONNX view is flagged as the dataset-fed approximation;
        # the traced graph's rendering is the export-time plan_onnx.txt
        assert "dataset-fed STATIC view" in out
        assert "plan_onnx.txt" in out
        # the no-op narrowed-to-nothing labels step is annotated (M4-review
        # fix: it used to look like live label loading in the ONNX plan)
        assert "labels" in out
        assert "[narrowed to 0 keys — no-op]" in out

    def test_validate_reports_fit_preds_as_info(self, data, capsys):
        rc = main([
            "graph",
            "validate",
            "-c",
            str(DUMMY_CFG),
            "--mode",
            "fit",
            *self.set_flags(data),
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "info:" in out
        assert "metric callback" in out


# ===========================================================================
# Wave-1 class_dict fan-out (formerly test_wave1_fanout.py)
#
# These exercise the --class_dict convenience flag on salt.core.main
# (Salt2CLI._fan_out_artifacts). Helpers/fixtures are wave1_*-prefixed to avoid
# colliding with the CLI-surface ones above. norm_dict is NOT a fan-out flag —
# it is the Normaliser module's own config (set on
# model.modules.norm.init_args.norm_dict); the TestNormDictOnModule section at
# the end pins that.
# ===========================================================================


# the two classification tasks in gn2v2-dummy.yaml, both with weight_source unset
CLS_TASKS = ("jets_classification", "track_origin")
NORM_MODULE = "norm"


@pytest.fixture(scope="module")
def wave1_data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("wave1_fanout")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    # a SECOND class dict, used to prove an explicit per-task weight_source is
    # left untouched even when --class_dict is also supplied
    cd_explicit = base / "class_dict_explicit.yaml"
    write_parity_norm_dict(base / "_throwaway_nd.yaml", cd_explicit)
    h5_path = base / "pp_output_train.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {
        "dir": base,
        "h5": h5_path,
        "nd": nd_path,
        "cd": cd_path,
        "cd_explicit": cd_explicit,
        "schema": schema_path,
    }


def wave1_base_overrides(wave1_data) -> list[str]:
    """The gn2v2-dummy.yaml required path overrides MINUS the per-task weight_source.

    ``norm_dict`` IS supplied here (the module form — it is the Normaliser's own
    REQUIRED config, no longer a fan-out flag). The per-task ``weight_source`` is
    deliberately omitted so the ``--class_dict`` flag (or the verbose block) is
    the ONLY source of those values.

    Returns
    -------
    list[str]
        The CLI args common to every case here.
    """
    return [
        "--config",
        str(DUMMY_CFG),
        f"--data.train_file={wave1_data['h5']}",
        f"--data.val_file={wave1_data['h5']}",
        f"--data.modules.reader.init_args.schema={wave1_data['schema']}",
        # norm_dict is the Normaliser module's own config (its sole consumer):
        # always set it the module way, NOT via a top-level fan-out flag
        f"--model.modules.{NORM_MODULE}.init_args.norm_dict={wave1_data['nd']}",
        "--trainer.logger=false",  # opt out of the default-ON CometLogger (plan-24 W0)
    ]


def wave1_verbose_block(wave1_data) -> list[str]:
    """Today's verbose per-task weight_source override block (the form being retired).

    Returns
    -------
    list[str]
        The two per-task ``--model.modules.*.init_args.weight_source`` overrides.
    """
    cd = wave1_data["cd"]

    def ws_override(task: str) -> str:
        return f'--model.modules.{task}.init_args.weight_source={{"from_class_dict": "{cd}"}}'

    return [
        ws_override("jets_classification"),
        ws_override("track_origin"),
    ]


def wave1_class_dict_flag(wave1_data) -> list[str]:
    """The Wave-1 one-flag form (``--class_dict``).

    Returns
    -------
    list[str]
        The ``--class_dict`` flag.
    """
    return [f"--class_dict={wave1_data['cd']}"]


def wave1_make_cli(wave1_data, extra: list[str]) -> Salt2CLI:
    return Salt2CLI(args=[*wave1_base_overrides(wave1_data), *extra], run=False)


def wave1_print_config(wave1_data, extra: list[str]) -> str:
    """Capture the ``--print_config`` dump for the given override block.

    Returns
    -------
    str
        The dumped YAML config.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit) as excinfo:
        wave1_make_cli(wave1_data, [*extra, "--print_config"])
    assert excinfo.value.code == 0
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Gate (a) — --print_config byte-equality (--class_dict == verbose two-line)
# ---------------------------------------------------------------------------


class TestPrintConfigByteEquality:
    def test_model_block_byte_equal(self, wave1_data):
        verbose = yaml.safe_load(wave1_print_config(wave1_data, wave1_verbose_block(wave1_data)))
        one_flag = yaml.safe_load(wave1_print_config(wave1_data, wave1_class_dict_flag(wave1_data)))
        # the whole assembled model block is byte-identical between the two forms
        # (norm_dict comes from the shared base overrides in both — module config)
        assert one_flag["model"] == verbose["model"]

    def test_norm_dict_field_from_module_config(self, wave1_data):
        # norm_dict is NOT a fan-out flag: it rides on the module config supplied
        # by wave1_base_overrides, identically for both the --class_dict and the
        # verbose weight_source form
        verbose = yaml.safe_load(wave1_print_config(wave1_data, wave1_verbose_block(wave1_data)))
        one_flag = yaml.safe_load(wave1_print_config(wave1_data, wave1_class_dict_flag(wave1_data)))

        def norm_path(cfg):
            return cfg["model"]["init_args"]["modules"][NORM_MODULE]["init_args"]["norm_dict"]

        assert norm_path(one_flag) == norm_path(verbose) == str(wave1_data["nd"])

    def test_weight_source_fields_equal(self, wave1_data):
        verbose = yaml.safe_load(wave1_print_config(wave1_data, wave1_verbose_block(wave1_data)))
        one_flag = yaml.safe_load(wave1_print_config(wave1_data, wave1_class_dict_flag(wave1_data)))

        def ws(cfg, task):
            return cfg["model"]["init_args"]["modules"][task]["init_args"]["weight_source"]

        for task in CLS_TASKS:
            assert ws(one_flag, task) == ws(verbose, task)
            assert ws(one_flag, task) == {"from_class_dict": str(wave1_data["cd"])}


# ---------------------------------------------------------------------------
# Gate (c) — the class_dict fan-out lands on the right tasks / leaves explicit
# ones alone (norm_dict, the module config, is covered in TestNormDictOnModule)
# ---------------------------------------------------------------------------


class TestFanOutInstantiated:
    def test_class_dict_lands_on_every_unset_task(self, wave1_data):
        cli = wave1_make_cli(wave1_data, wave1_class_dict_flag(wave1_data))
        for task in CLS_TASKS:
            mod = cli.model.net[task]
            assert isinstance(mod, ClassificationTaskModule)
            assert mod.weight_source == {"from_class_dict": str(wave1_data["cd"])}

    def test_class_dict_skips_non_classification_tasks(self, wave1_data):
        # track_vertexing is a VertexingTaskModule, NOT a ClassificationTaskModule,
        # so --class_dict must NOT fan a weight_source onto it (the .endswith
        # ClassificationTaskModule suffix-match in _fan_out_artifacts). Pins the
        # negative case so a future broadening of the match (e.g. to "TaskModule")
        # would be caught here.
        cli = wave1_make_cli(wave1_data, wave1_class_dict_flag(wave1_data))
        vtx = cli.model.net["track_vertexing"]
        assert not isinstance(vtx, ClassificationTaskModule)
        assert getattr(vtx, "weight_source", None) is None

    def test_explicit_weight_source_left_alone(self, wave1_data):
        # a task that ALREADY sets weight_source (here jets_classification, the
        # resume / saved-run-dir-config case) must NOT be overwritten by the
        # --class_dict fan-out; the other (unset) task still gets it.
        cd_explicit = wave1_data["cd_explicit"]
        explicit_ws = (
            "--model.modules.jets_classification.init_args.weight_source="
            f'{{"from_class_dict": "{cd_explicit}"}}'
        )
        cli = wave1_make_cli(wave1_data, [*wave1_class_dict_flag(wave1_data), explicit_ws])
        jets = cli.model.net["jets_classification"]
        track = cli.model.net["track_origin"]
        assert jets.weight_source == {"from_class_dict": str(cd_explicit)}  # untouched
        assert track.weight_source == {"from_class_dict": str(wave1_data["cd"])}  # fanned out

    def test_no_class_dict_is_a_noop(self, wave1_data):
        # with no --class_dict, an unset task stays unset (no accidental fan-out);
        # norm_dict supplied the module way still works (control that the fan-out
        # is purely additive on the --class_dict flag)
        cli = wave1_make_cli(wave1_data, [])
        for task in CLS_TASKS:
            assert cli.model.net[task].weight_source is None
        # norm_dict (module config from the base overrides) still landed
        assert str(cli.model.net[NORM_MODULE].norm_dict_path) == str(wave1_data["nd"])

    def test_class_dict_only_requires_norm_dict_elsewhere(self, wave1_data):
        # --class_dict fans out to the tasks; norm_dict must still be supplied
        # (it is REQUIRED on the Normaliser) — here via the module config in the
        # base overrides, proving the two knobs are independent
        cli = wave1_make_cli(wave1_data, wave1_class_dict_flag(wave1_data))
        for task in CLS_TASKS:
            assert cli.model.net[task].weight_source == {"from_class_dict": str(wave1_data["cd"])}


# ---------------------------------------------------------------------------
# norm_dict is the Normaliser module's OWN config (its sole consumer): set on
# model.modules.norm.init_args.norm_dict, NOT a top-level fan-out flag. These
# replace the retired --norm_dict fan-out tests.
# ---------------------------------------------------------------------------


class TestNormDictOnModule:
    def test_norm_dict_from_module_config_lands_on_normaliser(self, wave1_data):
        # the module-config form (model.modules.norm.init_args.norm_dict, supplied
        # by wave1_base_overrides) materialises on the instantiated Normaliser
        cli = wave1_make_cli(wave1_data, [])
        assert isinstance(cli.model, SaltModule)
        assert str(cli.model.net[NORM_MODULE].norm_dict_path) == str(wave1_data["nd"])

    def test_norm_dict_does_not_touch_tasks(self, wave1_data):
        # setting norm_dict on the module leaves the classification tasks'
        # weight_source untouched (norm_dict is module-local, not multi-task)
        cli = wave1_make_cli(wave1_data, [])
        for task in CLS_TASKS:
            assert cli.model.net[task].weight_source is None

    def test_norm_dict_unknown_flag_is_rejected(self, wave1_data):
        # the retired --norm_dict flag is now an unknown arg: the parser must
        # reject it (proves the top-level fan-out flag is gone, not silently
        # ignored). The norm_dict in base_overrides is dropped here so the only
        # norm_dict surface under test is the (now-invalid) flag.
        bad_args = [a for a in wave1_base_overrides(wave1_data) if "norm_dict" not in a]
        with pytest.raises(SystemExit):
            Salt2CLI(args=[*bad_args, f"--norm_dict={wave1_data['nd']}"], run=False)
