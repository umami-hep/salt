"""Tests for the ``salt2`` config surface (plan 05, stage C — design §5, §5.3).

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
"""

from __future__ import annotations

from pathlib import Path

import pytest
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary

from salt.core.data import GraphDataModule
from salt.core.main import CONFIG_DIR, Salt2CLI, main
from salt.core.nn.tasks import ClassificationTaskModule
from salt.core.saltmodule import SaltModule
from salt.core.schema import dump_schema, save_schema
from salt.core.writers import WriterCallback
from salt.tests.core.gn2_fixture import write_parity_norm_dict
from salt.utils.inputs import write_dummy_file

DUMMY_CFG = CONFIG_DIR / "gn2v2-dummy.yaml"
OPENDATA_CFG = CONFIG_DIR / "gn2v2-opendata.yaml"
TOY_GRAPH_CFG = Path(__file__).parent / "configs" / "toy.yaml"
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
    """Parse + instantiate (run=False) through the real CLI surface."""
    return Salt2CLI(
        args=["--config", str(config), *required_overrides(data), *(extra or [])],
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
        assert set(cli.model.net.keys()) == GN2V2_MODULES
        jets_task = cli.model.net["jets_classification"]
        assert isinstance(jets_task, ClassificationTaskModule)
        assert list(jets_task.class_names) == ["bjets", "cjets", "ujets", "taujets"]


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
        assert cli.model.lrs_config["max"] == pytest.approx(1.0e-3)
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
        # parse-level check only — a labels-less FIT would fail at compile
        cli = make_cli(data, extra=["--data.modules.labels=null"])
        dm_modules = cli.datamodule._modules  # noqa: SLF001 - assembly result
        assert set(dm_modules) == {"reader", "features"}


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
        cli = make_cli(data)
        kinds = [type(cb) for cb in cli.trainer.callbacks]
        assert ModelCheckpoint in kinds
        assert ModelSummary in kinds
        ckpt = next(cb for cb in cli.trainer.callbacks if isinstance(cb, ModelCheckpoint))
        assert ckpt.monitor == "val/loss"  # base2.yaml default

    def test_one_key_override(self, data):
        cli = make_cli(data, extra=["--callbacks.checkpoint.init_args.monitor=train/loss"])
        ckpt = next(cb for cb in cli.trainer.callbacks if isinstance(cb, ModelCheckpoint))
        assert ckpt.monitor == "train/loss"
        # the sibling dict entry survives the one-key override
        assert any(isinstance(cb, ModelSummary) for cb in cli.trainer.callbacks)

    def test_null_deletes_callback(self, data):
        cli = make_cli(data, extra=["--callbacks.checkpoint=null"])
        # no configured checkpoint left (Lightning may add its bare default,
        # which has no monitor)
        assert not any(getattr(cb, "monitor", None) == "val/loss" for cb in cli.trainer.callbacks)
        assert any(isinstance(cb, ModelSummary) for cb in cli.trainer.callbacks)

    def test_writers_defaults_assembled(self, data):
        # base2.yaml ships the v1-layout writer order (M3, design §8)
        cli = make_cli(data)
        wcb = next(cb for cb in cli.trainer.callbacks if isinstance(cb, WriterCallback))
        assert list(wcb.writers) == ["inputs_copy", "tasks", "pad_mask"]

    def test_writers_null_deletes(self, data):
        cli = make_cli(data, extra=["--writers.modules.pad_mask=null"])
        wcb = next(cb for cb in cli.trainer.callbacks if isinstance(cb, WriterCallback))
        assert "pad_mask" not in wcb.writers
        assert set(wcb.writers) == {"inputs_copy", "tasks"}  # siblings survive


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
            "--trainer.max_epochs=1",
            "--trainer.limit_train_batches=2",
            "--trainer.limit_val_batches=2",
            "--trainer.num_sanity_val_steps=0",
            "--trainer.log_every_n_steps=1",
            "--trainer.enable_progress_bar=false",
        ])
        assert rc == 0
        # base2's ModelCheckpoint wrote a checkpoint under the run dir, with
        # the 'loss=' stem the salt2-test fallback globs (M3-review fix)
        ckpts = list(tmp_path.rglob("*.ckpt"))
        assert ckpts, f"no checkpoint written under {tmp_path}"
        assert all("loss=" in ckpt.name for ckpt in ckpts), [c.name for c in ckpts]
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
            "--trainer.max_epochs=1",
            "--trainer.limit_train_batches=2",
            "--trainer.limit_val_batches=2",
            "--trainer.num_sanity_val_steps=0",
            "--trainer.log_every_n_steps=1",
            "--trainer.enable_progress_bar=false",
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

    def test_deadcode_sees_narrowed_writers(self, data, capsys):
        # M3-review MEDIUM fix: static tooling builds the writers' TEST
        # demand from the parsed writers: block — the §4.2 worked example
        # (narrowed TaskWriter -> dead-preds ERROR, exit non-zero, culprit
        # config address named), previously '[deadcode] mode=TEST: OK'
        rc = main([
            "graph",
            "deadcode",
            "-c",
            str(DUMMY_CFG),
            "--mode",
            "test",
            *self.set_flags(data),
            "--set",
            'writers.modules.tasks.init_args.streams=["jets"]',
        ])
        out = capsys.readouterr().out
        assert rc == 1
        assert "consumed by NO writer" in out
        assert "track_origin" in out and "track_vertexing" in out
        # §4.2-exemplar attribution: the culprit writer address + excluded
        # streams, and the per-task config addresses
        assert "writers.modules.tasks.init_args.streams" in out
        assert "excludes" in out
        assert "model.modules.track_origin" in out

    def test_validate_errors_on_narrowed_writers_other_modes_still_report(self, data, capsys):
        # validate (all modes) exits 1 on the stored TEST sink error while
        # FIT/VAL/ONNX still compile and print their OK lines
        rc = main([
            "graph",
            "validate",
            "-c",
            str(DUMMY_CFG),
            *self.set_flags(data),
            "--set",
            'writers.modules.tasks.init_args.streams=["jets"]',
        ])
        out, err = capsys.readouterr()
        assert rc == 1
        assert "OK [mode=FIT]" in out
        assert "consumed by NO writer" in err

    def test_plan_test_mode_raises_on_narrowed_writers(self, data, capsys):
        rc = main([
            "graph",
            "plan",
            "-c",
            str(DUMMY_CFG),
            "--mode",
            "test",
            *self.set_flags(data),
            "--set",
            'writers.modules.tasks.init_args.streams=["jets"]',
        ])
        assert rc == 1
        assert "consumed by NO writer" in capsys.readouterr().err

    def test_strict_validate_passes_stock_config(self, data):
        # M3-review fix: unconsumed FIT/VAL preds are info-level (design
        # §3.3), so the documented CI mode (--strict) works on a standard
        # tagger config — it used to exit 1 on six preds warnings
        assert (
            main(["graph", "validate", "-c", str(DUMMY_CFG), "--strict", *self.set_flags(data)])
            == 0
        )

    def test_validate_onnx_bad_export_port_fails(self, tmp_path, capsys):
        # M4-review HIGH fix: ONNX-mode sinks come from export.outputs —
        # a typo'd export port used to validate green (even --strict) and
        # fail months later at export time (design §3.1/§4.1, §9.3 CI gate)
        import yaml

        config = yaml.safe_load(DUMMY_CFG.read_text())
        config["export"]["outputs"][0]["port"] = "preds.jets.jets_clasification"
        bad = tmp_path / "bad_port.yaml"
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
        assert "preds.jets.jets_clasification" in err
        assert "export.outputs" in err  # the demanding config address

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
