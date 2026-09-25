"""Tests for the ``salt`` config surface / ``salt.main``."""

from __future__ import annotations

import contextlib
import io
import os
import re
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import pytest
import torch
import yaml
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary

from salt.callbacks import Checkpoint, ProgressBar
from salt.data import SaltDataModule
from salt.graph.errors import ConfigError
from salt.main import (
    CONFIG_DIR,
    RUN_DIR_TIMESTAMP_FORMAT,
    SALT_AUTO_RESUME_ENV,
    SaltCLI,
    _best_checkpoint,  # noqa: PLC2701 - the fallback glob under test
    latest_checkpoint,
    main,
    run_dir_path,
    run_root,
)
from salt.model.modules.tasks import ClassificationTaskModule
from salt.model.saltmodule import SaltModule
from salt.schema import dump_schema, save_schema
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict
from salt.tests._fixtures.gn2v2_test_config import small_config
from salt.testing.inputs import write_dummy_file
from salt.utils.config_utils import disable_logger_in_config

DUMMY_CFG = small_config()
NAME = yaml.safe_load(DUMMY_CFG.read_text())["name"]
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
    # the outputs: section writers are graph-folded into model.net
    # (inputs_copy is manifest-only, not folded).
    "jets_out",
    "origin_out",
    "pad_mask",
}

FOURTH_TASK_YAML = """
model:
  init_args:
    modules:
      track_type:
        class_path: salt.model.modules.tasks.ClassificationTaskModule
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
    # the committed YAML — the documented required overrides)
    base = tmp_path_factory.mktemp("salt_cli")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    h5_path = base / "pp_output_train.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


def required_overrides(data) -> list[str]:
    """The gn2v2-opendata.yaml documented required overrides (its header)."""
    return [
        f"--data.train_file={data['h5']}",
        f"--data.val_file={data['h5']}",
        f"--data.modules.reader.init_args.schema={data['schema']}",
        f"--model.modules.norm.init_args.norm_dict={data['nd']}",
    ]


def make_cli(data, extra: list[str] | None = None, config: Path = DUMMY_CFG) -> SaltCLI:
    """Parse + instantiate (run=False) through the real CLI surface."""
    cfg = disable_logger_in_config(str(config))
    return SaltCLI(
        args=["--config", cfg, *required_overrides(data), *(extra or [])],
        run=False,
    )


def write_yaml(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


# parse + instantiate (gn2v2-opendata.yaml is the shipped worked config)


class TestParseAndInstantiate:
    def test_dummy_config_instantiates_saltmodule(self, data):
        cli = make_cli(data)
        assert isinstance(cli.model, SaltModule)
        assert isinstance(cli.datamodule, SaltDataModule)
        assert set(cli.model.net.keys()) == GN2V2_MODULES
        assert cli.model.net["encoder"].name == "encoder"
        # data modules in YAML order: reader -> features -> labels
        assert str(cli.datamodule.train_file) == str(data["h5"])

    def test_name_linked_into_model(self, data):
        cli = make_cli(data)
        assert cli.config.name == "GN2v2_dummy"
        assert cli.model.name == "GN2v2_dummy"

    def test_opendata_config_parses(self, data):
        # same model at open-data scale; file paths are required overrides
        # (real-data runs are out of scope here — no /data paths)
        cli = make_cli(data, config=OPENDATA_CFG)
        assert isinstance(cli.model, SaltModule)
        # the open-data config is on the outputs: section path — model.net carries
        # the model modules (nets + tasks + loss) PLUS the composed section
        # GRAPH-node writers. Open-data uses ONE run_tasks writer (+ pad_mask;
        # inputs_copy is manifest-only, not an nn.Module) — the dummy's mode-split
        # jets_out/origin_out writers (in GN2V2_MODULES) do not appear here.
        base = GN2V2_MODULES - {"jets_out", "origin_out", "pad_mask"}
        assert set(cli.model.net.keys()) == base | {"run_tasks", "pad_mask"}
        jets_task = cli.model.net["jets_classification"]
        assert isinstance(jets_task, ClassificationTaskModule)
        assert list(jets_task.class_names) == ["bjets", "cjets", "ujets", "taujets"]
        # the outputs: section is composed onto the model
        section = cli.model._output_section  # noqa: SLF001
        assert set(section) == {"inputs_copy", "run_tasks", "pad_mask"}
        assert section["run_tasks"].is_run_task_output()


# --print_config round-trip


class TestPrintConfig:
    def test_round_trip(self, data, tmp_path, capsys):
        with pytest.raises(SystemExit) as excinfo:
            make_cli(data, extra=["--print_config"])
        assert excinfo.value.code == 0
        printed = capsys.readouterr().out
        # full class_path/init_args blocks with defaults made explicit
        assert "class_path: salt.model.SaltModule" in printed
        assert "class_path: salt.model.modules.TransformerEncoder" in printed
        assert "torch-math" in printed

        # feeding the printed config back yields the same parsed namespaces
        cli_orig = make_cli(data)
        printed_path = write_yaml(tmp_path, "printed.yaml", printed)
        cli_again = SaltCLI(args=["--config", printed_path], run=False)
        assert cli_again.config.model == cli_orig.config.model
        assert cli_again.config.data == cli_orig.config.data
        assert cli_again.config.callbacks == cli_orig.config.callbacks


# deep-merge across config files (DeepMergeParser)


class TestDeepMerge:
    def test_override_file_adds_fourth_task(self, data, tmp_path):
        override = write_yaml(tmp_path, "add_task.yaml", FOURTH_TASK_YAML)
        cli = make_cli(data, extra=["--config", override])
        # the new entry merged in; ALL siblings survive (the semantics that
        # native jsonargparse loses — dict leaves replace wholesale)
        assert set(cli.model.net.keys()) == GN2V2_MODULES | {"track_type"}
        task = cli.model.net["track_type"]
        assert isinstance(task, ClassificationTaskModule)
        # sibling init_args of the model itself survive the partial restate.
        # Read the expected value off the config under test rather than pinning
        # a literal: what is asserted is that the override did not clear it.
        base_lrs = yaml.safe_load(DUMMY_CFG.read_text())["model"]["init_args"]["lrs"]
        assert cli.model.lrs["max"] == pytest.approx(base_lrs["max"])
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
        assert enc_cfg["class_path"] == "salt.model.modules.TransformerEncoder"
        assert enc_cfg["init_args"]["num_layers"] == 3
        assert enc_cfg["init_args"]["dim"] == 16
        assert set(cli.model.net.keys()) == GN2V2_MODULES  # siblings survive


# null-deletion (parse-to-None + assembly-time filtering)


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
        # labels can be dropped in TEST-style configs;
        # parse-level check only — a labels-less FIT would fail at compile.
        # The deprecated --data.train_file alias synthesises an implicit
        # InputSamples, so it joins _modules.
        cli = make_cli(data, extra=["--data.modules.labels=null"])
        dm_modules = cli.datamodule._modules  # noqa: SLF001 - assembly result
        assert set(dm_modules) == {"reader", "features", "input_samples"}


# dotted CLI overrides into init_args


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


# callbacks: dict mechanics + assembly into trainer.callbacks


class TestCallbacksDict:
    def test_base_defaults_assembled(self, data):
        # base.yaml ships the salt.callbacks.Checkpoint port (a
        # ModelCheckpoint subclass) + ProgressBar + ModelSummary
        cli = make_cli(data)
        assert any(isinstance(cb, Checkpoint) for cb in cli.trainer.callbacks)
        assert any(isinstance(cb, ProgressBar) for cb in cli.trainer.callbacks)
        assert any(isinstance(cb, ModelSummary) for cb in cli.trainer.callbacks)
        ckpt = next(cb for cb in cli.trainer.callbacks if isinstance(cb, ModelCheckpoint))
        assert ckpt.monitor == "val/loss"  # base.yaml monitor_loss default
        assert ckpt.save_top_k == -1  # the v1 Checkpoint keeps every epoch

    def test_one_key_override(self, data):
        # monitor_loss is the new Checkpoint arg (it sets BOTH the monitor
        # and the filename loss tag)
        cli = make_cli(data, extra=["--callbacks.checkpoint.init_args.monitor_loss=train/loss"])
        ckpt = next(cb for cb in cli.trainer.callbacks if isinstance(cb, ModelCheckpoint))
        assert ckpt.monitor == "train/loss"
        assert ckpt.filename == "epoch={epoch:03d}-step={step}-loss={train/loss:.5f}"
        # the sibling dict entry survives the one-key override
        assert any(isinstance(cb, ModelSummary) for cb in cli.trainer.callbacks)

    def test_null_deletes_callback(self, data):
        cli = make_cli(data, extra=["--callbacks.checkpoint=null"])
        # no configured checkpoint left (Lightning may add its bare default,
        # which has no monitor)
        assert not any(getattr(cb, "monitor", None) == "val/loss" for cb in cli.trainer.callbacks)
        assert any(isinstance(cb, ModelSummary) for cb in cli.trainer.callbacks)



# FIT/VAL callback-declared sinks via the STATIC graph tooling — the runtime
# path is covered in test_saltmodule.py::TestCallbackSinks; here the static
# `salt graph` path (load_config / `salt graph validate`) must see the SAME
# FIT/VAL sinks.

CONFMAT_CALLBACK_YAML = """
callbacks:
  confmat:
    class_path: salt.callbacks.ConfusionMatrix
    init_args:
      task_name: jets_classification
"""


class TestStaticFitValCallbackSinks:
    def test_validate_fit_with_callback_passes(self, data, tmp_path):
        # a configured ConfusionMatrix must not break `salt graph validate
        # --mode fit` (it declares preds/labels the task already keeps alive)
        from salt.main import main as graph_main

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
        from salt.cli import load_config
        from salt.graph.spec import Mode

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


# 2-step fit smoke through the REAL CLI (run=True path)


class TestFitSmoke:
    def test_two_step_fit(self, data, tmp_path):
        rc = main([
            "fit",
            "--config",
            str(DUMMY_CFG),
            *required_overrides(data),
            f"--trainer.default_root_dir={tmp_path}",
            "--trainer.accelerator=cpu",
            # base ships a default-ON CometLogger; turn it off so
            # the smoke run emits no offline Comet archive (and lr_monitor drops)
            "--trainer.logger=false",
            "--trainer.max_epochs=1",
            "--trainer.limit_train_batches=2",
            "--trainer.limit_val_batches=2",
            "--trainer.num_sanity_val_steps=0",
            "--trainer.log_every_n_steps=1",
            # null-delete the base ProgressBar (can't combine with the stock
            # enable_progress_bar=false; the ProgressBar is default-on now)
            "--callbacks.progress=null",
        ])
        assert rc == 0
        # base's Checkpoint wrote a checkpoint under the run dir's ckpts/,
        # with the 'loss=' stem the salt-test fallback globs
        ckpts = list(tmp_path.rglob("*.ckpt"))
        assert ckpts, f"no checkpoint written under {tmp_path}"
        assert all("loss=" in ckpt.name for ckpt in ckpts), [c.name for c in ckpts]
        # the Checkpoint forces the v1 'ckpts/' run-dir layout
        assert any(ckpt.parent.name == "ckpts" for ckpt in ckpts), [str(c) for c in ckpts]
        # exactly one timestamped run dir, whose name's field after the last
        # '_' is the RUN_DIR_TIMESTAMP_FORMAT stamp
        run_dirs = [d for d in tmp_path.glob(f"{NAME}_*") if d.is_dir()]
        assert len(run_dirs) == 1, run_dirs
        run_dir = run_dirs[0]
        datetime.strptime(run_dir.name.rsplit("_", 1)[-1], RUN_DIR_TIMESTAMP_FORMAT)
        assert all(ckpt.name.startswith("epoch=000-step=2-loss=") for ckpt in ckpts), [
            c.name for c in ckpts
        ]
        # the resolved config was persisted (SaveConfigCallback) directly in the run dir
        configs = list(tmp_path.rglob("config.yaml"))
        assert configs, f"no config.yaml written under {tmp_path}"
        assert configs[0].parent == run_dir
        assert "class_path: salt.model.SaltModule" in configs[0].read_text()
        # the SAVED run config (which carries ckpt_path: null) round-trips
        # into the salt graph tooling
        assert "ckpt_path" in configs[0].read_text()
        saved = yaml.safe_load(configs[0].read_text())
        assert saved["trainer"]["default_root_dir"].endswith(run_dir.name)
        assert main(["graph", "validate", "-c", str(configs[0]), "--mode", "fit"]) == 0
        assert main(["graph", "plan", "-c", str(configs[0]), "--mode", "test"]) == 0


class TestCompileFlag:
    """``--compile``: in-place ``torch.compile`` of every graph module."""

    def test_defaults_off(self, data):
        assert make_cli(data).config.compile is False

    def test_flag_parses(self, data):
        assert make_cli(data, extra=["--compile"]).config.compile is True

    def test_fit_compiles_modules_and_checkpoint_round_trips(self, data, tmp_path, monkeypatch):
        # real dynamo (so _compiled_call_impl is genuine) on the eager backend
        # (so the unit suite pays no inductor codegen cost)
        real_compile = torch.compile
        monkeypatch.setattr(
            torch, "compile", lambda module, **kw: real_compile(module, backend="eager", **kw)
        )
        rc = main([
            "fit",
            "--config",
            str(DUMMY_CFG),
            *required_overrides(data),
            "--compile",
            f"--trainer.default_root_dir={tmp_path}",
            "--trainer.accelerator=cpu",
            "--trainer.logger=false",
            "--trainer.max_epochs=1",
            "--trainer.limit_train_batches=2",
            "--trainer.limit_val_batches=2",
            "--trainer.num_sanity_val_steps=0",
            "--trainer.log_every_n_steps=1",
            "--data.num_workers=0",
            "--callbacks.progress=null",
        ])
        assert rc == 0

        ckpts = list(tmp_path.rglob("*.ckpt"))
        assert ckpts, f"no checkpoint written under {tmp_path}"
        checkpoint = torch.load(ckpts[0], map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        # in-place compile => NO dynamo wrapper in the module tree, so the
        # checkpoint is byte-compatible with an uncompiled load
        assert not any("_orig_mod." in key for key in state_dict), sorted(state_dict)[:5]

        model = make_cli(data).model
        model.on_load_checkpoint(checkpoint)
        assert set(state_dict) == set(model.state_dict())
        model.load_state_dict(state_dict)

    def test_modules_are_compiled_in_place(self, data, tmp_path, monkeypatch):
        """The graph modules keep their identity — the plan/executor hold them."""
        real_compile = torch.compile
        monkeypatch.setattr(
            torch, "compile", lambda module, **kw: real_compile(module, backend="eager", **kw)
        )
        seen: dict[str, object] = {}
        real_setup = SaltModule.setup

        def spy(self, stage):
            real_setup(self, stage)
            seen.update(self._graph_modules)  # noqa: SLF001 - the surface under test

        monkeypatch.setattr(SaltModule, "setup", spy)
        rc = main([
            "fit",
            "--config",
            str(DUMMY_CFG),
            *required_overrides(data),
            "--compile",
            f"--trainer.default_root_dir={tmp_path}",
            "--trainer.accelerator=cpu",
            "--trainer.logger=false",
            "--trainer.max_epochs=1",
            "--trainer.limit_train_batches=1",
            "--trainer.limit_val_batches=1",
            "--trainer.num_sanity_val_steps=0",
            "--trainer.log_every_n_steps=1",
            "--data.num_workers=0",
            "--callbacks.progress=null",
        ])
        assert rc == 0
        assert seen, "setup spy never fired"
        # every nn.Module graph module got a compiled call impl, and none of them
        # was replaced by dynamo's OptimizedModule (which would fail the
        # executor's GraphModule protocol check)
        compiled = [
            name
            for name, module in seen.items()
            if getattr(module, "_compiled_call_impl", None) is not None
        ]
        assert compiled, sorted(seen)
        assert all(type(module).__name__ != "OptimizedModule" for module in seen.values())


class TestBestCheckpointFallback:
    """The salt-test no-``--ckpt_path`` fallback: ``_best_checkpoint`` globs
    ``{ckpts,checkpoints}/*.ckpt`` next to the saved config and picks the
    lowest embedded ``loss=`` (v1 best-epoch contract).
    """

    @staticmethod
    def _config(tmp_path) -> Path:
        config = tmp_path / "config.yaml"
        config.write_text("{}")
        return config

    def test_picks_lowest_loss_across_both_dirs(self, tmp_path, capsys):
        """Scans BOTH ckpts/ (v1 runs) and checkpoints/ (Lightning default)."""
        config = self._config(tmp_path)
        (tmp_path / "ckpts").mkdir()
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "ckpts" / "epoch=000-loss=0.75.ckpt").touch()
        (tmp_path / "ckpts" / "epoch=001-loss=0.50.ckpt").touch()
        (tmp_path / "checkpoints" / "epoch=002-loss=0.25.ckpt").touch()
        best = _best_checkpoint(config)
        assert best == str(tmp_path / "checkpoints" / "epoch=002-loss=0.25.ckpt")
        assert best in capsys.readouterr().out  # the chosen path is announced

    def test_non_loss_named_files_are_skipped(self, tmp_path):
        """A last.ckpt without a loss= stem never wins over a scored one."""
        config = self._config(tmp_path)
        (tmp_path / "ckpts").mkdir()
        (tmp_path / "ckpts" / "last.ckpt").touch()
        (tmp_path / "ckpts" / "epoch=000-loss=1.5.ckpt").touch()
        assert _best_checkpoint(config).endswith("epoch=000-loss=1.5.ckpt")

    def test_no_loss_named_checkpoints_raises_config_error(self, tmp_path):
        """No loss=-named checkpoint anywhere -> loud ConfigError naming --ckpt_path."""
        config = self._config(tmp_path)
        (tmp_path / "ckpts").mkdir()
        (tmp_path / "ckpts" / "last.ckpt").touch()  # present but unscoreable
        with pytest.raises(ConfigError, match="ckpt_path"):
            _best_checkpoint(config)

    def test_missing_ckpt_dirs_raise_config_error(self, tmp_path):
        """A config with no ckpts/checkpoints sibling dirs at all -> same error."""
        with pytest.raises(ConfigError, match="loss="):
            _best_checkpoint(self._config(tmp_path))


# graph/schema dispatch (the tooling keeps working through salt)


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
        # a leftover config.yaml from a previous run
        # must not abort the retry loop (SaveConfigCallback overwrite=True),
        # and the artifact locations are announced at end of fit.
        # --log_suffix pins both runs to the SAME dir, deterministically
        # exercising the overwrite path (no timestamp-collision guesswork).
        args = [
            "fit",
            "--config",
            str(DUMMY_CFG),
            *required_overrides(data),
            f"--trainer.default_root_dir={tmp_path}",
            "--trainer.accelerator=cpu",
            "--trainer.logger=false",  # opt out of the default CometLogger
            "--trainer.max_epochs=1",
            "--trainer.limit_train_batches=2",
            "--trainer.limit_val_batches=2",
            "--trainer.num_sanity_val_steps=0",
            "--trainer.log_every_n_steps=1",
            "--callbacks.progress=null",  # see test_two_step_fit
            "--log_suffix=retry",
        ]
        assert main(list(args)) == 0
        out = capsys.readouterr().out
        assert "salt fit artifacts" in out
        assert "not yet implemented" not in out
        run_dir = tmp_path / f"{NAME}_retry"
        assert run_dir.is_dir()
        assert main(list(args)) == 0  # used to raise: "expected ... to NOT exist"
        out = capsys.readouterr().out
        assert "not yet implemented" not in out
        # both runs landed in the SAME suffix dir — no second sibling minted
        run_dirs = [d for d in tmp_path.glob(f"{NAME}_*") if d.is_dir()]
        assert run_dirs == [run_dir]


class TestGraphFitConfigAdapter:
    """``salt graph`` over the REAL trainer configs."""

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
        # the class-names cross-check ran (jets flavour_label attr)
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
        # without a schema artifact the opt-out warning is emitted
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
        # the sink's 'model_name contains no _/-' validate check
        import yaml

        config = yaml.safe_load(DUMMY_CFG.read_text())
        config["outputs"]["onnx_export"]["init_args"]["model_name"] = "GN2_v2_dummy"
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
        # a trainer config declaring no export contract keeps the all-preds
        # fallback but says so — and --strict promotes it to an error
        import yaml

        config = yaml.safe_load(DUMMY_CFG.read_text())
        del config["outputs"]["onnx_export"]
        no_export = tmp_path / "no_export.yaml"
        no_export.write_text(yaml.dump(config, sort_keys=False))
        flags = ["--set", "model.modules.norm.init_args.norm_dict=unused.yaml"]
        rc = main(["graph", "validate", "-c", str(no_export), "--mode", "onnx", *flags])
        out, err = capsys.readouterr()
        assert rc == 0  # non-strict: warning only
        assert "OK [mode=ONNX]" in out
        assert "declares no export contract" in err
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
        # the no-op narrowed-to-nothing labels step is annotated, so it does
        # not look like live label loading in the ONNX plan
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

    def test_test_only_h5_column_error_surfaces_only_for_test(self, data, tmp_path, capsys):
        # a section-declared TEST H5 sink whose `consumes:` matches no leaf is
        # broken ONLY at declare_io(TEST) — `--mode fit` must not see it, and
        # `mode_errors[TEST]` must stay unpopulated, exactly as at the pin
        from salt.cli import load_config
        from salt.graph.spec import Mode

        config = yaml.safe_load(DUMMY_CFG.read_text())
        config["outputs"]["h5_output"] = {
            "class_path": "salt.outputs.H5OutputSink",
            "init_args": {"consumes": ["outputs.jets.no_such_leaf"]},
        }
        bad = tmp_path / "test_only_broken.yaml"
        bad.write_text(yaml.dump(config, sort_keys=False))
        flags = self.set_flags(data)

        cfg = load_config(
            [bad],
            set_overrides=[
                f"data.train_file={data['h5']}",
                f"data.val_file={data['h5']}",
                f"data.modules.reader.init_args.schema={data['schema']}",
                f"model.modules.norm.init_args.norm_dict={data['nd']}",
            ],
        )
        assert cfg.mode_errors.get(Mode.TEST) is None
        assert cfg.sinks[Mode.TEST] == ()

        assert main(["graph", "plan", "-c", str(bad), "--mode", "fit", *flags]) == 0
        capsys.readouterr()

        assert main(["graph", "plan", "-c", str(bad), "--mode", "test", *flags]) == 1
        assert "matches none of the declared output leaves" in capsys.readouterr().err

        assert main(["graph", "validate", "-c", str(bad), *flags]) == 1
        assert "matches none of the declared output leaves" in capsys.readouterr().err


# --class_dict fan-out (SaltCLI._fan_out_artifacts). Helpers are
# fanout_*-prefixed. norm_dict is NOT a fan-out flag — it is the Normaliser
# module's own config (pinned in TestNormDictOnModule).


# the two classification tasks in gn2v2-opendata.yaml, both with weight_source unset
CLS_TASKS = ("jets_classification", "track_origin")
NORM_MODULE = "norm"


@pytest.fixture(scope="module")
def fanout_data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("fanout")
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


def fanout_base_overrides(fanout_data) -> list[str]:
    """The gn2v2-opendata.yaml required path overrides MINUS the per-task weight_source."""
    return [
        "--config",
        str(DUMMY_CFG),
        f"--data.train_file={fanout_data['h5']}",
        f"--data.val_file={fanout_data['h5']}",
        f"--data.modules.reader.init_args.schema={fanout_data['schema']}",
        # norm_dict is the Normaliser module's own config (its sole consumer):
        # always set it the module way, NOT via a top-level fan-out flag
        f"--model.modules.{NORM_MODULE}.init_args.norm_dict={fanout_data['nd']}",
        "--trainer.logger=false",  # opt out of the default-ON CometLogger
    ]


def fanout_verbose_block(fanout_data) -> list[str]:
    """Today's verbose per-task weight_source override block (the form being retired)."""
    cd = fanout_data["cd"]

    def ws_override(task: str) -> str:
        return f'--model.modules.{task}.init_args.weight_source={{"from_class_dict": "{cd}"}}'

    return [
        ws_override("jets_classification"),
        ws_override("track_origin"),
    ]


def fanout_class_dict_flag(fanout_data) -> list[str]:
    """The one-flag form (``--class_dict``)."""
    return [f"--class_dict={fanout_data['cd']}"]


def fanout_make_cli(fanout_data, extra: list[str]) -> SaltCLI:
    return SaltCLI(args=[*fanout_base_overrides(fanout_data), *extra], run=False)


def fanout_print_config(fanout_data, extra: list[str]) -> str:
    """Capture the ``--print_config`` dump for the given override block."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit) as excinfo:
        fanout_make_cli(fanout_data, [*extra, "--print_config"])
    assert excinfo.value.code == 0
    return buf.getvalue()


# --print_config byte-equality (--class_dict == verbose two-line)


class TestPrintConfigByteEquality:
    def test_model_block_byte_equal(self, fanout_data):
        verbose = yaml.safe_load(
            fanout_print_config(fanout_data, fanout_verbose_block(fanout_data))
        )
        one_flag = yaml.safe_load(
            fanout_print_config(fanout_data, fanout_class_dict_flag(fanout_data))
        )
        # the whole assembled model block is byte-identical between the two forms
        # (norm_dict comes from the shared base overrides in both — module config)
        assert one_flag["model"] == verbose["model"]

    def test_norm_dict_field_from_module_config(self, fanout_data):
        # norm_dict is NOT a fan-out flag: it rides on the module config supplied
        # by fanout_base_overrides, identically for both the --class_dict and the
        # verbose weight_source form
        verbose = yaml.safe_load(
            fanout_print_config(fanout_data, fanout_verbose_block(fanout_data))
        )
        one_flag = yaml.safe_load(
            fanout_print_config(fanout_data, fanout_class_dict_flag(fanout_data))
        )

        def norm_path(cfg):
            return cfg["model"]["init_args"]["modules"][NORM_MODULE]["init_args"]["norm_dict"]

        assert norm_path(one_flag) == norm_path(verbose) == str(fanout_data["nd"])

    def test_weight_source_fields_equal(self, fanout_data):
        verbose = yaml.safe_load(
            fanout_print_config(fanout_data, fanout_verbose_block(fanout_data))
        )
        one_flag = yaml.safe_load(
            fanout_print_config(fanout_data, fanout_class_dict_flag(fanout_data))
        )

        def ws(cfg, task):
            return cfg["model"]["init_args"]["modules"][task]["init_args"]["weight_source"]

        for task in CLS_TASKS:
            assert ws(one_flag, task) == ws(verbose, task)
            assert ws(one_flag, task) == {"from_class_dict": str(fanout_data["cd"])}


# The class_dict fan-out lands on the right tasks / leaves explicit
# ones alone (norm_dict, the module config, is covered in TestNormDictOnModule)


class TestFanOutInstantiated:
    def test_class_dict_lands_on_every_unset_task(self, fanout_data):
        cli = fanout_make_cli(fanout_data, fanout_class_dict_flag(fanout_data))
        for task in CLS_TASKS:
            mod = cli.model.net[task]
            assert isinstance(mod, ClassificationTaskModule)
            assert mod.weight_source == {"from_class_dict": str(fanout_data["cd"])}

    def test_class_dict_skips_non_classification_tasks(self, fanout_data):
        # track_vertexing is a VertexingTaskModule, NOT a ClassificationTaskModule,
        # so --class_dict must NOT fan a weight_source onto it (the .endswith
        # ClassificationTaskModule suffix-match in _fan_out_artifacts). Pins the
        # negative case so a future broadening of the match (e.g. to "TaskModule")
        # would be caught here.
        cli = fanout_make_cli(fanout_data, fanout_class_dict_flag(fanout_data))
        vtx = cli.model.net["track_vertexing"]
        assert not isinstance(vtx, ClassificationTaskModule)
        assert getattr(vtx, "weight_source", None) is None

    def test_explicit_weight_source_left_alone(self, fanout_data):
        # a task that ALREADY sets weight_source (here jets_classification, the
        # resume / saved-run-dir-config case) must NOT be overwritten by the
        # --class_dict fan-out; the other (unset) task still gets it.
        cd_explicit = fanout_data["cd_explicit"]
        explicit_ws = (
            "--model.modules.jets_classification.init_args.weight_source="
            f'{{"from_class_dict": "{cd_explicit}"}}'
        )
        cli = fanout_make_cli(fanout_data, [*fanout_class_dict_flag(fanout_data), explicit_ws])
        jets = cli.model.net["jets_classification"]
        track = cli.model.net["track_origin"]
        assert jets.weight_source == {"from_class_dict": str(cd_explicit)}  # untouched
        assert track.weight_source == {"from_class_dict": str(fanout_data["cd"])}  # fanned out

    def test_no_class_dict_is_a_noop(self, fanout_data):
        # with no --class_dict, an unset task stays unset (no accidental fan-out);
        # norm_dict supplied the module way still works (control that the fan-out
        # is purely additive on the --class_dict flag)
        cli = fanout_make_cli(fanout_data, [])
        for task in CLS_TASKS:
            assert cli.model.net[task].weight_source is None
        # norm_dict (module config from the base overrides) still landed
        assert str(cli.model.net[NORM_MODULE].norm_dict_path) == str(fanout_data["nd"])


# top-level `training_schedule:` — a config-surface peer of trainer:/data:/
# model:, injected into the SaltModule constructor arg before instantiation;
# the nested model.init_args home is rejected fail-loud.


# structural twin of docs/tutorials/configs/finetuning/finetune_gn3large_add_charge_head.yaml:
# a head warm-up (only the new head trainable, per-stage lrs override) then a
# full-network fine-tune (frozen: [] = everything trainable, epochs omitted =
# remainder).
TOP_LEVEL_SCHEDULE_YAML = """
training_schedule:
  stages:
    head_warmup:
      epochs: 5
      trainable: [jets_classification]
      lrs:
        max: 1.0e-4
    full_finetune:
      frozen: []
      lrs:
        initial: 1.0e-7
        max: 1.0e-5
"""

SCHEDULE_OVERRIDE_YAML = """
training_schedule:
  stages:
    head_warmup:
      epochs: 2
"""

DELETE_STAGE_YAML = """
training_schedule:
  stages:
    full_finetune: null
"""

NESTED_SCHEDULE_YAML = """
model:
  init_args:
    training_schedule:
      stages:
        fit:
          frozen: [encoder]
"""


class StageCallbackProbe(Callback):
    """Test-only stage-scoped callback mirroring a real `StageLRTracer` spec
    shape (``init_args: {out_path, stage_name}``) — the CLI-path vector for the
    stage-callbacks bug (jsonargparse used to eager-instantiate this spec
    before salt saw the raw dict). Referenced by its full class_path from YAML.
    """

    def __init__(self, out_path: str, stage_name: str) -> None:
        self.out_path = out_path
        self.stage_name = stage_name


# a single-`fit`-stage schedule whose stage declares a scoped callback.
# ``{out}`` is formatted with a tmp path per test.
STAGE_CALLBACK_YAML = """
training_schedule:
  stages:
    fit:
      callbacks:
        - class_path: salt.tests.unit.test_main.StageCallbackProbe
          init_args:
            out_path: {out}
            stage_name: fit
"""


class _StubTrainer:
    """Minimal stand-in — the coordinator's setup/sync ``del`` the trainer arg."""


class TestTopLevelTrainingSchedule:
    def test_top_level_schedule_parses_into_model(self, data, tmp_path):
        # the example-config twin parses on the real CLI surface and builds the
        # 2-stage schedule on the instantiated SaltModule.
        override = write_yaml(tmp_path, "sched.yaml", TOP_LEVEL_SCHEDULE_YAML)
        cli = make_cli(data, extra=["--config", override])
        sched = cli.model.training_controller.schedule
        assert sched.is_multi_stage
        assert [s.name for s in sched.stages] == ["head_warmup", "full_finetune"]
        assert sched.stages[0].epochs == 5
        assert sched.stages[0].trainable == ("jets_classification",)
        assert sched.stages[0].lrs == {"max": 1.0e-4}
        assert sched.stages[1].frozen == ()
        # the top-level value was injected into the constructor arg
        assert cli.config.model.init_args.training_schedule is not None

    def test_nested_home_rejected_fail_loud(self, data, tmp_path):
        # the retired home (model.init_args.training_schedule, no top-level key)
        # is a ConfigError naming the new top-level location.
        override = write_yaml(tmp_path, "nested.yaml", NESTED_SCHEDULE_YAML)
        with pytest.raises(ConfigError, match="TOP-LEVEL config key"):
            make_cli(data, extra=["--config", override])

    def test_deep_cli_override_updates_one_stage_field(self, data, tmp_path):
        # a deep CLI override reaches into the schedule and updates one leaf,
        # leaving sibling stage fields intact.
        override = write_yaml(tmp_path, "sched.yaml", TOP_LEVEL_SCHEDULE_YAML)
        cli = make_cli(
            data,
            extra=["--config", override, "--training_schedule.stages.head_warmup.epochs=3"],
        )
        sched = cli.model.training_controller.schedule
        head = next(s for s in sched.stages if s.name == "head_warmup")
        assert head.epochs == 3  # CLI override won
        assert head.trainable == ("jets_classification",)  # sibling field survived

    def test_stacked_config_merges_per_stage_by_name(self, data, tmp_path):
        # a second --config overrides ONE field of ONE stage; the untouched stage
        # (full_finetune) and the untouched field (head_warmup.trainable) survive
        # — DeepMergeParser deep-merges the schedule per-stage-by-name.
        base = write_yaml(tmp_path, "base_sched.yaml", TOP_LEVEL_SCHEDULE_YAML)
        over = write_yaml(tmp_path, "over_sched.yaml", SCHEDULE_OVERRIDE_YAML)
        cli = make_cli(data, extra=["--config", base, "--config", over])
        sched = cli.model.training_controller.schedule
        assert [s.name for s in sched.stages] == ["head_warmup", "full_finetune"]
        head = next(s for s in sched.stages if s.name == "head_warmup")
        assert head.epochs == 2  # override won
        assert head.trainable == ("jets_classification",)  # base field survived

    def test_stacked_config_stage_null_deletes(self, data, tmp_path):
        # a `stage: null` override deletes that stage (the deep-merge deletion
        # idiom) — full_finetune is dropped, head_warmup remains.
        base = write_yaml(tmp_path, "base_sched.yaml", TOP_LEVEL_SCHEDULE_YAML)
        over = write_yaml(tmp_path, "del_sched.yaml", DELETE_STAGE_YAML)
        cli = make_cli(data, extra=["--config", base, "--config", over])
        sched = cli.model.training_controller.schedule
        assert [s.name for s in sched.stages] == ["head_warmup"]

    def test_top_level_schedule_round_trips_through_print_config(self, data, tmp_path, capsys):
        # --print_config dumps the top-level schedule AND the injected nested copy;
        # re-parsing that dump must NOT re-reject (round-trip: top-level present =>
        # the nested slot is ours) and must rebuild the same schedule. This is the
        # saved-run-config path that resume relies on (config home is location-
        # agnostic to the checkpoint's schedule.{stage_index,stage_name} payload).
        override = write_yaml(tmp_path, "sched.yaml", TOP_LEVEL_SCHEDULE_YAML)
        with pytest.raises(SystemExit) as excinfo:
            make_cli(data, extra=["--config", override, "--print_config"])
        assert excinfo.value.code == 0
        printed = capsys.readouterr().out
        printed_path = write_yaml(tmp_path, "printed.yaml", printed)
        cli_again = SaltCLI(args=["--config", printed_path], run=False)
        sched = cli_again.model.training_controller.schedule
        assert [s.name for s in sched.stages] == ["head_warmup", "full_finetune"]
        assert sched.stages[0].epochs == 5

    def test_no_schedule_is_a_noop_desugars(self, data):
        # no top-level schedule (and no nested) => the desugared single `fit` stage
        # (legacy parity path); nothing injected, no reject.
        cli = make_cli(data)
        sched = cli.model.training_controller.schedule
        assert not sched.is_multi_stage
        assert sched.initial_stage.name == "fit"
        assert getattr(cli.config.model.init_args, "training_schedule", None) is None


# main.py callback auto-injection. The TrainingScheduleCallback is
# auto-added on `fit` iff the schedule is multi-stage or freezes anything;
# never otherwise.
class TestScheduleCallbackAutoInjection:
    def test_callback_injected_for_multistage_fit(self, data, tmp_path):
        from salt.callbacks.schedule import TrainingScheduleCallback

        override = write_yaml(tmp_path, "sched.yaml", TOP_LEVEL_SCHEDULE_YAML)
        cli = make_cli(data, extra=["--config", override])
        cli.config.subcommand = "fit"  # the injector keys off the fit subcommand
        assembled = cli._maybe_add_schedule_callback([], [])  # noqa: SLF001
        assert any(isinstance(cb, TrainingScheduleCallback) for cb in assembled)

    def test_callback_absent_for_plain_fit(self, data):
        from salt.callbacks.schedule import TrainingScheduleCallback

        cli = make_cli(data)  # desugared single fit stage, no freeze
        cli.config.subcommand = "fit"
        assembled = cli._maybe_add_schedule_callback([], [])  # noqa: SLF001
        assert not any(isinstance(cb, TrainingScheduleCallback) for cb in assembled)

    def test_callback_absent_off_fit(self, data, tmp_path):
        # even a multi-stage schedule injects nothing when the subcommand is not
        # fit (test/graph/export are schedule-inert).
        from salt.callbacks.schedule import TrainingScheduleCallback

        override = write_yaml(tmp_path, "sched.yaml", TOP_LEVEL_SCHEDULE_YAML)
        cli = make_cli(data, extra=["--config", override])
        cli.config.subcommand = "test"
        assembled = cli._maybe_add_schedule_callback([], [])  # noqa: SLF001
        assert not any(isinstance(cb, TrainingScheduleCallback) for cb in assembled)

    def test_class_dict_only_requires_norm_dict_elsewhere(self, fanout_data):
        # --class_dict fans out to the tasks; norm_dict must still be supplied
        # (it is REQUIRED on the Normaliser) — here via the module config in the
        # base overrides, proving the two knobs are independent
        cli = fanout_make_cli(fanout_data, fanout_class_dict_flag(fanout_data))
        for task in CLS_TASKS:
            assert cli.model.net[task].weight_source == {"from_class_dict": str(fanout_data["cd"])}


# stage-scoped `training_schedule.stages.*.callbacks` through the REAL CLI:
# jsonargparse must NOT eager-instantiate the nested {class_path, init_args}
# spec (regression: it once did, crashing the CLI path).
class TestStageCallbacksCLI:
    def test_stage_callback_reaches_model_as_raw_spec(self, data, tmp_path):
        # a stage `callbacks:` entry must parse and arrive at the model as a
        # RAW spec dict, NOT an eager-instantiated object.
        text = STAGE_CALLBACK_YAML.format(out=tmp_path / "lr.json")
        override = write_yaml(tmp_path, "sched_cb.yaml", text)
        cli = make_cli(data, extra=["--config", override])
        stage = cli.model.training_controller.schedule.stages[0]
        assert stage.callbacks is not None
        spec = stage.callbacks[0]
        assert isinstance(spec, Mapping)  # raw spec, not a StageCallbackProbe instance
        assert not isinstance(spec, StageCallbackProbe)
        assert spec["class_path"] == "salt.tests.unit.test_main.StageCallbackProbe"
        assert spec["init_args"]["stage_name"] == "fit"

    def test_stage_callback_instantiates_at_fit_start(self, data, tmp_path):
        # the coordinator instantiates the CLI-parsed raw spec at fit start (its
        # setup validates every stage's callbacks; the active stage's delegates are
        # then built fresh) — proving the spec is usable, not merely well-formed.
        from salt.callbacks.schedule import StageScopedCallbacks

        text = STAGE_CALLBACK_YAML.format(out=tmp_path / "lr.json")
        override = write_yaml(tmp_path, "sched_cb.yaml", text)
        cli = make_cli(data, extra=["--config", override])
        coord = StageScopedCallbacks()
        coord.setup(_StubTrainer(), cli.model, "fit")  # validates all specs (no raise)
        coord._sync_active_stage(_StubTrainer(), cli.model)  # noqa: SLF001 - build delegates
        delegates = coord._active_delegates  # noqa: SLF001
        assert len(delegates) == 1
        # class-name compare (not isinstance): `_resolve_class_path` imports the test
        # module fresh, which under pytest can be a distinct module object.
        assert type(delegates[0]).__name__ == "StageCallbackProbe"
        assert delegates[0].stage_name == "fit"  # instantiated from the raw init_args

    def test_stage_callback_bad_class_path_fails_at_fit_start(self, data, tmp_path):
        # a bad class_path on a stage callback must fail at fit-start setup, not
        # silently at the boundary (import/instantiation validation).
        from salt.callbacks.schedule import StageScopedCallbacks

        text = STAGE_CALLBACK_YAML.replace(
            "salt.tests.unit.test_main.StageCallbackProbe", "salt.nonexistent.NoSuchCallback"
        ).format(out=tmp_path / "lr.json")
        override = write_yaml(tmp_path, "sched_bad.yaml", text)
        cli = make_cli(data, extra=["--config", override])
        coord = StageScopedCallbacks()
        with pytest.raises((ImportError, ModuleNotFoundError, AttributeError)):
            coord.setup(_StubTrainer(), cli.model, "fit")

    def test_stage_callback_merge_config_round_trip(self, data, tmp_path):
        # `salt merge-config` (the other CLI path that crashed) must dump the stage
        # callback spec faithfully AND render the per-stage freeze graph without
        # instantiating the spec.
        from salt.merge_config import main as merge_config_main

        text = STAGE_CALLBACK_YAML.format(out=tmp_path / "lr.json")
        override = write_yaml(tmp_path, "sched_cb.yaml", text)
        out = tmp_path / "merged.yaml"
        rc = merge_config_main([
            "--config", disable_logger_in_config(str(DUMMY_CFG)),
            *required_overrides(data),
            "--config", override,
            "--merged.output", str(out),
            "--merged.plots", "false",
        ])
        assert rc == 0
        dumped = out.read_text()
        assert "salt.tests.unit.test_main.StageCallbackProbe" in dumped  # faithful round-trip
        assert "stage_name: fit" in dumped
        # the per-stage freeze graph rendered (plots=false → .dot only) — the graph
        # path built the schedule from the raw spec without instantiating it.
        assert (tmp_path / "merged_stage00_fit.dot").exists()


# a two-stage schedule whose stages choose DIFFERENT scheduler classes.
LR_SCHEDULER_YAML = """
training_schedule:
  stages:
    warmup:
      epochs: 2
      lr_scheduler:
        class_path: torch.optim.lr_scheduler.CosineAnnealingLR
        init_args: {T_max: 2}
    full:
      lr_scheduler:
        class_path: torch.optim.lr_scheduler.ReduceLROnPlateau
        init_args: {mode: min, factor: 0.5}
        monitor: val/loss
"""


# per-stage `lr_scheduler:` through the REAL CLI (same nested-class-spec
# shape; a scheduler can never be eager-instantiated — it needs the stage
# optimizer, built at the boundary).
class TestLRSchedulerCLI:
    def test_lr_scheduler_specs_reach_model_unparsed(self, data, tmp_path):
        from salt.schedule import LRSchedulerConfig

        override = write_yaml(tmp_path, "sched_lr.yaml", LR_SCHEDULER_YAML)
        cli = make_cli(data, extra=["--config", override])
        stages = cli.model.training_controller.schedule.stages
        assert cli.model.training_controller.schedule.has_lr_scheduler
        # each stage carries a parsed LRSchedulerConfig — jsonargparse did NOT
        # eager-instantiate the nested scheduler class (which would be impossible
        # without an optimizer). The class_path survives as a string.
        assert isinstance(stages[0].lr_scheduler, LRSchedulerConfig)
        assert stages[0].lr_scheduler.class_path == "torch.optim.lr_scheduler.CosineAnnealingLR"
        assert stages[1].lr_scheduler.class_path == "torch.optim.lr_scheduler.ReduceLROnPlateau"
        assert stages[1].lr_scheduler.monitor == "val/loss"

    def test_lr_scheduler_merge_config_round_trip(self, data, tmp_path):
        from salt.merge_config import main as merge_config_main

        override = write_yaml(tmp_path, "sched_lr.yaml", LR_SCHEDULER_YAML)
        out = tmp_path / "merged.yaml"
        rc = merge_config_main([
            "--config", disable_logger_in_config(str(DUMMY_CFG)),
            *required_overrides(data),
            "--config", override,
            "--merged.output", str(out),
            "--merged.plots", "false",
        ])
        assert rc == 0
        dumped = out.read_text()
        assert "torch.optim.lr_scheduler.CosineAnnealingLR" in dumped
        assert "torch.optim.lr_scheduler.ReduceLROnPlateau" in dumped
        # a freeze graph per stage rendered without instantiating the scheduler spec
        assert (tmp_path / "merged_stage00_warmup.dot").exists()
        assert (tmp_path / "merged_stage01_full.dot").exists()

    def test_init_args_optimizer_rejected_via_cli(self, data, tmp_path):
        # a user-supplied init_args.optimizer is rejected on the CLI. The salt
        # ConfigError is raised in SaltModule.__init__ (schedule parse) and
        # jsonargparse wraps a model-instantiation failure as its Union-validation
        # ValueError — either way the bad config does not parse. (The exact
        # ConfigError message is asserted directly in test_schedule.py.)
        bad = LR_SCHEDULER_YAML.replace("init_args: {T_max: 2}", "init_args: {optimizer: foo}")
        override = write_yaml(tmp_path, "sched_bad.yaml", bad)
        with pytest.raises((ConfigError, ValueError)):
            make_cli(data, extra=["--config", override])


# norm_dict is the Normaliser module's OWN config (its sole consumer): set on
# model.modules.norm.init_args.norm_dict, NOT a top-level fan-out flag.


class TestNormDictOnModule:
    def test_norm_dict_from_module_config_lands_on_normaliser(self, fanout_data):
        # the module-config form (model.modules.norm.init_args.norm_dict, supplied
        # by fanout_base_overrides) materialises on the instantiated Normaliser
        cli = fanout_make_cli(fanout_data, [])
        assert isinstance(cli.model, SaltModule)
        assert str(cli.model.net[NORM_MODULE].norm_dict_path) == str(fanout_data["nd"])

    def test_norm_dict_does_not_touch_tasks(self, fanout_data):
        # setting norm_dict on the module leaves the classification tasks'
        # weight_source untouched (norm_dict is module-local, not multi-task)
        cli = fanout_make_cli(fanout_data, [])
        for task in CLS_TASKS:
            assert cli.model.net[task].weight_source is None

    def test_norm_dict_unknown_flag_is_rejected(self, fanout_data):
        # the retired --norm_dict flag is now an unknown arg: the parser must
        # reject it (proves the top-level fan-out flag is gone, not silently
        # ignored). The norm_dict in base_overrides is dropped here so the only
        # norm_dict surface under test is the (now-invalid) flag.
        bad_args = [a for a in fanout_base_overrides(fanout_data) if "norm_dict" not in a]
        with pytest.raises(SystemExit):
            SaltCLI(args=[*bad_args, f"--norm_dict={fanout_data['nd']}"], run=False)


class TestInitFromCLI:
    """The --init_from warm-start flag."""

    def _fit_args(self, data, extra: list[str]) -> list[str]:
        cfg = disable_logger_in_config(str(DUMMY_CFG))
        return [
            "fit", "--config", cfg, *required_overrides(data),
            "--trainer.accelerator=cpu", "--trainer.logger=false",
            "--trainer.fast_dev_run=1", "--callbacks.progress=null", *extra,
        ]

    def test_init_from_and_ckpt_path_mutually_exclusive(self, data, tmp_path):
        # a bogus-but-present path for each: the guard fires in
        # before_instantiate_classes, before either file is opened
        fake = tmp_path / "fake.ckpt"
        with pytest.raises(ConfigError, match="mutually exclusive"):
            SaltCLI(args=self._fit_args(
                data, [f"--init_from={fake}", f"--ckpt_path={fake}"]
            ))

    def test_init_from_flag_is_registered(self, data, tmp_path):
        # the flag is accepted + parsed by the real CLI surface (registered like
        # --class_dict); the full model plumbing is covered by the integration
        # tests (test_init_from.py).
        fake = tmp_path / "fake.ckpt"
        cli = make_cli(data, extra=[f"--init_from={fake}"])
        assert str(cli.config.get("init_from")) == str(fake)


# run-dir layout, native --auto_resume, step-based checkpointing (main.py:
# run_root/run_dir_path/latest_checkpoint/auto_resume_requested + the
# before_instantiate_classes fit-branch wiring).


class StopAfterFirstEpoch(Callback):
    """Cuts a fit after epoch 0 so a resume leg can continue it.

    Referenced by class_path ``salt.tests.unit.test_main.StopAfterFirstEpoch``
    (precedent: `StageCallbackProbe`, above).
    """

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        trainer.should_stop = True


def _fit_args(data, root: Path, *extra: str) -> list[str]:
    """`TestFitSmoke`'s CPU/tiny recipe, parametrised by run dir + extra flags.

    ``--trainer.max_epochs`` is deliberately NOT included — every call site
    passes it explicitly (the OneCycleLR total_steps constraint means a
    producer/resume pair must agree on it).
    """
    return [
        "fit",
        "--config",
        str(DUMMY_CFG),
        *required_overrides(data),
        f"--trainer.default_root_dir={root}",
        "--trainer.accelerator=cpu",
        "--trainer.logger=false",
        "--trainer.limit_train_batches=2",
        "--trainer.limit_val_batches=2",
        "--trainer.num_sanity_val_steps=0",
        "--trainer.log_every_n_steps=1",
        "--callbacks.progress=null",
        *extra,
    ]


def _newest_ckpt(run_dir: Path) -> Path:
    """The checkpoint under `run_dir` with the highest ``(epoch, step)``."""
    ckpts = list(run_dir.rglob("*.ckpt"))
    assert ckpts, f"no checkpoint under {run_dir}"

    def key(path: Path) -> tuple[int, int]:
        epoch = int(re.search(r"epoch=(\d+)", path.name).group(1))
        step = int(re.search(r"step=(\d+)", path.name).group(1))
        return (epoch, step)

    return max(ckpts, key=key)


def _run_dirs(root: Path, name: str = NAME) -> list[Path]:
    """Sibling ``<name>_*`` run dirs directly under `root`, sorted by name."""
    return sorted(d for d in root.glob(f"{name}_*") if d.is_dir())


class TestRunDirLayout:
    def test_no_root_is_logs(self):
        assert run_root(None, "n") == Path("logs")

    def test_strips_trailing_timestamp(self):
        assert run_root("/x/n_20260912-T134455", "n") == Path("/x")

    def test_strips_trailing_timestamp_with_underscored_name(self):
        # name-agnostic: only the text after the LAST '_' must parse
        assert run_root("/x/my_run_20260912-T134455", "my_run") == Path("/x")

    def test_strips_exact_suffix_only_when_it_matches(self):
        assert run_root("/x/n_foo", "n", "foo") == Path("/x")
        assert run_root("/x/n_foo", "n") == Path("/x/n_foo")  # no suffix given
        assert run_root("/x/n_bar", "n", "foo") == Path("/x/n_bar")  # wrong suffix

    def test_run_dir_path_with_suffix(self):
        assert run_dir_path("/x", "n", "foo") == Path("/x/n_foo")

    def test_run_dir_path_without_suffix_matches_timestamp_pattern(self):
        name = run_dir_path("/x", "n").name
        assert re.match(r"^n_\d{8}-T\d{6}$", name), name

    def test_run_dir_path_repass_yields_sibling(self):
        assert run_dir_path("/x/n_20260912-T134455", "n").parent == Path("/x")

    def test_log_suffix_restack_stays_one_run_dir(self, data, tmp_path):
        assert main(_fit_args(data, tmp_path, "--trainer.max_epochs=1", "--log_suffix=foo")) == 0
        run_dir = tmp_path / f"{NAME}_foo"
        assert run_dir.is_dir()
        saved_config = run_dir / "config.yaml"
        assert saved_config.is_file()
        # a second fit stacks the SAVED config.yaml (its own default_root_dir
        # already points AT the suffix run dir) as an extra --config
        rc = main([
            "fit",
            "--config",
            str(DUMMY_CFG),
            *required_overrides(data),
            "--config",
            str(saved_config),
            "--trainer.accelerator=cpu",
            "--trainer.logger=false",
            "--trainer.max_epochs=1",
            "--trainer.limit_train_batches=2",
            "--trainer.limit_val_batches=2",
            "--trainer.num_sanity_val_steps=0",
            "--trainer.log_every_n_steps=1",
            "--callbacks.progress=null",
            "--log_suffix=foo",
        ])
        assert rc == 0
        assert _run_dirs(tmp_path) == [run_dir]  # still exactly one, nothing nested
        assert not list(tmp_path.glob(f"{NAME}_*/{NAME}_*"))

    def test_timestamp_restack_yields_sibling_not_nested(self, data, tmp_path):
        assert main(_fit_args(data, tmp_path, "--trainer.max_epochs=1")) == 0
        run_dirs_a = _run_dirs(tmp_path)
        assert len(run_dirs_a) == 1
        saved_config = run_dirs_a[0] / "config.yaml"
        time.sleep(1.1)  # guarantee run B mints a distinct timestamp
        rc = main([
            "fit",
            "--config",
            str(DUMMY_CFG),
            *required_overrides(data),
            "--config",
            str(saved_config),
            "--trainer.accelerator=cpu",
            "--trainer.logger=false",
            "--trainer.max_epochs=1",
            "--trainer.limit_train_batches=2",
            "--trainer.limit_val_batches=2",
            "--trainer.num_sanity_val_steps=0",
            "--trainer.log_every_n_steps=1",
            "--callbacks.progress=null",
        ])
        assert rc == 0
        run_dirs_b = _run_dirs(tmp_path)
        assert len(run_dirs_b) == 2  # two SIBLINGS, not one nested inside the other
        assert not list(tmp_path.glob(f"{NAME}_*/{NAME}_*"))


class TestLatestCheckpoint:
    NAME = "run"

    def test_empty_root_is_none(self, tmp_path):
        assert latest_checkpoint(tmp_path, self.NAME) is None

    def test_missing_root_is_none(self, tmp_path):
        assert latest_checkpoint(tmp_path / "does_not_exist", self.NAME) is None

    def test_mid_epoch_step_beats_epoch_end_checkpoints(self, tmp_path):
        a = tmp_path / f"{self.NAME}_a" / "ckpts"
        b = tmp_path / f"{self.NAME}_b" / "checkpoints"
        a.mkdir(parents=True)
        b.mkdir(parents=True)
        winner = a / "epoch=001-step=7.ckpt"
        winner.touch()
        (a / "epoch=001-step=5-loss=0.1.ckpt").touch()
        (b / "epoch=000-step=9-loss=0.01.ckpt").touch()
        assert latest_checkpoint(tmp_path, self.NAME) == winner

    def test_same_epoch_higher_step_wins(self, tmp_path):
        ckpts = tmp_path / f"{self.NAME}_a" / "ckpts"
        ckpts.mkdir(parents=True)
        (ckpts / "epoch=002-step=3-loss=0.2.ckpt").touch()
        high = ckpts / "epoch=002-step=8-loss=0.1.ckpt"
        high.touch()
        assert latest_checkpoint(tmp_path, self.NAME) == high

    def test_unparseable_names_ignored(self, tmp_path):
        ckpts = tmp_path / f"{self.NAME}_a" / "ckpts"
        ckpts.mkdir(parents=True)
        (ckpts / "last.ckpt").touch()
        (ckpts / "epoch=3.ckpt").touch()  # no step=
        (ckpts / "garbage.ckpt").touch()
        assert latest_checkpoint(tmp_path, self.NAME) is None

    def test_other_run_dir_not_scanned(self, tmp_path):
        other = tmp_path / "other_x" / "ckpts"
        other.mkdir(parents=True)
        (other / "epoch=005-step=10.ckpt").touch()
        assert latest_checkpoint(tmp_path, self.NAME) is None

    def test_mtime_tie_break(self, tmp_path):
        ckpts = tmp_path / f"{self.NAME}_a" / "ckpts"
        ckpts.mkdir(parents=True)
        older = ckpts / "epoch=001-step=4-a.ckpt"
        newer = ckpts / "epoch=001-step=4-b.ckpt"
        older.touch()
        newer.touch()
        old_time = time.time() - 100
        os.utime(older, (old_time, old_time))
        assert latest_checkpoint(tmp_path, self.NAME) == newer

    def test_step_only_beats_best_checkpoints_disregard_of_it(self, tmp_path):
        # _best_checkpoint (loss=-keyed) ignores the step-only file at the
        # SAME step; latest_checkpoint accepts it as a legitimate candidate.
        run_dir = tmp_path / f"{self.NAME}_a"
        ckpts = run_dir / "ckpts"
        ckpts.mkdir(parents=True)
        step_only = ckpts / "epoch=000-step=3.ckpt"
        epoch_end = ckpts / "epoch=000-step=3-loss=0.5.ckpt"
        step_only.touch()
        epoch_end.touch()
        assert _best_checkpoint(run_dir / "config.yaml").endswith(
            "epoch=000-step=3-loss=0.5.ckpt"
        )
        assert latest_checkpoint(tmp_path, self.NAME) in (step_only, epoch_end)


class TestAutoResume:
    """Each test uses its own ``tmp_path`` root."""

    @staticmethod
    def _leg_a(data, root: Path, *extra: str) -> list[str]:
        return _fit_args(
            data,
            root,
            "--trainer.max_epochs=2",
            "--callbacks.stop.class_path=salt.tests.unit.test_main.StopAfterFirstEpoch",
            *extra,
        )

    def test_flag_resumes_from_sibling_checkpoint(self, data, tmp_path, capsys):
        root = tmp_path
        assert main(self._leg_a(data, root)) == 0
        a_dir = _run_dirs(root)[0]
        a_ckpt = _newest_ckpt(a_dir)
        a_state = torch.load(a_ckpt, map_location="cpu", weights_only=False)
        assert a_state["epoch"] == 0
        assert a_state["global_step"] == 2
        capsys.readouterr()  # drain leg A's stdout

        time.sleep(1.1)  # guarantee leg B mints a distinct timestamp
        assert main(_fit_args(data, root, "--trainer.max_epochs=2", "--auto_resume")) == 0
        out = capsys.readouterr().out
        assert f"auto-resume: resuming run '{NAME}' from " in out
        assert str(a_ckpt) in out

        run_dirs_after = _run_dirs(root)
        assert len(run_dirs_after) == 2  # a NEW sibling, not a reuse of A's dir
        b_dir = next(d for d in run_dirs_after if d != a_dir)
        assert not list(b_dir.rglob("epoch=000*"))  # never trained epoch 0
        b_ckpt = _newest_ckpt(b_dir)
        b_state = torch.load(b_ckpt, map_location="cpu", weights_only=False)
        assert b_state["epoch"] == 1
        assert b_state["global_step"] == 4
        assert b_ckpt.name.startswith("epoch=001-step=4-")

        saved = yaml.safe_load((b_dir / "config.yaml").read_text())
        assert saved["ckpt_path"] is None  # the resolved path never leaked
        assert saved["auto_resume"] is True

    @pytest.mark.parametrize("value", ["1", "TRUE", "yes"])
    def test_env_variants_resume(self, data, tmp_path, capsys, monkeypatch, value):
        root = tmp_path
        assert main(self._leg_a(data, root)) == 0
        a_dir = _run_dirs(root)[0]
        a_ckpt = _newest_ckpt(a_dir)
        capsys.readouterr()

        monkeypatch.setenv(SALT_AUTO_RESUME_ENV, value)
        time.sleep(1.1)
        assert main(_fit_args(data, root, "--trainer.max_epochs=2")) == 0
        out = capsys.readouterr().out
        assert f"auto-resume: resuming run '{NAME}' from " in out
        assert str(a_ckpt) in out

        run_dirs_after = _run_dirs(root)
        assert len(run_dirs_after) == 2
        b_dir = next(d for d in run_dirs_after if d != a_dir)
        assert not list(b_dir.rglob("epoch=000*"))
        b_state = torch.load(_newest_ckpt(b_dir), map_location="cpu", weights_only=False)
        assert b_state["epoch"] == 1
        assert b_state["global_step"] == 4

    def test_env_falsy_zero_trains_from_scratch(self, data, tmp_path, capsys, monkeypatch):
        root = tmp_path
        assert main(self._leg_a(data, root)) == 0
        a_dir = _run_dirs(root)[0]
        capsys.readouterr()

        monkeypatch.setenv(SALT_AUTO_RESUME_ENV, "0")
        time.sleep(1.1)
        assert main(_fit_args(data, root, "--trainer.max_epochs=2")) == 0
        out = capsys.readouterr().out
        assert "auto-resume:" not in out

        run_dirs_after = _run_dirs(root)
        assert len(run_dirs_after) == 2
        b_dir = next(d for d in run_dirs_after if d != a_dir)
        # no --auto_resume, no stop callback -> a full fresh 2-epoch run
        assert list(b_dir.rglob("epoch=000*"))
        b_state = torch.load(_newest_ckpt(b_dir), map_location="cpu", weights_only=False)
        assert b_state["epoch"] == 1
        assert b_state["global_step"] == 4

    def test_no_sibling_starts_fresh(self, data, tmp_path, capsys):
        rc = main(_fit_args(data, tmp_path, "--trainer.max_epochs=1", "--auto_resume"))
        assert rc == 0
        out = capsys.readouterr().out
        assert (
            f"auto-resume: no checkpoint of run '{NAME}' under {tmp_path} — starting fresh"
            in out
        )

    def test_no_sibling_falls_back_to_ckpt_path(self, data, tmp_path, capsys):
        root_a = tmp_path / "root_a"
        root_b = tmp_path / "root_b"
        root_a.mkdir()
        root_b.mkdir()
        assert main(self._leg_a(data, root_a)) == 0
        a_ckpt = _newest_ckpt(_run_dirs(root_a)[0])
        capsys.readouterr()

        rc = main(_fit_args(
            data, root_b, "--trainer.max_epochs=2", "--auto_resume", f"--ckpt_path={a_ckpt}"
        ))
        assert rc == 0
        out = capsys.readouterr().out
        assert "falling back to --ckpt_path" in out
        assert str(a_ckpt) in out

        b_state = torch.load(
            _newest_ckpt(_run_dirs(root_b)[0]), map_location="cpu", weights_only=False
        )
        assert b_state["global_step"] == 4
        assert b_state["epoch"] == 1

    def test_sibling_checkpoint_beats_ckpt_path(self, data, tmp_path, capsys):
        root = tmp_path / "root"
        root_d = tmp_path / "root_d"
        root.mkdir()
        root_d.mkdir()
        assert main(self._leg_a(data, root)) == 0
        a_dir = _run_dirs(root)[0]
        a_ckpt = _newest_ckpt(a_dir)

        rc = main(_fit_args(
            data,
            root_d,
            "--trainer.limit_train_batches=1",
            "--trainer.max_epochs=2",
            "--callbacks.stop.class_path=salt.tests.unit.test_main.StopAfterFirstEpoch",
        ))
        assert rc == 0
        d_ckpt = _newest_ckpt(_run_dirs(root_d)[0])
        assert d_ckpt.name.startswith("epoch=000-step=1-")
        capsys.readouterr()

        time.sleep(1.1)
        rc = main(_fit_args(
            data, root, "--trainer.max_epochs=2", "--auto_resume", f"--ckpt_path={d_ckpt}"
        ))
        assert rc == 0
        out = capsys.readouterr().out
        assert f"resuming run '{NAME}' from {a_ckpt}" in out
        assert f"(overriding --ckpt_path {d_ckpt})" in out

        run_dirs_after = _run_dirs(root)
        assert len(run_dirs_after) == 2
        b_dir = next(d for d in run_dirs_after if d != a_dir)
        b_state = torch.load(_newest_ckpt(b_dir), map_location="cpu", weights_only=False)
        assert b_state["global_step"] == 4  # A's 2 steps, not D's 1

    def test_init_from_and_resolved_resume_mutually_exclusive(self, data, tmp_path):
        root = tmp_path
        assert main(self._leg_a(data, root)) == 0
        a_ckpt = _newest_ckpt(_run_dirs(root)[0])

        cfg = disable_logger_in_config(str(DUMMY_CFG))
        args = [
            "fit",
            "--config",
            cfg,
            *required_overrides(data),
            "--trainer.accelerator=cpu",
            "--trainer.logger=false",
            "--trainer.fast_dev_run=1",
            "--callbacks.progress=null",
            f"--trainer.default_root_dir={root}",
            "--auto_resume",
            f"--init_from={a_ckpt}",
        ]
        with pytest.raises(ConfigError, match="auto-resume"):
            SaltCLI(args=args)

    def test_log_suffix_with_auto_resume_writes_into_same_dir(self, data, tmp_path, capsys):
        root = tmp_path
        assert main(self._leg_a(data, root, "--log_suffix=foo")) == 0
        run_dir = root / f"{NAME}_foo"
        assert run_dir.is_dir()
        a_ckpt = _newest_ckpt(run_dir)
        assert a_ckpt.name.startswith("epoch=000-step=2-")
        capsys.readouterr()

        rc = main(_fit_args(
            data, root, "--trainer.max_epochs=2", "--log_suffix=foo", "--auto_resume"
        ))
        assert rc == 0
        out = capsys.readouterr().out
        assert f"resuming run '{NAME}' from {a_ckpt}" in out

        assert _run_dirs(root) == [run_dir]  # same dir — no new sibling
        b_state = torch.load(_newest_ckpt(run_dir), map_location="cpu", weights_only=False)
        assert b_state["global_step"] == 4
        assert b_state["epoch"] == 1


class TestStepCheckpointFit:
    @staticmethod
    def _args(data, root: Path, save_top_k: int | None = None) -> list[str]:
        extra = [
            "--trainer.max_epochs=2",
            "--trainer.limit_train_batches=3",
            "--callbacks.stop.class_path=salt.tests.unit.test_main.StopAfterFirstEpoch",
            "--callbacks.step_checkpoint.class_path=salt.callbacks.StepCheckpoint",
            "--callbacks.step_checkpoint.init_args.every_n_train_steps=1",
        ]
        if save_top_k is not None:
            extra.append(f"--callbacks.step_checkpoint.init_args.save_top_k={save_top_k}")
        return _fit_args(data, root, *extra)

    @staticmethod
    def _step_only_names(run_dir: Path) -> set[str]:
        pattern = re.compile(r"^epoch=\d{3}-step=\d+\.ckpt$")
        return {p.name for p in (run_dir / "ckpts").glob("*.ckpt") if pattern.match(p.name)}

    def test_keep_latest_one_default_pruning(self, data, tmp_path):
        assert main(self._args(data, tmp_path)) == 0
        run_dir = _run_dirs(tmp_path)[0]
        assert self._step_only_names(run_dir) == {"epoch=000-step=3.ckpt"}
        assert list((run_dir / "ckpts").glob("epoch=000-step=3-loss=*.ckpt"))

    def test_save_top_k_minus_one_keeps_all(self, data, tmp_path):
        assert main(self._args(data, tmp_path, save_top_k=-1)) == 0
        run_dir = _run_dirs(tmp_path)[0]
        assert self._step_only_names(run_dir) == {
            "epoch=000-step=1.ckpt",
            "epoch=000-step=2.ckpt",
            "epoch=000-step=3.ckpt",
        }

    def test_auto_resume_picks_mid_epoch_step_checkpoint(self, data, tmp_path, capsys):
        root = tmp_path
        assert main(self._args(data, root, save_top_k=-1)) == 0
        capsys.readouterr()

        time.sleep(1.1)
        rc = main(_fit_args(
            data,
            root,
            "--trainer.max_epochs=2",
            "--trainer.limit_train_batches=3",
            "--auto_resume",
        ))
        assert rc == 0
        out = capsys.readouterr().out
        assert "auto-resume: resuming" in out
        assert "step=3" in out
        # NOT asserting the absence of an epoch=000 file here: resuming from a
        # mid-epoch step-only checkpoint legitimately re-emits an epoch-0
        # epoch-end checkpoint on the way through the rest of that epoch.

        run_dirs = _run_dirs(root)
        assert len(run_dirs) == 2
        b_state = torch.load(
            _newest_ckpt(run_dirs[-1]), map_location="cpu", weights_only=False
        )
        assert b_state["epoch"] == 1
        assert b_state["global_step"] == 6
