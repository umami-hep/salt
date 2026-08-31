"""jsonargparse capabilities the salt dict-of-modules config mechanism relies on."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace

MODULE = __name__

# Toy class hierarchy (stands in for NetModule / Processor / Writer)


class ToyModule:
    """Toy stand-in for the salt v2 module base class."""

    def __init__(self, dim: int = 8, name: str = "x"):
        self.dim = dim
        self.name = name


class Encoder(ToyModule):
    """Subclass with an extra init arg, to prove subclass-specific args parse."""

    def __init__(self, dim: int = 8, name: str = "x", heads: int = 2):
        super().__init__(dim, name)
        self.heads = heads


class Decoder(ToyModule):
    """Second subclass, used for dict values and replacement checks."""


class Model:
    """Holder with the ``dict[str, Base]`` annotation."""

    def __init__(self, modules: dict[str, ToyModule] | None = None):
        self.modules = modules or {}


class ModelOptionalValues:
    """Holder whose dict values admit None — the null-deletion workaround."""

    def __init__(self, modules: dict[str, ToyModule | None] | None = None):
        self.modules = modules or {}


# Toy callback hierarchy (capability 7, callbacks-dict assembly)


class Callback:
    """Toy stand-in for lightning.Callback."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose


class Checkpoint(Callback):
    """Framework callback with the monitor-override use case."""

    def __init__(self, monitor: str = "val_loss", verbose: bool = False):
        super().__init__(verbose)
        self.monitor = monitor


class LRMonitor(Callback):
    """Second framework callback."""


class ProgressBar(Callback):
    """Stand-in for a stock Lightning callback listed under trainer.callbacks."""


# Remediation shim: deep-merge for dict-typed values


class DeepMergeParser(ArgumentParser):
    """ArgumentParser whose config-file merge deep-merges dict-typed values."""

    def merge_config(self, cfg_from: Namespace, cfg_to: Namespace) -> Namespace:
        """Union dict leaves key-by-key before the standard namespace merge."""
        for key, val_from in list(cfg_from.items()):
            if not isinstance(val_from, dict):
                continue
            val_to = cfg_to.get(key)
            if isinstance(val_to, dict):
                merged = {**val_to, **val_from}
                cfg_from[key] = {k: v for k, v in merged.items() if v is not None}
        return super().merge_config(cfg_from, cfg_to)


# Helpers

YAML_BASE = f"""
model:
  modules:
    encoder:
      class_path: {MODULE}.Encoder
      init_args:
        dim: 64
        heads: 4
    decoder:
      class_path: {MODULE}.Decoder
      init_args:
        dim: 32
"""

YAML_ADD_HEAD = f"""
model:
  modules:
    head:
      class_path: {MODULE}.Decoder
      init_args:
        dim: 16
"""

YAML_NULL_DECODER = """
model:
  modules:
    decoder: null
"""

YAML_UPDATE_ENCODER = """
model:
  modules:
    encoder:
      init_args:
        dim: 999
"""


def make_parser(holder: type = Model, *, deep_merge: bool = False, **kwargs) -> ArgumentParser:
    parser_cls = DeepMergeParser if deep_merge else ArgumentParser
    parser = parser_cls(exit_on_error=False, **kwargs)
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_class_arguments(holder, "model")
    return parser


def write_yaml(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


# Capability 1: basic dict[str, Base] instantiation from YAML


def test_dict_of_base_instantiation_from_yaml(tmp_path):
    parser = make_parser()
    cfg = parser.parse_args(["--config", write_yaml(tmp_path, "base.yaml", YAML_BASE)])
    init = parser.instantiate_classes(cfg)

    modules = init.model.modules
    assert list(modules) == ["encoder", "decoder"]  # YAML insertion order preserved
    assert isinstance(modules["encoder"], Encoder)
    assert isinstance(modules["decoder"], Decoder)
    assert modules["encoder"].dim == 64
    assert modules["encoder"].heads == 4  # subclass-specific arg parsed
    assert modules["encoder"].name == "x"  # base default filled in
    assert modules["decoder"].dim == 32


def test_dict_of_base_rejects_bad_subclass(tmp_path):
    """Subclass validation is enforced: a non-ToyModule class path is an error."""
    bad = f"""
model:
  modules:
    encoder:
      class_path: {MODULE}.Callback
"""
    parser = make_parser()
    with pytest.raises(Exception, match=r"(?i)subclass|not.*valid"):
        parser.parse_args(["--config", write_yaml(tmp_path, "bad.yaml", bad)])


# Capability 2: --print_config round-trip


def test_print_config_round_trip(tmp_path, capsys):
    parser = make_parser()
    config = write_yaml(tmp_path, "base.yaml", YAML_BASE)

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--config", config, "--print_config"])
    assert excinfo.value.code == 0
    printed = capsys.readouterr().out

    # the dict serialises with class_path/init_args blocks intact
    assert f"class_path: {MODULE}.Encoder" in printed
    assert "heads: 4" in printed

    # feeding the printed config back yields an identical model namespace
    cfg_orig = parser.parse_args(["--config", config])
    cfg_again = parser.parse_string(printed)
    assert cfg_again.model == cfg_orig.model
    assert parser.dump(cfg_again) == parser.dump(cfg_orig)


# Capability 3: null-deletion (XFAIL natively + passing workarounds)


@pytest.mark.xfail(
    strict=True,
    reason="jsonargparse 4.46.0: None fails dict[str, Base] value validation, and a "
    "second config file replaces the dict wholesale (merge_config -> Namespace.update "
    "treats dict leaves atomically). Null-deletion needs the DeepMergeParser "
    "shim plus `| None` value types — see the workaround tests below.",
)
def test_null_deletion_via_second_config_file(tmp_path):
    """A second config file setting modules.decoder=null removes it."""
    parser = make_parser()
    cfg = parser.parse_args(
        [
            "--config",
            write_yaml(tmp_path, "base.yaml", YAML_BASE),
            "--config",
            write_yaml(tmp_path, "null.yaml", YAML_NULL_DECODER),
        ]
    )
    assert list(cfg.model.modules) == ["encoder"]


def test_null_deletion_workaround_merge_config_override(tmp_path):
    """DeepMergeParser + `| None` values: file-level null deletes the key."""
    parser = make_parser(ModelOptionalValues, deep_merge=True)
    cfg = parser.parse_args(
        [
            "--config",
            write_yaml(tmp_path, "base.yaml", YAML_BASE),
            "--config",
            write_yaml(tmp_path, "null.yaml", YAML_NULL_DECODER),
        ]
    )
    assert list(cfg.model.modules) == ["encoder"]
    init = parser.instantiate_classes(cfg)
    assert isinstance(init.model.modules["encoder"], Encoder)


def test_null_deletion_workaround_optional_values_cli(tmp_path):
    """CLI dotted null with `| None` values: entry becomes None, framework filters."""
    parser = make_parser(ModelOptionalValues)
    cfg = parser.parse_args(
        ["--config", write_yaml(tmp_path, "base.yaml", YAML_BASE), "--model.modules.decoder=null"]
    )
    init = parser.instantiate_classes(cfg)
    assert init.model.modules["decoder"] is None  # sibling keys survive
    assert isinstance(init.model.modules["encoder"], Encoder)
    # the framework-side assembly step drops None entries (deletion semantics)
    live = {k: v for k, v in init.model.modules.items() if v is not None}
    assert list(live) == ["encoder"]


# Capability 4: dotted CLI overrides into dict values


def test_dotted_cli_override_into_dict_value(tmp_path):
    """--model.modules.encoder.init_args.dim=128 updates one entry."""
    parser = make_parser()
    config = write_yaml(tmp_path, "base.yaml", YAML_BASE)

    cfg = parser.parse_args(["--config", config, "--model.modules.encoder.init_args.dim=128"])
    init = parser.instantiate_classes(cfg)
    assert init.model.modules["encoder"].dim == 128
    assert init.model.modules["encoder"].heads == 4  # other init_args survive
    assert isinstance(init.model.modules["decoder"], Decoder)  # sibling keys survive

    # the short form (without init_args) also works
    cfg = parser.parse_args(["--config", config, "--model.modules.encoder.dim=96"])
    assert cfg.model.modules["encoder"].init_args.dim == 96

    # a dotted override can also ADD a new key, merging with existing ones
    new_block = json.dumps({"class_path": f"{MODULE}.Decoder", "init_args": {"dim": 16}})
    cfg = parser.parse_args(["--config", config, f"--model.modules.head={new_block}"])
    assert list(cfg.model.modules) == ["encoder", "decoder", "head"]


# Capability 5: env-var overrides


def test_env_var_override(tmp_path, monkeypatch):
    """default_env=True: SALT_MODEL__MODULES sets the whole dict from JSON/YAML."""
    env_value = json.dumps({"envenc": {"class_path": f"{MODULE}.Encoder", "init_args": {"dim": 7}}})
    monkeypatch.setenv("SALT_MODEL__MODULES", env_value)

    parser = make_parser(default_env=True, env_prefix="SALT")
    cfg = parser.parse_args([])
    init = parser.instantiate_classes(cfg)
    assert list(init.model.modules) == ["envenc"]
    assert isinstance(init.model.modules["envenc"], Encoder)
    assert init.model.modules["envenc"].dim == 7

    # precedence: a config file given on the command line beats the env var
    cfg = parser.parse_args(["--config", write_yaml(tmp_path, "base.yaml", YAML_BASE)])
    assert list(cfg.model.modules) == ["encoder", "decoder"]


# Capability 6: deep-merge across two config files (XFAIL natively + workarounds)


@pytest.mark.xfail(
    strict=True,
    reason="jsonargparse 4.46.0: merge_config -> Namespace.update replaces dict leaves "
    "wholesale, so a second config file drops earlier dict keys. The 'module "
    "dicts merge' semantics need the DeepMergeParser shim — see the workaround tests below.",
)
def test_deep_merge_across_config_files(tmp_path):
    """A later file adds a key, earlier keys survive."""
    parser = make_parser()
    cfg = parser.parse_args(
        [
            "--config",
            write_yaml(tmp_path, "base.yaml", YAML_BASE),
            "--config",
            write_yaml(tmp_path, "add.yaml", YAML_ADD_HEAD),
        ]
    )
    assert list(cfg.model.modules) == ["encoder", "decoder", "head"]


def test_deep_merge_workaround_merge_config_override(tmp_path):
    """DeepMergeParser restores the intended semantics: add a key, keep the others."""
    parser = make_parser(deep_merge=True)
    cfg = parser.parse_args(
        [
            "--config",
            write_yaml(tmp_path, "base.yaml", YAML_BASE),
            "--config",
            write_yaml(tmp_path, "add.yaml", YAML_ADD_HEAD),
        ]
    )
    init = parser.instantiate_classes(cfg)
    assert list(init.model.modules) == ["encoder", "decoder", "head"]
    assert init.model.modules["encoder"].heads == 4
    assert init.model.modules["head"].dim == 16


def test_deep_merge_workaround_updates_entry_in_place(tmp_path):
    """DeepMergeParser + native per-entry merge: later file updates one init_arg."""
    parser = make_parser(deep_merge=True)
    cfg = parser.parse_args(
        [
            "--config",
            write_yaml(tmp_path, "base.yaml", YAML_BASE),
            "--config",
            write_yaml(tmp_path, "update.yaml", YAML_UPDATE_ENCODER),
        ]
    )
    init = parser.instantiate_classes(cfg)
    assert list(init.model.modules) == ["encoder", "decoder"]  # siblings survive
    assert init.model.modules["encoder"].dim == 999  # updated
    assert init.model.modules["encoder"].heads == 4  # other init_args survive


def test_per_entry_init_args_merge_is_native(tmp_path):
    """Documents native 4.46.0 semantics: per-entry merge works, sibling keys drop."""
    parser = make_parser()
    cfg = parser.parse_args(
        [
            "--config",
            write_yaml(tmp_path, "base.yaml", YAML_BASE),
            "--config",
            write_yaml(tmp_path, "update.yaml", YAML_UPDATE_ENCODER),
        ]
    )
    # sibling key 'decoder' is LOST (whole-dict replacement)...
    assert list(cfg.model.modules) == ["encoder"]
    # ...but the surviving entry merged with its previous value:
    encoder = cfg.model.modules["encoder"]
    assert encoder.class_path == f"{MODULE}.Encoder"  # inherited, not restated
    assert encoder.init_args.dim == 999  # updated
    assert encoder.init_args.heads == 4  # survived from the first file


# Capability 7: callbacks-dict assembly (trainer.callbacks pattern)


def test_callbacks_dict_assembly(tmp_path):
    """Parse dict[str, Callback] + stock trainer.callbacks list, assemble one list."""
    parser = ArgumentParser(exit_on_error=False)
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--callbacks", type=dict[str, Callback], default={})
    parser.add_argument("--trainer.callbacks", type=list[Callback] | None, default=None)

    config = write_yaml(
        tmp_path,
        "callbacks.yaml",
        f"""
callbacks:
  checkpoint:
    class_path: {MODULE}.Checkpoint
    init_args:
      monitor: val/jets_classification_loss
  lr_monitor:
    class_path: {MODULE}.LRMonitor
trainer:
  callbacks:
    - class_path: {MODULE}.ProgressBar
""",
    )

    # the one-key override into the callbacks dict
    cfg = parser.parse_args(
        ["--config", config, "--callbacks.checkpoint.init_args.monitor=val/other_loss"]
    )
    init = parser.instantiate_classes(cfg)

    # framework-side assembly: dict values (insertion order) + stock list entries
    assembled = [*init.callbacks.values(), *(init.trainer.callbacks or [])]
    assert [type(cb) for cb in assembled] == [Checkpoint, LRMonitor, ProgressBar]
    assert assembled[0].monitor == "val/other_loss"
