"""Deterministic reproducer for jsonargparse's ``sys.modules`` walk race (``RuntimeError:
dictionary changed size during iteration``) and salt's import-time retry patch
(`salt.main._patch_jsonargparse_sys_modules_race`)."""

from __future__ import annotations

import contextlib
import sys
import types
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import comet_ml
import pytest
from jsonargparse import _postponed_annotations as pa  # noqa: PLC2701 - the patch site under test

import salt.main
from salt.schema import dump_schema, save_schema
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict
from salt.tests._fixtures.gn2v2_test_config import small_config
from salt.testing.inputs import write_dummy_file

_RACE_MESSAGE = "changed size during iteration"
_PATCH_RETRIES = 40  # salt.main._patch_jsonargparse_sys_modules_race retry budget
_COMET_ENV = (
    "COMET_API_KEY",
    "COMET_EXPERIMENT_NAME",
    "COMET_OFFLINE_DIRECTORY",
    "COMET_DISABLE_AUTO_LOGGING",
)


class _SysModulesChurnProbe:
    """A ``sys.modules`` entry whose ``__dict__`` access (jsonargparse's walk calls
    ``vars(mod)``) inserts a dummy module while ``remaining > 0``, so the dict iterator
    raises on its next step wherever the probe sits — the failure a concurrently
    importing thread (e.g. Comet's) causes. Returns ``{}``, binding nothing.
    """

    __slots__ = ("accesses", "inserted", "remaining")

    def __init__(self, mutations: int) -> None:
        self.accesses = 0
        self.remaining = mutations
        self.inserted: list[str] = []

    @property
    def __dict__(self) -> dict[str, Any]:  # type: ignore[override]
        self.accesses += 1
        if self.remaining > 0:
            self.remaining -= 1
            name = f"_salt_race_dummy_{uuid.uuid4().hex}"
            self.inserted.append(name)
            sys.modules[name] = types.ModuleType(name)
        return {}


@contextlib.contextmanager
def _armed_probe(mutations: int) -> Iterator[_SysModulesChurnProbe]:
    """Insert an armed probe into ``sys.modules``; always remove it and its dummies on exit."""
    probe = _SysModulesChurnProbe(mutations)
    key = f"_salt_race_probe_{uuid.uuid4().hex}"
    sys.modules[key] = probe  # type: ignore[assignment]
    try:
        yield probe
    finally:
        sys.modules.pop(key, None)
        for name in probe.inserted:
            sys.modules.pop(name, None)


def _enrich_with_fresh_alias(helper: Callable[[dict[str, Any]], None]) -> None:
    """Call ``helper`` on a fresh, unresolvable alias: a new object misses jsonargparse's
    id-keyed ``_TRIGGER_MODULE_CACHE`` and its unique forward ref never resolves, so the
    helper always reaches the full ``sys.modules`` walk.
    """
    alias = list[f"_SaltRaceMissing_{uuid.uuid4().hex}"]
    global_vars: dict[str, Any] = {"Alias": alias}
    helper(global_vars)


def _leaked_race_keys() -> list[str]:
    prefixes = ("_salt_race_probe_", "_salt_race_dummy_")
    return [key for key in list(sys.modules) if key.startswith(prefixes)]


def test_patch_installed():
    """Importing ``salt.main`` installs the race-safe wrapper over the helper."""
    helper = pa._enrich_globals_for_string_forward_refs  # noqa: SLF001
    assert getattr(helper, "_salt_race_safe", False) is True
    assert hasattr(helper, "__wrapped__")


def test_unpatched_enrichment_reproduces_race():
    """The ORIGINAL jsonargparse helper raises on a single mid-walk insertion."""
    original = pa._enrich_globals_for_string_forward_refs.__wrapped__  # noqa: SLF001
    with _armed_probe(1) as probe, pytest.raises(RuntimeError, match=_RACE_MESSAGE):
        _enrich_with_fresh_alias(original)
    assert probe.accesses == 1
    assert len(probe.inserted) == 1
    assert not _leaked_race_keys()


def test_patched_enrichment_survives_bounded_churn():
    """The patched helper retries through 3 racing walks and completes on the 4th."""
    with _armed_probe(3) as probe:
        _enrich_with_fresh_alias(pa._enrich_globals_for_string_forward_refs)  # noqa: SLF001
    assert probe.accesses == 4  # 3 raising walks + 1 clean walk
    assert len(probe.inserted) == 3
    assert not _leaked_race_keys()


def test_patched_enrichment_surfaces_persistent_race():
    """A race outlasting 40 retries + 1 final attempt surfaces instead of being swallowed."""
    attempts = _PATCH_RETRIES + 1
    with (
        _armed_probe(attempts) as probe,
        pytest.raises(RuntimeError, match=_RACE_MESSAGE),
    ):
        _enrich_with_fresh_alias(pa._enrich_globals_for_string_forward_refs)  # noqa: SLF001
    assert probe.accesses == attempts
    assert len(probe.inserted) == attempts
    assert not _leaked_race_keys()


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    """Tiny dummy H5 + parity norm dict + schema artifact for a fit smoke run."""
    base = tmp_path_factory.mktemp("salt_comet_race")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    h5_path = base / "pp_output_train.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"h5": h5_path, "nd": nd_path, "schema": schema_path}


def test_logged_fit_survives_sys_modules_churn(data, tmp_path, monkeypatch):
    """A logged (CometLogger ON, offline) ``salt fit`` completes when, right after the real
    ``comet_ml.start``, the wrapper arms a 5-mutation probe and calls the installed helper
    itself (synthetic stress; NOT a claim that jsonargparse's own validation walk hits it).
    """
    # register restoration for every Comet env var (setenv records the original
    # value even when unset; delenv of an absent var would record nothing)
    for name in _COMET_ENV:
        monkeypatch.setenv(name, "x")
    monkeypatch.delenv("COMET_API_KEY")  # -> salt forces online=false (offline, disabled config)
    monkeypatch.delenv("COMET_EXPERIMENT_NAME")
    monkeypatch.setenv("COMET_OFFLINE_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("COMET_DISABLE_AUTO_LOGGING", "1")

    real_start = comet_ml.start
    started: list[Any] = []
    stress: list[dict[str, int]] = []

    def _start_then_churn(*args: Any, **kwargs: Any) -> Any:
        experiment = real_start(*args, **kwargs)
        started.append(experiment)
        with _armed_probe(5) as probe:
            _enrich_with_fresh_alias(pa._enrich_globals_for_string_forward_refs)  # noqa: SLF001
            stress.append({"accesses": probe.accesses, "inserted": len(probe.inserted)})
        return experiment

    # Lightning's CometLogger calls `comet_ml.start` by attribute at call time;
    # CometLogger.__init__ itself stays untouched (jsonargparse introspects it)
    monkeypatch.setattr(comet_ml, "start", _start_then_churn)

    argv = [
        "fit",
        "--config",
        str(small_config()),
        f"--data.train_file={data['h5']}",
        f"--data.val_file={data['h5']}",
        f"--data.modules.reader.init_args.schema={data['schema']}",
        f"--model.modules.norm.init_args.norm_dict={data['nd']}",
        f"--trainer.default_root_dir={tmp_path}",
        "--trainer.accelerator=cpu",
        "--trainer.max_epochs=1",
        "--trainer.limit_train_batches=2",
        "--trainer.limit_val_batches=2",
        "--trainer.num_sanity_val_steps=0",
        "--trainer.log_every_n_steps=1",
        "--callbacks.progress=null",
    ]
    try:
        rc = salt.main.main(argv)
    finally:
        # _armed_probe already cleans up inside the wrapper; this sweep guards
        # against any leak if fit fails mid-stress
        for key in _leaked_race_keys():
            sys.modules.pop(key, None)

    assert rc == 0
    assert started, "comet_ml.start was never called — the CometLogger was not built"
    assert started[0] is not None
    assert stress, "the installed enrichment helper was not invoked at the Comet boundary"
    for record in stress:
        assert record["inserted"] == 5  # all 5 mutations happened
        assert record["accesses"] == 6  # 5 raising walks + 1 clean walk
