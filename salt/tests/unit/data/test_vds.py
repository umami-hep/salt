"""Tests for the `VDS` setup module + its datamodule auto-injection (W3.B)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from salt.data import (
    VDS,
    Features,
    GraphDataModule,
    H5StructuredReader,
    InputSamples,
    Labels,
)
from salt.data.base import SaltDatasetModule, SetupBundle
from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.planner import _check_incompatibilities, compile_setup_plan
from salt.graph.setup_executor import run_setup_plan
from salt.graph.setup_spec import SetupIO, SetupStage, SourceSpec, flatten_source_spec
from salt.graph.spec import IO, Mode
from salt.schema import dump_schema, save_schema
from salt.testing.inputs import write_dummy_file, write_dummy_norm_dict

JET_VARS = ["pt_btagJes", "eta_btagJes"]
TRACK_VARS = ["d0", "z0SinTheta", "dphi", "deta"]
JET_LABELS = ["flavour_label"]
TRACK_LABELS = ["ftagTruthOriginLabel", "ftagTruthVertexIndex", "ftagTruthTypeLabel"]

FIT_SINKS = [
    "inputs.jets",
    "inputs.tracks",
    "masks.tracks",
    *[f"labels.jets.{label}" for label in JET_LABELS],
    *[f"labels.tracks.{label}" for label in TRACK_LABELS],
]
TEST_SINKS = ["inputs.jets", "inputs.tracks", "masks.tracks", "meta.rows"]
SINKS = {Mode.FIT: FIT_SINKS, Mode.VAL: FIT_SINKS, Mode.TEST: TEST_SINKS}
N_JETS = 1000


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("vds")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_dummy_norm_dict(nd_path, cd_path)
    h5_path = base / "pp_output_train.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


def build_reader(data) -> H5StructuredReader:
    return H5StructuredReader(groups={"jets": {}, "tracks": {}}, schema=data["schema"])


def _flat(nested):
    return flatten_source_spec(nested)


def _wired_vds(*, reader="reader", vds_capable=True, out=None) -> VDS:
    """A `VDS` with the assembly pokes applied (the datamodule wiring)."""
    vds = VDS(out=out)
    vds.name = "vds"
    vds._reader = reader
    vds._vds_capable = vds_capable
    return vds


# a stub ShmStage-named setup module for the incompatibility check (W3.S
# doesn't exist yet — the rule matches on class NAME, not type).


class ShmStage(SaltDatasetModule):
    """Name-only stub: the incompatibility check matches `type(other).__name__`."""

    def declare_io(self, mode: Mode) -> IO:
        del mode
        return IO()

    def declare_setup_io(self, stage: SetupStage) -> SetupIO:
        # produce an unrelated setup leaf so it is a legal setup-only module.
        del stage
        return SetupIO(produces={"artifacts": {"shm": {"root": SourceSpec(kind="scalar")}}})

    def setup(self, ctx: SetupBundle, stage: SetupStage) -> SetupBundle:
        del stage
        return ctx


# VDS unit behaviour: declare + setup (build / identity / ROOT identity)


class TestVdsUnit:
    def test_declare_requires_pattern_produces_vds_path(self):
        vds = _wired_vds()
        io = vds.declare_setup_io("train")
        req = _flat(io.requires)
        prod = _flat(io.produces)
        assert "source.reader.train.pattern" in req
        assert req["source.reader.train.pattern"].kind == "path"
        assert "source.reader.train.vds_path" in prod
        assert prod["source.reader.train.vds_path"].kind == "path"

    def test_reader_name_required(self):
        vds = VDS()
        vds.name = "vds"
        with pytest.raises(RuntimeError, match="no reader name wired"):
            vds.declare_setup_io("train")

    def test_declare_io_per_batch_empty(self):
        vds = _wired_vds()
        io = vds.declare_io(Mode.FIT)
        assert not io.requires
        assert not io.produces

    def test_non_wildcard_identity(self):
        """A non-wildcard pattern → identity edge (vds_path == pattern verbatim)."""
        vds = _wired_vds(vds_capable=True)
        ctx = Bundle()
        ctx.merge(
            {"source": {"reader": {"train": {"pattern": "/data/train.h5"}}}},
            who="input_samples",
            expected={"source.reader.train.pattern"},
        )
        vds.setup(ctx, "train")
        assert ctx.get("source.reader.train.vds_path") == "/data/train.h5"

    def test_wildcard_builds_vds(self, data, monkeypatch):
        """A wildcard pattern of a vds_capable reader → create_vds is called."""
        calls: list[tuple] = []
        sentinel = data["dir"] / "built_vds.h5"

        def fake_create_vds(pattern, out_fname=None):
            calls.append((Path(pattern), out_fname))
            return sentinel

        monkeypatch.setattr("salt.data.readers.vds_module.create_vds", fake_create_vds)
        vds = _wired_vds(vds_capable=True)
        ctx = Bundle()
        pattern = str(data["dir"] / "pp_output_*.h5")
        ctx.merge(
            {"source": {"reader": {"train": {"pattern": pattern}}}},
            who="input_samples",
            expected={"source.reader.train.pattern"},
        )
        vds.setup(ctx, "train")
        assert len(calls) == 1  # create_vds WAS called for the wildcard
        assert calls[0][0] == Path(pattern)
        assert ctx.get("source.reader.train.vds_path") == str(sentinel)

    def test_wildcard_uses_explicit_out_path(self, data, monkeypatch):
        """An explicit out[stage] overrides the default VDS path."""
        calls: list[tuple] = []
        out_path = data["dir"] / "explicit_vds.h5"

        def fake_create_vds(pattern, out_fname=None):
            calls.append((Path(pattern), out_fname))
            return Path(out_fname)

        monkeypatch.setattr("salt.data.readers.vds_module.create_vds", fake_create_vds)
        vds = _wired_vds(vds_capable=True, out={"train": out_path})
        ctx = Bundle()
        pattern = str(data["dir"] / "pp_output_*.h5")
        ctx.merge(
            {"source": {"reader": {"train": {"pattern": pattern}}}},
            who="input_samples",
            expected={"source.reader.train.pattern"},
        )
        vds.setup(ctx, "train")
        assert calls[0][1] == out_path  # explicit out passed to create_vds
        assert ctx.get("source.reader.train.vds_path") == str(out_path)

    def test_real_wildcard_build_writes_h5_vds(self, data, tmp_path):
        """End-to-end (NO spy): a real wildcard build produces a usable VDS file."""
        # the fixture file is pp_output_train.h5; a glob over it matches one member.
        vds = _wired_vds(vds_capable=True)
        ctx = Bundle()
        pattern = str(data["dir"] / "pp_output_*.h5")
        ctx.merge(
            {"source": {"reader": {"train": {"pattern": pattern}}}},
            who="input_samples",
            expected={"source.reader.train.pattern"},
        )
        vds.setup(ctx, "train")
        built = ctx.get("source.reader.train.vds_path")
        assert built != pattern  # a real VDS file path, not the glob
        assert Path(built).is_file()
        import h5py  # noqa: PLC0415 - test-only

        with h5py.File(built, "r") as f:
            assert "jets" in f  # the VDS exposes the member groups

    def test_root_glob_identity_edge_never_calls_create_vds(self, monkeypatch):
        """R-VDSROOT (HARD): vds_capable=False + a GLOB value → identity, NO create_vds."""
        called = {"n": 0}

        def spy_create_vds(pattern, out_fname=None):
            called["n"] += 1
            raise AssertionError("create_vds must NOT run for a non-vds_capable reader")

        monkeypatch.setattr("salt.data.readers.vds_module.create_vds", spy_create_vds)
        vds = _wired_vds(vds_capable=False)  # a ROOT reader (easyjet/ftag1lite)
        ctx = Bundle()
        glob_value = "/data/ntuples/*.root"
        ctx.merge(
            {"source": {"reader": {"train": {"pattern": glob_value}}}},
            who="input_samples",
            expected={"source.reader.train.pattern"},
        )
        vds.setup(ctx, "train")
        assert called["n"] == 0  # create_vds NEVER called on a ROOT glob
        # the glob passes through verbatim as vds_path (the ROOT reader globs natively)
        assert ctx.get("source.reader.train.vds_path") == glob_value


# topo order: VDS requires `pattern` → orders AFTER InputSamples


class TestSetupPlanOrdering:
    def test_vds_orders_after_input_samples(self):
        inp = InputSamples(files={"train": "/a.h5"})
        inp.name = "input_samples"
        inp._reader = "reader"
        vds = _wired_vds()
        modules = {"input_samples": inp, "vds": vds}
        plan = compile_setup_plan(modules, "train")
        names = plan.module_names
        assert names.index("input_samples") < names.index("vds")

    def test_setup_pass_chains_pattern_to_vds_path(self, monkeypatch):
        """End-to-end: InputSamples writes pattern, VDS reads it and writes vds_path."""
        inp = InputSamples(files={"train": "/data/train.h5"})
        inp.name = "input_samples"
        inp._reader = "reader"
        vds = _wired_vds(vds_capable=True)
        modules = {"input_samples": inp, "vds": vds}
        ctx = Bundle()
        run_setup_plan(compile_setup_plan(modules, "train"), "train", ctx)
        assert ctx.get("source.reader.train.pattern") == "/data/train.h5"
        # non-wildcard → identity
        assert ctx.get("source.reader.train.vds_path") == "/data/train.h5"


# incompatibility enforcement (plan-25 Rev-2)


class TestIncompatibility:
    def test_vds_plus_shmstage_raises(self):
        vds = _wired_vds()
        shm = ShmStage()
        shm.name = "shm"
        modules = {"vds": vds, "shm": shm}
        with pytest.raises(ConfigError, match="incompatible"):
            _check_incompatibilities(modules)

    def test_vds_alone_passes(self):
        vds = _wired_vds()
        _check_incompatibilities({"vds": vds})  # no raise

    def test_compile_setup_plan_rejects_vds_plus_shmstage(self):
        inp = InputSamples(files={"train": "/a.h5"})
        inp.name = "input_samples"
        inp._reader = "reader"
        vds = _wired_vds()
        shm = ShmStage()
        shm.name = "shm"
        modules = {"input_samples": inp, "vds": vds, "shm": shm}
        with pytest.raises(ConfigError, match="incompatible"):
            compile_setup_plan(modules, "train")

    def test_check_is_generic_symmetric_stub(self):
        """The check is generic: any module naming another's class raises."""

        class Alpha(SaltDatasetModule):
            incompatible_with = ("Beta",)

            def declare_io(self, mode):
                del mode
                return IO()

        class Beta(SaltDatasetModule):
            def declare_io(self, mode):
                del mode
                return IO()

        a, b = Alpha(), Beta()
        a.name, b.name = "a", "b"
        with pytest.raises(ConfigError, match="incompatible"):
            _check_incompatibilities({"a": a, "b": b})


# datamodule auto-injection + byte-identity served bytes


def build_modules(data) -> dict:
    return {
        "input_samples": InputSamples(
            files={"train": data["h5"], "val": data["h5"], "test": data["h5"]}
        ),
        "reader": build_reader(data),
        "features": Features(variables={"jets": list(JET_VARS), "tracks": list(TRACK_VARS)}),
        "labels": Labels(),
    }


class TestDatamoduleAutoInjection:
    def test_vds_auto_injected_and_wired(self, data):
        dm = GraphDataModule(modules=build_modules(data), sinks=SINKS, pin_memory=False)
        assert "vds" in dm._setup_modules
        injected = dm._setup_modules["vds"]
        assert isinstance(injected, VDS)
        # wired with the single reader's name + its vds_capable flag (True for H5)
        assert injected._reader == "reader"
        assert injected._vds_capable is True
        # setup-only: never in the per-batch namespace
        assert "vds" not in dm._batch_modules

    def test_vds_path_in_resolved_ctx(self, data):
        dm = GraphDataModule(modules=build_modules(data), sinks=SINKS, pin_memory=False)
        dm.setup("fit")
        assert dm._setup_ctx is not None
        # both pattern (InputSamples) AND vds_path (VDS) present; non-wildcard identity
        assert dm._setup_ctx.get("source.reader.train.pattern") == str(data["h5"])
        assert dm._setup_ctx.get("source.reader.train.vds_path") == str(data["h5"])

    def test_explicit_vds_overrides_auto_injection(self, data):
        modules = build_modules(data)
        modules["vds"] = VDS()
        dm = GraphDataModule(modules=modules, sinks=SINKS, pin_memory=False)
        # the explicit VDS is used and wired (not a second auto-injected one)
        assert dm._setup_modules["vds"] is modules["vds"]
        assert dm._vds is modules["vds"]
        assert dm._vds._reader == "reader"
        assert dm._vds._vds_capable is True

    def test_two_vds_rejected(self, data):
        modules = build_modules(data)
        modules["vds"] = VDS()
        modules["vds2"] = VDS()
        with pytest.raises(ConfigError, match="at most one VDS"):
            GraphDataModule(modules=modules, sinks=SINKS)

    def test_auto_injected_vds_carries_legacy_out_paths(self, data):
        explicit_out = data["dir"] / "my_train_vds.h5"
        dm = GraphDataModule(
            modules=build_modules(data),
            train_vds_path=explicit_out,
            sinks=SINKS,
            pin_memory=False,
        )
        assert dm._vds is not None
        assert dm._vds._out["train"] == explicit_out

    def test_no_vds_when_no_input_samples(self, data):
        """No InputSamples (and no aliases) → no pattern → no VDS auto-injected."""
        modules = {
            "reader": H5StructuredReader(
                groups={"jets": {}, "tracks": {}}, schema=data["schema"], filename=data["h5"]
            ),
            "features": Features(
                variables={"jets": list(JET_VARS), "tracks": list(TRACK_VARS)}
            ),
            "labels": Labels(),
        }
        dm = GraphDataModule(modules=modules, sinks=SINKS, pin_memory=False)
        assert "vds" not in dm._setup_modules
        assert dm._vds is None

    def test_first_batch_byte_identical_to_trunk(self, data):
        """Served bytes via the VDS-resolved path == via the W3.A alias trunk path."""
        dm_vds = GraphDataModule(
            modules=build_modules(data), batch_size=128, sinks=SINKS, pin_memory=False
        )
        dm_vds.setup("fit")
        # the trunk W3.A path: deprecated train_file/val_file aliases, reader bound
        # directly (no explicit InputSamples module in the dict).
        dm_trunk = GraphDataModule(
            modules={
                "reader": H5StructuredReader(
                    groups={"jets": {}, "tracks": {}}, schema=data["schema"], filename=data["h5"]
                ),
                "features": Features(
                    variables={"jets": list(JET_VARS), "tracks": list(TRACK_VARS)}
                ),
                "labels": Labels(),
            },
            train_file=data["h5"],
            val_file=data["h5"],
            batch_size=128,
            sinks=SINKS,
            pin_memory=False,
        )
        dm_trunk.setup("fit")
        rows = np.s_[0:500]
        a = dm_vds.train_dset[rows]
        b = dm_trunk.train_dset[rows]
        assert torch.equal(a["inputs"]["jets"], b["inputs"]["jets"])
        assert torch.equal(a["inputs"]["tracks"], b["inputs"]["tracks"])
        assert torch.equal(a["masks"]["tracks"], b["masks"]["tracks"])
        for label in JET_LABELS:
            assert torch.equal(a["labels"]["jets"][label], b["labels"]["jets"][label])
        for label in TRACK_LABELS:
            assert torch.equal(a["labels"]["tracks"][label], b["labels"]["tracks"][label])
