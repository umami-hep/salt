"""The generated `SinkAdapter`: lifecycle forwarding, registry wiring, and the
teardown-on-crash guarantee a sink used to get from being a Callback itself.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader

from salt.callbacks.sink_adapter import SinkAdapter, attach_runtime_sink
from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import IO, Mode
from salt.outputs import H5OutputSink, Node, OnnxExportSink, RuntimeSink
from salt.outputs.output_schema import OutputColumn
from salt.outputs.sinks.registry import iter_sinks, register_sink, sink_registry
from salt.outputs.sinks.sink import SinkContext

pytestmark = pytest.mark.cpu_always

_N = 128  # rows (>= the ftag H5Writer default chunk size of 100)
_T = 5  # file token length


# -- stand-ins ---------------------------------------------------------------


class _SpySink(RuntimeSink):
    """Records which lifecycle calls it received, in order."""

    name = "spy"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.batches = 0

    def declare_io(self, mode: Mode) -> IO:
        del mode
        return IO(requires={}, produces={})

    def open_schema(self, ctx: SinkContext) -> None:
        self.calls.append("open")
        self.ctx = ctx

    def consume(self, bundle: Bundle) -> None:
        del bundle
        self.calls.append("consume")
        self.batches += 1

    def flush(self) -> None:
        self.calls.append("flush")

    def close_if_open(self) -> None:
        self.calls.append("close")


@dataclass
class _Group:
    dataset: str
    global_object: bool


class _Reader:
    def __init__(self, source_path: Path) -> None:
        self.source_path = str(source_path)
        self.streams = ("jets", "tracks")
        self.groups = {
            "jets": _Group("jets", global_object=True),
            "tracks": _Group("tracks", global_object=False),
        }

    @property
    def h5_source(self) -> str:
        """Advertise the h5py-openable source (mirrors H5StructuredReader)."""
        return self.source_path


class _Dset:
    def __init__(self, reader: _Reader) -> None:
        self.reader = reader

    def __len__(self) -> int:
        return _N


class _BatchIndices(torch.utils.data.Dataset):
    """Three trivial batches; the payload the module returns is built in test_step."""

    def __init__(self, n: int = 3) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> int:
        return idx


class _DataModule(LightningDataModule):
    """Carries the schema handles `SinkContext.from_trainer` reads, plus a loader."""

    def __init__(self, reader: _Reader) -> None:
        super().__init__()
        self.test_dset = _Dset(reader)
        self.batch_size = _N
        self.test_suff = None

    def test_dataloader(self) -> DataLoader:
        """Three single-item batches, num_workers=0 (agent memcg)."""  # noqa: DOC201 - test helper
        return DataLoader(_BatchIndices(), batch_size=1, num_workers=0)


class _BundleModule(LightningModule):
    """Emits one full bundle on batch 0, then raises on `fail_at` if asked."""

    def __init__(self, bundle: Bundle, fail_at: int | None = None) -> None:
        super().__init__()
        self.name = "salt"
        self._bundle = bundle
        self._fail_at = fail_at
        self._emitted = False

    def forward(self, batch: object) -> object:
        """Identity; the module exists only to drive the test loop."""  # noqa: DOC201 - test helper
        return batch

    def test_step(self, batch: object, batch_idx: int) -> Bundle | None:
        """Emit the bundle once, then optionally crash mid-test."""  # noqa: DOC201, DOC501 - test helper
        del batch
        if self._fail_at is not None and batch_idx == self._fail_at:
            raise RuntimeError("boom mid-test")
        if self._emitted:
            return None
        self._emitted = True
        return self._bundle


def _source_file(path: Path) -> None:
    """A tiny source H5: float + int jets columns, per-token tracks group."""
    rng = np.random.default_rng(11)
    jets = np.zeros(_N, dtype=np.dtype([("pt", "f4"), ("n_trks", "i8")]))
    jets["pt"] = rng.random(_N).astype("f4") * 100
    jets["n_trks"] = rng.integers(0, _T, _N)
    tracks = np.zeros((_N, _T), dtype=np.dtype([("d0", "f4")]))
    tracks["d0"] = rng.random((_N, _T)).astype("f4")
    with h5py.File(path, "w") as f:
        f.create_dataset("jets", data=jets)
        f.create_dataset("tracks", data=tracks)


def _seed_columns(sink: H5OutputSink) -> None:
    """One float prob column (jets) + one bare int column (tracks)."""
    sink._columns = (  # noqa: SLF001 - the explicit table is retired as a config surface
        OutputColumn(key="outputs.jets.cls", suffixes=["pb", "pc"]),
        OutputColumn(key="outputs.tracks.vtx", suffixes=["VertexIndex"], dtype="i8", prefix=False),
    )
    sink._columns_resolved = True  # noqa: SLF001


def _bundle() -> Bundle:
    """One full-file batch of outputs the seeded columns demand."""  # noqa: DOC201 - test helper
    mask = torch.zeros(_N, _T, dtype=torch.bool)
    mask[:, 3:] = True
    torch.manual_seed(3)
    return Bundle({
        "meta": {"rows": torch.tensor([0, _N])},
        "outputs": {
            "jets": {"cls": torch.rand(_N, 2)},
            "tracks": {
                "vtx": torch.full((_N, _T), -(2**31), dtype=torch.int32).masked_fill(~mask, 4)
            },
        },
        "masks": {"tracks": mask},
    })


def _trainer(*callbacks: Callback) -> Trainer:
    """A silent CPU trainer carrying `callbacks`."""  # noqa: DOC201 - test helper
    return Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=list(callbacks),
    )


# -- registry ----------------------------------------------------------------


def test_registry_is_created_once_and_deduplicates():
    """`register_sink` attaches one list and never registers the same object twice."""
    owner = SimpleNamespace()
    sink = _SpySink()
    register_sink(owner, sink)
    register_sink(owner, sink)
    assert sink_registry(owner) is owner.salt_sinks
    assert sink_registry(owner) == [sink]


def test_iter_sinks_also_finds_a_sink_left_in_the_callbacks_list():
    """The alias path: a sink handed straight to a trainer is still discovered."""
    aliased = _SpySink()
    registered = _SpySink()
    owner = SimpleNamespace(callbacks=[Callback(), aliased])
    register_sink(owner, registered)
    found = iter_sinks(owner)
    assert found == [registered, aliased]


def test_iter_sinks_is_empty_without_a_trainer():
    """No trainer, no sinks — the run-free and stub paths must not explode."""
    assert iter_sinks(None) == []


def test_attach_generates_an_adapter_for_a_runtime_sink_but_not_a_node():
    """A `RuntimeSink` gets a Lightning bridge; a declare-only `Node` gets none."""
    trainer = _trainer()
    runtime, manifest = _SpySink(), OnnxExportSink()
    adapter = attach_runtime_sink(trainer, runtime)
    assert isinstance(adapter, SinkAdapter)
    assert attach_runtime_sink(trainer, manifest) is None
    assert iter_sinks(trainer) == [runtime, manifest]
    assert [cb for cb in trainer.callbacks if isinstance(cb, SinkAdapter)] == [adapter]


def test_adapters_have_distinct_state_keys():
    """Two adapters must not be conflated by Lightning's state bookkeeping."""
    first, second = _SpySink(), _SpySink()
    second.name = "other"
    assert SinkAdapter(first).state_key != SinkAdapter(second).state_key


# -- the base split ----------------------------------------------------------


def test_sink_module_imports_no_lightning():
    """`salt/outputs/sinks/sink.py` must stay driver-agnostic — no lightning import."""
    text = Path(inspect.getsourcefile(RuntimeSink)).read_text(encoding="utf-8")
    imports = [
        line
        for line in text.splitlines()
        if line.startswith(("import ", "from ")) and "lightning" in line
    ]
    assert imports == []


def test_the_shipped_sinks_sit_on_the_right_base():
    """Runtime sinks carry the lifecycle; the ONNX manifest is declare-only."""
    assert issubclass(H5OutputSink, RuntimeSink)
    assert issubclass(OnnxExportSink, Node)
    assert not issubclass(OnnxExportSink, RuntimeSink)
    assert not hasattr(OnnxExportSink, "consume")


def test_a_sink_is_not_a_lightning_callback():
    """The point of the split: the node stopped privileging one driver."""
    assert not issubclass(Node, Callback)
    assert not isinstance(H5OutputSink(), Callback)


# -- lifecycle forwarding ----------------------------------------------------


def test_adapter_drives_the_full_lifecycle_under_lightning(tmp_path):
    """A clean test run: open -> consume per batch -> flush -> close."""
    del tmp_path
    sink = _SpySink()
    model = _BundleModule(_bundle())
    dm = _DataModule(_Reader(Path("unused.h5")))
    trainer = _trainer()
    attach_runtime_sink(trainer, sink)
    trainer.test(model, datamodule=dm)
    assert sink.calls[0] == "open"
    assert sink.calls.count("consume") == 3
    assert sink.calls[-2:] == ["flush", "close"]
    assert sink.ctx.run_name == "salt"
    assert sink.ctx.datamodule is dm


def test_adapter_refuses_multi_device_test():
    """The single-device guard moved from the sink onto the adapter."""
    adapter = SinkAdapter(_SpySink())
    with pytest.raises(ConfigError, match="single device"):
        adapter.setup(SimpleNamespace(world_size=2), None, "test")


def test_adapter_setup_allows_multi_device_outside_test():
    """The guard is TEST-scoped; a multi-device fit is none of the sink's business."""
    SinkAdapter(_SpySink()).setup(SimpleNamespace(world_size=2), None, "fit")


# -- the guarantee that must not regress -------------------------------------


def test_a_crash_mid_test_still_closes_the_sink():
    """A raise inside the test loop must still close the sink.

    Lightning does NOT run `teardown` on that path — it unwinds past
    `on_test_end`/`teardown` and calls `on_exception` only. This is the test
    that pins which hook actually delivers the guarantee.
    """
    sink = _SpySink()
    model = _BundleModule(_bundle(), fail_at=1)
    dm = _DataModule(_Reader(Path("unused.h5")))
    trainer = _trainer()
    attach_runtime_sink(trainer, sink)
    with pytest.raises(RuntimeError, match="boom mid-test"):
        trainer.test(model, datamodule=dm)
    assert "flush" not in sink.calls  # the run never reached on_test_end
    assert sink.calls[-1] == "close"


def _opened_h5_sink(tmp_path: Path) -> H5OutputSink:
    """A real H5 sink with a live handle and one batch already written."""  # noqa: DOC201 - test helper
    src = tmp_path / "src.h5"
    _source_file(src)
    sink = H5OutputSink(
        copy_inputs={"jets": ["pt", "n_trks"]},
        write_pad_mask=["tracks"],
        output=str(tmp_path / "crashed.h5"),
    )
    _seed_columns(sink)
    sink.open_schema(
        SinkContext(
            run_name="salt",
            datamodule=_DataModule(_Reader(src)),
            ckpt_path=str(tmp_path / "e0-loss=0.1.ckpt"),
        )
    )
    sink.consume(_bundle())
    assert sink._h5 is not None  # noqa: SLF001 - precondition: the handle IS open
    return sink


@pytest.mark.parametrize("hook", ["on_exception", "teardown"])
def test_either_exit_hook_closes_the_real_h5(tmp_path, hook):
    """Both exit paths release the real H5 handle and leave a readable file."""
    sink = _opened_h5_sink(tmp_path)
    adapter = SinkAdapter(sink)
    if hook == "on_exception":
        adapter.on_exception(None, None, RuntimeError("boom mid-test"))
    else:
        adapter.teardown(None, None, "test")

    assert sink._h5 is None  # noqa: SLF001 - the handle-release assertion IS the gate
    out = Path(str(tmp_path / "crashed.h5"))
    assert out.exists()
    with h5py.File(out) as f:  # a leaked writer handle would make this raise
        assert len(f["jets"]) == _N


def test_close_if_open_is_idempotent(tmp_path):
    """teardown can fire more than once (crash then normal exit) — no double close."""
    src = tmp_path / "src.h5"
    _source_file(src)
    out = tmp_path / "idem.h5"
    sink = H5OutputSink(output=str(out))
    _seed_columns(sink)
    ctx = SinkContext(
        run_name="salt",
        datamodule=_DataModule(_Reader(src)),
        ckpt_path=str(tmp_path / "e0-loss=0.1.ckpt"),
    )
    sink.open_schema(ctx)
    adapter = SinkAdapter(sink)
    adapter.teardown(None, None, "test")
    adapter.teardown(None, None, "test")
    assert sink._h5 is None  # noqa: SLF001 - the handle-release assertion IS the gate
