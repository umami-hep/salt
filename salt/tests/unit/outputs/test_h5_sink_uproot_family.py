"""H5OutputSink capability gate for the uproot reader family.

The sink keys its structured-H5 path on the reader's advertised CAPABILITY
(``reader.h5_source``), NOT on reader type or group shape. The uproot/ROOT reader
(`UprootReader`) exposes a non-None
``.groups`` of a different (non-H5) config shape and a ROOT source, so it must
NOT be driven into the H5StructuredReader branch (which assumes ``.dataset`` /
``.global_object`` / an h5py-openable ``source_path``). It advertises no
``h5_source`` and so takes the no-source path: task outputs are written; a
pad-mask / input-copy demand raises a ConfigError naming the capability. A
MultiSampleReader-shaped reader whose ``.groups`` DELEGATE to a sub-reader (and
so DO expose ``.dataset``) must still route to the no-source path — proving the
probe is capability-keyed, not ``hasattr(group, "dataset")``-keyed.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from salt.graph.errors import ConfigError
from salt.outputs.h5_sink import H5OutputSink
from salt.outputs.output_schema import OutputColumn

pytestmark = pytest.mark.cpu_always

_N = 128  # rows (>= the ftag H5Writer default chunk size of 100)


class _UprootGroupCfg:
    """Minimal `UprootGroupConfig` stand-in: ``jagged``/``pad_max`` only, and
    deliberately NO ``.dataset`` / ``.global_object`` — the shape that crashes
    the H5StructuredReader branch (`AttributeError: no attribute 'dataset'`).
    """

    def __init__(self, jagged: bool) -> None:
        self.jagged = jagged
        self.pad_max = 15 if jagged else None


class _H5LikeGroupCfg:
    """H5StructuredReader `GroupConfig` shape: HAS ``.dataset``/``.global_object``."""

    def __init__(self, dataset: str, global_object: bool) -> None:
        self.dataset = dataset
        self.global_object = global_object


class _UprootReader:
    """A uproot-family reader as the sink sees it: non-None ``.groups`` of
    `UprootGroupConfig` shape + a ROOT ``.filename``, advertising NO
    ``h5_source`` (the base-`Reader` default). Mirrors `UprootReader`.
    """

    def __init__(self, filename: Path) -> None:
        self.filename = filename
        self.streams = ("jets", "event")
        self.groups = {"jets": _UprootGroupCfg(True), "event": _UprootGroupCfg(False)}
        # h5_source intentionally absent -> getattr(reader, "h5_source", None) is None


class _MultiSampleLikeReader:
    """MultiSampleReader-shaped: ``.groups`` DELEGATE to a sub-reader and so DO
    expose ``.dataset``/``.global_object`` — but the wrapper has N sources, no
    single ``filename``/``source_path``, and advertises no ``h5_source``. Keying
    on ``hasattr(group, "dataset")`` would wrongly take the structured path (and
    then crash on the missing ``source_path``); keying on ``h5_source`` routes it
    to the no-source path. `_output_path` must fall back to ``sources()[0]``.
    """

    def __init__(self, srcs: list[Path]) -> None:
        self._sources = list(srcs)
        self.streams = ("jets", "event")
        self.groups = {
            "jets": _H5LikeGroupCfg("jets", False),
            "event": _H5LikeGroupCfg("event", True),
        }

    def sources(self) -> list[Path]:
        return list(self._sources)


class _Dset:
    def __init__(self, reader: object) -> None:
        self.reader = reader

    def __len__(self) -> int:
        return _N


class _DataModule:
    def __init__(self, reader: object) -> None:
        self.test_dset = _Dset(reader)
        self.batch_size = _N
        self.test_suff = None


class _Module:
    name = "salt"


class _Trainer:
    def __init__(self, dm: _DataModule, ckpt_path: str) -> None:
        self.lightning_module = _Module()
        self.datamodule = dm
        self.ckpt_path = ckpt_path
        self.num_test_batches = None  # -> _expected_rows falls back to len(dset)


def _seed_global_column(sink: H5OutputSink) -> None:
    """Seed one global prob column on the 'event' stream (bypass section binding)."""
    sink._columns = (  # noqa: SLF001 - the explicit table is retired as a config surface
        OutputColumn(key="outputs.event.cls", suffixes=["p0", "p1"]),
    )
    sink._columns_resolved = True  # noqa: SLF001


def _trainer(dm: _DataModule, tmp_path: Path) -> _Trainer:
    return _Trainer(dm, ckpt_path=str(tmp_path / "e0-loss=0.1.ckpt"))


def test_uproot_reader_ok_writes_global_task_output(tmp_path):
    """A uproot-family reader (groups present, h5_source None) resolves its
    schema and writes a global 'event' task-output group — no crash on the
    UprootGroupConfig's missing ``.dataset``.
    """
    out = tmp_path / "eval.h5"
    reader = _UprootReader(tmp_path / "ttbar_test.root")
    sink = H5OutputSink(output=str(out))  # no copy_inputs, write_pad_mask=False default
    _seed_global_column(sink)
    sink.open_schema(_trainer(_DataModule(reader), tmp_path))
    assert sink._h5 is not None  # noqa: SLF001 - FIXED-mode writer was created
    assert out.exists()
    with h5py.File(out) as f:
        assert f["event"].shape == (_N,)  # global stream -> (total,) shape
        assert list(f["event"].dtype.names) == ["salt_p0", "salt_p1"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"copy_inputs": {"jets": ["pt"]}}, "input-copying"),
        ({"write_pad_mask": ["jets"]}, "pad-mask columns"),
    ],
)
def test_uproot_reader_rejected_when_source_needed(tmp_path, kwargs, match):
    """Binding input-copying OR pad-mask columns to a source-less uproot reader
    hard-fails, the ConfigError naming exactly the missing capability.
    """
    reader = _UprootReader(tmp_path / "ttbar_test.root")
    sink = H5OutputSink(output=str(tmp_path / "eval.h5"), **kwargs)
    _seed_global_column(sink)
    with pytest.raises(ConfigError, match=match):
        sink.open_schema(_trainer(_DataModule(reader), tmp_path))


def test_multisample_delegating_groups_still_no_source_path(tmp_path):
    """A MultiSampleReader-shaped reader whose groups DELEGATE (and so expose
    ``.dataset``) is still routed to the no-source path via h5_source, and
    ``_output_path`` falls back to ``sources()[0]`` (no ``filename``/``source_path``).
    """
    out = tmp_path / "eval.h5"
    reader = _MultiSampleLikeReader([tmp_path / "ttbar_test.root", tmp_path / "hh4b_test.root"])
    sink = H5OutputSink(output=str(out))
    _seed_global_column(sink)
    sink.open_schema(_trainer(_DataModule(reader), tmp_path))
    assert sink._h5 is not None  # noqa: SLF001
    assert out.exists()
    with h5py.File(out) as f:
        assert f["event"].shape == (_N,)
