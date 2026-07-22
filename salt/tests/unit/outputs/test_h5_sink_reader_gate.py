"""H5OutputSink open_schema reader-shape gate — a structured-reader-less
(global-only, no ``.groups``/``.source_path``) reader is accepted when nothing
demands the source file, and rejected (naming the capability) when pad-mask
columns or input-copying are requested (plan 51 T0).
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


class _GlobalReader:
    """A minimal global-only reader: one 'mnist' stream, NO groups/source_path.

    Mirrors the plan-51 IdxReader — carries ``.filename`` (for _output_path)
    but exposes neither ``.groups`` nor ``.source_path``.
    """

    def __init__(self, filename: Path) -> None:
        self.filename = filename
        self.streams = ("mnist",)


class _Dset:
    def __init__(self, reader: _GlobalReader) -> None:
        self.reader = reader

    def __len__(self) -> int:
        return _N


class _DataModule:
    def __init__(self, reader: _GlobalReader) -> None:
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
    """Seed one global prob column on the 'mnist' stream (bypass section binding)."""
    sink._columns = (  # noqa: SLF001 - the explicit table is retired as a config surface
        OutputColumn(key="outputs.mnist.cls", suffixes=["p0", "p1"]),
    )
    sink._columns_resolved = True  # noqa: SLF001


def _trainer(tmp_path: Path) -> _Trainer:
    return _Trainer(
        _DataModule(_GlobalReader(tmp_path / "t10k-images-idx3-ubyte")),
        ckpt_path=str(tmp_path / "e0-loss=0.1.ckpt"),
    )


def test_groupless_reader_ok_when_nothing_needs_source(tmp_path):
    """No copy + no pad mask: a groups-less global-only reader resolves its schema."""
    out = tmp_path / "eval.h5"
    sink = H5OutputSink(output=str(out))  # no copy_inputs, write_pad_mask=False default
    _seed_global_column(sink)
    sink.open_schema(_trainer(tmp_path))
    assert sink._h5 is not None  # noqa: SLF001 - FIXED-mode writer was created
    assert out.exists()  # schema file written to disk
    with h5py.File(out) as f:
        assert f["mnist"].shape == (_N,)  # global stream -> (total,) shape
        cols = list(f["mnist"].dtype.names)
    assert cols == ["salt_p0", "salt_p1"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"copy_inputs": {"mnist": ["digit"]}}, "input-copying"),
        ({"write_pad_mask": ["mnist"]}, "pad-mask columns"),
    ],
)
def test_groupless_reader_rejected_when_source_needed(tmp_path, kwargs, match):
    """Binding input-copying OR pad-mask columns to a groups-less reader hard-fails,
    the ConfigError naming exactly the missing capability.
    """
    sink = H5OutputSink(output=str(tmp_path / "eval.h5"), **kwargs)
    _seed_global_column(sink)
    with pytest.raises(ConfigError, match=match):
        sink.open_schema(_trainer(tmp_path))


# Case 3 (H5StructuredReader-shaped reader path UNCHANGED) is already exercised
# end-to-end by test_h5_sink_precision.py (`_Reader` exposes .groups/.source_path
# and drives open_schema -> consume -> flush through the groups-present branch)
# and by test_sink_node.py; the goldens gate (generate_goldens.py) additionally
# proves the groups-present schema is byte-identical. Not duplicated here.
