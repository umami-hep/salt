"""H5OutputSink dtype-policy coverage on the MAIN column path (input copies,
task columns, pad mask) — half_precision downcasts floats only, ints/bools
untouched (re-anchored from the retired ``test_half_precision_v1_flag`` e2e).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from salt.core.graph.bundle import Bundle
from salt.core.outputs.h5_sink import H5OutputSink
from salt.core.outputs.output_column import OutputColumn

pytestmark = pytest.mark.cpu_always

_N = 128  # rows (>= the ftag H5Writer default chunk size of 100)
_T = 5  # file token length


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


class _Dset:
    def __init__(self, reader: _Reader) -> None:
        self.reader = reader

    def __len__(self) -> int:
        return _N


class _DataModule:
    def __init__(self, reader: _Reader) -> None:
        self.test_dset = _Dset(reader)
        self.batch_size = _N
        self.test_suff = None


class _Module:
    name = "salt"


class _Trainer:
    """The minimal trainer surface ``open_schema``/``_output_path`` read."""

    def __init__(self, dm: _DataModule, ckpt_path: str) -> None:
        self.lightning_module = _Module()
        self.datamodule = dm
        self.ckpt_path = ckpt_path
        self.num_test_batches = None  # -> _expected_rows falls back to len(dset)


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


def _run_sink(tmp_path: Path, *, half_precision: bool) -> Path:
    """Drive open_schema -> consume -> flush over stub trainer/reader plumbing."""  # noqa: DOC201 - test helper, no Returns block per docstring policy
    src = tmp_path / "src.h5"
    _source_file(src)
    out = tmp_path / ("half.h5" if half_precision else "full.h5")
    sink = H5OutputSink(
        copy_inputs={"jets": ["pt", "n_trks"]},
        write_pad_mask=["tracks"],
        output=str(out),
        half_precision=half_precision,
    )
    _seed_columns(sink)
    trainer = _Trainer(_DataModule(_Reader(src)), ckpt_path=str(tmp_path / "e0-loss=0.1.ckpt"))
    sink.open_schema(trainer)
    torch.manual_seed(3)
    mask = torch.zeros(_N, _T, dtype=torch.bool)
    mask[:, 3:] = True  # last two positions padded
    bundle = Bundle({
        "meta": {"rows": torch.tensor([0, _N])},
        "outputs": {
            "jets": {"cls": torch.rand(_N, 2)},
            # the vertexing leaf reaches the sink as the int-cast union-find
            # value (int32; -inf padding -> int32 min)
            "tracks": {
                "vtx": torch.full((_N, _T), -(2**31), dtype=torch.int32).masked_fill(~mask, 4)
            },
        },
        "masks": {"tracks": mask},
    })
    sink.consume(bundle)
    sink.flush()
    assert out.exists()
    return out


def test_half_precision_downcasts_floats_only(tmp_path):
    """half_precision: float copies AND prob columns -> f2; int/bool untouched."""
    out = _run_sink(tmp_path, half_precision=True)
    with h5py.File(out) as f:
        jets_dtype, tracks_dtype = f["jets"].dtype, f["tracks"].dtype
    assert jets_dtype["pt"] == np.dtype("f2")  # float input copy downcast
    assert jets_dtype["salt_pb"] == np.dtype("f2")  # prob column downcast
    assert jets_dtype["n_trks"] == np.dtype("i8")  # int copy untouched
    assert tracks_dtype["VertexIndex"] == np.dtype("i8")  # int task column untouched
    assert tracks_dtype["mask"] == np.dtype("?")  # bool pad mask untouched


def test_full_precision_keeps_f4(tmp_path):
    """Default (full) precision: the v1 f4 policy on floats, ints/bools as-is."""
    out = _run_sink(tmp_path, half_precision=False)
    with h5py.File(out) as f:
        jets_dtype, tracks_dtype = f["jets"].dtype, f["tracks"].dtype
        tracks = f["tracks"][:]
    assert jets_dtype["pt"] == np.dtype("f4")
    assert jets_dtype["salt_pb"] == np.dtype("f4")
    assert jets_dtype["n_trks"] == np.dtype("i8")
    assert tracks_dtype["VertexIndex"] == np.dtype("i8")
    assert tracks_dtype["mask"] == np.dtype("?")
    # value spot-checks through the packing path: padded sentinel + mask polarity
    padded = tracks["mask"]
    assert (padded == np.tile([False] * 3 + [True] * 2, (_N, 1))).all()
    assert (tracks["VertexIndex"][padded] == np.int64(-(2**31))).all()
    assert (tracks["VertexIndex"][~padded] == 4).all()
