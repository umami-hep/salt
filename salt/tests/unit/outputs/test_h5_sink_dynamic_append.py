"""H5OutputSink dynamic-append: the ftag `H5Writer` is built LAZILY on the first
`consume` from the actual batch dtypes/shapes and resized on every write — the
streaming (source-less iterable, uneven batches) and map-style paths, the
empty-run warning, and `close_if_open`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from salt.graph.bundle import Bundle
from salt.outputs import PadMaskWriter
from salt.outputs.output_schema import OutputColumn
from salt.outputs.sinks.h5_sink import H5OutputSink
from salt.outputs.sinks.sink import SinkContext

pytestmark = pytest.mark.cpu_always

# -- TEST 1: source-less iterable loader (streaming path) -------------------

_BATCH_SIZES = [7, 100, 3, 250, 1]  # crosses the 100-row chunk; includes size 1
_TOTAL = sum(_BATCH_SIZES)  # 361


class _UprootReader:
    """A source-less reader (mirrors `UprootReader`): one global 'jets' stream, no ``h5_source``."""

    def __init__(self, filename: Path) -> None:
        self.filename = filename
        self.streams = ("jets",)
        # h5_source intentionally absent


class _IterableDset:
    """Mirrors `IterableSaltDataset`: ``__len__`` is the per-shard BATCH count,
    ``n_rows()`` the split's row total — deliberately different numbers.
    """

    def __init__(self, reader: _UprootReader, n_batches: int, total_rows: int) -> None:
        self.reader = reader
        self._n_batches = n_batches
        self._total_rows = total_rows

    def __len__(self) -> int:
        return self._n_batches

    def n_rows(self) -> int:
        return self._total_rows


class _DataModule:
    def __init__(self, dset: _IterableDset) -> None:
        self.test_dset = dset
        self.batch_size = 100
        self.test_suff = None


def _seed_columns(sink: H5OutputSink) -> None:
    """Two bundle-sourced copy columns (pt, eta) + one float prob column (jets)."""
    sink._columns = (  # noqa: SLF001 - OutputColumn is the sink's internal value object; seeded directly for this white-box test
        OutputColumn(key="outputs.jets.pt", suffixes=["pt"], dtype=None, prefix=False),
        OutputColumn(key="outputs.jets.eta", suffixes=["eta"], dtype=None, prefix=False),
        OutputColumn(key="outputs.jets.cls", suffixes=["p0", "p1"]),
    )
    sink._columns_resolved = True  # noqa: SLF001


def _open(
    tmp_path: Path, name: str, *, n_rows: int, num_test_batches: list | None = None
) -> tuple[H5OutputSink, Path]:
    """Build + open a sink over a fresh `_IterableDset` sized `n_rows`."""  # noqa: DOC201 - test helper
    reader = _UprootReader(tmp_path / "ttbar_test.root")
    dset = _IterableDset(reader, n_batches=len(_BATCH_SIZES), total_rows=n_rows)
    out = tmp_path / name
    sink = H5OutputSink(output=str(out))
    _seed_columns(sink)
    ctx = SinkContext(
        run_name="salt",
        datamodule=_DataModule(dset),
        ckpt_path=str(tmp_path / "e0-loss=0.1.ckpt"),
        num_test_batches=num_test_batches,
    )
    sink.open_schema(ctx)
    return sink, out


def _feed(sink: H5OutputSink, sizes: list[int]) -> dict[str, list[np.ndarray]]:
    """Consume `sizes`-shaped batches in order; return the fed pt/eta/cls arrays."""  # noqa: DOC201 - test helper
    fed: dict[str, list[np.ndarray]] = {"pt": [], "eta": [], "cls": []}
    start = 0
    for b in sizes:
        pt = torch.arange(start, start + b, dtype=torch.float32)
        eta = -torch.arange(start, start + b, dtype=torch.float32)
        cls = torch.rand(b, 2, dtype=torch.float32)
        fed["pt"].append(pt.numpy().copy())
        fed["eta"].append(eta.numpy().copy())
        fed["cls"].append(cls.numpy().copy())
        # meta.rows is batch-local here; the sink no longer reads it at all.
        sink.consume(
            Bundle({
                "meta": {"rows": torch.tensor([0, b])},
                "outputs": {"jets": {"pt": pt, "eta": eta, "cls": cls}},
            })
        )
        start += b
    return fed


def test_dynamic_append_from_a_source_less_iterable_loader(tmp_path):
    """Load-bearing: varying-size batches carrying bundle-sourced copies append
    into one resizable file in arrival order.
    """
    sink, out = _open(tmp_path, "eval.h5", n_rows=_TOTAL)
    assert sink._h5 is None  # noqa: SLF001 - lazy: no writer until the first consume
    assert not out.exists()

    fed = _feed(sink, _BATCH_SIZES)
    sink.flush()

    with h5py.File(out) as f:
        jets = f["jets"]
        assert jets.shape == (_TOTAL,)
        assert jets.maxshape == (None,)
        assert list(jets.dtype.names) == ["pt", "eta", "salt_p0", "salt_p1"]
        for name in ("pt", "eta", "salt_p0", "salt_p1"):
            assert jets.dtype[name] == np.dtype("f4")
        data = jets[:]
    np.testing.assert_array_equal(data["pt"], np.concatenate(fed["pt"]))
    np.testing.assert_array_equal(data["eta"], np.concatenate(fed["eta"]))
    np.testing.assert_allclose(data["salt_p0"], np.concatenate([c[:, 0] for c in fed["cls"]]))
    np.testing.assert_allclose(data["salt_p1"], np.concatenate([c[:, 1] for c in fed["cls"]]))


def test_no_completeness_warning_when_the_known_total_matches(tmp_path, caplog):
    """No advisory warning when rows written == the dataset's known row total."""
    sink, _ = _open(tmp_path, "eval_ok.h5", n_rows=_TOTAL)
    _feed(sink, _BATCH_SIZES)
    with caplog.at_level(logging.WARNING, logger="salt"):
        caplog.clear()
        sink.flush()
    assert "the test split has" not in caplog.text


def test_completeness_warning_when_the_known_total_is_not_met(tmp_path, caplog):
    """flush warns when rows written != the dataset's known row total, uncapped."""
    sink, _ = _open(tmp_path, "eval_short.h5", n_rows=_TOTAL + 39)
    _feed(sink, _BATCH_SIZES)
    with caplog.at_level(logging.WARNING, logger="salt"):
        caplog.clear()
        sink.flush()
    assert "the test split has" in caplog.text


def test_no_completeness_warning_when_the_run_is_capped(tmp_path, caplog):
    """A `limit_test_batches`-capped run never warns, even with a mismatched total."""
    sink, _ = _open(tmp_path, "eval_capped.h5", n_rows=_TOTAL + 39, num_test_batches=[2])
    _feed(sink, _BATCH_SIZES)
    with caplog.at_level(logging.WARNING, logger="salt"):
        caplog.clear()
        sink.flush()
    assert "the test split has" not in caplog.text


def test_completeness_warning_still_fires_when_the_batch_limit_covers_the_split(tmp_path, caplog):
    """The advisory must not be silenced by a finite batch count that covers the
    whole split — only a limit BELOW the split total is a real cap. Lightning
    reports ``[len(dataloader)]`` even when `limit_test_batches` is unset (here
    5 batches x the datamodule's batch_size 100 = 500 rows >= 400), so the loop
    was NOT capped and the mismatch must still warn.
    """
    sink, _ = _open(
        tmp_path, "eval_len_dl.h5", n_rows=_TOTAL + 39, num_test_batches=[len(_BATCH_SIZES)]
    )
    _feed(sink, _BATCH_SIZES)
    with caplog.at_level(logging.WARNING, logger="salt"):
        caplog.clear()
        sink.flush()
    assert "the test split has" in caplog.text


# -- TEST 2: map-style H5 reader, copies + pad mask -------------------------

_MAP_BATCH_SIZES = [50, 60, 18]  # deliberately uneven; sums to 128
_MAP_TOTAL = sum(_MAP_BATCH_SIZES)
_T = 5  # file token length


@dataclass
class _Group:
    dataset: str
    global_object: bool


class _MapReader:
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


class _MapDset:
    def __init__(self, reader: _MapReader, n: int) -> None:
        self.reader = reader
        self._n = n

    def __len__(self) -> int:
        return self._n


class _MapDataModule:
    def __init__(self, dset: _MapDset) -> None:
        self.test_dset = dset
        self.batch_size = _MAP_TOTAL
        self.test_suff = None


def _map_source_file(path: Path) -> None:
    """A source H5 that only advertises the sequence-stream token length (`_T`) —
    no field values are read back from it (copies arrive from the bundle).
    """
    jets = np.zeros(_MAP_TOTAL, dtype=np.dtype([("dummy", "f4")]))
    tracks = np.zeros((_MAP_TOTAL, _T), dtype=np.dtype([("dummy", "f4")]))
    with h5py.File(path, "w") as f:
        f.create_dataset("jets", data=jets)
        f.create_dataset("tracks", data=tracks)


def test_dynamic_append_map_style_with_copies_and_pad_mask(tmp_path):
    """An int64 copy, a per-token float copy and a pad-mask column, over three uneven batches."""
    src = tmp_path / "src.h5"
    _map_source_file(src)
    out = tmp_path / "eval.h5"
    sink = H5OutputSink(output=str(out))
    # bind BEFORE seeding: binding resets the resolved column table
    sink.bind_output_section({"pad_mask": PadMaskWriter(streams=["tracks"])})
    sink._columns = (  # noqa: SLF001 - OutputColumn is the sink's internal value object; seeded directly for this white-box test
        OutputColumn(key="outputs.jets.n_trks", suffixes=["n_trks"], dtype=None, prefix=False),
        OutputColumn(key="outputs.tracks.d0", suffixes=["d0"], dtype=None, prefix=False),
        OutputColumn(key="outputs.jets.cls", suffixes=["pb", "pc"]),
    )
    sink._columns_resolved = True  # noqa: SLF001
    ctx = SinkContext(
        run_name="salt",
        datamodule=_MapDataModule(_MapDset(_MapReader(src), _MAP_TOTAL)),
        ckpt_path=str(tmp_path / "e0-loss=0.1.ckpt"),
    )
    sink.open_schema(ctx)
    assert sink._h5 is None  # noqa: SLF001 - lazy: no writer until the first consume

    torch.manual_seed(9)
    fed_n_trks, fed_d0 = [], []
    for b in _MAP_BATCH_SIZES:
        n_trks = torch.randint(0, 20, (b,), dtype=torch.int64)
        d0 = torch.rand(b, _T, dtype=torch.float32)
        cls = torch.rand(b, 2, dtype=torch.float32)
        mask = torch.zeros(b, _T, dtype=torch.bool)
        mask[:, -1] = True  # last token padded
        fed_n_trks.append(n_trks.numpy().copy())
        fed_d0.append(d0.numpy().copy())
        # the pad mask is read off the PadMaskWriter's graph-fed
        # outputs.<stream>.mask leaf.
        sink.consume(
            Bundle({
                "meta": {"rows": torch.tensor([0, b])},
                "outputs": {
                    "jets": {"n_trks": n_trks, "cls": cls},
                    "tracks": {"d0": d0, "mask": mask},
                },
            })
        )
    sink.flush()

    with h5py.File(out) as f:
        jets, tracks = f["jets"], f["tracks"]
        assert jets.dtype["n_trks"] == np.dtype("i8")
        assert tracks.shape == (_MAP_TOTAL, _T)
        names = list(tracks.dtype.names)
        assert names[-1] == "mask"
        jets_data, tracks_data = jets[:], tracks[:]
    np.testing.assert_array_equal(jets_data["n_trks"], np.concatenate(fed_n_trks))
    np.testing.assert_allclose(tracks_data["d0"], np.concatenate(fed_d0))


# -- TEST 3: empty run ------------------------------------------------------


def test_empty_run_writes_no_file_and_warns_once(tmp_path, caplog):
    """`open_schema` + `flush` with no batch: no file, exactly one WARNING naming the gap."""
    sink, out = _open(tmp_path, "empty.h5", n_rows=0)
    with caplog.at_level(logging.WARNING, logger="salt"):
        caplog.clear()
        sink.flush()
    assert not out.exists()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "no test batch reached the sink" in warnings[0].getMessage()


# -- TEST 4: close_if_open --------------------------------------------------


class _FakeHandle:
    """A stand-in exposing only `close()` — the surface `close_if_open` needs."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_close_if_open_is_idempotent_after_partial_rows(tmp_path):
    """A leaked handle mid-run closes once, and a second call is a no-op."""
    sink = H5OutputSink(output=str(tmp_path / "unused.h5"))
    fake = _FakeHandle()
    sink._h5 = fake  # noqa: SLF001 - white-box cleanup contract
    sink._rows_written = 3  # noqa: SLF001

    sink.close_if_open()
    assert fake.closed is True
    assert sink._h5 is None  # noqa: SLF001 - handle released
    sink.close_if_open()
    assert sink._h5 is None  # noqa: SLF001
