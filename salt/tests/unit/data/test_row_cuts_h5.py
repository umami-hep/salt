"""Unit tests for the shared row-cut engine (`Reader._apply_row_cuts`) and
`H5StructuredReader` sample-axis row cuts (kept-index at prepare + filtered read;
cuts=None identity).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from salt.data import GlobalObjectCuts, H5StructuredReader
from salt.data.base import WorkerCtx
from salt.graph.spec import Mode
from salt.testing.inputs import write_dummy_file, write_dummy_norm_dict

READ_FIELDS = {"jets": {"pt": "t", "flavour_label": "t"}, "tracks": {"d0": "t"}}


@pytest.fixture(scope="module")
def dummy(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("row_cuts")
    nd, cd = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_dummy_norm_dict(nd, cd)
    path = base / "data.h5"
    write_dummy_file(path, nd)
    return path


def _reader(
    path: Path, cuts: GlobalObjectCuts | None = None, stage: str | None = None, num: int = -1
):
    r = H5StructuredReader(
        groups={"jets": {"global_object": True}, "tracks": {"global_object": False}},
        filename=path,
        cuts=cuts,
        stage=stage,
        num=num,
    )
    r.prepare()
    r.bind(WorkerCtx(mode=Mode.FIT, read_fields=READ_FIELDS, seed=0))
    return r


def _jets(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        return f["jets"]["pt"][:], f["jets"]["flavour_label"][:]


# --------------------------------------------------------------------------- #
# 1. base engine — Reader._apply_row_cuts / _row_record (no file)
# --------------------------------------------------------------------------- #


def test_engine_no_cuts_keeps_all() -> None:
    r = H5StructuredReader(groups={"jets": {"global_object": True}})
    rec = r._row_record({"pt": np.array([0.1, 0.6, 0.9])}, 3)
    np.testing.assert_array_equal(r._apply_row_cuts(rec), [True, True, True])


def test_engine_applies_global_cut() -> None:
    r = H5StructuredReader(
        groups={"jets": {"global_object": True}},
        cuts=GlobalObjectCuts(global_cuts=("pt > 0.5",)),
    )
    rec = r._row_record({"pt": np.array([0.1, 0.6, 0.9])}, 3)
    np.testing.assert_array_equal(r._apply_row_cuts(rec), [False, True, True])


# --------------------------------------------------------------------------- #
# 2. cuts=None identity — byte-identical to a contiguous read
# --------------------------------------------------------------------------- #


def test_identity_len_is_full_file(dummy) -> None:
    assert len(_reader(dummy)) == 1000
    assert _reader(dummy)._kept is None  # identity sentinel preserves the slab path


@pytest.mark.parametrize("sl", [slice(0, 128), slice(300, 428), slice(900, 1000)])
def test_identity_read_matches_contiguous_oracle(dummy, sl) -> None:
    pt, _ = _jets(dummy)
    out = _reader(dummy).read(sl, Mode.FIT)
    np.testing.assert_array_equal(out["raw.jets"]["pt"], pt[sl])
    # tracks pad mask is the contiguous slab's ~valid
    with h5py.File(dummy, "r") as f:
        valid = f["tracks"]["valid"][sl]
    np.testing.assert_array_equal(out["masks.tracks"], ~valid)


# --------------------------------------------------------------------------- #
# 3. row-cut correctness — kept-index, __len__, slice reads vs numpy/h5py oracle
# --------------------------------------------------------------------------- #


def _oracle_kept(path: Path, thr: float) -> np.ndarray:
    pt, _labels = _jets(path)
    return np.flatnonzero(pt > thr)


def test_cut_len_is_filtered(dummy) -> None:
    spec = GlobalObjectCuts(global_cuts=("pt > 0.5",))
    r = _reader(dummy, cuts=spec)
    kept = _oracle_kept(dummy, 0.5)
    assert len(r) == len(kept)
    np.testing.assert_array_equal(r._kept, kept)  # ascending kept file rows


def test_cut_full_read_matches_oracle(dummy) -> None:
    spec = GlobalObjectCuts(global_cuts=("pt > 0.5",))
    r = _reader(dummy, cuts=spec)
    kept = _oracle_kept(dummy, 0.5)
    pt, _ = _jets(dummy)
    out = r.read(slice(0, len(r)), Mode.FIT)
    np.testing.assert_array_equal(out["raw.jets"]["pt"], pt[kept])
    assert np.all(out["raw.jets"]["pt"] > 0.5)


@pytest.mark.parametrize("frac", [(0, 1), (0.5, 0.5), (-1, -1)])  # first / middle / last
def test_cut_endpoint_reads_map_through_kept(dummy, frac) -> None:
    spec = GlobalObjectCuts(global_cuts=("pt > 0.5",))
    r = _reader(dummy, cuts=spec)
    kept = _oracle_kept(dummy, 0.5)
    pt, _ = _jets(dummy)
    n = len(r)
    a, b = frac
    lo = n - 1 if a == -1 else int(a * n)
    hi = lo + 1 if b in {1, -1} else int(b * n) + 1
    lo, hi = max(0, lo), min(n, max(lo + 1, hi))
    out = r.read(slice(lo, hi), Mode.FIT)
    np.testing.assert_array_equal(out["raw.jets"]["pt"], pt[kept[lo:hi]])


def test_cut_non_contiguous_and_batch_boundary(dummy) -> None:
    spec = GlobalObjectCuts(global_cuts=("pt > 0.5",))
    r = _reader(dummy, cuts=spec)
    kept = _oracle_kept(dummy, 0.5)
    # non-contiguous: kept file rows are scattered (not a contiguous run)
    assert np.any(np.diff(kept) > 1)
    pt, _ = _jets(dummy)
    n = len(r)
    mid = n // 2
    first = r.read(slice(0, mid), Mode.FIT)["raw.jets"]["pt"]
    second = r.read(slice(mid, n), Mode.FIT)["raw.jets"]["pt"]
    # batch-boundary: two halves concatenate to the whole, in kept order
    np.testing.assert_array_equal(np.concatenate([first, second]), pt[kept])
    # tracks pad mask follows the same kept rows
    with h5py.File(dummy, "r") as f:
        valid = f["tracks"]["valid"][:]
    out = r.read(slice(0, n), Mode.FIT)
    np.testing.assert_array_equal(out["masks.tracks"], ~valid[kept])


def test_cut_is_stage_independent(dummy) -> None:
    """The engine no longer carries per-split cuts — every stage reads identically."""
    spec = GlobalObjectCuts(global_cuts=("pt > 0.5",))
    kept = _oracle_kept(dummy, 0.5)
    for stage in ("train", "val", "test"):
        assert len(_reader(dummy, cuts=spec, stage=stage)) == len(kept)


def test_cut_missing_field_raises(dummy) -> None:
    from salt.graph.errors import SchemaError

    r = H5StructuredReader(
        groups={"jets": {"global_object": True}, "tracks": {"global_object": False}},
        filename=dummy,
        cuts=GlobalObjectCuts(global_cuts=("not_a_field > 0",)),
    )
    with pytest.raises(SchemaError, match="not_a_field"):
        r.prepare()


def test_cut_num_caps_served_rows(dummy) -> None:
    spec = GlobalObjectCuts(global_cuts=("pt > 0.5",))
    kept = _oracle_kept(dummy, 0.5)
    r = _reader(dummy, cuts=spec, num=10)
    assert len(r) == 10
    pt, _ = _jets(dummy)
    out = r.read(slice(0, 10), Mode.FIT)
    np.testing.assert_array_equal(out["raw.jets"]["pt"], pt[kept[:10]])
