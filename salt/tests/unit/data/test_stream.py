"""Unit tests for the shared `Reader`-base stream helpers (plan 24, Wave 2)."""

from __future__ import annotations

import numpy as np
import pytest

from salt.data import Cut, OffsetIndex, StreamConfig
from salt.data.readers.stream import INT_PAD_SENTINEL, _cut_sort_truncate_pad, pad_fill
from salt.graph.errors import ConfigError
from salt.schema import GroupSchema

ak = pytest.importorskip("awkward")


# --------------------------------------------------------------------------- #
# 1. StreamConfig validation
# --------------------------------------------------------------------------- #


def test_stream_config_pad_max_must_be_positive() -> None:
    with pytest.raises(ConfigError):
        StreamConfig(pad_max=0)


def test_stream_config_sort_normalises_default_mode() -> None:
    cfg = StreamConfig(pad_max=4, sort={"var": "pt"})
    assert cfg.sort == {"var": "pt", "mode": "descending"}
    assert cfg.engages_pipeline


def test_stream_config_sort_rejects_unknown_mode() -> None:
    with pytest.raises(ConfigError):
        StreamConfig(pad_max=4, sort={"var": "pt", "mode": "sideways"})


def test_stream_config_sort_rejects_empty_var() -> None:
    with pytest.raises(ConfigError):
        StreamConfig(pad_max=4, sort={"var": ""})


def test_stream_config_cuts_must_be_cut_instances() -> None:
    with pytest.raises(ConfigError):
        StreamConfig(pad_max=4, cuts=("not a cut",))  # type: ignore[arg-type]


def test_stream_config_cuts_sort_rejected_on_scalar_stream() -> None:
    with pytest.raises(ConfigError):
        StreamConfig(pad_max=4, jagged=False, cuts=(Cut(field="pt", op=">", value=0),))
    with pytest.raises(ConfigError):
        StreamConfig(pad_max=4, jagged=False, sort={"var": "pt"})


def test_stream_config_default_does_not_engage_pipeline() -> None:
    assert not StreamConfig(pad_max=8).engages_pipeline


# --------------------------------------------------------------------------- #
# pad_fill dtype-aware sentinels
# --------------------------------------------------------------------------- #


def test_pad_fill_by_dtype_kind() -> None:
    assert pad_fill(np.dtype("float32")) == pytest.approx(0.0)
    assert pad_fill(np.dtype("int32")) == INT_PAD_SENTINEL == -1
    assert pad_fill(np.dtype("uint8")) == 0
    assert pad_fill(np.dtype("bool")) is False


# --------------------------------------------------------------------------- #
# 2. _cut_sort_truncate_pad PARITY (no cuts/sort == byte-identical contiguous path)
# --------------------------------------------------------------------------- #


def _jagged_cols(counts: list[int], seed: int = 0) -> tuple[dict, list[str], GroupSchema]:
    """Build deterministic jagged float-pt + int-label awkward columns."""
    rng = np.random.default_rng(seed)
    pt = ak.Array([list(rng.normal(size=c).astype(np.float32)) for c in counts])
    label = ak.Array([list(rng.integers(0, 5, size=c).astype(np.int32)) for c in counts])
    cols = {"pt": pt, "label": label}
    gschema = GroupSchema(fields={"pt": "float32", "label": "int32", "valid": "bool"})
    return cols, ["pt", "label"], gschema


def _explicit_fill(dt: np.dtype) -> object:
    """The pre-Wave-2 inline pad fill (float 0.0 / int -1 / bool False / unsigned 0)."""
    if dt.kind == "f":
        return 0.0
    if dt.kind == "i":
        return -1
    if dt.kind == "b":
        return False
    return 0


def _manual_contiguous(
    cols: dict, fields: list[str], t: int, gschema: GroupSchema, b: int
) -> tuple[np.ndarray, np.ndarray]:
    """A hand-rolled contiguous truncate+pad+valid build (the pre-Wave-2 path)."""
    first = cols[fields[0]]
    counts = np.asarray(ak.num(first, axis=1)) if b > 0 else np.zeros(0, dtype=np.int64)
    valid = np.arange(t)[None, :] < np.minimum(counts, t)[:, None]
    dtype_fields = []
    blocks = {}
    for f in fields:
        arr = cols[f][:, :t]
        padded = ak.pad_none(arr, t, axis=1, clip=True)
        dt = np.dtype(gschema.fields[f])
        dense = ak.to_numpy(ak.fill_none(padded, _explicit_fill(dt), axis=1))
        block = np.asarray(dense).astype(dt, copy=False)
        blocks[f] = block
        dtype_fields.append((f, block.dtype))
    dtype_fields.append(("valid", np.dtype("bool")))
    raw = np.empty((b, t), dtype=np.dtype(dtype_fields))
    for f in fields:
        raw[f] = blocks[f]
    raw["valid"] = valid
    return raw, valid


def test_no_cuts_no_sort_is_byte_identical_to_contiguous_path() -> None:
    counts = [0, 1, 3, 8, 5, 12]
    cols, fields, gschema = _jagged_cols(counts, seed=42)
    b = len(counts)
    t = 8
    cfg = StreamConfig(pad_max=t)  # no cuts, no sort
    raw, valid = _cut_sort_truncate_pad(cols, fields, cfg, b, gschema)
    exp_raw, exp_valid = _manual_contiguous(cols, fields, t, gschema, b)

    # byte-for-byte identical (the locked parity invariant)
    assert raw.dtype == exp_raw.dtype
    assert raw.tobytes() == exp_raw.tobytes()
    np.testing.assert_array_equal(valid, exp_valid)
    # field order = config order + valid
    assert list(raw.dtype.names) == ["pt", "label", "valid"]
    # valid == per-row count clipped to T
    np.testing.assert_array_equal(valid.sum(axis=1), np.minimum(counts, t))


def test_no_cuts_no_sort_dtype_sentinels_on_padded_positions() -> None:
    counts = [2, 5, 0, 7]
    cols, fields, gschema = _jagged_cols(counts, seed=7)
    b, t = len(counts), 6
    cfg = StreamConfig(pad_max=t)
    raw, _ = _cut_sort_truncate_pad(cols, fields, cfg, b, gschema)
    for row, c in enumerate(counts):
        k = min(c, t)
        # padded FLOAT positions are 0.0; padded INTEGER LABEL positions are -1
        np.testing.assert_array_equal(raw["pt"][row, k:], 0.0)
        assert np.all(raw["label"][row, k:] == -1)


# --------------------------------------------------------------------------- #
# 3. drop-then-pad cuts (a cut constituent is REMOVED, never wastes a pad_max slot)
# --------------------------------------------------------------------------- #


def test_drop_then_pad_removes_failing_constituent() -> None:
    # one row: pt = [5, 1, 9, 2], cut pt > 3 keeps [5, 9]; pad_max=4 -> [5,9,pad,pad]
    cols = {
        "pt": ak.Array([[5.0, 1.0, 9.0, 2.0]]),
        "label": ak.Array([[10, 11, 12, 13]]),
    }
    gschema = GroupSchema(fields={"pt": "float32", "label": "int32", "valid": "bool"})
    cfg = StreamConfig(pad_max=4, cuts=(Cut(field="pt", op=">", value=3.0),))
    raw, valid = _cut_sort_truncate_pad(cols, ["pt", "label"], cfg, 1, gschema)
    # only the 2 passing constituents are kept, padded — NOT masked-in-place
    np.testing.assert_array_equal(raw["pt"][0], [5.0, 9.0, 0.0, 0.0])
    # labels carried in lockstep with the kept pts (10 with 5, 12 with 9)
    np.testing.assert_array_equal(raw["label"][0], [10, 12, -1, -1])
    # valid reflects the POST-CUT count (2), not the original (4)
    np.testing.assert_array_equal(valid[0], [True, True, False, False])


def test_drop_then_pad_does_not_waste_pad_max_slots() -> None:
    # pad_max=2, original 4 constituents, cut keeps 3 leading -> truncate to 2 KEPT
    cols = {"pt": ak.Array([[9.0, 1.0, 8.0, 7.0]])}
    gschema = GroupSchema(fields={"pt": "float32", "valid": "bool"})
    cfg = StreamConfig(pad_max=2, cuts=(Cut(field="pt", op=">", value=3.0),))
    raw, valid = _cut_sort_truncate_pad(cols, ["pt"], cfg, 1, gschema)
    # kept = [9, 8, 7]; truncate to leading 2 -> [9, 8]; NO pad slot is a dropped const
    np.testing.assert_array_equal(raw["pt"][0], [9.0, 8.0])
    np.testing.assert_array_equal(valid[0], [True, True])


# --------------------------------------------------------------------------- #
# 4. sort (argsort by sort.var, ALL fields permuted in lockstep)
# --------------------------------------------------------------------------- #


def test_sort_descending_permutes_all_fields_in_lockstep() -> None:
    cols = {
        "pt": ak.Array([[1.0, 5.0, 3.0]]),
        "label": ak.Array([[10, 50, 30]]),
    }
    gschema = GroupSchema(fields={"pt": "float32", "label": "int32", "valid": "bool"})
    cfg = StreamConfig(pad_max=3, sort={"var": "pt", "mode": "descending"})
    raw, _ = _cut_sort_truncate_pad(cols, ["pt", "label"], cfg, 1, gschema)
    np.testing.assert_array_equal(raw["pt"][0], [5.0, 3.0, 1.0])
    # labels follow the SAME permutation (50 with 5, 30 with 3, 10 with 1)
    np.testing.assert_array_equal(raw["label"][0], [50, 30, 10])


def test_sort_ascending() -> None:
    cols = {"pt": ak.Array([[3.0, 1.0, 2.0]])}
    gschema = GroupSchema(fields={"pt": "float32", "valid": "bool"})
    cfg = StreamConfig(pad_max=3, sort={"var": "pt", "mode": "ascending"})
    raw, _ = _cut_sort_truncate_pad(cols, ["pt"], cfg, 1, gschema)
    np.testing.assert_array_equal(raw["pt"][0], [1.0, 2.0, 3.0])


def test_cut_then_sort_compose() -> None:
    # cut pt > 1 keeps [5,3,2], then sort descending -> [5,3,2], pad_max=4
    cols = {"pt": ak.Array([[5.0, 1.0, 3.0, 2.0]])}
    gschema = GroupSchema(fields={"pt": "float32", "valid": "bool"})
    cfg = StreamConfig(
        pad_max=4, cuts=(Cut(field="pt", op=">", value=1.0),), sort={"var": "pt"}
    )
    raw, valid = _cut_sort_truncate_pad(cols, ["pt"], cfg, 1, gschema)
    np.testing.assert_array_equal(raw["pt"][0], [5.0, 3.0, 2.0, 0.0])
    np.testing.assert_array_equal(valid[0], [True, True, True, False])


def test_missing_cut_or_sort_field_raises() -> None:
    cols = {"pt": ak.Array([[1.0, 2.0]])}
    gschema = GroupSchema(fields={"pt": "float32", "valid": "bool"})
    with pytest.raises(KeyError):
        _cut_sort_truncate_pad(
            cols, ["pt"], StreamConfig(pad_max=2, cuts=(Cut(field="nope", op=">", value=0),)),
            1, gschema,
        )
    with pytest.raises(KeyError):
        _cut_sort_truncate_pad(
            cols, ["pt"], StreamConfig(pad_max=2, sort={"var": "nope"}), 1, gschema
        )


# --------------------------------------------------------------------------- #
# 5. OffsetIndex
# --------------------------------------------------------------------------- #


def test_offset_index_cumulative_and_total() -> None:
    idx = OffsetIndex([3, 0, 5, 2])
    assert idx.offsets == [0, 3, 3, 8, 10]
    assert idx.total == 10
    assert idx.file_starts() == [0, 3, 3, 8]


def test_offset_index_runs_single_file() -> None:
    idx = OffsetIndex([10, 10])
    # slice entirely within file 0
    assert idx.runs(slice(2, 7)) == [(0, 2, 7)]


def test_offset_index_runs_crossing_boundary() -> None:
    idx = OffsetIndex([10, 10])
    # slice [8, 13) -> file 0 local [8,10), file 1 local [0,3)
    assert idx.runs(slice(8, 13)) == [(0, 8, 10), (1, 0, 3)]


def test_offset_index_runs_skips_empty_files() -> None:
    idx = OffsetIndex([5, 0, 5])
    assert idx.runs(slice(3, 8)) == [(0, 3, 5), (2, 0, 3)]


def test_offset_index_covering_range() -> None:
    # per-event kept-jet counts: [2, 0, 3, 1] -> cum [0,2,2,5,6]
    cum = np.array([0, 2, 2, 5, 6])
    # kept-jet range [3, 5) -> covered by event index 2 only (cum[2]=2, cum[3]=5)
    c0, c1, off = OffsetIndex.covering_range(cum, 3, 5)
    assert (c0, c1) == (2, 3)
    assert off == 1  # jlo=3 is at offset 1 within event 2's [2,5) block
    # range [0, 6) covers all events 0..3 (event 1 empty)
    c0, c1, off = OffsetIndex.covering_range(cum, 0, 6)
    assert (c0, c1, off) == (0, 4, 0)
