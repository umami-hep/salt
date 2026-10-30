"""Behavioural tests for `salt.data.readers.ftag1lite_reader`: `FTAG1LITEReader` (``jets:
small_r | large_r | both``) over a real (uproot-written) two-collection FTAG1LITE-like corpus,
and `FTAG1LITEStreamingDataset` with its shard scopes and row schedules over an in-memory stub.
"""

# Streaming guarantee -> retained test (real multi-process DDP: test_ftag1lite_reader_ddp.py):
# - coverage + disjointness: test_coverage_and_disjointness (SCOPES x real 236 containers at
#   S=4/16/236/1024 + 3000 random groups at S=64: every row in exactly one shard);
#   test_owner_coverage_and_disjointness (R x W in 1x1/1x4/3x4/8x2/3x1 x G=3/30/236 with a 45%
#   group: tiling, exact worker strides, adjacent abutting straddlers, balance, W=1 == reference
#   rank cut).
# - owner super-batch stratification: test_window_schedule_exact (1000 random + 9 labelled
#   adversarial cells: quotas, per-window and Pfair prefix deviation < 1, counts_at / windows vs
#   the matrix); test_owner_super_batch_composition (R=3 x W=2: each rank batch == its window,
#   super batch within d_cw of ideal, tiny containers cumulative < 1, no row twice);
#   test_pf_tie_order_is_a_strict_total_order (antisymmetric strict order, lower id last tie).
# - per_rank pool composition: test_pool_schedule_multiset_exact_over_an_epoch (6 cells x pools
#   7/128/1000); test_pool_composition_is_exact_at_pool_granularity (pooled == unpooled counts).
# - pool_rows normalisation: test_pool_rows_floors_to_whole_batches (8 cases, floor to batches);
#   test_pool_below_one_batch_is_rejected (3 cases, ConfigError).
# - epoch lifecycle (both scopes): test_second_iter_reshuffles_with_the_same_ownership,
#   test_constructing_at_epoch_e_reproduces_the_eth_pass, test_setting_epoch_resets_the_pass_counter.
# - residency guard: test_guard_raises_below_group_count (scope x cap 1/2/3 x len / iter seam).
# - FIT len == batches emitted: test_fit_emits_only_full_batches_and_exactly_len (6 x 5 pools).
# - eval is per_rank: test_eval_modes_are_per_rank_even_when_the_reader_says_owner (VAL/TEST x pool).
# - datamodule hook: test_datamodule_builds_the_ftag1lite_streaming_dataset (type + full val split).
# - owner contract: test_owner_requires_null_pool, test_owner_rejects_bad_layout. Containers =
#   parent dirs: test_read_equals_concatenated_read_blocks, test_row_blocks_tile_and_match_manifest.

from __future__ import annotations

import copy
import json
import pickle
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from salt.data.base import RowBlock, WorkerCtx
from salt.data.datamodule import SaltDataModule
from salt.data.manifest import CorpusManifest, apply_schema
from salt.data.processors.features import Features
from salt.data.readers.cuts import GlobalObjectCuts
from salt.data.readers.ftag1lite_reader import (
    FTAG1LITEReader,
    FTAG1LITEStreamingDataset,
    PoolSchedule,
    WindowSchedule,
    _partition,
    _pf_successor_cmp,
    _pf_tie_order,
    shard_assignment,
)
from salt.data.readers.uproot_reader import UprootReader
from salt.graph.errors import ConfigError, SchemaError
from salt.graph.spec import Mode, flatten_spec
from salt.tests._fixtures.ftag1lite_ddp_stream import (
    SEED,
    SIZES,
    InterleavedStubReader,
    _adversarial_window_schedule_fixtures,
    _assignment,
    _batches,
    _dataset,
    _expected_uids,
    _group_counts,
    _owner_group_rows,
    _random_group_rows,
    _random_window_schedule_fixtures,
    _row_counts,
    _scoped,
    _shard_dataset,
    _uids,
)
from salt.tests._fixtures.joint_ftag1lite_corpus import (
    LARGE_CUTS,
    LARGE_GROUPS,
    LARGE_TYPE,
    SMALL_CUTS,
    SMALL_GROUPS,
    SMALL_TYPE,
    TOWER_PAD,
    TRACK_PAD,
    TREE,
    JointCorpusFacts,
    SeamUprootReader,
    build_joint_corpus,
    corpus_glob,
    make_plain_reader,
    make_reader,
    make_side_reader,
)

_ONE_JET = {"groups": {"jets": {"branches": {"pt": "pt"}, "jagged": False}}}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> JointCorpusFacts:  # noqa: ANN001
    """One corpus for the module (uproot/awkward are installed wherever this suite gates)."""
    pytest.importorskip("uproot")
    pytest.importorskip("awkward")
    return build_joint_corpus(tmp_path_factory.mktemp("ftag1lite_corpus"))


def _full(reader, mode: Mode = Mode.FIT) -> dict[str, np.ndarray]:  # noqa: ANN001
    """Every served row in one `read`."""
    return reader.read(slice(0, len(reader)), mode)


def _assert_same(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    """Same keys, dtypes, shapes and values (field by field, NaN == NaN)."""
    assert set(a) == set(b)
    for key in a:
        x, y = a[key], b[key]
        assert x.dtype == y.dtype, key
        assert x.shape == y.shape, key
        if x.dtype.names:
            for name in x.dtype.names:
                np.testing.assert_array_equal(x[name], y[name], err_msg=f"{key}.{name}")
        else:
            np.testing.assert_array_equal(x, y, err_msg=key)


def _slice(batch: dict[str, np.ndarray], sel) -> dict[str, np.ndarray]:  # noqa: ANN001
    """Every key of a batch indexed by `sel` (TEST meta excluded)."""
    return {k: v[sel] for k, v in batch.items() if k != "meta.rows"}


def _concat(batches: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Per-key concatenation of batches."""
    return {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}


def _identities(batch: dict[str, np.ndarray]) -> list[tuple[int, int, float]]:
    """``(eventNumber, jet_type, pt)`` per row."""
    jets, event = batch["raw.jets"], batch["raw.event"]
    return [
        (int(e), int(t), float(p))
        for e, t, p in zip(event["eventNumber"], jets["jet_type"], jets["pt"], strict=True)
    ]


def _assert_blocks_tile(reader, corpus: JointCorpusFacts) -> list[int]:  # noqa: ANN001
    """One block per file, contiguous over ``[0, len)`` in global rows, group = its file's
    container id; the manifest agrees (group names, per-container rows, paths).
    """
    from salt.data.manifest import build_manifest

    names = list(dict.fromkeys(p.parent.name for p in corpus.files))
    assert reader.group_names == names
    blocks = reader.row_blocks()
    assert len(blocks) <= len(corpus.files)
    group_rows = [0] * len(names)
    cursor = 0
    for b, path in zip(blocks, corpus.files, strict=False):
        assert b.group == names.index(path.parent.name)
        assert b.start == cursor
        assert b.stop > b.start
        group_rows[b.group] += b.n_rows
        cursor = b.stop
    assert cursor == len(reader)
    manifest = build_manifest(reader)
    assert manifest.group_names == reader.group_names == names
    assert manifest.n_rows == len(reader)
    assert manifest.group_rows == group_rows
    paths = [e.path for e in manifest.entries]
    assert paths == [str(p) for p in corpus.files][: len(paths)]
    return group_rows


def test_first_file_block_is_mixed_and_counts_add_up(corpus) -> None:  # noqa: ANN001
    reader = make_reader(corpus.root)
    out = reader.read_block(reader.row_blocks()[0], Mode.FIT)
    assert set(np.unique(out["raw.jets"]["jet_type"]).tolist()) == {SMALL_TYPE, LARGE_TYPE}
    for i, entry in enumerate(reader._table):  # noqa: SLF001
        expected = corpus.rows_in_file(i)
        n_small = sum(r.jet_type == SMALL_TYPE for r in expected)
        assert int(entry.cum["small"][-1]) == n_small
        assert int(entry.cum["large"][-1]) == len(expected) - n_small
        assert entry.n_rows == len(expected)
    assert len(reader) == len(corpus.rows)


def test_row_order_is_per_event_small_then_large(corpus) -> None:  # noqa: ANN001
    reader = make_reader(corpus.root)
    out = _full(reader)
    assert _identities(out) == [r.identity for r in corpus.rows]
    jt = out["raw.jets"]["jet_type"]
    evn = out["raw.event"]["eventNumber"]
    for k in range(1, len(jt)):
        if evn[k] == evn[k - 1]:
            assert jt[k] >= jt[k - 1], "a small-R row follows a large-R row within one event"


def test_rows_align_with_single_type_readers(corpus) -> None:  # noqa: ANN001
    joint = _full(make_reader(corpus.root))
    jt = joint["raw.jets"]["jet_type"]
    for side, jet_type in (("small", SMALL_TYPE), ("large", LARGE_TYPE)):
        single = _full(make_side_reader(corpus.root, side))
        rows = jt == jet_type
        assert int(rows.sum()) == len(single["raw.jets"])
        for key, ref in single.items():
            got = joint[key][rows]
            if ref.dtype.names:
                for name in ref.dtype.names:
                    assert got.dtype[name] == ref.dtype[name], f"{side} {key}.{name}"
                    np.testing.assert_array_equal(got[name], ref[name], err_msg=f"{key}.{name}")
            else:
                np.testing.assert_array_equal(got, ref, err_msg=f"{side} {key}")


def test_towers_are_all_padding_on_large_rows(corpus) -> None:  # noqa: ANN001
    out = _full(make_reader(corpus.root))
    large = out["raw.jets"]["jet_type"] == LARGE_TYPE
    assert large.any()
    assert out["raw.towers"].shape[1] == TOWER_PAD
    assert out["masks.towers"][large].all()
    assert not out["raw.towers"]["valid"][large].any()
    for name in ("pt", "deta"):
        assert (out["raw.towers"][name][large] == 0).all()
    small = ~large
    assert (~out["masks.towers"][small]).any(), "small-R rows must carry real towers"
    assert out["raw.tracks"].shape[1] == TRACK_PAD
    assert (~out["masks.tracks"][large]).any(), "large-R rows must carry real tracks"


def test_fill_values_on_the_other_types_rows(corpus) -> None:  # noqa: ANN001
    out = _full(make_reader(corpus.root))
    jets = out["raw.jets"]
    assert jets["jet_type"].dtype == np.dtype("int64")
    large = jets["jet_type"] == LARGE_TYPE
    small = ~large
    assert np.isnan(jets["ptFromTruthDressedWZJet"][large]).all()
    for name in ("matchedToTruthDressedWZJet", "HadronGhostTruthLabelID", "PartonTruthLabelID"):
        assert (jets[name][large] == -1).all(), name
    assert np.isnan(jets["m"][small]).all()
    for name in ("R10TruthLabel_R22v1", "GhostBHadronsFinalCount", "GhostCHadronsFinalCount"):
        assert (jets[name][small] == -1).all(), name
    assert (jets["R10TruthLabel_R22v1"][large] != -1).all()
    assert (jets["matchedToTruthDressedWZJet"][small] >= 0).all()


def test_read_block_subranges_equal_full_file_slices(corpus) -> None:  # noqa: ANN001
    reader = make_reader(corpus.root)
    block = reader.row_blocks()[0]
    whole = reader.read_block(block, Mode.FIT)
    n = block.n_rows
    ranges = {(lo, min(lo + k, n)) for lo in range(0, n, 3) for k in (1, 2, 5, 11)}
    # a range crossing an event that has large-R rows only
    expected = corpus.rows_in_file(0)
    small_events = {r.entry for r in expected if r.jet_type == SMALL_TYPE}
    lone = [i for i, r in enumerate(expected) if r.entry not in small_events]
    assert lone, "fixture must contain a large-only event in file 0"
    ranges.add((max(0, lone[0] - 1), min(n, lone[0] + 2)))
    for lo, hi in sorted(ranges):
        if hi <= lo:
            continue
        sub = reader.read_block(block.subrange(block.start + lo, block.start + hi), Mode.FIT)
        _assert_same(sub, _slice(whole, slice(lo, hi)))


def test_read_equals_concatenated_read_blocks(corpus) -> None:  # noqa: ANN001
    reader = make_reader(corpus.root)
    full = _full(reader)
    blocks = reader.row_blocks()
    names = reader.group_names
    assert [b.group for b in blocks] == [names.index(p.parent.name) for p in corpus.files]
    assert {b.group for b in blocks} == {0, 1}
    _assert_same(full, _concat([reader.read_block(b, Mode.FIT) for b in blocks]))
    for cut in (blocks[1].start, blocks[2].start):  # straddles file 0 -> 1, then 1 -> 2
        lo, hi = cut - 3, cut + 4
        _assert_same(reader.read(slice(lo, hi), Mode.FIT), _slice(full, slice(lo, hi)))


def test_row_blocks_tile_and_match_manifest(corpus) -> None:  # noqa: ANN001
    group_rows = _assert_blocks_tile(make_reader(corpus.root), corpus)
    names = list(dict.fromkeys(p.parent.name for p in corpus.files))
    expected = [0] * len(names)
    for i, path in enumerate(corpus.files):
        expected[names.index(path.parent.name)] += len(corpus.rows_in_file(i))
    assert group_rows == expected


def test_num_caps_joint_rows(corpus) -> None:  # noqa: ANN001
    full = _full(make_reader(corpus.root))
    rows = corpus.rows
    n_file0 = len(corpus.rows_in_file(0))
    # a cap INSIDE an event of file 1
    inside = next(
        k
        for k in range(n_file0 + 1, len(rows))
        if rows[k].event_number == rows[k - 1].event_number and rows[k].file == 1
    )
    for num in (inside, n_file0 + 2):
        reader = make_reader(corpus.root, num=num)
        assert len(reader) == num
        _assert_blocks_tile(reader, corpus)
        _assert_same(_full(reader), _slice(full, slice(0, num)))
        _assert_same(
            _concat([reader.read_block(b, Mode.FIT) for b in reader.row_blocks()]),
            _slice(full, slice(0, num)),
        )
    assert len(make_reader(corpus.root, num=10 * len(rows))) == len(rows)


def test_test_mode_meta_rows_are_global(corpus) -> None:  # noqa: ANN001
    reader = make_reader(corpus.root)
    block = reader.row_blocks()[2]
    out = reader.read_block(block, Mode.TEST)
    start = reader.row_blocks()[2].start
    assert start == len(corpus.rows_in_file(0)) + len(corpus.rows_in_file(1))
    np.testing.assert_array_equal(out["meta.rows"], [start, start + block.n_rows])


def test_with_source_stage_splits_are_disjoint_and_exhaustive(corpus) -> None:  # noqa: ANN001
    proto = make_reader(corpus.root, filename=None)
    proto.name = "reader"
    seen: Counter = Counter()
    residues = {"train": set(range(8)), "val": {8}, "test": {9}}
    for stage in ("train", "val", "test"):
        reader = proto.with_source(corpus.glob, stage=stage)
        assert reader.name == "reader"
        out = _full(reader)
        res = set((out["raw.event"]["eventNumber"] % 10).tolist())
        assert res <= residues[stage]
        for jet_type in (SMALL_TYPE, LARGE_TYPE):
            assert (out["raw.jets"]["jet_type"] == jet_type).any(), (stage, jet_type)
        ids = _identities(out)
        assert ids == [r.identity for r in corpus.rows_for(stage)]
        seen.update(ids)
    assert max(seen.values()) == 1
    assert set(seen) == {r.identity for r in corpus.rows}


def test_per_type_cuts_drop_ineligible_jets(corpus) -> None:  # noqa: ANN001
    out = _full(make_reader(corpus.root))
    jets = out["raw.jets"]
    assert corpus.ineligible_small_pt and corpus.ineligible_large_pt
    small_pt = set(jets["pt"][jets["jet_type"] == SMALL_TYPE].tolist())
    large_pt = set(jets["pt"][jets["jet_type"] == LARGE_TYPE].tolist())
    assert not small_pt & corpus.ineligible_small_pt
    assert not large_pt & corpus.ineligible_large_pt
    assert (jets["R10TruthLabel_R22v1"][jets["jet_type"] == LARGE_TYPE] != 3).all()


def test_config_fingerprint_tracks_both_sides(corpus) -> None:  # noqa: ANN001
    base = make_reader(corpus.root).config_fingerprint()
    no_split = {"groups": SMALL_GROUPS, "cuts": {"global_cuts": SMALL_CUTS["global_cuts"]}}
    assert make_reader(corpus.root, small=no_split).config_fingerprint() != base
    groups = {**LARGE_GROUPS, "jets": {**LARGE_GROUPS["jets"]}}
    groups["jets"]["branches"] = {
        k: v for k, v in groups["jets"]["branches"].items() if k != "GhostCHadronsFinalCount"
    }
    large = {"groups": groups, "cuts": LARGE_CUTS}
    assert make_reader(corpus.root, large=large).config_fingerprint() != base
    assert make_reader(corpus.root, num=5).config_fingerprint() != base
    assert make_reader(corpus.root, large_type=2).config_fingerprint() != base
    assert make_reader(corpus.root, jets="small_r").config_fingerprint() != base

    first = make_reader(corpus.root)
    again = make_reader(corpus.root)
    assert again.config_fingerprint() == first.config_fingerprint()
    assert len(again) == len(first)
    _assert_same(_full(again), _full(first))


def _large_with_tracks(**changes) -> dict:  # noqa: ANN003
    """The large-R side config with its ``tracks`` group modified."""
    tracks = {**LARGE_GROUPS["tracks"], **changes}
    return {"groups": {**LARGE_GROUPS, "tracks": tracks}, "cuts": LARGE_CUTS}


def test_mismatched_jagged_field_sets_raise_at_construction(corpus) -> None:  # noqa: ANN001
    branches = dict(LARGE_GROUPS["tracks"]["branches"])
    del branches["qOverP"]
    with pytest.raises(ConfigError, match="tracks"):
        make_reader(corpus.root, large=_large_with_tracks(branches=branches))


def test_mismatched_pad_max_raises_at_construction(corpus) -> None:  # noqa: ANN001
    with pytest.raises(ConfigError, match="pad_max"):
        make_reader(corpus.root, large=_large_with_tracks(pad_max=TRACK_PAD + 1))


def test_mismatched_on_disk_dtype_raises_at_prepare(corpus) -> None:  # noqa: ANN001
    branches = dict(LARGE_GROUPS["tracks"]["branches"])
    branches["d0"] = "ft1lR_trk_ftagTruthOriginLabel"  # int32 on disk vs small's float32
    reader = make_reader(corpus.root, large=_large_with_tracks(branches=branches))
    with pytest.raises(ConfigError, match="d0"):
        len(reader)


@pytest.mark.parametrize("key", ["tree", "filename", "num", "stage", "unroll"])
def test_reader_owned_key_in_side_mapping_raises(corpus, key: str) -> None:  # noqa: ANN001
    small = {"groups": SMALL_GROUPS, "cuts": SMALL_CUTS, key: None}
    with pytest.raises(ConfigError, match=key):
        make_reader(corpus.root, small=small)


def test_unknown_side_key_raises(corpus) -> None:  # noqa: ANN001
    small = {"groups": SMALL_GROUPS, "cuts": SMALL_CUTS, "group_field": None}
    with pytest.raises(ConfigError, match="unknown"):
        make_reader(corpus.root, small=small)


def test_equal_jet_types_raise(corpus) -> None:  # noqa: ANN001
    with pytest.raises(ConfigError, match="differ"):
        make_reader(corpus.root, small_type=1, large_type=1)


def test_jet_type_field_collision_raises(corpus) -> None:  # noqa: ANN001
    with pytest.raises(ConfigError, match="pt"):
        make_reader(corpus.root, jet_type_field="pt")


_TRACKS_NO_PAD = {"groups": {**_ONE_JET["groups"], "tracks": {"branches": {"d0": "d0"}}}}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"jets": "medium", "small": _ONE_JET}, "jets='medium'"),
        ({"jets": "small_r"}, "needs a 'small' mapping"),
        ({"jets": "small_r", "small": _ONE_JET, "large": _ONE_JET}, "takes no 'large' mapping"),
        ({"jets": "both", "small": _ONE_JET, "large": _ONE_JET, "unroll": None}, "needs unroll"),
        ({"jets": "small_r", "small": _TRACKS_NO_PAD}, "must set pad_max"),
    ],
    ids=["unknown-mode", "missing-side", "inactive-side", "both-without-unroll", "no-pad-max"],
)
def test_mode_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        FTAG1LITEReader(**kwargs)


def test_no_grouping_surface() -> None:
    for name in (
        *("group_by", "group_field", "group_stream", "group_rows", "warm_open"),
        *("first_touch_paths", "file_entries", "read_global_rows", "max_open_files"),
    ):
        assert not hasattr(FTAG1LITEReader, name), name
        assert not hasattr(UprootReader, name), name
    # the container list is FTAG1LITE-only: UprootReader stays ungrouped
    assert hasattr(FTAG1LITEReader, "group_names")
    assert not hasattr(UprootReader, "group_names")


def test_declare_io_and_label_universe_are_unions(corpus) -> None:  # noqa: ANN001
    reader = make_reader(corpus.root)
    keys = set(flatten_spec(reader.declare_io(Mode.FIT).produces))
    assert {"raw.jets", "raw.event", "raw.tracks", "masks.tracks"} <= keys
    assert {"raw.towers", "masks.towers", "meta.rows"} <= keys
    assert not any("jet_type" in k for k in keys)
    assert reader.streams == ("jets", "tracks", "towers", "event")
    universe = set(reader.label_universe())
    assert "labels.jets.jet_type" in universe
    assert "labels.jets.R10TruthLabel_R22v1" in universe
    assert "labels.jets.ptFromTruthDressedWZJet" in universe
    assert "labels.towers.pt" in universe
    assert reader.schema_group("jets").fields["jet_type"] == "int64"


def test_schema_lookups_use_a_seeded_schema_without_preparing(corpus, monkeypatch) -> None:  # noqa: ANN001
    ref = make_reader(corpus.root)
    ref.prepare()
    assert ref.schema is not None
    manifest = CorpusManifest(
        entries=[],
        schema={s: dict(g.fields) for s, g in ref.schema.groups.items()},
    )
    fresh = make_reader(corpus.root)
    assert apply_schema(manifest, fresh)

    def _boom(*a, **k) -> None:  # noqa: ANN002, ANN003
        raise AssertionError("prepare called despite seeded schema")

    monkeypatch.setattr(fresh, "prepare", _boom)
    for side in fresh.sides.values():
        monkeypatch.setattr(side, "prepare", _boom)

    assert fresh.label_universe() == ref.label_universe()
    for stream in ("jets", "tracks"):
        assert fresh.schema_group(stream).fields == ref.schema_group(stream).fields
    assert fresh._table is None


def test_bind_with_union_demand_serves_exactly_the_demanded_fields(corpus) -> None:  # noqa: ANN001
    reader = make_reader(corpus.root)
    demand = {"jets": dict.fromkeys(("ptFromTruthDressedWZJet", "R10TruthLabel_R22v1", "jet_type"), "test")}
    reader.bind(WorkerCtx(mode=Mode.FIT, read_fields=demand, seed=0))
    out = reader.read_block(reader.row_blocks()[0], Mode.FIT)
    jets = out["raw.jets"]
    assert jets.dtype.names == ("ptFromTruthDressedWZJet", "R10TruthLabel_R22v1", "jet_type")
    large = jets["jet_type"] == LARGE_TYPE
    assert np.isnan(jets["ptFromTruthDressedWZJet"][large]).all()
    assert (jets["R10TruthLabel_R22v1"][~large] == -1).all()
    assert (jets["R10TruthLabel_R22v1"][large] != -1).all()

    bad = make_reader(corpus.root)
    with pytest.raises(SchemaError, match="nope"):
        bad.bind(WorkerCtx(mode=Mode.FIT, read_fields={"jets": {"nope": "test"}}, seed=0))


def test_pickled_reader_reprepares_and_reads_identically(corpus) -> None:  # noqa: ANN001
    reader = make_reader(corpus.root)
    full = _full(reader)
    clone = pickle.loads(pickle.dumps(reader))  # noqa: S301 - own object round-trip
    assert clone._table is None  # noqa: SLF001
    _assert_same(_full(clone), full)


def test_exported_from_salt_data_and_sources_delegate(corpus) -> None:  # noqa: ANN001
    import salt.data

    reader = make_reader(corpus.root)
    assert isinstance(reader, salt.data.FTAG1LITEReader)
    assert reader.sources() == corpus.files
    assert isinstance(reader.source_path, Path)
    for side, kept in (("small", corpus.small_kept), ("large", corpus.large_kept)):
        seam = reader.sides[side].entry_rows()
        assert [p for p, _ in seam] == corpus.files
        for (_, got), want in zip(seam, kept, strict=True):
            np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("stage", [None, "val"])
@pytest.mark.parametrize("side", ["small", "large"])
def test_single_mode_equals_plain_uproot_reader(corpus, side: str, stage: str | None) -> None:
    single = make_side_reader(corpus.root, side, stage)
    plain = make_plain_reader(corpus.root, side, stage)
    n = len(plain)
    assert len(single) == n
    from salt.data.manifest import build_manifest

    blocks = plain.row_blocks()
    # A plain UprootReader is ungrouped (group 0); FTAG1LITE blocks carry container ids.
    single_blocks = single.row_blocks()
    assert [(b.start, b.stop) for b in single_blocks] == [(b.start, b.stop) for b in blocks]
    names = single.group_names
    assert names == list(dict.fromkeys(p.parent.name for p in corpus.files))
    entries = build_manifest(plain).entries
    assert all(e.path for e in entries)
    paths = [Path(e.path) for e in entries]
    assert [b.group for b in single_blocks] == [names.index(p.parent.name) for p in paths]
    _assert_same(_full(single), _full(plain))
    _assert_same(_full(single, Mode.TEST), _full(plain, Mode.TEST))  # meta.rows included
    cut = blocks[-1].start
    for rows in (slice(0, 3), slice(max(cut - 2, 0), min(cut + 3, n)), slice(max(n - 5, 0), n)):
        _assert_same(single.read(rows, Mode.FIT), plain.read(rows, Mode.FIT))
    io_single = flatten_spec(single.declare_io(Mode.FIT).produces)
    io_plain = flatten_spec(plain.declare_io(Mode.FIT).produces)
    assert {k: v.shape for k, v in io_single.items()} == {k: v.shape for k, v in io_plain.items()}
    assert single.schema_group("jets").fields == plain.schema_group("jets").fields
    assert "jet_type" not in single.schema_group("jets").fields
    assert single.label_universe() == plain.label_universe()


@pytest.mark.parametrize("side", ["small", "large"])
def test_single_mode_num_caps_rows(corpus, side: str) -> None:  # noqa: ANN001
    plain = _full(make_plain_reader(corpus.root, side))
    total = len(plain["raw.jets"])
    capped = make_side_reader(corpus.root, side, num=7)
    assert len(capped) == 7
    _assert_blocks_tile(capped, corpus)
    _assert_same(_full(capped), _slice(plain, slice(0, 7)))
    assert len(make_side_reader(corpus.root, side, num=10 * total)) == total


def _valid_groups(side: str) -> dict:
    """The side's groups with a scalar jet field named ``valid`` mapped onto ``pt``."""
    groups = copy.deepcopy(SMALL_GROUPS if side == "small" else LARGE_GROUPS)
    groups["jets"]["branches"]["valid"] = "pt"
    return groups


@pytest.mark.parametrize("side", ["small", "large"])
def test_scalar_valid_field_single_modes_equal_plain_uproot_reader(corpus, side: str) -> None:  # noqa: ANN001
    """A configured scalar ``valid`` is served exactly as a plain `UprootReader` serves it."""
    groups, cuts = _valid_groups(side), SMALL_CUTS if side == "small" else LARGE_CUTS
    single = make_reader(corpus.root, jets=f"{side}_r", **{side: {"groups": groups, "cuts": cuts}})
    plain = SeamUprootReader(
        groups=groups,
        cuts=GlobalObjectCuts(**cuts),
        filename=corpus_glob(corpus.root),
        tree=TREE,
        unroll="jets",
    )
    assert len(single) == len(plain)
    _assert_same(_full(single), _full(plain))
    assert "valid" in single.schema_group("jets").fields
    assert single.schema_group("jets").fields == plain.schema_group("jets").fields
    io_single = flatten_spec(single.declare_io(Mode.FIT).produces)
    io_plain = flatten_spec(plain.declare_io(Mode.FIT).produces)
    assert {k: v.shape for k, v in io_single.items()} == {k: v.shape for k, v in io_plain.items()}
    assert single.label_universe() == plain.label_universe()
    outs = []
    for reader in (single, plain):
        demand = {"jets": {"valid": "test", "pt": "test"}}
        reader.bind(WorkerCtx(mode=Mode.FIT, read_fields=demand, seed=0))
        outs.append(reader.read_block(reader.row_blocks()[0], Mode.FIT))
    out = outs[0]
    assert out["raw.jets"].dtype.names == ("pt", "valid")
    np.testing.assert_array_equal(out["raw.jets"]["valid"], out["raw.jets"]["pt"])
    _assert_same(outs[0], outs[1])


def _valid_both(corpus) -> FTAG1LITEReader:  # noqa: ANN001
    """A ``jets: both`` reader whose sides both map a scalar ``valid`` jet field onto ``pt``."""
    return make_reader(
        corpus.root,
        small={"groups": _valid_groups("small"), "cuts": SMALL_CUTS},
        large={"groups": _valid_groups("large"), "cuts": LARGE_CUTS},
    )


def test_scalar_valid_field_in_both_mode_is_side_ordered(corpus) -> None:  # noqa: ANN001
    """``both`` serves one float32 ``jets.valid`` per side-ordered row, outside the labels."""
    both = _valid_both(corpus)
    jets = _full(both)["raw.jets"]
    assert jets.dtype.names.count("valid") == 1
    assert jets["valid"].dtype == np.dtype("float32")
    np.testing.assert_array_equal(jets["valid"], jets["pt"])
    assert "labels.jets.valid" not in both.label_universe()


def test_jagged_stream_keeps_exactly_one_synthetic_valid(corpus) -> None:  # noqa: ANN001
    """A jagged stream still carries one synthetic bool ``valid`` that the mask inverts."""
    both = _valid_both(corpus)
    out = _full(both)
    tracks = out["raw.tracks"]
    assert tracks.dtype.names.count("valid") == 1
    assert tracks["valid"].dtype == np.dtype(bool)
    assert both.schema_group("tracks").fields["valid"] == "bool"
    assert np.array_equal(out["masks.tracks"], ~tracks["valid"])


def test_defaults() -> None:
    reader = FTAG1LITEReader(small=_ONE_JET)
    assert reader.jets == "small_r"
    assert reader.tree == "CollectionTree"
    assert reader.unroll == "jets"
    assert tuple(reader.sides) == ("small",)
    side = reader.sides["small"]
    assert type(side) is UprootReader
    assert (side.tree, side.unroll, side.num) == ("CollectionTree", "jets", -1)


def test_defaults_are_overridable() -> None:
    reader = FTAG1LITEReader(jets="large_r", large=_ONE_JET, tree="OtherTree", unroll=None)
    assert reader.jets == "large_r"
    assert reader.tree == "OtherTree"
    assert reader.unroll is None
    assert tuple(reader.sides) == ("large",)
    assert (reader.sides["large"].tree, reader.sides["large"].unroll) == ("OtherTree", None)


def test_with_source_and_restage_keep_class_and_mode(corpus, tmp_path) -> None:  # noqa: ANN001
    reader = make_reader(corpus.root, jets="large_r", stage="val")
    reader.name = "reader"
    clone = reader.with_source(corpus.glob, num=5, stage="train")
    assert type(clone) is type(reader)
    assert (clone.jets, clone.name, clone.stage, clone.num) == ("large_r", "reader", "train", 5)
    assert tuple(clone.sides) == ("large",)
    assert type(clone.sides["large"]) is type(reader.sides["large"])
    assert len(clone) == 5

    staged = reader.restage(tmp_path / "stage")
    assert type(staged) is type(reader)
    assert (staged.jets, staged.name, staged.stage) == ("large_r", "reader", "val")
    assert all(p.is_relative_to(tmp_path / "stage") for p in staged.sources())
    assert len(staged) == len(reader)


def test_broadcast_event_group_on_a_minitree(tmp_path: Path) -> None:
    """A ``broadcast`` event group repeats its entry-axis value onto every jet row."""
    pytest.importorskip("uproot")
    pytest.importorskip("awkward")
    from salt.tests._fixtures.easyjet_minitree import build_fixture_arrays, write_minitree

    container_dir = tmp_path / (
        "mc23_13p6TeV_mc23_13p6TeV.601190.PhPy8EG_AZNLO_Zmumu.deriv.DAOD_FTAG1LITE."
        "e8514_s4159_r15530_p7356"
    )
    container_dir.mkdir()
    arrays = build_fixture_arrays(seed=11)
    write_minitree(container_dir / "file_0.root", arrays)
    groups = {
        "jets": {"branches": {"pt": "recojet_antikt4PFlow_pt_NOSYS"}, "jagged": False},
        "event": {"branches": {"eventNumber": "eventNumber"}, "jagged": False, "broadcast": True},
    }
    reader = FTAG1LITEReader(
        jets="small_r",
        small={"groups": groups},
        filename=str(container_dir),
        tree="AnalysisMiniTree",
    )
    njets = np.array(arrays["njets"])
    n = len(reader)
    assert n == int(njets.sum())
    out = reader.read(slice(0, n), Mode.FIT)
    expected_event_numbers = np.repeat(arrays["eventNumber"], njets)
    np.testing.assert_array_equal(out["raw.event"]["eventNumber"], expected_event_numbers)


def test_nested_tree_adaptor_reshapes_two_flat_reads(tmp_path) -> None:  # noqa: ANN001
    """`_NestedTree` turns a flat ``var * T`` branch + a per-jet count branch into
    ``[event][jet][const]``: empty first event, unequal per-jet counts, a zero-constituent
    jet, float32/int32 preserved, entry windows honoured by ``.array`` and ``.arrays``.
    """
    uproot = pytest.importorskip("uproot")
    ak = pytest.importorskip("awkward")
    from salt.tests._fixtures.joint_ftag1lite_corpus import (
        SMALL_PREFIX,
        TREE,
        _jagged,
        _NestedBranch,
        _NestedTree,
    )

    pt = f"{SMALL_PREFIX}pt"
    count = f"{SMALL_PREFIX}ft1l_trk_n"
    d0 = f"{SMALL_PREFIX}ft1l_trk_d0"
    label = f"{SMALL_PREFIX}ft1l_trk_ftagTruthOriginLabel"
    f32, i32 = np.float32, np.int32
    branches = {
        pt: _jagged(
            [np.array([], f32), np.array([10.0, 20.0, 30.0], f32), np.array([40.0], f32)], f32
        ),
        count: _jagged([np.array([], i32), np.array([2, 0, 1], i32), np.array([1], i32)], i32),
        d0: _jagged([np.array([], f32), np.array([1.0, 2.0, 3.0], f32), np.array([4.0], f32)], f32),
        label: _jagged([np.array([], i32), np.array([7, 8, 9], i32), np.array([10], i32)], i32),
    }
    path = tmp_path / "nested.root"
    with uproot.recreate(path) as f:
        f.mktree(TREE, branches)

    with uproot.open(f"{path}:{TREE}") as real:
        tree = _NestedTree(real)

        # (1) full read: nested per event, leaf dtype preserved
        full = tree[d0].array(library="ak")
        assert isinstance(full, ak.Array)
        assert full.to_list() == [[], [[1.0, 2.0], [], [3.0]], [[4.0]]]
        assert "var * var * float32" in str(full.type)
        assert full.ndim == 3
        ints = tree[label].array(library="ak")
        assert ints.to_list() == [[], [[7, 8], [], [9]], [[10]]]
        assert "var * var * int32" in str(ints.type)
        assert ints.ndim == 3

        # (2) the probe's one-entry read of an EMPTY first event still types concretely
        first = tree[d0].array(entry_stop=1, library="ak")
        assert first.to_list() == [[]]
        assert "var * var * float32" in str(first.type)
        assert "unknown" not in str(first.type)
        assert first.ndim == 3

        # (3) a nonzero entry window through `.array`
        window = tree[d0].array(entry_start=1, entry_stop=3, library="ak")
        assert window.to_list() == full[1:3].to_list()

        # (4) the same window through `.arrays`: nested + real branches merged
        got = tree.arrays([d0, label, pt], entry_start=1, entry_stop=3, library="ak", how=dict)
        assert set(got) == {d0, label, pt}
        assert got[d0].to_list() == window.to_list()
        assert got[label].to_list() == ints[1:3].to_list()
        assert "var * var * int32" in str(got[label].type)
        assert got[pt].to_list() == [[10.0, 20.0, 30.0], [40.0]]

        # (5) metadata: no interpretation (-> probe), per-file C++ typename from the leaf dtype
        assert tree[d0].interpretation is None
        assert tree[d0].typename == "std::vector<std::vector<float>>"
        assert tree[label].typename == "std::vector<std::vector<int32_t>>"

        # (6) the count branch and plain jet branches resolve to the real uproot branches
        assert not isinstance(tree[count], _NestedBranch)
        assert tree[count].array(library="ak").to_list() == [[], [2, 0, 1], [1]]
        assert not isinstance(tree[pt], _NestedBranch)

        # (7) everything else delegates to the real tree
        assert tree.num_entries == 3
        assert count in tree.keys()


# --------------------------------------------------------------------------- #
# FTAG1LITEStreamingDataset: shard scopes, schedules, epochs (interleaved in-memory stub)
# --------------------------------------------------------------------------- #

REAL_GROUP_ROWS: list[int] = json.loads(
    (Path(__file__).parents[3] / "_fixtures" / "ftag1lite_group_rows.json").read_text()
)["group_rows"]  # the real 236 per-container FTAG1LITE row counts
EVEN_SIZES = [40, 130, 8, 62]  # 240 rows: a drop-last FIT epoch emits every row
RAGGED_SIZES = [37, 411, 5]  # shards do NOT divide evenly: the drop-last / tail cases
SIZES4 = [60, 200, 45, 90]  # residency guard: 4 groups, 395 rows, one dominant group
SCOPES = ("per_rank", "owner")


def _interval(index: int, n_shards: int, n_rows: int) -> tuple[int, int]:
    """Independent restatement of ``[floor(i*N/S), floor((i+1)*N/S))``."""
    return (index * n_rows) // n_shards, ((index + 1) * n_rows) // n_shards


def _assert_config_error(build: Callable[[], object], match: str | None = None) -> None:
    """`build()` (or sizing / starting it) raises `ConfigError`, wherever it is detected."""
    with pytest.raises(ConfigError, match=match):
        dataset = build()
        len(dataset)  # type: ignore[arg-type]
        next(iter(dataset))  # type: ignore[call-overload]


def _assert_tiles_exactly(
    assignment: list[dict[int, tuple[int, int]]], group_rows: list[int]
) -> None:
    """Every group's intervals, unioned over all shards, tile ``[0, n_g)`` (``O(G + S)``)."""
    per_group: dict[int, list[tuple[int, int]]] = {g: [] for g in range(len(group_rows))}
    for mapping in assignment:
        for g, (lo, hi) in mapping.items():
            assert hi > lo, f"empty interval ({lo}, {hi}) should have been omitted"
            per_group[g].append((lo, hi))
    for g, n_g in enumerate(group_rows):
        cur = 0
        for lo, hi in sorted(per_group[g]):
            assert lo == cur, f"group {g}: gap or overlap at row {cur} (next interval starts {lo})"
            cur = hi
        assert cur == n_g, f"group {g}: covered up to {cur}, expected {n_g}"


def _reference_rank_cut(rows: list[int], n_ranks: int, seed: int) -> list[dict]:
    """Independent ``O(G*R)`` owner rank cut: concatenate in ``default_rng(seed).permutation(G)``
    order, cut with `_interval`, map each piece back to group-local coordinates.
    """
    order = np.random.default_rng(seed).permutation(len(rows)).tolist()
    starts = np.concatenate([[0], np.cumsum([rows[g] for g in order])]).astype(np.int64)
    total = int(starts[-1])
    out: list[dict[int, tuple[int, int]]] = [{} for _ in range(n_ranks)]
    for r in range(n_ranks):
        lo, hi = _interval(r, n_ranks, total)
        for j, g in enumerate(order):
            a, b = max(lo, int(starts[j])), min(hi, int(starts[j + 1]))
            if b > a:
                out[r][g] = (a - int(starts[j]), b - int(starts[j]))
    return out


_GRID_REAL = [
    (REAL_GROUP_ROWS, 4),
    (REAL_GROUP_ROWS, 16),
    (REAL_GROUP_ROWS, 236),
    (REAL_GROUP_ROWS, 1024),
]
_GRID_RANDOM = [(_random_group_rows(seed=0, n_groups=3000), 64)]


@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize(("group_rows", "n_shards"), _GRID_REAL + _GRID_RANDOM)
def test_coverage_and_disjointness(group_rows: list[int], n_shards: int, scope: str) -> None:
    """Every row of every group lands in exactly one shard, at S<G, S==G and S>G."""
    assignment = shard_assignment(group_rows, n_shards, scope=scope)
    assert len(assignment) == n_shards
    _assert_tiles_exactly(assignment, group_rows)


_FIXED_CELL_COUNTS = [
    {0: 55, 1: 464, 2: 3375, 3: 2405, 4: 2050},
    dict.fromkeys(range(9), 3) | {9: 100},
    {0: 1217, 1: 4292, 2: 3094, 3: 51, 4: 494, 5: 107, 6: 3928, 7: 89},
    {0: 108, 1: 4830, 2: 4074, 3: 226, 4: 62},
    {g: n // 1000 for g, n in enumerate(REAL_GROUP_ROWS)},
    {g: hi - lo for g, (lo, hi) in shard_assignment(REAL_GROUP_ROWS, 1024, "per_rank")[0].items()},
]

_SMALL_POOL_SIZES = [7, 128, 1000]


@pytest.mark.parametrize("pool_rows", _SMALL_POOL_SIZES)
@pytest.mark.parametrize("counts", _FIXED_CELL_COUNTS)
def test_pool_schedule_multiset_exact_over_an_epoch(counts: dict[int, int], pool_rows: int) -> None:
    """Concatenating every `pools()` output reproduces `Counter(counts)` exactly."""
    seen: Counter = Counter()
    for pool in PoolSchedule(counts).pools(pool_rows):
        seen.update(pool.tolist())
    assert seen == Counter(counts)


_OWNER_RW = [(1, 1), (1, 4), (3, 4), (8, 2), (3, 1)]
_OWNER_G = [3, 30, 236]


def _owner_rank_atoms(
    assignment: list[dict[int, tuple[int, int]]], rank: int, n_workers: int
) -> dict[int, tuple[int, int]]:
    """Rank `rank`'s atoms: ``{group: (min lo, max hi)}`` over its worker shards."""
    atoms: dict[int, tuple[int, int]] = {}
    for w in range(n_workers):
        for g, (lo, hi) in assignment[rank * n_workers + w].items():
            cur = atoms.get(g, (lo, hi))
            atoms[g] = (min(cur[0], lo), max(cur[1], hi))
    return atoms


@pytest.mark.parametrize("n_groups", _OWNER_G)
@pytest.mark.parametrize(("n_ranks", "n_workers"), _OWNER_RW)
def test_owner_coverage_and_disjointness(n_ranks: int, n_workers: int, n_groups: int) -> None:
    """``owner``: tiling, exact worker strides, adjacent abutting straddlers, balance, rank cut."""
    seed = n_ranks * 10_000 + n_workers * 100 + n_groups
    group_rows = _owner_group_rows(seed=seed, n_groups=n_groups)
    n_shards = n_ranks * n_workers
    assignment = shard_assignment(group_rows, n_shards, "owner", seed=SEED, n_workers=n_workers)
    assert len(assignment) == n_shards
    _assert_tiles_exactly(assignment, group_rows)

    atoms_by_rank = [_owner_rank_atoms(assignment, r, n_workers) for r in range(n_ranks)]
    for r in range(n_ranks):
        for g, (lo, hi) in atoms_by_rank[r].items():
            for w in range(n_workers):
                a, b = _interval(w, n_workers, hi - lo)
                shard = assignment[r * n_workers + w]
                if b > a:
                    assert shard.get(g) == (lo + a, lo + b), f"rank {r} worker {w} group {g}"
                else:
                    assert g not in shard, f"rank {r} worker {w} group {g}: empty slice kept"

    ranks_by_group: dict[int, list[int]] = {}
    for r in range(n_ranks):
        for g in atoms_by_rank[r]:
            ranks_by_group.setdefault(g, []).append(r)
    two_rank = [g for g, ranks in ranks_by_group.items() if len(ranks) == 2]
    assert len(two_rank) <= n_ranks - 1
    for g in two_rank:
        r1, r2 = sorted(ranks_by_group[g])
        assert r2 - r1 == 1, f"group {g} spans non-adjacent ranks {r1}, {r2}"
        lo1, hi1 = atoms_by_rank[r1][g]
        lo2, hi2 = atoms_by_rank[r2][g]
        assert (lo1, hi2) == (0, group_rows[g])
        assert hi1 == lo2, f"group {g}: ranks {r1}/{r2} intervals don't abut"

    n_total = sum(group_rows)
    for r in range(n_ranks):
        rank_rows = sum(_row_counts(assignment[r * n_workers : (r + 1) * n_workers]))
        assert abs(rank_rows - n_total / n_ranks) <= 1
        n_atoms = len(atoms_by_rank[r])
        for w in range(n_workers):
            shard_rows = _row_counts([assignment[r * n_workers + w]])[0]
            # one worker can take every atom's remainder: the bound scales with the atom count
            assert abs(shard_rows - n_total / n_shards) <= 1 + n_atoms

    if n_workers == 1:
        assert assignment == _reference_rank_cut(group_rows, n_ranks, SEED)


_N_RANDOM_FIXTURES = 1000
_WINDOWS_FULL_CHECK_BUDGET = 200_000  # T * window entries


_ADVERSARIAL_LABEL_CHECKS: dict[str, Callable[[WindowSchedule], bool]] = {
    "mixed_m_zero_and_nonzero": lambda s: bool((s.extra == 0).any()) and bool((s.extra > 0).any()),
    "m_extremes_T_minus_1_and_1": lambda s: int(s.extra[0]) == s.n_windows - 1
    and int(s.extra[1]) == 1,
    "one_row_group_M_zero": lambda s: int(s.quotas[0]) == 0,
    "single_atom": lambda s: len(s.groups) == 1,
    "k_zero_all_integer": lambda s: s.k == 0,
    "k_equals_g_minus_one": lambda s: s.k == len(s.groups) - 1,
    "T_equals_one": lambda s: s.n_windows == 1,
    "window_equals_one": lambda s: s.window == 1,
    "plan_real_shape": lambda s: len(s.groups) == 12 and s.n_windows == 2000,
}


def _check_windows_against_matrix(
    schedule: WindowSchedule, matrix: np.ndarray, window: int, t_windows: int
) -> None:
    """`windows()` vs the `counts()` matrix: length, ascending ids, bincount == row (<= 5 if big)."""
    groups = schedule.groups
    n_check = t_windows if t_windows * window <= _WINDOWS_FULL_CHECK_BUDGET else min(t_windows, 5)
    minlength = int(groups.max()) + 1 if len(groups) else 0
    for t, window_groups in enumerate(schedule.windows()):
        if t >= n_check:
            break
        window_groups = np.asarray(window_groups)
        assert len(window_groups) == window
        assert bool((np.diff(window_groups) >= 0).all()), f"window {t}: ids not ascending"
        bincount = np.bincount(window_groups, minlength=minlength)
        for gi, g in enumerate(groups):
            assert int(bincount[g]) == int(matrix[t, gi]), f"window {t} group {g}"


def _window_schedule_invariants(counts: dict[int, int], n_rows: int, window: int) -> None:
    """Every invariant `test_window_schedule_exact` claims, for one fixture."""
    schedule = WindowSchedule(counts, n_rows, window)
    lens, total = schedule.lens, schedule.total
    quotas, base, extra, k = schedule.quotas, schedule.base, schedule.extra, schedule.k
    t_windows = schedule.n_windows

    assert int(quotas.sum()) == n_rows
    assert bool((quotas <= lens).all())
    assert np.all(np.abs(quotas.astype(np.float64) - n_rows * lens.astype(np.float64) / total) < 1)
    assert int(base.sum()) + int(k) == window
    assert int(extra.sum()) == t_windows * int(k)
    if t_windows == 0:
        assert list(schedule.counts()) == []
        return

    matrix = np.stack([np.asarray(row, dtype=np.int64) for row in schedule.counts()])
    assert matrix.shape[0] == t_windows
    assert np.all(matrix.sum(axis=1) == window)
    assert bool(((matrix == base[None, :]) | (matrix == base[None, :] + 1)).all())
    per_window_dev = np.abs(matrix.astype(np.float64) - (quotas / t_windows)[None, :])
    assert np.all(per_window_dev < 1.0)
    np.testing.assert_array_equal(matrix.sum(axis=0), quotas)

    # Pfair lag: |T * bonus_c(t) - t * m_c| < T for every prefix t
    prefix_bonus = np.cumsum(matrix - base[None, :], axis=0)
    t_vals = np.arange(1, t_windows + 1, dtype=np.int64)[:, None]
    prefix_gap = np.abs(t_windows * prefix_bonus - t_vals * extra[None, :])
    assert np.all(prefix_gap < t_windows)

    for t in sorted({0, t_windows // 2, t_windows - 1}):
        np.testing.assert_array_equal(schedule.counts_at(t), matrix[t])
    _check_windows_against_matrix(schedule, matrix, window, t_windows)
    assert schedule.verify() < 1.0


def test_window_schedule_exact() -> None:
    """`WindowSchedule` per-window and prefix deviation ``< 1`` on random + adversarial cells."""
    # PF (Baruah et al. 1996) is Pfair-optimal for every k: the prefix bound holds by theorem.
    for counts, n_rows, window in _random_window_schedule_fixtures(_N_RANDOM_FIXTURES, seed=3030):
        _window_schedule_invariants(counts, n_rows, window)

    fixtures = _adversarial_window_schedule_fixtures()
    assert set(_ADVERSARIAL_LABEL_CHECKS) == {label for label, *_ in fixtures}
    for label, counts, n_rows, window in fixtures:
        assert _ADVERSARIAL_LABEL_CHECKS[label](WindowSchedule(counts, n_rows, window)), label
        try:
            _window_schedule_invariants(counts, n_rows, window)
        except AssertionError as exc:
            raise AssertionError(f"adversarial fixture {label!r} failed: {exc}") from exc


def test_owner_rejects_bad_layout() -> None:
    """Bad owner layouts and `WindowSchedule` arguments raise; ``n_rows == 0`` is empty."""
    with pytest.raises(ConfigError):
        shard_assignment(REAL_GROUP_ROWS, 6, scope="owner", n_workers=4)
    with pytest.raises(ConfigError):
        shard_assignment(REAL_GROUP_ROWS, 3, scope="owner", n_workers=2)
    for scope in SCOPES:
        with pytest.raises(ConfigError):
            shard_assignment(REAL_GROUP_ROWS, 4, scope=scope, n_workers=0)

    counts = {0: 100, 1: 200}
    with pytest.raises(ConfigError):
        WindowSchedule(counts, n_rows=10, window=0)
    with pytest.raises(ConfigError):
        WindowSchedule(counts, n_rows=17, window=5)
    with pytest.raises(ConfigError):
        WindowSchedule(counts, n_rows=1000, window=10)
    with pytest.raises(ConfigError):
        WindowSchedule(counts, n_rows=-5, window=5)
    for empty in (WindowSchedule({}, n_rows=0, window=5), WindowSchedule(counts, 0, 5)):
        assert list(empty.counts()) == []
        assert list(empty.windows()) == []
        assert empty.verify() == 0.0


def test_pf_tie_order_is_a_strict_total_order() -> None:
    """`_pf_successor_cmp` is a strict antisymmetric order, and `_pf_tie_order` sorts by it."""
    extra = np.array([3, 3, 6, 2, 3], dtype=np.int64)
    ceils = np.array([1, 1, 2, 0, 1], dtype=np.int64)
    n_win = 9
    groups = [4, 2, 0, 3, 1]
    for a in groups:
        assert _pf_successor_cmp(a, a, extra, ceils, n_win) == 0
        for c in groups:
            if a != c:
                ab = _pf_successor_cmp(a, c, extra, ceils, n_win)
                assert ab != 0
                assert ab == -_pf_successor_cmp(c, a, extra, ceils, n_win)
    order = _pf_tie_order(groups, extra, ceils, n_win)
    assert sorted(order) == sorted(groups)
    for i, a in enumerate(order):
        for c in order[i + 1 :]:
            assert _pf_successor_cmp(a, c, extra, ceils, n_win) < 0
    # identical subtask sequences tie all the way down -> the lower group id wins
    assert _pf_tie_order([1, 0, 4], extra, ceils, n_win) == [0, 1, 4]


@pytest.mark.parametrize("mode", [Mode.VAL, Mode.TEST])
@pytest.mark.parametrize("pool", [None, 100_000])
def test_eval_modes_are_per_rank_even_when_the_reader_says_owner(
    mode: Mode, pool: int | None
) -> None:
    """VAL/TEST stream ``per_rank`` even for an ``owner`` reader (pool or not)."""
    world, workers = 2, 2
    per_rank = _assignment(SIZES, 4, "per_rank")
    owner = _assignment(SIZES, 4, "owner", n_workers=workers)
    differs = False
    for shard in range(4):
        reader = InterleavedStubReader(SIZES, shard_scope="owner", shuffle_pool_rows=pool)
        dataset = _shard_dataset(reader, world, workers, shard, mode=mode)
        assert dataset.shard_scope == "per_rank"
        uids = set(_uids(_batches(dataset)))
        assert uids == _expected_uids(reader, per_rank[shard])
        differs |= uids != _expected_uids(reader, owner[shard])
    assert differs, "fixture cannot tell per_rank from owner"


def _guard_dataset(scope: str, cap: int | None, **kw) -> FTAG1LITEStreamingDataset:
    reader = _scoped(SIZES4, scope)
    kw.setdefault("block_rows", 32)
    return _dataset(reader, max_live_streams=cap, **kw)


@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("cap", [1, 2, 3])
@pytest.mark.parametrize("seam", ["len", "iter"])
def test_guard_raises_below_group_count(cap: int, scope: str, seam: str) -> None:
    """A cap below the shard's container count raises from `len` and from `__iter__`."""
    dataset = _guard_dataset(scope, cap, world_size=1, rank=0, num_workers=1, worker_id=0)
    with pytest.raises(ConfigError, match=r"max_live_streams=\d+ < G_per_shard=4"):
        if seam == "len":
            len(dataset)
        else:
            next(iter(dataset))


def test_owner_requires_null_pool() -> None:
    """FIT ``owner`` needs a null pool; VAL streams ``per_rank`` and accepts one."""
    for pool in (100_000, 100):
        _assert_config_error(
            lambda p=pool: _dataset(InterleavedStubReader(shard_scope="owner", shuffle_pool_rows=p))
        )
    dataset = _dataset(InterleavedStubReader(shard_scope="owner", shuffle_pool_rows=None))
    assert dataset.shard_scope == "owner"
    assert dataset.shuffle_pool_rows is None
    val = _dataset(InterleavedStubReader(shard_scope="owner"), mode=Mode.VAL)
    assert val.shard_scope == "per_rank"


def test_owner_super_batch_composition() -> None:
    """``owner``: the super batch (all R rank batches at one step) is stratified by container."""
    # Each shard batch == its WindowSchedule(counts_s, T*B, B) window; super batch within
    # d_{c,w} (owner shards holding c in lane w) of ideal; tiny containers: cumulative < 1.
    rng = np.random.default_rng(3034)
    others = rng.integers(100, 800, size=26).tolist()
    sizes = [round(0.45 / 0.55 * sum(others)), 1, 1, 1, *others]  # group 0 ~45%; 1-3 tiny
    n_groups, tiny_groups = len(sizes), (1, 2, 3)
    r_ranks, w_workers, b_size = 3, 2, 64
    total = sum(sizes)
    p = [n / total for n in sizes]

    batches: list[list[np.ndarray]] = []
    lens = []
    for r in range(r_ranks):
        for w in range(w_workers):
            reader = InterleavedStubReader(sizes, shard_scope="owner", shuffle_pool_rows=None)
            dataset = _dataset(
                reader,
                batch_size=b_size,
                world_size=r_ranks,
                rank=r,
                num_workers=w_workers,
                worker_id=w,
                shuffle=False,
            )
            batches.append(_batches(dataset))
            lens.append(len(dataset))
    t = lens[0]
    assert t > 0
    assert set(lens) == {t} == {len(b) for b in batches}
    assert all(len(batch) == b_size for shard in batches for batch in shard)

    assignment = _assignment(sizes, r_ranks * w_workers, "owner", n_workers=w_workers)
    sched: list[WindowSchedule] = []
    x: list[dict[int, float]] = []
    for s in range(r_ranks * w_workers):
        schedule = WindowSchedule(
            {g: hi - lo for g, (lo, hi) in assignment[s].items()}, t * b_size, b_size
        )
        sched.append(schedule)
        quotas = zip(schedule.groups, schedule.quotas, strict=True)
        x.append({int(g): float(q) / t for g, q in quotas})

    d_cw = [
        Counter(g for r in range(r_ranks) for g in assignment[r * w_workers + w])
        for w in range(w_workers)
    ]
    ideal_sched = [
        {g: sum(x[r * w_workers + w].get(g, 0.0) for r in range(r_ranks)) for g in range(n_groups)}
        for w in range(w_workers)
    ]

    for s, schedule in enumerate(sched):
        for tau in range(t):
            cnt = Counter(batches[s][tau][:, 1].tolist())
            got = np.array([cnt.get(int(g), 0) for g in schedule.groups], dtype=np.int64)
            assert int(got.sum()) == b_size, f"shard {s} tau {tau}: a group outside its schedule"
            np.testing.assert_array_equal(got, schedule.counts_at(tau), err_msg=f"s={s} tau={tau}")

    for step in range(t * w_workers):
        w, tau = step % w_workers, step // w_workers
        q = Counter(int(g) for r in range(r_ranks) for g in batches[r * w_workers + w][tau][:, 1])
        assert sum(q.values()) == r_ranks * b_size
        for g in range(n_groups):
            dd, ideal = d_cw[w].get(g, 0), ideal_sched[w][g]
            if dd == 0:
                assert q.get(g, 0) == 0 and ideal == 0
                continue
            diff = abs(q.get(g, 0) - ideal)
            assert diff < dd, f"group {g} super batch {step}: q={q.get(g, 0)} ideal={ideal:.4f}"

    for w in range(w_workers):
        for g in range(n_groups):
            off = abs(ideal_sched[w][g] - r_ranks * b_size * p[g])
            assert off < 0.5, f"FIXTURE-SANITY: group {g} lane {w}: off={off:.4f}"

    for g in tiny_groups:
        owners = [s for s in range(r_ranks * w_workers) if g in assignment[s]]
        assert len(owners) == 1, f"tiny group {g} expected exactly one owning shard"
        w_star = owners[0] % w_workers
        x_star = x[owners[0]].get(g, 0.0)
        cum_g = 0
        for step in range(t * w_workers):
            w, tau = step % w_workers, step // w_workers
            cum_g += sum(
                int(np.count_nonzero(batches[r * w_workers + w][tau][:, 1] == g))
                for r in range(r_ranks)
            )
            tau_w = (step - w_star) // w_workers + 1 if step >= w_star else 0
            assert abs(cum_g - tau_w * x_star) < 1, f"tiny group {g} shard prefix, step {step}"
            assert abs(cum_g - (step + 1) * r_ranks * b_size * p[g]) < 1, f"tiny {g} corpus"

    emitted = Counter(u for shard in batches for u in _uids(shard))
    assert max(emitted.values()) == 1, "a row was emitted more than once"
    n_emitted = sum(emitted.values())
    assert n_emitted == r_ranks * w_workers * t * b_size
    rows_per_shard = _row_counts(assignment)
    assert total - n_emitted == sum(rows - t * b_size for rows in rows_per_shard)


@pytest.mark.parametrize(
    ("shuffle_pool_rows", "batch_size", "expected"),
    [
        (None, 8, 8),
        (100, 8, 96),
        (13, 8, 8),
        (8, 8, 8),
        (100_000, 1000, 100_000),
        (100_000, 200, 100_000),
        (100_000, 3, 99_999),
        (1999, 1000, 1000),
    ],
)
def test_pool_rows_floors_to_whole_batches(
    shuffle_pool_rows: int | None, batch_size: int, expected: int
) -> None:
    """`pool_rows()`: one batch when unset, else floored to a whole number of batches."""
    reader = InterleavedStubReader([16], shuffle_pool_rows=shuffle_pool_rows)
    assert _dataset(reader, batch_size=batch_size).pool_rows() == expected


@pytest.mark.parametrize(("shuffle_pool_rows", "batch_size"), [(4, 8), (999, 1000), (1, 2)])
def test_pool_below_one_batch_is_rejected(shuffle_pool_rows: int, batch_size: int) -> None:
    """A pool smaller than one batch cannot be dealt in whole batches: `ConfigError`."""
    reader = InterleavedStubReader([16], shuffle_pool_rows=shuffle_pool_rows)
    with pytest.raises(ConfigError, match="batch_size"):
        _dataset(reader, batch_size=batch_size).pool_rows()


_POOLS = [None, 8, 13, 100, 100_000]


def test_fit_emits_only_full_batches_and_exactly_len() -> None:
    """FIT at every pool size: exactly `len` full batches per shard, the unpooled rows."""
    n_shards, batch_size = 6, 8
    for shard in range(n_shards):
        reference = None
        for pool in _POOLS:
            reader = InterleavedStubReader(RAGGED_SIZES, shuffle_pool_rows=pool)
            dataset = _shard_dataset(reader, n_shards, 1, shard, batch_size=batch_size)
            batches = _batches(dataset)
            assert len(batches) == len(dataset) > 0
            assert all(len(b) == batch_size for b in batches)
            if reference is None:
                reference = (len(dataset), Counter(_uids(batches)))
            assert (len(dataset), Counter(_uids(batches))) == reference, f"pool {pool}"


def test_pool_composition_is_exact_at_pool_granularity() -> None:
    """Pool-sized windows of batches carry exactly the unpooled run's per-container counts."""
    sizes = np.random.default_rng(202).integers(50, 900, size=32).tolist()
    batch_size, batches_per_pool = 8, 16

    def run(pool: int | None) -> list[np.ndarray]:
        reader = InterleavedStubReader(sizes, shuffle_pool_rows=pool)
        return _batches(_dataset(reader, batch_size=batch_size, seed=33))

    unpooled, pooled = run(None), run(batch_size * batches_per_pool)
    assert len(unpooled) == len(pooled)
    n_windows = len(pooled) // batches_per_pool
    assert n_windows >= 2
    for w in range(n_windows):
        lo, hi = w * batches_per_pool, (w + 1) * batches_per_pool
        assert _group_counts(pooled[lo:hi]) == _group_counts(unpooled[lo:hi]), f"window {w}"


def _lifecycle_reader(scope: str) -> InterleavedStubReader:
    return _scoped(EVEN_SIZES, scope)


_SHARD_KW = {"world_size": 2, "rank": 1, "num_workers": 2, "worker_id": 1}


@pytest.mark.parametrize("scope", SCOPES)
def test_second_iter_reshuffles_with_the_same_ownership(scope: str) -> None:
    """A second `__iter__` reshuffles; owned rows, container counts and length are unchanged."""
    reader = _lifecycle_reader(scope)
    dataset = _dataset(reader, **_SHARD_KW)
    owned = _expected_uids(reader, _assignment(EVEN_SIZES, 4, scope, n_workers=2)[3])
    first, second = _batches(dataset), _batches(dataset)
    assert _uids(first) != _uids(second)
    assert set(_uids(first)) <= owned and set(_uids(second)) <= owned
    assert _group_counts(first) == _group_counts(second)
    assert len(first) == len(second) == len(dataset)


@pytest.mark.parametrize("scope", SCOPES)
def test_constructing_at_epoch_e_reproduces_the_eth_pass(scope: str) -> None:
    """A dataset built at epoch ``e`` streams the ``e``-th pass of a continuous run."""
    continuous = _dataset(_lifecycle_reader(scope), **_SHARD_KW)
    passes = [_uids(_batches(continuous)) for _ in range(3)]
    assert len({tuple(p) for p in passes}) == 3, "consecutive passes must reshuffle"
    for e, expected in enumerate(passes):
        assert _uids(_batches(_dataset(_lifecycle_reader(scope), epoch=e, **_SHARD_KW))) == expected


@pytest.mark.parametrize("scope", SCOPES)
def test_setting_epoch_resets_the_pass_counter(scope: str) -> None:
    """A changed `epoch` restarts the pass count (loader reconstruction)."""

    def fresh(e: int) -> list[int]:
        return _uids(_batches(_dataset(_lifecycle_reader(scope), epoch=e, **_SHARD_KW)))

    dataset = _dataset(_lifecycle_reader(scope), epoch=0, **_SHARD_KW)
    _batches(dataset)
    _batches(dataset)
    dataset.epoch = 5
    assert _uids(_batches(dataset)) == fresh(5)
    assert _uids(_batches(dataset)) == fresh(6)
    dataset.epoch = 1
    assert _uids(_batches(dataset)) == fresh(1)


def test_datamodule_builds_the_ftag1lite_streaming_dataset(corpus, tmp_path) -> None:
    """`SaltDataModule` builds `FTAG1LITEStreamingDataset` for an `FTAG1LITEReader`."""
    sinks = ["inputs.jets", "inputs.event"]
    dm = SaltDataModule(
        modules={
            "reader": make_reader(corpus.root),
            "feats": Features(variables={"jets": ["pt", "jet_type"], "event": ["eventNumber"]}),
        },
        train_file=corpus.glob,
        val_file=corpus.glob,
        batch_size=8,
        num_workers=0,
        iterable=True,
        block_rows=None,
        max_live_streams=None,
        manifest={"train": str(tmp_path / "train.json"), "val": str(tmp_path / "val.json")},
        sinks={Mode.FIT: sinks, Mode.VAL: sinks},
    )
    dm.setup("fit")
    for dset in (dm.train_dset, dm.val_dset):
        assert type(dset) is FTAG1LITEStreamingDataset
        assert dset.shard_scope == "per_rank"
    assert len(dm.train_dset) > 0
    val = Counter(
        (int(e), int(t), float(p))
        for batch in dm.val_dset
        for e, t, p in zip(
            batch["inputs"]["event"][:, 0].tolist(),
            batch["inputs"]["jets"][:, 1].tolist(),
            batch["inputs"]["jets"][:, 0].tolist(),
            strict=True,
        )
    )
    assert val == Counter(r.identity for r in corpus.rows_for("val"))


# -- ROOT handle lifecycle: a file closes right after its last block in the worker's order ------


def _real_stream(corpus, *, scope="per_rank", jets="both", block_rows=16, drop_last=None, **kw):
    """One FIT worker over the real corpus (its slice spans both files of each container)."""
    pool = None if scope == "owner" else 100_000
    reader = make_reader(corpus.root, jets=jets, shard_scope=scope, shuffle_pool_rows=pool)
    feats = Features(variables={"jets": ["pt"], "event": ["eventNumber"]})
    shard = {"world_size": 1, "rank": 0, "num_workers": 1, "worker_id": 0}
    kw = {**shard, "block_rows": block_rows, "max_live_streams": None, "seed": SEED, **kw}
    kw.update(shuffle=True, drop_last=(scope == "owner") if drop_last is None else drop_last)
    sinks = ["inputs.jets", "inputs.event"]
    return FTAG1LITEStreamingDataset({"reader": reader, "feats": feats}, Mode.FIT, sinks, 8, **kw)


def _traced_pass(monkeypatch, dataset) -> tuple[list[tuple], list]:
    """One full pass logging real ``open`` / ``close`` (side, path) and completed ``read``s."""
    log: list[tuple] = []
    orig_tree, orig_close = UprootReader._tree, UprootReader.close_tree  # noqa: SLF001
    orig_read = FTAG1LITEStreamingDataset._read_block  # noqa: SLF001

    def spy_tree(self, path):  # noqa: ANN202
        if str(path) not in self._open_trees:  # this call opens the file
            log.append(("open", self.name, str(path)))
        return orig_tree(self, path)

    def spy_close(self, path) -> None:
        if str(path) in self._open_trees:  # a no-op close is not a closure
            log.append(("close", self.name, str(path)))
        orig_close(self, path)

    def spy_read(self, block):  # noqa: ANN202
        out = orig_read(self, block)
        log.append(("read", block.group, block.start, block.stop))
        return out

    monkeypatch.setattr(UprootReader, "_tree", spy_tree)
    monkeypatch.setattr(UprootReader, "close_tree", spy_close)
    monkeypatch.setattr(FTAG1LITEStreamingDataset, "_read_block", spy_read)
    return log, [{k: v.numpy() for k, v in b["inputs"].items()} for b in list(dataset)]


def _shuffled_blocks(dataset) -> dict[int, list[RowBlock]]:  # noqa: ANN001
    """The one-worker shard's blocks per group, shuffled exactly as a first `__iter__` does."""
    blocks, group_rows = dataset._resolve_layout()  # noqa: SLF001
    assignment = shard_assignment(group_rows, 1, dataset.shard_scope, seed=SEED, n_workers=1)
    mine = _partition(blocks, assignment[0], dataset.block_rows)
    rng = np.random.default_rng([SEED, 0, 0])  # (seed, effective epoch 0, shard 0)
    for group_blocks in mine.values():
        rng.shuffle(group_blocks)
    return mine


@pytest.mark.parametrize("scope", SCOPES)
def test_streaming_closes_each_file_after_its_last_block(corpus, monkeypatch, scope) -> None:
    """Every opened (side, file) closes once: right after its last block, else at pass end."""
    dataset = _real_stream(corpus, scope=scope)
    log, _ = _traced_pass(monkeypatch, dataset)
    reader = dataset._reader  # noqa: SLF001
    paths = [str(e.path) for e in reader._table]  # noqa: SLF001
    last = {paths[reader.file_index(b)]: b for bs in _shuffled_blocks(dataset).values() for b in bs}
    read_at = {RowBlock(*e[1:]): i for i, e in enumerate(log) if e[0] == "read"}
    closes = [(j, e[1:]) for j, e in enumerate(log) if e[0] == "close"]
    opened = [e[1:] for e in log if e[0] == "open"]
    assert opened and read_at, "the pass must open files and read blocks"
    assert Counter(key for _, key in closes) == Counter(set(opened)) == Counter(opened)
    for j, (side, path) in closes:
        r = read_at.get(last[path])
        if r is None:  # owner drop_last: this file's last shuffled block was never streamed
            assert scope == "owner" and j > max(read_at.values()), (side, path)
        else:  # the closes for both sides follow that read with nothing in between
            assert r < j and all(log[k][0] == "close" for k in range(r + 1, j)), (side, path)


def test_streaming_holds_at_most_the_files_a_slice_spans_per_container(corpus, monkeypatch) -> None:
    """Per side, open handles per container never exceed the 2 files the slice spans."""
    log, _ = _traced_pass(monkeypatch, _real_stream(corpus))
    live: dict[str, set[str]] = {"small": set(), "large": set()}
    peak = 0
    for kind, *rest in (e for e in log if e[0] != "read"):
        (live[rest[0]].add if kind == "open" else live[rest[0]].discard)(rest[1])
        for files in live.values():
            peak = max(peak, 0, *Counter(Path(p).parent.name for p in files).values())
        assert peak <= 2, (kind, *rest)
    per_container = Counter(p.parent.name for p in corpus.files)
    assert set(per_container.values()) == {2}  # the fixture's n_files_per_container
    for side in live:  # both files of every container are opened: the bound of 2 is tight
        opened = {e[2] for e in log if e[:2] == ("open", side)}
        assert Counter(Path(p).parent.name for p in opened) == per_container


def test_streaming_releases_every_handle_at_pass_end_and_on_generator_close(corpus) -> None:
    """Exhausting a pass, or closing its generator mid-pass, leaves no tree open on any side."""
    dataset = _real_stream(corpus)
    list(dataset)
    assert all(side._open_trees == {} for side in dataset._reader.sides.values())  # noqa: SLF001
    partial = _real_stream(corpus)
    gen = iter(partial)
    [next(gen) for _ in range(2)]
    assert any(side._open_trees for side in partial._reader.sides.values())  # noqa: SLF001
    gen.close()
    assert all(side._open_trees == {} for side in partial._reader.sides.values())  # noqa: SLF001


def test_streaming_block_order_and_batches_unchanged_by_closing(corpus, monkeypatch) -> None:
    """Closing draws no rng: block order and every emitted batch match the never-closing path."""
    dataset = _real_stream(corpus, drop_last=False)
    log, with_close = _traced_pass(monkeypatch, dataset)
    reads = [RowBlock(*e[1:]) for e in log if e[0] == "read"]
    expected = _shuffled_blocks(dataset)
    assert {g: [r for r in reads if r.group == g] for g in expected} == expected
    assert len(reads) == sum(map(len, expected.values()))
    monkeypatch.setattr(UprootReader, "close_tree", lambda self, path: None)  # handles stay open
    _, without = _traced_pass(monkeypatch, _real_stream(corpus, drop_last=False))
    assert len(with_close) == len(without) > 0
    for a, b in zip(with_close, without, strict=True):
        _assert_same(a, b)  # keys, dtypes, shapes, values (NaN == NaN)


def test_both_sides_close_their_handles(corpus, monkeypatch) -> None:
    """``jets: both``: each opened (side, file) closes exactly once and nothing stays open."""
    dataset = _real_stream(corpus, jets="both")
    log, _ = _traced_pass(monkeypatch, dataset)
    closes = Counter(e[1:] for e in log if e[0] == "close")
    assert {side for side, _ in closes} == {"small", "large"}
    assert set(closes.values()) == {1}
    assert {e[1:] for e in log if e[0] == "open"} == set(closes)
    assert all(not side._open_trees for side in dataset._reader.sides.values())  # noqa: SLF001
