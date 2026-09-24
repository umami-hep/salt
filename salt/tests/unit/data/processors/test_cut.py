"""`ConstituentSelection` — hand-rolled-oracle byte parity, sort, config surface."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.lib.recfunctions import structured_to_unstructured as s2u

from salt.data import ConstituentSelection, Features, H5StructuredReader, Labels, SaltDataset
from salt.data.processors.cut import _ConstituentCuts
from salt.data.readers.cuts import VALID_FIELD
from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import Mode
from salt.schema import dump_schema, save_schema
from salt.testing.inputs import write_dummy_file, write_dummy_norm_dict

JET_VARS = ["pt_btagJes", "eta_btagJes"]
TRACK_VARS = ["d0", "z0SinTheta", "dphi", "deta"]

_OPS = {
    ">": lambda a, v: a > v,
    "<": lambda a, v: a < v,
    ">=": lambda a, v: a >= v,
    "<=": lambda a, v: a <= v,
    "==": lambda a, v: a == v,
    "!=": lambda a, v: a != v,
}


# --------------------------------------------------------------------------- #
# (i) independent numpy oracle
# --------------------------------------------------------------------------- #


def _pad_fill(kind: str) -> float | int | bool:
    """The pad-fill value for a dtype kind, matching the shipped convention."""
    if kind == "f":
        return 0.0
    if kind == "i":
        return -1
    if kind == "b":
        return False
    return 0  # unsigned / other


def _manual_select(
    raw: np.ndarray,
    cuts: list[tuple[str, str, float]],
    pad_max: int | None,
    sort: dict[str, str] | None,
) -> np.ndarray:
    """Independent hand-rolled oracle: cut (drop) -> sort -> truncate, no processor code reused."""
    b, t = raw.shape
    valid = raw[VALID_FIELD].copy()
    keep = np.ones((b, t), dtype=bool)
    for field, op, value in cuts:
        keep &= _OPS[op](raw[field], value)
    if not cuts:
        out = np.array(raw, copy=True)
    else:  # drop: compact kept-in-order per row, then pad
        out = np.zeros_like(raw)
        for row in range(b):
            idx = [j for j in range(t) if valid[row, j] and keep[row, j]]
            n = len(idx)
            for dst, src in enumerate(idx):
                out[row, dst] = raw[row, src]
            for dst in range(n, t):
                for name in out.dtype.names or ():
                    if name != VALID_FIELD:
                        out[row, dst][name] = _pad_fill(out.dtype[name].kind)
            out[row, :n][VALID_FIELD] = True
            out[row, n:][VALID_FIELD] = False
    if sort is not None:
        var, ascending = sort["var"], sort["mode"] == "ascending"
        for row in range(b):
            v = out[row][VALID_FIELD]
            key = out[row][var].astype(np.float64)
            order_key = np.where(v, key if ascending else -key, np.inf)
            order = np.argsort(order_key, kind="stable")
            out[row] = out[row][order]
    if pad_max is not None:
        out = out[:, :pad_max]
    return out


_SLAB_DTYPE = np.dtype([
    ("pt", "f4"),
    ("label", "i4"),
    ("flag", "u1"),
    ("active", "?"),
    (VALID_FIELD, "?"),
])


def _random_slab(seed: int, b: int = 4, t: int = 6) -> np.ndarray:
    """A synthetic (B, T) structured slab: float32/int32/uint8/bool fields + valid."""
    rng = np.random.default_rng(seed)
    arr = np.zeros((b, t), dtype=_SLAB_DTYPE)
    counts = rng.integers(1, t + 1, size=b)
    for row, n in enumerate(counts):
        arr[row, :n]["pt"] = rng.uniform(0, 5, size=n).astype(np.float32)
        arr[row, :n]["label"] = rng.integers(0, 8, size=n).astype(np.int32)
        arr[row, :n]["flag"] = rng.integers(0, 5, size=n).astype(np.uint8)
        arr[row, :n]["active"] = rng.integers(0, 2, size=n).astype(bool)
        arr[row, :n][VALID_FIELD] = True
        arr[row, n:]["label"] = -1
        arr[row, n:][VALID_FIELD] = False
    return arr


# --------------------------------------------------------------------------- #
# (ii) process() vs oracle: drop / no-cuts noop / never-wastes-a-slot
# --------------------------------------------------------------------------- #


def test_process_matches_oracle_for_drop():
    """`process()` output is byte-identical to the hand-rolled drop oracle."""
    raw = _random_slab(seed=1)
    sel = ConstituentSelection({"s": {"cuts": ["pt > 2.0"]}})
    out = sel.process(Bundle({"raw": {"s": raw}}), slice(0, 4), Mode.FIT)["raw.s"]
    expected = _manual_select(raw, [("pt", ">", 2.0)], None, None)
    assert out.tobytes() == expected.tobytes()


def test_no_cuts_process_is_a_tobytes_noop():
    """A stream with no cuts/sort/pad_max passes through byte-identical."""
    raw = _random_slab(seed=3)
    sel = ConstituentSelection({"s": {}})
    out = sel.process(Bundle({"raw": {"s": raw}}), slice(0, 4), Mode.FIT)["raw.s"]
    assert out.tobytes() == raw.tobytes()


def test_masks_output_is_the_inverse_of_valid():
    """The produced ``masks.s`` is exactly ``~raw.s['valid']``."""
    raw = _random_slab(seed=4)
    sel = ConstituentSelection({"s": {}})
    produced = sel.process(Bundle({"raw": {"s": raw}}), slice(0, 4), Mode.FIT)
    np.testing.assert_array_equal(produced["masks.s"], ~produced["raw.s"][VALID_FIELD])


def test_drop_then_pad_never_wastes_a_slot():
    """4 constituents, 3 pass, pad_max=2 -> the leading 2 kept ones, both valid."""
    raw = np.zeros((1, 4), dtype=_SLAB_DTYPE)
    raw[0, :3]["pt"] = [1.0, 2.0, 3.0]
    raw[0, :3][VALID_FIELD] = True
    sel = ConstituentSelection({"s": {"cuts": ["pt >= 0"], "pad_max": 2}})
    out = sel.process(Bundle({"raw": {"s": raw}}), slice(0, 1), Mode.FIT)["raw.s"]
    assert out.shape == (1, 2)
    assert out[VALID_FIELD].all()
    np.testing.assert_array_equal(out["pt"][0], [1.0, 2.0])


def test_process_raises_keyerror_when_stream_lacks_valid_field():
    """A stream with no ``valid`` field fails loudly, naming the stream."""
    raw = np.zeros((2, 3), dtype=np.dtype([("pt", "f4")]))
    sel = ConstituentSelection({"s": {}})
    with pytest.raises(KeyError, match="valid"):
        sel.process(Bundle({"raw": {"s": raw}}), slice(0, 2), Mode.FIT)


# --------------------------------------------------------------------------- #
# (iii) sort lockstep
# --------------------------------------------------------------------------- #

_SORT_DTYPE = np.dtype([("pt", "f4"), ("id", "i4"), (VALID_FIELD, "?")])


def _sort_row(pts: list[float]) -> np.ndarray:
    n = len(pts)
    arr = np.zeros((1, n), dtype=_SORT_DTYPE)
    arr[0]["pt"] = pts
    arr[0]["id"] = np.arange(n)
    arr[0][VALID_FIELD] = True
    return arr


def test_sort_descending_permutes_all_fields_together():
    """Descending sort on pt reorders every field of the row in lockstep."""
    raw = _sort_row([5, 1, 3, 2])
    sel = ConstituentSelection({"s": {"sort": {"var": "pt", "mode": "descending"}}})
    out = sel.process(Bundle({"raw": {"s": raw}}), slice(0, 1), Mode.FIT)["raw.s"]
    np.testing.assert_array_equal(out["pt"][0], [5, 3, 2, 1])
    np.testing.assert_array_equal(out["id"][0], [0, 2, 3, 1])


def test_sort_ascending():
    """Ascending sort on pt orders the row smallest-first."""
    raw = _sort_row([5, 1, 3, 2])
    sel = ConstituentSelection({"s": {"sort": {"var": "pt", "mode": "ascending"}}})
    out = sel.process(Bundle({"raw": {"s": raw}}), slice(0, 1), Mode.FIT)["raw.s"]
    np.testing.assert_array_equal(out["pt"][0], [1, 2, 3, 5])


def test_cut_then_sort_composes():
    """A cut removing one constituent composes with a following sort."""
    raw = _sort_row([5, 1, 3, 2])
    sel = ConstituentSelection({
        "s": {"cuts": ["pt > 1"], "sort": {"var": "pt", "mode": "descending"}}
    })
    out = sel.process(Bundle({"raw": {"s": raw}}), slice(0, 1), Mode.FIT)["raw.s"]
    np.testing.assert_array_equal(out["pt"][0], [5, 3, 2, 0])
    np.testing.assert_array_equal(out[VALID_FIELD][0], [True, True, True, False])


def test_missing_sort_var_raises_keyerror():
    """A sort var absent from the stream fails loudly, naming the field."""
    raw = _sort_row([5, 1, 3, 2])
    sel = ConstituentSelection({"s": {"sort": {"var": "nope"}}})
    with pytest.raises(KeyError, match="nope"):
        sel.process(Bundle({"raw": {"s": raw}}), slice(0, 1), Mode.FIT)


# --------------------------------------------------------------------------- #
# declare_io: rewrite shape carries pad_max
# --------------------------------------------------------------------------- #


def test_declare_io_rewrites_carry_pad_max_shape():
    """`declare_io` rewrites raw+masks at the configured pad_max width."""
    io = ConstituentSelection({"s": {"pad_max": 5}}).declare_io(Mode.FIT)
    assert io.rewrites["raw"]["s"].shape == ("B", 5)
    assert io.rewrites["masks"]["s"].shape == ("B", 5)
    assert io.rewrites["masks"]["s"].dtype == "bool"


# --------------------------------------------------------------------------- #
# (iv) dataset-driven H5: plan order, shapes, byte parity vs oracle
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def h5_data(tmp_path_factory) -> dict[str, object]:
    base = tmp_path_factory.mktemp("cut_h5")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_dummy_norm_dict(nd_path, cd_path)
    h5_path = base / "data.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"file": h5_path, "schema": schema_path}


def _h5_reader(data: dict[str, object]) -> H5StructuredReader:
    return H5StructuredReader(
        groups={"jets": {}, "tracks": {}}, schema=data["schema"], filename=data["file"]
    )


def test_dataset_plan_orders_sel_between_reader_and_downstream(h5_data):
    """`sel` runs after `reader` and before `features`/`labels`; shapes + byte parity hold."""
    ds = SaltDataset(
        modules={
            "reader": _h5_reader(h5_data),
            "sel": ConstituentSelection(
                streams={"tracks": {"cuts": ["d0 < 3.5"], "pad_max": 10}}
            ),
            "features": Features(variables={"jets": JET_VARS, "tracks": TRACK_VARS}),
            "labels": Labels(),
        },
        mode=Mode.TEST,
        sinks=["inputs.jets", "inputs.tracks", "masks.tracks", "meta.rows"],
    )
    names = [s.name for s in ds.plan.steps]
    assert names.index("reader") < names.index("sel") < names.index("features")
    assert names.index("sel") < names.index("labels")

    batch = ds[np.s_[0:128]]
    assert batch["inputs"]["tracks"].shape == (128, 10, len(TRACK_VARS))
    assert batch["masks"]["tracks"].shape == (128, 10)

    pre_cut = np.array(ds.reader.read(slice(0, 128), Mode.TEST)["raw.tracks"], copy=True)
    assert np.all(pre_cut["d0"][pre_cut[VALID_FIELD]] < 3.5)  # trivial precondition of the cut
    expected = _manual_select(pre_cut, [("d0", "<", 3.5)], 10, None)
    expected_inputs = s2u(expected[TRACK_VARS], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(batch["inputs"]["tracks"]), expected_inputs)


# --------------------------------------------------------------------------- #
# (v) uproot-emitted slab: byte parity + identical plan-order path
# --------------------------------------------------------------------------- #


def test_uproot_slab_byte_parity_vs_hand_rolled_oracle(tmp_path):
    """The uproot-emitted slab takes the identical cut/sort/truncate path as H5."""
    pytest.importorskip("uproot")
    pytest.importorskip("awkward")
    from salt.data import UprootReader
    from salt.tests._fixtures.easyjet_minitree import build_fixture_arrays, write_minitree

    arrays = build_fixture_arrays(seed=99)
    path = write_minitree(tmp_path / "ej.root", arrays)
    assert 8 >= max(arrays["njets"])  # byte-parity precondition: reader pad >= pre-cut max

    branches = {
        "pt": "recojet_antikt4PFlow_pt_NOSYS",
        "eta": "recojet_antikt4PFlow_eta",
        "label": "recojet_antikt4PFlow_HadronConeExclTruthLabelID",
    }
    reader = UprootReader(
        groups={"jets": {"branches": branches, "jagged": True, "pad_max": 8}},
        filename=path,
        tree="AnalysisMiniTree",
        unroll=None,
    )
    n = len(reader)
    pre_cut = np.array(reader.read(slice(0, n), Mode.FIT)["raw.jets"], copy=True)

    sel = ConstituentSelection({"jets": {"cuts": ["pt > 100000"], "pad_max": 4}})
    out = sel.process(Bundle({"raw": {"jets": pre_cut}}), slice(0, n), Mode.FIT)["raw.jets"]

    expected = _manual_select(pre_cut, [("pt", ">", 100000.0)], 4, None)
    assert out.tobytes() == expected.tobytes()


def test_uproot_slab_through_dataset_matches_plan_order(tmp_path):
    """`SaltDataset` over an `UprootReader` places `sel` between reader and features too."""
    pytest.importorskip("uproot")
    pytest.importorskip("awkward")
    from salt.data import UprootReader
    from salt.tests._fixtures.easyjet_minitree import build_fixture_arrays, write_minitree

    arrays = build_fixture_arrays(seed=7)
    path = write_minitree(tmp_path / "ej2.root", arrays)
    branches = {"pt": "recojet_antikt4PFlow_pt_NOSYS", "eta": "recojet_antikt4PFlow_eta"}
    ds = SaltDataset(
        modules={
            "reader": UprootReader(
                groups={"jets": {"branches": branches, "jagged": True, "pad_max": 8}},
                filename=path,
                tree="AnalysisMiniTree",
                unroll=None,
            ),
            "sel": ConstituentSelection({"jets": {"pad_max": 4}}),
            "features": Features(variables={"jets": ["pt", "eta"]}),
            "labels": Labels(),
        },
        mode=Mode.TEST,
        sinks=["inputs.jets", "masks.jets", "meta.rows"],
    )
    names = [s.name for s in ds.plan.steps]
    assert names.index("reader") < names.index("sel") < names.index("features")
    batch = ds[np.s_[0 : arrays["n_events"]]]
    assert batch["inputs"]["jets"].shape[1] == 4


# --------------------------------------------------------------------------- #
# (vi) read-field demand: cut-only fields are still pulled from the reader
# --------------------------------------------------------------------------- #


def test_read_fields_demand_includes_cut_only_field(h5_data):
    """A field only referenced by the selection's cuts is still demanded, attributed to `sel`."""
    ds = SaltDataset(
        modules={
            "reader": _h5_reader(h5_data),
            "sel": ConstituentSelection(streams={"tracks": {"cuts": ["z0SinTheta < 0"]}}),
            "features": Features(variables={"jets": JET_VARS, "tracks": ["d0"]}),
            "labels": Labels(),
        },
        mode=Mode.TEST,
        sinks=["inputs.jets", "inputs.tracks", "masks.tracks", "meta.rows"],
    )
    assert ds.read_fields["tracks"]["z0SinTheta"] == "sel"


# --------------------------------------------------------------------------- #
# (vii) config surface
# --------------------------------------------------------------------------- #


def test_unknown_key_raises_config_error():
    """An unrecognised per-stream key is rejected, naming the stream."""
    with pytest.raises(ConfigError, match="unknown keys"):
        ConstituentSelection({"tracks": {"bogus": 1}})


def test_pad_max_zero_raises_config_error():
    """`pad_max: 0` is rejected."""
    with pytest.raises(ConfigError, match="pad_max"):
        ConstituentSelection({"tracks": {"pad_max": 0}})


def test_on_fail_key_is_rejected_as_unknown():
    """``on_fail`` is not a key: rejected as unknown (no compatibility path)."""
    with pytest.raises(ConfigError, match="unknown keys"):
        ConstituentSelection({"tracks": {"cuts": ["d0 < 1"], "on_fail": "drop"}})


def test_null_stream_entry_is_skipped():
    """A ``None`` stream entry is dropped, not an error, and may empty the selection."""
    sel = ConstituentSelection({"tracks": None})
    assert sel.streams == {}


def test_empty_streams_raises_config_error():
    """An empty ``streams`` mapping is rejected outright."""
    with pytest.raises(ConfigError, match="at least one entry"):
        ConstituentSelection({})


def test_constituent_selection_importable_from_salt_data():
    """`ConstituentSelection` is exported at the `salt.data` top level."""
    from salt.data import ConstituentSelection as top_level

    assert top_level is ConstituentSelection


def _instantiate_processor(init_args: dict):
    """Build a `ConstituentSelection` through jsonargparse, exactly as the CLI does."""
    from jsonargparse import ArgumentParser

    from salt.data.base import Processor

    parser = ArgumentParser(exit_on_error=False)
    parser.add_subclass_arguments(Processor, "p")
    cfg = parser.parse_object(
        {"p": {"class_path": "salt.data.ConstituentSelection", "init_args": init_args}}
    )
    return parser.instantiate_classes(cfg).p


def test_shipped_gn3_baseline_track_selection_instantiates():
    """The migrated GN3_baseline track-selection block instantiates via jsonargparse."""
    sel = _instantiate_processor({
        "streams": {
            "tracks": {
                "cuts": [
                    "d0 < 3.5",
                    "(numberOfPixelSharedHits + numberOfSCTSharedHits / 2) < 1.1",
                ],
            }
        }
    })
    assert isinstance(sel, ConstituentSelection)
    assert sel.streams["tracks"].cuts.fields == (
        "d0",
        "numberOfPixelSharedHits",
        "numberOfSCTSharedHits",
    )


def test_shipped_physlite_jet_selection_instantiates():
    """The migrated physlite jet-selection block instantiates via jsonargparse."""
    sel = _instantiate_processor({
        "streams": {
            "jets": {
                "pad_max": 20,
                "cuts": ["pt > 20000", "eta < 4.5", "eta > -4.5", "NNJvtPass > 0.5"],
            }
        }
    })
    assert isinstance(sel, ConstituentSelection)
    assert sel.streams["jets"].pad_max == 20


# --------------------------------------------------------------------------- #
# (viii) _ConstituentCuts helper
# --------------------------------------------------------------------------- #


def _tracks(n_rows: int = 2, t_dim: int = 4) -> np.ndarray:
    """A (n_rows, t_dim) constituent batch: 3 valid slots per row, 1 pad."""
    rec = np.zeros(
        (n_rows, t_dim),
        dtype=[("d0", "f4"), ("npix", "u1"), ("label", "i4"), ("flag", "?"), ("valid", "?")],
    )
    rec["d0"] = [[0.5, 9.0, 1.0, 0.0], [4.0, 0.25, 7.0, 0.0]]
    rec["npix"] = [[1, 2, 3, 0], [4, 5, 6, 0]]
    rec["label"] = [[10, 11, 12, -1], [13, 14, 15, -1]]
    rec["flag"] = True
    rec["valid"] = [[True, True, True, False], [True, True, True, False]]
    return rec


def test_constituent_cuts_fields_dedup_union() -> None:
    cc = _ConstituentCuts(cuts=("d0 < 3.5", "d0 + npix < 9"))
    assert cc.fields == ("d0", "npix")


def test_constituent_cuts_drop_compacts_and_repads() -> None:
    batch = _tracks()
    out = _ConstituentCuts(cuts=("d0 < 3.5",)).apply(batch)
    # row 0 keeps slots 0, 2 (order preserved); row 1 keeps slot 1
    np.testing.assert_array_equal(out["d0"][0], [0.5, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(out["label"][0], [10, 12, -1, -1])
    np.testing.assert_array_equal(out["valid"][0], [True, True, False, False])
    np.testing.assert_array_equal(out["d0"][1], [0.25, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(out["label"][1], [14, -1, -1, -1])
    np.testing.assert_array_equal(out["valid"][1], [True, False, False, False])
    np.testing.assert_array_equal(out["npix"][1], [5, 0, 0, 0])


def test_constituent_cuts_drop_never_keeps_a_failing_constituent() -> None:
    batch = _tracks()
    out = _ConstituentCuts(cuts=("d0 < 3.5",)).apply(batch)
    kept = out["d0"][out["valid"]]
    assert (kept < 3.5).all()


def test_constituent_cuts_no_cuts_is_a_noop() -> None:
    batch = _tracks()
    before = batch.copy()
    out = _ConstituentCuts(cuts=()).apply(batch)
    assert out.tobytes() == before.tobytes()


def test_constituent_cuts_need_the_valid_field() -> None:
    rec = np.zeros((1, 2), dtype=[("d0", "f4")])
    with pytest.raises(KeyError):
        _ConstituentCuts(cuts=("d0 < 1",)).apply(rec)


def test_constituent_cuts_refuse_a_reduction() -> None:
    with pytest.raises(ConfigError, match="reduce the constituent axis"):
        _ConstituentCuts(cuts=("sum(jets.valid) >= 4",))


def test_constituent_cuts_derived_expression_drops_like_the_oracle() -> None:
    batch = _tracks()
    npix, valid, d0 = batch["npix"].copy(), batch["valid"].copy(), batch["d0"].copy()
    out = _ConstituentCuts(cuts=("npix + npix / 2 < 3.0",)).apply(batch)
    expected_kept = valid & ((npix + npix / 2) < 3.0)
    np.testing.assert_array_equal(out["valid"].sum(axis=1), expected_kept.sum(axis=1))
    for row in range(batch.shape[0]):
        n = int(expected_kept[row].sum())
        np.testing.assert_array_equal(out["d0"][row, :n], d0[row][expected_kept[row]])
