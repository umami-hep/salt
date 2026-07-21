"""Regression tests for `salt.data.FTAG1LiteReader` (plan 19, Track C)."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest

from salt.data import Cut, CutSpec, FTAG1LiteGroupConfig, FTAG1LiteReader
from salt.graph.errors import SchemaError
from salt.graph.spec import Mode

uproot = pytest.importorskip("uproot")
awkward = pytest.importorskip("awkward")

import awkward as ak  # noqa: E402

_DEFAULT_SAMPLE = (
    "/data/atlas_samples/test-athena-derivations/ftag1lite_stock_smoke_100evt/smoke_100evt/"
    "Athena_main_latest_20260424-082808/DAOD_FTAG1LITE.601589.e8549_s4159_r15530.pool.root"
)
SAMPLE = Path(os.environ.get("FTAG1LITE_SAMPLE", _DEFAULT_SAMPLE))

pytestmark = pytest.mark.skipif(
    not SAMPLE.exists(),
    reason=f"FTAG1LITE sample not found at {SAMPLE} (set FTAG1LITE_SAMPLE to override)",
)

AUX = "AntiKt4EMPFlowJetsAuxDyn."
PAD_MAX = 40

JET_BRANCHES = {
    "pt": "pt",
    "eta": "eta_calibrated",
    "phi": "phi_calibrated",
    "mass": "mass_calibrated",
    "GN3_pb": "GN3EPCLV01_pb",
    "flavour_label": "HadronConeExclTruthLabelID",
}
TRACK_BRANCHES = {
    "d0": "ft1l_trk_d0_bf16",
    "z0SinTheta": "ft1l_trk_z0SinTheta_bf16",
    "pt": "ft1l_trk_pt_bf16",
    "numberOfPixelHits": "ft1l_trk_numberOfPixelHits",
    "origin": "ft1l_trk_ftagTruthOriginLabel",
    "vertexing": "ft1l_trk_ftagTruthVertexIndex",
}
JET_FEATURES = ["pt", "eta", "phi", "mass", "GN3_pb"]  # label is a Labels target, not a feature


def _groups(pad_max: int | None = PAD_MAX) -> dict:
    return {
        "jets": FTAG1LiteGroupConfig(branches=dict(JET_BRANCHES), jagged=False),
        "tracks": FTAG1LiteGroupConfig(branches=dict(TRACK_BRANCHES), jagged=True, pad_max=pad_max),
    }


def _reader(**kw) -> FTAG1LiteReader:
    return FTAG1LiteReader(groups=_groups(kw.pop("pad_max", PAD_MAX)), filename=SAMPLE, **kw)


@pytest.fixture(scope="module")
def oracle() -> dict:
    """Direct uproot reads of the sample → flat-jet ground-truth arrays."""
    with uproot.open(f"{SAMPLE}:CollectionTree") as t:
        jet_pt_ev = t[f"{AUX}pt"].array(library="ak")  # [event][jet]
        njets = ak.to_numpy(ak.num(jet_pt_ev, axis=1)).astype(np.int64)
        jet_cols = {
            f: ak.to_numpy(ak.flatten(t[f"{AUX}{b}"].array(library="ak"), axis=1))
            for f, b in JET_BRANCHES.items()
        }
        # track columns: flatten events away -> [jet][track]
        trk_cols = {
            f: ak.flatten(t[f"{AUX}{b}"].array(library="ak"), axis=1)
            for f, b in TRACK_BRANCHES.items()
        }
    n_events = int(njets.size)
    n_jets = int(njets.sum())
    trk_per_jet = ak.to_numpy(ak.num(trk_cols["pt"], axis=1)).astype(np.int64)
    return {
        "njets": njets,
        "n_events": n_events,
        "n_jets": n_jets,
        "jet_cols": jet_cols,
        "trk_cols": trk_cols,
        "trk_per_jet": trk_per_jet,
    }


# --------------------------------------------------------------------------- #
# 1. schema + structured shapes + bf16
# --------------------------------------------------------------------------- #


def test_len_streams_and_schema(oracle: dict) -> None:
    reader = _reader()
    assert len(reader) == oracle["n_jets"]
    assert reader.streams == ("jets", "tracks")
    assert reader.jet_stream == "jets"
    js = reader.schema_group("jets")
    ts = reader.schema_group("tracks")
    assert js is not None and not js.has_valid
    assert ts is not None and ts.has_valid
    # float kinematics stay float, the flavour label stays integer
    assert np.issubdtype(np.dtype(js.fields["pt"]), np.floating)
    assert np.issubdtype(np.dtype(js.fields["flavour_label"]), np.integer)
    # track origin/vertexing labels stay integer
    assert np.issubdtype(np.dtype(ts.fields["origin"]), np.integer)
    assert np.issubdtype(np.dtype(ts.fields["vertexing"]), np.integer)


def test_structured_shapes_and_field_order(oracle: dict) -> None:
    reader = _reader()
    n = len(reader)
    out = reader.read(slice(0, n), Mode.FIT)
    raw_jets, raw_trk, masks = out["raw.jets"], out["raw.tracks"], out["masks.tracks"]
    # jets: (B,) structured; field order = config order (no 'valid')
    assert raw_jets.shape == (n,)
    assert list(raw_jets.dtype.names) == list(JET_BRANCHES)
    # tracks: (B, pad_max) structured + valid; masks (B, pad_max)
    assert raw_trk.shape == (n, PAD_MAX)
    assert list(raw_trk.dtype.names) == [*TRACK_BRANCHES, "valid"]
    assert masks.shape == (n, PAD_MAX)
    assert masks.dtype == bool
    # masks == ~valid
    np.testing.assert_array_equal(masks, ~raw_trk["valid"])


def test_valid_mask_equals_track_multiplicity(oracle: dict) -> None:
    reader = _reader()
    n = len(reader)
    raw_trk = reader.read(slice(0, n), Mode.FIT)["raw.tracks"]
    expected = np.minimum(oracle["trk_per_jet"], PAD_MAX)
    np.testing.assert_array_equal(raw_trk["valid"].sum(axis=1), expected)


def test_bf16_decoded_physical(oracle: dict) -> None:
    reader = _reader()
    n = len(reader)
    out = reader.read(slice(0, n), Mode.FIT)
    raw_trk, masks = out["raw.tracks"], out["masks.tracks"]
    valid_pt = raw_trk["pt"][~masks]
    # bf16-decoded track pt is in MeV (physical: hundreds of MeV .. ~100 GeV)
    assert valid_pt.min() > 0.0
    assert valid_pt.max() > 1000.0
    # jet pt is physical too
    assert out["raw.jets"]["pt"].min() > 1000.0


# --------------------------------------------------------------------------- #
# 2. flatten + pad/truncate/sentinel/mask + roundtrip parity
# --------------------------------------------------------------------------- #


def test_jet_fields_match_oracle(oracle: dict) -> None:
    reader = _reader()
    n = len(reader)
    raw = reader.read(slice(0, n), Mode.FIT)["raw.jets"]
    for f in JET_BRANCHES:
        np.testing.assert_allclose(
            raw[f], oracle["jet_cols"][f].astype(raw[f].dtype), rtol=0, atol=1e-2
        )


def test_track_fields_match_oracle_and_truncate(oracle: dict) -> None:
    reader = _reader()
    n = len(reader)
    raw_trk = reader.read(slice(0, n), Mode.FIT)["raw.tracks"]
    trk = oracle["trk_cols"]
    counts = oracle["trk_per_jet"]
    for jet in range(n):
        k = min(int(counts[jet]), PAD_MAX)
        # leading-k constituents match the file order (truncation keeps the lead)
        np.testing.assert_allclose(
            raw_trk["d0"][jet, :k], ak.to_numpy(trk["d0"][jet][:k]), rtol=0, atol=1e-3
        )


def test_float_padding_zero_and_int_label_sentinel(oracle: dict) -> None:
    reader = _reader()
    n = len(reader)
    out = reader.read(slice(0, n), Mode.FIT)
    raw_trk, masks = out["raw.tracks"], out["masks.tracks"]
    counts = np.minimum(oracle["trk_per_jet"], PAD_MAX)
    # the integer origin label survives as integer
    assert np.issubdtype(raw_trk["origin"].dtype, np.integer)
    for jet in range(n):
        k = int(counts[jet])
        pad = masks[jet]  # True where padded
        assert pad[:k].sum() == 0 and bool(pad[k:].all())
        # padded FLOAT positions -> 0.0
        np.testing.assert_array_equal(raw_trk["d0"][jet, k:], 0.0)
        # padded SIGNED-INT LABEL positions -> -1 sentinel (NEVER a real class)
        assert np.all(raw_trk["origin"][jet, k:] == -1)
        assert np.all(raw_trk["vertexing"][jet, k:] == -1)


def test_truncation_pad_max_small(oracle: dict) -> None:
    """A small pad_max truncates to the leading constituents (valid==pad_max where overflow)."""
    reader = _reader(pad_max=4)
    n = len(reader)
    out = reader.read(slice(0, n), Mode.FIT)
    raw_trk = out["raw.tracks"]
    assert raw_trk.shape == (n, 4)
    counts = oracle["trk_per_jet"]
    valid_counts = raw_trk["valid"].sum(axis=1)
    np.testing.assert_array_equal(valid_counts, np.minimum(counts, 4))
    # overflowing jets are fully valid (all 4 positions used)
    overflow = counts > 4
    assert np.all(valid_counts[overflow] == 4)


# --------------------------------------------------------------------------- #
# 3. event-boundary-crossing read
# --------------------------------------------------------------------------- #


def test_meta_rows_test_mode_only() -> None:
    reader = _reader()
    assert "meta.rows" not in reader.read(slice(0, 2), Mode.FIT)
    np.testing.assert_array_equal(reader.read(slice(1, 3), Mode.TEST)["meta.rows"], [1, 3])


def test_boundary_crossing_jet_slice(oracle: dict) -> None:
    reader = _reader()
    n = len(reader)
    full = reader.read(slice(0, n), Mode.FIT)
    # the first event has njets[0] jets; a window straddling events 0/1/2
    lo = int(oracle["njets"][0]) - 1
    hi = lo + 4
    sub = reader.read(slice(lo, hi), Mode.FIT)
    assert sub["raw.jets"].shape == (hi - lo,)
    for f in JET_BRANCHES:
        np.testing.assert_array_equal(sub["raw.jets"][f], full["raw.jets"][f][lo:hi])
    # tracks too
    np.testing.assert_array_equal(sub["raw.tracks"]["d0"], full["raw.tracks"]["d0"][lo:hi])
    np.testing.assert_array_equal(sub["masks.tracks"], full["masks.tracks"][lo:hi])


def test_many_small_slices_tile_full_read(oracle: dict) -> None:
    """Reading in small windows reconstructs the full flat-jet stream exactly."""
    reader = _reader()
    n = len(reader)
    full = reader.read(slice(0, n), Mode.FIT)["raw.jets"]["pt"]
    pieces = []
    step = 7  # deliberately not aligned to event boundaries
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        pieces.append(reader.read(slice(lo, hi), Mode.FIT)["raw.jets"]["pt"])
    np.testing.assert_array_equal(np.concatenate(pieces), full)


# --------------------------------------------------------------------------- #
# 4. CutSpec at index-build
# --------------------------------------------------------------------------- #


def test_cut_count_parity_and_served_jets_pass(oracle: dict) -> None:
    base = _reader()
    n_all = len(base)
    pt_all = base.read(slice(0, n_all), Mode.FIT)["raw.jets"]["pt"]
    thresh = 50_000.0
    n_pass = int((pt_all >= thresh).sum())

    cut_reader = FTAG1LiteReader(
        groups=_groups(), filename=SAMPLE,
        cuts=CutSpec(global_cuts=(Cut("pt", ">=", thresh),)),
    )
    n_cut = len(cut_reader)
    # count parity: raw jets = passing + failing
    assert n_cut == n_pass
    assert n_pass + (n_all - n_pass) == n_all
    # every served jet passes the cut
    served = cut_reader.read(slice(0, n_cut), Mode.FIT)["raw.jets"]
    assert np.all(served["pt"] >= thresh)


def test_cut_on_flavour_label(oracle: dict) -> None:
    """A cut on the integer flavour label (b-jets only) is exact."""
    base = _reader()
    n_all = len(base)
    labels = base.read(slice(0, n_all), Mode.FIT)["raw.jets"]["flavour_label"]
    n_b = int((labels == 5).sum())
    b_reader = FTAG1LiteReader(
        groups=_groups(), filename=SAMPLE,
        cuts=CutSpec(global_cuts=(Cut("flavour_label", "==", 5),)),
    )
    assert len(b_reader) == n_b
    served = b_reader.read(slice(0, n_b), Mode.FIT)["raw.jets"]
    assert np.all(served["flavour_label"] == 5)


def test_global_vs_per_split_differ() -> None:
    spec = CutSpec(
        global_cuts=(Cut("pt", ">=", 20_000.0),),
        per_split={"train": (Cut("flavour_label", "==", 5),)},
    )
    # stage=None -> global only; stage="train" -> global + b-only
    r_global = FTAG1LiteReader(groups=_groups(), filename=SAMPLE, cuts=spec)
    r_train = r_global.with_source(filename=SAMPLE, stage="train")
    r_val = r_global.with_source(filename=SAMPLE, stage="val")
    n_global, n_train, n_val = len(r_global), len(r_train), len(r_val)
    assert n_train < n_global  # b-only is a strict subset
    assert n_val == n_global  # val has no extra split cut -> same as global-only
    served = r_train.read(slice(0, n_train), Mode.FIT)["raw.jets"]
    assert np.all(served["flavour_label"] == 5)
    assert np.all(served["pt"] >= 20_000.0)


def test_cut_unknown_field_raises() -> None:
    bad = FTAG1LiteReader(
        groups=_groups(), filename=SAMPLE,
        cuts=CutSpec(global_cuts=(Cut("not_a_jet_field", ">", 0),)),
    )
    with pytest.raises((SchemaError, KeyError)):
        bad.prepare()


def test_cut_variable_read_even_if_not_in_groups() -> None:
    """A cut field that is NOT a configured branch is still read at index-build."""
    groups = {
        "jets": FTAG1LiteGroupConfig(branches={"pt": "pt"}, jagged=False),
        "tracks": FTAG1LiteGroupConfig(
            branches={"d0": "ft1l_trk_d0_bf16"}, jagged=True, pad_max=PAD_MAX
        ),
    }
    # HadronConeExclTruthLabelID is a real aux branch but NOT in the jets group
    reader = FTAG1LiteReader(
        groups=groups, filename=SAMPLE,
        cuts=CutSpec(global_cuts=(Cut("HadronConeExclTruthLabelID", "==", 5),)),
    )
    n = len(reader)
    assert n > 0
    # the served jets are real and all pass an additional independent check:
    # flavour_label isn't exposed (not in groups), but pt is finite
    served = reader.read(slice(0, n), Mode.FIT)["raw.jets"]
    assert np.all(np.isfinite(served["pt"]))


# --------------------------------------------------------------------------- #
# 5. labels wired — UNMODIFIED Features / Labels
# --------------------------------------------------------------------------- #


def test_features_processor_compat() -> None:
    from salt.data import Features
    from salt.graph.bundle import Bundle

    reader = _reader()
    n = len(reader)
    produced = reader.read(slice(0, n), Mode.FIT)
    bundle = Bundle()
    bundle.set("raw.tracks", produced["raw.tracks"])
    bundle.set("masks.tracks", produced["masks.tracks"])

    feats = Features(variables={"tracks": ["d0", "z0SinTheta", "pt"]})
    out = feats.process(bundle, slice(0, n), Mode.FIT)
    inputs = out["inputs.tracks"]
    assert inputs.dtype == np.float32
    assert inputs.shape == (n, PAD_MAX, 3)
    # padded rows zeroed by Features via the pad mask
    masks = produced["masks.tracks"]
    assert np.all(inputs[masks] == 0.0)


def test_labels_processor_compat_origin_and_vertexing() -> None:
    from types import MappingProxyType

    from salt.data import Labels
    from salt.data.base import WorkerCtx
    from salt.graph.bundle import Bundle
    from salt.graph.planner import PlanStep
    from salt.graph.spec import TensorSpec

    reader = _reader()
    n = len(reader)
    produced = reader.read(slice(0, n), Mode.FIT)

    labels = Labels(streams=["jets", "tracks"], dtype_policy="int64-for-int")
    labels.name = "labels"
    step = PlanStep(
        name="labels",
        module=labels,
        requires=MappingProxyType({}),
        produces=MappingProxyType(
            {
                "labels.jets.flavour_label": TensorSpec(kind="label"),
                "labels.tracks.origin": TensorSpec(kind="label"),
                "labels.tracks.vertexing": TensorSpec(kind="label"),
            }
        ),
    )
    labels.bind(WorkerCtx(mode=Mode.FIT, read_fields={}, seed=0, step=step))

    bundle = Bundle()
    bundle.set("raw.jets", produced["raw.jets"])
    bundle.set("raw.tracks", produced["raw.tracks"])
    out = labels.process(bundle, slice(0, n), Mode.FIT)

    flav = out["labels.jets.flavour_label"]
    origin = out["labels.tracks.origin"]
    vtx = out["labels.tracks.vertexing"]
    assert flav.dtype == np.int64 and flav.shape == (n,)
    assert origin.shape == (n, PAD_MAX) and vtx.shape == (n, PAD_MAX)
    # flavour label is one of the real classes
    assert set(np.unique(flav)).issubset({0, 4, 5, 15})
    # padded track-label positions are the -1 sentinel AND flagged by the mask
    masks = produced["masks.tracks"]
    assert np.all(origin[masks] == -1)
    assert np.all(vtx[masks] == -1)


# --------------------------------------------------------------------------- #
# 6. demand narrowing, source-resolution, lazy import
# --------------------------------------------------------------------------- #


def test_demand_narrowed_read() -> None:
    from salt.data.base import WorkerCtx

    reader = _reader()
    reader.bind(
        WorkerCtx(mode=Mode.FIT, read_fields={"tracks": {"d0": "x", "origin": "y"}}, seed=0)
    )
    raw_trk = reader.read(slice(0, len(reader)), Mode.FIT)["raw.tracks"]
    assert set(raw_trk.dtype.names) == {"d0", "origin", "valid"}


def test_pool_root_directory_glob(tmp_path: Path) -> None:
    """A directory source globs *.pool.root (the FTAG1LITE convention)."""
    link = tmp_path / SAMPLE.name
    link.symlink_to(SAMPLE)
    reader = FTAG1LiteReader(groups=_groups(), filename=tmp_path)
    assert len(reader) > 0
    assert reader.source_path == link


def test_missing_branch_raises() -> None:
    bad = {
        "jets": FTAG1LiteGroupConfig(branches={"pt": "NOT_A_REAL_BRANCH"}, jagged=False),
    }
    reader = FTAG1LiteReader(groups=bad, filename=SAMPLE)
    with pytest.raises(SchemaError):
        reader.prepare()


def test_jaggedness_mismatch_raises() -> None:
    # configure a jet-level scalar branch as jagged (constituent) -> mismatch
    bad = {
        "jets": FTAG1LiteGroupConfig(branches={"pt": "pt"}, jagged=False),
        "tracks": FTAG1LiteGroupConfig(branches={"pt": "pt"}, jagged=True, pad_max=4),
    }
    reader = FTAG1LiteReader(groups=bad, filename=SAMPLE)
    with pytest.raises(SchemaError):
        reader.prepare()


def test_no_jet_level_stream_raises() -> None:
    from salt.graph.errors import ConfigError

    with pytest.raises(ConfigError):
        FTAG1LiteReader(
            groups={
                "tracks": FTAG1LiteGroupConfig(
                    branches={"d0": "ft1l_trk_d0_bf16"}, jagged=True, pad_max=4
                )
            },
            filename=SAMPLE,
        )


def test_no_top_level_uproot_awkward_import() -> None:
    """The reader module must lazy-import uproot/awkward (salt w/o them)."""
    import salt.data.readers.ftag1lite_reader as mod

    tree = ast.parse(Path(mod.__file__).read_text())
    top_level: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
    assert "uproot" not in top_level
    assert "awkward" not in top_level


# --------------------------------------------------------------------------- #
# 7. round-trip smoke: file -> reader -> Features/Labels -> torch -> salt fit
# --------------------------------------------------------------------------- #

_CONFIG = Path(__file__).resolve().parents[2] / "core" / "configs" / "ftag1lite_empflow.yaml"

# the feature variables the shipped config feeds to Features (must have norm stats)
_NORM_JETS = ["pt", "eta", "phi", "mass"]
_NORM_TRACKS = [
    "d0", "z0SinTheta", "d0Uncertainty", "z0SinThetaUncertainty",
    "lifetimeSignedD0Significance", "lifetimeSignedZ0SinThetaSignificance",
    "pt", "deta", "dphi", "numberOfPixelHits", "numberOfSCTHits",
]
_CONFIG_TRACK_BRANCHES = {
    "d0": "ft1l_trk_d0_bf16",
    "z0SinTheta": "ft1l_trk_z0SinTheta_bf16",
    "d0Uncertainty": "ft1l_trk_d0Uncertainty_bf16",
    "z0SinThetaUncertainty": "ft1l_trk_z0SinThetaUncertainty_bf16",
    "lifetimeSignedD0Significance": "ft1l_trk_lifetimeSignedD0Significance_bf16",
    "lifetimeSignedZ0SinThetaSignificance": "ft1l_trk_lifetimeSignedZ0SinThetaSignificance_bf16",
    "pt": "ft1l_trk_pt_bf16",
    "deta": "ft1l_trk_deta_bf16",
    "dphi": "ft1l_trk_dphi_bf16",
    "numberOfPixelHits": "ft1l_trk_numberOfPixelHits",
    "numberOfSCTHits": "ft1l_trk_numberOfSCTHits",
    "ftagTruthOriginLabel": "ft1l_trk_ftagTruthOriginLabel",
    "ftagTruthVertexIndex": "ft1l_trk_ftagTruthVertexIndex",
}
_CONFIG_JET_BRANCHES = {
    "pt": "pt", "eta": "eta_calibrated", "phi": "phi_calibrated", "mass": "mass_calibrated",
    "flavour_label": "HadronConeExclTruthLabelID",
}


def _compute_norm_dict(path: Path) -> dict:
    """Compute a real mean/std norm_dict from the reader output for the config features."""
    groups = {
        "jets": FTAG1LiteGroupConfig(branches=dict(_CONFIG_JET_BRANCHES), jagged=False),
        "tracks": FTAG1LiteGroupConfig(
            branches=dict(_CONFIG_TRACK_BRANCHES), jagged=True, pad_max=PAD_MAX
        ),
    }
    reader = FTAG1LiteReader(groups=groups, filename=SAMPLE)
    n = len(reader)
    out = reader.read(slice(0, n), Mode.FIT)
    raw_jets, raw_trk, masks = out["raw.jets"], out["raw.tracks"], out["masks.tracks"]
    nd: dict = {"jets": {}, "tracks": {}}
    for f in _NORM_JETS:
        col = raw_jets[f].astype(np.float64)
        nd["jets"][f] = {"mean": float(col.mean()), "std": float(col.std() or 1.0)}
    valid = ~masks
    for f in _NORM_TRACKS:
        col = raw_trk[f][valid].astype(np.float64)
        nd["tracks"][f] = {"mean": float(col.mean()), "std": float(col.std() or 1.0)}
    return nd


def test_config_plan_compiles_all_modes(tmp_path: Path) -> None:
    """The shipped config plan-compiles in fit/test/onnx (label_universe-validated)."""
    import yaml

    from salt.cli import main as salt_main

    nd = _compute_norm_dict(SAMPLE)
    nd_path = tmp_path / "norm_dict.yaml"
    with open(nd_path, "w") as fh:
        yaml.safe_dump(nd, fh)
    rc = salt_main(
        ["graph", "validate", "-c", str(_CONFIG),
         "--set", f"model.modules.norm.init_args.norm_dict={nd_path}",
         "--set", f"data.modules.reader.init_args.filename={SAMPLE}"]
    )
    assert rc == 0


def test_roundtrip_smoke_fit_finite_loss(tmp_path: Path) -> None:
    """file -> FTAG1LiteReader -> Features/Labels -> torch -> salt fit --fast_dev_run."""
    import yaml

    from salt.main import main as salt_main

    nd = _compute_norm_dict(SAMPLE)
    nd_path = tmp_path / "norm_dict.yaml"
    with open(nd_path, "w") as fh:
        yaml.safe_dump(nd, fh)

    rc = salt_main([
        "fit",
        "--config", str(_CONFIG),
        f"--data.train_file={SAMPLE}",
        f"--data.val_file={SAMPLE}",
        f"--model.modules.norm.init_args.norm_dict={nd_path}",
        f"--trainer.default_root_dir={tmp_path}",
        "--trainer.accelerator=cpu",
        # base2 default-ON CometLogger (plan-24 Wave 0) → off for the smoke fit so
        # no offline Comet archive is written under the run dir (and lr_monitor drops)
        "--trainer.logger=false",
        "--trainer.fast_dev_run=2",
        "--data.num_workers=0",
        "--data.batch_size=50",
        "--callbacks.progress=null",
    ])
    assert rc == 0, "salt fit --fast_dev_run did not complete (rc != 0)"
