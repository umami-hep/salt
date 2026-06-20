"""Regression tests for `salt.core.data.EasyjetReader` (plan 01, v2 dataloaders).

Gates the modular-Reader boundary on a brand-new file type:

1. **round-trip parity** — the reader's ``raw.jets[field]`` / ``raw.event[field]``
   match direct uproot/oracle reads elementwise; the ``valid`` mask equals the
   true per-event jet multiplicity; padded float positions are 0.0 and padded
   INTEGER LABEL positions are the -1 sentinel; field order = config order;
2. **file-boundary slice** — a global slice crossing TWO fixture files returns the
   correct global rows;
3. **processor compatibility** — ``raw.jets`` flows through the UNMODIFIED
   `Features` (-> ``inputs.jets`` float32 ``(B, T, F)``) and `Labels` (->
   ``labels.jets.HadronConeExclTruthLabelID``), and a padded label position is
   masked (folds to ``ignore_index`` territory: the pad mask is True there);
4. **lazy import** — the reader module declares no top-level ``uproot`` / ``awkward``
   import (``salt.core`` imports without them).

All fixtures are deterministic synthetic ``AnalysisMiniTree`` files written with
uproot at test time — no dependency on ``/data/atlas_samples``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from salt.core.data import EasyjetGroupConfig, EasyjetReader
from salt.core.graph.spec import Mode

uproot = pytest.importorskip("uproot")
awkward = pytest.importorskip("awkward")

from salt.tests.core.fixtures.easyjet_minitree import (  # noqa: E402
    build_fixture_arrays,
    write_minitree,
)

JET_FIELDS = ["pt", "eta", "phi", "m", "GN2v01_pb", "GN2v01_pc", "GN2v01_pu", "label"]
JET_BRANCHES = {
    "pt": "recojet_antikt4PFlow_pt_NOSYS",
    "eta": "recojet_antikt4PFlow_eta",
    "phi": "recojet_antikt4PFlow_phi",
    "m": "recojet_antikt4PFlow_m_NOSYS",
    "GN2v01_pb": "recojet_antikt4PFlow_GN2v01_pb",
    "GN2v01_pc": "recojet_antikt4PFlow_GN2v01_pc",
    "GN2v01_pu": "recojet_antikt4PFlow_GN2v01_pu",
    "label": "recojet_antikt4PFlow_HadronConeExclTruthLabelID",
}
EVENT_BRANCHES = {"eventNumber": "eventNumber", "mcChannelNumber": "mcChannelNumber"}
# the Features-consumed float input variables (label is a Labels target, not a feature)
FEATURE_VARS = ["pt", "eta", "phi", "m", "GN2v01_pb", "GN2v01_pc", "GN2v01_pu"]


def _groups(truncate: int | None = 8) -> dict:
    return {
        "jets": EasyjetGroupConfig(branches=dict(JET_BRANCHES), jagged=True, truncate=truncate),
        "event": EasyjetGroupConfig(branches=dict(EVENT_BRANCHES), jagged=False),
    }


@pytest.fixture
def single_file(tmp_path: Path) -> tuple[Path, dict]:
    """A single deterministic fixture file + its ground-truth arrays."""
    arrays = build_fixture_arrays(seed=1234)
    path = write_minitree(tmp_path / "ej_single.root", arrays)
    return path, arrays


@pytest.fixture
def two_files(tmp_path: Path) -> tuple[Path, list[dict]]:
    """A directory of TWO fixture files (different seeds) for boundary tests."""
    a = build_fixture_arrays(seed=1)
    b = build_fixture_arrays(seed=2)
    d = tmp_path / "ej_dir"
    d.mkdir()
    write_minitree(d / "file_000001.root", a)
    write_minitree(d / "file_000002.root", b)
    return d, [a, b]


# --------------------------------------------------------------------------- #
# 1. round-trip parity
# --------------------------------------------------------------------------- #


def test_len_and_prepare(single_file: tuple[Path, dict]) -> None:
    path, arrays = single_file
    reader = EasyjetReader(groups=_groups(truncate=None), filename=path)
    assert len(reader) == arrays["n_events"]
    # truncate=None auto-resolves T = max multiplicity
    assert reader._mult["jets"] == max(arrays["njets"])
    assert reader.streams == ("jets", "event")


def test_roundtrip_jet_fields_and_valid(single_file: tuple[Path, dict]) -> None:
    path, arrays = single_file
    t = 8
    reader = EasyjetReader(groups=_groups(truncate=t), filename=path)
    n = len(reader)
    out = reader.read(slice(0, n), Mode.FIT)
    raw = out["raw.jets"]
    masks = out["masks.jets"]

    # field order = config order + valid
    assert list(raw.dtype.names) == JET_FIELDS + ["valid"]
    assert raw.shape == (n, t)

    njets = np.array(arrays["njets"])
    valid = raw["valid"]
    # valid mask == true per-event jet multiplicity (clipped to T)
    assert np.array_equal(valid.sum(axis=1), np.minimum(njets, t))
    # masks = ~valid (True = padded)
    assert np.array_equal(masks, ~valid)

    for fname in FEATURE_VARS:
        branch = JET_BRANCHES[fname]
        truth = arrays[branch]  # list of per-event np arrays
        for ev in range(n):
            k = min(njets[ev], t)
            np.testing.assert_array_equal(
                raw[fname][ev, :k], truth[ev][:k].astype(np.float32)
            )
            # padded FLOAT positions are 0.0
            np.testing.assert_array_equal(raw[fname][ev, k:], 0.0)


def test_roundtrip_integer_label_sentinel(single_file: tuple[Path, dict]) -> None:
    path, arrays = single_file
    t = 8
    reader = EasyjetReader(groups=_groups(truncate=t), filename=path)
    n = len(reader)
    raw = reader.read(slice(0, n), Mode.FIT)["raw.jets"]
    njets = np.array(arrays["njets"])
    truth = arrays["recojet_antikt4PFlow_HadronConeExclTruthLabelID"]
    # integer label dtype survives as int
    assert np.issubdtype(raw["label"].dtype, np.integer)
    for ev in range(n):
        k = min(njets[ev], t)
        np.testing.assert_array_equal(raw["label"][ev, :k], truth[ev][:k])
        # padded INTEGER LABEL positions == -1 sentinel (NOT 0)
        assert np.all(raw["label"][ev, k:] == -1)


def test_roundtrip_scalar_event_fields(single_file: tuple[Path, dict]) -> None:
    path, arrays = single_file
    reader = EasyjetReader(groups=_groups(), filename=path)
    n = len(reader)
    raw = reader.read(slice(0, n), Mode.FIT)["raw.event"]
    assert raw.shape == (n,)
    assert "masks.event" not in reader.read(slice(0, n), Mode.FIT)
    np.testing.assert_array_equal(raw["eventNumber"], arrays["eventNumber"])
    np.testing.assert_array_equal(raw["mcChannelNumber"], arrays["mcChannelNumber"])


def test_meta_rows_test_mode_only(single_file: tuple[Path, dict]) -> None:
    path, _ = single_file
    reader = EasyjetReader(groups=_groups(), filename=path)
    assert "meta.rows" not in reader.read(slice(0, 2), Mode.FIT)
    test_out = reader.read(slice(1, 3), Mode.TEST)
    np.testing.assert_array_equal(test_out["meta.rows"], [1, 3])


# --------------------------------------------------------------------------- #
# 2. file-boundary crossing slice
# --------------------------------------------------------------------------- #


def test_multifile_len_and_offsets(two_files: tuple[Path, list[dict]]) -> None:
    d, [a, b] = two_files
    reader = EasyjetReader(groups=_groups(), filename=d)
    assert len(reader) == a["n_events"] + b["n_events"]
    # deterministic, sorted file table with cumulative offsets
    assert [e.start for e in reader._table] == [0, a["n_events"]]


def test_slice_crossing_file_boundary(two_files: tuple[Path, list[dict]]) -> None:
    d, [a, b] = two_files
    t = 8
    reader = EasyjetReader(groups=_groups(truncate=t), filename=d)
    na = a["n_events"]
    # slice spanning the last event of file A and the first of file B
    lo, hi = na - 1, na + 2
    raw = reader.read(slice(lo, hi), Mode.FIT)["raw.jets"]
    assert raw.shape == (hi - lo, t)

    # global row -> (file, local row) oracle
    files = [a, b]
    counts = [a["n_events"], b["n_events"]]
    starts = [0, counts[0]]
    for gi, grow in enumerate(range(lo, hi)):
        fidx = 1 if grow >= starts[1] else 0
        local = grow - starts[fidx]
        truth_pt = files[fidx]["recojet_antikt4PFlow_pt_NOSYS"][local].astype(np.float32)
        k = min(len(truth_pt), t)
        np.testing.assert_array_equal(raw["pt"][gi, :k], truth_pt[:k])
        # event-scalar oracle too
        ev_raw = reader.read(slice(lo, hi), Mode.FIT)["raw.event"]
        assert ev_raw["eventNumber"][gi] == files[fidx]["eventNumber"][local]
    del counts


# --------------------------------------------------------------------------- #
# 3. processor compatibility — UNMODIFIED Features / Labels
# --------------------------------------------------------------------------- #


def test_features_processor_compat(single_file: tuple[Path, dict]) -> None:
    from salt.core.data import Features
    from salt.core.graph.bundle import Bundle

    path, arrays = single_file
    t = 8
    reader = EasyjetReader(groups=_groups(truncate=t), filename=path)
    n = len(reader)
    produced = reader.read(slice(0, n), Mode.FIT)
    bundle = Bundle()
    bundle.set("raw.jets", produced["raw.jets"])
    bundle.set("masks.jets", produced["masks.jets"])

    features = Features(variables={"jets": list(FEATURE_VARS)})
    out = features.process(bundle, slice(0, n), Mode.FIT)
    inputs = out["inputs.jets"]
    assert inputs.dtype == np.float32
    assert inputs.shape == (n, t, len(FEATURE_VARS))
    # padded rows zeroed by Features via the pad mask
    njets = np.array(arrays["njets"])
    for ev in range(n):
        k = min(njets[ev], t)
        np.testing.assert_array_equal(inputs[ev, k:], 0.0)
        # real rows match raw, column order = config order
        for fi, fname in enumerate(FEATURE_VARS):
            np.testing.assert_array_equal(
                inputs[ev, :k, fi], produced["raw.jets"][fname][ev, :k]
            )


def test_labels_processor_compat_and_padding_masked(single_file: tuple[Path, dict]) -> None:
    from types import MappingProxyType

    from salt.core.data import Labels
    from salt.core.data.base import WorkerCtx
    from salt.core.graph.bundle import Bundle
    from salt.core.graph.planner import PlanStep
    from salt.core.graph.spec import TensorSpec

    path, arrays = single_file
    t = 8
    reader = EasyjetReader(groups=_groups(truncate=t), filename=path)
    n = len(reader)
    produced = reader.read(slice(0, n), Mode.FIT)

    # narrow the Labels wildcard to the one demanded label (the planner does this
    # for real; here we build the narrowed PlanStep directly — TestLabelsUnit
    # pattern in test_data_pipeline.py)
    labels = Labels(streams=["jets"], dtype_policy="int64-for-int")
    labels.name = "labels"
    step = PlanStep(
        name="labels",
        module=labels,
        requires=MappingProxyType({}),
        produces=MappingProxyType({"labels.jets.label": TensorSpec(kind="label")}),
    )
    labels.bind(WorkerCtx(mode=Mode.FIT, read_fields={}, seed=0, step=step))

    bundle = Bundle()
    bundle.set("raw.jets", produced["raw.jets"])
    out = labels.process(bundle, slice(0, n), Mode.FIT)
    lab = out["labels.jets.label"]
    assert lab.dtype == np.int64
    njets = np.array(arrays["njets"])
    masks = produced["masks.jets"]
    truth = arrays["recojet_antikt4PFlow_HadronConeExclTruthLabelID"]
    for ev in range(n):
        k = min(njets[ev], t)
        np.testing.assert_array_equal(lab[ev, :k], truth[ev][:k])
        # padded label positions: -1 sentinel AND flagged by the pad mask (so the
        # sequence task folds them to ignore_index, never a real class)
        assert np.all(lab[ev, k:] == -1)
        assert np.all(masks[ev, k:])
        assert not np.any(masks[ev, :k])


# --------------------------------------------------------------------------- #
# 4. demand narrowing + lazy import + schema
# --------------------------------------------------------------------------- #


def test_demand_narrowed_read(single_file: tuple[Path, dict]) -> None:
    from salt.core.data.base import WorkerCtx

    path, _ = single_file
    reader = EasyjetReader(groups=_groups(truncate=8), filename=path)
    reader.bind(WorkerCtx(mode=Mode.FIT, read_fields={"jets": {"pt": "x", "label": "y"}}, seed=0))
    raw = reader.read(slice(0, len(reader)), Mode.FIT)["raw.jets"]
    # only the demanded fields (+ valid) are present
    assert set(raw.dtype.names) == {"pt", "label", "valid"}


def test_schema_built_and_label_universe(single_file: tuple[Path, dict]) -> None:
    path, _ = single_file
    reader = EasyjetReader(groups=_groups(), filename=path)
    reader.prepare()
    assert reader.schema is not None
    js = reader.schema_group("jets")
    assert js is not None and js.has_valid
    # float kinematics stay float, the flavour label stays integer (dtype kinds
    # survive the read; exact width depends on the writer — RNTuple fixtures may
    # widen f4->f8, real easyjet TTrees store f4)
    assert np.issubdtype(np.dtype(js.fields["pt"]), np.floating)
    assert np.issubdtype(np.dtype(js.fields["label"]), np.integer)
    universe = reader.label_universe()
    assert "labels.jets.label" in universe
    assert "labels.event.eventNumber" in universe
    assert "labels.jets.valid" not in universe  # 'valid' is not a label


def test_missing_branch_raises(single_file: tuple[Path, dict]) -> None:
    from salt.core.graph.errors import SchemaError

    path, _ = single_file
    bad = {
        "jets": EasyjetGroupConfig(
            branches={"pt": "recojet_antikt4PFlow_NOT_A_BRANCH"}, jagged=True, truncate=4
        )
    }
    reader = EasyjetReader(groups=bad, filename=path)
    with pytest.raises(SchemaError):
        reader.prepare()


def test_no_top_level_uproot_awkward_import() -> None:
    """The reader module must lazy-import uproot/awkward (salt.core w/o them)."""
    src = Path(EasyjetReader.__module__.replace(".", "/") + ".py")
    # resolve via the imported module's file
    import salt.core.data.easyjet_reader as mod

    tree = ast.parse(Path(mod.__file__).read_text())
    top_level_imports: set[str] = set()
    for node in tree.body:  # only module-level statements
        if isinstance(node, ast.Import):
            top_level_imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".")[0])
    assert "uproot" not in top_level_imports
    assert "awkward" not in top_level_imports
    del src


# --------------------------------------------------------------------------- #
# 5. pyproject.toml optional-dependency structure
# --------------------------------------------------------------------------- #


def test_uproot_awkward_are_extras_not_base_deps() -> None:
    """uproot/awkward must NOT appear in base [project.dependencies].

    They must appear in the `root` and/or `easyjet` optional-dependency groups,
    confirming `pip install salt` stays uproot-free.
    """
    import importlib.util
    import tomllib
    from pathlib import Path

    # Locate pyproject.toml relative to the installed package location
    spec = importlib.util.find_spec("salt")
    assert spec is not None and spec.origin is not None
    pkg_root = Path(spec.origin).parent  # salt/
    proj_root = pkg_root.parent  # worktree root
    pyproject = proj_root / "pyproject.toml"
    assert pyproject.exists(), f"pyproject.toml not found at {pyproject}"

    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)

    base_deps: list[str] = data["project"].get("dependencies", [])
    base_names = {d.split("[")[0].split(">=")[0].split("==")[0].split("!=")[0].strip().lower()
                  for d in base_deps}
    assert "uproot" not in base_names, "uproot must not be in base [project.dependencies]"
    assert "awkward" not in base_names, "awkward must not be in base [project.dependencies]"

    optional: dict = data["project"].get("optional-dependencies", {})
    # The `easyjet` extra must exist and pull in uproot/awkward (directly or via root)
    assert "easyjet" in optional, "Missing [project.optional-dependencies.easyjet]"
    easyjet_deps = optional["easyjet"]
    # Either easyjet lists them directly, or it includes the `root` meta-extra
    root_deps = optional.get("root", [])
    all_easyjet = easyjet_deps + root_deps
    all_easyjet_lower = {d.lower() for d in all_easyjet}
    assert any("uproot" in d for d in all_easyjet_lower), (
        "uproot not found in easyjet or root optional-dependency group"
    )
    assert any("awkward" in d for d in all_easyjet_lower), (
        "awkward not found in easyjet or root optional-dependency group"
    )


def test_missing_root_deps_raises_helpful_error(single_file: tuple[Path, dict]) -> None:
    """EasyjetReader with blocked uproot must raise a helpful ImportError.

    The error message must mention `salt[easyjet]` so the user knows exactly
    what to install — NOT a bare ModuleNotFoundError from deep inside uproot.
    """
    import importlib
    import sys

    path, _ = single_file

    # Save and remove real modules from sys.modules to simulate absence
    saved: dict[str, object] = {}
    for key in list(sys.modules):
        if key == "uproot" or key.startswith("uproot."):
            saved[key] = sys.modules.pop(key)

    # Install a meta_path blocker for uproot (same pattern as the existing
    # test_no_top_level_uproot_awkward_import style used in the codebase)
    class _BlockUproot:
        @staticmethod
        def find_spec(name, path, target=None):  # noqa: ANN001, ANN202
            if name == "uproot" or name.startswith("uproot."):
                raise ModuleNotFoundError(f"blocked: {name}")
            return None

    blocker = _BlockUproot()
    sys.meta_path.insert(0, blocker)
    # Also ensure the easyjet_reader module re-runs its guard by busting the
    # cached imports inside the module (clear uproot from its namespace if loaded)
    import salt.core.data.easyjet_reader as _mod

    _orig_uproot = _mod.__dict__.pop("uproot", None)

    try:
        reader = EasyjetReader(
            groups={"jets": EasyjetGroupConfig(branches={"pt": "recojet_antikt4PFlow_pt_NOSYS"})},
            filename=path,
        )
        with pytest.raises(ImportError) as exc_info:
            reader.prepare()
        msg = str(exc_info.value)
        assert "salt[easyjet]" in msg, (
            f"Error message must mention 'salt[easyjet]', got: {msg!r}"
        )
        # Must NOT be a bare ModuleNotFoundError without context
        assert "pip install" in msg, (
            f"Error message must include install instructions, got: {msg!r}"
        )
    finally:
        sys.meta_path.remove(blocker)
        # Restore uproot so subsequent tests work
        for key, mod in saved.items():
            sys.modules[key] = mod  # type: ignore[assignment]
        if _orig_uproot is not None:
            _mod.__dict__["uproot"] = _orig_uproot
        # Force uproot reimport into the module cache
        importlib.import_module("uproot")


# --------------------------------------------------------------------------- #
# 5. M8 reader-owned staging — restage() round-trips a MULTI-FILE reader
#    (the new capability the v1 train_file/val_file-only staging lacked) +
#    the GraphDataModule thin trigger restages the per-stage reader so VDS
#    precreation sees the staged paths.
# --------------------------------------------------------------------------- #


def _read_all(reader: EasyjetReader, mode: Mode = Mode.FIT) -> dict:
    """Read every served row of a (bound or standalone) reader as one batch."""
    n = len(reader)
    return reader.read(slice(0, n), mode)


def test_sources_lists_every_member_of_a_multifile_reader(
    two_files: tuple[Path, list[dict]],
) -> None:
    """sources() returns the FULL resolved member list — not just one file.

    This is the multi-file declaration the old datamodule (single train_file) could
    not express; restage() builds on it to stage EVERY member.
    """
    d, _ = two_files
    reader = EasyjetReader(groups=_groups(), filename=d)
    srcs = reader.sources()
    assert [p.name for p in srcs] == ["file_000001.root", "file_000002.root"]
    assert all(p.parent == d for p in srcs)
    # an unbound reader has nothing to stage (no error)
    assert EasyjetReader(groups=_groups()).sources() == []


def test_restage_roundtrips_a_multifile_reader(
    two_files: tuple[Path, list[dict]], tmp_path: Path
) -> None:
    """restage() copies ALL members to root and reads byte-identical data.

    The core M8 wave-3 capability: a MULTI-file reader (2-file easyjet directory)
    restages every member under a fresh root, the clone's sources() point there, the
    ORIGINALS survive, and a full read of the staged clone equals the original read
    elementwise (jets + valid mask + scalar event fields). The old train_file/val_file
    staging would have copied at most one file and silently dropped the rest.
    """
    d, [a, b] = two_files
    t = 8
    orig = EasyjetReader(groups=_groups(truncate=t), filename=d)
    orig_out = _read_all(orig)

    root = tmp_path / "stage_root"
    staged = orig.restage(root)

    # the clone reads copies UNDER the staging root (a per-reader subdir under root)
    staged_srcs = staged.sources()
    assert {p.name for p in staged_srcs} == {"file_000001.root", "file_000002.root"}
    assert all(root in p.parents for p in staged_srcs)
    # both members were physically copied; originals survive
    assert {p.name for p in root.rglob("*.root")} == {"file_000001.root", "file_000002.root"}
    assert (d / "file_000001.root").is_file() and (d / "file_000002.root").is_file()

    # byte-identical READ through the staged reader (the data didn't change, only
    # WHERE the bytes live) — full epoch, jets + valid + event scalars
    assert len(staged) == a["n_events"] + b["n_events"] == len(orig)
    staged_out = _read_all(staged)
    np.testing.assert_array_equal(staged_out["raw.jets"]["pt"], orig_out["raw.jets"]["pt"])
    np.testing.assert_array_equal(staged_out["raw.jets"]["valid"], orig_out["raw.jets"]["valid"])
    np.testing.assert_array_equal(staged_out["masks.jets"], orig_out["masks.jets"])
    np.testing.assert_array_equal(
        staged_out["raw.event"]["eventNumber"], orig_out["raw.event"]["eventNumber"]
    )


def test_restage_is_filelock_coordinated_and_idempotent(
    two_files: tuple[Path, list[dict]], tmp_path: Path
) -> None:
    """A second restage to the same root reuses the copies (the .done markers).

    Proves the reuse of the vds.py FileLock + .done-marker machinery: restaging twice
    leaves one copy per member (no duplication, no error) — the stampede-safe path a
    DDP run relies on.
    """
    d, _ = two_files
    reader = EasyjetReader(groups=_groups(), filename=d)
    root = tmp_path / "stage_root"
    reader.restage(root)
    reader.restage(root)  # second pass: copies present -> fast path
    roots = sorted(p.name for p in root.rglob("*.root"))
    assert roots == ["file_000001.root", "file_000002.root"]
    # the FileLock + completion markers (vds.stage_file machinery) are present
    assert sorted(p.name for p in root.rglob("*.done")) == [
        "file_000001.root.done",
        "file_000002.root.done",
    ]


def test_datamodule_trigger_restages_per_stage_reader_for_vds_precreation(
    two_files: tuple[Path, list[dict]], tmp_path: Path
) -> None:
    """The GraphDataModule thin trigger restages so VDS precreation sees staged paths.

    Drives the collapsed datamodule path: with move_files_temp set, _resolve_stage_root
    arms _stage_root, and _stage(reader) restages the per-stage reader. _precreate_vds_
    rank0 calls .prepare() on exactly this restaged reader, so it resolves files UNDER
    the staging root — the reader-owned analogue of the v1 prepare_data/setup repoint.
    With move_files_temp=None the per-stage reader is returned UNCHANGED (byte-identical
    read path).
    """
    from salt.core.data.datamodule import GraphDataModule

    d, _ = two_files
    proto = EasyjetReader(groups=_groups(), filename=d)
    root = tmp_path / "stage_root"

    dm = GraphDataModule(
        modules={"reader": proto},
        train_file=str(d),
        val_file=str(d),
        move_files_temp=str(root),
    )
    # default-off twin: with no stage root, _stage() is identity (read path unchanged)
    dm._stage_root = None
    same = dm._stage(proto.with_source(filename=str(d), stage="train"))
    assert all(p.parent == d for p in same.sources())

    # armed: setup('fit') sets _stage_root; the per-stage reader restages to root
    dm._stage_root = dm._resolve_stage_root("fit")
    assert dm._stage_root == root
    per_stage = dm.reader.with_source(filename=str(d), stage="train")
    staged = dm._stage(per_stage)
    assert all(root in p.parents for p in staged.sources())
    # this is the reader _precreate_vds_rank0 prepares — it now resolves under root
    staged.prepare()
    assert all(root in e.path.parents for e in staged._table)

    # teardown removes the whole staged tree
    dm.teardown("fit")
    assert not root.exists()
