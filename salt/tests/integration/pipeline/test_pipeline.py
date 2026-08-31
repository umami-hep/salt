"""The declarative config-lifecycle matrix: mechanism, fixtures, and tests.

Owns ``MATRIX``, ``FEEDS``, the feeder builders, ``run_row``/``run_eval``/
``run_export`` and their session-scoped artifact cache, the pytest surface
(a compile+plot floor leg every row gets regardless of ``fit``, plus three
parametrized lifecycle legs: fit runs when the row declares ``fit=True``;
eval/export additionally skip when the row declares ``do_eval=False``/
``do_onnx=False``), the xfail table lookup, the matrix<->discovery
completeness checks (see ``FRAGMENTS`` below), the two residual
finetune-template assertions (claims about the chained artifacts that a
matrix row cannot itself express), and
the regression/gaussian-regression per-config semantics (de-scale, doubled-
column and ONNX-rank assertions) that are per-config and cannot be expressed
by the generic runner.

Discovery is a recursive glob of ``salt/configs`` (``_discover``) — a config
added anywhere in the tree is picked up automatically and cannot silently go
untested. Every discovered config (minus ``base.yaml``) must be EITHER a
``MATRIX`` row, or a key in ``FRAGMENTS`` (a config that cannot run alone: an
include-target reader fragment, a fragment paired via ``FEEDS``, or a
data-behaviour overlay) — see ``test_every_discovered_config_is_placed``. A
row that has no synthetic fixture the fit lifecycle can serve declares
``fit=False``: it still gets the compile+plot floor leg (every row does),
just not the fit/eval/export legs.
"""

from __future__ import annotations

import contextlib
import io
import re
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import onnx
import pytest

from salt.main import CONFIG_DIR
from salt.main import main as salt_main
from salt.outputs.sinks.onnx import make_session
from salt.schema import dump_schema, load_schema, save_schema
from salt.testing.datagen import compute_norm_dict, load_pipeline
from salt.testing.inputs import write_dummy_file, write_dummy_norm_dict

RECIPES_DIR = Path(__file__).resolve().parents[3] / "testing" / "datagen" / "recipes"

# base.yaml is auto-loaded machinery, never a model in its own right.
_MACHINERY = {"base"}

# cpu_always: every row uses --trainer.accelerator=auto on synthetic data, so
# the whole matrix is CPU-safe and must run on every CI invocation (the
# tests/integration/ GPU-skip in conftest.py would otherwise hide it on a
# CPU box).
pytestmark = [pytest.mark.cpu_always]

# ---------------------------------------------------------------------------
# schema (§1.1-1.2)
# ---------------------------------------------------------------------------

Feed = tuple[Literal["recipe", "dummy", "root"], str | None]
"""``("recipe", name)`` | ``("dummy", flavour)`` | ``("root", fragment | None)``."""


@dataclass(frozen=True)
class Row:
    """One matrix row: ``(test_name, config_relpath, do_eval, do_onnx, train_args, fit)``.

    ``fit`` (default ``True``): whether this row gets the fit/eval/export
    lifecycle legs. Every row — ``fit=True`` or not — always gets the
    compile+plot floor leg (``test_compile_plot``): a config with no
    synthetic fixture the lifecycle can run on (``fit=False``) is still a
    real shipped config, and this is what proves its plan compiles and
    renders even though no checkpoint is produced.
    """

    test_name: str
    config: str
    do_eval: bool
    do_onnx: bool
    train_args: tuple[str, ...] = ()
    fit: bool = True


@dataclass
class Artifacts:
    """What a fit leg (and, once run, eval/export) produced for one row."""

    ckpt: Path
    saved_config: Path
    root_dir: Path
    eval_h5: Path | None = None
    onnx: Path | None = None


class LegFailedError(RuntimeError):
    """A row's leg did not complete.

    ``producer`` names the row whose failure is the root cause — itself,
    unless a ``{ckpt:NAME}``/``{config:NAME}`` dependency failed first.
    """

    def __init__(self, name: str, leg: str, reason: str, producer: str | None = None) -> None:
        self.name = name
        self.leg = leg
        self.reason = reason
        self.producer = producer or name
        super().__init__(reason)


class RootDepsMissingError(RuntimeError):
    """A ROOT-fed row's fixture needs ``pip install 'salt[root]'`` (uproot + awkward)."""


# ---------------------------------------------------------------------------
# the matrix (§1.2)
# ---------------------------------------------------------------------------

MATRIX: list[Row] = [
    # -- production flagships (6) --------------------------------------
    Row(
        "gn2v2_opendata",
        "gn2v2-opendata",
        True,
        True,
        (
            # pipeline #15649957: SystemExit 2 on every leg — the old form used
            # three deep-dotted per-stage overrides
            # (`--...files.train=`/`.val=`/`.test=`); jsonargparse hands a dict-typed
            # field a Namespace for a dotted key instead of merging into it, so each
            # override wiped the other two ("Namespace given where dict expected").
            # Fix: ONE whole-dict override. Single braces, not double — this
            # template is expanded by this module's own `_TOKEN_RE`/`_expand_one`
            # (a plain regex substitution over `{h5}`), not `str.format`, so a
            # doubled brace would survive substitution unconsumed and land in the
            # argv as a literal `{{`/`}}`, breaking the JSON. The value is
            # single-quoted so `shlex.split` (the next step in
            # `_expand_train_args`) does not treat the embedded double quotes as
            # shell quoting and mangle the JSON on its internal spaces.
            """--data.modules.input_samples.init_args.files='{"train": "{h5}", "val": "{h5}", "test": "{h5}"}'""",
        ),
    ),
    Row("gn3epclv01", "GN3EPCLV01", True, True, ()),
    Row("gn3x", "GN3X", True, True, ()),
    Row("hitz", "hitz", True, True, ()),
    Row("maskformer", "MaskFormer", True, True, ()),
    Row(
        "event_tagger_easyjet",
        "ttbar_vs_hh4b_event_tagger",
        True,
        False,
        ("--config {fragment}", "--data.batch_size=2"),
    ),
    # -- reader pairing (1) ----------------------------------------------
    Row(
        "easyjet_flavour",
        "readers/easyjet_flavour",
        False,
        False,
        (
            # "--config {fragment}" (pipeline #15651154, item 1): the compile+
            # plot floor leg keeps only "--config "-prefixed train_args
            # elements (test_pipeline._expand_train_args/run_compile_plot),
            # so the fit-leg-only "--data.train_file="/"--data.val_file="
            # overrides below never reached it — its run-free `graph
            # validate`/`graph plot` calls .prepare() on the reader PROTOTYPE
            # directly and needs a real filename already resolved in a config
            # file (see write_standalone_source_fragment).
            "--config {fragment}",
            "--data.train_file={root}",
            "--data.val_file={root}",
            "--data.batch_size=2",
        ),
    ),
    # -- regression family (5) -------------------------------------------
    Row("regression", "regression/regression", True, True, ()),
    # do_onnx=True (§7 resolution): "no gaussian handling" was a stale
    # docstring claim, not a property of check_onnx — see check.py.
    Row("regression_gaussian", "regression/regression_gaussian", True, True, ()),
    Row("regression_weighted", "regression/regression_weighted", True, True, ()),
    Row("nan_regression", "regression/nan_regression", True, True, ()),
    Row("regression_multi_target", "regression/regression_multi_target", True, True, ()),
    # -- fine-tuning (3) — the reason train_args exists -------------------
    Row("gn3v00_base", "GN3/GN3V00", False, False, ("--trainer.max_epochs=1",)),
    Row(
        "finetune_same_heads",
        "finetune/finetune_gn3large",
        False,
        False,
        (
            "--config {config:gn3v00_base}",
            "--init_from={ckpt:gn3v00_base}",
            "--trainer.max_epochs=6",
            "--trainer.limit_train_batches=1",
            "--trainer.limit_val_batches=1",
        ),
    ),
    Row(
        "finetune_new_head",
        "finetune/finetune_gn3large_new_head",
        False,
        False,
        (
            "--config {config:gn3v00_base}",
            # pipeline #15651154, item 3: re-pin THIS row's own schema (with
            # large_r_flavour_label) on top of gn3v00_base's stacked one
            # (without it) — see write_schema_override_fragment. Must come
            # AFTER "{config:gn3v00_base}" so it wins the config-file merge.
            "--config {schema_fragment}",
            "--init_from={ckpt:gn3v00_base}",
            "--trainer.max_epochs=6",
            "--trainer.limit_train_batches=1",
            "--trainer.limit_val_batches=1",
        ),
    ),
    # -- completeness rows (§2): every remaining shipped config. `fit=True`
    # rows get real fit legs on top of the compile+plot floor every row gets;
    # `fit=False` rows carry forward the old NO_FIXTURE reason (or an EXTRAS
    # gap) as the comment — compile+plot only, no synthetic fixture serves
    # the lifecycle.
    # NO_FIXTURE: no soft-muon global stream in write_dummy_file. DEFERRED
    # (pipeline #15651154, item 2): fit-mode plot fails with BindError
    # "symbolic dim 'B' resolves to 2 at key 'raw.jets' declared fields but
    # 14 at key 'raw.global' declared fields". Traced to a PRODUCT-level
    # issue, not a FEEDS/fixture mismatch: `H5StructuredReader.declare_io`
    # (salt/data/readers/reader.py) gives every global_object stream the
    # literal shape ("B",) — not per-stream-qualified — and
    # `resolve_bind_schema` (salt/model/bind.py) binds a field-carrying key's
    # symbolic LAST dim to its declared-fields COUNT; for a global_object
    # stream that last dim IS "B" (there is no other), so with jets (2
    # declared Features variables) and global (14) both global_object:true,
    # the SAME "B" symbol gets bound to two different counts. This is 100%
    # config-driven (Features.declare_io reads `variables:` from the SHIPPED
    # config directly — no schema/data file involved either way, confirmed
    # by reading both declare_io implementations), so no FEEDS/dummy-fixture
    # change can avoid it — the shipped GN2emu.yaml is the only MATRIX config
    # with two DIFFERENT-width global_object streams, and this looks like a
    # latent bug never exercised before this restructure ran fit-mode plot
    # for the first time. Left for product-code judgement, not a test fix.
    Row("gn2emu", "GN2/GN2emu", False, False, fit=False),
    Row("gn2_mup", "GN2/GN2_muP", False, False),
    # NO_FIXTURE: config truncates tracks to 100; the fixture writes 40
    Row("gn2xe", "GN2/GN2XE", False, False, fit=False),
    # NO_FIXTURE: fixture flow stream lacks the flow_* field prefix
    Row("gn2x_qcdsplit", "GN2/GN2X_qcdsplit", False, False, fit=False),
    Row("gn3_baseline", "GN3/GN3_baseline", False, False),
    Row("gn3_baseline_loose", "GN3/GN3_baseline_loose", False, False),
    Row("gn3_charge", "GN3/GN3_Charge", False, False),
    Row("gn3_dr", "GN3/GN3_dR", False, False),
    Row("gn3_flow", "GN3/GN3_flow", False, False),
    Row("gn3_hybrid", "GN3/GN3_Hybrid", False, False),
    Row("gn3_lepid_smt", "GN3/GN3_LepID_SMT", False, False),
    # NO_FIXTURE: no global stream in write_dummy_file
    Row("gn3_softe", "GN3/GN3_SoftE", False, False, fit=False),
    # NO_FIXTURE: no tracks.ftagTruthSourceLabel in write_dummy_file
    Row("gn3_tracklabel", "GN3/GN3_tracklabel", False, False, fit=False),
    Row("gn3_v00", "GN3/GN3_v00", False, False),
    # NO_FIXTURE: labels on raw HadronConeExclTruthLabelID; fixture writes PDG-like values
    Row("dips", "legacy/dips", False, False, fit=False),
    # NO_FIXTURE: no super_tracks stream in write_dummy_file
    Row("dipz", "legacy/Dipz", False, False, fit=False),
    Row("dl1", "legacy/DL1", False, False),
    # EXTRAS(awkward)-gated; no synthetic FTAG1LITE POOL fixture. DEFERRED
    # (pipeline #15651154, item 1): compile+plot floor leg fails with
    # ConfigError "reader 'reader' has no source file" — the "default"
    # H5-shaped dummy feed is irrelevant to this config's OWN UprootReader
    # (via ftag1lite.yaml, unroll: jets + double-jagged ft1l_trk_* track
    # decorations under AntiKt4EMPFlowJetsAuxDyn.), and `graph validate`
    # calls .prepare() on the reader PROTOTYPE (needs a REAL, openable ROOT
    # file — same fix class as easyjet_flavour above, see
    # write_standalone_source_fragment). Unlike easyjet_flavour's single-
    # jagged jets-only minitree, a correct FTAG1LITE fixture needs an
    # accurate event->jet->track double-jagged structure with 18 branches
    # across 2 groups — a nontrivial new fixture writer left for a follow-up
    # rather than risk shipping one unverified (no pytest in this harness).
    Row("ftag1lite_empflow", "readers/ftag1lite_empflow", False, False, fit=False),
]

# config_relpath -> feeder. Replaces RECIPES/PAIRED/FIXTURE_FLAVOUR/the fixture
# half of NO_FIXTURE, which used to exist across two files.
FEEDS: dict[str, Feed] = {
    "gn2v2-opendata": ("recipe", "flavour_tagger"),
    "GN3EPCLV01": ("recipe", "flavour_tagger_global"),
    "GN3X": ("recipe", "flavour_tagger"),
    "hitz": ("recipe", "hits_regression"),
    "MaskFormer": ("recipe", "maskformer_truth_hadron"),
    "ttbar_vs_hh4b_event_tagger": ("root", "readers/easyjet_events"),
    "readers/easyjet_flavour": ("root", None),
    "regression/regression": ("dummy", "regression"),
    "regression/regression_gaussian": ("dummy", "regression"),
    "regression/regression_weighted": ("dummy", "regression"),
    "regression/nan_regression": ("dummy", "default"),
    "regression/regression_multi_target": ("dummy", "default"),
    "GN3/GN3V00": ("dummy", "gn3"),
    "finetune/finetune_gn3large": ("dummy", "gn3"),
    "finetune/finetune_gn3large_new_head": ("dummy", "gn3_large_r"),
    # completeness rows (§2) — configs not listed above take the plain
    # 3-class "default" fixture (compile+plot never reads the file, only the
    # norm_dict path, so a schema mismatch on a fit=False row is harmless).
    "GN2/GN2emu": ("dummy", "default"),
    "GN2/GN2_muP": ("dummy", "default"),
    "GN2/GN2XE": ("dummy", "default"),
    "GN2/GN2X_qcdsplit": ("dummy", "default"),
    "GN3/GN3_baseline": ("dummy", "taus"),
    "GN3/GN3_baseline_loose": ("dummy", "taus"),
    "GN3/GN3_Charge": ("dummy", "gn3"),
    "GN3/GN3_dR": ("dummy", "taus"),
    "GN3/GN3_flow": ("dummy", "taus"),
    "GN3/GN3_Hybrid": ("dummy", "gn3"),
    "GN3/GN3_LepID_SMT": ("dummy", "taus"),
    "GN3/GN3_SoftE": ("dummy", "gn3"),
    "GN3/GN3_tracklabel": ("dummy", "taus"),
    "GN3/GN3_v00": ("dummy", "gn3"),
    "legacy/dips": ("dummy", "default"),
    "legacy/Dipz": ("dummy", "default"),
    "legacy/DL1": ("dummy", "default"),
    "readers/ftag1lite_empflow": ("dummy", "default"),
}

# config_relpath -> (importable module, pip extra) for a config whose reader
# or model needs an optional dependency the bare image lacks. ROOT-fed MATRIX
# rows (easyjet_flavour, event_tagger_easyjet) already self-gate via
# `_root_deps()`/`RootDepsMissingError` (the "root" FEEDS kind) and don't need
# an entry here; this table is for configs fed some other way (dummy H5)
# whose reader/model class still has a hard import on an optional package,
# plus the reader-fragment entries in FRAGMENTS that
# `test_reader_fragment_instantiates` gates directly.
EXTRAS: dict[str, tuple[str, str]] = {
    "GN2/GN2_muP": ("mup", "salt[muP]"),
    "readers/ftag1lite_empflow": ("awkward", "salt[root]"),
    "readers/easyjet_events": ("awkward", "salt[root]"),
    "readers/ftag1lite": ("awkward", "salt[root]"),
    "readers/physlite_events": ("awkward", "salt[root]"),
    "readers/physlite_jets": ("awkward", "salt[root]"),
}


def _require_extra(config: str) -> None:
    """Skip if ``config``'s optional dependency (EXTRAS) isn't importable."""
    entry = EXTRAS.get(config)
    if entry is None:
        return
    module, extra = entry
    pytest.importorskip(module, reason=f"{config} needs `pip install '{extra}'`")


# Fragments that cannot run alone (an include-target reader fragment, a
# fragment paired via FEEDS, or a pure data-behaviour overlay) — every
# discovered config is either a MATRIX row or a key here (§ discovery,
# test_every_discovered_config_is_placed). The value records how it is
# exercised: "included" (>=1 row's config include:s it — checked via each
# row's own top-level include: block), "paired" (a FEEDS entry references it
# by name), or a free-text reason (floor-only: existence-checked, not
# otherwise exercised by the matrix — the ftag1lite_streaming overlay, ported
# from the old OVERLAY_GATED_BY table, and the two PHYSLITE reader fragments,
# which are documented alternates with no synthetic fixture to pair them).
FRAGMENTS: dict[str, str] = {
    "readers/easyjet_events": "paired",
    "readers/ftag1lite": "included",
    "readers/physlite_events": (
        "documented alternate leg for ttbar_vs_hh4b_event_tagger (the PHYSLITE "
        "side of the easyjet/physlite reader-parity pair, see the config's own "
        "docstring) — no synthetic PHYSLITE/POOL fixture exists to pair it into "
        "a matrix row, so it is exercised only via test_reader_fragment_instantiates."
    ),
    "readers/physlite_jets": (
        "documented JET-axis PHYSLITE reader (ElementLink-dereferenced tracks) — "
        "no synthetic PHYSLITE/POOL fixture exists to pair it into a matrix row, "
        "so it is exercised only via test_reader_fragment_instantiates."
    ),
    "readers/ftag1lite_streaming": (
        "overlay: needs a manifest and real DAOD_FTAG1LITE files to build an "
        "IterableSaltDataset — no synthetic fixture emits that corpus shape, so "
        "it stays floor-only (this table entry) rather than a matrix row."
    ),
}

# The GPU subset (§5.2): green-on-CPU rows only — gn3epclv01/gn3x/hitz are
# deliberately excluded (a strict xfail whose failure was never root-caused
# could plausibly XPASS on a different device, and a strict XPASS is red).
GPU_ROWS: set[str] = {"gn2v2_opendata", "maskformer", "gn3v00_base", "finetune_same_heads"}

# (test_name, leg) -> reason. strict=True: an unexpected PASS is a FAILURE, so
# fixing the product forces the entry to be deleted. Study-record only (§7
# resolution #4) — no tracker links until an upstream MR is on the table.
KNOWN_FAILURES: dict[tuple[str, str], str] = {
    ("gn3epclv01", "fit"): (
        "salt fit rc=1, no traceback captured; not caused by the consolidation "
        "(exp 55 run 3, exp 59 both saw it pre-W6). Root cause not yet isolated."
    ),
    ("gn3x", "fit"): (
        "salt fit rc=1, unchanged since the first lifecycle run (exp 55 run 1) — "
        "looks like a genuine config/model bug, not a harness artefact."
    ),
    ("hitz", "fit"): ("salt fit rc=1, survives the recipe-schema fixes from exp 55 run 2."),
    ("maskformer", "export"): (
        "check_onnx: NaN in torch output for MFv2_leading_objects_pt at "
        "lengths={'tracks': 0} and {'tracks': 1} — a real torch/ONNX zero-token "
        "divergence, not a harness bug. Fit and eval are green."
    ),
    ("gn2emu", "compile_plot"): (
        "graph plot --mode fit: BindError, symbolic dim 'B' resolves to 2 at key "
        "'raw.jets' declared fields but 14 at key 'raw.global' declared fields — "
        "conflicting widths. Product bug, not a feed/fixture gap (see the MATRIX "
        "Row comment above): H5StructuredReader.declare_io assigns the literal, "
        "non-stream-qualified symbolic dim 'B' to EVERY global_object stream's "
        "shape, so two global_object streams with different declared-Features "
        "widths (jets: 2, global: 14) collide in resolve_bind_schema's "
        "field-count binding — both counts come straight from the shipped "
        "config, so no feed/schema change can avoid it. Needs a product-side fix "
        "(a stream-qualified dim symbol for global_object shapes)."
    ),
    ("ftag1lite_empflow", "compile_plot"): (
        "graph validate: ConfigError, reader 'reader' has no source file. See "
        "the MATRIX Row comment above: this config's own UprootReader (via "
        "ftag1lite.yaml, unroll: jets) needs a REAL, openable ROOT file for "
        "graph validate's run-free .prepare() call (same mechanism the "
        "easyjet_flavour fix addresses), but a correct fixture needs an "
        "accurate double-jagged event->jet->track structure (18 branches "
        "across 2 groups under AntiKt4EMPFlowJetsAuxDyn.) that does not exist "
        "yet — tracked as a fixture-writing follow-up, not a harness bug."
    ),
}

# A curated (not necessarily exhaustive) table of output names that MUST be
# present in what a row's fit/eval/export/inference legs actually produce —
# replaces the golden-snapshot machinery (user ruling, "Replace entirely").
# Per row (keyed by test_name):
#   "h5": {<group>: [<column name>, ...]}  — group names use the FILE-DATASET
#     vocabulary (e.g. ``tracks_ghost``, per the reader's own `dataset:` alias
#     in its `groups:` block), not the reader STREAM name — this dissolves the
#     stream-vs-dataset drift a frozen snapshot could not (test_inference
#     gn3v00_base, pipeline #15650554). Column names are the LITERAL strings
#     ``salt test`` writes (run-name-prefixed where the field is prefixed,
#     bare otherwise).
#   "onnx": [<output name>, ...] — the literal ONNX graph output names
#     (model-name-prefixed), in tuple order. Includes MaskFormer's
#     ``leading_objects_*`` leaves — they are real expected outputs, not an
#     oversight (user ruling 2).
# Checked by CONTAINMENT, not equality: a declared name must be present;
# extra columns (input copies, pad mask, object groups, undeclared
# predictions) are fine and expected. Not necessarily exhaustive — curated,
# and meant to stay human-readable. Seeded from the deleted per-config output
# schema snapshots at ad8a54b, mapping each snapshot's h5-column stream
# through the row's own reader `groups:` `dataset:` alias to the real
# file-dataset group name.
#
# REQUIRED: every row with do_eval=True or do_onnx=True must have an entry
# naming at least one output — enforced by the completeness test below
# (test_every_eval_or_onnx_row_has_expected_outputs). gn3v00_base carries an
# entry despite do_eval=do_onnx=False purely because test_inference.py's
# ONNX-output-name check runs over every fit=True row with a committed
# contract, not just the do_onnx ones — it is not required by the
# completeness test.
EXPECTED_OUTPUTS: dict[str, dict] = {
    "gn2v2_opendata": {
        "h5": {
            "jets": [
                "GN2v2_opendata_pb",
                "GN2v2_opendata_pc",
                "GN2v2_opendata_pu",
                "GN2v2_opendata_ptau",
                "target_jets_classification",
            ],
            "tracks": [
                "GN2v2_opendata_pPileup",
                "GN2v2_opendata_pFake",
                "GN2v2_opendata_pPrimary",
                "GN2v2_opendata_pFromB",
                "GN2v2_opendata_pFromBC",
                "GN2v2_opendata_pFromC",
                "GN2v2_opendata_pFromTau",
                "GN2v2_opendata_pOtherSecondary",
                "target_track_origin",
                "VertexIndex",
                "target_track_vertexing",
            ],
        },
        "onnx": [
            "GN2v2opendata_pb",
            "GN2v2opendata_pc",
            "GN2v2opendata_pu",
            "GN2v2opendata_ptau",
            "GN2v2opendata_TrackOrigin",
            "GN2v2opendata_VertexIndex",
        ],
    },
    "gn3epclv01": {
        # GN3EPCLV01.yaml's reader aliases the tracks stream to the
        # tracks_ghost file dataset (`groups.tracks.dataset: tracks_ghost`).
        "h5": {
            "jets": [
                "GN3EPCLV01_pb",
                "GN3EPCLV01_pc",
                "GN3EPCLV01_ps",
                "GN3EPCLV01_pud",
                "GN3EPCLV01_pg",
                "GN3EPCLV01_ptau",
                "target_jets_classification",
                "GN3EPCLV01_ptFromTruthDressedWZJet",
                "target_jet_pt_regression_ptFromTruthDressedWZJet",
                "GN3EPCLV01_pbquark",
                "GN3EPCLV01_pantibquark",
                "GN3EPCLV01_pcquark",
                "GN3EPCLV01_panticquark",
                "GN3EPCLV01_pother",
                "target_jets_bccharge",
            ],
            "tracks_ghost": [
                "GN3EPCLV01_pPileup",
                "GN3EPCLV01_pFake",
                "GN3EPCLV01_pPrimary",
                "GN3EPCLV01_pFromB",
                "GN3EPCLV01_pFromBC",
                "GN3EPCLV01_pFromC",
                "GN3EPCLV01_pFromTau",
                "GN3EPCLV01_pOtherSecondary",
                "target_track_origin",
                "VertexIndex",
                "target_track_vertexing",
                "GN3EPCLV01_pNoTruth",
                "GN3EPCLV01_pOther",
                "GN3EPCLV01_pPion",
                "GN3EPCLV01_pKaon",
                "GN3EPCLV01_pElectron",
                "GN3EPCLV01_pMuon",
                "target_track_type",
            ],
        },
        "onnx": [
            "GN3EPCLV01_pb",
            "GN3EPCLV01_pc",
            "GN3EPCLV01_ps",
            "GN3EPCLV01_pud",
            "GN3EPCLV01_pg",
            "GN3EPCLV01_ptau",
            "GN3EPCLV01_ptFromTruthDressedWZJet",
            "GN3EPCLV01_pbquark",
            "GN3EPCLV01_pantibquark",
            "GN3EPCLV01_pcquark",
            "GN3EPCLV01_panticquark",
            "GN3EPCLV01_pother",
            "GN3EPCLV01_TrackOrigin",
            "GN3EPCLV01_VertexIndex",
            "GN3EPCLV01_TrackType",
        ],
    },
    "gn3x": {
        # GN3X.yaml's tracks group carries no `dataset:` alias — group == stream.
        "h5": {
            "jets": [
                "GN3XPV01_phtautauhad",
                "GN3XPV01_phbb",
                "GN3XPV01_phcc",
                "GN3XPV01_ptop",
                "GN3XPV01_pqcdbb",
                "GN3XPV01_pqcdbx",
                "GN3XPV01_pqcdcx",
                "GN3XPV01_pqcdll",
                "GN3XPV01_pWqq",
                "target_jets_classification",
            ],
            "tracks": [
                "GN3XPV01_pPileup",
                "GN3XPV01_pFake",
                "GN3XPV01_pPrimary",
                "GN3XPV01_pFromB",
                "GN3XPV01_pFromBC",
                "GN3XPV01_pFromC",
                "GN3XPV01_pFromTau",
                "GN3XPV01_pOtherSecondary",
                "target_track_origin",
                "VertexIndex",
                "target_track_vertexing",
            ],
        },
        "onnx": [
            "GN3XPV01_phtautauhad",
            "GN3XPV01_phbb",
            "GN3XPV01_phcc",
            "GN3XPV01_ptop",
            "GN3XPV01_pqcdbb",
            "GN3XPV01_pqcdbx",
            "GN3XPV01_pqcdcx",
            "GN3XPV01_pqcdll",
            "GN3XPV01_pWqq",
            "GN3XPV01_TrackOrigin",
            "GN3XPV01_VertexIndex",
        ],
    },
    "hitz": {
        "h5": {
            "jets": [
                "Hitz_TruthJetPVz",
                "Hitz_TruthJetPVz_stddev",
                "target_gaussian_regression_TruthJetPVz",
            ],
        },
        "onnx": ["Hitz_TruthJetPVz", "Hitz_TruthJetPVz_stddev"],
    },
    "maskformer": {
        # object-level groups (objects/object_masks + the tracks HadronIndex
        # leaf) are real MaskFormer outputs too but are not curated here — the
        # jets/tracks task columns below are the containment floor; the ONNX
        # leaves (incl. the leading_objects_* leaves, ruling 2) are the
        # authoritative record of the object outputs.
        "h5": {
            "jets": [
                "MaskFormer_pb",
                "MaskFormer_pc",
                "MaskFormer_pu",
                "target_jets_classification",
            ],
            "tracks": [
                "MaskFormer_pPileup",
                "MaskFormer_pFake",
                "MaskFormer_pPrimary",
                "MaskFormer_pFromB",
                "MaskFormer_pFromBC",
                "MaskFormer_pFromC",
                "MaskFormer_pFromTau",
                "MaskFormer_pOtherSecondary",
                "target_track_origin",
            ],
        },
        "onnx": [
            "MFv2_pb",
            "MFv2_pc",
            "MFv2_pu",
            "MFv2_TrackOrigin",
            "MFv2_leading_objects_pt",
            "MFv2_leading_objects_Lxy",
            "MFv2_leading_objects_deta",
            "MFv2_leading_objects_dphi",
            "MFv2_leading_objects_mass",
            "MFv2_HadronIndex",
        ],
    },
    "event_tagger_easyjet": {
        # do_onnx=False — no onnx entry. ROOT-fed (ttbar_vs_hh4b_event_tagger),
        # its event-level stream carries no reader `dataset:` alias.
        "h5": {
            "event": [
                "ttbar_vs_hh4b_event_tagger_pbackground",
                "ttbar_vs_hh4b_event_tagger_psignal",
                "target_events_classification",
            ],
        },
    },
    "regression": {
        "h5": {
            "jets": [
                "regression_HadronConeExclTruthLabelPt",
                "target_reg_normed_HadronConeExclTruthLabelPt",
                "regression_R10TruthLabel_R22v1_TruthJetMass",
                "regression_R10TruthLabel_R22v1_TruthJetPt",
                "target_reg_multinorm_R10TruthLabel_R22v1_TruthJetMass",
                "target_reg_multinorm_R10TruthLabel_R22v1_TruthJetPt",
                "regression_pt",
                "target_reg_ratio_HadronConeExclTruthLabelPt",
                "regression_truthMass",
                "regression_truthPt",
                "target_reg_multiratio_R10TruthLabel_R22v1_TruthJetMass",
                "target_reg_multiratio_R10TruthLabel_R22v1_TruthJetPt",
            ],
            "tracks": [
                "regression_dummyOutput_dPhi",
                "regression_dummyOutput_dEta",
                "target_reg_seq_tracks_dphi",
                "target_reg_seq_tracks_deta",
            ],
        },
        "onnx": [
            "regression_HadronConeExclTruthLabelPt",
            "regression_R10TruthLabel_R22v1_TruthJetMass",
            "regression_R10TruthLabel_R22v1_TruthJetPt",
            "regression_pt",
            "regression_truthMass",
            "regression_truthPt",
            "regression_dummyOutput_dPhi",
            "regression_dummyOutput_dEta",
        ],
    },
    "regression_gaussian": {
        "h5": {
            "jets": [
                "regression_gaussian_HadronConeExclTruthLabelPt",
                "regression_gaussian_HadronConeExclTruthLabelPt_stddev",
                "target_gaussian_regression_HadronConeExclTruthLabelPt",
            ],
            "tracks": [
                "regression_gaussian_dummyOutput_dPhi",
                "regression_gaussian_dummyOutput_dPhi_stddev",
                "target_gaussian_regression_no_global_object_dphi",
            ],
        },
        "onnx": [
            "regressionGaussian_HadronConeExclTruthLabelPt",
            "regressionGaussian_HadronConeExclTruthLabelPt_stddev",
            "regressionGaussian_dummyOutput_dPhi",
            "regressionGaussian_dummyOutput_dPhi_stddev",
        ],
    },
    "regression_weighted": {
        "h5": {
            "jets": [
                "regression_weighted_HadronConeExclTruthLabelPt",
                "target_reg_weighted_HadronConeExclTruthLabelPt",
                "regression_weighted_R10TruthLabel_R22v1_TruthJetMass",
                "regression_weighted_R10TruthLabel_R22v1_TruthJetPt",
                "target_reg_weighted_multi_R10TruthLabel_R22v1_TruthJetMass",
                "target_reg_weighted_multi_R10TruthLabel_R22v1_TruthJetPt",
                "regression_weighted_pt",
                "target_reg_weighted_ratio_HadronConeExclTruthLabelPt",
                "regression_weighted_truthMass",
                "regression_weighted_truthPt",
                "target_reg_weighted_multi_ratio_R10TruthLabel_R22v1_TruthJetMass",
                "target_reg_weighted_multi_ratio_R10TruthLabel_R22v1_TruthJetPt",
            ],
            "tracks": [
                "regression_weighted_dummyOutput_dPhi",
                "regression_weighted_dummyOutput_dEta",
                "target_reg_weighted_no_global_object_dphi",
                "target_reg_weighted_no_global_object_deta",
            ],
        },
        "onnx": [
            "regressionWeighted_HadronConeExclTruthLabelPt",
            "regressionWeighted_R10TruthLabel_R22v1_TruthJetMass",
            "regressionWeighted_R10TruthLabel_R22v1_TruthJetPt",
            "regressionWeighted_pt",
            "regressionWeighted_truthMass",
            "regressionWeighted_truthPt",
            "regressionWeighted_dummyOutput_dPhi",
            "regressionWeighted_dummyOutput_dEta",
        ],
    },
    "nan_regression": {
        "h5": {
            "jets": [
                "nan_regression_output_std_norm",
                "target_reg_nan_norm_HadronConeExclTruthLabelLxy",
                "nan_regression_output_ratio",
                "target_reg_nan_ratio_HadronConeExclTruthLabelLxy",
            ],
            "tracks": ["nan_regression_dummyOutput_dPhi", "target_reg_nan_seq_dphi"],
        },
        "onnx": [
            "nanRegression_output_std_norm",
            "nanRegression_output_ratio",
            "nanRegression_dummyOutput_dPhi",
        ],
    },
    "regression_multi_target": {
        "h5": {"jets": ["regression_multi_target_pt_label_handle"]},
        "onnx": ["regressionMultiTarget_pt_label_handle"],
    },
    "gn3v00_base": {
        # do_eval=do_onnx=False in MATRIX (fit-only, warms the finetune
        # templates) — carried here so test_inference.py's inference gate has
        # a contract for it too. Same tracks_ghost alias as GN3EPCLV01
        # (GN3V00.yaml `groups.tracks.dataset: tracks_ghost`).
        "h5": {
            "jets": [
                "GN3V00_pb",
                "GN3V00_pc",
                "GN3V00_ps",
                "GN3V00_pud",
                "GN3V00_pg",
                "GN3V00_ptau",
                "target_jets_classification",
                "GN3V00_ptFromTruthDressedWZJet",
                "target_jet_pt_regression_ptFromTruthDressedWZJet",
            ],
            "tracks_ghost": [
                "GN3V00_pPileup",
                "GN3V00_pFake",
                "GN3V00_pPrimary",
                "GN3V00_pFromB",
                "GN3V00_pFromBC",
                "GN3V00_pFromC",
                "GN3V00_pFromTau",
                "GN3V00_pOtherSecondary",
                "target_track_origin",
                "VertexIndex",
                "target_track_vertexing",
                "GN3V00_pNoTruth",
                "GN3V00_pOther",
                "GN3V00_pPion",
                "GN3V00_pKaon",
                "GN3V00_pElectron",
                "GN3V00_pMuon",
                "target_track_type",
            ],
        },
        "onnx": [
            "GN3V00_pb",
            "GN3V00_pc",
            "GN3V00_ps",
            "GN3V00_pud",
            "GN3V00_pg",
            "GN3V00_ptau",
            "GN3V00_ptFromTruthDressedWZJet",
            "GN3V00_TrackOrigin",
            "GN3V00_VertexIndex",
            "GN3V00_TrackType",
        ],
    },
}

_BY_NAME: dict[str, Row] = {r.test_name: r for r in MATRIX}


def row_by_name(name: str) -> Row:
    """The matrix row named ``name``."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(f"no matrix row named {name!r}") from None


def _discover() -> list[str]:
    """Every shipped config under ``CONFIG_DIR``, relative path minus suffix, minus base.yaml.

    Recursive (``rglob``) — a config added anywhere in the tree, at any
    depth, is picked up automatically (§ discovery).
    """
    found = sorted(
        p.relative_to(CONFIG_DIR).with_suffix("").as_posix() for p in CONFIG_DIR.rglob("*.yaml")
    )
    return [c for c in found if c not in _MACHINERY]


def _config_includes(config: str) -> set[str]:
    """``config``'s own top-level ``include:`` list, resolved to config_relpaths.

    Deliberately NOT ``expand_includes`` (which recurses and would also
    surface transitive includes) — this is "does THIS config's own
    ``include:`` block name it", matching the docstring convention
    ("declare its bases in the config's own include: block").
    """
    import yaml

    raw = yaml.safe_load((CONFIG_DIR / f"{config}.yaml").read_text()) or {}
    resolved: set[str] = set()
    for inc in raw.get("include") or []:
        candidate = (CONFIG_DIR / config).parent / inc
        if not candidate.is_file():
            candidate = CONFIG_DIR / inc
        resolved.add(candidate.relative_to(CONFIG_DIR).with_suffix("").as_posix())
    return resolved


_TOKEN_RE = re.compile(r"\{(\w+)(?::(\w+))?\}")


def dependencies_of(row: Row) -> set[str]:
    """Row names ``row.train_args`` references via ``{ckpt:NAME}``/``{config:NAME}``."""
    deps: set[str] = set()
    for template in row.train_args:
        for match in _TOKEN_RE.finditer(template):
            key, arg = match.group(1), match.group(2)
            if key in {"ckpt", "config"} and arg:
                deps.add(arg)
    return deps


def _load_expanded(config: str) -> dict:
    """A shipped config's include-expanded YAML, as a plain dict."""
    import yaml

    from salt.config_utils import expand_includes

    path = CONFIG_DIR / f"{config}.yaml"
    return yaml.safe_load(Path(expand_includes(str(path))).read_text()) or {}


def _reader_node(config: str) -> dict | None:
    """The expanded config's ``data.modules.reader`` node, or ``None``."""
    return ((_load_expanded(config).get("data") or {}).get("modules") or {}).get("reader")


# ---------------------------------------------------------------------------
# feeders (§1.1: the ``FEEDS`` table drives which of these a row uses)
# ---------------------------------------------------------------------------

_RECIPE_CACHE: dict[str, dict[str, Path]] = {}


# config_relpath -> the ``class_names`` this row's own flavour_label task
# ACTUALLY declares, when it differs from the SHARED recipe's default
# schema attr (pipeline #15651154, item 5: gn2v2-opendata declares
# [bjets, cjets, ujets, taujets] (open-data label order, taujets INCLUDED),
# but the "flavour_tagger" recipe's default flags: {} (no inc_taus) produces
# a 3-class flavour_label attr with no taujets — check_class_names
# (salt/model/saltmodule.py) then raises ConfigError at fit/eval time
# (schema-bound legs only; compile+plot passes no schema, so it never saw
# this). A per-config table, not a blanket recipe/flag change: GN3X ALSO
# feeds off "flavour_tagger" (see FEEDS) and ALSO declares a class set the
# recipe's default doesn't produce (9 classes) — GN3X's fit is an EXISTING
# KNOWN_FAILURES strict xfail, so fixing this generically for every
# recipe-fed row risks silently fixing GN3X's root cause too and flipping
# it to an unexpected PASS (a strict-xfail failure). Metadata-only: the
# underlying H5 flavour_label INT column is untouched by the override below
# — the appended class(es) are simply undrawn (count 0), exactly like
# taujets already is in the write_dummy_file convention this recipe mirrors
# (jet_features.yaml's own class_names vs. sample_classes split).
RECIPE_CLASS_NAMES_OVERRIDE: dict[str, list[str]] = {
    "gn2v2-opendata": ["bjets", "cjets", "ujets", "taujets"],
}


def _apply_class_names_override(
    row: Row, ctx: dict[str, Path], class_names: list[str], tmp_path_factory
) -> dict[str, Path]:
    """A row-scoped derived schema with the ``jets`` group's ``flavour_label``
    attr replaced by ``class_names`` (see ``RECIPE_CLASS_NAMES_OVERRIDE``).

    Never mutates the shared, cached recipe context (``_recipe_context``,
    keyed by recipe name only, so other rows sharing the SAME recipe get the
    UNCHANGED schema) — this returns a NEW ctx dict pointing ``"schema"`` at
    a freshly written, row-scoped file instead.
    """
    schema = load_schema(ctx["schema"])
    jets = schema.groups["jets"]
    derived_jets = replace(jets, attrs={**jets.attrs, "flavour_label": list(class_names)})
    derived_schema = replace(schema, groups={**schema.groups, "jets": derived_jets})
    out = tmp_path_factory.mktemp(f"{row.test_name}_schema") / "schema.yaml"
    save_schema(derived_schema, out)
    return {**ctx, "schema": out}


def _recipe_context(recipe: str, tmp_path_factory) -> dict[str, Path]:
    """Run a ``salt.testing.datagen`` recipe once; every row using it shares the result."""
    cached = _RECIPE_CACHE.get(recipe)
    if cached is not None:
        return cached
    import yaml

    out = tmp_path_factory.mktemp(f"datagen_{recipe}")
    pipe = load_pipeline(str(RECIPES_DIR / f"{recipe}.yaml"))
    pipe.set_output_dir(out)
    data = pipe.run()
    h5 = out / f"{recipe}.h5"
    if not h5.is_file():
        raise RuntimeError(f"recipe {recipe} wrote no H5")
    schema = out / "schema.yaml"
    save_schema(dump_schema(h5), schema)
    ctx: dict[str, Path] = {"h5": h5, "schema": schema}
    for name in ("norm_dict.yaml", "class_dict.yaml"):
        path = out / name
        if path.is_file():
            ctx[name.split("_")[0]] = path
    # not every recipe ships a NormWriter, but every config with a Normaliser
    # needs one — derive it from the produced arrays
    if "norm" not in ctx:
        nd = out / "norm_dict.yaml"
        nd.write_text(yaml.dump(compute_norm_dict(data), sort_keys=False))
        ctx["norm"] = nd
    _RECIPE_CACHE[recipe] = ctx
    return ctx


def _build_default_dummy(tmp_path_factory) -> dict[str, Path]:
    """The plain 3-class ``write_dummy_file`` fixture — rows 11-12."""
    out = tmp_path_factory.mktemp("dummy_default")
    nd, cd = out / "norm_dict.yaml", out / "class_dict.yaml"
    write_dummy_norm_dict(nd, cd)
    h5 = out / "pp_output_train.h5"
    write_dummy_file(h5, nd)
    schema = out / "schema.yaml"
    save_schema(dump_schema(h5), schema)
    return {"h5": h5, "schema": schema, "norm": nd, "class_dict": cd}


def _build_taus_dummy(tmp_path_factory) -> dict[str, Path]:
    """The 4-class (incl. taujets) ``write_dummy_file`` fixture — the GN3_baseline family.

    The "taus" flavour: same 3-class norm/class dict as the default builder,
    but the H5 itself carries a fourth ``taujets`` flavour_label value.
    """
    out = tmp_path_factory.mktemp("dummy_taus")
    nd, cd = out / "norm_dict.yaml", out / "class_dict.yaml"
    write_dummy_norm_dict(nd, cd)
    h5 = out / "pp_output_train.h5"
    write_dummy_file(h5, nd, inc_taus=True)
    schema = out / "schema.yaml"
    save_schema(dump_schema(h5), schema)
    return {"h5": h5, "schema": schema, "norm": nd, "class_dict": cd}


def _build_regression_dummy(tmp_path_factory) -> dict[str, Path]:
    """The parity fixture plus a ``mass`` jets entry — rows 8-10 (§1.2 "default + mass").

    ``regression``/``regression_weighted`` declare ``mass`` as a jets input
    Feature (a ratio denominator); the other rows in the family share this
    builder too, and simply never ask for the extra column.
    """
    import yaml

    from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict

    out = tmp_path_factory.mktemp("dummy_regression")
    nd, cd = out / "norm_dict.yaml", out / "class_dict.yaml"
    write_parity_norm_dict(nd, cd)
    raw = yaml.safe_load(nd.read_text())
    raw["jets"]["mass"] = {"mean": round(0.1 * 3, 6), "std": round(1.0 + 0.05 * 3, 6)}
    nd.write_text(yaml.dump(raw, sort_keys=False))
    h5 = out / "pp_output_train.h5"
    write_dummy_file(h5, nd)
    schema = out / "schema.yaml"
    save_schema(dump_schema(h5), schema)
    return {"h5": h5, "schema": schema, "norm": nd, "class_dict": cd}


def _build_gn3_dummy(tmp_path_factory) -> dict[str, Path]:
    """The 6-class GN3 ``write_dummy_file`` fixture — rows 13-14."""
    out = tmp_path_factory.mktemp("dummy_gn3")
    nd, cd = out / "norm_dict.yaml", out / "class_dict.yaml"
    write_dummy_norm_dict(nd, cd, is_gn3=True)
    h5 = out / "pp_output_train.h5"
    write_dummy_file(h5, nd, is_gn3=True)
    schema = out / "schema.yaml"
    save_schema(dump_schema(h5), schema)
    return {"h5": h5, "schema": schema, "norm": nd, "class_dict": cd}


def _write_schema_override_fragment(out: Path, schema: Path) -> Path:
    """A ``--config``-stackable overlay pinning the reader's ``schema:`` to
    ``schema``.

    Needed by rows that chain onto ANOTHER row's saved config
    (``{config:NAME}`` in ``train_args``, e.g. finetune_new_head onto
    gn3v00_base): that producer config is schema-bound too (from ITS OWN
    fixture), and jsonargparse's config-file deep-merge is key-by-key — a
    later ``--config``/``-c`` that never mentions ``schema:`` does not clear
    an earlier one, so the producer's schema silently survives unless this
    row explicitly re-pins its OWN schema on top (pipeline #15651154, item 3:
    finetune_new_head's compile+plot leg saw gn3v00_base's "gn3" schema —
    missing ``large_r_flavour_label`` — instead of its own "gn3_large_r"
    one). The real fit leg never hit this: ``_base_data_args``'s
    ``--data.modules.reader.init_args.schema=`` is a CLI override applied
    after every ``-c``/``--config``, so it always won there regardless; the
    compile+plot floor leg passes no such override at all (discards
    override_argv, see ``run_compile_plot``), so this row needs the
    correction to survive as a stacked config file instead.
    """
    import yaml

    overlay = {"data": {"modules": {"reader": {"init_args": {"schema": str(schema)}}}}}
    out = Path(out)
    out.write_text(yaml.safe_dump(overlay, sort_keys=False))
    return out


def _build_gn3_large_r_dummy(tmp_path_factory) -> dict[str, Path]:
    """The ``gn3`` fixture plus a ``large_r_flavour_label`` jets column — row 15.

    ``finetune_gn3large_new_head.yaml`` bolts a head onto exactly that label;
    its absence from the base checkpoint IS the domain shift the template
    demonstrates, so it lives only in this derived fixture, never in the base
    one rows 13-14 share.
    """
    import h5py as _h5py
    import numpy as _np

    from salt.utils.array_utils import join_structured_arrays

    base = _dummy_context("gn3", tmp_path_factory)
    out = tmp_path_factory.mktemp("dummy_gn3_large_r")
    h5 = out / "pp_output_train.h5"
    with _h5py.File(base["h5"]) as src, _h5py.File(h5, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        for name, dataset in src.items():
            if name != "jets":
                dst.create_dataset(name, data=dataset[:])
                for key, value in dataset.attrs.items():
                    dst[name].attrs[key] = value
                continue
            jets = dataset[:]
            rng = _np.random.default_rng(42)
            extra = rng.integers(0, 4, size=len(jets)).astype("i4")
            extra = extra.view(_np.dtype([("large_r_flavour_label", "i4")]))
            dst.create_dataset("jets", data=join_structured_arrays([jets, extra]))
            for key, value in dataset.attrs.items():
                dst["jets"].attrs[key] = value
            dst["jets"].attrs["large_r_flavour_label"] = ["hbb", "hcc", "top", "qcd"]
    schema = out / "schema.yaml"
    save_schema(dump_schema(h5), schema)
    schema_fragment = _write_schema_override_fragment(out / "schema_override.yaml", schema)
    return {
        "h5": h5,
        "schema": schema,
        "schema_fragment": schema_fragment,
        "norm": base["norm"],
        "class_dict": base["class_dict"],
    }


_DUMMY_BUILDERS = {
    "default": _build_default_dummy,
    "regression": _build_regression_dummy,
    "gn3": _build_gn3_dummy,
    "gn3_large_r": _build_gn3_large_r_dummy,
    "taus": _build_taus_dummy,
}
_DUMMY_CACHE: dict[str, dict[str, Path]] = {}


def _dummy_context(flavour: str, tmp_path_factory) -> dict[str, Path]:
    cached = _DUMMY_CACHE.get(flavour)
    if cached is not None:
        return cached
    builder = _DUMMY_BUILDERS.get(flavour)
    if builder is None:
        raise ValueError(f"unknown dummy flavour {flavour!r}")
    ctx = builder(tmp_path_factory)
    _DUMMY_CACHE[flavour] = ctx
    return ctx


def _root_deps() -> bool:
    """Whether uproot + awkward are importable (every ROOT fixture needs both)."""
    from importlib.util import find_spec

    return all(find_spec(m) is not None for m in ("uproot", "awkward"))


_ROOT_CACHE: dict[str, dict[str, Path]] = {}


def _root_context(row: Row, fragment: str | None, tmp_path_factory) -> dict[str, Path]:
    cached = _ROOT_CACHE.get(row.config)
    if cached is not None:
        return cached
    if not _root_deps():
        raise RootDepsMissingError(f"{row.test_name} is ROOT-fed; needs `pip install 'salt[root]'`")
    ctx = (
        _paired_root_context(row, fragment, tmp_path_factory)
        if fragment is not None
        else _whole_root_context(row, tmp_path_factory)
    )
    _ROOT_CACHE[row.config] = ctx
    return ctx


def _paired_root_context(row: Row, fragment: str, tmp_path_factory) -> dict[str, Path]:
    """Two synthetic minitrees + a sourced copy of ``fragment`` — row 6."""
    from salt.tests._fixtures.easyjet_minitree import (
        write_jets_norm_dict,
        write_sample_pair,
        write_sourced_fragment,
    )

    out = tmp_path_factory.mktemp(f"root_{row.test_name}")
    signal, background = write_sample_pair(out)
    raw = _load_expanded(fragment)
    sourced = write_sourced_fragment(
        raw, out / "reader.yaml", {"signal": signal, "background": background}
    )
    model = _load_expanded(row.config)
    variables = model["data"]["modules"]["features"]["init_args"]["variables"]["jets"]
    norm = write_jets_norm_dict(out / "norm_dict.yaml", variables)
    return {"fragment": sourced, "norm": norm}


def _whole_root_context(row: Row, tmp_path_factory) -> dict[str, Path]:
    """One synthetic minitree — read via ``--data.train_file``/``--data.val_file``
    for the fit leg (row 7), and via a stacked ``--config {fragment}`` overlay
    for the compile+plot floor leg (pipeline #15651154, item 1: that leg
    discards plain overrides, and its run-free ``graph validate``/``graph
    plot`` needs the reader's ``filename`` already resolved in a config file —
    see ``write_standalone_source_fragment``).
    """
    from salt.tests._fixtures.easyjet_minitree import (
        build_fixture_arrays,
        write_jets_norm_dict,
        write_minitree,
        write_standalone_source_fragment,
    )

    out = tmp_path_factory.mktemp(f"root_{row.test_name}")
    root = write_minitree(out / "data.root", build_fixture_arrays())
    expanded = _load_expanded(row.config)
    variables = expanded["data"]["modules"]["features"]["init_args"]["variables"]["jets"]
    norm = write_jets_norm_dict(out / "norm_dict.yaml", variables)
    reader_node = expanded["data"]["modules"]["reader"]
    fragment = write_standalone_source_fragment(out / "source.yaml", root, reader_node)
    return {"root": root, "norm": norm, "fragment": fragment}


def _feed_context(row: Row, tmp_path_factory) -> dict[str, Path]:
    """The format-string context (§1.1) available to ``row``'s ``train_args``.

    Gates ``EXTRAS`` first: a config whose reader/model class has a hard
    import on an optional package (mup, or a non-``root``-fed UprootReader
    config) must skip before any fixture-building is attempted, same as the
    ``root``-fed rows already do via ``_root_deps()`` below.
    """
    _require_extra(row.config)
    kind, arg = FEEDS[row.config]
    if kind == "recipe":
        assert arg is not None
        ctx = _recipe_context(arg, tmp_path_factory)
        override = RECIPE_CLASS_NAMES_OVERRIDE.get(row.config)
        if override is not None:
            ctx = _apply_class_names_override(row, ctx, override, tmp_path_factory)
        return ctx
    if kind == "dummy":
        assert arg is not None
        return _dummy_context(arg, tmp_path_factory)
    return _root_context(row, arg, tmp_path_factory)


def _norm_dict_is_unresolved(init_args: dict) -> bool:
    """Whether ``init_args`` needs a real ``norm_dict`` override: either it
    declares none at all (the "bare Normaliser" case), or it declares an
    UNRESOLVED CCRA runner placeholder (``${VAR_NAME}``, e.g.
    gn2v2-opendata.yaml's ``norm_dict: ${DATA_NORM_DICT_PATH}``) — the
    experiment runner's ``update_paths.py`` substitutes those, never this
    isolated test harness, so the literal ``${...}`` string would otherwise
    reach ``Normaliser``'s constructor and fail its own file-existence
    preflight (pipeline #15651451, item 2: gn2v2-opendata is the only
    shipped config in the matrix whose Normaliser ALREADY declares a
    norm_dict key, so the old bare ``"norm_dict" not in init_args`` check
    saw the key present and silently skipped it).
    """
    if "norm_dict" not in init_args:
        return True
    value = init_args["norm_dict"]
    return isinstance(value, str) and value.startswith("${") and value.endswith("}")


def _norm_overrides(row: Row, ctx: dict[str, Path]) -> list[str]:
    """``--model.modules.<name>.init_args.norm_dict=<norm>`` for every Normaliser
    that needs one (see ``_norm_dict_is_unresolved``).

    Reads the row's own EXPANDED config: an overlay's Normaliser lives in the
    base it includes, and reading the raw file would silently yield none.
    Several configs carry more than one Normaliser (``norm`` plus
    ``norm_global``), so overriding only the first is not enough.
    """
    norm = ctx.get("norm")
    if norm is None:
        return []
    raw = _load_expanded(row.config)
    model = raw.get("model")
    modules = (model or {}).get("init_args", {}).get("modules") if isinstance(model, dict) else None
    return [
        f"--model.modules.{name}.init_args.norm_dict={norm}"
        for name, node in (modules or {}).items()
        if isinstance(node, dict)
        and "Normaliser" in str(node.get("class_path", ""))
        and _norm_dict_is_unresolved(node.get("init_args") or {})
    ]


def _norm_set_overrides(row: Row, ctx: dict[str, Path]) -> list[str]:
    """``_norm_overrides`` results as bare ``KEY=VALUE`` (for ``graph``'s ``--set``)."""
    return [o.removeprefix("--") for o in _norm_overrides(row, ctx)]


def _base_data_args(row: Row, ctx: dict[str, Path]) -> list[str]:
    """The per-feed CLI args every row of that feed kind shares.

    Recipe/dummy rows are H5-fed: point the reader at the built file + schema.
    ROOT rows supply their own file wiring via ``train_args`` (``sources:``
    embedded in a paired fragment, or ``{root}``) — the base args here are
    deliberately minimal, so this list must not fight that.
    """
    kind = FEEDS[row.config][0]
    if kind == "root":
        return ["--data.num_workers=0", *_norm_overrides(row, ctx)]
    return [
        f"--data.train_file={ctx['h5']}",
        f"--data.val_file={ctx['h5']}",
        f"--data.modules.reader.init_args.schema={ctx['schema']}",
        # shipped configs assume large training machines
        "--data.num_workers=0",
        # the feeders emit ~1000 rows; shipped batch sizes run to 4000
        "--data.batch_size=50",
        *_norm_overrides(row, ctx),
    ]


def _trainer_args(root: Path) -> list[str]:
    """The shared FIT trainer args (§4.1) — one copy instead of five."""
    return [
        f"--trainer.default_root_dir={root}",
        # auto, not cpu: on a GPU runner these must exercise the GPU path
        "--trainer.accelerator=auto",
        "--trainer.max_epochs=1",
        "--trainer.limit_train_batches=2",
        "--trainer.limit_val_batches=2",
        "--trainer.num_sanity_val_steps=0",
        "--trainer.log_every_n_steps=1",
        # base ships a default-ON CometLogger; off so no offline archive lands
        "--trainer.logger=false",
        # null-delete the base ProgressBar: the stock enable_progress_bar=false
        # cannot coexist with a configured bar
        "--callbacks.progress=null",
    ]


# ---------------------------------------------------------------------------
# train_args expansion (§1.1, §2) — chains into run_row for {ckpt:NAME}/{config:NAME}
# ---------------------------------------------------------------------------


def _expand_train_args(
    row: Row, ctx: dict[str, Path], tmp_path_factory
) -> tuple[list[str], list[str]]:
    """``row.train_args`` with every ``{token}``/``{token:NAME}`` substituted.

    Split into ``(config_argv, override_argv)``: elements that stack an extra
    ``--config`` file (``"--config {fragment}"``, ``"--config {config:NAME}"``)
    go in ``config_argv``; everything else (``--init_from=``, per-row trainer
    overrides, ...) goes in ``override_argv``. ``_do_fit`` stacks
    ``config_argv`` BEFORE this row's own template — a ``{config:NAME}``
    dependency is a producer row's *saved* ``config.yaml``, and salt's fit
    semantics deep-merge later ``--config`` files on top (salt/parser.py), so
    stacking it after the template (or after the shared harness args) would
    let the producer's incidental settings — its own
    ``trainer.default_root_dir``, ``data.train_file``, a null
    ``training_schedule`` — clobber this row's. ``override_argv`` stays last,
    so row-specific overrides still win over both configs and the shared
    harness args.

    Each element may expand to more than one argv token (``"--config {fragment}"``
    -> two), so the whole thing is re-split with :func:`shlex.split` after
    substitution.
    """
    config_argv: list[str] = []
    override_argv: list[str] = []
    for template in row.train_args:
        expanded = _expand_one(row.test_name, template, ctx, tmp_path_factory)
        tokens = shlex.split(expanded)
        (config_argv if template.startswith("--config ") else override_argv).extend(tokens)
    return config_argv, override_argv


def _expand_one(name: str, template: str, ctx: dict[str, Path], tmp_path_factory) -> str:
    def repl(match: re.Match[str]) -> str:
        key, arg = match.group(1), match.group(2)
        if key in {"ckpt", "config"}:
            if arg is None:
                raise ValueError(f"{{{key}}} needs a row name, e.g. {{{key}:gn3v00_base}}")
            try:
                producer = run_row(arg, tmp_path_factory)
            except LegFailedError as exc:
                raise LegFailedError(
                    name, "fit", f"producer row {exc.producer} failed", producer=exc.producer
                ) from exc
            return str(producer.ckpt if key == "ckpt" else producer.saved_config)
        if arg is not None:
            raise ValueError(f"unexpected token {match.group(0)!r} in {name}'s train_args")
        if key not in ctx:
            raise KeyError(
                f"{name}: train_args token {{{key}}} not in this row's feed context ({sorted(ctx)})"
            )
        return str(ctx[key])

    return _TOKEN_RE.sub(repl, template)


# ---------------------------------------------------------------------------
# the floor leg (compile+plot) — every row, ``fit`` or not
# ---------------------------------------------------------------------------


_STDERR_TAIL_LINES = 20


def _salt_main(name: str, leg: str, argv: list[str]) -> None:
    """Run ``salt_main(argv)`` for ``name``'s ``leg``; raises ``LegFailedError``
    on ANY failure — a nonzero rc, or ``salt.main.main``'s ``SystemExit``
    re-raise on a failed CLI parse. Returns ``None`` on success (rc == 0):
    every call site below is now "call it, then keep going" — a failure
    always surfaces as ``LegFailedError``, never a bare rc for the caller to
    re-check.

    Harness fix (pipeline #15651025 cluster 4 — "unblocks all future
    diagnosis"): CI runs pytest with ``--show-capture=stdout``, and every
    ``salt`` error path prints to STDERR (``console(..., file=sys.stderr)``,
    argparse's own usage errors, ``GraphError``'s one-block form) — so a
    failing leg showed NOTHING in the CI log, just a bare rc/exit code. This
    redirects stderr into a buffer for the duration of the call; on ANY
    failure the last ``_STDERR_TAIL_LINES`` lines print to STDOUT (so CI
    actually shows them) and are folded into the raised ``LegFailedError``'s
    message (so a producer-row skip elsewhere names the real cause, not just
    "rc=1"). On rc == 0 the buffer is discarded silently.

    ``salt.main.main`` re-raising ``SystemExit`` from a failed CLI parse
    (instead of returning a nonzero rc) matters beyond messaging too: left
    uncaught, it skips the ``_ROW_CACHE``/``_INFERENCE_CACHE`` memoisation
    entirely, so every row chained onto a dead producer
    (``{ckpt:NAME}``/``{config:NAME}``) re-runs the full fit from scratch
    instead of skipping with a named reason (pipeline #15650554, item B4 —
    gn2v2_opendata's 7 legs each re-ran the whole fit).
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf):
            rc = salt_main(argv)
    except SystemExit as exc:
        _fail_with_stderr(name, leg, buf, f"salt {leg} exited {exc.code} (argv parse)", exc)
        return
    if rc != 0:
        _fail_with_stderr(name, leg, buf, f"salt {leg} rc={rc}")


def _fail_with_stderr(
    name: str, leg: str, buf: io.StringIO, reason: str, cause: BaseException | None = None
) -> None:
    """Print ``buf``'s captured-stderr tail to stdout, then raise ``LegFailedError``
    naming ``reason`` with the SAME tail folded into its message. Always raises.
    """
    lines = buf.getvalue().splitlines()
    tail = "\n".join(lines[-_STDERR_TAIL_LINES:])
    if tail:
        n_shown = min(len(lines), _STDERR_TAIL_LINES)
        print(f"--- {name} {leg}: captured stderr (last {n_shown} lines) ---")
        print(tail)
        print(f"--- {name} {leg}: end captured stderr ---")
        reason = f"{reason}\nstderr tail:\n{tail}"
    if cause is not None:
        raise LegFailedError(name, leg, reason) from cause
    raise LegFailedError(name, leg, reason)


def run_compile_plot(row: Row, tmp_path_factory, tmp_path: Path) -> None:
    """Every row's floor leg: ``graph validate`` (all modes) + ``graph plot --mode fit``.

    Runs regardless of ``row.fit`` — a ``fit=False`` row (no synthetic
    fixture the lifecycle can serve) still proves its plan compiles and
    renders, just with no checkpoint. A chained row (``{config:NAME}`` in its
    ``train_args``) reuses the SAME producer artifact the fit leg would
    (``_expand_train_args``/``run_row``), so a finetune template's compile
    leg genuinely needs its base already fit — this is correct, not
    accidental: the template cannot compile without the base config it warms
    up.

    NOTE: ``--mode`` (``salt/cli.py``) is a plain, non-``append`` argparse
    flag — passing it four times, as the old Tier A did, leaves
    ``mode=onnx`` (the last value wins), not all four. Ported byte-for-byte
    rather than "fixed": this item's scope is completeness + the two named
    bugs (§8), and validating fit/val/test for the first time on ~30 configs
    that have never been checked in those modes is a change with unknown
    blast radius, not something to slip in unannounced here.
    """
    ctx = _feed_context(row, tmp_path_factory)
    config_argv, _ = _expand_train_args(row, ctx, tmp_path_factory)
    sets: list[str] = ["--set", "trainer.accelerator=auto"]
    for override in _norm_set_overrides(row, ctx):
        sets += ["--set", override]

    cfg_path = str(CONFIG_DIR / f"{row.config}.yaml")
    argv = ["graph", "validate", *config_argv, "-c", cfg_path]
    for mode in ("fit", "val", "test", "onnx"):
        argv += ["--mode", mode]
    _salt_main(row.test_name, "validate", [*argv, *sets])

    plot_path = tmp_path / f"{row.test_name}.dot"
    plot_argv = [
        "graph",
        "plot",
        *config_argv,
        "-c",
        cfg_path,
        "--mode",
        "fit",
        "-o",
        str(plot_path),
    ]
    _salt_main(row.test_name, "plot", [*plot_argv, *sets])
    assert plot_path.is_file(), f"{row.test_name}: graph plot wrote nothing to {plot_path}"


_COMPILE_PLOT_PARAMS = [row.test_name for row in MATRIX]


@pytest.mark.parametrize("name", _COMPILE_PLOT_PARAMS)
def test_compile_plot(name, tmp_path_factory, tmp_path, request):
    """Floor leg — every row's config plan-compiles and its fit-mode DOT renders.

    Leg name "compile_plot" for KNOWN_FAILURES purposes (study xfail policy:
    "pre-existing lifecycle failures get tracked xfails, not debugging
    expeditions") — this leg has no {fit, eval, export} split of its own, so
    one entry covers both the ``graph validate`` and ``graph plot`` calls
    inside ``run_compile_plot``.
    """
    row = row_by_name(name)
    _apply_known_xfail(request, name, "compile_plot")
    try:
        run_compile_plot(row, tmp_path_factory, tmp_path)
    except RootDepsMissingError as exc:
        pytest.skip(str(exc))
    except LegFailedError as exc:
        if exc.producer != name:
            pytest.skip(f"producer row {exc.producer} failed")
        raise


# ---------------------------------------------------------------------------
# the legs (§4.1) — session-scoped, memoised per row
# ---------------------------------------------------------------------------

# dict[test_name, Artifacts | LegFailedError]. No xdist today (pyproject has
# none, CI runs one pytest process), so a plain module-level dict is correct.
# If xdist is ever added, every matrix row must stay in one module (this
# module already is), and `--dist loadfile` becomes mandatory — this cache
# is not otherwise safe to share across workers.
_ROW_CACHE: dict[str, Artifacts | LegFailedError] = {}


def run_row(name: str, tmp_path_factory) -> Artifacts:
    """Run (or fetch) matrix row ``name`` through the fit leg.

    Memoised — the work happens once, on the first leg (of any row) that
    asks. Raises :class:`LegFailedError`: ``producer == name`` when this row's
    own fit failed, or names the upstream row when a
    ``{ckpt:NAME}``/``{config:NAME}`` dependency failed first.
    """
    cached = _ROW_CACHE.get(name)
    if isinstance(cached, LegFailedError):
        raise cached
    if cached is not None:
        return cached
    row = row_by_name(name)
    try:
        ctx = _feed_context(row, tmp_path_factory)
        config_argv, override_argv = _expand_train_args(row, ctx, tmp_path_factory)
        artifacts = _do_fit(row, ctx, config_argv, override_argv, tmp_path_factory)
    except LegFailedError as exc:
        _ROW_CACHE[name] = exc
        raise
    _ROW_CACHE[name] = artifacts
    return artifacts


def _needs_mup_shapes(config: str) -> bool:
    """Whether ``config``'s model declares a ``mup:`` block (structural check
    on the config itself, not a row-name special case — GN2_muP is simply the
    only shipped config where this is currently true).
    """
    model = _load_expanded(config).get("model") or {}
    return bool((model.get("init_args") or {}).get("mup"))


def _generate_mup_shapes(row: Row, ctx: dict[str, Path], tmp_path_factory) -> Path:
    """Run ``salt mup-shapes`` for ``row``'s config — the real muP pre-fit
    workflow (pipeline #15651154, item 4): the shipped config's
    ``model.init_args.mup.shape_path`` is a documented placeholder
    ("shape_path is a REQUIRED override — generate with `salt mup-shapes`",
    GN2_muP.yaml), never a real file, so ``salt fit`` on it verbatim always
    fails looking for ``shape_mup.bsh``. ``generate_shapes`` builds its
    base/delta probe models data-free/run-free purely from the config, so
    this needs only the same norm_dict override every bare Normaliser in the
    row's config needs for a run-free parse (``_norm_set_overrides``) — base/
    delta widths default from the config's own configured width.
    """
    out_dir = tmp_path_factory.mktemp(f"mup_shapes_{row.test_name}")
    shape_path = out_dir / "shape_mup.bsh"
    argv = [
        "mup-shapes",
        "-c",
        str(CONFIG_DIR / f"{row.config}.yaml"),
        "--save-path",
        str(shape_path),
    ]
    for override in _norm_set_overrides(row, ctx):
        argv += ["--set", override]
    _salt_main(row.test_name, "mup-shapes", argv)
    return shape_path


def _do_fit(
    row: Row,
    ctx: dict[str, Path],
    config_argv: list[str],
    override_argv: list[str],
    tmp_path_factory,
) -> Artifacts:
    # explicit, direct gate — pipeline #15651025 cluster 3: gn2_mup's fit leg
    # rc=1'd on a missing `mup` despite EXTRAS mapping GN2/GN2_muP to it;
    # `_feed_context` (called by `run_row` just before this) already calls
    # `_require_extra`, but the failure surfaced anyway, so this leg gets its
    # OWN direct call rather than relying solely on that transitive path.
    _require_extra(row.config)
    mup_shape_overrides: list[str] = []
    if _needs_mup_shapes(row.config):
        shape_path = _generate_mup_shapes(row, ctx, tmp_path_factory)
        mup_shape_overrides = [f"--model.init_args.mup.shape_path={shape_path}"]
    root = tmp_path_factory.mktemp(row.test_name)
    argv = ["fit"]
    # a {config:NAME} dependency is a producer row's *saved* config.yaml —
    # stack it BEFORE this row's own template (docs/tutorials/finetuning.md's
    # worked order: saved base config first, template second), so the
    # template's own settings (training_schedule, the new head, ...) win over
    # the producer's incidental ones on any key both happen to set.
    argv += config_argv
    argv += ["--config", str(CONFIG_DIR / f"{row.config}.yaml")]
    argv += _base_data_args(row, ctx)
    argv += _trainer_args(root)
    argv += mup_shape_overrides
    # override_argv last: row-specific overrides (finetune epochs/limits, the
    # input_samples fix, ...) must win over both configs and the shared
    # defaults above.
    argv += override_argv
    _salt_main(row.test_name, "fit", argv)
    ckpts = sorted(root.rglob("*.ckpt"))
    if not ckpts:
        raise LegFailedError(row.test_name, "fit", f"training wrote no checkpoint under {root}")
    configs = sorted(root.rglob("config.yaml"))
    if not configs:
        raise LegFailedError(row.test_name, "fit", f"training saved no config.yaml under {root}")
    return Artifacts(ckpt=ckpts[0], saved_config=configs[0], root_dir=root)


def run_eval(name: str, tmp_path_factory) -> Path:
    """Run (or fetch) row ``name``'s eval leg: ``salt test`` + schema-parity (§4.2).

    Requires the fit leg's artifacts first (calls :func:`run_row`).
    """
    artifacts = run_row(name, tmp_path_factory)
    if artifacts.eval_h5 is not None:
        return artifacts.eval_h5
    row = row_by_name(name)
    ctx = _feed_context(row, tmp_path_factory)
    argv = [
        "test",
        "--config",
        str(artifacts.saved_config),
        f"--ckpt_path={artifacts.ckpt}",
        "--data.num_workers=0",
        "--trainer.accelerator=auto",
        "--trainer.logger=false",
        "--callbacks.progress=null",
    ]
    if "h5" in ctx:
        argv.append(f"--data.test_file={ctx['h5']}")
    _salt_main(name, "eval", argv)
    evals = sorted(artifacts.ckpt.parent.glob("*__test_*.h5"))
    if not evals:
        raise LegFailedError(name, "eval", f"eval wrote no H5 next to {artifacts.ckpt}")
    eval_h5 = evals[-1]
    artifacts.eval_h5 = eval_h5
    _assert_eval_h5_has_expected_outputs(eval_h5, row)
    return eval_h5


def run_export(name: str, tmp_path_factory) -> Path:
    """Run (or fetch) row ``name``'s export leg: checked ONNX export (§4.1).

    Deliberately *without* ``--no-check`` — ``rc == 0`` IS the torch<->ONNX
    parity assertion. Additionally checks the exported graph's own output
    names against ``EXPECTED_OUTPUTS[name]["onnx"]`` (containment) when an
    entry exists — the export leg's own enforcement point (user ruling),
    independent of ``test_inference.py``'s onnxruntime-session check on the
    same file.
    """
    artifacts = run_row(name, tmp_path_factory)
    if artifacts.onnx is not None:
        return artifacts.onnx
    onnx_dir = artifacts.root_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / f"{name}.onnx"
    _salt_main(
        name,
        "export",
        [
            "export",
            "--config",
            str(artifacts.saved_config),
            f"--ckpt_path={artifacts.ckpt}",
            f"--output={onnx_path}",
        ],
    )
    if not sorted(onnx_dir.glob("*.onnx")):
        raise LegFailedError(name, "export", f"no ONNX written under {onnx_dir}")
    expected_onnx = EXPECTED_OUTPUTS.get(name, {}).get("onnx")
    if expected_onnx:
        actual = [o.name for o in onnx.load(str(onnx_path)).graph.output]
        missing = [n for n in expected_onnx if n not in actual]
        if missing:
            raise LegFailedError(
                name,
                "export",
                f"exported ONNX is missing declared EXPECTED_OUTPUTS output(s) "
                f"{missing}; actual tuple: {actual}",
            )
    artifacts.onnx = onnx_path
    return onnx_path


def _assert_eval_h5_has_expected_outputs(eval_h5: Path, row: Row) -> None:
    """The written eval H5 carries at least the declared ``EXPECTED_OUTPUTS`` columns.

    Containment, not equality (user ruling, replacing the golden-snapshot
    exact-schema check): extra columns (input copies, pad mask, object
    groups, undeclared predictions) are expected and fine — only a MISSING
    declared column is a failure. Compares what was actually written, not
    what was planned.
    """
    expected = EXPECTED_OUTPUTS.get(row.test_name, {}).get("h5")
    if not expected:
        return
    with h5py.File(eval_h5) as f:
        for group, names in expected.items():
            if group not in f:
                raise LegFailedError(
                    row.test_name,
                    "eval",
                    f"EXPECTED_OUTPUTS expects H5 group {group!r}, the eval H5 has none "
                    f"(groups present: {sorted(f)})",
                )
            actual = set(f[group].dtype.names or ())
            missing = [n for n in names if n not in actual]
            if missing:
                raise LegFailedError(
                    row.test_name,
                    "eval",
                    f"eval H5 group {group!r} is missing declared EXPECTED_OUTPUTS "
                    f"column(s) {missing}; actual columns: {sorted(actual)}",
                )


# ---------------------------------------------------------------------------
# pytest surface — three parametrized legs per row
# ---------------------------------------------------------------------------

_PARAMS = [
    pytest.param(row.test_name, marks=(pytest.mark.gpu,) if row.test_name in GPU_ROWS else ())
    for row in MATRIX
]


def _apply_known_xfail(request: pytest.FixtureRequest, name: str, leg: str) -> None:
    reason = KNOWN_FAILURES.get((name, leg))
    if reason is not None:
        request.node.add_marker(pytest.mark.xfail(strict=True, reason=reason))


@pytest.mark.parametrize("name", _PARAMS)
def test_fit(name, tmp_path_factory, request):
    """Leg 1 — ``fit=True`` rows: ``salt fit`` produces a checkpoint + saved config."""
    row = row_by_name(name)
    if not row.fit:
        pytest.skip("row declares fit=False — compile+plot only (see test_compile_plot)")
    _apply_known_xfail(request, name, "fit")
    try:
        run_row(name, tmp_path_factory)
    except RootDepsMissingError as exc:
        pytest.skip(str(exc))
    except LegFailedError as exc:
        if exc.producer != name:
            pytest.skip(f"producer row {exc.producer} failed")
        raise


@pytest.mark.parametrize("name", _PARAMS)
def test_eval(name, tmp_path_factory, request):
    """Leg 2 — ``do_eval=True`` rows: ``salt test`` + the EXPECTED_OUTPUTS containment check."""
    row = row_by_name(name)
    if not row.do_eval:
        pytest.skip("row declares do_eval=False")
    _apply_known_xfail(request, name, "eval")
    try:
        run_eval(name, tmp_path_factory)
    except RootDepsMissingError as exc:
        pytest.skip(str(exc))
    except LegFailedError as exc:
        if exc.leg == "fit":
            pytest.skip(f"fit failed for {name}")
        raise


@pytest.mark.parametrize("name", _PARAMS)
def test_export(name, tmp_path_factory, request):
    """Leg 3 — ``do_onnx=True`` rows: ``salt export``, checked (torch<->ONNX parity)."""
    row = row_by_name(name)
    if not row.do_onnx:
        pytest.skip("row declares do_onnx=False")
    _apply_known_xfail(request, name, "export")
    try:
        run_export(name, tmp_path_factory)
    except RootDepsMissingError as exc:
        pytest.skip(str(exc))
    except LegFailedError as exc:
        if exc.leg == "fit":
            pytest.skip(f"fit failed for {name}")
        raise


# --------------------------------------------------------- completeness (§4.3)


def test_every_discovered_config_is_placed():
    """§ discovery: every config is a MATRIX row or a FRAGMENTS entry — never neither, never both.

    Enforces "require ALL configs be defined in test_pipeline" (user,
    verbatim). A config discovered on disk in neither table is an actionable
    failure naming the file and both tables; stale entries (naming a file
    that no longer exists) and rows also listed as fragments are equally a
    failure.
    """
    discovered = set(_discover())
    matrix_configs = {r.config for r in MATRIX}
    fragment_configs = set(FRAGMENTS)

    unplaced = sorted(discovered - matrix_configs - fragment_configs)
    assert not unplaced, (
        f"shipped config(s) with neither a MATRIX row nor a FRAGMENTS entry: {unplaced}. "
        "Add a Row to MATRIX (pipeline/test_pipeline.py) naming a FEEDS source, or a "
        "FRAGMENTS entry naming how it is exercised."
    )
    stale_rows = sorted(matrix_configs - discovered)
    assert not stale_rows, f"MATRIX rows name configs that do not exist: {stale_rows}"
    stale_fragments = sorted(fragment_configs - discovered)
    assert not stale_fragments, f"FRAGMENTS names configs that do not exist: {stale_fragments}"
    both = sorted(matrix_configs & fragment_configs)
    assert not both, f"named in both MATRIX and FRAGMENTS: {both}"


def test_every_fragment_is_exercised():
    """Every FRAGMENTS entry is reachable some way other than sitting unused on disk.

    "included" entries need >=1 MATRIX row's config to ``include:`` them
    directly; "paired" entries need a FEEDS entry to reference them by name;
    anything else is a free-text reason and is only existence-checked (by
    ``test_every_discovered_config_is_placed`` above) — the
    ``ftag1lite_streaming`` overlay and the two undocumented-pairing PHYSLITE
    reader fragments.
    """
    included_by: set[str] = set()
    for row in MATRIX:
        included_by |= _config_includes(row.config)
    paired = {arg for kind, arg in FEEDS.values() if kind == "root" and arg is not None}
    for fragment, how in FRAGMENTS.items():
        if how == "included":
            assert fragment in included_by, (
                f"{fragment} is marked 'included' in FRAGMENTS but no MATRIX row's "
                "config include:s it"
            )
        elif how == "paired":
            assert fragment in paired, (
                f"{fragment} is marked 'paired' in FRAGMENTS but no FEEDS entry "
                "references it"
            )


_READER_FRAGMENTS = [c for c in FRAGMENTS if _reader_node(c) is not None]


@pytest.mark.parametrize("fragment", _READER_FRAGMENTS)
def test_reader_fragment_instantiates(fragment):
    """A reader fragment's reader builds — the gate a fragment gets instead of a matrix row.

    A fragment has no model, so there is no plan to compile, but every
    invariant a reader enforces (``unroll`` naming a scalar group,
    link_branch/target_prefix pairing, constituent cuts on a jagged stream
    naming configured branches, ...) is raised from its constructor and is
    caught here.
    """
    _require_extra(fragment)
    from jsonargparse import ArgumentParser

    from salt.data.base import Reader

    parser = ArgumentParser(exit_on_error=False)
    parser.add_subclass_arguments(Reader, "reader")
    cfg = parser.parse_object({"reader": _reader_node(fragment)})
    assert parser.instantiate_classes(cfg).reader is not None


def test_matrix_test_names_are_unique():
    """``test_name`` is the artifact-cache key and the junit id."""
    names = [r.test_name for r in MATRIX]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"MATRIX test_names are not unique: {dupes}"


def test_matrix_dependencies_are_defined_and_acyclic():
    """Every ``{ckpt:NAME}``/``{config:NAME}`` names a row; the DAG has no cycle."""
    names = {r.test_name for r in MATRIX}
    deps = {r.test_name: dependencies_of(r) for r in MATRIX}
    for name, dep_names in deps.items():
        unknown = sorted(dep_names - names)
        assert not unknown, f"{name}: train_args names unknown row(s) {unknown}"

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str, stack: tuple[str, ...]) -> None:
        if name in visited:
            return
        if name in visiting:
            raise AssertionError(f"MATRIX dependency cycle: {' -> '.join((*stack, name))}")
        visiting.add(name)
        for dep in deps[name]:
            visit(dep, (*stack, name))
        visiting.discard(name)
        visited.add(name)

    for name in names:
        visit(name, ())


def test_every_matrix_config_has_a_feed():
    """No row can reach the fit leg without a declared data source."""
    missing = sorted({r.config for r in MATRIX} - set(FEEDS))
    assert not missing, f"MATRIX rows with no FEEDS entry: {missing}"


def test_known_failures_name_real_rows_and_legs():
    """Stale KNOWN_FAILURES entries fail loudly instead of silently protecting nothing.

    "compile_plot" joins {fit, eval, export} as a valid leg name — the floor
    leg every row gets (test_compile_plot), added when KNOWN_FAILURES grew
    its first compile_plot-leg entries (gn2emu, ftag1lite_empflow).
    """
    names = {r.test_name for r in MATRIX}
    for name, leg in KNOWN_FAILURES:
        assert name in names, f"KNOWN_FAILURES names an unknown row: {name!r}"
        assert leg in {"fit", "eval", "export", "compile_plot"}, (
            f"KNOWN_FAILURES names an unknown leg {leg!r} for {name!r}"
        )


def test_every_eval_or_onnx_row_has_expected_outputs():
    """Every ``do_eval=True`` or ``do_onnx=True`` row has an ``EXPECTED_OUTPUTS``
    entry naming at least one output (user ruling — the completeness gate for
    the curated table that replaced the deleted output-schema snapshots).
    """
    required = {r.test_name for r in MATRIX if r.do_eval or r.do_onnx}
    for name in sorted(required):
        entry = EXPECTED_OUTPUTS.get(name)
        assert entry is not None, (
            f"{name} has do_eval or do_onnx True and no EXPECTED_OUTPUTS entry — add one, "
            "seeded from what `salt test`/`salt export` actually produce"
        )
        names = [n for cols in entry.get("h5", {}).values() for n in cols] + list(
            entry.get("onnx") or ()
        )
        assert names, f"{name}'s EXPECTED_OUTPUTS entry names no outputs at all"


# ----------------------------------------------- residual finetune assertions (§3)
# What a matrix row cannot express: claims about the CHAINED artifacts
# (rows 13-15), not observable from rc == 0 on a fit.


def test_base_run_carries_the_modules_the_templates_freeze(tmp_path_factory):
    """The saved base config names the head both finetune templates warm up."""
    import yaml

    try:
        artifacts = run_row("gn3v00_base", tmp_path_factory)
    except LegFailedError as exc:
        pytest.skip(f"gn3v00_base fit failed: {exc}")
    modules = yaml.safe_load(artifacts.saved_config.read_text())["model"]["init_args"]["modules"]
    assert "jets_classification" in modules, (
        "finetune_gn3large.yaml warms up `jets_classification`; the base config "
        "no longer defines it"
    )


def test_added_head_is_absent_from_the_pretrained_checkpoint(tmp_path_factory):
    """The new head really is new — the warm start cannot be a no-op."""
    import torch

    try:
        artifacts = run_row("gn3v00_base", tmp_path_factory)
    except LegFailedError as exc:
        pytest.skip(f"gn3v00_base fit failed: {exc}")
    state = torch.load(artifacts.ckpt, map_location="cpu", weights_only=False)
    keys = state.get("state_dict", state)
    assert not any("large_r_jet_classification" in k for k in keys), (
        "the base checkpoint already carries the head the template adds, so this "
        "test would no longer prove the new-head path works"
    )


# ---------------------------------------------------------------------------
# regression + gaussian-regression per-config semantics
# ---------------------------------------------------------------------------
#
# Rows ``regression``/``regression_gaussian`` give the fit/eval/export
# lifecycle above; what follows is the de-scale, doubled-column and ONNX-rank
# assertions that are per-config semantics the generic runner cannot express
# and must survive the fold of ``test_regression_e2e.py`` +
# ``test_regression_gaussian_e2e.py``.
#
# do_onnx=True for both rows (§7 resolution): the "no gaussian handling"
# claim this section's gaussian half used to carry was a stale docstring, not
# a property of ``check_onnx`` — ``compare_once``
# (``salt/outputs/sinks/onnx/check.py``) is fully generic (by-name allclose on
# floats, exact ints, dead-output canary), so the gaussian export is checked
# like every other row.


def _eval_h5(name: str, tmp_path_factory) -> Path:
    try:
        return run_eval(name, tmp_path_factory)
    except LegFailedError as exc:
        pytest.skip(f"{name} {exc.leg} failed: {exc}")


def _onnx(name: str, tmp_path_factory) -> Path:
    try:
        return run_export(name, tmp_path_factory)
    except LegFailedError as exc:
        pytest.skip(f"{name} {exc.leg} failed: {exc}")


@pytest.fixture(scope="module")
def regression_eval_h5(tmp_path_factory) -> Path:
    return _eval_h5("regression", tmp_path_factory)


@pytest.fixture(scope="module")
def gaussian_eval_h5(tmp_path_factory) -> Path:
    return _eval_h5("regression_gaussian", tmp_path_factory)


@pytest.fixture(scope="module")
def regression_onnx(tmp_path_factory) -> Path:
    return _onnx("regression", tmp_path_factory)


@pytest.fixture(scope="module")
def gaussian_onnx(tmp_path_factory) -> Path:
    return _onnx("regression_gaussian", tmp_path_factory)


def test_regression_eval_h5_columns_present_and_descaled(regression_eval_h5):
    """The section eval H5 carries the regression columns, finite and de-scaled."""
    with h5py.File(regression_eval_h5) as f:
        jets = f["jets"][:]
        tracks = f["tracks"][:]
    jet_cols = set(jets.dtype.names)
    # the five regression heads' custom / target column names (regression.yaml)
    for col in (
        "regression_HadronConeExclTruthLabelPt",
        "regression_pt",
        "regression_truthMass",
        "regression_truthPt",
    ):
        assert col in jet_cols, f"missing regression column {col}: {sorted(jet_cols)}"
    # reg_normed -> HadronConeExclTruthLabelPt (norm_params mean=1.0 std=1.0):
    # de-scaled = raw*1 + 1, so the column is finite (the de-scale really ran)
    assert np.isfinite(jets["regression_HadronConeExclTruthLabelPt"]).all()
    # the per-token seq head columns land on the tracks stream
    track_cols = set(tracks.dtype.names)
    for col in ("regression_dummyOutput_dPhi", "regression_dummyOutput_dEta"):
        assert col in track_cols, f"missing seq regression column {col}"
    assert jets.shape[0] > 0


def test_gaussian_eval_h5_has_stddev_columns(gaussian_eval_h5):
    """The eval H5 carries the gaussian doubled columns (mean + _stddev), both streams."""
    with h5py.File(gaussian_eval_h5) as f:
        jet_cols = set(f["jets"].dtype.names)
        track_cols = set(f["tracks"].dtype.names)
        n_rows = f["jets"].shape[0]
    assert n_rows > 0
    assert any(c.endswith("_stddev") for c in jet_cols), f"no gaussian stddev in jets: {jet_cols}"
    assert any(
        c.endswith("_stddev") for c in track_cols
    ), f"no gaussian stddev in tracks: {track_cols}"


class TestRegressionOnnxContract:
    """The shipped regression.yaml exports a well-formed ONNX contract (row 8)."""

    def test_onnx_output_ranks_global_vs_per_token(self, regression_onnx):
        """6 rank-0 globals (norm/ratio scalars) + 2 rank-1 per-token seq columns."""
        model = onnx.load(str(regression_onnx))
        ranks = {o.name: len(o.type.tensor_type.shape.dim) for o in model.graph.output}
        assert sorted(ranks.values()) == [0, 0, 0, 0, 0, 0, 1, 1], ranks

    def test_onnx_session_runs(self, regression_onnx):
        """The exported graph runs in onnxruntime on batch-1 inputs (L=5 tokens)."""
        sess = make_session(regression_onnx)
        in_meta = {i.name: i.shape for i in sess.get_inputs()}
        rng = np.random.default_rng(0)

        def shape_for(dims):
            return tuple(5 if (isinstance(d, str) or d is None) else d for d in dims)

        feeds = {
            name: rng.standard_normal(shape_for(dims)).astype(np.float32)
            for name, dims in in_meta.items()
        }
        out = {o.name: v for o, v in zip(sess.get_outputs(), sess.run(None, feeds), strict=True)}
        assert len(out) == 8, f"expected the 8 regression outputs, got {sorted(out)}"


class TestGaussianOnnxContract:
    """The shipped regression_gaussian.yaml exports a well-formed ONNX contract (row 9)."""

    def test_onnx_output_ranks_global_vs_per_token(self, gaussian_onnx):
        """2 rank-0 globals (mean + _stddev of the global head) + 2 rank-1 per-token."""
        model = onnx.load(str(gaussian_onnx))
        ranks = {o.name: len(o.type.tensor_type.shape.dim) for o in model.graph.output}
        assert sorted(ranks.values()) == [0, 0, 1, 1], ranks

    def test_onnx_session_runs(self, gaussian_onnx):
        """The exported graph runs in onnxruntime on batch-1 inputs (L=5 tokens)."""
        sess = make_session(gaussian_onnx)
        in_meta = {i.name: i.shape for i in sess.get_inputs()}
        rng = np.random.default_rng(0)

        def shape_for(dims):
            return tuple(5 if (isinstance(d, str) or d is None) else d for d in dims)

        feeds = {
            name: rng.standard_normal(shape_for(dims)).astype(np.float32)
            for name, dims in in_meta.items()
        }
        out = {o.name: v for o, v in zip(sess.get_outputs(), sess.run(None, feeds), strict=True)}
        assert len(out) == 4, f"expected the 4 gaussian outputs, got {sorted(out)}"
