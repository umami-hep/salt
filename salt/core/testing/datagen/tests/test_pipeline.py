"""Unit tests for the modular generator ``Pipeline`` + contract validator.

Covers contract ordering (inserter-before-Tracks raises ``RecipeError``),
field-level ``requires`` resolution, duplicate-producer / undeclared-mutation /
self-cycle / no-writer errors, and jsonargparse ``class_path``/``init_args``
load + end-to-end run to a written H5.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from salt.core.testing.datagen import RecipeError
from salt.core.testing.datagen.modules import (
    ClassDictWriter,
    Constituents,
    H5Writer,
    Jets,
    NormWriter,
    Tracks,
    TruthHadronInserter,
)
from salt.core.testing.datagen.pipeline import Pipeline, load_pipeline

_RECIPES_DIR = Path(__file__).resolve().parent.parent / "recipes"

_JET_FIELDS = [
    {"name": "pt", "type": "distribution", "dtype": "f4"},
    {"name": "eta", "type": "distribution", "dtype": "f4"},
    {
        "name": "flavour_label",
        "type": "label",
        "dtype": "i4",
        "classes": [0, 1, 2, 3],
        "sample_classes": [0, 1, 2],
    },
]
_TRACK_FIELDS = [{"name": "d0", "type": "distribution", "dtype": "f4"}]
_HADRON_FIELDS = [
    {"name": "pt", "type": "distribution", "dtype": "f4"},
    {"name": "barcode", "type": "id", "dtype": "i4", "range": [0, 10000],
     "scope": "jet", "unique": True},
    {"name": "flavour", "type": "label", "dtype": "i4",
     "classes": [-1, 4, 5], "invalid_fill": -1},
]


# --------------------------------------------------------------------------- #
# Contract / ordering
# --------------------------------------------------------------------------- #
def test_inserter_before_tracks_raises(tmp_path):
    """The done-criterion: a mis-ordered recipe (inserter before Tracks) is
    rejected at ``Pipeline.__init__`` with a clear error."""
    modules = [
        Jets(fields=_JET_FIELDS),
        TruthHadronInserter(hadron_fields=_HADRON_FIELDS),  # requires "tracks"
        Tracks(fields=_TRACK_FIELDS),
        H5Writer(path=str(tmp_path / "x.h5")),
    ]
    with pytest.raises(RecipeError, match="requires 'tracks'"):
        Pipeline(modules)


def test_inserter_after_tracks_validates(tmp_path):
    """Correct order validates cleanly (no raise)."""
    modules = [
        Jets(fields=_JET_FIELDS),
        Tracks(fields=_TRACK_FIELDS),
        TruthHadronInserter(hadron_fields=_HADRON_FIELDS),
        H5Writer(path=str(tmp_path / "x.h5")),
    ]
    pipe = Pipeline(modules)
    assert len(pipe.modules) == 4


def test_field_level_requires_satisfied_by_group_producer(tmp_path):
    """A field-level ``requires`` (``tracks.valid``) is satisfied by the group
    producer (``Tracks`` produces ``tracks`` + ``tracks.valid``)."""

    class _NeedsValid(H5Writer):
        def __init__(self, path):
            super().__init__(path)
            self.requires = ["tracks.valid"]

    modules = [
        Tracks(fields=_TRACK_FIELDS),
        _NeedsValid(path=str(tmp_path / "x.h5")),
    ]
    Pipeline(modules)  # no raise


def test_field_level_requires_unmet_raises(tmp_path):
    class _NeedsField(H5Writer):
        def __init__(self, path):
            super().__init__(path)
            self.requires = ["tracks.ftagTruthParentBarcode"]

    modules = [
        Tracks(fields=_TRACK_FIELDS),  # produces tracks + tracks.valid, NOT the link
        _NeedsField(path=str(tmp_path / "x.h5")),
    ]
    with pytest.raises(RecipeError, match="ftagTruthParentBarcode"):
        Pipeline(modules)


def test_duplicate_producer_raises(tmp_path):
    """Two modules producing the same group key is rejected."""
    modules = [
        Constituents(name="tracks", max_items=5, fields=_TRACK_FIELDS),
        Constituents(name="tracks", max_items=5, fields=_TRACK_FIELDS),
        H5Writer(path=str(tmp_path / "x.h5")),
    ]
    with pytest.raises(RecipeError, match="produced by both"):
        Pipeline(modules)


def test_undeclared_mutation_raises(tmp_path):
    """A module mutating a key that no earlier module produced is rejected."""

    class _BadMutator(Jets):
        def __init__(self, fields):
            super().__init__(fields=fields)
            self.mutates = ["tracks"]  # nothing produced "tracks" yet

    modules = [
        _BadMutator(fields=_JET_FIELDS),
        H5Writer(path=str(tmp_path / "x.h5")),
    ]
    with pytest.raises(RecipeError, match="mutates 'tracks'"):
        Pipeline(modules)


def test_self_referential_require_raises(tmp_path):
    class _SelfDep(Jets):
        def __init__(self, fields):
            super().__init__(fields=fields)
            self.requires = ["jets"]  # requires what it also produces

    modules = [
        _SelfDep(fields=_JET_FIELDS),
        H5Writer(path=str(tmp_path / "x.h5")),
    ]
    with pytest.raises(RecipeError, match="requires and produces"):
        Pipeline(modules)


def test_no_terminal_writer_raises():
    modules = [Jets(fields=_JET_FIELDS)]
    with pytest.raises(RecipeError, match="no terminal writer"):
        Pipeline(modules)


# --------------------------------------------------------------------------- #
# run() threads the nested dict + injects pipeline-level n_samples
# --------------------------------------------------------------------------- #
def test_run_threads_dict_and_injects_n_samples(tmp_path):
    modules = [
        Jets(fields=_JET_FIELDS),
        Tracks(fields=_TRACK_FIELDS),
        H5Writer(path=str(tmp_path / "out.h5")),
    ]
    pipe = Pipeline(modules, seed=1, n_samples=37)
    data = pipe.run()
    assert data["jets"].shape == (37,)
    assert data["tracks"].shape[0] == 37
    assert "valid" in data["tracks"].dtype.names
    assert (tmp_path / "out.h5").exists()


def test_run_is_deterministic():
    def _data():
        pipe = Pipeline(
            [Jets(fields=_JET_FIELDS), Tracks(fields=_TRACK_FIELDS),
             H5Writer(path="/dev/null")],
            seed=5, n_samples=20,
        )
        return pipe.run()

    # H5Writer to /dev/null would fail; instead drop the writer for determinism
    pipe_a = Pipeline(
        [Jets(fields=_JET_FIELDS), Tracks(fields=_TRACK_FIELDS),
         NormWriter(path="/dev/null")],
        seed=5, n_samples=20,
    )
    pipe_b = Pipeline(
        [Jets(fields=_JET_FIELDS), Tracks(fields=_TRACK_FIELDS),
         NormWriter(path="/dev/null")],
        seed=5, n_samples=20,
    )
    # only compare the producer outputs (don't run the writer's IO)
    rng_a = np.random.default_rng(5)
    da = {}
    for m in pipe_a.modules[:2]:
        m.n_samples = 20
        da = m(da, rng_a)
    rng_b = np.random.default_rng(5)
    db = {}
    for m in pipe_b.modules[:2]:
        m.n_samples = 20
        db = m(db, rng_b)
    for k in da:
        assert np.array_equal(da[k].view(np.uint8), db[k].view(np.uint8))


# --------------------------------------------------------------------------- #
# jsonargparse load + end-to-end run (the done-criterion)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "recipe_name",
    [
        "flavour_tagger",
        "maskformer_truth_hadron",
        "truth_hadron_regression",
        "flavour_tagger_charged_neutral",
        "lepton_tagger",
        "event_objects",
        "jets_only",
        "single_constituent_regression",
    ],
)
def test_load_pipeline_runs_end_to_end(tmp_path, recipe_name):
    """A recipe loads via jsonargparse (class_path/init_args) and runs end-to-end
    to a written HDF5 file."""
    pipe = load_pipeline(str(_RECIPES_DIR / f"{recipe_name}.yaml"))
    pipe.set_output_dir(tmp_path)
    data = pipe.run()
    h5 = tmp_path / f"{recipe_name}.h5"
    assert h5.exists(), f"{recipe_name}: H5 not written"
    with h5py.File(h5, "r") as f:
        # every produced group is a top-level dataset
        for g in data:
            assert g in f, f"{recipe_name}: group {g} missing from H5"
        assert f.attrs["unique_jets"] > 0


def test_load_pipeline_modules_are_an_ordered_list():
    """The recipe wires an ordered list (not a name-keyed dict)."""
    pipe = load_pipeline(str(_RECIPES_DIR / "flavour_tagger.yaml"))
    assert isinstance(pipe.modules, list)
    assert type(pipe.modules[0]).__name__ == "Jets"
    assert type(pipe.modules[-1]).__name__ == "ClassDictWriter"


def test_maskformer_link_corruption_free(tmp_path):
    """Inserter done-criterion: every valid track's ftagTruthParentBarcode is
    in the SAME jet's valid hadron barcodes (or the -1 unmatched sentinel),
    with zero cross-jet leakage and no -1 masquerading as a valid hadron id."""
    pipe = load_pipeline(str(_RECIPES_DIR / "maskformer_truth_hadron.yaml"))
    pipe.set_output_dir(tmp_path)
    data = pipe.run()
    had = data["truth_hadrons"]
    trk = data["tracks"]
    assert "ftagTruthParentBarcode" in trk.dtype.names
    assert np.dtype(trk.dtype["ftagTruthParentBarcode"]).kind == "i"  # i4, not f4
    n = had.shape[0]
    for i in range(n):
        valid_bc = had["barcode"][i][had["valid"][i]]
        valid_bc_set = set(valid_bc.tolist())
        # no valid hadron carries the -1 fill sentinel as its id
        assert -1 not in valid_bc_set, f"jet {i}: valid hadron with id -1"
        tvalid = trk["valid"][i]
        for pb in trk["ftagTruthParentBarcode"][i][tvalid].tolist():
            assert pb in valid_bc_set or pb == -1, (
                f"jet {i}: link {pb} not a same-jet valid hadron barcode nor -1"
            )
        # invalid track slots are the -1 fill
        assert np.all(trk["ftagTruthParentBarcode"][i][~tvalid] == -1)
