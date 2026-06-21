"""CPU-only reader round-trip: recipe -> Pipeline.run -> H5Writer -> SaltDataset.

Parametrized over the 8 confirmed recipes. Uses the REAL legacy reader
``salt.data.SaltDataset`` (the captured API) to prove ``H5Writer``'s on-disk
layout matches a real reader: declared variables read back with correct
shapes/dtypes, constituent pad masks bool with invalid slots zeroed, global
features finite. A second labelled block (design §4.3) asserts every label head
round-trips as ``torch.long``, including the explicit MaskFormer
``truth_hadrons.flavour`` (classes [-1, 4, 5], invalid_fill -1) assertion.

No GPU, no training, no datamodule/H5StructuredReader.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from salt.core.testing.datagen.pipeline import load_pipeline

SaltDataset = pytest.importorskip("salt.data").SaltDataset

_RECIPES_DIR = Path(__file__).resolve().parent.parent / "recipes"


def _run_recipe(tmp_path, recipe_name):
    pipe = load_pipeline(str(_RECIPES_DIR / f"{recipe_name}.yaml"))
    pipe.set_output_dir(tmp_path)
    pipe.run()
    return pipe, tmp_path / f"{recipe_name}.h5"


# (recipe, variables, global_object)
RECIPES = [
    ("flavour_tagger", {"jets": ["pt", "eta"], "tracks": ["d0", "z0SinTheta"]}, "jets"),
    ("maskformer_truth_hadron",
     {"jets": ["pt"], "tracks": ["d0"], "truth_hadrons": ["pt", "Lxy"]}, "jets"),
    ("truth_hadron_regression",
     {"jets": ["pt"], "tracks": ["d0"], "truth_hadrons": ["pt"]}, "jets"),
    ("flavour_tagger_charged_neutral",
     {"jets": ["pt"], "charged": ["pt"], "neutral": ["pt"]}, "jets"),
    ("lepton_tagger", {"electrons": ["pt"], "electron_tracks": ["pt"]}, "electrons"),
    ("event_objects", {"events": ["mBB"], "objects": ["pt", "eta"]}, "events"),
    ("jets_only", {"jets": ["pt", "eta"]}, "jets"),
    ("single_constituent_regression", {"jets": ["pt"], "super_tracks": ["pt"]}, "jets"),
]


@pytest.mark.parametrize("recipe_name, variables, global_object", RECIPES)
def test_reader_roundtrip(tmp_path, recipe_name, variables, global_object):
    pipe, h5 = _run_recipe(tmp_path, recipe_name)

    ds = SaltDataset(
        filename=str(h5),
        norm_dict={},
        variables=variables,
        stage="train",
        global_object=global_object,
    )

    assert len(ds) == pipe.n_samples  # __len__ == rows in global group

    B = 50
    for i in range(0, len(ds), B):
        inputs, pad_masks, labels = ds[i:i + B]  # slice, never int
        b = min(B, len(ds) - i)

        # global: (b, F) float32, finite, no pad mask
        gfeat = variables[global_object]
        assert inputs[global_object].shape == (b, len(gfeat))
        assert inputs[global_object].dtype == torch.float32
        assert torch.isfinite(inputs[global_object]).all()
        assert global_object not in pad_masks

        # constituents: (b, M, F); mask (b, M) bool; invalid slots zeroed
        for g, feats in variables.items():
            if g == global_object:
                continue
            assert inputs[g].shape[0] == b
            assert inputs[g].shape[2] == len(feats)
            assert inputs[g].dtype == torch.float32
            assert pad_masks[g].shape == (b, inputs[g].shape[1])
            assert pad_masks[g].dtype == torch.bool
            assert (inputs[g][pad_masks[g]] == 0).all()


# --------------------------------------------------------------------------- #
# Labelled round-trip (design §4.3): every label head -> torch.long
# --------------------------------------------------------------------------- #
LABELLED = [
    ("flavour_tagger", {"jets": ["flavour_label"]}, "jets"),
    ("jets_only", {"jets": ["flavour_label"]}, "jets"),
    ("flavour_tagger_charged_neutral", {"jets": ["flavour_label"]}, "jets"),
    ("truth_hadron_regression", {"jets": ["flavour_label"]}, "jets"),
    ("lepton_tagger", {"electrons": ["iffClass"]}, "electrons"),
    ("maskformer_truth_hadron",
     {"jets": ["flavour_label"], "truth_hadrons": ["flavour"]}, "jets"),
]


@pytest.mark.parametrize("recipe_name, labels, global_object", LABELLED)
def test_label_roundtrip(tmp_path, recipe_name, labels, global_object):
    _, h5 = _run_recipe(tmp_path, recipe_name)

    ds = SaltDataset(
        filename=str(h5),
        norm_dict={},
        variables={g: [] for g in labels},  # labels need their group present
        labels=labels,
        stage="train",
        global_object=global_object,
    )
    _, _, lbls = ds[0:50]
    for g, names in labels.items():
        for n in names:
            assert lbls[g][n].dtype == torch.long, f"{recipe_name}: {g}.{n} not long"

    # EXPLICIT maskformer object-class assertion (fix #6).
    if recipe_name == "maskformer_truth_hadron":
        flav = lbls["truth_hadrons"]["flavour"]  # classes [-1, 4, 5], invalid_fill -1
        assert flav.dtype == torch.long
        uniq = set(flav.unique().tolist())
        assert uniq <= {-1, 4, 5}, f"unexpected flavour classes {uniq}"
        assert -1 in uniq, "invalid_fill -1 not preserved on invalid hadron slots"
