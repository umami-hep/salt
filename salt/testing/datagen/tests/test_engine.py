"""Unit tests for the schema-driven test-data generation engine."""

from __future__ import annotations

import math

import numpy as np
import pytest

from salt.testing.datagen import (
    SchemaError,
    compute_class_dict,
    compute_norm_dict,
    generate_data,
    load_schema,
    parse_schema,
    write_h5,
)


def _maskformer_schema(n_samples=200, seed=7):
    return {
        "n_samples": n_samples,
        "seed": seed,
        "fill": {"float": math.nan, "int": -1},
        "file_attrs": {"config": "{}", "unique_jets_from": "jets"},
        "groups": [
            {
                "name": "jets",
                "kind": "global",
                "fields": [
                    {"name": "pt", "type": "distribution", "dtype": "f4"},
                    {
                        "name": "lxy",
                        "type": "distribution",
                        "dtype": "f4",
                        "nan_where": "flavour_label == 0",
                    },
                    {
                        "name": "flavour_label",
                        "type": "label",
                        "dtype": "i4",
                        "classes": [0, 1, 2, 3],
                        "sample_classes": [0, 1, 2],
                        "class_names": ["bjets", "cjets", "ujets"],
                    },
                ],
            },
            {
                "name": "truth_hadrons",
                "kind": "constituent",
                "max_items": 5,
                "valid_fraction": 0.5,
                "min_valid": 1,
                "fields": [
                    {"name": "pt", "type": "distribution", "dtype": "f4"},
                    {
                        "name": "barcode",
                        "type": "id",
                        "dtype": "i4",
                        "range": [0, 10000],
                        "scope": "jet",
                        "unique": True,
                    },
                ],
            },
            {
                "name": "tracks",
                "kind": "constituent",
                "max_items": 40,
                "valid_fraction": 0.5,
                "min_valid": 1,
                "fields": [
                    {"name": "d0", "type": "distribution", "dtype": "f4"},
                    {
                        "name": "ftagTruthOriginLabel",
                        "type": "label",
                        "dtype": "i4",
                        "classes": [0, 1, 2, 3, 4, 5, 6, 7],
                        "sample_classes": [0, 1, 2, 3],
                    },
                    {
                        "name": "ftagTruthTypeLabel",
                        "type": "label",
                        "dtype": "i4",
                        "classes": [-2, -3, 5, -5, 6, -6],
                        "invalid_fill": 0,
                    },
                    {
                        "name": "ftagTruthParentBarcode",
                        "type": "link",
                        "dtype": "i4",
                        "references": "truth_hadrons.barcode",
                        "scope": "jet",
                        "unmatched_fraction": 0.0,
                    },
                ],
            },
            {"name": "tracks_dr", "kind": "constituent", "alias_of": "tracks"},
        ],
    }


# --------------------------------------------------------------------------- #
def test_id_uniqueness_within_jet_scope():
    data = generate_data(_maskformer_schema())
    had = data["truth_hadrons"]
    for i in range(had.shape[0]):
        valid = had["valid"][i]
        bc = had["barcode"][i][valid]
        assert len(bc) == len(set(bc.tolist())), f"duplicate barcode in jet {i}"
        # invalid slots are -1
        assert np.all(had["barcode"][i][~valid] == -1)


def test_link_membership_in_valid_referenced_ids():
    data = generate_data(_maskformer_schema())
    had = data["truth_hadrons"]
    trk = data["tracks"]
    for i in range(trk.shape[0]):
        valid_bc = set(had["barcode"][i][had["valid"][i]].tolist())
        for t in range(trk.shape[1]):
            if not trk["valid"][i, t]:
                # invalid track -> -1
                assert trk["ftagTruthParentBarcode"][i, t] == -1
                continue
            pb = int(trk["ftagTruthParentBarcode"][i, t])
            # valid track: parent barcode in {valid hadron barcodes} or unmatched sentinel
            assert pb in valid_bc or pb == -1


def test_invalid_fill_float_nan_int_minus_one():
    data = generate_data(_maskformer_schema())
    trk = data["tracks"]
    invalid = ~trk["valid"]
    # float distribution -> NaN on invalid
    assert np.all(np.isnan(trk["d0"][invalid]))
    # int label -> -1 on invalid (default)
    assert np.all(trk["ftagTruthOriginLabel"][invalid] == -1)
    # ftagTruthTypeLabel overrides invalid fill -> 0
    assert np.all(trk["ftagTruthTypeLabel"][invalid] == 0)


def test_electrons_no_invalid_fill():
    schema = {
        "n_samples": 100,
        "seed": 1,
        "groups": [
            {
                "name": "electrons",
                "kind": "constituent",
                "max_items": 10,
                "valid_fraction": 0.5,
                "mask_invalid": False,
                "fields": [{"name": "pt", "type": "distribution", "dtype": "f4"}],
            }
        ],
    }
    data = generate_data(schema)
    el = data["electrons"]
    invalid = ~el["valid"]
    # mask_invalid False -> invalid slots keep (non-NaN) random values
    assert not np.any(np.isnan(el["pt"][invalid]))


def test_nan_where_label_driven():
    data = generate_data(_maskformer_schema())
    jets = data["jets"]
    light = jets["flavour_label"] == 0
    assert np.all(np.isnan(jets["lxy"][light]))
    assert not np.any(np.isnan(jets["lxy"][~light]))


def test_valid_mask_sorted_to_front():
    data = generate_data(_maskformer_schema())
    trk = data["tracks"]
    for i in range(trk.shape[0]):
        v = trk["valid"][i].astype(int)
        # valid (1) packed to front: non-increasing
        assert np.all(np.diff(v) <= 0)
        assert v[0] == 1  # min_valid: 1


def test_alias_is_byte_identical():
    data = generate_data(_maskformer_schema())
    assert np.array_equal(data["tracks"].view(np.uint8), data["tracks_dr"].view(np.uint8))


# --------------------------------------------------------------------------- #
def test_class_dict_length_equals_output_size():
    schema = _maskformer_schema()
    data = generate_data(schema)
    cd = compute_class_dict(data, schema)
    # flavour_label declared 4 classes -> length 4 (even though data spans 3)
    assert len(cd["jets"]["flavour_label"]) == 4
    # class 3 never drawn -> zero-count -> weight 0.0
    assert cd["jets"]["flavour_label"][3] == 0.0
    # ftagTruthOriginLabel declared 8 -> length 8
    assert len(cd["tracks"]["ftagTruthOriginLabel"]) == 8


def test_class_dict_is_gn3_flavour_length_6():
    # reuse example: flavour_label classes_by_flag is_gn3 -> 6
    schema = {
        "n_samples": 100,
        "seed": 2,
        "groups": [
            {
                "name": "jets",
                "kind": "global",
                "fields": [
                    {
                        "name": "flavour_label",
                        "type": "label",
                        "dtype": "i4",
                        "classes": [0, 1, 2, 3],
                        "sample_classes": [0, 1, 2],
                        "classes_by_flag": {"is_gn3": [0, 1, 2, 3, 4, 5]},
                    }
                ],
            }
        ],
    }
    data = generate_data(schema, flags={"is_gn3": True})
    cd = compute_class_dict(data, schema, flags={"is_gn3": True})
    assert len(cd["jets"]["flavour_label"]) == 6


def test_class_dict_excludes_invalid_fill():
    schema = _maskformer_schema()
    data = generate_data(schema)
    cd = compute_class_dict(data, schema)
    # all weights finite and non-negative
    for w in cd["tracks"]["ftagTruthOriginLabel"]:
        assert np.isfinite(w) and w >= 0.0


def test_norm_dict_skips_non_features_and_falls_back():
    schema = _maskformer_schema()
    data = generate_data(schema)
    nd = compute_norm_dict(data, schema)
    # labels / ids / links / valid are NOT in the norm dict
    assert "flavour_label" not in nd["jets"]
    assert "barcode" not in nd["truth_hadrons"]
    assert "ftagTruthParentBarcode" not in nd["tracks"]
    assert "valid" not in nd["tracks"]
    # distribution features ARE present, with finite, non-zero std
    assert "d0" in nd["tracks"]
    assert np.isfinite(nd["tracks"]["d0"]["mean"])
    assert nd["tracks"]["d0"]["std"] != 0.0


def test_norm_dict_zero_variance_fallback():
    schema = {
        "n_samples": 50,
        "seed": 3,
        "groups": [
            {
                "name": "jets",
                "kind": "global",
                "fields": [
                    {
                        "name": "const_feat",
                        "type": "distribution",
                        "dtype": "f4",
                        "dist": "constant",
                        "params": {"value": 7.0},
                    }
                ],
            }
        ],
    }
    data = generate_data(schema)
    nd = compute_norm_dict(data, schema)
    # zero variance -> loadable no-op
    assert nd["jets"]["const_feat"] == {"mean": 0.0, "std": 1.0}


def test_norm_dict_all_invalid_fallback():
    # all hadron pt invalid (valid_fraction tiny + no min_valid) -> some columns
    # still loadable; here we force an all-NaN column via a single-sample mask trick
    schema = {
        "n_samples": 1,
        "seed": 4,
        "groups": [
            {
                "name": "c",
                "kind": "constituent",
                "max_items": 4,
                "valid_fraction": 0.5,
                "min_valid": 0,
                "fields": [{"name": "x", "type": "distribution", "dtype": "f4"}],
            }
        ],
    }
    # try many seeds until we get an all-invalid sample; otherwise just check loadable
    for s in range(50):
        d = generate_data({**schema, "seed": s})
        nd = compute_norm_dict(d, schema)
        entry = nd["c"]["x"]
        assert np.isfinite(entry["mean"]) and entry["std"] != 0.0


def test_class_names_round_trip_through_write_h5(tmp_path):
    schema = _maskformer_schema()
    data = generate_data(schema)
    out = tmp_path / "dummy.h5"
    write_h5(data, out, schema=schema)
    import h5py

    with h5py.File(out, "r") as f:
        assert f.attrs["unique_jets"] == data["jets"].shape[0]
        assert f.attrs["config"] == "{}"
        names = [
            n.decode() if isinstance(n, bytes) else n for n in f["jets"].attrs["flavour_label"]
        ]
        assert names == ["bjets", "cjets", "ujets"]


def test_write_h5_flag_class_names(tmp_path):
    # make_xbb flag -> hbb/hcc/top/qcd attr
    schema = {
        "n_samples": 50,
        "seed": 5,
        "groups": [
            {
                "name": "jets",
                "kind": "global",
                "fields": [
                    {
                        "name": "flavour_label",
                        "type": "label",
                        "dtype": "i4",
                        "classes": [0, 1, 2, 3],
                        "sample_classes": [0, 1, 2],
                        "class_names": ["bjets", "cjets", "ujets"],
                        "classes_by_flag": {"make_xbb": [0, 1, 2, 3]},
                        "class_names_by_flag": {"make_xbb": ["hbb", "hcc", "top", "qcd"]},
                    }
                ],
            }
        ],
    }
    data = generate_data(schema, flags={"make_xbb": True})
    out = tmp_path / "xbb.h5"
    write_h5(data, out, schema=schema, attrs={"_flags": {"make_xbb": True}})
    import h5py

    with h5py.File(out, "r") as f:
        names = [
            n.decode() if isinstance(n, bytes) else n for n in f["jets"].attrs["flavour_label"]
        ]
        assert names == ["hbb", "hcc", "top", "qcd"]


# --------------------------------------------------------------------------- #
def test_schema_validation_rejects_bad_reference():
    bad = {
        "n_samples": 10,
        "groups": [
            {
                "name": "tracks",
                "kind": "constituent",
                "max_items": 5,
                "fields": [
                    {
                        "name": "link",
                        "type": "link",
                        "references": "nonexistent.barcode",
                    }
                ],
            }
        ],
    }
    with pytest.raises(SchemaError):
        parse_schema(bad)


def test_schema_validation_rejects_link_to_non_id():
    bad = {
        "n_samples": 10,
        "groups": [
            {
                "name": "hads",
                "kind": "constituent",
                "max_items": 5,
                "fields": [{"name": "flavour", "type": "label", "classes": [0, 1]}],
            },
            {
                "name": "tracks",
                "kind": "constituent",
                "max_items": 5,
                "fields": [{"name": "lk", "type": "link", "references": "hads.flavour"}],
            },
        ],
    }
    with pytest.raises(SchemaError):
        parse_schema(bad)


def test_schema_validation_rejects_bad_nan_where():
    bad = {
        "n_samples": 10,
        "groups": [
            {
                "name": "jets",
                "kind": "global",
                "fields": [
                    {
                        "name": "x",
                        "type": "distribution",
                        "nan_where": "flavour_label > 0",
                    },
                    {"name": "flavour_label", "type": "label", "classes": [0, 1]},
                ],
            }
        ],
    }
    with pytest.raises(SchemaError):
        parse_schema(bad)


def test_schema_validation_rejects_class_names_length_mismatch():
    bad = {
        "n_samples": 10,
        "groups": [
            {
                "name": "jets",
                "kind": "global",
                "fields": [
                    {
                        "name": "flavour_label",
                        "type": "label",
                        "classes": [0, 1, 2, 3],
                        "sample_classes": [0, 1, 2],
                        "class_names": ["a", "b"],  # should be 3
                    }
                ],
            }
        ],
    }
    with pytest.raises(SchemaError):
        parse_schema(bad)


def test_determinism_same_seed_same_bytes():
    schema = _maskformer_schema()
    d1 = generate_data(schema)
    d2 = generate_data(schema)
    for k in d1:
        assert np.array_equal(d1[k].view(np.uint8), d2[k].view(np.uint8))


def test_load_schema_yaml(tmp_path):
    import yaml

    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(_maskformer_schema()))
    s = load_schema(p)
    data = generate_data(s)
    assert "tracks" in data


# --------------------------------------------------------------------------- #
# required_match=True / select_over coverage
# --------------------------------------------------------------------------- #
def _required_match_schema(n_samples=2000, seed=11):
    """A schema where the referenced group would otherwise have 0 valid items in
    many samples, with a required_match link.
    """
    return {
        "n_samples": n_samples,
        "seed": seed,
        "fill": {"float": math.nan, "int": -1},
        "groups": [
            {
                "name": "truth_hadrons",
                "kind": "constituent",
                "max_items": 5,
                "valid_fraction": 0.05,
                "min_valid": 0,  # deliberately allows zero-valid samples
                "fields": [
                    {"name": "pt", "type": "distribution", "dtype": "f4"},
                    {
                        "name": "barcode",
                        "type": "id",
                        "dtype": "i4",
                        "range": [0, 10000],
                        "scope": "jet",
                        "unique": True,
                    },
                    {
                        "name": "flavour",
                        "type": "label",
                        "dtype": "i4",
                        "classes": [0, 1, 2, 3],
                    },
                ],
            },
            {
                "name": "tracks",
                "kind": "constituent",
                "max_items": 10,
                "valid_fraction": 0.5,
                "min_valid": 1,
                "fields": [
                    {"name": "d0", "type": "distribution", "dtype": "f4"},
                    {
                        "name": "parent_barcode",
                        "type": "link",
                        "dtype": "i4",
                        "references": "truth_hadrons.barcode",
                        "scope": "jet",
                        "unmatched_fraction": 0.0,
                        "required_match": True,
                    },
                ],
            },
        ],
    }


def test_required_match_propagates_min_valid_and_never_links_fill():
    """Every valid source link resolves to a real referent id, never the fill sentinel."""
    schema = _required_match_schema()
    fill_int = -1

    # schema resolution raises the referenced group's effective min_valid to >=1,
    # while leaving the source group's own min_valid untouched.
    resolved = parse_schema(schema)
    assert resolved.group("truth_hadrons").min_valid >= 1
    assert resolved.group("tracks").min_valid == 1

    data = generate_data(schema)
    had = data["truth_hadrons"]
    trk = data["tracks"]

    n = had.shape[0]
    for i in range(n):
        valid = had["valid"][i]
        valid_bc = had["barcode"][i][valid]

        # (1) every sample has >=1 valid referent
        assert valid.any(), f"sample {i}: no valid referent despite required_match"

        # (2) NO referent marked valid carries the fill sentinel as its id
        assert not np.any(valid_bc == fill_int), (
            f"sample {i}: a valid referent has barcode == fill ({fill_int})"
        )
        # its payload is genuine too (id is in-range; pt not NaN on valid slots)
        assert np.all((valid_bc >= 0) & (valid_bc < 10000))
        assert not np.any(np.isnan(had["pt"][i][valid]))

        # (3) every VALID source link resolves to a real valid referent id,
        # never fill / unmatched (unmatched_fraction == 0 here)
        valid_bc_set = set(valid_bc.tolist())
        tvalid = trk["valid"][i]
        links = trk["parent_barcode"][i][tvalid]
        for pb in links.tolist():
            assert pb != fill_int, f"sample {i}: valid link resolved to fill ({fill_int})"
            assert pb in valid_bc_set, f"sample {i}: link {pb} not a valid referent id"
        # invalid source slots are still the fill
        assert np.all(trk["parent_barcode"][i][~tvalid] == fill_int)


def test_select_over_all_slots_vs_valid_ids():
    """select_over=all_slots may draw from ANY referenced slot (incl. invalid, -1
    fill); the default valid_ids never points a valid link at an invalid slot.
    """
    fill_int = -1

    def _schema(select_over):
        return {
            "n_samples": 1500,
            "seed": 23,
            "fill": {"float": math.nan, "int": fill_int},
            "groups": [
                {
                    "name": "hadrons",
                    "kind": "constituent",
                    "max_items": 5,
                    "valid_fraction": 0.4,
                    "min_valid": 1,  # so valid_ids pool is never empty
                    "fields": [
                        {
                            "name": "barcode",
                            "type": "id",
                            "dtype": "i4",
                            "range": [0, 10000],
                            "scope": "jet",
                            "unique": True,
                        },
                    ],
                },
                {
                    "name": "tracks",
                    "kind": "constituent",
                    "max_items": 20,
                    "valid_fraction": 0.6,
                    "min_valid": 1,
                    "fields": [
                        {
                            "name": "parent",
                            "type": "link",
                            "dtype": "i4",
                            "references": "hadrons.barcode",
                            "scope": "jet",
                            "unmatched_fraction": 0.0,
                            "select_over": select_over,
                        },
                    ],
                },
            ],
        }

    # default valid_ids: every valid link points at a VALID referent slot,
    # never the invalid -1 fill.
    data_valid = generate_data(_schema("valid_ids"))
    had_v, trk_v = data_valid["hadrons"], data_valid["tracks"]
    saw_valid_link = False
    for i in range(had_v.shape[0]):
        valid_bc = set(had_v["barcode"][i][had_v["valid"][i]].tolist())
        tvalid = trk_v["valid"][i]
        for pb in trk_v["parent"][i][tvalid].tolist():
            saw_valid_link = True
            assert pb in valid_bc, "valid_ids link pointed at an invalid slot"
            assert pb != fill_int
    assert saw_valid_link  # the assertions above were actually exercised

    # all_slots: does not crash, and CAN draw an invalid slot member (the -1
    # fill). Over 1500 jets x ~12 valid tracks with ~60% invalid hadron slots,
    # at least one valid link lands on a -1 slot.
    data_all = generate_data(_schema("all_slots"))
    had_a, trk_a = data_all["hadrons"], data_all["tracks"]
    saw_fill_on_valid_link = False
    for i in range(had_a.shape[0]):
        all_bc = set(had_a["barcode"][i].tolist())  # all slots incl invalid (-1)
        tvalid = trk_a["valid"][i]
        for pb in trk_a["parent"][i][tvalid].tolist():
            assert pb in all_bc, "all_slots link drew an id from no referenced slot"
            if pb == fill_int:
                saw_fill_on_valid_link = True
    assert saw_fill_on_valid_link, (
        "all_slots should be able to draw an invalid (-1) referenced slot"
    )
