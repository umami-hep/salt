"""Tests for salt.graph.spec (design §2.2)."""

import dataclasses

import pytest

from salt.graph.spec import (
    IO,
    KINDS,
    PRIMARY_MODES,
    GraphModule,
    Mode,
    TensorSpec,
    check_key_component,
    flatten_spec,
    is_symbolic_dim,
    iter_spec_leaves,
    join_key,
    split_key,
    split_symbolic_dim,
    sym_dim,
    unflatten_spec,
)

# Mode flag algebra


class TestMode:
    def test_composites(self):
        assert Mode.TRAINING == Mode.FIT | Mode.VAL
        assert Mode.ALL == Mode.FIT | Mode.VAL | Mode.TEST | Mode.ONNX

    def test_membership(self):
        assert Mode.FIT in Mode.TRAINING
        assert Mode.VAL in Mode.TRAINING
        assert Mode.TEST not in Mode.TRAINING
        assert Mode.ONNX not in Mode.TRAINING
        for mode in PRIMARY_MODES:
            assert mode in Mode.ALL

    def test_intersection_truthiness(self):
        assert bool(Mode.TRAINING & Mode.FIT)
        assert not bool(Mode.TRAINING & Mode.TEST)
        assert not bool(Mode.TRAINING & Mode.ONNX)
        assert (Mode.TRAINING & Mode.ALL) == Mode.TRAINING

    def test_primary_modes_atomic_and_disjoint(self):
        assert len(PRIMARY_MODES) == 4
        for i, a in enumerate(PRIMARY_MODES):
            for b in PRIMARY_MODES[i + 1 :]:
                assert not (a & b)

    def test_iteration_decomposes_composites(self):
        assert list(Mode.TRAINING) == [Mode.FIT, Mode.VAL]
        assert list(Mode.ALL) == [Mode.FIT, Mode.VAL, Mode.TEST, Mode.ONNX]

    def test_union_and_difference(self):
        assert (Mode.TRAINING | Mode.TEST) == (Mode.FIT | Mode.VAL | Mode.TEST)
        assert (Mode.ALL & ~Mode.ONNX) == (Mode.FIT | Mode.VAL | Mode.TEST)
        assert Mode.ALL & ~Mode.ALL == Mode(0)


# TensorSpec


class TestTensorSpec:
    def test_defaults(self):
        spec = TensorSpec()
        assert spec.shape is None
        assert spec.dtype is None
        assert spec.kind == "data"
        assert spec.modes == Mode.ALL
        assert spec.optional is False
        assert spec.fields is None

    def test_full_construction(self):
        spec = TensorSpec(
            shape=("B", "T:tracks", 23),
            dtype="float32",
            kind="data",
            modes=Mode.TRAINING,
            optional=True,
            fields=tuple(f"f{i}" for i in range(23)),
        )
        assert spec.shape == ("B", "T:tracks", 23)
        assert spec.modes == Mode.TRAINING

    def test_frozen(self):
        spec = TensorSpec()
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.kind = "label"
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.shape = ("B",)

    def test_invalid_kind(self):
        with pytest.raises(ValueError, match="invalid kind"):
            TensorSpec(kind="mask")  # the kind is "pad_mask", not "mask"

    def test_all_kinds_accepted(self):
        for kind in KINDS:
            assert TensorSpec(kind=kind).kind == kind

    def test_invalid_symbolic_dim(self):
        for bad in ("T:", ":tracks", "T:a:b", ""):
            with pytest.raises(ValueError, match="symbolic dim"):
                TensorSpec(shape=("B", bad))

    def test_negative_concrete_dim(self):
        with pytest.raises(ValueError, match="non-negative"):
            TensorSpec(shape=("B", -1))

    def test_non_int_non_str_dim(self):
        with pytest.raises(TypeError, match="shape entries"):
            TensorSpec(shape=("B", 1.5))

    def test_shape_rejects_bare_string(self):
        # strings are iterable: shape="BT" would silently become ("B", "T")
        with pytest.raises(TypeError, match="shape must be a sequence"):
            TensorSpec(shape="BT")
        with pytest.raises(TypeError, match="shape must be a sequence"):
            TensorSpec(shape="B")

    def test_fields_rejects_bare_string(self):
        # fields="pt" would silently become ("p", "t")
        with pytest.raises(TypeError, match="fields must be a sequence"):
            TensorSpec(fields="pt")

    def test_list_inputs_coerced_to_tuples(self):
        spec = TensorSpec(shape=["B", 3], fields=["a", "b", "c"])
        assert spec.shape == ("B", 3)
        assert spec.fields == ("a", "b", "c")

    def test_fields_length_must_match_concrete_last_dim(self):
        with pytest.raises(ValueError, match="does not match"):
            TensorSpec(shape=("B", 3), fields=("a", "b"))

    def test_fields_with_symbolic_last_dim_ok(self):
        spec = TensorSpec(shape=("B", "F:tracks"), fields=("a", "b"))
        assert spec.fields == ("a", "b")

    def test_empty_modes_rejected(self):
        with pytest.raises(ValueError, match="modes"):
            TensorSpec(modes=Mode(0))

    def test_non_mode_modes_rejected(self):
        with pytest.raises(TypeError, match="Mode flag"):
            TensorSpec(modes=3)

    def test_active_in(self):
        spec = TensorSpec(modes=Mode.TRAINING)
        assert spec.active_in(Mode.FIT)
        assert spec.active_in(Mode.VAL)
        assert spec.active_in(Mode.TRAINING)
        assert not spec.active_in(Mode.TEST)
        assert not spec.active_in(Mode.ONNX)
        # overlap semantics: a TRAINING port is "active in" any composite containing FIT or VAL
        assert spec.active_in(Mode.ALL)

    def test_equality_and_hash(self):
        a = TensorSpec(shape=("B", 3), dtype="float32")
        b = TensorSpec(shape=("B", 3), dtype="float32")
        assert a == b
        assert hash(a) == hash(b)
        assert a != TensorSpec(shape=("B", 4), dtype="float32")


# IO


class TestIO:
    def test_defaults_empty(self):
        io = IO()
        assert io.requires == {}
        assert io.produces == {}

    def test_frozen(self):
        io = IO()
        with pytest.raises(dataclasses.FrozenInstanceError):
            io.requires = {}

    def test_validates_leaves_eagerly(self):
        with pytest.raises(TypeError, match="spec leaf"):
            IO(produces={"preds": {"jets": "not-a-spec"}})
        with pytest.raises(ValueError, match="key component"):
            IO(requires={"inputs.jets": TensorSpec()})

    def test_round_trip_through_flatten(self):
        io = IO(
            requires={"inputs": {"jets": TensorSpec(shape=("B", 2))}},
            produces={"embed": {"jets": TensorSpec(shape=("B", 16))}},
        )
        assert list(flatten_spec(io.requires)) == ["inputs.jets"]
        assert list(flatten_spec(io.produces)) == ["embed.jets"]


# GraphModule protocol


class TestGraphModule:
    def test_structural_conformance(self):
        class Toy:
            def __init__(self):
                self.name = "toy"

            def declare_io(self, mode):
                return IO()

        assert isinstance(Toy(), GraphModule)

    def test_non_conformance(self):
        class NotAModule:
            pass

        assert not isinstance(NotAModule(), GraphModule)


# flatten / unflatten / iter


def _nested():
    return {
        "inputs": {
            "jets": TensorSpec(shape=("B", 2), dtype="float32"),
            "tracks": TensorSpec(shape=("B", "T:tracks", 23), dtype="float32"),
        },
        "masks": {"tracks": TensorSpec(shape=("B", "T:tracks"), dtype="bool", kind="pad_mask")},
        "labels": {
            "jets": {"flavour_label": TensorSpec(shape=("B",), kind="label", modes=Mode.TRAINING)}
        },
    }


class TestFlattenUnflatten:
    def test_flatten_keys(self):
        flat = flatten_spec(_nested())
        assert list(flat) == [
            "inputs.jets",
            "inputs.tracks",
            "masks.tracks",
            "labels.jets.flavour_label",
        ]
        assert all(isinstance(v, TensorSpec) for v in flat.values())

    def test_round_trip(self):
        nested = _nested()
        assert unflatten_spec(flatten_spec(nested)) == nested

    def test_flat_round_trip(self):
        flat = flatten_spec(_nested())
        assert flatten_spec(unflatten_spec(flat)) == flat

    def test_flatten_empty(self):
        assert flatten_spec({}) == {}
        assert unflatten_spec({}) == {}

    def test_flatten_ignores_empty_subtrees(self):
        assert flatten_spec({"inputs": {}}) == {}

    def test_flatten_rejects_dotted_component(self):
        with pytest.raises(ValueError, match="key component"):
            flatten_spec({"inputs.jets": TensorSpec()})

    def test_flatten_rejects_non_spec_leaf(self):
        with pytest.raises(TypeError, match="spec leaf"):
            flatten_spec({"inputs": {"jets": 42}})

    def test_flatten_rejects_non_string_key(self):
        with pytest.raises(TypeError, match="key component"):
            flatten_spec({3: TensorSpec()})

    def test_iter_spec_leaves_is_lazy_and_ordered(self):
        it = iter_spec_leaves(_nested())
        key, spec = next(it)
        assert key == "inputs.jets"
        assert isinstance(spec, TensorSpec)
        assert [k for k, _ in it] == ["inputs.tracks", "masks.tracks", "labels.jets.flavour_label"]

    def test_unflatten_prefix_clash_leaf_first(self):
        flat = {"a.b": TensorSpec(), "a.b.c": TensorSpec()}
        with pytest.raises(ValueError, match="clashes with leaf 'a.b'"):
            unflatten_spec(flat)

    def test_unflatten_prefix_clash_subtree_first(self):
        flat = {"a.b.c": TensorSpec(), "a.b": TensorSpec()}
        with pytest.raises(ValueError, match="clashes with existing subtree"):
            unflatten_spec(flat)

    def test_unflatten_rejects_bad_keys(self):
        with pytest.raises(ValueError, match="empty component"):
            unflatten_spec({"a..b": TensorSpec()})

    def test_unflatten_rejects_non_spec_value(self):
        with pytest.raises(TypeError, match="must be TensorSpec"):
            unflatten_spec({"a.b": "nope"})


# dotted-key helpers


class TestKeyHelpers:
    def test_split_key(self):
        assert split_key("a") == ("a",)
        assert split_key("preds.jets.cls") == ("preds", "jets", "cls")

    @pytest.mark.parametrize("bad", ["", "a..b", ".a", "a.", "."])
    def test_split_key_invalid(self, bad):
        with pytest.raises(ValueError):
            split_key(bad)

    def test_split_key_non_string(self):
        with pytest.raises(TypeError, match="must be str"):
            split_key(7)

    def test_join_key_round_trip(self):
        for key in ("a", "preds.jets.cls"):
            assert join_key(split_key(key)) == key

    def test_join_key_rejects_dotted_component(self):
        with pytest.raises(ValueError, match="must not contain"):
            join_key(("a", "b.c"))

    def test_join_key_rejects_empty(self):
        with pytest.raises(ValueError, match="empty sequence"):
            join_key(())

    def test_check_key_component(self):
        assert check_key_component("jets") == "jets"
        with pytest.raises(ValueError):
            check_key_component("")
        with pytest.raises(ValueError):
            check_key_component("a.b")
        with pytest.raises(TypeError):
            check_key_component(None)


# symbolic dims


class TestSymbolicDims:
    def test_is_symbolic_dim(self):
        assert is_symbolic_dim("B")
        assert is_symbolic_dim("T:tracks")
        assert not is_symbolic_dim(23)

    def test_sym_dim(self):
        assert sym_dim("B") == "B"
        assert sym_dim("T", "tracks") == "T:tracks"
        assert sym_dim("F", "tracks") == "F:tracks"

    def test_sym_dim_invalid(self):
        with pytest.raises(ValueError):
            sym_dim("")
        with pytest.raises(ValueError):
            sym_dim("T:x")
        with pytest.raises(ValueError):
            sym_dim("T", "")
        with pytest.raises(ValueError):
            sym_dim("T", "a:b")

    def test_split_symbolic_dim(self):
        assert split_symbolic_dim("B") == ("B", None)
        assert split_symbolic_dim("T:tracks") == ("T", "tracks")

    def test_split_round_trip(self):
        for dim in ("B", "T:tracks", "F:electrons"):
            assert sym_dim(*split_symbolic_dim(dim)) == dim

    @pytest.mark.parametrize("bad", ["", "T:", ":tracks", "T:a:b"])
    def test_split_symbolic_dim_invalid(self, bad):
        with pytest.raises(ValueError):
            split_symbolic_dim(bad)
