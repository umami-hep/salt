"""Tests for salt.graph.bundle."""

import numpy as np
import pytest
import torch

from salt.graph.bundle import Bundle
from salt.graph.errors import DeclarationError, GraphError, KeyCollisionError

# construction


class TestConstruction:
    def test_empty(self):
        b = Bundle()
        assert b.keys() == []
        assert len(b) == 0
        assert b.data == {}

    def test_adopt_nested_dict(self):
        x = torch.zeros(2, 3)
        mask = torch.ones(2, 4, dtype=torch.bool)
        b = Bundle({"inputs": {"jets": x}, "masks": {"tracks": mask}})
        assert b.keys() == ["inputs.jets", "masks.tracks"]
        assert b.get("inputs.jets") is x
        assert b.get("masks.tracks") is mask

    def test_adopt_rejects_dotted_keys(self):
        with pytest.raises(ValueError, match="must not contain"):
            Bundle({"inputs.jets": torch.zeros(1)})

    def test_adopt_rejects_non_string_keys(self):
        with pytest.raises(TypeError, match="key component"):
            Bundle({1: torch.zeros(1)})

    def test_adopt_empty_subtree_contributes_no_keys(self):
        b = Bundle({"inputs": {}})
        assert b.keys() == []

    def test_payload_stays_plain(self):
        b = Bundle({"inputs": {"jets": torch.zeros(1)}})
        b.set("embed.jets", torch.zeros(1))
        assert type(b.data) is dict
        assert type(b.data["inputs"]) is dict
        assert type(b.data["embed"]) is dict
        assert isinstance(b.data["inputs"]["jets"], torch.Tensor)

    def test_numpy_payloads(self):
        arr = np.zeros((2, 3), dtype=np.float32)
        b = Bundle({"raw": {"jets": arr}})
        assert b.get("raw.jets") is arr


# get / set / contains


class TestGetSet:
    def test_set_get_round_trip(self):
        b = Bundle()
        x = torch.randn(4, 8)
        b.set("embed.jets", x)
        assert b.get("embed.jets") is x
        assert "embed.jets" in b
        assert "embed" not in b  # subtree roots are not leaves

    def test_set_builds_nested_structure(self):
        b = Bundle()
        b.set("preds.jets.cls", torch.zeros(1))
        assert list(b.data["preds"]["jets"]) == ["cls"]

    def test_get_missing_key(self):
        b = Bundle()
        with pytest.raises(KeyError, match="'nope.missing' not found"):
            b.get("nope.missing")

    def test_get_subtree_node_is_an_error(self):
        b = Bundle()
        b.set("preds.jets.cls", torch.zeros(1))
        with pytest.raises(KeyError, match="is a subtree"):
            b.get("preds.jets")

    def test_get_descending_through_leaf(self):
        b = Bundle()
        b.set("encoded.seq", torch.zeros(1))
        with pytest.raises(KeyError, match="'encoded.seq' is a leaf"):
            b.get("encoded.seq.deeper")

    def test_get_invalid_key(self):
        b = Bundle()
        with pytest.raises(ValueError, match="empty component"):
            b.get("a..b")
        with pytest.raises(ValueError, match="non-empty"):
            b.get("")

    def test_set_invalid_key(self):
        b = Bundle()
        with pytest.raises(ValueError):
            b.set("", torch.zeros(1))
        with pytest.raises(ValueError):
            b.set("a..b", torch.zeros(1))

    def test_contains_validates_key(self):
        # malformed probe keys must fail loudly (mirroring get), not read as a
        # permanent absence — and must match debug-view behaviour
        b = Bundle()
        with pytest.raises(ValueError, match="non-empty"):
            "" in b  # noqa: B015 - the membership test itself is the assertion
        with pytest.raises(ValueError, match="empty component"):
            "a..b" in b  # noqa: B015

    def test_keys_insertion_order(self):
        b = Bundle()
        b.set("z.last", 1)
        b.set("a.first", 2)
        b.set("z.other", 3)
        assert b.keys() == ["z.last", "a.first", "z.other"]

    def test_dict_valued_leaf(self):
        # e.g. the seq.layout meta leaf — a plain dict stored opaquely
        layout = {"jets": slice(0, 1), "tracks": slice(1, 41)}
        b = Bundle()
        b.set("seq.layout", layout)
        assert b.get("seq.layout") is layout
        assert b.keys() == ["seq.layout"]
        # the dict-leaf blocks descent like any other leaf
        with pytest.raises(KeyCollisionError, match="'seq.layout' is already a leaf"):
            b.set("seq.layout.jets", slice(0, 1))
        with pytest.raises(KeyError, match="is a leaf"):
            b.get("seq.layout.jets")


# write-once collisions


class TestWriteOnce:
    def test_set_twice_collides(self):
        b = Bundle()
        b.set("embed.jets", torch.zeros(1))
        with pytest.raises(KeyCollisionError, match="'embed.jets' already exists"):
            b.set("embed.jets", torch.ones(1))
        # original value untouched
        assert torch.all(b.get("embed.jets") == 0)

    def test_set_onto_existing_subtree_collides(self):
        b = Bundle()
        b.set("preds.jets.cls", torch.zeros(1))
        with pytest.raises(KeyCollisionError, match="already exists as a subtree"):
            b.set("preds.jets", torch.zeros(1))
        with pytest.raises(KeyCollisionError, match="already exists as a subtree"):
            b.set("preds", torch.zeros(1))

    def test_set_under_existing_leaf_collides(self):
        b = Bundle()
        b.set("encoded.seq", torch.zeros(1))
        with pytest.raises(KeyCollisionError, match="'encoded.seq' is already a leaf"):
            b.set("encoded.seq.sub.key", torch.zeros(1))

    def test_collision_on_adopted_keys(self):
        b = Bundle({"inputs": {"jets": torch.zeros(1)}})
        with pytest.raises(KeyCollisionError):
            b.set("inputs.jets", torch.zeros(1))

    def test_key_collision_error_is_graph_error(self):
        assert issubclass(KeyCollisionError, GraphError)
        assert issubclass(DeclarationError, GraphError)


# subtree


class TestSubtree:
    def make(self):
        b = Bundle()
        b.set("preds.jets.cls", torch.zeros(2))
        b.set("preds.tracks.origin", torch.zeros(2, 40))
        b.set("embed.jets", torch.zeros(2, 16))
        return b

    def test_subtree_contents(self):
        b = self.make()
        sub = b.subtree("preds")
        assert set(sub) == {"jets", "tracks"}
        assert sub["jets"]["cls"] is b.get("preds.jets.cls")
        assert sub["tracks"]["origin"] is b.get("preds.tracks.origin")

    def test_subtree_nested_prefix(self):
        b = self.make()
        assert list(b.subtree("preds.jets")) == ["cls"]

    def test_subtree_isolation(self):
        # mutating the returned dict must not bypass write-once semantics
        b = self.make()
        sub = b.subtree("preds")
        sub["new_key"] = torch.zeros(1)
        sub["jets"]["smuggled"] = torch.zeros(1)
        sub["jets"].clear()
        assert b.keys() == ["preds.jets.cls", "preds.tracks.origin", "embed.jets"]
        assert "preds.new_key" not in b
        assert "preds.jets.smuggled" not in b
        assert b.get("preds.jets.cls") is not None
        assert "smuggled" not in b.data["preds"]["jets"]

    def test_subtree_leaves_shared(self):
        b = self.make()
        assert b.subtree("preds")["jets"]["cls"] is b.get("preds.jets.cls")

    def test_subtree_missing(self):
        b = self.make()
        with pytest.raises(KeyError, match="'nope' not found"):
            b.subtree("nope")

    def test_subtree_of_leaf_is_an_error(self):
        b = self.make()
        with pytest.raises(KeyError, match="is a leaf"):
            b.subtree("embed.jets")

    def test_subtree_invalid_prefix(self):
        b = self.make()
        with pytest.raises(ValueError):
            b.subtree("a..b")


# merge (executor path: write-once + declaration check)


class TestMerge:
    def test_merge_happy_path(self):
        b = Bundle()
        x = torch.randn(2, 16)
        y = torch.randn(2, 40, 8)
        produced = {"embed": {"jets": x, "tracks": y}}
        b.merge(produced, who="init_nets", expected={"embed.jets", "embed.tracks"})
        assert b.get("embed.jets") is x
        assert b.get("embed.tracks") is y
        assert b.keys() == ["embed.jets", "embed.tracks"]

    def test_merge_empty(self):
        b = Bundle()
        b.merge({}, who="noop", expected=set())
        assert b.keys() == []

    def test_merge_missing_key(self):
        b = Bundle()
        with pytest.raises(
            DeclarationError,
            match=r"module 'enc' .* missing=\['encoded.tracks'\] unexpected=\[\]",
        ):
            b.merge(
                {"encoded": {"seq": torch.zeros(1)}},
                who="enc",
                expected={"encoded.seq", "encoded.tracks"},
            )
        assert b.keys() == []  # nothing written

    def test_merge_extra_key(self):
        b = Bundle()
        with pytest.raises(
            DeclarationError,
            match=r"module 'enc' .* missing=\[\] unexpected=\['encoded.extra'\]",
        ):
            b.merge(
                {"encoded": {"seq": torch.zeros(1), "extra": torch.zeros(1)}},
                who="enc",
                expected={"encoded.seq"},
            )
        assert b.keys() == []

    def test_merge_extra_and_missing(self):
        b = Bundle()
        with pytest.raises(DeclarationError, match="missing=.*unexpected="):
            b.merge({"a": {"wrong": 1}}, who="m", expected={"a.right"})

    def test_merge_undeclared_nested_subtree_reported_per_leaf(self):
        b = Bundle()
        with pytest.raises(
            DeclarationError, match=r"unexpected=\['junk.deep.x', 'junk.deep.y'\]"
        ):
            b.merge(
                {"a": torch.zeros(1), "junk": {"deep": {"x": 1, "y": 2}}},
                who="m",
                expected={"a"},
            )

    def test_merge_empty_dict_value_is_unexpected(self):
        b = Bundle()
        with pytest.raises(DeclarationError, match=r"unexpected=\['junk'\]"):
            b.merge({"a": torch.zeros(1), "junk": {}}, who="m", expected={"a"})

    def test_merge_collision_with_existing_key(self):
        b = Bundle()
        b.set("embed.jets", torch.zeros(1))
        with pytest.raises(
            KeyCollisionError, match=r"'embed.jets' already exists .*module 'init'"
        ):
            b.merge({"embed": {"jets": torch.ones(1)}}, who="init", expected={"embed.jets"})

    def test_merge_collision_is_atomic(self):
        # a collision on one key must leave ALL produced keys unwritten
        b = Bundle()
        b.set("embed.jets", torch.zeros(1))
        produced = {"embed": {"tracks": torch.zeros(1), "jets": torch.ones(1)}}
        with pytest.raises(KeyCollisionError):
            b.merge(produced, who="init", expected={"embed.tracks", "embed.jets"})
        assert b.keys() == ["embed.jets"]
        assert "embed.tracks" not in b

    def test_merge_collision_with_leaf_prefix(self):
        b = Bundle()
        b.set("encoded.seq", torch.zeros(1))
        with pytest.raises(KeyCollisionError, match="'encoded.seq' is already a leaf"):
            b.merge(
                {"encoded": {"seq": {"sub": torch.zeros(1)}}},
                who="enc",
                expected={"encoded.seq.sub"},
            )

    def test_merge_declared_dict_leaf_kept_whole(self):
        # a dict value whose dotted path is declared is a dict-valued leaf (seq.layout)
        layout = {"jets": slice(0, 1), "tracks": slice(1, 41)}
        b = Bundle()
        b.merge(
            {"seq": {"x": torch.zeros(2, 41, 16), "layout": layout}},
            who="concat",
            expected={"seq.x", "seq.layout"},
        )
        assert b.get("seq.layout") is layout
        assert b.keys() == ["seq.x", "seq.layout"]

    def test_merge_rejects_dotted_component(self):
        b = Bundle()
        with pytest.raises(ValueError, match="must not contain"):
            b.merge({"embed.jets": torch.zeros(1)}, who="m", expected={"embed.jets"})

    def test_merge_then_set_collision(self):
        b = Bundle()
        b.merge({"embed": {"jets": torch.zeros(1)}}, who="init", expected={"embed.jets"})
        with pytest.raises(KeyCollisionError):
            b.set("embed.jets", torch.zeros(1))

    def test_repr_lists_keys(self):
        b = Bundle()
        b.set("a.b", 1)
        assert "a.b" in repr(b)
