"""Generic coverage for the H5OutputSink ``object_groups`` capability.

Proves the seam has NO per-consumer (MaskFormer) knowledge: a synthetic
non-MaskFormer group ("particles") declares fields sourcing arbitrary bundle
leaves and the sink sizes/packs it correctly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import Mode, flatten_spec
from salt.outputs.h5_sink import H5OutputSink
from salt.outputs.output_schema import ObjectGroup, ObjectGroupField, OutputColumn

pytestmark = pytest.mark.cpu_always

_STREAMS = ("jets", "tracks")
_SEQ_STREAMS = ("tracks",)
_TOTAL = 123
_P = 3   # the synthetic object axis (particles)
_T = 40  # the tracks file token length


# a deliberately NON-MaskFormer group: "particles" [B, P] with a 2-class prob
# field + a truth label, plus a per-token "tracks" index column. None of these
# names appear in the sink — genericity by construction.
def _particle_groups() -> list[ObjectGroup]:
    return [
        ObjectGroup(
            name="particles",
            shape=[_P],
            fields=[
                ObjectGroupField(leaf="net.particle_probs", suffixes=["pa", "pb"], dtype="f4"),
                ObjectGroupField(
                    leaf="labels.particles.truth", suffixes=["truth"], dtype="i8",
                    prefix=False, kind="label",
                ),
            ],
        ),
        ObjectGroup(
            name="particle_masks",
            shape=[_P, "tracks"],
            fields=[
                ObjectGroupField(
                    leaf="net.particle_masks", suffixes=["logit"], dtype="f4", prefix=False
                ),
            ],
        ),
        ObjectGroup(
            name="tracks",
            fields=[
                ObjectGroupField(leaf="outputs.tracks.owner", suffixes=["Owner"], dtype="i8"),
            ],
        ),
    ]


def _seeded(sink: H5OutputSink) -> H5OutputSink:
    """Seed one task column directly (the explicit-``outputs`` surface was retired)."""
    sink._columns = (OutputColumn(key="outputs.jets.cls", suffixes=["pb"]),)  # noqa: SLF001
    sink._columns_resolved = True  # noqa: SLF001
    return sink


def _sink(groups=None, *, half=False) -> H5OutputSink:
    sink = _seeded(H5OutputSink(object_groups=groups or _particle_groups(), half_precision=half))
    sink._run_name = "salt"  # noqa: SLF001
    sink._seq_lengths = {"tracks": _T}  # noqa: SLF001
    return sink


def _open(sink: H5OutputSink) -> H5OutputSink:
    """Resolve object shapes as open_schema would."""
    sink._object_shapes = sink._resolve_object_shapes(_STREAMS)  # noqa: SLF001
    return sink


def _batch(batch_size=4, n_tracks=_T):
    gen = torch.Generator().manual_seed(7)
    b = Bundle()
    b.set("net.particle_probs", torch.rand(batch_size, _P, 2, generator=gen))
    b.set("labels.particles.truth", torch.randint(0, 2, (batch_size, _P), generator=gen))
    b.set("net.particle_masks", torch.randn(batch_size, _P, n_tracks, generator=gen))
    b.set("outputs.tracks.owner", torch.randint(-2, _P, (batch_size, n_tracks), generator=gen))
    return b


class TestObjectGroupFieldNaming:
    def test_suffix_defaults_to_leaf_terminal_segment(self):
        """Single-source naming: omitting suffixes defaults the one column to the leaf terminal."""
        field = ObjectGroupField(leaf="outputs.tracks.HadronIndex", dtype="i8")
        assert field.suffixes == ("HadronIndex",)
        assert field.column_names("GN3") == ["GN3_HadronIndex"]

    def test_explicit_suffixes_still_honoured(self):
        field = ObjectGroupField(leaf="objects.class_probs", suffixes=["pb", "pc", "pnull"])
        assert list(field.suffixes) == ["pb", "pc", "pnull"]


class TestResolveObjectShapes:
    def test_non_reader_and_reader_groups(self):
        sink = _open(_sink())
        # non-reader groups carry a trailing shape (stream token resolved)
        assert sink._object_shapes == {"particles": (_P,), "particle_masks": (_P, _T)}  # noqa: SLF001
        # trailing dims are plain python ints
        for trailing in sink._object_shapes.values():  # noqa: SLF001
            assert all(type(d) is int for d in trailing)

    def test_empty_is_noop(self):
        sink = _seeded(H5OutputSink())
        sink._seq_lengths = {"tracks": _T}  # noqa: SLF001
        assert sink._resolve_object_shapes(_STREAMS) == {}  # noqa: SLF001

    def test_non_reader_shadowing_reader_stream_raises(self):
        sink = _sink([ObjectGroup(name="jets", shape=[_P],
                                  fields=[ObjectGroupField(leaf="x.y", suffixes=["z"])])])
        with pytest.raises(ConfigError, match="shadows reader stream"):
            sink._resolve_object_shapes(_STREAMS)  # noqa: SLF001

    def test_duplicate_group_name_raises(self):
        g = ObjectGroup(name="particles", shape=[_P],
                        fields=[ObjectGroupField(leaf="x.y", suffixes=["z"])])
        sink = _sink([g, g])
        with pytest.raises(ConfigError, match="declared twice"):
            sink._resolve_object_shapes(_STREAMS)  # noqa: SLF001

    def test_reader_stream_group_naming_non_stream_raises(self):
        sink = _sink([ObjectGroup(name="ghost",
                                  fields=[ObjectGroupField(leaf="x.y", suffixes=["z"])])])
        with pytest.raises(ConfigError, match="not a reader stream"):
            sink._resolve_object_shapes(_STREAMS)  # noqa: SLF001

    def test_unknown_shape_token_raises(self):
        sink = _sink([ObjectGroup(name="p", shape=[_P, "nope"],
                                  fields=[ObjectGroupField(leaf="x.y", suffixes=["z"])])])
        with pytest.raises(ConfigError, match="not a sequence stream"):
            sink._resolve_object_shapes(_STREAMS)  # noqa: SLF001


class TestGroupShape:
    def test_object_and_reader_shapes(self):
        sink = _open(_sink())
        gs = sink._group_shape  # noqa: SLF001
        assert gs("particles", _SEQ_STREAMS, _TOTAL) == (_TOTAL, _P)
        assert gs("particle_masks", _SEQ_STREAMS, _TOTAL) == (_TOTAL, _P, _T)
        # reader streams keep the prior inline rule
        assert gs("tracks", _SEQ_STREAMS, _TOTAL) == (_TOTAL, _T)
        assert gs("jets", _SEQ_STREAMS, _TOTAL) == (_TOTAL,)


class TestMergeColumns:
    def test_object_dtypes_and_shapes_appended(self):
        sink = _open(_sink())
        dtypes, shapes = sink._merge_columns(  # noqa: SLF001
            _STREAMS, _SEQ_STREAMS, {"jets": "jets", "tracks": "tracks"}, _TOTAL
        )
        # the task column still lands in its reader group
        assert "salt_pb" in dtypes["jets"].names
        # the object groups carry the field-declared columns + shapes
        assert dtypes["particles"].names == ("salt_pa", "salt_pb", "truth")
        assert dtypes["particle_masks"].names == ("logit",)
        # the reader-stream group appends its column to tracks (after task/mask)
        assert "salt_Owner" in dtypes["tracks"].names
        assert shapes["particles"] == (_TOTAL, _P)
        assert shapes["particle_masks"] == (_TOTAL, _P, _T)

    def test_object_column_colliding_with_task_column_raises(self):
        # a field minting 'salt_pb' into the reader stream 'jets' collides with
        # the seeded task column via the shared _add owners map.
        groups = [ObjectGroup(name="jets",
                              fields=[ObjectGroupField(leaf="x.y", suffixes=["pb"])])]
        sink = _open(_sink(groups))
        with pytest.raises(ConfigError, match="is declared by"):
            sink._merge_columns(  # noqa: SLF001
                _STREAMS, _SEQ_STREAMS, {"jets": "jets", "tracks": "tracks"}, _TOTAL
            )

    def test_half_precision_demotes_float_columns(self):
        sink = _open(_sink(half=True))
        dtypes, _ = sink._merge_columns(  # noqa: SLF001
            _STREAMS, _SEQ_STREAMS, {"jets": "jets", "tracks": "tracks"}, _TOTAL
        )
        assert dtypes["particles"]["salt_pa"] == np.dtype("f2")
        assert dtypes["particle_masks"]["logit"] == np.dtype("f2")
        # int columns are unaffected by half precision
        assert dtypes["particles"]["truth"] == np.dtype("i8")


class TestObjectGroupRequires:
    def test_declare_io_test_demands_every_field_leaf(self):
        sink = _sink()
        req = flatten_spec(sink.declare_io(Mode.TEST).requires)
        for leaf in (
            "net.particle_probs",
            "labels.particles.truth",
            "net.particle_masks",
            "outputs.tracks.owner",
        ):
            assert leaf in req, f"{leaf} not demanded"
        # kinds propagate for the planner's kind-unify
        assert req["labels.particles.truth"].kind == "label"
        assert req["net.particle_probs"].kind == "data"

    def test_non_test_modes_declare_nothing_extra(self):
        sink = _sink()
        assert flatten_spec(sink.declare_io(Mode.ONNX).requires) == {}


class TestObjectGroupFragments:
    def test_packs_bytes_from_leaves(self):
        from numpy.lib.recfunctions import unstructured_to_structured as u2s

        sink = _open(_sink())
        b = _batch()
        frags = sink._object_group_fragments(b)  # noqa: SLF001
        assert set(frags) == {"particles", "particle_masks", "tracks"}
        # particles: 2 prob cols + truth, byte-identical to a direct u2s pack
        exp_probs = u2s(
            b.get("net.particle_probs").numpy(),
            np.dtype([("salt_pa", "f4"), ("salt_pb", "f4")]),
        )
        for col in ("salt_pa", "salt_pb"):
            assert frags["particles"][col].tobytes() == exp_probs[col].tobytes()
        exp_truth = u2s(
            b.get("labels.particles.truth").unsqueeze(-1).numpy(),
            np.dtype([("truth", "i8")]),
        )
        assert frags["particles"]["truth"].tobytes() == exp_truth["truth"].tobytes()
        # the tracks index column rides the reader stream
        assert frags["tracks"].dtype.names == ("salt_Owner",)

    def test_non_reader_group_shape_guard(self):
        """A per-row shape narrower than the declared file width raises ConfigError."""
        sink = _open(_sink())
        b = _batch(n_tracks=_T - 10)  # particle_masks per-row (P, T-10) != declared (P, T)
        with pytest.raises(ConfigError, match="particle_masks"):
            sink._object_group_fragments(b)  # noqa: SLF001

    def test_reader_stream_fragment_is_reexpanded(self):
        """A reader-stream group fragment is padded to the file token length."""
        groups = [
            ObjectGroup(
                name="tracks",
                fields=[ObjectGroupField(leaf="outputs.tracks.owner", suffixes=["Owner"],
                                         dtype="i8")],
            )
        ]
        sink = _open(_sink(groups))
        b = Bundle()
        b.set("outputs.tracks.owner", torch.zeros(4, _T - 5, dtype=torch.int64))  # model < file
        frags = sink._object_group_fragments(b)  # noqa: SLF001
        assert frags["tracks"].shape == (4, _T)  # re-expanded to file width
