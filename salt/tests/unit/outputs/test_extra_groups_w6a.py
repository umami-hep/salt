"""W6a — POSITIVE coverage for the H5OutputSink ``extra_groups`` seam."""

from __future__ import annotations

import numpy as np
import pytest

from salt.core.graph.errors import ConfigError
from salt.core.outputs.h5_sink import H5OutputSink, _ExtraGroupCtx
from salt.core.outputs.output_column import OutputColumn

pytestmark = pytest.mark.cpu_always

_STREAMS = ("jets", "tracks")
_SEQ_STREAMS = ("tracks",)
_TOTAL = 123
_M = 5  # the object axis (num_objects)
_T = 40  # the constituent (tracks) file token length


class _StubExtraNode:
    """A minimal extra-group node (MaskFormer-object-writer shaped, W6b stand-in)."""

    def __init__(
        self,
        extra: dict[str, tuple[int, ...]] | None = None,
        cols: dict[str, np.dtype] | None = None,
    ) -> None:
        self._extra = extra
        self._cols = cols

    def extra_groups(self, ctx: _ExtraGroupCtx) -> dict[str, tuple[int, ...]]:
        if self._extra is not None:
            return self._extra
        tok = ctx.seq_lengths["tracks"]
        return {"objects": (_M,), "object_masks": (_M, tok)}

    def columns(self, ctx: _ExtraGroupCtx) -> dict[str, np.dtype]:
        if self._cols is not None:
            return self._cols
        fmt = "f2" if ctx.precision == "half" else "f4"
        return {
            "objects": np.dtype([(f"{ctx.run_name}_pX", fmt), ("MaskTarget", "i8")]),
            "object_masks": np.dtype([("MaskTruth", "i8"), ("MaskLogits", fmt)]),
        }


def _ctx(precision: str = "full") -> _ExtraGroupCtx:
    return _ExtraGroupCtx(
        streams=_STREAMS,
        seq_lengths={"tracks": _T},
        total=_TOTAL,
        run_name="salt",
        precision=precision,
    )


def _sink(node: _StubExtraNode, *, name: str = "mf") -> H5OutputSink:
    """A sink with ONE explicit task column + the extra-group node bound."""
    sink = H5OutputSink(
        outputs=[OutputColumn(key="outputs.jets.cls", suffixes=["pb"])],
        extra_groups=[name],
    )
    sink.bind_output_section({name: node})
    return sink


class TestCollectExtraGroups:
    def test_shapes_and_owner_map(self):
        """The populated collect mirrors the MaskFormer extra_groups declaration."""
        sink = _sink(_StubExtraNode())
        shapes, owner = sink._collect_extra_groups(_ctx(), _STREAMS)  # noqa: SLF001
        assert shapes == {"objects": (_M,), "object_masks": (_M, _T)}
        assert owner == {"objects": "mf", "object_masks": "mf"}
        # trailing dims are plain python ints (port: tuple(int(d) for d in trailing))
        for trailing in shapes.values():
            assert all(type(d) is int for d in trailing)

    def test_empty_is_noop(self):
        """No extra_groups -> ({}, {}) (the byte-identical seam the configs ship)."""
        sink = H5OutputSink(outputs=[OutputColumn(key="outputs.jets.cls", suffixes=["pb"])])
        assert sink._collect_extra_groups(_ctx(), _STREAMS) == ({}, {})  # noqa: SLF001

    def test_group_shadowing_reader_stream_raises(self):
        """An extra group named after a reader stream is a ConfigError (callback.py:891)."""
        sink = _sink(_StubExtraNode(extra={"jets": (_M,)}))
        with pytest.raises(ConfigError, match="shadows reader stream"):
            sink._collect_extra_groups(_ctx(), _STREAMS)  # noqa: SLF001

    def test_duplicate_owner_raises(self):
        """Two nodes claiming the same extra group is a ConfigError (callback.py:897)."""
        sink = H5OutputSink(
            outputs=[OutputColumn(key="outputs.jets.cls", suffixes=["pb"])],
            extra_groups=["a", "b"],
        )
        sink.bind_output_section({
            "a": _StubExtraNode(extra={"objects": (_M,)}),
            "b": _StubExtraNode(extra={"objects": (_M,)}),
        })
        with pytest.raises(ConfigError, match="one node owns one extra group"):
            sink._collect_extra_groups(_ctx(), _STREAMS)  # noqa: SLF001

    def test_unbound_name_raises(self):
        """An extra_groups name with no bound section node is a ConfigError."""
        sink = H5OutputSink(
            outputs=[OutputColumn(key="outputs.jets.cls", suffixes=["pb"])],
            extra_groups=["ghost"],
        )
        sink.bind_output_section({"mf": _StubExtraNode()})
        with pytest.raises(ConfigError, match="not a bound outputs: section node"):
            sink._collect_extra_groups(_ctx(), _STREAMS)  # noqa: SLF001


class TestGroupShape:
    def test_extra_branch_shapes(self):
        """_group_shape sizes extra groups (total, *trailing); readers unchanged."""
        sink = _sink(_StubExtraNode())
        sink._extra_shapes, _ = sink._collect_extra_groups(_ctx(), _STREAMS)  # noqa: SLF001
        sink._seq_lengths = {"tracks": _T}  # noqa: SLF001 (open_schema sets this)
        gs = sink._group_shape  # noqa: SLF001
        assert gs("objects", _SEQ_STREAMS, _TOTAL) == (_TOTAL, _M)
        assert gs("object_masks", _SEQ_STREAMS, _TOTAL) == (_TOTAL, _M, _T)
        # reader streams keep the prior inline rule
        assert gs("tracks", _SEQ_STREAMS, _TOTAL) == (_TOTAL, _T)
        assert gs("jets", _SEQ_STREAMS, _TOTAL) == (_TOTAL,)


class TestMergeColumnsExtraBranch:
    def test_extra_dtypes_and_shapes_appended(self):
        """The populated merge appends the extra-group dtypes + shapes after tasks."""
        sink = _sink(_StubExtraNode())
        ctx = _ctx()
        sink._extra_shapes, _ = sink._collect_extra_groups(ctx, _STREAMS)  # noqa: SLF001
        sink._seq_lengths = {"tracks": _T}  # noqa: SLF001
        dtypes, shapes = sink._merge_columns(  # noqa: SLF001
            _STREAMS, _SEQ_STREAMS, {"jets": "jets", "tracks": "tracks"}, _TOTAL, ctx
        )
        # the task column still lands in its reader group
        assert "salt_pb" in dtypes["jets"].names
        # the extra groups are present with the node-declared columns + shapes
        assert dtypes["objects"].names == ("salt_pX", "MaskTarget")
        assert dtypes["object_masks"].names == ("MaskTruth", "MaskLogits")
        assert shapes["objects"] == (_TOTAL, _M)
        assert shapes["object_masks"] == (_TOTAL, _M, _T)

    def test_extra_column_colliding_with_task_column_raises(self):
        """An extra column shadowing a task field collides via the shared _add map."""
        # node writes a 'salt_pb' field into the reader stream 'jets' — the exact
        # column the task OutputColumn already mints there.
        node = _StubExtraNode(
            extra={"objects": (_M,)},
            cols={"jets": np.dtype([("salt_pb", "f4")])},
        )
        sink = _sink(node)
        ctx = _ctx()
        sink._extra_shapes, _ = sink._collect_extra_groups(ctx, _STREAMS)  # noqa: SLF001
        sink._seq_lengths = {"tracks": _T}  # noqa: SLF001
        with pytest.raises(ConfigError, match="is declared by"):
            sink._merge_columns(  # noqa: SLF001
                _STREAMS, _SEQ_STREAMS, {"jets": "jets", "tracks": "tracks"}, _TOTAL, ctx
            )

    def test_half_precision_extra_columns(self):
        """The extra ctx threads precision into the node's column dtypes (f2)."""
        sink = H5OutputSink(
            outputs=[OutputColumn(key="outputs.jets.cls", suffixes=["pb"])],
            extra_groups=["mf"],
            half_precision=True,
        )
        sink.bind_output_section({"mf": _StubExtraNode()})
        ctx = _ctx(precision="half")
        sink._extra_shapes, _ = sink._collect_extra_groups(ctx, _STREAMS)  # noqa: SLF001
        sink._seq_lengths = {"tracks": _T}  # noqa: SLF001
        dtypes, _ = sink._merge_columns(  # noqa: SLF001
            _STREAMS, _SEQ_STREAMS, {"jets": "jets", "tracks": "tracks"}, _TOTAL, ctx
        )
        assert dtypes["objects"]["salt_pX"] == np.dtype("f2")
