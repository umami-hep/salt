"""Unit tests for VertexingTaskModule and origin-weighting config."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from salt.core.graph import (
    ConfigError,
    Mode,
    flatten_spec,
)
from salt.core.nn import (
    ResolvedSchema,
)
from salt.core.nn.tasks import (
    VertexingTaskModule,
)
from salt.core.schema import GroupSchema, Schema
from salt.tests._fixtures.gn2v2_fixture import (
    ORIGIN_CLASSES,
)


class TestVertexingTaskModule:
    def test_label_must_contain_vertex_index(self):
        with pytest.raises(ConfigError, match="VertexIndex"):
            VertexingTaskModule(stream="tracks", label="vtx", origin_label="o")

    def test_unknown_origin_weighting_key(self):
        with pytest.raises(ConfigError, match="origin_weighting"):
            VertexingTaskModule(
                stream="tracks",
                label="ftagTruthVertexIndex",
                origin_label="o",
                origin_weighting={"heavy": [3], "bogus": [1]},
            )

    def test_declares_origin_label_dependency(self):
        task = VertexingTaskModule(
            stream="tracks", label="ftagTruthVertexIndex", origin_label="ftagTruthOriginLabel"
        )
        task.name = "track_vertexing"
        req = flatten_spec(task.declare_io(Mode.FIT).requires)
        assert "labels.tracks.ftagTruthOriginLabel" in req
        assert "labels.tracks.ftagTruthVertexIndex" in req

    def test_bind_pair_widths_and_reduction_check(self, gn2v2):
        modules, _, _ = gn2v2
        head = modules["track_vertexing"].task
        assert head.net.input_size == 2 * 16  # pair concat (task.py:894-896)
        assert head.net.context_size == 16
        task = VertexingTaskModule(
            stream="tracks",
            label="ftagTruthVertexIndex",
            origin_label="o",
            loss={"class_path": "torch.nn.BCEWithLogitsLoss", "init_args": {"reduction": "mean"}},
        )
        task.name = "v"
        with pytest.raises(ConfigError, match="reduction"):
            task.bind(ResolvedSchema(widths={"encoded.tracks": 16}))

    # DEL-1: test_default_origin_weighting_matches_v1 retired with the v1 tree
    # (parity-closure doctrine: v1 comparisons = git checkout 29c67a1).


def _origin_schema_reader(origin_label: str = "ftagTruthOriginLabel") -> SimpleNamespace:
    """A duck-typed reader exposing the tracks origin class names (design §2.6)."""
    schema = Schema(groups={"tracks": GroupSchema(fields={}, attrs={origin_label: ORIGIN_CLASSES})})
    return SimpleNamespace(schema_group=schema.groups.get)


class TestOriginWeightingConfig:
    """Name-based origin_weighting (design §5.1, M5 sub-wave D): names resolve to ids."""

    def _name_based(self) -> VertexingTaskModule:
        task = VertexingTaskModule(
            stream="tracks",
            label="ftagTruthVertexIndex",
            origin_label="ftagTruthOriginLabel",
            origin_weighting={"heavy": ["FromB", "FromBC", "FromC"], "fake": ["Fake"]},
        )
        task.name = "track_vertexing"
        return task

    def test_names_pending_until_resolved(self):
        # __init__ captures names but does NOT resolve them (no reader yet)
        task = self._name_based()
        assert task._names_pending is True  # noqa: SLF001
        assert task.heavy_ids is None
        assert task.fake_ids is None

    def test_names_resolve_to_v1_default_ids(self):
        # the GN3 origin names map to exactly v1's hardcoded heavy 3,4,5 / fake 1
        task = self._name_based()
        resolved = task.resolve_origin_names(_origin_schema_reader())
        assert resolved is True
        assert task.heavy_ids == (3, 4, 5)
        assert task.fake_ids == (1,)
        assert task._names_pending is False  # noqa: SLF001

    def test_name_resolved_weights_match_integer_configured(self):
        # name-resolved ids produce weights bit-identical to a FRESH,
        # independently-constructed head configured with the literal default
        # integer ids (3,4,5)/1 — never the resolved module's own head.
        # (DEL-1: the retired v1 reference hardcoded these same ids; the
        # id-parity itself is pinned by test_names_resolve_to_v1_default_ids.)
        task = self._name_based()
        task.resolve_origin_names(_origin_schema_reader())
        task.bind(ResolvedSchema(widths={"encoded.tracks": 16}))
        indep = VertexingTaskModule(
            stream="tracks",
            label="ftagTruthVertexIndex",
            origin_label="ftagTruthOriginLabel",
            origin_weighting={"heavy": [3, 4, 5], "fake": [1]},
        )
        indep.name = "track_vertexing_int"
        indep.bind(ResolvedSchema(widths={"encoded.tracks": 16}))
        labels = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
        n = labels.shape[1]
        adjmat = ~torch.eye(n, dtype=torch.bool).unsqueeze(0)
        ref_weights = indep.task.get_weights(labels, adjmat)
        v2_weights = task.task.get_weights(labels, adjmat)
        assert torch.equal(ref_weights, v2_weights)

    def test_resolve_origin_weighting_module_helper(self):
        # the saltmodule helper resolves over a module dict, counting resolutions
        from salt.core.saltmodule import resolve_origin_weighting

        task = self._name_based()
        # an int-id sibling must NOT count (already resolved)
        intd = VertexingTaskModule(
            stream="tracks", label="ftagTruthVertexIndex", origin_label="o",
            origin_weighting={"heavy": [3], "fake": [1]},
        )
        intd.name = "int_vtx"
        n = resolve_origin_weighting(
            {"track_vertexing": task, "int_vtx": intd}, _origin_schema_reader()
        )
        assert n == 1
        assert task.heavy_ids == (3, 4, 5)

    def test_unknown_name_raises_quality_error(self):
        task = VertexingTaskModule(
            stream="tracks",
            label="ftagTruthVertexIndex",
            origin_label="ftagTruthOriginLabel",
            origin_weighting={"heavy": ["NotAClass"], "fake": ["Fake"]},
        )
        task.name = "vtx"
        with pytest.raises(ConfigError, match="NotAClass") as excinfo:
            task.resolve_origin_names(_origin_schema_reader())
        assert "ftagTruthOriginLabel" in str(excinfo.value)

    def test_no_schema_attr_raises(self):
        # a name-based config but the schema has no origin class-name attr
        task = self._name_based()
        empty = SimpleNamespace(schema_group=lambda s: GroupSchema(fields={}, attrs={}))
        with pytest.raises(ConfigError, match="no string-list attr"):
            task.resolve_origin_names(empty)

    def test_name_based_bind_without_resolution_fails_loudly(self):
        # binding a name-based task that never reached a schema is a loud error,
        # NOT a silent mis-weighting
        task = self._name_based()
        with pytest.raises(ConfigError, match="not resolved") as excinfo:
            task.bind(ResolvedSchema(widths={"encoded.tracks": 16}))
        assert "schema artifact" in str(excinfo.value)

    def test_no_schema_reader_resolves_nothing(self):
        # a reader without schema support leaves names pending (the bind error
        # is the loud surface, not this no-op)
        task = self._name_based()
        assert task.resolve_origin_names(SimpleNamespace()) is False
        assert task._names_pending is True  # noqa: SLF001

    def test_integer_ids_are_noop_for_resolution(self):
        task = VertexingTaskModule(
            stream="tracks", label="ftagTruthVertexIndex", origin_label="o",
            origin_weighting={"heavy": [3, 4, 5], "fake": [1]},
        )
        task.name = "vtx"
        assert task.resolve_origin_names(_origin_schema_reader()) is False
        assert task.heavy_ids == (3, 4, 5)

    def test_mixed_ids_and_names_rejected(self):
        with pytest.raises(ConfigError, match="mixes integer ids with class names"):
            VertexingTaskModule(
                stream="tracks",
                label="ftagTruthVertexIndex",
                origin_label="o",
                origin_weighting={"heavy": [3, "FromBC", 5], "fake": [1]},
            )

    def test_non_integer_id_rejected(self):
        with pytest.raises(ConfigError, match="INTEGER origin ids or class NAMES"):
            VertexingTaskModule(
                stream="tracks",
                label="ftagTruthVertexIndex",
                origin_label="o",
                origin_weighting={"heavy": [3.5], "fake": [1]},
            )
