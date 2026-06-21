"""Tests for `salt.core.writers.callback` — the `WriterCallback` column merge.

Covers `WriterCallback` construction validation, the column-merge collision
check, writer-order preservation in the merged v1 layout, and the runtime
export-only-stub backstop (design §8; M4.5 merge condition 3).

(Split out of the former monolithic ``test_writers.py``; shared fixtures /
toy-writers / constants come from ``salt.tests._fixtures.writers_common``.)
"""

from __future__ import annotations

import h5py
import pytest

from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import TensorSpec
from salt.core.writers import (
    InputCopyWriter,
    PadMaskWriter,
    TaskWriter,
    Writer,
    WriterCallback,
)
from salt.tests._fixtures.writers_common import (  # noqa: F401  (data/modules are fixtures)
    JET_PROB_COLS,
    L_FILE,
    N_JETS,
    data,
    modules,
    write_ctx,
)

# ---------------------------------------------------------------------------
# WriterCallback internals: column merge collision check (design §8)
# ---------------------------------------------------------------------------


class TestWriterCallback:
    def test_empty_modules_rejected(self):
        with pytest.raises(ConfigError, match="non-empty writer dict"):
            WriterCallback(modules={})
        with pytest.raises(ConfigError, match="non-empty writer dict"):
            WriterCallback(modules={"tasks": None})

    def test_non_writer_rejected(self):
        with pytest.raises(ConfigError, match="not a"):
            WriterCallback(modules={"tasks": object()})  # type: ignore[dict-item]

    def test_column_collision_names_both_writers(self, data, modules):
        cb = WriterCallback(modules={"a": PadMaskWriter(), "b": PadMaskWriter()})
        ctx = write_ctx(data, modules)
        for writer in cb.writers.values():
            writer.setup(ctx)
        with pytest.raises(ConfigError, match="'a' AND 'b'"):
            cb._merge_columns(ctx)

    def test_merge_preserves_writer_order(self, data, modules):
        cb = WriterCallback(
            modules={"inputs_copy": InputCopyWriter(), "tasks": TaskWriter()},
        )
        ctx = write_ctx(data, modules)
        for writer in cb.writers.values():
            writer.setup(ctx)
        dtypes, shapes = cb._merge_columns(ctx)
        with h5py.File(data["h5"]) as f:
            source_jets = list(f["jets"].dtype.names)
        # v1 layout: input copies FIRST, then task columns
        assert list(dtypes["jets"].names) == source_jets + JET_PROB_COLS
        assert shapes["tracks"] == (N_JETS, L_FILE)
        cb.writers["inputs_copy"].finalize()

    def test_runtime_stub_backstop_in_merge_columns(self, data, modules):
        # M4.5 merge condition 3, the RUNTIME half (fix-stage regression):
        # a writer that slips past the static role check (it declares TEST
        # requires) but produces NO columns while declaring ONNX outputs
        # must hit the explicit stub error at column merge — never a silent
        # zero-contribution fall-through
        from salt.core.onnx.config import ExportOutput

        class RuntimeStub(Writer):
            def requires(self, ctx):
                del ctx
                return {"meta.rows": TensorSpec(shape=(2,), dtype="int64", kind="meta")}

            def columns(self, ctx):
                del ctx
                return {}

            def write(self, bundle, rows):
                del bundle, rows
                return {}

            def onnx_outputs(self, ctx):
                del ctx
                return [ExportOutput(port="pooled.global", names=["s0", "s1"])]

        cb = WriterCallback(modules={"tasks": TaskWriter(), "stub": RuntimeStub()})
        ctx = write_ctx(data, modules)
        for writer in cb.writers.values():
            writer.setup(ctx)
        with pytest.raises(ConfigError, match="export-only stub shape"):
            cb._merge_columns(ctx)
