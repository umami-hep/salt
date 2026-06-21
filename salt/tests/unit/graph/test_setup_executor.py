"""Kernel tests for the setup-graph compile/execute pipeline.

Covers (plan-24 §3/§4, plan-25 §3.6 / W3.0): the `compile_setup_plan` planner
reuse (topo + validation, shape-unification SKIPPED), the bespoke
`run_setup_plan` setup-execution loop (write-once ctx threading), the namespace
split that keeps setup-only modules OUT of the per-batch compile (the
`AllModesDeadError` blocker, via `datamodule._is_setup_only`), and the existing
tensor `compile_plan` as a control proving the `_compile_core` refactor did not
perturb it.

(Split out of the former ``test_setup_graph.py``: the pure `SourceSpec` /
`SetupIO` type-system tests live in ``test_setup_spec.py``. This file owns the
shared setup-only toy DatasetModules used across all the pipeline groups.)
"""

from __future__ import annotations

import pytest

from salt.core.data.base import DatasetModule
from salt.core.data.datamodule import _is_setup_only
from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import (
    ConnectivityError,
    DeclarationError,
    KeyCollisionError,
)
from salt.core.graph.planner import compile_plan, compile_setup_plan
from salt.core.graph.setup_executor import run_setup_plan
from salt.core.graph.setup_spec import (
    SetupIO,
    SourceSpec,
    unflatten_source_spec,
)
from salt.core.graph.spec import IO, Mode, TensorSpec, unflatten_spec

# ---------------------------------------------------------------------------
# dummy setup-only modules (no physics — kernel scope)
# ---------------------------------------------------------------------------


class SetupToy(DatasetModule):
    """A setup-only DatasetModule: non-empty declare_setup_io, empty declare_io.

    Mirrors the InputSamples/VDS/ShmStage shape — a pure-source node whose only
    face is the setup graph. The canonical module style RETURNS its newly
    produced dict (flat dotted keys ok) and the setup-execution loop owns the
    write-once merge, exactly like the per-batch executor merges `read()` /
    `process()` returns.
    """

    def __init__(self, name, requires=None, produces=None, values=None):
        super().__init__()
        self.name = name
        self._req = dict(requires or {})
        self._prod = dict(produces or {})
        # values to write for each produced key; default the key string itself
        self._values = dict(values or {})

    def declare_io(self, mode: Mode) -> IO:  # empty per-batch face
        del mode
        return IO()

    def declare_setup_io(self, stage) -> SetupIO:
        del stage
        return SetupIO(
            requires=unflatten_source_spec(self._req),
            produces=unflatten_source_spec(self._prod),
        )

    def setup(self, ctx, stage):
        del ctx, stage
        return {key: self._values.get(key, f"<{key}>") for key in self._prod}


class SelfMergeToy(SetupToy):
    """A setup-only module that merges into ctx ITSELF and returns ctx (plan-25 §3.8).

    Uses `canonical_produced` to turn its flat dotted produces into the nested
    single-component form `Bundle.merge` expects (the same canonicalisation the
    executor applies), then returns the ctx unchanged — the loop adds nothing.
    """

    def setup(self, ctx, stage):
        del stage
        from salt.core.graph.executor import canonical_produced

        out = {key: self._values.get(key, f"<{key}>") for key in self._prod}
        if out:
            expected = set(out)
            ctx.merge(canonical_produced(out, expected, self.name), who=self.name, expected=expected)
        return ctx


class BatchToy(DatasetModule):
    """A per-batch DatasetModule (non-empty declare_io), e.g. a processor."""

    def __init__(self, name, requires=None, produces=None):
        super().__init__()
        self.name = name
        self._io = IO(
            requires=unflatten_spec(requires or {}),
            produces=unflatten_spec(produces or {}),
        )

    def declare_io(self, mode: Mode) -> IO:
        del mode
        return self._io


def src(kind="path", **kwargs):
    return SourceSpec(kind=kind, **kwargs)


def mods(*modules):
    return {m.name: m for m in modules}


# ---------------------------------------------------------------------------
# §4.3 — compile_setup_plan (topo + validation, no shape-unification)
# ---------------------------------------------------------------------------


def chain_modules():
    """InputSamples → VDS → ShmStage style chain."""
    inp = SetupToy("inp", produces={"source.r.train.pattern": src()})
    vds = SetupToy(
        "vds",
        requires={"source.r.train.pattern": src()},
        produces={"source.r.train.vds_path": src()},
    )
    shm = SetupToy(
        "shm",
        requires={"source.r.train.vds_path": src()},
        produces={"source.r.train.staged_path": src()},
    )
    return inp, vds, shm


class TestCompileSetupPlan:
    def test_topo_order(self):
        inp, vds, shm = chain_modules()
        # declared out of dependency order to prove topo-sort, not config order
        plan = compile_setup_plan(mods(shm, inp, vds), "train")
        assert list(plan.module_names) == ["inp", "vds", "shm"]

    def test_plan_hash_stable_and_deterministic(self):
        inp, vds, shm = chain_modules()
        h1 = compile_setup_plan(mods(inp, vds, shm), "train").plan_hash
        h2 = compile_setup_plan(mods(shm, vds, inp), "train").plan_hash
        assert h1 == h2 and len(h1) == 64

    def test_scalar_and_path_kinds_in_one_plan(self):
        inp = SetupToy(
            "inp",
            produces={"source.r.train.pattern": src(), "artifacts.r.num": src(kind="scalar")},
        )
        plan = compile_setup_plan(mods(inp), "train")
        step = plan.step("inp")
        assert step.produces["source.r.train.pattern"].kind == "path"
        assert step.produces["artifacts.r.num"].kind == "scalar"

    def test_duplicate_producer_raises(self):
        a = SetupToy("a", produces={"source.r.train.pattern": src()})
        b = SetupToy("b", produces={"source.r.train.pattern": src()})
        with pytest.raises(ConnectivityError, match="two producers"):
            compile_setup_plan(mods(a, b), "train")

    def test_unproduced_consumer_raises(self):
        vds = SetupToy(
            "vds",
            requires={"source.r.train.pattern": src()},
            produces={"source.r.train.vds_path": src()},
        )
        # no InputSamples producing the pattern
        with pytest.raises(ConnectivityError, match="no module or source produces"):
            compile_setup_plan(mods(vds), "train")

    def test_kind_mismatch_raises(self):
        from salt.core.graph.errors import KindError

        prod = SetupToy("prod", produces={"k": src(kind="scalar")})
        cons = SetupToy("cons", requires={"k": src(kind="path")}, produces={"out": src()})
        with pytest.raises(KindError, match="kind"):
            compile_setup_plan(mods(prod, cons), "train")

    def test_shape_unification_is_skipped(self):
        # SourceSpec has no .shape/.dtype; if _unify_edge ran it would AttributeError.
        # Two chained setup edges compile cleanly precisely because unify is bypassed.
        inp, vds, shm = chain_modules()
        plan = compile_setup_plan(mods(inp, vds, shm), "train")
        assert len(plan.steps) == 3

    def test_stage_gating_excludes_inactive_module(self):
        # a module whose only produce is gated to 'test' is inactive in 'train'
        inp = SetupToy("inp", produces={"source.r.train.pattern": src()})
        test_only = SetupToy(
            "test_only", produces={"source.r.test.pattern": src(stages=("test",))}
        )
        plan = compile_setup_plan(mods(inp, test_only), "train")
        assert "test_only" not in plan.module_names
        assert "inp" in plan.module_names


# ---------------------------------------------------------------------------
# §4.3(b) — the setup-execution loop (write-once ctx threading)
# ---------------------------------------------------------------------------


class TestRunSetupPlan:
    def test_threads_ctx_in_topo_order(self):
        inp = SetupToy("inp", produces={"source.r.train.pattern": src()}, values={
            "source.r.train.pattern": "/data/train.h5"
        })
        vds = SetupToy(
            "vds",
            requires={"source.r.train.pattern": src()},
            produces={"source.r.train.vds_path": src()},
            values={"source.r.train.vds_path": "/data/train.vds.h5"},
        )
        plan = compile_setup_plan(mods(inp, vds), "train")
        ctx = run_setup_plan(plan, "train")
        assert ctx.get("source.r.train.pattern") == "/data/train.h5"
        assert ctx.get("source.r.train.vds_path") == "/data/train.vds.h5"

    def test_self_merge_style_module_merges_into_ctx(self):
        # plan-25 §3.8 "merge yourself + return ctx" style is also supported.
        inp = SelfMergeToy(
            "inp",
            produces={"source.r.train.pattern": src()},
            values={"source.r.train.pattern": "/data/x.h5"},
        )
        plan = compile_setup_plan(mods(inp), "train")
        ctx = run_setup_plan(plan, "train")
        assert ctx.get("source.r.train.pattern") == "/data/x.h5"

    def test_returner_extra_key_raises_declaration_error(self):
        class Bad(SetupToy):
            def setup(self, ctx, stage):
                del ctx, stage
                return {"source.r.train.pattern": "x", "source.r.train.extra": "y"}

        bad = Bad("inp", produces={"source.r.train.pattern": src()})
        plan = compile_setup_plan(mods(bad), "train")
        with pytest.raises(DeclarationError):
            run_setup_plan(plan, "train")

    def test_write_once_collision_across_two_stages(self):
        # one shared ctx for setup("fit"): train then val. Same key twice collides.
        inp = SetupToy("inp", produces={"artifacts.r.num": src(kind="scalar")}, values={
            "artifacts.r.num": {"train": -1, "val": -1}
        })
        plan = compile_setup_plan(mods(inp), "train")
        ctx = run_setup_plan(plan, "train")
        # re-running into the SAME ctx for 'val' rewrites the same key -> collision
        with pytest.raises(KeyCollisionError):
            run_setup_plan(plan, "val", ctx)

    def test_disjoint_stage_keys_accumulate_in_one_ctx(self):
        inp = SetupToy("inp", produces={"source.r.train.pattern": src()}, values={
            "source.r.train.pattern": "/train.h5"
        })
        inp_val = SetupToy("inp", produces={"source.r.val.pattern": src()}, values={
            "source.r.val.pattern": "/val.h5"
        })
        ctx = Bundle()
        run_setup_plan(compile_setup_plan(mods(inp), "train"), "train", ctx)
        run_setup_plan(compile_setup_plan(mods(inp_val), "val"), "val", ctx)
        assert ctx.get("source.r.train.pattern") == "/train.h5"
        assert ctx.get("source.r.val.pattern") == "/val.h5"


# ---------------------------------------------------------------------------
# §4.4 / §3.6 — the namespace split (AllModesDeadError blocker)
# ---------------------------------------------------------------------------


class TestNamespaceSplit:
    def test_is_setup_only_classifies_pure_source(self):
        setup_mod = SetupToy("inp", produces={"source.r.train.pattern": src()})
        assert _is_setup_only(setup_mod)

    def test_is_setup_only_false_for_dual_face(self):
        class DualFace(SetupToy):
            def declare_io(self, mode):
                del mode
                return IO(produces=unflatten_spec({"raw.jets": TensorSpec()}))

        dual = DualFace("reader", produces={"source.r.train.pattern": src()})
        assert not _is_setup_only(dual)

    def test_is_setup_only_false_for_pure_batch(self):
        batch = BatchToy("proc", requires={"raw.x": TensorSpec()}, produces={"inputs.x": TensorSpec()})
        assert not _is_setup_only(batch)

    def test_setup_only_module_absent_from_per_batch_plan_no_dead_error(self):
        # The core blocker: a setup-only module in the per-batch module dict
        # must be partitioned out so compile_plan never sees it (else
        # AllModesDeadError). We emulate the GraphDataModule partition and
        # assert the surviving per-batch compile is clean AND excludes it.
        a = BatchToy("a", requires={"inputs.x": TensorSpec()}, produces={"preds.x": TensorSpec()})
        setup_mod = SetupToy("inp", produces={"source.r.train.pattern": src()})
        full = mods(a, setup_mod)
        batch_modules = {n: m for n, m in full.items() if not _is_setup_only(m)}
        assert "inp" not in batch_modules
        src_x = unflatten_spec({"inputs.x": TensorSpec()})
        plan = compile_plan(batch_modules, Mode.FIT, src_x, sinks=["preds.x"])
        assert "inp" not in plan.module_names
        assert "a" in plan.module_names

    def test_unpartitioned_setup_only_would_trip_dead_error(self):
        # documents WHY the split is required: without it, compile_plan raises.
        from salt.core.graph.errors import AllModesDeadError

        a = BatchToy("a", requires={"inputs.x": TensorSpec()}, produces={"preds.x": TensorSpec()})
        setup_mod = SetupToy("inp", produces={"source.r.train.pattern": src()})
        src_x = unflatten_spec({"inputs.x": TensorSpec()})
        with pytest.raises(AllModesDeadError, match="'inp'"):
            compile_plan(mods(a, setup_mod), Mode.FIT, src_x, sinks=["preds.x"])


# ---------------------------------------------------------------------------
# control — the tensor compile_plan is unaffected by the _compile_core refactor
# ---------------------------------------------------------------------------


class TestTensorPlannerUnaffected:
    def test_chain_compiles_and_unifies(self):
        a = BatchToy(
            "a",
            requires={"inputs.x": TensorSpec(shape=("B", 4))},
            produces={"embed.x": TensorSpec(shape=("B", 8))},
        )
        b = BatchToy(
            "b",
            requires={"embed.x": TensorSpec(shape=("B", 8))},
            produces={"preds.x": TensorSpec(shape=("B", 2))},
        )
        src_x = unflatten_spec({"inputs.x": TensorSpec(shape=("B", 4))})
        plan = compile_plan(mods(a, b), Mode.FIT, src_x, sinks=["preds.x"])
        assert list(plan.module_names) == ["a", "b"]

    def test_shape_conflict_still_raises(self):
        from salt.core.graph.errors import ShapeError

        a = BatchToy("a", requires={"inputs.x": TensorSpec(shape=("B", 4))},
                     produces={"embed.x": TensorSpec(shape=("B", 8))})
        b = BatchToy("b", requires={"embed.x": TensorSpec(shape=("B", 16))},
                     produces={"preds.x": TensorSpec(shape=("B", 2))})
        src_x = unflatten_spec({"inputs.x": TensorSpec(shape=("B", 4))})
        with pytest.raises(ShapeError):
            compile_plan(mods(a, b), Mode.FIT, src_x, sinks=["preds.x"])
