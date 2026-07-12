"""Regression gates for the plan-29 W1 sink-node surface (design §4, §5, §7)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from salt.core.cli import load_config
from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.executor import Executor
from salt.core.graph.planner import compile_plan
from salt.core.graph.spec import (
    IO,
    Mode,
    SinkModule,
    TensorSpec,
    unflatten_spec,
)
from salt.core.outputs import CollectOutputs, H5OutputSink, OutputColumn
from salt.core.render import dot_source
from salt.core.saltmodule import SaltModule

# this file is at salt/tests/unit/outputs/ — the configs live at salt/core/configs/
_CONFIGS = Path(__file__).parents[3] / "core" / "configs"
_DUMMY = str(_CONFIGS / "gn2v2-dummy.yaml")
_CUTOVER = str(_CONFIGS / "gn2v2-dummy-cutover.yaml")
_OVERRIDES = [
    "model.modules.norm.init_args.norm_dict=unused.yaml",
    "trainer.logger=false",
]

# the cutover producer leaves the H5 sink consumes (design §7 exp-15 assertion).
_JET_OUT = "outputs.jets.jets_classification"
_TRK_OUT = "outputs.tracks.track_origin"


@pytest.fixture(scope="module")
def cutover_cfg():
    """The live P1.5 cutover config — the H5OutputSink folded as a node (design §4.3)."""
    return load_config([_DUMMY, _CUTOVER], _OVERRIDES)


def _compile(cfg, mode):
    """Compile one mode from a loaded `GraphConfig` (the static CLI plan)."""
    return compile_plan(
        cfg.modules,
        mode,
        cfg.sources,
        schema=cfg.schema,
        sinks=cfg.sinks,
        sink_origins=cfg.sink_origins.get(mode),
    )


# (1) the SinkModule marker — H5OutputSink is a sink node, CollectOutputs is NOT


def test_h5_output_sink_is_a_sink_module():
    """`H5OutputSink` satisfies the `SinkModule` Protocol (``is_sink() -> True``)."""
    sink = H5OutputSink(outputs=[OutputColumn(key=_JET_OUT, suffixes=["pb", "pc", "pu"])])
    assert isinstance(sink, SinkModule)
    assert sink.is_sink() is True


class _FakeH5:
    """A stand-in for the ftag `H5Writer` recording whether its file handle closed."""

    def __init__(self) -> None:
        self.file = self
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_close_if_open_closes_handle_without_full_count_assertion():
    """`close_if_open` closes a leaked handle on an interrupted test and is idempotent (§5.3)."""
    sink = H5OutputSink(outputs=[OutputColumn(key=_JET_OUT, suffixes=["pb", "pc", "pu"])])
    fake = _FakeH5()
    # simulate an open writer mid-test with FEWER rows written than expected
    # (the interrupted-batch shape `flush`'s full-count branch would assert on)
    sink._h5 = fake  # noqa: SLF001 - white-box cleanup contract
    sink._rows_written = 3  # noqa: SLF001
    sink._expected = 10  # noqa: SLF001

    sink.close_if_open()
    assert fake.closed is True
    assert sink._h5 is None  # noqa: SLF001 - handle released
    # idempotent: a second close on the already-released handle is a no-op
    sink.close_if_open()
    assert sink._h5 is None  # noqa: SLF001


def test_collect_outputs_is_not_a_sink_module():
    """The legacy duck-typed `CollectOutputs` is NOT a `SinkModule` (legacy path preserved)."""
    legacy = CollectOutputs(outputs=[_TRK_OUT])
    assert not isinstance(legacy, SinkModule)
    assert not hasattr(legacy, "is_sink")


# (2) executor partition — folded sink is in plan.steps, NOT in the forward loop


class _ToyProducer:
    """A minimal callable producer: doubles its input into ``outputs.x``."""

    def __init__(self) -> None:
        self.name = "prod"
        self.called = False

    def declare_io(self, mode):
        del mode
        return IO(
            unflatten_spec({"inputs.x": TensorSpec()}),
            unflatten_spec({"outputs.x": TensorSpec()}),
        )

    def __call__(self, b, mode):
        del mode
        self.called = True
        return {"outputs.x": b.get("inputs.x") * 2}


class _ExplodingSink:
    """A terminal sink whose tensor-forward call would raise — proving it is never invoked."""

    name = "toy_sink"

    def is_sink(self) -> bool:
        return True

    def declare_io(self, mode):
        if not (mode & Mode.TEST):
            return IO(requires={}, produces={})
        return IO(
            unflatten_spec({"outputs.x": TensorSpec(shape=None, dtype=None, kind="data")}),
            produces={},
        )

    def __call__(self, b, mode):  # pragma: no cover - must never run
        del b, mode
        raise AssertionError("a terminal sink must never be invoked as a tensor forward")


def test_executor_partitions_sink_out_of_forward_loop():
    """A folded sink step is in ``plan.steps`` but NOT in the executor forward loop."""
    prod = _ToyProducer()
    sink = _ExplodingSink()
    modules = {prod.name: prod, sink.name: sink}
    sources = unflatten_spec({"inputs.x": TensorSpec(shape=("B", 4), dtype="float32")})
    plan = compile_plan(modules, Mode.TEST, sources, sinks=[])

    # the sink is a real PlanStep (renders its own card) ...
    assert sink.name in plan.module_names
    executor = Executor(plan)
    # ... but excluded from the forward loop (the producer stays in it)
    forward_names = [step.name for step in executor._forward_steps]  # noqa: SLF001 - white-box partition assertion
    assert sink.name not in forward_names
    assert prod.name in forward_names

    out = executor.run(Bundle({"inputs": {"x": torch.ones(3, 4)}}))
    # the producer ran (its output is present); the sink forward never raised
    assert prod.called is True
    assert out.get("outputs.x") is not None


# (3) plan_hash invariance — the folded sink perturbs nothing in FIT/VAL/ONNX


@pytest.mark.parametrize("mode", [Mode.FIT, Mode.VAL, Mode.ONNX])
def test_folded_sink_contributes_no_step_or_edge_outside_test(cutover_cfg, mode):
    """The sink node is ABSENT (inactive) from the FIT/VAL/ONNX plans."""
    plan = _compile(cutover_cfg, mode)
    assert "h5_output" not in plan.module_names
    touching = [e for e in plan.edges if "h5_output" in {e.producer, e.consumer}]
    assert touching == []


@pytest.mark.parametrize("mode", [Mode.FIT, Mode.VAL])
def test_fit_val_plan_hash_byte_identical_with_vs_without_sink(cutover_cfg, mode):
    """FIT/VAL ``plan_hash`` is byte-identical with the sink folded vs absent."""
    full = _compile(cutover_cfg, mode).plan_hash
    eval_only = {"jet_probs", "track_origin_probs", "h5_output"}
    base_modules = {k: v for k, v in cutover_cfg.modules.items() if k not in eval_only}
    base = compile_plan(
        base_modules,
        mode,
        cutover_cfg.sources,
        schema=cutover_cfg.schema,
        sinks=cutover_cfg.sinks,
        sink_origins=cutover_cfg.sink_origins.get(mode),
    ).plan_hash
    assert full == base


# (4) render payoff — named h5_output card, no <sinks> sentinel, ONNX prunes


def test_cutover_test_render_has_named_h5_sink_card(cutover_cfg):
    """The cutover TEST DOT renders the H5 sink as its OWN named card (design §7 exp-15)."""
    test = _compile(cutover_cfg, Mode.TEST)
    dot = dot_source(test, cutover_cfg.modules)

    assert "<sinks>" not in dot  # the sentinel branch is dead (the W1 payoff)
    assert "h5_output" in dot
    assert "H5OutputSink" in dot
    for leaf in (_JET_OUT, _TRK_OUT, "meta.rows", "masks.tracks"):
        assert leaf in dot
    # named-consumer edges flow into the NAMED node, not a sentinel
    assert '"jet_probs" -> "h5_output";' in dot
    assert '"track_origin_probs" -> "h5_output";' in dot


def test_cutover_onnx_render_prunes_h5_sink(cutover_cfg):
    """The ONNX DOT prunes the H5 sink entirely (TEST-only declare_io)."""
    onnx = _compile(cutover_cfg, Mode.ONNX)
    assert "h5_output" not in onnx.module_names
    dot = dot_source(onnx, cutover_cfg.modules)
    assert "h5_output" not in dot


# (5) dead-preds safety net on the folded-sink TEST path (parity with M4.5)


class _Stub:
    """A minimal `GraphModule` declaring the given requires/produces (compile-only)."""

    def __init__(self, name, requires=None, produces=None):
        self.name = name
        self._req = requires or {}
        self._prod = produces or {}

    def declare_io(self, mode):
        del mode
        return IO(unflatten_spec(self._req), unflatten_spec(self._prod))


class _StubSalt:
    """A stand-in `self` carrying only ``_graph_modules`` for the unbound gate call."""

    def __init__(self, graph_modules):
        self._graph_modules = graph_modules


def _folded_test_plan(*, include_dead):
    """Compile a TEST plan: a converted track pred (+ optionally a DEAD jet pred)."""
    from salt.core.outputs import TaskOutput  # noqa: PLC0415 - test-local

    pred_key = "preds.tracks.track_origin"
    out_key = "outputs.tracks.track_origin"
    dead_pred = "preds.jets.jets_classification"
    track_task = _Stub(
        "track_task", produces={pred_key: TensorSpec(shape=("B", "L", 5), dtype="float32")}
    )
    producer = TaskOutput(task="track_origin", stream="tracks")
    producer.name = "track_origin_output"
    sink = H5OutputSink(outputs=[OutputColumn(key=out_key, suffixes=list("abcde"))])
    modules = {
        track_task.name: track_task,
        producer.name: producer,
        sink.name: sink,
    }
    if include_dead:
        # a SECOND task whose preds.* no producer converts and no output persists
        modules["jet_task"] = _Stub(
            "jet_task", produces={dead_pred: TensorSpec(shape=("B", 3), dtype="float32")}
        )
    sources = unflatten_spec({"meta.rows": TensorSpec(shape=(2,), dtype="int64", kind="meta")})
    plan = compile_plan(modules, Mode.TEST, sources, sinks=[])
    graph_modules = {k: v for k, v in modules.items() if k != sink.name}
    return graph_modules, plan


def test_folded_sink_dead_preds_gate_passes_when_all_consumed():
    """No dead pred when every produced ``preds.*`` feeds a demanded output."""
    graph_modules, plan = _folded_test_plan(include_dead=False)
    # the production gate is a no-op on a fully-consumed folded plan
    SaltModule._assert_no_dead_preds(_StubSalt(graph_modules), plan)  # noqa: SLF001 - white-box gate


def test_folded_sink_dead_preds_gate_fires_on_unconsumed_pred():
    """A computed-but-never-persisted ``preds.*`` is a hard error on the folded-sink path."""
    graph_modules, plan = _folded_test_plan(include_dead=True)
    with pytest.raises(ConfigError) as excinfo:
        SaltModule._assert_no_dead_preds(_StubSalt(graph_modules), plan)  # noqa: SLF001 - white-box gate
    assert "preds.jets.jets_classification" in str(excinfo.value)
