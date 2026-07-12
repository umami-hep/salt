"""P0 gates for the ``outputs.*`` output-writing redesign (plan 01, P0)."""

from __future__ import annotations

import pytest

from salt.core.graph import (
    IO,
    ConfigError,
    ConnectivityError,
    Mode,
    TensorSpec,
    compile_plan,
    sym_dim,
    unflatten_spec,
)
from salt.core.nn import from_v1, resolve_bind_schema, v1_sinks, v1_sources
from salt.core.outputs import CollectOutputs, TaskOutput
from salt.tests._fixtures.gn2_fixture import build_test_gn2

# The producer copies this real GN2 task's prediction into outputs.*.
_SOURCE_TASK = "track_origin"
_SOURCE_STREAM = "tracks"
_PRED_KEY = f"preds.{_SOURCE_STREAM}.{_SOURCE_TASK}"
_OUTPUT_KEY = f"outputs.{_SOURCE_STREAM}.{_SOURCE_TASK}"
_PRODUCER_NAME = "track_origin_output"

# Small fixture dims (matching parity_gn2's scale).
_EMBED_DIM = 16
_OUT_DIM = 16
_NUM_LAYERS = 2
_NUM_HEADS = 2


def _gn2_modules(tmp_path):
    """Build the real GN2 wrapper module dict (shared instances, design §2.2)."""
    v1 = build_test_gn2(
        tmp_path,
        embed_dim=_EMBED_DIM,
        out_dim=_OUT_DIM,
        num_layers=_NUM_LAYERS,
        num_heads=_NUM_HEADS,
        seed=42,
    )
    modules = from_v1(v1)
    return modules, v1_sources(v1), v1_sinks(v1)


def _with_producer(modules):
    """Add the `TaskOutput` producer to a copy of `modules` (instance name set)."""
    out = dict(modules)
    producer = TaskOutput(task=_SOURCE_TASK, stream=_SOURCE_STREAM)
    producer.name = _PRODUCER_NAME
    out[_PRODUCER_NAME] = producer
    return out


def _per_mode_sinks(pred_sinks):
    """Per-mode sink map: TEST also demands the producer's output leaf."""
    return {
        Mode.FIT: list(pred_sinks),
        Mode.VAL: list(pred_sinks),
        Mode.TEST: [*pred_sinks, _OUTPUT_KEY],
    }


# GATE (a): TRAINING UNPERTURBED — producer demand-pruned from FIT/VAL


def test_gate_a_producer_pruned_from_fit_and_val(tmp_path):
    """The producer is ABSENT from the FIT and VAL plans (demand-pruned)."""
    modules, sources, pred_sinks = _gn2_modules(tmp_path)
    with_producer = _with_producer(modules)
    sinks = _per_mode_sinks(pred_sinks)

    # the producer is genuinely a configured node ...
    assert _PRODUCER_NAME in with_producer

    fit = compile_plan(with_producer, Mode.FIT, sources=sources, sinks=sinks)
    val = compile_plan(with_producer, Mode.VAL, sources=sources, sinks=sinks)

    # ... yet demand-pruned from both training plans (its outputs reach no
    # FIT/VAL sink). Absence is a PRUNE, not mode-inactivity: the producer's
    # ports are active in every mode (declare_io carries modes=ALL).
    assert _PRODUCER_NAME not in fit.module_names
    assert _PRODUCER_NAME not in val.module_names
    # the producer's output leaf is therefore never produced in FIT/VAL
    assert _OUTPUT_KEY not in {k for s in fit.steps for k in s.produces}
    assert _OUTPUT_KEY not in {k for s in val.steps for k in s.produces}


def test_gate_a_fit_plan_byte_identical_to_no_producer(tmp_path):
    """The FIT plan with the producer EQUALS the FIT plan without it."""
    modules, sources, pred_sinks = _gn2_modules(tmp_path)
    sinks = _per_mode_sinks(pred_sinks)

    baseline = compile_plan(modules, Mode.FIT, sources=sources, sinks=sinks)
    with_producer = compile_plan(
        _with_producer(modules), Mode.FIT, sources=sources, sinks=sinks
    )

    # structural plan hash is mode-independent and covers steps/edges/sources:
    # equality proves the producer added ZERO training compute (design §4 risk 4)
    assert with_producer.plan_hash == baseline.plan_hash
    assert with_producer.module_names == baseline.module_names


@pytest.mark.integration  # runs a real GN2 forward via the parity harness
def test_gate_a_parity_gn2_stays_bitwise_with_producer_configured(tmp_path):
    """parity_gn2 stays bitwise when the producer is added to the model."""
    from salt.tests.integration.parity_gn2 import run_parity  # noqa: PLC0415 - integration-only

    def hook(modules):
        return _with_producer(modules)

    code, report = run_parity(tmp_path, modules_hook=hook)
    assert code == 0
    assert report["passed"] is True
    # producer pruned from the parity TEST plan (no outputs.* sink there)
    assert _PRODUCER_NAME not in report["plan"]["module_names"]
    # every compared leaf still bitwise-identical
    assert all(r["bitwise"] and r["passed"] for r in report["leaves"])
    assert all(r["bitwise"] and r["passed"] for r in report["intermediates"])


# GATE (b): TEST PATH GETS IT — present in TEST + width resolves TEST-only


def test_gate_b_producer_present_in_test_plan(tmp_path):
    """With the outputs.* sink demanded, the producer IS in the TEST plan."""
    modules, sources, pred_sinks = _gn2_modules(tmp_path)
    with_producer = _with_producer(modules)
    sinks = _per_mode_sinks(pred_sinks)

    test = compile_plan(with_producer, Mode.TEST, sources=sources, sinks=sinks)

    assert _PRODUCER_NAME in test.module_names
    # the producer's output leaf is genuinely produced in TEST
    produced = {k for step in test.steps for k in step.produces}
    assert _OUTPUT_KEY in produced
    # and its source preds.* leaf is pulled in transitively
    assert _PRED_KEY in produced


def test_gate_b_width_resolves_in_test_only_bind(tmp_path):
    """ResolvedSchema resolves a CONCRETE width for the output leaf, TEST-only."""
    modules, sources, pred_sinks = _gn2_modules(tmp_path)
    with_producer = _with_producer(modules)
    sinks = _per_mode_sinks(pred_sinks)

    test = compile_plan(with_producer, Mode.TEST, sources=sources, sinks=sinks)
    # single-plan bind: the gap §4 risk 6 warns against (a FIT+TEST fixture
    # would mask a missing derived_widths)
    schema = resolve_bind_schema([test])

    output_width = schema.width(_OUTPUT_KEY)
    pred_width = schema.width(_PRED_KEY)
    # the copy preserves the last dim: output width == source pred width
    assert output_width == pred_width
    assert isinstance(output_width, int)
    assert output_width > 0


# GATE (c): EXPOSE INTERACTION — expose:[fit,val] source -> deterministic reject


class _StubModule:
    """A minimal `GraphModule` for the focused expose / pruning controls."""

    def __init__(self, name, requires=None, produces=None):
        self.name = name
        self._requires = requires or {}
        self._produces = produces or {}

    def declare_io(self, mode):
        del mode
        return IO(
            requires=unflatten_spec(self._requires),
            produces=unflatten_spec(self._produces),
        )


def _stub_graph(*, expose_modes):
    """Stub graph: an expose-gated task -> TaskOutput producer -> sink demand."""
    pred = TensorSpec(shape=("B", 5), dtype="float32", modes=expose_modes)
    task = _StubModule("aux_task", produces={_PRED_KEY: pred})
    producer = TaskOutput(task=_SOURCE_TASK, stream=_SOURCE_STREAM)
    producer.name = _PRODUCER_NAME
    modules = {"aux_task": task, _PRODUCER_NAME: producer}
    return modules, {}, [_OUTPUT_KEY]


def test_gate_c_expose_fit_val_source_rejected_in_test(tmp_path):
    """A producer on an expose:[fit,val] task is rejected DETERMINISTICALLY in TEST."""
    del tmp_path
    modules, sources, test_sinks = _stub_graph(expose_modes=Mode.FIT | Mode.VAL)

    with pytest.raises(ConnectivityError) as excinfo:
        compile_plan(modules, Mode.TEST, sources=sources, sinks=test_sinks)
    assert _PRED_KEY in str(excinfo.value)


def test_gate_c_unexposed_source_compiles_in_test(tmp_path):
    """Control: a normally-exposed (modes=ALL) source task compiles fine in TEST."""
    del tmp_path
    modules, sources, test_sinks = _stub_graph(expose_modes=Mode.ALL)

    test = compile_plan(modules, Mode.TEST, sources=sources, sinks=test_sinks)
    assert _PRODUCER_NAME in test.module_names
    assert "aux_task" in test.module_names


# kernel-level demand-pruning control (focused on the pruning mechanism)


def test_producer_demand_pruned_when_output_reaches_no_sink(tmp_path):
    """Pure-kernel control: the producer is pruned exactly when no sink demands it."""
    del tmp_path
    pred = TensorSpec(shape=("B", 5), dtype="float32")
    task = _StubModule("aux_task", produces={_PRED_KEY: pred})
    producer = TaskOutput(task=_SOURCE_TASK, stream=_SOURCE_STREAM)
    producer.name = _PRODUCER_NAME
    modules = {"aux_task": task, _PRODUCER_NAME: producer}
    # per-mode demand so the producer is alive in TEST (avoids all-modes-dead)
    sinks = {Mode.FIT: [_PRED_KEY], Mode.TEST: [_PRED_KEY, _OUTPUT_KEY]}

    fit = compile_plan(modules, Mode.FIT, sources={}, sinks=sinks)
    test = compile_plan(modules, Mode.TEST, sources={}, sinks=sinks)

    assert _PRODUCER_NAME not in fit.module_names  # pruned: no FIT sink
    assert _PRODUCER_NAME in test.module_names  # demanded by the TEST sink


# outputs.* namespace acceptance + the CollectOutputs sink demand surface


def test_outputs_namespace_accepted_by_kernel(tmp_path):
    """The kernel accepts the open ``outputs.*`` namespace (no enum gate)."""
    del tmp_path
    pred = TensorSpec(shape=("B", sym_dim("C", "x")), dtype="float32")
    task = _StubModule("aux_task", produces={_PRED_KEY: pred})
    producer = TaskOutput(task=_SOURCE_TASK, stream=_SOURCE_STREAM)
    producer.name = _PRODUCER_NAME
    modules = {"aux_task": task, _PRODUCER_NAME: producer}

    test = compile_plan(modules, Mode.TEST, sources={}, sinks=[_OUTPUT_KEY])
    produced = {k for step in test.steps for k in step.produces}
    assert _OUTPUT_KEY in produced


def test_collect_outputs_sink_demand_surface():
    """`CollectOutputs.writer_demand` returns the configured outputs.* keys."""
    sink = CollectOutputs(outputs=[_OUTPUT_KEY])
    demand = sink.writer_demand(model_modules={}, reader=None)
    assert set(demand) == {_OUTPUT_KEY}
    assert sink.outputs == (_OUTPUT_KEY,)


def test_collect_outputs_rejects_non_outputs_key():
    """`CollectOutputs` rejects a key outside the outputs namespace."""
    with pytest.raises(ConfigError):
        CollectOutputs(outputs=[_PRED_KEY])


def test_collect_outputs_rejects_empty_and_wildcard():
    """`CollectOutputs` rejects an empty list and a wildcard key."""
    with pytest.raises(ConfigError):
        CollectOutputs(outputs=[])
    with pytest.raises(ConfigError):
        CollectOutputs(outputs=["outputs.tracks.*"])
