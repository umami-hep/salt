"""Unit gates for the plan-29 W2 `Combination` conversion node (design §6.2 / Q2).

A combination (e.g. ``pbc = pb + pc``) is a NEW conversion plan node that reads a
SOURCE prob/pred bundle leaf (``outputs.<stream>.<src>``) and produces a NEW
``outputs.<stream>.<name>`` scalar leaf as a weighted sum over the source's
last-dim channels — bitwise-equal to v1's ``pb + pc`` computed on the renamed
scalars, but reading the BUNDLE leaf so the name-space dependency disappears
(user decision 2026-06-22, folding the ``adapter.py:279-280`` combine loop). This
file pins:

- the weighted-sum math (bitwise-equal to the indexed channel adds, in order);
- the demand contract (requires the source ``outputs.*`` leaf, kind=data;
  produces the new ``outputs.<stream>.<name>`` leaf; collapses to a scalar);
- it runs inside the executor (per-batch in TEST, once in the ONNX trace);
- config validation (source must be an ``outputs.*`` leaf, terms non-empty).
"""

from __future__ import annotations

import pytest
import torch

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.executor import Executor
from salt.core.graph.planner import compile_plan
from salt.core.graph.spec import IO, Mode, TensorSpec, flatten_spec, unflatten_spec
from salt.core.outputs import Combination

_SRC = "outputs.jets.jets_classification"


def test_combination_sum_matches_indexed_channel_adds_bitwise():
    """``pbc = probs[..., 0] + probs[..., 1]`` bitwise-equal to the v1 renamed-scalar add.

    The combination reads the SOURCE prob leaf and sums the selected channels in
    `terms` order — the SAME float adds v1 did on the renamed ``pb``/``pc``
    scalars (``to_onnx.py:404-412``), so the result is bit-for-bit identical.
    """
    torch.manual_seed(1)
    probs = torch.rand(5, 3)  # [B, C=3] softmaxed probs (pb, pc, pu)
    node = Combination(source=_SRC, name="pbc", terms={0: 1.0, 1: 1.0})
    node.name = "pbc"
    out = node.forward(
        Bundle({"outputs": {"jets": {"jets_classification": probs.clone()}}}), Mode.TEST
    )
    got = out["outputs.jets.pbc"]
    oracle = probs[..., 0] + probs[..., 1]
    torch.testing.assert_close(got, oracle, rtol=0, atol=0)
    assert got.shape == (5,)  # the last dim collapsed to a scalar


def test_combination_weighted_sum_respects_scales_and_order():
    """A weighted combination ``2*p0 + 0.5*p2`` matches the scaled adds in `terms` order."""
    torch.manual_seed(2)
    probs = torch.rand(4, 4)
    node = Combination(source=_SRC, name="w", terms={0: 2.0, 2: 0.5})
    node.name = "w"
    got = node.forward(
        Bundle({"outputs": {"jets": {"jets_classification": probs.clone()}}}), Mode.TEST
    )["outputs.jets.w"]
    oracle = 2.0 * probs[..., 0] + 0.5 * probs[..., 2]
    torch.testing.assert_close(got, oracle, rtol=0, atol=0)


def test_combination_onnx_and_test_branches_agree():
    """The combination is mode-agnostic — the same sum runs in TEST and ONNX (one path)."""
    torch.manual_seed(3)
    probs = torch.rand(1, 3)
    node = Combination(source=_SRC, name="pbc", terms={0: 1.0, 1: 1.0})
    node.name = "pbc"
    b_test = Bundle({"outputs": {"jets": {"jets_classification": probs.clone()}}})
    b_onnx = Bundle({"outputs": {"jets": {"jets_classification": probs.clone()}}})
    torch.testing.assert_close(
        node.forward(b_test, Mode.TEST)["outputs.jets.pbc"],
        node.forward(b_onnx, Mode.ONNX)["outputs.jets.pbc"],
        rtol=0,
        atol=0,
    )


def test_combination_declare_io_requires_source_produces_new_leaf():
    """``declare_io`` requires the source ``outputs.*`` leaf (kind=data) -> the new leaf."""
    node = Combination(source=_SRC, name="pbc", terms={0: 1.0, 1: 1.0})
    node.name = "pbc"
    io = node.declare_io(Mode.ONNX)
    assert list(flatten_spec(io.requires)) == [_SRC]
    assert list(flatten_spec(io.produces)) == ["outputs.jets.pbc"]
    # the source require is kind=data so it kind-unifies against the producer leaf
    assert flatten_spec(io.requires)[_SRC].kind == "data"


def test_combination_derived_width_collapses_to_one():
    """The combination collapses the source last dim to a single scalar column (design §6.6)."""
    node = Combination(source=_SRC, name="pbc", terms={0: 1.0, 1: 1.0})
    node.name = "pbc"
    assert node.derived_widths({_SRC: 3}) == {"outputs.jets.pbc": 1}


def test_combination_runs_inside_the_executor():
    """The combination is a normal plan node run by the executor (per-batch TEST / trace ONNX)."""

    class _Probs:
        """A toy producer minting the source prob leaf (stands in for ClassProbs)."""

        name = "probs"

        def declare_io(self, mode):
            del mode
            return IO(
                unflatten_spec({"inputs.jets": TensorSpec(shape=("B", 3), dtype="float32")}),
                unflatten_spec({_SRC: TensorSpec(shape=("B", 3), dtype="float32", kind="data")}),
            )

        def __call__(self, b, mode):
            del mode
            return {_SRC: torch.softmax(b.get("inputs.jets"), dim=-1)}

    probs = _Probs()
    combo = Combination(source=_SRC, name="pbc", terms={0: 1.0, 1: 1.0})
    combo.name = "pbc"
    modules = {probs.name: probs, combo.name: combo}
    sources = unflatten_spec({"inputs.jets": TensorSpec(shape=("B", 3), dtype="float32")})
    # demand the combined leaf so both nodes stay alive
    plan = compile_plan(modules, Mode.ONNX, sources, sinks=["outputs.jets.pbc"])
    assert "pbc" in plan.module_names

    raw = torch.randn(4, 3)
    out = Executor(plan).run(Bundle({"inputs": {"jets": raw}}))
    soft = torch.softmax(raw, dim=-1)
    torch.testing.assert_close(
        out.get("outputs.jets.pbc"), soft[..., 0] + soft[..., 1], rtol=0, atol=0
    )


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------


def test_combination_rejects_non_outputs_source():
    """A combination reads a bundle ``outputs.*`` leaf — not a raw prediction."""
    with pytest.raises(ConfigError, match="must be an 'outputs"):
        Combination(source="preds.jets.jets_classification", name="pbc", terms={0: 1.0, 1: 1.0})


def test_combination_rejects_wildcard_source():
    """A combination source is a concrete key (no wildcard, design §2.2)."""
    with pytest.raises(ConfigError, match="wildcard"):
        Combination(source="outputs.jets.*", name="pbc", terms={0: 1.0})


def test_combination_rejects_empty_terms():
    """A combination needs at least one source channel term."""
    with pytest.raises(ConfigError, match="at least one"):
        Combination(source=_SRC, name="pbc", terms={})


def test_combination_rejects_negative_index():
    """A source channel index must be a non-negative int."""
    with pytest.raises(ConfigError, match="non-negative"):
        Combination(source=_SRC, name="pbc", terms={-1: 1.0})
