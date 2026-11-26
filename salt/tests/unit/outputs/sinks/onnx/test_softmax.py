"""Unit tests for the Softmax decomposition applied to exported graphs."""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from salt.outputs.sinks.onnx.softmax import decompose_softmax


@pytest.mark.parametrize("axis", [-1, 1])
def test_decompose_softmax_keeps_outputs(axis):
    """The decomposed graph has no Softmax node and matches the onnxruntime Softmax."""
    graph = helper.make_graph(
        nodes=[helper.make_node("Softmax", ["x"], ["y"], axis=axis)],
        name="softmax",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 40, 40])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 40, 40])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    expected_session = ort.InferenceSession(model.SerializeToString())

    decomposed = decompose_softmax(model)
    onnx.checker.check_model(decomposed)
    assert "Softmax" not in {node.op_type for node in decomposed.graph.node}

    x = {"x": np.random.default_rng(42).normal(0, 5, (2, 40, 40)).astype(np.float32)}
    session = ort.InferenceSession(decomposed.SerializeToString())
    np.testing.assert_allclose(
        session.run(None, x)[0], expected_session.run(None, x)[0], rtol=1e-6, atol=1e-7
    )
