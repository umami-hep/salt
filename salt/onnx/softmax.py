import onnx
from onnx import helper


def decompose_softmax(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace Softmax nodes by ReduceMax, Sub, Exp, ReduceSum and Div.

    The onnxruntime Softmax kernel sums in a different order with and without
    AVX-512, so its outputs depend on the CPU, see ATLINFR-6129. Athena applies
    the same decomposition when it loads older models, see
    https://gitlab.cern.ch/atlas/athena/-/merge_requests/91915.

    Parameters
    ----------
    model : onnx.ModelProto
        Exported model with opset 18 or later, modified in place.

    Returns
    -------
    onnx.ModelProto
        The model with each Softmax node of the main graph replaced.
    """
    nodes = []
    for node in model.graph.node:
        if node.op_type != "Softmax" or node.domain not in {"", "ai.onnx"}:
            nodes.append(node)
            continue
        axis = next((a.i for a in node.attribute if a.name == "axis"), -1)
        x, y = node.input[0], node.output[0]
        axes, max_, shifted, exp, total = (
            f"{y}/softmax_{s}" for s in ("axes", "max", "shifted", "exp", "sum")
        )
        nodes += [
            helper.make_node("Constant", [], [axes], name=axes, value_ints=[axis]),
            helper.make_node("ReduceMax", [x, axes], [max_], name=max_, keepdims=1),
            helper.make_node("Sub", [x, max_], [shifted], name=shifted),
            helper.make_node("Exp", [shifted], [exp], name=exp),
            helper.make_node("ReduceSum", [exp, axes], [total], name=total, keepdims=1),
            helper.make_node("Div", [exp, total], [y], name=y),
        ]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model
