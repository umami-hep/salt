"""Export output reduces: declarative bundle-port -> ONNX-output conversions (design §7.3).

Each registry entry binds one resolved `ExportOutput` to a `BoundReduce`
that knows its output names, dtypes and dynamic axes — so the exporter's
``output_names``/``dynamic_axes`` are GENERATED from the export config
instead of v1's hand-ordered list surgery (``to_onnx.py:182-187,243-338``).
The reduce functions run INSIDE the traced graph, at exactly v1's
placement:

- ``split_scalars`` — per-class probability scalars
  (``task.py:285-301``: ``torch.split(probs, 1, -1)`` + squeeze). The task
  already published converted probabilities in ONNX mode (design §3.3), so
  this reduce is a pure splitter.
- ``argmax`` — int8 per-token argmax with the zero-row append/strip trick
  kept verbatim (``to_onnx.py:415-423``; the appended row keeps the trace
  valid for zero-token jets). v1 argmaxes RAW logits where the v2 task
  publishes run_inference-converted probabilities in ONNX mode — argmax is
  invariant under the (masked) softmax, so the int8 output is identical.
- ``vertex_union_find`` — the in-graph union-find on the RAW edge scores
  the vertexing task publishes in ONNX mode (design §3.3 per-family
  exception): ``get_node_assignment_jit`` (still ``@torch.jit.script``,
  fake-pad-track workaround inside, ``union_find.py:151-153``) +
  ``mask_fill_flattened`` + ``.reshape(-1).char()`` — v1's exact chain
  (``to_onnx.py:426-432``).

MaskFormer reduces (``leading_object``/``object_index``) are M5.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import TensorSpec, split_key
from salt.core.onnx.config import ExportOutput
from salt.models.task import mask_fill_flattened
from salt.utils.union_find import get_node_assignment_jit

__all__ = ["BoundReduce", "ReduceCtx", "bind_reduce"]


@dataclass(frozen=True)
class ReduceCtx:
    """Static context for binding reduces (config + plan derived, no tensors).

    `model_name` prefixes every output name; `seq_dyn_axis` maps sequence
    streams to their configured dynamic-axis names (from ``export.inputs``);
    `produced_specs` are the ONNX plan's produced specs, used for static
    class-count validation.
    """

    model_name: str
    seq_dyn_axis: Mapping[str, str]
    produced_specs: Mapping[str, TensorSpec]


@dataclass(frozen=True)
class BoundReduce:
    """One bound output group: traced-graph function + generated ONNX metadata.

    `fn` consumes the executed bundle and returns the group's output
    tensors, in `output_names` order; `dynamic_axes` covers only this
    group's outputs.
    """

    port: str
    output_names: tuple[str, ...]
    dtypes: tuple[str, ...]
    dynamic_axes: Mapping[str, dict[int, str]]
    fn: Callable[[Bundle], tuple[Tensor, ...]]


def bind_reduce(out_cfg: ExportOutput, ctx: ReduceCtx) -> BoundReduce:
    """Bind one resolved `ExportOutput` to its reduce implementation.

    Returns
    -------
    BoundReduce
        The bound reduce.

    Raises
    ------
    ConfigError
        On an unknown reduce key or a statically-detectable mismatch
        (class count, missing sequence stream).
    """
    try:
        binder = _BINDERS[str(out_cfg.reduce)]
    except KeyError:
        raise ConfigError(
            f"export output {out_cfg.port!r}: unknown reduce {out_cfg.reduce!r} — registry: "
            f"{sorted(_BINDERS)} (design §7.3)"
        ) from None
    return binder(out_cfg, ctx)


def _aux_stream(out_cfg: ExportOutput, ctx: ReduceCtx) -> tuple[str, str]:
    """Resolve the sequence stream + dynamic axis of a per-token output.

    The port must follow the ``preds.<stream>.<task>`` convention
    (design §3.3) and the stream must be a declared sequence export input
    (its mask is the all-valid pad mask the reduce consumes, and its
    ``dyn_axis`` names the output's dynamic axis — v1 hardcoded
    ``n_tracks``, ``to_onnx.py:326-334``).

    Returns
    -------
    tuple[str, str]
        ``(stream, dyn_axis)``.

    Raises
    ------
    ConfigError
        On a malformed port or a stream without a sequence export input.
    """
    parts = split_key(out_cfg.port)
    if len(parts) != 3 or parts[0] != "preds":
        raise ConfigError(
            f"export output {out_cfg.port!r}: the {out_cfg.reduce!r} reduce needs a "
            "'preds.<stream>.<task>' port to resolve its sequence stream (design §3.3)"
        )
    stream = parts[1]
    dyn_axis = ctx.seq_dyn_axis.get(stream)
    if dyn_axis is None:
        raise ConfigError(
            f"export output {out_cfg.port!r}: stream {stream!r} has no sequence entry in "
            "export.inputs — per-token reduces need the stream's pad mask and dynamic axis "
            f"(declared sequence streams: {sorted(ctx.seq_dyn_axis)})"
        )
    return stream, dyn_axis


def _bind_split_scalars(out_cfg: ExportOutput, ctx: ReduceCtx) -> BoundReduce:
    """Per-class probability scalars (v1 ``task.py:285-301``).

    Returns
    -------
    BoundReduce
        One float32 scalar output per configured class suffix.

    Raises
    ------
    ConfigError
        When the port's declared class count contradicts ``names``.
    """
    assert out_cfg.names is not None  # resolve_export_config guarantees it
    port = out_cfg.port
    spec = ctx.produced_specs.get(port)
    if spec is not None and spec.shape:
        last = spec.shape[-1]
        if isinstance(last, int) and last != len(out_cfg.names):
            raise ConfigError(
                f"export output {port!r} declares {len(out_cfg.names)} names "
                f"{out_cfg.names} but the port produces {last} classes — one scalar per "
                "class (design §7.3)"
            )
    names = tuple(f"{ctx.model_name}_{suffix}" for suffix in out_cfg.names)

    def fn(b: Bundle) -> tuple[Tensor, ...]:
        probs = b.get(port)  # already converted in ONNX mode (design §3.3)
        return tuple(out.squeeze() for out in torch.split(probs, 1, -1))  # task.py:301

    return BoundReduce(
        port=port,
        output_names=names,
        dtypes=("float32",) * len(names),
        dynamic_axes={},  # global scalars carry no dynamic axis (to_onnx.py:309-338)
        fn=fn,
    )


def _bind_argmax(out_cfg: ExportOutput, ctx: ReduceCtx) -> BoundReduce:
    """int8 per-token argmax with the zero-row trick (v1 ``to_onnx.py:415-423``).

    Returns
    -------
    BoundReduce
        One int8 ``[L]`` output with a dynamic token axis.
    """
    port = out_cfg.port
    _, dyn_axis = _aux_stream(out_cfg, ctx)
    name = f"{ctx.model_name}_{out_cfg.name}"

    def fn(b: Bundle) -> tuple[Tensor, ...]:
        scores = b.get(port)  # [1, L, C] probs (argmax-equivalent to v1's raw logits)
        # zero-row append/strip verbatim (to_onnx.py:418-421): keeps the
        # traced argmax valid for zero-token jets
        scores = torch.concatenate([scores, torch.zeros((1, 1, scores.shape[-1]))], dim=1)
        out = torch.argmax(scores, dim=-1)[:, :-1]
        return (out.squeeze(0).char(),)

    return BoundReduce(
        port=port,
        output_names=(name,),
        dtypes=("int8",),
        dynamic_axes={name: {0: dyn_axis}},
        fn=fn,
    )


def _bind_vertex_union_find(out_cfg: ExportOutput, ctx: ReduceCtx) -> BoundReduce:
    """In-graph union-find on RAW edge scores (v1 ``to_onnx.py:426-432``).

    The vertexing task publishes raw ``[E, 1]`` edge scores in ONNX mode
    (design §3.3 per-family exception); union-find runs INSIDE the traced
    graph on the adapter's all-valid pad mask — `get_node_assignment_jit`
    is ``@torch.jit.script`` and inlines into the trace, with the
    fake-pad-track workaround kept inside (``union_find.py:151-153``).

    Returns
    -------
    BoundReduce
        One int8 ``[L]`` output with a dynamic token axis.
    """
    port = out_cfg.port
    stream, dyn_axis = _aux_stream(out_cfg, ctx)
    name = f"{ctx.model_name}_{out_cfg.name}"
    mask_key = f"masks.{stream}"

    def fn(b: Bundle) -> tuple[Tensor, ...]:
        edge_scores = b.get(port)  # RAW [E, 1] scores (design §3.3)
        pad_mask = b.get(mask_key)  # the adapter's all-valid mask (to_onnx.py:428)
        vertex_indices = get_node_assignment_jit(edge_scores, pad_mask)
        vertex_list = mask_fill_flattened(vertex_indices, pad_mask)
        return (vertex_list.reshape(-1).char(),)

    return BoundReduce(
        port=port,
        output_names=(name,),
        dtypes=("int8",),
        dynamic_axes={name: {0: dyn_axis}},
        fn=fn,
    )


_BINDERS: dict[str, Callable[[ExportOutput, ReduceCtx], BoundReduce]] = {
    "split_scalars": _bind_split_scalars,
    "argmax": _bind_argmax,
    "vertex_union_find": _bind_vertex_union_find,
}
