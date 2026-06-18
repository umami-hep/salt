"""Export output reduces: the LIVE registry of bundle-port -> ONNX-output conversions (design §7.3).

This is the M5 ``register_reduce`` deliverable (M4.5 amendment 555-567): the
export reduce set is a LIVE registry — `register_reduce` binds a reduce *name*
to a binder + its DECLARED output dtype + a per-token flag, and
`salt.core.onnx.config` validates ``export.outputs`` against it (and defaults
each entry's dtype from the registered declaration) instead of the frozen
``KNOWN_REDUCES`` tuple it carried through M4.5. A custom (e.g. export-only)
writer can now ship its own export math: register a reduce once, then name it
in `ExportOutput.reduce` — the field knowledge (output names, dtype, dynamic
axes) has one owner, the registered binder. This unblocks sub-wave C's two new
MaskFormer reduces (``leading_object``/``object_index``), which register here.

Each entry binds one resolved `ExportOutput` to a `BoundReduce` that knows its
output names, dtypes and dynamic axes — so the exporter's
``output_names``/``dynamic_axes`` are GENERATED from the export config instead
of v1's hand-ordered list surgery (``to_onnx.py:182-187,243-338``). The reduce
functions run INSIDE the traced graph, at exactly v1's placement. The three
SHIPPED reduces (registered at import via `register_reduce`, byte-unchanged
from M4):

- ``split_scalars`` — per-class probability scalars (float32, global)
  (``task.py:285-301``: ``torch.split(probs, 1, -1)`` + squeeze). The task
  already published converted probabilities in ONNX mode (design §3.3), so
  this reduce is a pure splitter.
- ``argmax`` — int8 per-token argmax with the zero-row append/strip trick
  kept verbatim (``to_onnx.py:415-423``; the appended row keeps the trace
  valid for zero-token jets). v1 argmaxes RAW logits where the v2 task
  publishes run_inference-converted probabilities in ONNX mode — argmax is
  invariant under the (masked) softmax, so the int8 output is identical.
- ``vertex_union_find`` — int8 per-token in-graph union-find on the RAW edge
  scores the vertexing task publishes in ONNX mode (design §3.3 per-family
  exception): ``get_node_assignment_jit`` (still ``@torch.jit.script``,
  fake-pad-track workaround inside, ``union_find.py:151-153``) +
  ``mask_fill_flattened`` + ``.reshape(-1).char()`` — v1's exact chain
  (``to_onnx.py:426-432``).

The two MaskFormer reduces register here too (sub-wave C, plan 10), composing
v1's `get_maskformer_outputs` (null suppression + pT reorder + index math,
maskformer.py:244-349) byte-faithfully:

- ``leading_object`` — R float32 GLOBAL scalars, the leading object's de-scaled
  regression values (v1 ``to_onnx.py:461-468``). The object-regression task
  already publishes DE-SCALED predictions in ONNX mode (the TEST|ONNX de-scaling
  branch, ``tasks.py:1063-1067``), so this reduce reorders + selects, never
  re-inverting scaling.
- ``object_index`` — int8 PER-TOKEN constituent->object index (v1
  ``indices.reshape(-1).char()``, ``to_onnx.py:469``); its single suffix is the
  writer-declared `OBJECT_INDEX.onnx` (``HadronIndex``).

Both are declared on ONE port (``preds.<object_stream>.regression`` /
``<object_stream>.masks``) yet additionally read the OTHER object keys from the
executed bundle — the decoder is kept alive by the ``object_index`` ``masks``
sink, so ALL its products (class_probs/masks/embed) are present.

Torch-free seam: `salt.core.onnx.config`'s MODULE BODY introduces no
top-level torch or registry import — it does NOT import this module at top
level (this module imports ``config.ExportOutput``, so the dependency is
one-way). It queries the live registry through `registered_reduces` /
`reduce_dtype` / `per_token_reduces` via a DEFERRED import inside its
export-only resolution path (``config._resolve_output`` /
``combine_insertion_index`` / the ``KNOWN_REDUCES`` lazy attribute) — so this
torch-importing registry is reached only when an `export:` block is actually
resolved, never at fit-time parse. (A bare ``import salt.core.onnx.config``
still pulls torch into ``sys.modules`` transitively via the
``salt.core.onnx`` package ``__init__``, which imports the torch-using
adapter; the seam is config.py's own torch-free module body and the
deferred-import discipline, not the whole import path.)
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
from salt.models.maskformer import get_maskformer_outputs as v1_get_maskformer_outputs
from salt.models.task import mask_fill_flattened
from salt.core.utils.union_find import get_node_assignment_jit

__all__ = [
    "BoundReduce",
    "ReduceCtx",
    "ReduceSpec",
    "bind_reduce",
    "per_token_reduces",
    "reduce_dtype",
    "register_reduce",
    "registered_reduces",
    "unregister_reduce",
]


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


Binder = Callable[[ExportOutput, "ReduceCtx"], BoundReduce]
"""A reduce binder: resolved `ExportOutput` + static `ReduceCtx` -> `BoundReduce`."""


@dataclass(frozen=True)
class ReduceSpec:
    """One LIVE registry entry: a reduce's binder + its DECLARED output dtype.

    `register_reduce` builds and registers these; `bind_reduce` looks up the
    binder by name, and `salt.core.onnx.config._resolve_output` reads `dtype`
    (the reduce's declared ONNX output dtype) and `per_token` (whether the
    reduce emits dynamic-axis outputs) to default + validate the manifest entry
    without hard-coding the per-reduce dtype rules it carried through M4.5.

    Parameters
    ----------
    name : str
        The registry key named in `ExportOutput.reduce`.
    binder : Binder
        Builds the traced-graph `BoundReduce` for one resolved entry.
    dtype : str
        The reduce's declared ONNX output dtype (``float32``/``int8``) — the
        single owner of the per-reduce dtype rule (e.g. v1 ``.char()`` int8 for
        the aux reduces, ``to_onnx.py:422,432``). `_resolve_output` defaults an
        entry's ``dtype`` from this and rejects any other declared dtype.
    per_token : bool
        Whether the reduce emits per-token (dynamic sequence axis) outputs —
        the "sequence-aux" entries of v1's output order. `config`'s
        `combine_insertion_index` keys off this set (amendment merge
        condition 5: combines insert before the first per-token entry).
    expects_names : bool
        Whether the reduce consumes the plural ``names`` field (per-class
        scalars, ``split_scalars``) vs the singular ``name`` field (the single-
        output reduces). Drives `_resolve_output`'s name/names exclusivity rule.
    """

    name: str
    binder: Binder
    dtype: str
    per_token: bool
    expects_names: bool


_REGISTRY: dict[str, ReduceSpec] = {}
"""The live reduce registry — the M5 replacement for the frozen ``KNOWN_REDUCES``."""


def register_reduce(
    name: str,
    binder: Binder,
    *,
    dtype: str,
    per_token: bool = False,
    expects_names: bool = False,
) -> None:
    """Register a reduce under `name` with its declared dtype (the M5 public surface).

    The single entry point that adds a reduce to the LIVE registry validated by
    `salt.core.onnx.config`. Shipped reduces register at import; custom /
    export-only writers (and sub-wave C's MaskFormer reduces) register their own
    export math the same way — the field knowledge (binder, dtype, per-token
    placement) lives with the registration, not in a frozen config tuple.

    Parameters
    ----------
    name : str
        The registry key named in `ExportOutput.reduce`. Must be unique.
    binder : Binder
        The binder building the traced-graph `BoundReduce` (see `bind_reduce`).
    dtype : str
        The reduce's declared ONNX output dtype (``float32``/``int8``).
    per_token : bool, optional
        Whether the reduce emits dynamic-axis (per-token) outputs, by default
        False (a global output). Drives the combine-insertion order in
        `config.combine_insertion_index`.
    expects_names : bool, optional
        Whether the reduce consumes the plural ``names`` field (per-class
        scalars) rather than the singular ``name``, by default False.

    Raises
    ------
    ConfigError
        On a non-string/empty name, an unknown declared dtype, or a duplicate
        registration (re-registering a name is a programming error — there is
        no silent override).
    """
    if not isinstance(name, str) or not name:
        raise ConfigError(f"register_reduce: name must be a non-empty string, got {name!r}")
    if dtype not in {"float32", "int8"}:
        raise ConfigError(
            f"register_reduce {name!r}: dtype must be 'float32' or 'int8', got {dtype!r} "
            "(the ONNX output dtypes salt2 export supports — adapter.output_dtypes/check.py:149)"
        )
    if name in _REGISTRY:
        raise ConfigError(
            f"register_reduce: reduce {name!r} is already registered "
            f"(registry: {sorted(_REGISTRY)}) — names are unique, no silent override"
        )
    _REGISTRY[name] = ReduceSpec(
        name=name,
        binder=binder,
        dtype=dtype,
        per_token=per_token,
        expects_names=expects_names,
    )


def unregister_reduce(name: str) -> None:
    """Remove a reduce from the live registry (the symmetric cleanup surface).

    The registry is process-global module state. A consumer that registers a
    TRANSIENT reduce (a gate probe, a test fixture, a notebook experiment) must
    remove it again so it leaves no residue — a leaked entry would make the
    registered set order-dependent for everything sharing the process. No-ops if
    the name is not registered (idempotent teardown).

    Parameters
    ----------
    name : str
        The reduce name to remove.
    """
    _REGISTRY.pop(name, None)


def registered_reduces() -> tuple[str, ...]:
    """The registered reduce names, sorted (the live replacement for ``KNOWN_REDUCES``).

    Returns
    -------
    tuple[str, ...]
        Sorted registry keys.
    """
    return tuple(sorted(_REGISTRY))


def reduce_spec(name: str) -> ReduceSpec:
    """Look up one registered reduce spec by name.

    Returns
    -------
    ReduceSpec
        The registered spec.

    Raises
    ------
    ConfigError
        When `name` is not a registered reduce.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ConfigError(
            f"unknown reduce {name!r} — registry: {registered_reduces()} "
            "(register it via salt.core.onnx.reduces.register_reduce; design §7.3)"
        ) from None


def reduce_dtype(name: str) -> str:
    """The DECLARED ONNX output dtype of a registered reduce.

    `reduce_spec` raises `ConfigError` for an unregistered name.

    Returns
    -------
    str
        ``float32`` or ``int8``.
    """
    return reduce_spec(name).dtype


def per_token_reduces() -> tuple[str, ...]:
    """The registered reduces that emit per-token (dynamic-axis) outputs, sorted.

    The live replacement for the frozen ``PER_TOKEN_REDUCES`` tuple; consumed by
    `config.combine_insertion_index` (combines insert before the first per-token
    entry, the v1 order, amendment merge condition 5).

    Returns
    -------
    tuple[str, ...]
        Sorted per-token reduce names.
    """
    return tuple(sorted(name for name, spec in _REGISTRY.items() if spec.per_token))


def bind_reduce(out_cfg: ExportOutput, ctx: ReduceCtx) -> BoundReduce:
    """Bind one resolved `ExportOutput` to its registered reduce implementation.

    `reduce_spec` raises `ConfigError` on an unknown reduce key; the registered
    binder raises it on a statically-detectable mismatch (class count, missing
    sequence stream).

    Returns
    -------
    BoundReduce
        The bound reduce.
    """
    spec = reduce_spec(str(out_cfg.reduce))
    return spec.binder(out_cfg, ctx)


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


def _object_stream(port: str) -> str:
    """The object stream name from a MaskFormer reduce's declared port.

    The two MaskFormer reduces are declared on ``preds.<object_stream>.regression``
    (``leading_object``) and ``<object_stream>.masks`` (``object_index``); both
    additionally read the OTHER object keys from the bundle. The object stream is
    the dotted component preceding the trailing leaf (``objects`` for both shipped
    ports), so a single convention recovers it from either declared port.

    Returns
    -------
    str
        The object stream name (e.g. ``objects``).

    Raises
    ------
    ConfigError
        On a port too short to carry an object stream.
    """
    parts = split_key(port)
    # preds.<stream>.<task> -> parts[1]; <stream>.<leaf> -> parts[-2]
    if len(parts) == 3 and parts[0] == "preds":
        return parts[1]
    if len(parts) == 2:
        return parts[0]
    raise ConfigError(
        f"export output {port!r}: the MaskFormer object reduces need a "
        "'preds.<object_stream>.regression' (leading_object) or '<object_stream>.masks' "
        "(object_index) port to resolve the object stream (design §7.3)"
    )


def _maskformer_objects(b: Bundle, stream: str, reg_key: str) -> dict[str, Tensor]:
    """Gather the v1 ``objects`` dict (class_probs / masks / regression) from the bundle.

    The shape `get_maskformer_outputs` consumes (maskformer.py:294-297). The
    object-regression predictions are read from `reg_key` (the
    writer-declared ``preds.<stream>.<regression_task>`` port — threaded in so
    the regression key is NEVER hardcoded to the default ``regression`` task
    name; a non-default ``regression_task`` would otherwise fetch a stale/absent
    key at trace time). They are read DE-SCALED — the v2 ``RegressionTaskModule``
    already inverts the per-target scaling in ONNX mode (tasks.py:1063-1067, the
    TEST|ONNX de-scaling branch), exactly the values v1's exporter held after its
    own per-target ``scaler.inverse`` loop (to_onnx.py:454-459). Tensors are
    CLONED: v1's `get_maskformer_outputs` mutates ``masks``/``regression`` in
    place (null suppression, pT reorder), and the v2 bundle is write-once/shared.

    Returns
    -------
    dict[str, Tensor]
        ``{"class_probs", "masks", "regression"}`` for `get_maskformer_outputs`.
    """
    return {
        "class_probs": b.get(f"{stream}.class_probs").clone(),
        "masks": b.get(f"{stream}.masks").clone(),
        "regression": b.get(reg_key).clone(),
    }


def _bind_leading_object(out_cfg: ExportOutput, ctx: ReduceCtx) -> BoundReduce:
    """Per-target leading-object regression scalars (v1 ``to_onnx.py:461-468``).

    The GLOBAL half of v1's MaskFormer object outputs: null objects are
    suppressed and the objects re-ordered so the leading object (highest
    ``regression[..., 0]``, the pT in vertexing) is first, then the leading
    object's R de-scaled regression values are emitted as R squeezed float32
    scalars (v1 ``for r in leading_reg[0]: onnx_outputs += (r,)``). The R suffixes
    are the writer-declared per-target leading names (``MaskFormerObjectWriter``).

    Returns
    -------
    BoundReduce
        R float32 global scalar outputs (one per leading-object regression target).

    Raises
    ------
    ConfigError
        When the port produces a regression width contradicting ``names``.
    """
    assert out_cfg.names is not None  # split_scalars-style plural names
    port = out_cfg.port
    stream = _object_stream(port)
    spec = ctx.produced_specs.get(port)
    if spec is not None and spec.shape:
        last = spec.shape[-1]
        if isinstance(last, int) and last != len(out_cfg.names):
            raise ConfigError(
                f"export output {port!r} declares {len(out_cfg.names)} leading-object names "
                f"{out_cfg.names} but the object-regression port produces {last} targets — one "
                "leading scalar per regression target (v1 to_onnx.py:461-468)"
            )
    names = tuple(f"{ctx.model_name}_{suffix}" for suffix in out_cfg.names)

    def fn(b: Bundle) -> tuple[Tensor, ...]:
        # the leading_object port IS the object-regression port, so the regression
        # is read from the DECLARED port (no hardcoded task name) — port-faithful
        leading_reg, _, _, _ = v1_get_maskformer_outputs(
            _maskformer_objects(b, stream, port), apply_reorder=True
        )  # leading_reg: [B, R] (maskformer.py:344)
        # v1 emits leading_reg[0][r] per target (B == 1 in the traced export);
        # squeeze the per-target column to a scalar output (to_onnx.py:467-468)
        return tuple(leading_reg[0, i] for i in range(len(names)))

    return BoundReduce(
        port=port,
        output_names=names,
        dtypes=("float32",) * len(names),
        dynamic_axes={},  # global leading-object scalars carry no dynamic axis
        fn=fn,
    )


def _bind_object_index(out_cfg: ExportOutput, ctx: ReduceCtx) -> BoundReduce:
    """int8 per-constituent object index (v1 ``to_onnx.py:462,469``).

    The PER-TOKEN half of v1's MaskFormer object outputs: after the same
    null-suppression + pT reorder, each constituent's owning object is emitted as
    an int8 index (``-2`` where no object claims the constituent — v1
    ``indices_from_mask`` noindex, mask_utils.py:97-148), then
    ``indices.reshape(-1).char()`` (to_onnx.py:469). The single suffix is the
    writer-declared `OBJECT_INDEX.onnx` (``HadronIndex``); the dynamic axis is the
    sole declared sequence stream's (v1 ``aux_sequence_object = "tracks"``).

    Returns
    -------
    BoundReduce
        One int8 ``[L]`` output with a dynamic constituent-token axis.

    Raises
    ------
    ConfigError
        When no (or more than one) sequence export input is declared — the
        constituent token axis is ambiguous otherwise (v1 had one fixed
        ``aux_sequence_object``).
    """
    port = out_cfg.port
    stream = _object_stream(port)
    if len(ctx.seq_dyn_axis) != 1:
        raise ConfigError(
            f"export output {port!r}: the object_index reduce needs EXACTLY one sequence "
            f"export input to name its constituent token axis, got {sorted(ctx.seq_dyn_axis)} "
            "(v1's fixed aux_sequence_object='tracks', to_onnx.py:469) — declare the "
            "constituent stream as the single sequence export.input"
        )
    dyn_axis = next(iter(ctx.seq_dyn_axis.values()))
    name = f"{ctx.model_name}_{out_cfg.name}"

    # object_index is declared on '<stream>.masks' (not the regression port), so its
    # own ExportOutput cannot carry the regression task name. It reorders by
    # regression[..., 0] like leading_object, so it reads the default
    # 'preds.<stream>.regression' key — kept correct by MaskFormerObjectWriter
    # asserting regression_task == 'regression' at onnx_outputs() time (a loud
    # config-resolution failure, never a stale-key trace error).
    reg_key = f"preds.{stream}.regression"

    def fn(b: Bundle) -> tuple[Tensor, ...]:
        _, indices, _, _ = v1_get_maskformer_outputs(
            _maskformer_objects(b, stream, reg_key), apply_reorder=True
        )  # indices: [B, L] dense object index per constituent (maskformer.py:347)
        return (indices.reshape(-1).char(),)  # v1 to_onnx.py:469

    return BoundReduce(
        port=port,
        output_names=(name,),
        dtypes=("int8",),
        dynamic_axes={name: {0: dyn_axis}},
        fn=fn,
    )


# The three SHIPPED reduces, registered through the public M5 surface at import
# (byte-unchanged binders, dtypes/per-token flags matching the frozen M4.5
# KNOWN_REDUCES/PER_TOKEN_REDUCES they replace):
#   split_scalars     float32, global,   plural names  (per-class probability scalars)
#   argmax            int8,    per-token, single name  (v1 .char(), to_onnx.py:422)
#   vertex_union_find int8,    per-token, single name  (v1 .char(), to_onnx.py:432)
register_reduce(
    "split_scalars", _bind_split_scalars, dtype="float32", per_token=False, expects_names=True
)
register_reduce("argmax", _bind_argmax, dtype="int8", per_token=True)
register_reduce("vertex_union_find", _bind_vertex_union_find, dtype="int8", per_token=True)

# The two MaskFormer object reduces (M5 sub-wave C, plan 10) — the v1 object
# outputs the exporter appended after the per-stream tasks (to_onnx.py:445-469):
#   leading_object   float32, global,   plural names  (R leading-object reg scalars)
#   object_index     int8,    per-token, single name  (v1 indices.char(), to_onnx.py:469)
register_reduce(
    "leading_object", _bind_leading_object, dtype="float32", per_token=False, expects_names=True
)
register_reduce("object_index", _bind_object_index, dtype="int8", per_token=True)
