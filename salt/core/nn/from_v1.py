"""Build a v2 GN2 module graph that SHARES instances with a v1 `ModelWrapper`.

`from_v1` is the weight-sharing half of the forward-parity gate (plan 04):
it wraps the live v1 submodules (`InputNorm`, `InitNet`s, `Transformer`,
`GlobalAttentionPooling`, task heads) in GraphModules without copying a
single weight, so any output difference between the v1 forward and the
executed plan is a wiring bug by construction.

`v1_sources` / `v1_sinks` derive the matching plan boundary (the dataset-side
``inputs.*`` / ``masks.*`` leaves and the ``preds.*`` sink anchors) from the
same v1 instance, so the harness compiles with one source of truth.
"""

from __future__ import annotations

from salt.core.graph.spec import GraphModule, NestedSpec, TensorSpec, sym_dim, unflatten_spec
from salt.core.nn.wrappers import (
    Concat,
    ConstituentTask,
    GlobalObjectTask,
    Normaliser,
    Pooling,
    StreamEmbed,
    TransformerEncoder,
)
from salt.modelwrapper import ModelWrapper

__all__ = ["from_v1", "v1_sinks", "v1_sources"]


def _check_scope(wrapper: ModelWrapper) -> None:
    """Reject v1 features outside the GN2 parity scope.

    Raises
    ------
    ValueError
        On edge constructors/init nets, mask decoders, merged streams,
        global featurewise transforms, or a missing encoder/pool net.
    """
    model = wrapper.model
    if wrapper.edge_constructor is not None:
        raise ValueError("from_v1: edge_constructor is outside the GN2 parity scope")
    if model.edge_init_nets:
        raise ValueError("from_v1: edge_init_nets are outside the GN2 parity scope")
    if model.mask_decoder is not None:
        raise ValueError("from_v1: mask_decoder is outside the GN2 parity scope")
    if model.merge_dict is not None:
        raise ValueError("from_v1: merge_dict is outside the GN2 parity scope")
    if getattr(model, "featurewise_global", None):
        raise ValueError("from_v1: global featurewise transforms are outside the parity scope")
    if model.encoder is None or model.pool_net is None:
        raise ValueError("from_v1: the GN2 parity graph needs both an encoder and a pool_net")


def from_v1(wrapper: ModelWrapper) -> dict[str, GraphModule]:
    """Build the v2 GN2 module dict by wrapping the v1 model's live submodules.

    Every wrapper holds a REFERENCE to the corresponding v1 ``nn.Module``
    (``wrapper.norm``, ``wrapper.model.init_nets[*]``, ``.encoder``,
    ``.pool_net``, ``.tasks[*]``) — weights identical by construction. The
    glue between them (init dispatch, encoder tuple unpack, pooling over the
    post-register masks, three-way task routing) is reproduced as graph
    edges; each load-bearing ordering is commented at the wrapper that owns
    it, citing the v1 line it reproduces.

    No `Split` is included: the v1 task heads slice the full sequence
    internally, and a `Split` on the task path would change GEMM shapes and
    risk breaking bitwise parity (stage-1 recipe; `salt.core.nn.Split`
    docstring).

    Parameters
    ----------
    wrapper : ModelWrapper
        A constructed v1 model (its ``__init__`` monkey-patches
        ``task.global_object`` onto every task, modelwrapper.py:111-114 —
        the task routing below relies on it, as v1's run_tasks does at
        saltmodel.py:216).

    Returns
    -------
    dict[str, GraphModule]
        Instance-named modules for `compile_plan`: ``norm``,
        ``<stream>_embed`` per init net, ``concat``, ``encoder``, ``pool``,
        and one module per task (keyed by the task's own name).

    Raises
    ------
    ValueError
        If the v1 model uses features outside the GN2 parity scope, or on
        a module-name collision.
    """
    _check_scope(wrapper)
    model = wrapper.model

    # Sequence-stream order = init_nets order: v1 builds xs in this order
    # (saltmodel.py:128-129) and the encoder concatenates in dict insertion
    # order (transformer.py:684-686).
    streams = tuple(init_net.input_name for init_net in model.init_nets)

    modules: dict[str, GraphModule] = {"norm": Normaliser(wrapper.norm)}
    for init_net in model.init_nets:
        modules[f"{init_net.input_name}_embed"] = StreamEmbed(init_net)
    modules["concat"] = Concat(streams)
    modules["encoder"] = TransformerEncoder(model.encoder)
    modules["pool"] = Pooling(model.pool_net)

    for task in model.tasks:
        if task.name in modules:
            raise ValueError(f"from_v1: task name {task.name!r} collides with a module name")
        # v1 three-way routing (saltmodel.py:215-223): the "objects" branch
        # needs a mask_decoder, which _check_scope rejected — two ways left.
        modules[task.name] = (
            GlobalObjectTask(task)
            if task.input_name == task.global_object
            else ConstituentTask(task, streams)
        )

    # Instance name = dict key (design §2.2); the executor enforces the match
    # (executor.py:96-99).
    for key, module in modules.items():
        module.name = key
    return modules


def v1_sources(wrapper: ModelWrapper) -> NestedSpec:
    """Derive the plan sources (dataset boundary) from the v1 model.

    Mirrors the v1 batch keying (datasets.py:435-441, 448-559): one
    ``inputs.<stream>`` data leaf per normed stream — ``("B", F)`` for the
    global object, ``("B", "T:<stream>", F)`` for sequence streams — plus a
    ``masks.<stream>`` pad-mask leaf (True = padded, datasets.py:523) for
    sequence streams only.

    Returns
    -------
    NestedSpec
        The nested source spec for `compile_plan`.
    """
    norm = wrapper.norm
    flat: dict[str, TensorSpec] = {}
    for stream, variables in norm.variables.items():
        fields = tuple(variables)
        if stream == norm.global_object:
            flat[f"inputs.{stream}"] = TensorSpec(
                shape=("B", len(fields)), dtype="float32", fields=fields
            )
        else:
            flat[f"inputs.{stream}"] = TensorSpec(
                shape=("B", sym_dim("T", stream), len(fields)), dtype="float32", fields=fields
            )
            flat[f"masks.{stream}"] = TensorSpec(
                shape=("B", sym_dim("T", stream)), dtype="bool", kind="pad_mask"
            )
    return unflatten_spec(flat)


def v1_sinks(wrapper: ModelWrapper) -> list[str]:
    """Derive the plan sinks: one ``preds.<stream>.<task>`` anchor per v1 task.

    These are the parity comparison targets — exactly the leaves
    ``ModelWrapper.forward`` nests into its preds dict (saltmodel.py:224-226).

    Returns
    -------
    list[str]
        Dotted sink keys for `compile_plan`.
    """
    return [f"preds.{task.input_name}.{task.name}" for task in wrapper.model.tasks]
