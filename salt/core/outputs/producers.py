"""Output producers — in-graph `GraphModule`s populating the ``outputs.*`` dict.

Design §2 (layer 1): a producer reads ``preds.<stream>.<task>`` (the raw
training-space prediction a task always publishes) and writes one or more
``outputs.<stream>.<name>`` leaves — the eval/export-ready tensors a sink
serialises. Producers are ordinary model-pipeline modules: ``declare_io(mode)
-> IO`` plus a ``forward(bundle, mode)``, so the existing planner /
executor / bind machinery carries them with no kernel change.

Mode handling (design §2 "no more per-task forward branching"): producers run
ONLY when demanded. Their ports are active in every mode (``modes=ALL``); what
gates them is DEMAND, not a hard mode flag — the keystone the architecture
rests on (design §4 risk 4, gated at P0):

- **FIT/VAL**: nothing demands ``outputs.*`` (losses/metrics read ``preds.*``
  directly), so the planner's demand closure (planner.py ``_demand_closure``)
  drops the producer — it is absent from the FIT/VAL ``plan.steps`` and the
  training path is bitwise-unperturbed.
- **TEST**: a sink demands the ``outputs.*`` leaf, which pulls the producer
  (and transitively its ``preds.*`` source) into the plan.
- **ONNX**: the export ports demand it and the producer is in the traced
  subgraph.

Gating by demand (not by ``modes=TEST|ONNX``) is what makes the absence in
FIT/VAL provably a PRUNE: the producer is genuinely collected as a node and
then dropped because its output reaches no sink, exactly the mechanism the
design relies on. (A hard ``modes`` flag would also hide it, but then a wiring
slip that left it demanded in FIT/VAL could go unnoticed.)

This module ships the generic ``TaskOutput`` producer for the trivial
majority (design §4b: "one generic ``TaskOutput`` producer for the
copy/softmax/argmax/de-scale majority, with dedicated classes only for the
genuinely non-trivial heads"). For P0 (scaffold + demand-gating proof) the op
is an identity copy — a real torch passthrough (``torch.clone``, so the
produced leaf never aliases the source ``preds.*`` leaf, design §2.1). The
conversion ops (softmax, de-scale, ...) land in P1; the demand-gating /
width-resolution contract this class establishes is identical for them.
"""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.spec import IO, Mode, TensorSpec, unflatten_spec

__all__ = ["TaskOutput"]

_UNNAMED = "unnamed"
"""Placeholder instance name — config assembly assigns the dict key (design §2.2)."""


class TaskOutput(nn.Module):
    """Generic producer: ``preds.<stream>.<task>`` -> ``outputs.<stream>.<name>``.

    The copy/softmax/argmax/de-scale majority needs no dedicated class — one
    parameterised producer reads a task's published prediction leaf and writes
    an ``outputs.*`` leaf (design §4b). For P0 the op is an identity copy
    (a real torch passthrough); P1 parameterises it by a conversion op.

    The produced leaf re-emits the upstream prediction's last-dim width via the
    `derived_widths` hook (design §6.6, §4 risk 6): a producer that minted a
    *fresh* symbolic last dim would get no resolved width in a TEST-only bind
    (no FIT plan to unify against), so the H5 sink could not size columns and
    the exporter would have no concrete last dim. `derived_widths` copies the
    bound input width onto the output instead, so the width resolves from the
    TEST plan alone.

    Both ports declare ``shape=None`` (rank-agnostic): a task's prediction is
    ``[B, C]`` for a global head and ``[B, T, C]`` for a sequence head, and the
    generic producer copies either without knowing the rank statically. The
    last-dim width still resolves — the source task's own produce spec carries
    the concrete last dim, which `resolve_bind_schema` records for the
    ``preds.*`` key and `derived_widths` then copies onto the output.

    Parameters
    ----------
    task : str
        The source task's instance name — the producer reads
        ``preds.<stream>.<task>``.
    stream : str
        The stream the source task publishes under (``preds.<stream>.<task>``).
        Also the stream the output is written under
        (``outputs.<stream>.<name>``).
    name : str, optional
        The output leaf name (``outputs.<stream>.<name>``); defaults to `task`
        so a "just copy the task output" producer needs no extra config.
    """

    def __init__(self, task: str, stream: str, name: str | None = None) -> None:
        super().__init__()
        self.name = _UNNAMED
        self.task = task
        self.stream = stream
        self.output_name = name if name is not None else task
        self.pred_key = f"preds.{stream}.{task}"
        self.output_key = f"outputs.{stream}.{self.output_name}"

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``preds.<stream>.<task>`` -> ``outputs.<stream>.<name>``.

        Both ports are active in every mode (``modes=ALL``, the default): the
        producer is gated by DEMAND, not by a hard mode flag (see the module
        docstring), so FIT/VAL drop it via the planner's demand closure rather
        than via mode-inactivity — the absence is then provably a prune (design
        §4 risk 4). Both shapes are ``None`` (rank-agnostic — a global vs
        sequence head differ in rank); the output last-dim width is bound via
        `derived_widths` (design §6.6) from the source task's resolved
        ``preds.*`` width, so it resolves in a TEST-only bind (design §4
        risk 6). ``kind="data"`` (the default) — a prediction tensor, never a
        loss/label leaf — so the producer<-task edge kind-unifies.

        Returns
        -------
        IO
            The declared requires/produces for this producer.
        """
        del mode
        pred_spec = TensorSpec(shape=None, dtype="float32")
        out_spec = TensorSpec(shape=None, dtype="float32")
        return IO(
            requires=unflatten_spec({self.pred_key: pred_spec}),
            produces=unflatten_spec({self.output_key: out_spec}),
        )

    def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
        """Re-emit the bound input prediction width onto the output leaf (design §6.6).

        The bind fixpoint (`salt.core.nn.bind._apply_derived_widths`) calls this
        once the input width is resolved; returning ``{}`` until then keeps the
        hook order-insensitive across plans.

        Returns
        -------
        dict[str, int]
            ``{outputs.<stream>.<name>: width}`` once the input width is
            known, else an empty dict.
        """
        pred_width = widths.get(self.pred_key)
        if pred_width is None:
            return {}
        return {self.output_key: pred_width}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Copy the prediction leaf into the ``outputs.*`` namespace (P0 identity op).

        Returns
        -------
        dict[str, Tensor]
            The newly produced ``outputs.<stream>.<name>`` leaf only
            (design §2.5). A fresh tensor (``clone``) so the output never
            aliases the source ``preds.*`` leaf (write-once, design §2.1).
        """
        del mode
        return {self.output_key: b.get(self.pred_key).clone()}
