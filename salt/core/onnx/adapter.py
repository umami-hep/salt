"""The traceable export wrapper around a compiled ONNX plan (design §7).

`OnnxAdapter` is the `nn.Module` handed to ``torch.onnx.export``: positional
tensor inputs named/ordered by ``export.inputs``, in-wrapper bundle
assembly (sequence ``[L, F]`` -> ``[1, L, F]`` + all-valid pad masks — the
same assumption v1's traced pattern makes, ``to_onnx.py:374,386-390``),
frozen-plan execution through the M1 `Executor`, and a flat output tuple
built by the bound reduces (design §7.3).

Trace-safety notes (verified by the M4 recipe spike): the executor's step
loop is a plain Python sequence of module invocations over MappingProxyType
plan steps — ``torch.onnx.export(dynamo=False)`` neither pickles nor
deep-copies the module, so holding the frozen `Plan` is fine; the bundle's
write-once checks and the dict-valued ``seq.layout`` meta leaf trace
cleanly. Mode is a plan property resolved before tracing, never a tensor
input (design §2.5).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.executor import Executor
from salt.core.graph.planner import Plan
from salt.core.graph.spec import Mode
from salt.core.onnx.config import (
    ExportConfig,
    ExportInput,
    stream_of_input_port,
)

__all__ = ["OnnxAdapter"]


class OnnxAdapter(nn.Module):
    """Traceable export wrapper: positional Athena tensors -> flat ONNX outputs (design §7).

    Parameters
    ----------
    plan : Plan
        The compiled ``Mode.ONNX`` plan (sinks = the `OnnxExportSink`'s
        declared conversion-leaf requires). The step modules are registered
        in an ``nn.ModuleDict`` so their parameters/buffers are visible to
        ``torch.onnx.export``.
    export : ExportConfig
        The RESOLVED export config (`resolve_export_config` output — every
        default filled, ``model_name`` validated).
    feature_fields : Mapping[str, tuple[str, ...]]
        Declared column names per ``inputs.<stream>`` port (the dataset
        `Features` declaration — the ONE place column order is defined,
        design §5.1). Sizes the example inputs (replacing v1
        ``ModelWrapper.input_dims``, ``modelwrapper.py:383-392``) and
        resolves ``alias:`` column gathers at construction time.

    Raises
    ------
    ConfigError
        If the plan is not an ONNX plan, the export config is unresolved,
        or an alias gather names unknown columns.
    """

    def __init__(
        self,
        plan: Plan,
        export: ExportConfig,
        feature_fields: Mapping[str, tuple[str, ...]],
    ) -> None:
        super().__init__()
        if plan.mode is not Mode.ONNX:
            raise ConfigError(
                f"OnnxAdapter needs a Mode.ONNX plan, got {plan.mode.name} (design §7)"
            )
        if export.model_name is None:
            raise ConfigError(
                "OnnxAdapter needs a RESOLVED export config — call resolve_export_config first"
            )
        self.model_name = export.model_name
        self.plan = plan
        # registering the plan modules makes their parameters/buffers visible
        # to torch.onnx.export; the Executor runs the SAME instances (the
        # plan steps hold live references, executor.py)
        self.net = nn.ModuleDict({
            step.name: step.module for step in plan.steps if isinstance(step.module, nn.Module)
        })
        # fail BEFORE tracing on unmaterialised buffers (e.g. a fresh
        # Normaliser): Normaliser.forward skips its eager guard under
        # tracing (TracerWarning hygiene), so the trace would silently bake
        # un-materialised values without this construction-time check
        for name, module in self.net.items():
            flag = getattr(module, "materialised", None)
            if isinstance(flag, Tensor) and not bool(flag.reshape(-1)[0]):
                raise ConfigError(
                    f"module {name!r} carries materialised=False — load a checkpoint or call "
                    "materialise() before export; tracing would bake un-materialised buffer "
                    "values into the ONNX graph (design §2.3)"
                )
        self._executor = Executor(plan)
        self._positional: list[ExportInput] = [e for e in export.inputs if e.alias is None]
        self._aliases: list[ExportInput] = [e for e in export.inputs if e.alias is not None]
        self._fields = {port: tuple(fields) for port, fields in feature_fields.items()}
        self._alias_gathers: list[Tensor | None] = [
            self._resolve_alias_gather(entry) for entry in self._aliases
        ]
        for i, idx in enumerate(self._alias_gathers):
            if idx is not None:
                # buffer (non-persistent): moves with .to()/.float() and is
                # a bind-time constant in the trace (design §7 alias gather)
                self.register_buffer(f"_alias_index_{i}", idx, persistent=False)
        # plan-29 W4 atomic cutover (R8 closed): the folded OnnxExportSink in the
        # plan is the SOLE ONNX-output authority. It NAMES the demanded outputs.*
        # leaves the conversion nodes (ClassProbs/SeqClassIndex/VertexUnionFind/
        # MaskFormerObjects/Combination) minted in the traced executor pass — NO
        # per-batch compute, NO post-executor reduce.fn loop, NO combine loop
        # (combinations are Combination conversion nodes now). The off-graph reduce
        # manifest (export.outputs) is retired; every ONNX output comes from a
        # conversion leaf the sink declares (design §4.2, §6).
        self._export_sink = self._find_export_sink(plan)
        if self._export_sink is None:
            raise ConfigError(
                "OnnxAdapter needs a folded OnnxExportSink in the plan — the off-graph reduce "
                "manifest was retired at plan-29 W4. Declare an OnnxExportSink naming the "
                "conversion outputs.* leaves (design §4.2/§6); the conversion nodes "
                "(ClassProbs/SeqClassIndex/VertexUnionFind/MaskFormerObjects/Combination) own "
                "the math inside the traced graph."
            )
        if self._export_sink.model_name is None:
            self._export_sink.model_name = self.model_name
        self._ordered = [
            (name, dtype, "onnx_export (folded conversion node)")
            for name, dtype in zip(
                self._export_sink.output_names(),
                self._export_sink.output_dtypes(),
                strict=True,
            )
        ]
        # the formal export-mode protocol (design §7.2): every module
        # recursively receives set_export_mode() — e.g. the encoder's
        # attention switch to torch-math (modelwrapper.py:331-335,
        # to_onnx.py:670,700), required for Athena agreement
        self.set_export_mode()
        self.eval()

    # -- generated export metadata (design §7.3 — no hand-ordered lists) -------

    @property
    def input_names(self) -> list[str]:
        """The ONNX graph input names, in positional order.

        Returns
        -------
        list[str]
            One name per positional input (alias entries excluded).
        """
        return [str(entry.name) for entry in self._positional]

    @property
    def output_names(self) -> list[str]:
        """The flat ONNX output names, in declared OnnxExportSink tuple order.

        Returns
        -------
        list[str]
            The generated names (``{model_name}_{suffix}``), ordered by the
            `OnnxExportSink`'s declared leaf list — the single ordering
            authority (globals, combines, per-token aux), independent of the
            executor topo order (design §6.3).
        """
        return [name for name, _, _ in self._ordered]

    @property
    def output_dtypes(self) -> list[str]:
        """Per-output dtypes, aligned with `output_names`.

        Returns
        -------
        list[str]
            ``"float32"`` / ``"int8"`` per output.
        """
        return [dtype for _, dtype, _ in self._ordered]

    @property
    def dynamic_axes(self) -> dict[str, dict[int, str]]:
        """Dynamic-axes mapping for ``torch.onnx.export``.

        Sequence inputs get ``{0: dyn_axis}`` (v1 ``to_onnx.py:319-323``);
        per-token outputs register theirs via their bound reduces.

        Returns
        -------
        dict[str, dict[int, str]]
            Tensor name -> ``{axis: name}``.
        """
        axes: dict[str, dict[int, str]] = {
            str(entry.name): {0: str(entry.dyn_axis)}
            for entry in self._positional
            if entry.sequence
        }
        # folded export-sink per-token outputs (argmax/union-find/object-index int8
        # leaves) register their dynamic axis from the sink's output table (W4)
        axes.update(self._export_sink.dynamic_axes())
        return axes

    def example_inputs(self, sequence_length: int = 40) -> tuple[Tensor, ...]:
        """Random example inputs for tracing, sized from the `Features` declaration.

        Globals are ``[1, F]`` (batch dim kept), sequences ``[L, F]``
        (no batch dim) — the v1 example-input contract
        (``to_onnx.py:209-229``).

        Returns
        -------
        tuple[Tensor, ...]
            One tensor per positional input.
        """
        example: list[Tensor] = []
        for entry in self._positional:
            width = len(self._field_list(entry.port))
            if entry.sequence:
                example.append(torch.rand(sequence_length, width))
            else:
                example.append(torch.rand(1, width))
        return tuple(example)

    # -- the traced forward (design §7) -----------------------------------------

    def forward(self, *args: Tensor) -> tuple[Tensor, ...]:
        """Assemble the bundle, run the frozen plan, reduce to the flat tuple.

        Parameters
        ----------
        *args : Tensor
            Tensors in ``export.inputs`` positional order (alias entries
            consume none): globals ``[1, F]``, sequences ``[L, F]``
            (Athena-style, no batch dim — ``to_onnx.py:340-374``).

        Returns
        -------
        tuple[Tensor, ...]
            Outputs in `output_names` order.
        """
        assert len(args) == len(self._positional), (
            f"expected {len(self._positional)} positional inputs "
            f"({self.input_names}), got {len(args)}"
        )
        b = Bundle()
        for entry, tensor in zip(self._positional, args, strict=True):
            if entry.sequence:
                b.set(entry.port, tensor.unsqueeze(0))  # [L, F] -> [1, L, F] (to_onnx.py:374)
                # all-valid pad mask — the v1 export-time assumption
                # (to_onnx.py:386-390); byte-identical traced pattern
                b.set(
                    f"masks.{stream_of_input_port(entry.port)}",
                    torch.zeros((1, tensor.shape[0]), dtype=torch.bool),
                )
            else:
                assert tensor.dim() == 2, (
                    f"global input {entry.name!r} must be [batch, features] (to_onnx.py:369-371)"
                )
                b.set(entry.port, tensor)
        for i, entry in enumerate(self._aliases):
            source = b.get(str(entry.alias))
            if self._alias_gathers[i] is None:
                b.set(entry.port, source.clone())  # the GN3 pseudo-input (to_onnx.py:377-378)
            else:
                b.set(entry.port, source.index_select(-1, getattr(self, f"_alias_index_{i}")))
        b = self._executor.run(b)
        # plan-29 W4: the OnnxExportSink NAMES the demanded outputs.* leaves the
        # folded conversion nodes minted in the traced executor pass above — NO
        # per-batch compute (only the split_scalars naming split), NO post-executor
        # reduce.fn loop, NO combine loop. The sink is the sole output authority.
        named = self._export_sink.named_outputs(b)
        return tuple(named[name] for name, _, _ in self._ordered)

    # -- helpers -----------------------------------------------------------------

    def set_export_mode(self) -> None:
        """Recursively invoke the ``set_export_mode`` protocol (design §7.2).

        Every registered (sub)module exposing a callable ``set_export_mode``
        receives it — the formal replacement for v1's name-based
        ``change_attn_backends`` recursion.
        """
        for module in self.net.modules():
            hook = getattr(module, "set_export_mode", None)
            if callable(hook):
                hook()

    def _field_list(self, port: str) -> tuple[str, ...]:
        """The declared columns of an ``inputs.<stream>`` port.

        Returns
        -------
        tuple[str, ...]
            The column names.

        Raises
        ------
        ConfigError
            When the port has no `Features` declaration.
        """
        try:
            return self._fields[port]
        except KeyError:
            raise ConfigError(
                f"export input port {port!r} has no Features variable declaration — every "
                "export input must be a declared dataset feature stream (design §7; known: "
                f"{sorted(self._fields)})"
            ) from None

    @staticmethod
    def _find_export_sink(plan: Plan) -> Any:
        """Find the folded `OnnxExportSink` among the plan steps (the output authority, W4).

        Returns
        -------
        OnnxExportSink | None
            The single export sink in the ONNX plan, or None when no sink is
            wired (which the adapter rejects — the off-graph reduce manifest
            was retired at W4).
        """
        from salt.core.outputs import OnnxExportSink  # noqa: PLC0415 - heavy/circular

        for step in plan.steps:
            if isinstance(step.module, OnnxExportSink):
                return step.module
        return None

    def _resolve_alias_gather(self, entry: ExportInput) -> Tensor | None:
        """Resolve an alias entry to its column gather (or None = identity clone).

        Identity when the two ports' `Features` declarations are equal
        (GN3V01's case); otherwise a name-resolved ``index_select`` gather
        with bind-time constant indices (design §7).

        Returns
        -------
        Tensor | None
            int64 gather indices, or None for the identity clone.

        Raises
        ------
        ConfigError
            When a target column is missing from the alias source.
        """
        src_fields = self._field_list(str(entry.alias))
        dst_fields = self._field_list(entry.port)
        if src_fields == dst_fields:
            return None
        missing = [name for name in dst_fields if name not in src_fields]
        if missing:
            raise ConfigError(
                f"export input {entry.port!r}: alias source {entry.alias!r} lacks columns "
                f"{missing} — the alias gather resolves by name from the Features "
                f"declarations (design §7; source columns: {list(src_fields)})"
            )
        return torch.tensor([src_fields.index(name) for name in dst_fields], dtype=torch.int64)
