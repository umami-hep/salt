"""The ONNX sink — `OnnxExportSink` + its per-output `OnnxExportLeaf` config."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import Tensor

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import IO, KEY_SEP, Mode, TensorSpec, flatten_spec, unflatten_spec
from salt.outputs.output_schema import _OUTPUTS_NAMESPACE
from salt.outputs.sink import Node


@dataclass(frozen=True)
class OnnxExportLeaf:
    """One ONNX output: the conversion ``outputs.*`` leaf + its Athena naming.

    The conversion already ran in the trace, so the sink does no per-batch
    compute — it just NAMES the demanded conversion leaves into the flat
    Athena output tuple.

    Three leaf shapes:

    - **split_scalars** (``names`` plural, float32 global): one converted
      prob leaf (``ClassProbs`` already softmaxed) -> N named scalars. The
      split is a naming concern owned here (``torch.split(probs, 1, -1)`` +
      squeeze), not a new conversion node.
    - **single per-token leaf** (``name`` singular, int8, ``per_token=True``):
      the conversion node (e.g. ``SeqClassIndex``'s ONNX branch) already
      produced the int8 ``[L]`` leaf — the sink passes it through under its
      Athena name and registers its dynamic axis.
    - **single global leaf** (``name`` singular, float32, ``per_token=False``):
      a ``Combination`` node's scalar leaf — passed through under its Athena
      name, no dynamic axis.

    Parameters
    ----------
    key : str
        The ``outputs.<stream>.<name>`` conversion leaf this output names.
        Concrete, under the ``outputs`` namespace.
    name : str | None, optional
        Single-output Athena suffix (full name ``{model_name}_{name}``).
        Exclusive with `names`. When both `name` and `names` are omitted the
        suffix defaults to the leaf key's terminal segment (single-source
        naming — the producing node names the leaf once).
    names : Sequence[str] | None, optional
        Per-class scalar suffixes for the split (one converted leaf -> N
        named scalars). Exclusive with `name`.
    dtype : str, optional
        The ONNX output dtype (``"float32"``/``"int8"``), by default
        ``"float32"``.
    per_token : bool, optional
        Whether the output carries a dynamic per-token sequence axis, by
        default False (a global scalar).
    dyn_axis : str | None, optional
        The dynamic-axis name for a per-token output (default ``n_<stream>``
        from the leaf's stream), ignored for global outputs.

    Raises
    ------
    ConfigError
        For a non-``outputs`` / wildcard key, a name/names arity violation,
        an unsupported dtype, or names combined with per_token.
    """

    key: str
    name: str | None = None
    names: Sequence[str] | None = None
    dtype: str = "float32"
    per_token: bool = False
    dyn_axis: str | None = None

    def __post_init__(self) -> None:
        parts = self.key.split(KEY_SEP)
        if any(part in {"*", "**"} for part in parts):
            raise ConfigError(
                f"OnnxExportLeaf key {self.key!r} contains a wildcard — export output keys are "
                "concrete (design §2.2)"
            )
        if len(parts) < 2 or parts[0] != _OUTPUTS_NAMESPACE:
            raise ConfigError(
                f"OnnxExportLeaf key {self.key!r} is not under the {_OUTPUTS_NAMESPACE!r} "
                "namespace — the ONNX sink names the conversion outputs.* leaves the folded "
                "nodes mint, not raw predictions (design §6.2)"
            )
        if self.name is not None and self.names is not None:
            raise ConfigError(
                f"OnnxExportLeaf {self.key!r} sets BOTH 'name' (single output) and 'names' "
                "(per-class split_scalars) — pick one (design §6.2)"
            )
        if self.name is None and self.names is None:
            # single-source naming: default the ONNX suffix to the leaf key's
            # terminal segment (the producing node names the leaf once).
            object.__setattr__(self, "name", self.key.split(KEY_SEP)[-1])
        if self.names is not None:
            if not list(self.names) or len(set(self.names)) != len(self.names):
                raise ConfigError(
                    f"OnnxExportLeaf {self.key!r}: 'names' must be a non-empty list without "
                    f"duplicates, got {self.names!r}"
                )
            if self.per_token:
                raise ConfigError(
                    f"OnnxExportLeaf {self.key!r}: per-class split_scalars outputs ('names') are "
                    "GLOBAL float scalars — per_token applies to single-name index leaves only "
                    "(design §6.2)"
                )
        if self.dtype not in {"float32", "int8"}:
            raise ConfigError(
                f"OnnxExportLeaf {self.key!r}: dtype must be 'float32' or 'int8', got "
                f"{self.dtype!r} (the ONNX output dtypes salt export supports)"
            )

    @property
    def stream(self) -> str:
        """The leaf's stream (``outputs.<stream>.<name>`` second component)."""
        return self.key.split(KEY_SEP)[1]

    @property
    def suffixes(self) -> tuple[str, ...]:
        """The Athena suffix list (the plural names, or the single name as a 1-tuple)."""
        return tuple(self.names) if self.names is not None else (str(self.name),)

    def resolved_dyn_axis(self) -> str:
        """The dynamic-axis name for a per-token output (default ``n_<stream>``)."""
        return self.dyn_axis or f"n_{self.stream}"


class OnnxExportSink(Node):
    """The ONNX sink: a declare-only terminal node naming the conversion leaves.

    A pure terminal `SinkModule` for ``Mode.ONNX``: its ONNX-mode
    ``declare_io`` requires the export-output conversion leaves
    (``kind=data``) the folded nodes mint — ``SeqClassIndex``'s int8 leaf,
    ``Combination``'s scalar leaf, the ``ClassProbs`` probs leaf the
    ``split_scalars`` split names — and produces nothing. Because every
    conversion ran inside the traced ``executor.run``, the sink does no
    per-batch compute: it just flattens/names the populated ``outputs.*``
    into the flat Athena output tuple.

    It is the folded-path counterpart to the legacy `salt.onnx.reduces`
    path: `compile_onnx_plan` sources its ONNX sinks from
    ``declare_io(Mode.ONNX).requires`` when an export node is present, and
    the `OnnxAdapter` reads the named leaves from the executed bundle instead
    of running a post-executor ``reduce.fn`` loop for these outputs.
    union_find / MaskFormer outputs are NOT folded here and keep the legacy
    reduce path — a config may MIX folded leaves (declared here) with legacy
    reduce outputs, and the adapter dispatches per output without drift.

    Outside ``Mode.ONNX`` the node declares empty requires AND empty
    produces, so the planner prunes it from FIT/VAL/TEST — the FIT
    ``plan_hash`` is unchanged. It is a `Node`, not a `RuntimeSink`: export
    never runs a test loop, so there is no lifecycle to have. Its only job is
    naming the leaves at adapter construction, which happens at compile time.

    Parameters
    ----------
    outputs : Sequence[OnnxExportLeaf | Mapping[str, Any]]
        The export outputs, in flat Athena tuple order — globals, then
        combines, then per-token aux (the export node's list is the
        authority, not executor topo order). Each entry is an
        `OnnxExportLeaf` (or a mapping jsonargparse builds into one).
    model_name : str | None, optional
        The Athena output-name prefix (``{model_name}_{suffix}``). When None
        it is supplied at adapter construction from the resolved export
        config, by default None.

    Raises
    ------
    ConfigError
        For an empty outputs list, a duplicate leaf key, or a duplicate flat
        Athena suffix.
    """

    name = "onnx_export"
    """The graph-node instance name (overridable by the config dict key)."""

    allowed_modes: ClassVar[frozenset[Mode]] = frozenset({Mode.ONNX})
    """Export only — a manifest node has nothing to declare in any other mode."""

    def __init__(
        self,
        outputs: Sequence[OnnxExportLeaf | Mapping[str, Any]] | None = None,
        model_name: str | None = None,
    ) -> None:
        super().__init__()
        leaves = [
            leaf if isinstance(leaf, OnnxExportLeaf) else OnnxExportLeaf(**dict(leaf))
            for leaf in outputs or []
        ]
        # The export tuple comes from EITHER the explicit `outputs:` leaf list (the
        # W4 export configs / the MaskFormer escape hatch) OR a bound dumb `outputs:`
        # section. With explicit leaves the tuple is resolved up front; with a
        # section it resolves lazily on first access. One MUST resolve.
        self._leaves_resolved = bool(leaves)
        # dumb-section binding (see H5OutputSink.bind_output_section).
        self._output_section: Mapping[str, Any] | None = None
        if leaves:
            self._validate_leaves(leaves)
        self._leaves: tuple[OnnxExportLeaf, ...] = tuple(leaves)
        # the EXPLICIT leaves (the MaskFormer object reduces — leading_object +
        # object_index) survive a later `bind_output_section`: when a dumb section
        # is ALSO bound (the MaskFormer cutover names the 1:1 head leaves through
        # the section AND the object reduces explicitly), `_resolve_section_leaves`
        # merges these on top of the section's RunTaskOutput leaves. Empty for the
        # section-only and explicit-only configs.
        self._explicit_leaves: tuple[OnnxExportLeaf, ...] = tuple(leaves)
        self.model_name = model_name

    @staticmethod
    def _validate_leaves(leaves: Sequence[OnnxExportLeaf]) -> None:
        """Reject a duplicate leaf key or duplicate flat Athena suffix; raises `ConfigError`."""
        seen_keys: set[str] = set()
        seen_suffixes: set[str] = set()
        for leaf in leaves:
            if leaf.key in seen_keys:
                raise ConfigError(
                    f"OnnxExportSink: duplicate output key {leaf.key!r} — one OnnxExportLeaf per "
                    "conversion leaf (design §6.2)"
                )
            seen_keys.add(leaf.key)
            for suffix in leaf.suffixes:
                if suffix in seen_suffixes:
                    raise ConfigError(
                        f"OnnxExportSink: duplicate flat ONNX output name {suffix!r} — the Athena "
                        "output namespace is flat (design §6.3 / plan 31 W5.1 dup guard)"
                    )
                seen_suffixes.add(suffix)

    def bind_output_section(self, section: Mapping[str, Any]) -> None:
        """Capture the ``outputs:`` section so the dumb ONNX sink names its leaves.

        When bound, the sink NAMES the per-field ``outputs.*`` leaves the
        section's `RunTaskOutput` mints (in section declaration order, the
        canonical globals -> combines -> per-token tuple order, not executor
        topo order). It does no math and no ``torch.split`` / ``.squeeze`` —
        ``get_output`` already squeezed each global per-class value to a
        0-dim scalar, so the dumb sink only names the already-scalar values.
        """
        self._output_section = section
        self._leaves_resolved = False

    def _is_dumb_section(self) -> bool:
        """Whether an ``outputs:`` section is bound (the dumb-section path is active)."""
        return bool(self._output_section)

    def _section_run_task_outputs(self) -> list[Any]:
        """The bound section's `RunTaskOutput` writers, in section declaration order."""
        if not self._output_section:
            return []
        return [
            w
            for w in self._output_section.values()
            if callable(getattr(w, "is_run_task_output", None)) and w.is_run_task_output()
        ]

    def _resolve_section_leaves(self) -> tuple[OnnxExportLeaf, ...]:
        """Resolve ONNX export leaves from the bound ``outputs:`` section.

        Walks each `RunTaskOutput`'s ``manifest_fields(Mode.ONNX)`` in
        SECTION DECLARATION ORDER, keeps FINAL fields with an ``onnx_name``,
        and mints one single-``name`` `OnnxExportLeaf` per field. Ordered
        GLOBAL scalars first then PER-TOKEN aux (independent of executor
        topo order). Raises `ConfigError` when the section mints no ONNX
        leaf or two fields mint the same suffix.
        """
        globals_block: list[OnnxExportLeaf] = []
        per_token_block: list[OnnxExportLeaf] = []
        for run_task in self._section_run_task_outputs():
            for leaf_key, field in run_task.manifest_fields(Mode.ONNX):
                if field.resolved_onnx_name is None:
                    continue
                suffix = field.resolved_onnx_name
                if field.axis == "per_token":
                    per_token_block.append(
                        OnnxExportLeaf(
                            key=leaf_key, name=suffix, dtype=field.onnx_dtype, per_token=True
                        )
                    )
                else:
                    # a single already-scalar per-class value -> ONE single-name
                    # leaf (not a split): the dumb sink only names it.
                    globals_block.append(
                        OnnxExportLeaf(key=leaf_key, name=suffix, dtype=field.onnx_dtype)
                    )
        # the EXPLICIT leaves (the MaskFormer object reduces — leading_object +
        # object_index) are appended AFTER the entire section block, preserving
        # the v1 manifest/writer order. They form their own globals-then-per-token
        # sub-block appended last — NOT merged into the section's blocks (merging
        # would hoist the leading_object globals ahead of the section's per-token
        # TrackOrigin, breaking the v1 tuple order). The dup guard below rejects
        # any key/suffix clash between section and explicit leaves.
        explicit_globals = [leaf for leaf in self._explicit_leaves if not leaf.per_token]
        explicit_per_token = [leaf for leaf in self._explicit_leaves if leaf.per_token]
        ordered = (*globals_block, *per_token_block, *explicit_globals, *explicit_per_token)
        if not ordered:
            raise ConfigError(
                "OnnxExportSink (dumb-section) found no RunTaskOutput field with an ONNX leaf — "
                "wire a RunTaskOutput([tasks]) in the outputs: section (plan 34 W34.2)"
            )
        self._validate_leaves(ordered)
        self._leaves = ordered
        self._leaves_resolved = True
        return ordered

    def _ensure_leaves(self) -> tuple[OnnxExportLeaf, ...]:
        """Resolve the export leaves — explicit, or from the bound ``outputs:``
        section (the section's `RunTaskOutput` fields in declaration order);
        raises `ConfigError` when neither is configured.
        """
        if self._leaves_resolved:
            return self._leaves
        if self._is_dumb_section():
            return self._resolve_section_leaves()
        raise ConfigError(
            "OnnxExportSink has no export leaves — give it an explicit `outputs:` "
            "OnnxExportLeaf list, or compose a top-level `outputs:` section "
            "(RunTaskOutput) that binds to it (plan 34 W34.2)"
        )

    @property
    def leaves(self) -> tuple[OnnxExportLeaf, ...]:
        """The declared export leaves, in flat Athena tuple order."""
        return self._ensure_leaves()

    @property
    def outputs(self) -> tuple[str, ...]:
        """The demanded ``outputs.*`` conversion leaf keys, in declaration order."""
        return tuple(leaf.key for leaf in self._ensure_leaves())

    # -- graph node surface -------------------------------------------------

    def declare_io(self, mode: Mode) -> IO:
        """ONNX: requires every declared ``outputs.*`` conversion leaf
        (``kind=data``, shape/dtype None), produces nothing (a terminal node
        keeping the folded conversion nodes demanded). FIT/VAL/TEST: empty
        requires/produces (pruned).
        """
        if mode is not Mode.ONNX:
            return IO(requires={}, produces={})
        req = {
            leaf.key: TensorSpec(shape=None, dtype=None, kind="data")
            for leaf in self._ensure_leaves()
        }
        return IO(requires=unflatten_spec(req), produces={})

    # -- generated export metadata --------

    def resolved_model_name(self) -> str:
        """The Athena output prefix, asserting it was supplied.

        Raises
        ------
        ConfigError
            When no `model_name` was set (config or adapter construction).
        """
        if self.model_name is None:
            raise ConfigError(
                "OnnxExportSink has no model_name — set export.model_name (or the sink's "
                "model_name) before deriving the ONNX output names (design §6.3)"
            )
        return self.model_name

    def output_names(self) -> list[str]:
        """The flat ONNX output names, in declared tuple order (``{model_name}_{suffix}``).

        The single ordering authority for the folded path: the export
        node's leaf list order IS the Athena tuple order (globals, combines,
        per-token aux), independent of executor topo order — so reordering
        ``model.modules`` for memory tuning never reorders the tuple.
        """
        prefix = self.resolved_model_name()
        return [f"{prefix}_{suffix}" for leaf in self._ensure_leaves() for suffix in leaf.suffixes]

    def output_dtypes(self) -> list[str]:
        """Per-output dtypes, aligned 1:1 with `output_names`."""
        return [leaf.dtype for leaf in self._ensure_leaves() for _ in leaf.suffixes]

    def dynamic_axes(self) -> dict[str, dict[int, str]]:
        """Dynamic-axes mapping for the per-token outputs (``{name: {0: dyn_axis}}``).

        Only per-token leaves register an axis; global scalars
        (split_scalars, combines) carry none.
        """
        prefix = self.resolved_model_name()
        axes: dict[str, dict[int, str]] = {}
        for leaf in self._ensure_leaves():
            if leaf.per_token:
                axes[f"{prefix}_{leaf.suffixes[0]}"] = {0: leaf.resolved_dyn_axis()}
        return axes

    def named_outputs(self, bundle: Bundle) -> dict[str, Tensor]:
        """Flatten the executed bundle's conversion leaves into named Athena tensors.

        The declare-only sink's ONE realisation step: no conversion math
        (that ran in the trace) — only the ``split_scalars`` naming split.
        For a plural-``names`` leaf the converted prob vector is split into
        per-class scalars (``torch.split(probs, 1, -1)`` + squeeze);
        single-name leaves (the int8 index leaf, a combination scalar) pass
        through under their Athena name. The
        `OnnxAdapter` calls this to source the folded outputs from the
        bundle instead of running a ``reduce.fn`` loop.

        Raises
        ------
        ConfigError
            When a split leaf's last dim contradicts its declared names
            count.
        """
        prefix = self.resolved_model_name()
        named: dict[str, Tensor] = {}
        for leaf in self._ensure_leaves():
            value = bundle.get(leaf.key)
            if leaf.names is not None:
                # the split-count guard runs only in EAGER eval, never inside the
                # trace (it compares the leaf's last dim — a Python-bool branch the
                # tracer would constant-fold with a TracerWarning); the count is
                # validated equally by `torch.split(..., strict=True)` zip below
                if not torch.jit.is_tracing() and value.shape[-1] != len(leaf.names):
                    raise ConfigError(
                        f"OnnxExportSink: leaf {leaf.key!r} produces {value.shape[-1]} channels "
                        f"but declares {len(leaf.names)} names {list(leaf.names)} — one scalar "
                        "per class (design §6.2)"
                    )
                for suffix, part in zip(
                    leaf.names, torch.split(value, 1, -1), strict=True
                ):  # v1 task.py:301
                    named[f"{prefix}_{suffix}"] = part.squeeze()
            else:
                named[f"{prefix}_{leaf.name}"] = value
        return named

    # -- static demand (consumed by SaltModule for the static onnx plan) --------

    def writer_demand(self, model_modules: Mapping[str, Any], reader: Any) -> dict[str, str]:
        """The ONNX demand this sink anchors — GENERATED from `declare_io`.

        Mirrors `H5OutputSink.writer_demand`: returns the sink's ONNX-mode
        ``declare_io`` requires (the conversion leaves), each mapped to a
        demander description, so the static ``salt graph plot --mode onnx``
        path (which folds duck-typed ``writer_demand`` into the plan sinks)
        keeps the sink's leaves demanded and the folded conversion nodes
        alive.
        """
        del model_modules, reader
        who = "sink 'OnnxExportSink' demanding"
        return {key: f"{who} {key}" for key in flatten_spec(self.declare_io(Mode.ONNX).requires)}
