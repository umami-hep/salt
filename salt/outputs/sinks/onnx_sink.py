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
from salt.outputs.output_schema import OutputField, check_output_key, coerce

# salt.outputs already imports salt.outputs.sinks.onnx at module level
# (salt.outputs.maskformer -> salt.outputs.sinks.onnx.reduces), and nothing under
# salt.outputs.sinks.onnx imports salt.outputs at module level — so the export
# dataclasses can be named in the signature, which is what lets jsonargparse
# resolve `inputs:`/`combine:` config entries.
from salt.outputs.sinks.onnx.config import (
    ExportCombine,
    ExportConfig,
    ExportInput,
    resolve_export_config,
)
from salt.outputs.sinks.sink import Node, collect_manifest_fields


@dataclass(frozen=True)
class OnnxExportLeaf:
    """One ONNX output: the conversion ``outputs.*`` leaf + its Athena naming.

    Internal representation of one resolved output (not a config surface),
    minted only by `OnnxExportSink._leaves_from_fields`, which sets exactly one
    of `name`/`names`. Three shapes: **split_scalars** (``names`` plural,
    float32 global — one converted prob leaf split into N named scalars; the
    split is a naming concern owned here, not a conversion node); **single
    per-token leaf** (``name`` singular, int8, ``per_token=True`` — passed
    through with dynamic axis ``n_<stream>``); **single global leaf** (``name``
    singular, float32). `ConfigError` on a non-``outputs``/wildcard key.
    `nan_ok` (default False) marks an output whose NaN is a declared semantic
    (e.g. a MaskFormer leading-object/lead-vertex scalar for a jet with no
    qualifying object) — the ONNX checker compares it with ``equal_nan``
    instead of refusing NaN.
    """

    key: str
    name: str | None = None
    names: Sequence[str] | None = None
    dtype: str = "float32"
    per_token: bool = False
    nan_ok: bool = False

    def __post_init__(self) -> None:
        check_output_key(self.key, "OnnxExportLeaf key")

    @property
    def stream(self) -> str:
        """The leaf's stream (``outputs.<stream>.<name>`` second component)."""
        return self.key.split(KEY_SEP)[1]

    @property
    def suffixes(self) -> tuple[str, ...]:
        """The Athena suffix list (the plural names, or the single name as a 1-tuple)."""
        return tuple(self.names) if self.names is not None else (str(self.name),)


class OnnxExportSink(Node):
    """The ONNX sink: a declare-only terminal node naming the conversion leaves.

    A pure terminal `SinkModule` for ``Mode.ONNX``: requires the export-output
    conversion leaves the folded nodes mint and produces nothing. Every
    conversion ran inside the traced ``executor.run``, so the sink does no
    per-batch compute — it just flattens/names the populated ``outputs.*``
    into the flat Athena output tuple.

    The tuple is AUTO-COLLECTED from every bound manifest source declaring
    ``manifest_fields(Mode.ONNX)``. Flat tuple ORDER: global float scalars
    before per-token aux outputs, applied within each manifest group (section
    writers, then model graph modules), declaration order inside each block —
    pinned by the curated ``EXPECTED_OUTPUTS`` table in
    ``salt/tests/integration/pipeline/test_pipeline.py`` so a reordering is a
    visible, actionable test failure. union_find / MaskFormer outputs are NOT
    folded and keep the legacy `reduces` path; a config may MIX both and the
    adapter dispatches per output without drift.

    Outside ``Mode.ONNX`` the node declares empty IO, so FIT/VAL/TEST prune
    it (FIT ``plan_hash`` unchanged). It is a `Node`, not a `RuntimeSink` —
    export never runs a test loop. It is also the CONFIG HOME of the ONNX
    artifact contract (`model_name`, `inputs`, `track_selection`, `rename`,
    `combine`); `export_config` assembles the resolved `ExportConfig` that
    ``salt export`` / ``salt inference`` trace against.

    Parameters
    ----------
    model_name : str | None, optional
        The Athena output-name prefix (``{model_name}_{suffix}``) and
        ``doc_string``; no ``_``/``-`` allowed. None (the default) defaults it
        to the run ``name:`` with ``_``/``-`` stripped, at export time.
    inputs : Sequence[ExportInput | Mapping[str, Any]] | None, optional
        The ONNX graph inputs in positional order, each an `ExportInput` (or
        the equivalent mapping) naming the bundle port it feeds. Required to
        export; None (the default) leaves the signature undeclared.
    track_selection : str, optional
        Athena-side track selection used in the default metadata input names,
        by default ``r22default``.
    rename : Mapping[str, str] | None, optional
        Manifest suffix renames ``old -> new``, recorded in the ``gnn_config``
        metadata, by default None.
    combine : Sequence[ExportCombine | Mapping[str, Any]] | None, optional
        Combined outputs (``sum(scale * output(suffix))``), by default None.
    modes : Sequence[str] | None, optional
        Which planner modes to run in. `allowed_modes` is ``[onnx]``, so
        ``[onnx]`` (or its writer-vocabulary spelling ``[export]``) is the
        only accepted list, and omitting it (the default) means the same.
    consumes : Sequence[str] | None, optional
        fnmatch patterns over the ``outputs.*`` leaf key narrowing the
        collected tuple, by default None (every declared final ONNX leaf).
    outputs : None
        RETIRED as a config surface. Only ``None``/``[]`` is accepted; any
        truthy value raises `ConfigError`. A module minting ``outputs.*``
        leaves names them by implementing ``manifest_fields(mode)``; use
        `consumes` to narrow what this sink takes.

    Raises
    ------
    ConfigError
        For any truthy ``outputs`` value (the retired explicit-leaf surface),
        or (at resolution) an empty collected tuple, a duplicate leaf key, or
        a duplicate flat Athena suffix.
    """

    name = "onnx_export"
    """The graph-node instance name (overridable by the config dict key)."""

    allowed_modes: ClassVar[frozenset[Mode]] = frozenset({Mode.ONNX})
    """Export only — a manifest node has nothing to declare in any other mode."""

    def __init__(
        self,
        model_name: str | None = None,
        inputs: Sequence[ExportInput | Mapping[str, Any]] | None = None,
        track_selection: str = "r22default",
        rename: Mapping[str, str] | None = None,
        combine: Sequence[ExportCombine | Mapping[str, Any]] | None = None,
        modes: Sequence[str] | None = None,
        consumes: Sequence[str] | None = None,
        outputs: Sequence[OnnxExportLeaf | Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__(modes=modes, consumes=consumes)
        if outputs:
            raise ConfigError(
                "OnnxExportSink no longer accepts an explicit `outputs:` OnnxExportLeaf list — "
                "the ONNX tuple is collected from the producers' own manifests. A module that "
                "mints outputs.* leaves names them by implementing manifest_fields(mode) "
                "(RunTaskOutput does it for the section's tasks; ClassProbs / "
                "Combination / MaskFormerObjects do it for the model graph). Use `consumes:` "
                "(fnmatch patterns over the leaf key) to narrow what this sink takes."
            )
        self._leaves: tuple[OnnxExportLeaf, ...] = ()
        self._leaves_resolved = False
        self.model_name = model_name
        # the export-only half of the contract, coerced from dataclasses OR plain
        # config mappings (jsonargparse resolves the union either way)
        self.inputs: list[ExportInput] = [coerce(ExportInput, e) for e in inputs or ()]
        self.track_selection = track_selection
        self.rename: dict[str, str] = dict(rename or {})
        self.combine: list[ExportCombine] = [coerce(ExportCombine, c) for c in combine or ()]

    @staticmethod
    def _validate_leaves(leaves: Sequence[OnnxExportLeaf]) -> None:
        """Reject a duplicate leaf key or duplicate flat Athena suffix; raises `ConfigError`."""
        seen_keys: set[str] = set()
        seen_suffixes: set[str] = set()
        for leaf in leaves:
            if leaf.key in seen_keys:
                raise ConfigError(
                    f"OnnxExportSink: duplicate output key {leaf.key!r} — one OnnxExportLeaf "
                    "per conversion leaf"
                )
            seen_keys.add(leaf.key)
            for suffix in leaf.suffixes:
                if suffix in seen_suffixes:
                    raise ConfigError(
                        f"OnnxExportSink: duplicate flat ONNX output name {suffix!r} — the Athena "
                        "output namespace is flat"
                    )
                seen_suffixes.add(suffix)

    def _invalidate_manifest(self) -> None:
        """Drop the cached leaf tuple after a manifest source rebinds."""
        self._leaves_resolved = False

    def _leaves_from_fields(
        self, fields: Sequence[tuple[str, OutputField]]
    ) -> list[OnnxExportLeaf]:
        """Mint one leaf per ``outputs.*`` key, grouping the key's fields.

        Keys in first-appearance order: one field -> a single-``name`` leaf
        (per-token when the field is), N fields -> a ``names`` split leaf.
        A split leaf is a set of GLOBAL float scalars, so a per-token field
        mixed into one is a `ConfigError`.
        """
        by_key: dict[str, list[OutputField]] = {}
        for leaf_key, field in fields:
            if field.resolved_onnx_name is not None:
                by_key.setdefault(leaf_key, []).append(field)
        leaves: list[OnnxExportLeaf] = []
        for leaf_key, group in by_key.items():
            if len(group) == 1:
                field = group[0]
                leaves.append(
                    OnnxExportLeaf(
                        key=leaf_key,
                        name=field.resolved_onnx_name,
                        dtype=field.onnx_dtype,
                        per_token=field.axis == "per_token",
                        nan_ok=field.nan_ok,
                    )
                )
                continue
            if any(f.axis == "per_token" for f in group):
                raise ConfigError(
                    f"OnnxExportSink: leaf {leaf_key!r} declares several ONNX fields including a "
                    "per-token one — a multi-name split leaf is a set of GLOBAL float scalars, so "
                    "a per-token field must be the leaf's only field"
                )
            leaves.append(
                OnnxExportLeaf(
                    key=leaf_key,
                    names=[str(f.resolved_onnx_name) for f in group],
                    dtype=group[0].onnx_dtype,
                    nan_ok=all(f.nan_ok for f in group),
                )
            )
        return leaves

    def _resolve_leaves(self) -> tuple[OnnxExportLeaf, ...]:
        """Resolve the export leaves from the bound manifest sources.

        The Athena tuple order is GLOBAL float scalars before PER-TOKEN aux
        outputs, applied WITHIN each manifest-source group (the ``outputs:``
        section, then the model's graph modules) and the groups concatenated
        in that order. Ordering per group rather than once over everything is
        what keeps a model-graph producer's globals (e.g. the MaskFormer
        ``leading_objects_*`` reduces) behind the section's own per-token
        leaves (e.g. ``TrackOrigin``) instead of hoisting them to the front.
        Within a block, declaration order.

        ``consumes:`` narrows the collected manifest first, validated against
        every declared leaf so a pattern matching nothing anywhere is loud.
        Raises `ConfigError` when nothing is declared, when a split leaf mixes
        in a per-token field, or when two fields mint the same key/suffix.
        """
        per_group = [
            collect_manifest_fields(group, Mode.ONNX) for group in self._manifest_source_groups()
        ]
        # validate + narrow ONCE over the flat manifest: a `consumes:` pattern is
        # checked against every declared leaf, not just one group's. The filter is
        # a pure function of the leaf key, so replaying it per group by key
        # membership is the same narrowing.
        kept = {key for key, _ in self._filter_consumed([f for g in per_group for f in g])}
        leaves: list[OnnxExportLeaf] = []
        for group_fields in per_group:
            group_leaves = self._leaves_from_fields([f for f in group_fields if f[0] in kept])
            leaves.extend(leaf for leaf in group_leaves if not leaf.per_token)
            leaves.extend(leaf for leaf in group_leaves if leaf.per_token)
        if not leaves:
            raise ConfigError(
                "OnnxExportSink collected no ONNX output — no bound producer declared a final "
                "Mode.ONNX field. Sources searched: "
                f"{self._manifest_source_names()}. Wire a RunTaskOutput whose `modes:` include "
                "'export' in the outputs: section, or a conversion producer "
                "(ClassProbs / Combination / MaskFormerObjects) in model.modules."
            )
        self._validate_leaves(leaves)
        self._leaves = tuple(leaves)
        self._leaves_resolved = True
        return self._leaves

    def _ensure_leaves(self) -> tuple[OnnxExportLeaf, ...]:
        """The resolved export leaves (cached until a manifest source rebinds)."""
        if self._leaves_resolved:
            return self._leaves
        return self._resolve_leaves()

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

    # -- the export contract ------------------------------------------------

    def export_config(self, run_name: str = "salt") -> ExportConfig:
        """The RESOLVED export contract this sink declares.

        Assembles the sink's own fields into an `ExportConfig` and hands it to
        `salt.outputs.sinks.onnx.resolve_export_config`, which fills the defaults (input
        names, dynamic axes, Athena metadata names, the `run_name`-derived
        model name) and validates them.

        A `ConfigError` propagates from that resolution when the contract is
        incomplete or malformed (no inputs, an invalid model name / track
        selection, a bad rename or combine).

        Parameters
        ----------
        run_name : str, optional
            The run ``name:``, the default `model_name` source, by default
            ``"salt"``.

        Returns
        -------
        ExportConfig
            The resolved export-only half.
        """
        return resolve_export_config(
            ExportConfig(
                model_name=self.model_name,
                track_selection=self.track_selection,
                inputs=list(self.inputs),
                rename=dict(self.rename),
                combine=list(self.combine),
            ),
            run_name,
        )

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
                "OnnxExportSink has no model_name — set the sink's `model_name:` (or let "
                "`salt export` default it from the run name) before deriving the ONNX "
                "output names"
            )
        return self.model_name

    def output_names(self) -> list[str]:
        """The flat ONNX output names, in resolved tuple order (``{model_name}_{suffix}``).

        The order is the manifest-source order (section writers, then model
        modules, each in declaration order), independent of executor topo
        order. It is not a contract — Athena consumes the outputs by name.
        """
        prefix = self.resolved_model_name()
        return [f"{prefix}_{suffix}" for leaf in self._ensure_leaves() for suffix in leaf.suffixes]

    def output_dtypes(self) -> list[str]:
        """Per-output dtypes, aligned 1:1 with `output_names`."""
        return [leaf.dtype for leaf in self._ensure_leaves() for _ in leaf.suffixes]

    def nan_ok_outputs(self) -> frozenset[str]:
        """The flat Athena output names whose NaN is a declared semantic.

        Returns
        -------
        frozenset[str]
            ``{model_name}_{suffix}`` for every suffix of a leaf whose
            (group-)``nan_ok`` is True — the ONNX checker compares these
            with ``equal_nan`` instead of refusing NaN.
        """
        prefix = self.resolved_model_name()
        return frozenset(
            f"{prefix}_{suffix}"
            for leaf in self._ensure_leaves()
            if leaf.nan_ok
            for suffix in leaf.suffixes
        )

    def dynamic_axes(self) -> dict[str, dict[int, str]]:
        """Dynamic-axes mapping for the per-token outputs (``{name: {0: "n_<stream>"}}``).

        Only per-token leaves register an axis; global scalars
        (split_scalars, combines) carry none.
        """
        prefix = self.resolved_model_name()
        axes: dict[str, dict[int, str]] = {}
        for leaf in self._ensure_leaves():
            if leaf.per_token:
                axes[f"{prefix}_{leaf.suffixes[0]}"] = {0: f"n_{leaf.stream}"}
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
        bundle instead of running a ``reduce.fn`` loop. A split leaf whose
        last dim contradicts its names count fails the strict ``zip`` (a
        ValueError in eager eval).
        """
        prefix = self.resolved_model_name()
        named: dict[str, Tensor] = {}
        for leaf in self._ensure_leaves():
            value = bundle.get(leaf.key)
            if leaf.names is not None:
                # the strict zip is the split-count check (a ValueError in eager eval)
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
