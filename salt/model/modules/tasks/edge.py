"""Edge-classification vertexing task module."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import IO, Mode, TensorSpec
from salt.model.bind import ResolvedSchema
from salt.model.modules.tasks.base import _loss_class, _TaskModuleBase
from salt.model.nn.dense import Dense
from salt.outputs.output_schema import VERTEX_INDEX, OutputField
from salt.utils.union_find import get_node_assignment_jit, mask_fill_flattened

_DEFAULT_VTX_LOSS: dict[str, Any] = {
    "class_path": "torch.nn.BCEWithLogitsLoss",
    "init_args": {"reduction": "none"},
}


class VertexingTaskModule(_TaskModuleBase):
    """Edge-classification vertexing head.

    The origin-label dependency is declared (``origin_label:`` config ->
    ``labels.<stream>.<origin_label>``). ``origin_weighting`` gives the
    heavy/fake origins as integer ids OR class names (defaults reproduce the
    hardcoded 3,4,5 / 1). Names are resolved to ids at fit/test setup against
    the dataset schema's origin class-name attr (`resolve_origin_names`,
    called before `bind`); a name-based config without a schema artifact is a
    `ConfigError` at bind. Mixing names and ids in one heavy/fake list is
    rejected.

    TEST publishes per-node vertex assignments (union-find); ONNX publishes
    RAW edge scores that `get_output` folds into the union-find vertex index.
    """

    sequence = True  # always per-token (no `sequence` config parameter)

    def __init__(
        self,
        stream: str,
        label: str,
        origin_label: str,
        input: str | None = None,  # noqa: A002 - matches the YAML config key name
        context: str | None = None,
        dense: dict[str, Any] | None = None,
        loss: str | dict[str, Any] | None = None,
        weight: float = 1.0,
        origin_weighting: Mapping[str, Sequence[int | str]] | None = None,
        prefix_vertex_column: bool = False,
        expose: Sequence[str] | None = None,
        write_targets: bool = True,
    ) -> None:
        """Capture config only.

        Parameters
        ----------
        label : str
            Vertex-index label; must contain ``"VertexIndex"`` (the vertexing
            loss derives the origin key by string-replace).
        origin_label : str
            Declared origin-label dependency for edge weighting (and, for
            name-based weighting, the schema attr the names resolve against).
        loss : str | dict[str, Any] | None, optional
            Loss config, by default ``BCEWithLogitsLoss(reduction="none")``.
            Reduction MUST be ``none`` — per-edge weighting multiplies the
            unreduced loss.
        origin_weighting : Mapping[str, Sequence[int | str]] | None, optional
            ``{"heavy": [...], "fake": [...]}`` origin ids OR class names, by
            default ``{"heavy": [3, 4, 5], "fake": [1]}``. Names are resolved
            at setup against the schema's origin class-name attr; ids bind
            directly.
        prefix_vertex_column : bool, optional
            Name the TEST vertexing column ``{run_name}_VertexIndex`` instead
            of the bare ``VertexIndex``, by default False. The ONNX name is
            always the shared `VERTEX_INDEX` constant.
        expose : Sequence[str] | None, optional
            Modes the ``preds.*`` port is published in, by default all modes.
            ``[fit, val]`` opts a train-only aux task out of TEST/ONNX.
        write_targets : bool, optional
            Also emit the per-token vertex-index label (padded positions
            ``-1``) as the TEST eval column ``target_{task}``, by default
            True. Never emitted in ONNX/export mode.

        Raises
        ------
        ConfigError
            If `label` lacks ``"VertexIndex"``, `origin_weighting` has unknown
            keys, a heavy/fake list mixes integer ids with class names, or
            `expose` is a bad mode list.
        """
        super().__init__(
            stream,
            label,
            input,
            context,
            dense,
            loss,
            weight,
            _DEFAULT_VTX_LOSS,
            expose,
            write_targets,
        )
        if "VertexIndex" not in label:
            raise ConfigError(
                f"VertexingTaskModule: label {label!r} must contain 'VertexIndex' — the "
                "vertexing loss derives the origin key as "
                "label.replace('VertexIndex', 'OriginLabel')"
            )
        self.origin_label = origin_label
        self.origin_label_key = f"labels.{stream}.{origin_label}"
        self.prefix_vertex_column = bool(prefix_vertex_column)
        weighting = dict(origin_weighting or {"heavy": [3, 4, 5], "fake": [1]})
        if unknown := sorted(set(weighting) - {"heavy", "fake"}):
            raise ConfigError(
                f"VertexingTaskModule: unknown origin_weighting keys {unknown} — expected "
                "'heavy' and 'fake'"
            )
        heavy = tuple(weighting.get("heavy", (3, 4, 5)))
        fake = tuple(weighting.get("fake", (1,)))
        # ids bind standalone; names defer to resolve_origin_names(reader) at setup,
        # so heavy_ids/fake_ids stay None until resolved (a name-based bind without
        # a schema then fails loudly instead of silently mis-weighting)
        self._heavy_cfg, self._fake_cfg = heavy, fake
        ids = (_origin_ids("heavy", heavy), _origin_ids("fake", fake))
        self._names_pending = None in ids
        self.heavy_ids, self.fake_ids = (None, None) if self._names_pending else ids

    @property
    def has_pad_mask(self) -> bool:
        """Always: the edge adjacency is built from the stream pad mask."""
        return True

    def resolve_origin_names(self, reader: Any) -> bool:
        """Resolve name-based ``origin_weighting`` to ids against the schema.

        No-op for integer-id weighting. For name-based weighting, the origin
        label's class-name attr (``schema_group(stream).attrs[origin_label]``)
        maps each name to its index. Must be called (via
        `salt.model.validation.resolve_origin_weighting`) before `bind`.

        Parameters
        ----------
        reader : Any
            The stage dataset reader; consulted via ``schema_group(stream)``
            (duck-typed — a reader without schema support resolves nothing,
            leaving a name-based config to fail loudly at bind).

        Returns
        -------
        bool
            True when names were resolved here, False when there was nothing
            to resolve (integer ids) or the reader has no schema artifact.

        Raises
        ------
        ConfigError
            When the stream/origin-label class-name attr is absent or a
            configured name is not among the schema's origin classes.
        """
        if not self._names_pending:
            return False
        schema_group = getattr(reader, "schema_group", None)
        if not callable(schema_group):
            return False
        gschema = schema_group(self.stream)
        attr = gschema.attrs.get(self.origin_label) if gschema is not None else None
        if not (
            isinstance(attr, (list, tuple)) and attr and all(isinstance(item, str) for item in attr)
        ):
            raise ConfigError(
                f"VertexingTaskModule {self.name!r}: name-based origin_weighting needs the "
                f"origin label's class names, but the schema artifact has no string-list "
                f"attr {self.origin_label!r} on the {self.stream!r} group (config: "
                f"model.modules.{self.name}.init_args.origin_weighting) — "
                "dump the schema with the origin class names, or use integer origin ids"
            )
        index = {name: i for i, name in enumerate(attr)}
        self.heavy_ids = self._resolve_names("heavy", self._heavy_cfg, index, attr)
        self.fake_ids = self._resolve_names("fake", self._fake_cfg, index, attr)
        self._names_pending = False
        return True

    def _resolve_names(
        self, role: str, names: Sequence[Any], index: Mapping[str, int], classes: Sequence[str]
    ) -> tuple[int, ...]:
        """Map one heavy/fake class-name list to integer origin ids, in config
        order; raises `ConfigError` on a name absent from the schema's origin
        classes.
        """
        if missing := [name for name in names if name not in index]:
            raise ConfigError(
                f"VertexingTaskModule {self.name!r}: origin_weighting {role!r} class "
                f"{missing[0]!r} is not among the {self.origin_label!r} classes {list(classes)} "
                f"(config: model.modules.{self.name}.init_args.origin_weighting)"
            )
        return tuple(index[name] for name in names)

    def declare_io(self, mode: Mode) -> IO:
        """Requires both vertex and origin labels (FIT|VAL); the shape-unconstrained
        pred spec covers data-dependent ``[E, 1]`` edge scores and per-node TEST
        assignments.
        """
        del mode
        label = TensorSpec(shape=self._tokens, dtype="int64", kind="label", modes=Mode.TRAINING)
        return self._head_io(
            label_specs={self.label_key: label, self.origin_label_key: label},
            pred_shape=None,
            context_last=True,
        )

    def bind(self, schema: ResolvedSchema) -> None:
        """``input_size = 2 * width(input)`` (edge-pair concat); requires the
        loss's ``reduction="none"`` and origin ids already resolved
        (`resolve_origin_names`).
        """
        if self._names_pending or self.heavy_ids is None or self.fake_ids is None:
            raise ConfigError(
                f"VertexingTaskModule {self.name!r}: name-based origin_weighting "
                f"(heavy={list(self._heavy_cfg)}, fake={list(self._fake_cfg)}) was not resolved "
                "to integer ids before bind — a dataset schema artifact carrying the "
                f"{self.origin_label!r} class names is required. "
                "resolve_origin_names(reader) runs at fit/test setup; standalone bind needs "
                "integer origin ids instead"
            )
        init_args = dict(self.loss_cfg.get("init_args", {}))
        loss_module = _loss_class(self.loss_cfg)(**init_args)
        if getattr(loss_module, "reduction", "none") != "none":
            raise ConfigError(
                f"VertexingTaskModule {self.name!r}: loss reduction must be 'none' — the "
                "origin weighting multiplies the per-edge loss (task.py:938-947)"
            )
        self.loss = loss_module
        width = schema.width(self.input_key)
        self.net = Dense(
            input_size=2 * width,
            output_size=1,
            **({"context_size": schema.width(self.context)} if self.context else {}),
            **self.dense_cfg,
        )

    @torch.compiler.disable
    def head_forward(
        self,
        x: Tensor,
        mask: Tensor | None,
        labels: Tensor | None,
        origin_labels: Tensor | None,
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute pair classification for vertexing and its loss.

        ``torch.compiler.disable`` cuts one deliberate seam here (a no-op unless
        compiled), the same trade as the flash-varlen unpad/repad. This head is
        structurally uncapturable: it compresses a ``[B, N, N]`` adjacency into
        one row per valid edge, so both the allocation size (``adjmat.sum()``,
        a ``.item()``) and every ``[adjmat]`` compression are data-dependent.
        The compressions lower to ``aten.nonzero``, which inductor will not
        lower on CUDA — capturing them is not an option on the GPUs this runs
        on. Left un-seamed the head cost eight scattered breaks, which fragment
        the graph AROUND it as well; one break is strictly better.

        The loss is the per-edge pairwise-matching BCE, weighted by
        `get_weights` and normalised by the number of matched pairs, times
        ``weight``.

        Returns
        -------
        tuple[Tensor, Tensor | None]
            Predicted edge logits ``[E, 1]`` and the scalar loss (``None`` without labels).
        """
        b, n, d = x.shape
        ex_size = (b, n, n, d)
        t_mask = torch.ones(b, n, device=x.device) if mask is None else ~mask
        t_mask = torch.cat(
            [t_mask, torch.zeros(b, 1, device=x.device)], dim=1
        )  # pad t_mask for onnx compatibility
        adjmat = t_mask.unsqueeze(-1) * t_mask.unsqueeze(-2)
        adjmat = (
            adjmat.bool() & ~torch.eye(n + 1, n + 1, device=adjmat.device).repeat(b, 1, 1).bool()
        )

        context_matrix = None
        if context is not None:
            context_d = context.shape[-1]
            context = context.unsqueeze(1).expand(b, n, context_d)
            context_matrix = torch.zeros(
                (adjmat.sum(), 2 * context_d), device=x.device, dtype=x.dtype
            )
            context_matrix = context.unsqueeze(-2).expand((b, n, n, context_d))[adjmat[:, :-1, :-1]]

        # compressed track-track matrix (one row per valid edge, not [B, N, N])
        tt_matrix = torch.zeros((adjmat.sum(), d * 2), device=x.device, dtype=x.dtype)
        tt_matrix[:, :d] = x.unsqueeze(-2).expand(ex_size)[adjmat[:, :-1, :-1]]
        tt_matrix[:, d:] = x.unsqueeze(-3).expand(ex_size)[adjmat[:, :-1, :-1]]
        pred = self.net(tt_matrix, context_matrix)
        if labels is None or origin_labels is None:
            return pred, None
        adjmat = adjmat[:, :-1, :-1]
        match_matrix = labels.unsqueeze(-1) == labels.unsqueeze(-2)
        # negative-class labels never count as a match, even to each other
        unique_matrix = labels < 0
        match_matrix *= ~(unique_matrix.unsqueeze(-1) | unique_matrix.unsqueeze(-2))
        match_matrix = match_matrix[adjmat].float()
        loss = self.loss(pred.squeeze(-1), match_matrix) * self.get_weights(origin_labels, adjmat)
        return pred, loss.sum() / match_matrix.sum() * self.weight

    def get_weights(self, labels: Tensor, adjmat: Tensor) -> Tensor:
        """Per-edge weights ``[E]`` (after adjacency compression) from the configured
        heavy/fake origin ids; the default ids (3, 4, 5 / 1) reproduce the historic
        hardcoded weighting bit-for-bit.
        """
        heavy = torch.clip(sum(labels == i for i in self.heavy_ids), 0, 1)
        fake = torch.clip(sum(labels == i for i in self.fake_ids), 0, 1)
        weights = heavy - fake.int()
        return 1 + (weights.unsqueeze(-1) & weights.unsqueeze(-2))[adjmat]

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Publishes RAW ``[E, 1]`` edge scores in every non-training mode; the
        union-find conversion happens exactly once downstream in `get_output`.
        """
        assert self.net is not None, "forward before bind()"
        x = b.get(self.input_key)
        ctx = b.get(self.context) if self.context is not None else None
        mask = b.get(f"masks.{self.stream}")
        if not (mode & Mode.TRAINING):
            preds, _ = self.head_forward(x, mask, None, None, context=ctx)
            return {self.pred_key: preds}
        labels, origins = b.get(self.label_key), b.get(self.origin_label_key)
        preds, loss = self.head_forward(x, mask, labels, origins, context=ctx)
        return {self.pred_key: preds, self.loss_key: loss}

    # -- output rendering ---------------------------------------------------

    def output_time_requires(self, mode: Mode) -> list[str]:
        """``["masks.<stream>"]`` — a vertexing head always requires a pad mask —
        plus the vertex-index label key in TEST when ``write_targets`` (never in ONNX).
        """
        return self._output_deps(mode, [self.label_key])

    def _output_values(self, b: Bundle, mode: Mode) -> list[Tensor]:
        """Union-find vertex assignments: ONNX -> flattened int8 ``[L]``; H5 -> int-cast
        (``-inf`` padding -> int32 min) plus, in TEST, the raw vertex-index label the
        pairwise matching loss targets (negative = no vertex; padded positions ``-1``).
        """
        assert self.net is not None, "get_output before bind()"
        mask = b.get(f"masks.{self.stream}")
        vertices = mask_fill_flattened(get_node_assignment_jit(b.get(self.pred_key), mask), mask)
        if mode & Mode.ONNX:
            return [vertices.reshape(-1).char()]
        values = [vertices.int()]
        if self._emit_targets(mode):
            values.append(torch.masked_fill(b.get(self.label_key), mask, -1))
        return values

    def get_output_manifest(self, mode: Mode, run_name: str) -> list[OutputField]:
        """`get_output` derives its fields from this list via ``replace(field, value=...)``."""
        del run_name
        if mode & Mode.ONNX:
            return [OutputField(None, VERTEX_INDEX, "int8", "per_token", final=True)]
        fields = [
            OutputField(
                VERTEX_INDEX, None, "i8", "per_token", final=True, prefix=self.prefix_vertex_column
            )
        ]
        if self._emit_targets(mode):
            fields += self._target_fields("i4")
        return fields


def _origin_ids(role: str, entries: Sequence[Any]) -> tuple[int, ...] | None:
    """One ``origin_weighting`` role list as integer origin ids in config order, or None
    for a class-name list (resolved at setup). Raises `ConfigError` on a list mixing ids
    and names, or on a non-int id (bool included) — at construction, never half-resolved.
    """
    is_name = [isinstance(e, str) for e in entries]
    if any(is_name):
        if not all(is_name):
            raise ConfigError(
                f"VertexingTaskModule: origin_weighting {role!r} list mixes integer ids with "
                f"class names ({list(entries)!r}) — use one or the other"
            )
        return None
    if bad := [e for e in entries if isinstance(e, bool) or not isinstance(e, int)]:
        raise ConfigError(
            f"VertexingTaskModule: origin_weighting {role!r} entries must be INTEGER origin "
            f"ids or class NAMES (got {bad[0]!r} in {list(entries)!r}); v1's hardcoded ids are "
            "heavy: [3, 4, 5], fake: [1]"
        )
    return tuple(int(e) for e in entries)
