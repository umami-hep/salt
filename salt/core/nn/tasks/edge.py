"""Edge-classification vertexing task module."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from numpy.lib.recfunctions import unstructured_to_structured as u2s
from torch import Tensor

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.core.nn.bind import ResolvedSchema
from salt.core.nn.dense import Dense
from salt.core.nn.stream_embed import _stream_len
from salt.core.nn.tasks.base import _loss_class, _TaskModuleBase
from salt.core.onnx.config import ExportOutput
from salt.core.onnx.reduces import mask_fill_flattened
from salt.core.outputs.names import VERTEX_INDEX
from salt.core.outputs.output_field import OutputField
from salt.core.utils.union_find import get_node_assignment_jit

_DEFAULT_VTX_LOSS: dict[str, Any] = {
    "class_path": "torch.nn.BCEWithLogitsLoss",
    "init_args": {"reduction": "none"},
}


# convert flattened array to shape of mask (ntracks, ...) -> (njets, maxtracks, ...)
@torch.jit.script
def _mask_fill_flattened(flat_array: Tensor, mask: Tensor) -> Tensor:
    """Unflatten a per-node array back to a batch-shaped tensor; padded positions read -inf.

    Returns
    -------
    Tensor
        Filled tensor of shape ``[B, L, F]``.
    """
    filled = torch.full((mask.shape[0], mask.shape[1], flat_array.shape[1]), float("-inf"))
    mask = mask.to(torch.bool)
    start_index = end_index = 0

    for i in range(mask.shape[0]):
        if mask[i].shape[0] > 0:
            end_index += (~mask[i]).to(torch.long).sum()
            filled[i, : end_index - start_index] = flat_array[start_index:end_index]
            start_index = end_index

    return filled


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
    RAW edge scores for the export-side ``vertex_union_find`` reduce.
    """

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

        Raises
        ------
        ConfigError
            If `label` lacks ``"VertexIndex"``, `origin_weighting` has unknown
            keys, a heavy/fake list mixes integer ids with class names, or
            `expose` is a bad mode list.
        """
        super().__init__(
            stream, label, input, context, dense, loss, weight, _DEFAULT_VTX_LOSS, expose
        )
        if "VertexIndex" not in label:
            raise ConfigError(
                f"VertexingTaskModule: label {label!r} must contain 'VertexIndex' — the "
                "vertexing loss derives the origin key as "
                "label.replace('VertexIndex', 'OriginLabel')"
            )
        self.origin_label = origin_label
        self.prefix_vertex_column = bool(prefix_vertex_column)
        weighting = dict(origin_weighting or {"heavy": [3, 4, 5], "fake": [1]})
        if unknown := sorted(set(weighting) - {"heavy", "fake"}):
            raise ConfigError(
                f"VertexingTaskModule: unknown origin_weighting keys {unknown} — expected "
                "'heavy' and 'fake' (design §3.3)"
            )
        heavy = tuple(weighting.get("heavy", (3, 4, 5)))
        fake = tuple(weighting.get("fake", (1,)))
        # ids bind standalone; names defer to resolve_origin_names(reader) at setup,
        # so heavy_ids/fake_ids stay None until resolved (a name-based bind without
        # a schema then fails loudly instead of silently mis-weighting)
        self._heavy_cfg, self._fake_cfg = heavy, fake
        self._names_pending = _is_name_weighting(heavy, fake)
        if self._names_pending:
            self.heavy_ids: tuple[int, ...] | None = None
            self.fake_ids: tuple[int, ...] | None = None
        else:
            self.heavy_ids = _coerce_origin_ids(heavy, "heavy")
            self.fake_ids = _coerce_origin_ids(fake, "fake")

    @property
    def origin_label_key(self) -> str:
        """The declared origin-label dependency.

        Returns
        -------
        str
            ``labels.<stream>.<origin_label>``.
        """
        return f"labels.{self.stream}.{self.origin_label}"

    def resolve_origin_names(self, reader: Any) -> bool:
        """Resolve name-based ``origin_weighting`` to ids against the schema.

        No-op for integer-id weighting. For name-based weighting, the origin
        label's class-name attr (``schema_group(stream).attrs[origin_label]``)
        maps each name to its index. Must be called (via
        `salt.core.saltmodule.resolve_origin_weighting`) before `bind`.

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
                f"model.modules.{self.name}.init_args.origin_weighting; design §5.1, §2.6) — "
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
        """Map one heavy/fake class-name list to integer origin ids.

        Returns
        -------
        tuple[int, ...]
            The resolved integer ids, in config order.

        Raises
        ------
        ConfigError
            On any name absent from the schema's origin classes.
        """
        ids: list[int] = []
        for name in names:
            if name not in index:
                raise ConfigError(
                    f"VertexingTaskModule {self.name!r}: origin_weighting {role!r} class "
                    f"{name!r} is not among the {self.origin_label!r} classes {list(classes)} "
                    f"(config: model.modules.{self.name}.init_args.origin_weighting; design §5.1)"
                )
            ids.append(index[name])
        return tuple(ids)

    def declare_io(self, mode: Mode) -> IO:
        """Declare input/context/mask (+ FIT|VAL vertex AND origin labels) -> preds/loss.

        The prediction spec is shape-unconstrained: the edge count is
        data-dependent in FIT|VAL|ONNX (``[E, 1]`` raw scores) and the TEST
        assignments are per-node.

        Returns
        -------
        IO
            The declared requires/produces.
        """
        del mode
        width = sym_dim("D", self.name)
        label_spec = TensorSpec(
            shape=("B", _stream_len(self.stream)),
            dtype="int64",
            kind="label",
            modes=Mode.TRAINING,
        )
        requires: dict[str, TensorSpec] = {
            self.input_key: TensorSpec(
                shape=("B", _stream_len(self.stream), width), dtype="float32"
            ),
            f"masks.{self.stream}": TensorSpec(
                shape=("B", _stream_len(self.stream)), dtype="bool", kind="pad_mask"
            ),
            self.label_key: label_spec,
            self.origin_label_key: label_spec,
        }
        if self.context is not None:
            requires[self.context] = TensorSpec(shape=None, dtype="float32")
        produces: dict[str, TensorSpec] = {
            self.pred_key: self._pred_spec(TensorSpec(shape=None, dtype=None)),
            self.loss_key: TensorSpec(shape=(), kind="loss", modes=Mode.TRAINING),
        }
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def bind(self, schema: ResolvedSchema) -> None:
        """Build the head layers with inferred widths.

        ``input_size = 2 * width(input)`` (pair concat); ``context_size = width(context)``.

        Raises
        ------
        ConfigError
            If the configured loss does not use ``reduction="none"`` (per-edge
            weighting requires the unreduced loss), or name-based
            ``origin_weighting`` was never resolved before bind.
        """
        if self._names_pending or self.heavy_ids is None or self.fake_ids is None:
            raise ConfigError(
                f"VertexingTaskModule {self.name!r}: name-based origin_weighting "
                f"(heavy={list(self._heavy_cfg)}, fake={list(self._fake_cfg)}) was not resolved "
                "to integer ids before bind — a dataset schema artifact carrying the "
                f"{self.origin_label!r} class names is required (design §5.1, §2.6). "
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

    def head_forward(
        self,
        x: Tensor,
        labels_dict: Mapping | None,
        pad_masks: Mapping | None = None,
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute pair classification for vertexing and its loss.

        Returns
        -------
        tuple[Tensor, Tensor | None]
            Predicted edge logits ``[E, 1]`` and the scalar loss.
        """
        if pad_masks is not None:
            input_name_mask = self.input_name_mask(pad_masks)
            mask = pad_masks[self.input_name]
            x = x[:, input_name_mask]
        else:
            mask = None
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
        loss: Tensor | None = None
        if labels_dict:
            loss = self.calculate_loss(pred, labels_dict, adjmat=adjmat[:, :-1, :-1])

        return pred, loss

    def calculate_loss(self, pred: Tensor, labels_dict: Mapping, adjmat: Tensor) -> Tensor:
        """Compute the vertexing loss against pairwise matching labels.

        Returns
        -------
        Tensor
            Weighted average loss scaled by ``self.weight``.
        """
        labels = labels_dict[self.input_name][self.label]

        match_matrix = labels.unsqueeze(-1) == labels.unsqueeze(-2)

        # negative-class labels never count as a match, even to each other
        unique_matrix = labels < 0
        unique_matrix = unique_matrix.unsqueeze(-1) | unique_matrix.unsqueeze(-2)
        match_matrix *= ~unique_matrix

        match_matrix = match_matrix[adjmat].float()

        loss = self.loss(pred.squeeze(-1), match_matrix)

        origin_label = self.label.replace("VertexIndex", "OriginLabel")
        weights = self.get_weights(labels_dict[self.input_name][origin_label], adjmat)
        weighted_loss = loss * weights

        num_non_masked_elements = match_matrix.sum()
        loss = weighted_loss.sum() / num_non_masked_elements

        return loss * self.weight

    def get_weights(self, labels: Tensor, adjmat: Tensor) -> Tensor:
        """Compute per-edge weights from the configured heavy/fake origin ids.

        With the default ids (3, 4, 5 / 1) this reproduces the historic
        hardcoded weighting bit-for-bit.

        Returns
        -------
        Tensor
            Per-edge weights of shape ``[E]`` after adjacency compression.
        """
        heavy = torch.clip(sum(labels == i for i in self.heavy_ids), 0, 1)
        fake = torch.clip(sum(labels == i for i in self.fake_ids), 0, 1)
        weights = heavy - fake.int()
        weights = weights.unsqueeze(-1) & weights.unsqueeze(-2)
        weights = weights[adjmat]
        return 1 + weights

    def run_inference(self, preds: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        """Per-node assignments from edge predictions.

        Returns
        -------
        Tensor
            Flattened per-node assignments with paddings filled to ``-inf``.
        """
        preds = get_node_assignment_jit(preds, pad_mask)
        return _mask_fill_flattened(preds, pad_mask)

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Run the head; publishes RAW ``[E, 1]`` edge scores in EVERY non-training mode.

        The labels dict carries the origin labels under the derived key
        (``label.replace("VertexIndex", "OriginLabel")``) so ``calculate_loss``
        finds them — the graph dependency is the declared ``origin_label`` port.

        The union-find conversion (edge scores -> per-node assignments) is
        owned by ``get_output`` on the live path and by ``get_h5`` on the
        oracle path, both of which read this raw leaf and convert exactly once.

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only.
        """
        assert self.net is not None, "forward before bind()"
        x = b.get(self.input_key)
        ctx = b.get(self.context) if self.context is not None else None
        mask = b.get(f"masks.{self.stream}")
        pad_masks = {self.stream: mask}
        if mode & Mode.TRAINING:
            derived = self.label.replace("VertexIndex", "OriginLabel")
            labels_dict = {
                self.stream: {
                    self.label: b.get(self.label_key),
                    derived: b.get(self.origin_label_key),
                }
            }
            preds, loss = self.head_forward(x, labels_dict, pad_masks, context=ctx)
            return {self.pred_key: preds, self.loss_key: loss}
        preds, _ = self.head_forward(x, None, pad_masks, context=ctx)
        return {self.pred_key: preds}

    # -- output rendering ---------------------------------------------------

    def output_names(self, run_name: str) -> list[tuple[str, str]]:
        """A single ``('VertexIndex', 'i8')`` column.

        Bare ``VertexIndex`` by default; with ``prefix_vertex_column`` it is
        ``{run_name}_VertexIndex``.

        Returns
        -------
        list[tuple[str, str]]
            One ``(column, "i8")`` pair.
        """
        column = f"{run_name}_{VERTEX_INDEX}" if self.prefix_vertex_column else VERTEX_INDEX
        return [(column, "i8")]

    def get_h5(self, b: Bundle, run_name: str) -> np.ndarray:
        """Per-node vertex assignments as one ``i8`` column.

        Union-finds the raw ``preds.*`` edge scores (``get_node_assignment_jit``
        -> ``_mask_fill_flattened``) then casts to int. Padded positions read
        the int32 cast of ``-inf`` (-2147483648).

        Returns
        -------
        np.ndarray
            ``[B, L]`` structured array with one ``i8`` field.
        """
        assert self.net is not None, "get_h5 before bind()"
        mask = b.get(f"masks.{self.stream}")
        preds = self.run_inference(b.get(self.pred_key), mask)
        dtype = np.dtype(self.output_names(run_name))
        return u2s(preds.int().cpu().numpy(), dtype)

    def onnx_outputs(self) -> list[ExportOutput]:
        """One ``vertex_union_find`` int8 entry on the shared `VERTEX_INDEX` suffix.

        The export suffix is the same `VERTEX_INDEX` constant the TEST column
        uses (the exporter prepends ``{model_name}_``). The in-graph union-find
        lives in the shipped ``vertex_union_find`` reduce; ONNX publishes raw
        edge scores that the reduce consumes.

        Returns
        -------
        list[ExportOutput]
            One ``vertex_union_find`` int8 entry.
        """
        return [
            ExportOutput(
                port=self.pred_key,
                name=VERTEX_INDEX,
                reduce="vertex_union_find",
                dtype="int8",
            )
        ]

    def output_time_requires(self, mode: Mode) -> list[str]:
        """The non-pred dep `get_output` reads: the stream pad mask.

        A vertexing head always has a pad mask (its ``forward`` requires
        ``masks.<stream>`` unconditionally), so this is mode-independent.

        Returns
        -------
        list[str]
            ``["masks.<stream>"]``.
        """
        del mode
        return [f"masks.{self.stream}"]

    def get_output(self, b: Bundle, mode: Mode, run_name: str) -> list[OutputField]:
        """Union-find the RAW edge scores into a graph-visible int8 leaf.

        Reads the RAW ``preds.*`` ``[E, 1]`` edge scores + the stream pad mask
        and runs the union-find conversion in TRACEABLE torch ops. Per mode:

        - **ONNX**: ``get_node_assignment_jit`` -> ``mask_fill_flattened`` ->
          ``.reshape(-1).char()`` -> one int8 ``[L]`` per-token field under the
          shared `VERTEX_INDEX` suffix.
        - **H5 modes** (TEST): ``run_inference`` -> ``.int()`` — the exact value
          ``get_h5`` packs. One int8 per-token field, ``onnx_name=None``. The
          column ``prefix`` follows ``prefix_vertex_column``.

        ``run_name`` is not baked in (the sink prefixes it).

        Returns
        -------
        list[OutputField]
            One int8 vertex-index field (the per-token union-find assignment).
        """
        del run_name
        assert self.net is not None, "get_output before bind()"
        edge_scores = b.get(self.pred_key)
        mask = b.get(f"masks.{self.stream}")
        if mode & Mode.ONNX:
            vertex_indices = get_node_assignment_jit(edge_scores, mask)
            vertex_list = mask_fill_flattened(vertex_indices, mask)
            return [
                OutputField(
                    h5_name=None,
                    onnx_name=VERTEX_INDEX,
                    dtype="int8",
                    axis="per_token",
                    final=True,
                    value=vertex_list.reshape(-1).char(),
                )
            ]
        # H5 (TEST): union-find then cast to int (-inf padding -> int32 -2147483648)
        preds = self.run_inference(edge_scores, mask).int()
        return [
            OutputField(
                h5_name=VERTEX_INDEX,
                onnx_name=None,
                dtype="i8",
                axis="per_token",
                final=True,
                prefix=self.prefix_vertex_column,
                value=preds,
            )
        ]

    def get_output_manifest(self, mode: Mode, run_name: str) -> list[OutputField]:
        """The value-free field metadata mirroring `get_output` for `mode` (``value=None``).

        Returns
        -------
        list[OutputField]
            The value-free serialisation field.
        """
        del run_name
        if mode & Mode.ONNX:
            return [
                OutputField(
                    h5_name=None,
                    onnx_name=VERTEX_INDEX,
                    dtype="int8",
                    axis="per_token",
                    final=True,
                )
            ]
        return [
            OutputField(
                h5_name=VERTEX_INDEX,
                onnx_name=None,
                dtype="i8",
                axis="per_token",
                final=True,
                prefix=self.prefix_vertex_column,
            )
        ]


def _is_name_weighting(heavy: Sequence[Any], fake: Sequence[Any]) -> bool:
    """Whether an ``origin_weighting`` config is class-name based.

    Name-based iff ANY heavy/fake entry is a string. A list mixing strings and
    ints is rejected here so the caller fails at construction, not silently
    half-resolved.

    Returns
    -------
    bool
        True for a (consistent) name-based config; False for all-integer ids.

    Raises
    ------
    ConfigError
        If any single role list mixes integer ids with class names.
    """
    for role, entries in (("heavy", heavy), ("fake", fake)):
        has_name = any(isinstance(e, str) for e in entries)
        has_id = any(not isinstance(e, str) for e in entries)
        if has_name and has_id:
            raise ConfigError(
                f"VertexingTaskModule: origin_weighting {role!r} list mixes integer ids with "
                f"class names ({list(entries)!r}) — use one or the other (design §5.1)"
            )
    return any(isinstance(e, str) for e in (*heavy, *fake))


def _coerce_origin_ids(entries: Sequence[Any], role: str) -> tuple[int, ...]:
    """Coerce an all-integer ``origin_weighting`` role list to a tuple of ids.

    Returns
    -------
    tuple[int, ...]
        The integer origin ids, in config order.

    Raises
    ------
    ConfigError
        On a non-integer entry (e.g. a float) — bool is rejected too (an origin
        id is never True/False).
    """
    ids: list[int] = []
    for e in entries:
        if isinstance(e, bool) or not isinstance(e, int):
            raise ConfigError(
                f"VertexingTaskModule: origin_weighting {role!r} entries must be INTEGER origin "
                f"ids or class NAMES (got {e!r} in {list(entries)!r}); v1's hardcoded ids are "
                "heavy: [3, 4, 5], fake: [1] (design §5.1)"
            )
        ids.append(int(e))
    return tuple(ids)
