"""Classification task module."""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import yaml
from torch import Tensor

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import IO, Mode, TensorSpec
from salt.model.bind import ResolvedSchema
from salt.model.modules.tasks.base import (
    _checked_weight_source,
    _loss_class,
    _TaskModuleBase,
)
from salt.model.nn.dense import Dense
from salt.outputs.output_schema import OutputField, pascal_case
from salt.utils.tensor_utils import masked_softmax

_DEFAULT_CLS_LOSS: dict[str, Any] = {"class_path": "torch.nn.CrossEntropyLoss"}


class ClassificationTaskModule(_TaskModuleBase):
    """Classification head over a pooled vector or a per-stream sequence.

    ``class_names`` is REQUIRED and explicit. When the dataset schema names the
    label's classes, the configured list is cross-checked (set and order) by
    `salt.model.validation.check_class_names`. The head width is
    ``len(class_names)``; input/context widths are inferred at `bind`.

    Declarative class weights: ``weight_source: {from_class_dict: <path>}``
    allocates the CE ``weight`` buffer at `bind`, fills it at `materialise()`
    on fresh fits (the only file I/O), and inherits it from the checkpoint on
    resume. A literal ``loss.init_args.weight`` list is exclusive with
    ``weight_source``. The class dict is indexed by the stream's dataset name
    (`ResolvedSchema.dataset_of`), not the raw stream name.

    Class weights and per-row ``sample_weight`` weights (one label key, or a list of
    keys whose per-row PRODUCT is the weight, e.g. ``[nn_kinematic_weight,
    flavour_class_weight]`` = a `KinematicWeight` times a `ClassWeight` column) are
    independent and MULTIPLY. Rows with a consumed label of ``-1``
    (padded, invalid, or another jet type's rows) are inert on every loss path,
    including an all-ignored batch (loss 0, finite gradients).
    """

    def __init__(
        self,
        stream: str,
        label: str,
        class_names: Sequence[str],
        input: str | None = None,  # noqa: A002 - matches the YAML config key name
        context: str | None = None,
        sequence: bool | None = None,
        dense: dict[str, Any] | None = None,
        loss: str | dict[str, Any] | None = None,
        weight: float = 1.0,
        weight_source: Mapping[str, str] | None = None,
        label_map: dict[int, int] | None = None,
        expose: Sequence[str] | None = None,
        write_targets: bool = True,
        sample_weight: str | Sequence[str] | None = None,
    ) -> None:
        """Capture config only.

        Parameters
        ----------
        class_names : Sequence[str]
            Ordered class names — REQUIRED, index-aligned with outputs.
        sequence : bool | None, optional
            Whether the task is per-token, by default inferred from `input`.
        weight_source : Mapping[str, str] | None, optional
            ``{"from_class_dict": <path>}`` declarative class-weight source
            (see class docstring), by default None.
        expose : Sequence[str] | None, optional
            Modes the ``preds.*`` port is published in, by default all modes.
            ``[fit, val]`` opts a train-only aux task out of TEST/ONNX.
        write_targets : bool, optional
            Also emit the consumed class label (post ``label_map`` remap) as
            the TEST eval column ``target_{task}``, by default True. Never
            emitted in ONNX/export mode.
        sample_weight : str | Sequence[str] | None, optional
            Per-sample loss-weight label(s) ``labels.<stream>.<name>`` (FIT|VAL
            only), by default None; several keys multiply per row. Requires
            ``loss.reduction == 'none'``; the weighted mean is taken in `calculate_loss`.

        Raises
        ------
        ConfigError
            On empty/duplicate class names, malformed `weight_source`, a
            literal loss weight combined with `weight_source`, an empty or
            duplicated `sample_weight` list, or one with a non-``none`` loss
            reduction, or a bad `expose` list.
        """
        super().__init__(
            stream,
            label,
            input,
            context,
            dense,
            loss,
            weight,
            _DEFAULT_CLS_LOSS,
            expose,
            write_targets,
        )
        if not class_names:
            raise ConfigError(
                f"ClassificationTaskModule: class_names is required and explicit for label "
                f"{label!r} (no CLASS_NAMES fallback)"
            )
        if len(set(class_names)) != len(tuple(class_names)):
            raise ConfigError(
                f"ClassificationTaskModule: duplicate class names in {tuple(class_names)}"
            )
        self.class_names = tuple(class_names)
        self.sequence = sequence if sequence is not None else input is None
        self.label_map = dict(label_map) if label_map is not None else None
        self.weight_source = _checked_weight_source(weight_source)
        self._class_dict_key: str = stream
        if self.weight_source is not None and "weight" in self.loss_cfg.get("init_args", {}):
            raise ConfigError(
                "ClassificationTaskModule: class weights already specified in the loss config — "
                "remove them or drop weight_source (v1 use_class_dict contract, cli.py:476-479)"
            )
        self.sample_weights: tuple[str, ...] = (
            ()
            if sample_weight is None
            else (sample_weight,)
            if isinstance(sample_weight, str)
            else tuple(sample_weight)
        )
        if sample_weight is not None and not self.sample_weights:
            raise ConfigError(
                "ClassificationTaskModule: sample_weight must name at least one label"
            )
        if len(set(self.sample_weights)) != len(self.sample_weights):
            raise ConfigError(
                f"ClassificationTaskModule: duplicate sample_weight keys {self.sample_weights}"
            )
        if self.sample_weights and (
            self.loss_cfg.get("init_args", {}).get("reduction", "mean") != "none"
        ):
            raise ConfigError(
                f"ClassificationTaskModule: sample_weight {list(self.sample_weights)!r} requires "
                "the loss reduction to be 'none' — the per-sample weight multiplies the unreduced "
                "loss before the weighted-mean reduction; set loss: "
                "{class_path: torch.nn.CrossEntropyLoss, init_args: {reduction: none}}"
            )

    @property
    def weight_label_keys(self) -> tuple[str, ...]:
        """``labels.<stream>.<key>`` for every `sample_weights` entry (empty when unset)."""
        return tuple(f"labels.{self.stream}.{k}" for k in self.sample_weights)

    def declare_io(self, mode: Mode) -> IO:
        """Global or per-token pred spec depending on `sequence`; label/loss ports
        (and the ``sample_weight`` labels) are TRAINING-only.
        """
        del mode
        label = TensorSpec(shape=self._tokens, dtype="int64", kind="label", modes=Mode.TRAINING)
        labels = {self.label_key: label}
        for key in self.weight_label_keys:
            labels[key] = TensorSpec(
                shape=self._tokens, dtype="float32", kind="label", modes=Mode.TRAINING
            )
        return self._head_io(label_specs=labels, pred_shape=(*self._tokens, len(self.class_names)))

    def bind(self, schema: ResolvedSchema) -> None:
        """Builds the head; when `weight_source` is set, allocates a ones-initialised
        loss ``weight`` buffer for `materialise` to fill on fresh fits.
        """
        self._class_dict_key = schema.dataset_of(self.stream)
        init_args = dict(self.loss_cfg.get("init_args", {}))
        if self.weight_source is not None:
            init_args["weight"] = torch.ones(len(self.class_names))
        elif isinstance(init_args.get("weight"), (list, tuple)):
            init_args["weight"] = torch.as_tensor(init_args["weight"], dtype=torch.float32)
        loss_module = _loss_class(self.loss_cfg)(**init_args)
        if hasattr(loss_module, "ignore_index"):
            loss_module.ignore_index = -1
        self.loss = loss_module
        self.net = Dense(
            input_size=schema.width(self.input_key),
            output_size=len(self.class_names),
            **({"context_size": schema.width(self.context)} if self.context else {}),
            **self.dense_cfg,
        )

    def run_inference(self, preds: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        """Convert logits to probabilities.

        Returns
        -------
        Tensor
            Probabilities with the same leading dimensions as ``preds``.
        """
        if isinstance(self.loss, torch.nn.BCEWithLogitsLoss):
            probs = torch.sigmoid(preds)
        elif pad_mask is None:
            assert preds.ndim == 2
            probs = torch.softmax(preds, dim=-1)
        else:
            assert preds.ndim == 3
            probs = masked_softmax(preds, pad_mask.unsqueeze(-1))
        return probs

    def materialise(self) -> None:
        """Resolve ``weight_source`` from the class dict (the ONLY file I/O).

        Raises
        ------
        RuntimeError
            If called before `bind`.
        ValueError
            If the class dict lacks the stream/label or the weight count
            does not match ``class_names``.
        """
        if self.weight_source is None:
            return
        if self.loss is None:
            raise RuntimeError(f"{type(self).__name__} {self.name!r}: materialise before bind()")
        path = self.weight_source["from_class_dict"]
        with open(path) as fh:
            class_dict = yaml.safe_load(fh)
        try:
            values = class_dict[self._class_dict_key][self.label]
        except (KeyError, TypeError):
            raise ValueError(
                f"Label {self.label!r} for stream {self.stream!r} "
                f"(class-dict key {self._class_dict_key!r}) "
                f"not found in class dict {path} — drop weight_source and specify class "
                f"weights manually (cli.py:483-488 semantics)"
            ) from None
        if len(values) != len(self.class_names):
            raise ValueError(
                f"class dict {path} has {len(values)} weights for {self.stream}.{self.label} "
                f"but the task declares {len(self.class_names)} class_names"
            )
        with torch.no_grad():
            self.loss.weight.copy_(torch.as_tensor(values, dtype=torch.float32))

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Publishes RAW logits in every non-training mode; conversion happens
        exactly once downstream in `get_output`.
        """
        assert self.net is not None, "forward before bind()"
        ctx = b.get(self.context) if self.context is not None else None
        preds = self.net(b.get(self.input_key), ctx)
        if not (mode & Mode.TRAINING):
            return {self.pred_key: preds}
        labels = self._consumed_labels(b)
        keys = self.weight_label_keys
        sample_weights = functools.reduce(torch.mul, (b.get(k) for k in keys)) if keys else None
        loss = self.calculate_loss(preds, labels, sample_weights)
        return {self.pred_key: preds, self.loss_key: loss * self.weight}

    def calculate_loss(
        self, preds: Tensor, labels: Tensor, sample_weights: Tensor | None = None
    ) -> Tensor:
        """The head loss over the consumed `labels`, before the task ``weight``.

        Without `sample_weights`, a ``CrossEntropyLoss(reduction='mean')`` reduces via
        `_masked_ce_mean` for both global ``[B, C]`` and sequence ``[B, T, C]`` heads
        (the latter after moving the class axis to ``[B, C, T]``). With them, the
        ``reduction='none'`` loss is reduced inline to
        ``sum(w_i * c_{y_i} * l_i) / sum(w_i * c_{y_i})`` over rows with ``label != -1``:
        ``CrossEntropyLoss(reduction='none')`` already returns the class-weighted
        ``c_{y_i} * l_i``, so this mirrors torch's weighted
        ``reduction='mean'``. ``c`` is the loss's class ``weight`` (``1`` when unset,
        and always for `torch.nn.BCEWithLogitsLoss`, whose ``weight`` is per element).

        Returns
        -------
        Tensor
            The scalar loss (an all-ignored or all-zero-weight batch gives 0).
        """
        masked_mean = (
            sample_weights is None
            and isinstance(self.loss, torch.nn.CrossEntropyLoss)
            and self.loss.reduction == "mean"
        )
        if preds.ndim != 3 and isinstance(self.loss, torch.nn.BCEWithLogitsLoss):
            loss = self.loss(preds.squeeze(-1), labels.float())
        else:  # a sequence head puts the class axis in place: [B, T, C] -> [B, C, T]
            logits = preds.permute(0, 2, 1) if preds.ndim == 3 else preds
            loss = (
                self._masked_ce_mean(logits, labels) if masked_mean else self.loss(logits, labels)
            )
        if sample_weights is None:
            return loss
        w_row = sample_weights * (labels != -1).to(loss.dtype)
        class_weight = (
            None
            if isinstance(self.loss, torch.nn.BCEWithLogitsLoss)
            else getattr(self.loss, "weight", None)
        )
        c_row = class_weight[labels.clamp(min=0)] if class_weight is not None else 1.0
        denominator = (w_row * c_row).sum()
        return (w_row * loss).sum() / denominator.clamp(min=torch.finfo(denominator.dtype).tiny)

    def _masked_ce_mean(self, preds: Tensor, labels: Tensor) -> Tensor:
        """``CrossEntropyLoss(reduction='mean')`` that survives an all-ignored batch.

        Covers global ``[B, C]`` logits with ``[B]`` labels and sequence ``[B, C, T]``
        logits with ``[B, T]`` labels (per-element losses and masks are elementwise).
        Torch's weighted mean ``sum(c_y * l) / sum(c_y)`` over ``label != -1`` rows or
        tokens is 0/0 = NaN when none is supervised (e.g. a batch holding only the
        other jet type, or a sequence batch with every token ignored). The clamped
        denominator makes that 0 with zero gradients; with at least one valid row or
        token it equals torch's mean.
        """
        assert isinstance(self.loss, torch.nn.CrossEntropyLoss)
        class_weight = self.loss.weight
        per = torch.nn.functional.cross_entropy(
            preds,
            labels,
            weight=class_weight,
            ignore_index=-1,
            reduction="none",
            label_smoothing=self.loss.label_smoothing,
        )
        valid = (labels != -1).to(per.dtype)
        c_row = (
            class_weight[labels.clamp(min=0)].to(per.dtype)
            if class_weight is not None
            else torch.ones_like(per)
        )
        denominator = (c_row * valid).sum()
        return (per * valid).sum() / denominator.clamp(min=torch.finfo(per.dtype).tiny)

    # -- output rendering ---------------------------------------------------

    @property
    def class_suffixes(self) -> list[str]:
        """Per-class logical suffixes, in class order — ``Flavours[c].px`` else ``p{c}``."""
        # Must stay a LOCAL import: `Flavours` (ftag LabelContainer) raises
        # KeyError (not AttributeError) from __getattr__ on unknown names, which
        # breaks jsonargparse's hasattr(value, "__args__") protocol walk over
        # this module's globals if imported at module level — the whole
        # model.modules config would fail to parse.
        from ftag import Flavours

        return [Flavours[c].px if c in Flavours else f"p{c}" for c in self.class_names]

    def output_time_requires(self, mode: Mode) -> list[str]:
        """``["masks.<stream>"]`` for a padded seq head, plus the task's label key in
        TEST when ``write_targets`` (never in ONNX).
        """
        return self._output_deps(mode, [self.label_key])

    def _output_values(self, b: Bundle, mode: Mode) -> list[Tensor]:
        """Global head: `run_inference` probs, one ``f4`` field per class (ONNX squeezes to
        rank-0 scalars). Sequence head: masked softmax, one ``f4`` field per class in H5 modes;
        ONNX instead emits a single int8 argmax field named ``pascal_case(name)``.
        """
        assert self.net is not None, "get_output before bind()"
        logits = b.get(self.pred_key)
        classes = range(len(self.class_names))
        if not self.sequence:
            probs = self.run_inference(logits)
            # ONNX sink squeezes [1, C] per-class columns to rank-0 scalars; H5
            # sink packs the full [B] column, so only squeeze in ONNX mode
            values = [
                probs[..., c].squeeze() if mode & Mode.ONNX else probs[..., c] for c in classes
            ]
        else:
            mask = b.get(f"masks.{self.stream}") if self.has_pad_mask else None
            probs = masked_softmax(logits, mask.unsqueeze(-1) if mask is not None else None)
            if mode & Mode.ONNX:
                # zero-row append/strip -> [L] int8 argmax leaf under the pascal-case task name
                padded = torch.concatenate([probs, torch.zeros((1, 1, probs.shape[-1]))], dim=1)
                values = [torch.argmax(padded, dim=-1)[:, :-1].squeeze(0).char()]
            else:
                # H5-only: the ONNX side of a seq head is the argmax leaf above, not the
                # manifest's per-class fields. Callers MUST select ONNX outputs by calling
                # with mode=Mode.ONNX, not by scanning resolved_onnx_name — these fields
                # fall back to a (misleading) per-class ONNX suffix since onnx_name=None.
                values = [probs[..., c] for c in classes]
        if self._emit_targets(mode):
            values.append(self._consumed_labels(b))
        return values

    def _consumed_labels(self, b: Bundle) -> Tensor:
        """The class label exactly as the loss consumes it (`forward`'s training
        target and the TEST ``target_{task}`` column): post ``label_map`` remap;
        for a padded seq head, padded and invalid (``-2``) positions read ``-1``.
        """
        labels = b.get(self.label_key)
        if self.label_map is not None:
            mapped = torch.clone(labels)
            for k, v in self.label_map.items():
                mapped[labels == k] = v
            labels = mapped
        if self.has_pad_mask:
            pad_mask = b.get(f"masks.{self.stream}")
            # TODO @npond: remove the -2 fill once fixed upstream (MR!60199)
            pad_mask = torch.masked_fill(pad_mask, labels == -2, True)
            labels = torch.masked_fill(labels, pad_mask, -1)
        return labels

    def get_output_manifest(self, mode: Mode, run_name: str) -> list[OutputField]:
        """The static, value-free descriptors for `mode` (``value=None``) — `get_output` derives its
        fields from this list via ``replace(field, value=...)``.
        """
        del run_name
        if self.sequence and mode & Mode.ONNX:
            name = pascal_case(self.name)
            return [OutputField(None, onnx_name=name, dtype="int8", axis="per_token", final=True)]
        axis = "per_token" if self.sequence else "global"
        fields = [OutputField(px, dtype="f4", axis=axis, final=True) for px in self.class_suffixes]
        if self._emit_targets(mode):
            fields += self._target_fields("i4")
        return fields
