"""Classification task module."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import yaml
from torch import Tensor

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.model.bind import ResolvedSchema
from salt.model.modules.stream_embed import _stream_len
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
    `salt.model.saltmodule.check_class_names`. The head width is
    ``len(class_names)``; input/context widths are inferred at `bind`.

    Declarative class weights: ``weight_source: {from_class_dict: <path>}``
    allocates the CE ``weight`` buffer at `bind`, fills it at `materialise()`
    on fresh fits (the only file I/O), and inherits it from the checkpoint on
    resume. A literal ``loss.init_args.weight`` list is exclusive with
    ``weight_source``.
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

        Raises
        ------
        ConfigError
            On empty/duplicate class names, malformed `weight_source`, a
            literal loss weight combined with `weight_source`, or a bad
            `expose` list.
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
                f"{label!r} (design §3.3 — no CLASS_NAMES fallback)"
            )
        if len(set(class_names)) != len(tuple(class_names)):
            raise ConfigError(
                f"ClassificationTaskModule: duplicate class names in {tuple(class_names)}"
            )
        self.class_names = tuple(class_names)
        self.sequence = sequence if sequence is not None else input is None
        self.label_map = dict(label_map) if label_map is not None else None
        # per-sample loss weighting is not part of this module's config surface
        self.sample_weight: str | None = None
        self.weight_source = _checked_weight_source(weight_source)
        if self.weight_source is not None and "weight" in self.loss_cfg.get("init_args", {}):
            raise ConfigError(
                "ClassificationTaskModule: class weights already specified in the loss config — "
                "remove them or drop weight_source (v1 use_class_dict contract, cli.py:476-479)"
            )

    def declare_io(self, mode: Mode) -> IO:
        """Global or per-token pred spec depending on `sequence`; label/loss ports
        are TRAINING-only.
        """
        del mode
        n_classes = len(self.class_names)
        width = sym_dim("D", self.name)
        if self.sequence:
            input_spec = TensorSpec(shape=("B", _stream_len(self.stream), width), dtype="float32")
            label_spec = TensorSpec(
                shape=("B", _stream_len(self.stream)),
                dtype="int64",
                kind="label",
                modes=Mode.TRAINING,
            )
            pred_spec = TensorSpec(
                shape=("B", _stream_len(self.stream), n_classes), dtype="float32"
            )
        else:
            input_spec = TensorSpec(shape=("B", width), dtype="float32")
            label_spec = TensorSpec(shape=("B",), dtype="int64", kind="label", modes=Mode.TRAINING)
            pred_spec = TensorSpec(shape=("B", n_classes), dtype="float32")
        requires: dict[str, TensorSpec] = {self.input_key: input_spec}
        if self.context is not None:
            requires[self.context] = TensorSpec(shape=None, dtype="float32")
        if self.has_pad_mask:
            requires[f"masks.{self.stream}"] = TensorSpec(
                shape=("B", _stream_len(self.stream)), dtype="bool", kind="pad_mask"
            )
        requires[self.label_key] = label_spec
        produces: dict[str, TensorSpec] = {
            self.pred_key: self._pred_spec(pred_spec),
            self.loss_key: TensorSpec(shape=(), kind="loss", modes=Mode.TRAINING),
        }
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def bind(self, schema: ResolvedSchema) -> None:
        """Builds the head; when `weight_source` is set, allocates a ones-initialised
        loss ``weight`` buffer for `materialise` to fill on fresh fits.
        """
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

    def apply_sample_weight(self, loss: Tensor, labels_dict: Mapping) -> Tensor:
        """Apply per-sample weights to a loss tensor if configured.

        Returns
        -------
        Tensor
            Weighted mean loss if ``sample_weight`` is set; otherwise the input.
        """
        if self.sample_weight is None:
            return loss
        return (loss * labels_dict[self.input_name][self.sample_weight]).mean()

    def head_forward(
        self,
        x: Tensor,
        labels_dict: Mapping | None,
        pad_masks: Mapping | None = None,
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute logits and classification loss.

        Returns
        -------
        tuple[Tensor, Tensor | None]
            Predicted logits and the loss (``None`` when no labels).
        """
        if pad_masks is not None:
            preds = self.net(x[:, self.input_name_slice(pad_masks)], context)
            pad_mask = pad_masks[self.input_name]
        else:
            preds = self.net(x, context)
            pad_mask = None

        labels = labels_dict[self.input_name][self.label] if labels_dict else None
        if labels is not None and self.label_map is not None:
            mapped_labels = torch.clone(labels)
            for k, v in self.label_map.items():
                mapped_labels[labels == k] = v
            labels = mapped_labels

        if pad_mask is not None and labels is not None:
            # TODO @npond: remove once fixed upstream (MR!60199)
            pad_mask = torch.masked_fill(pad_mask, labels == -2, True)
            labels = torch.masked_fill(labels, pad_mask, -1)

        loss: Tensor | None = None
        if labels is not None:
            if preds.ndim == 3:
                loss = self.loss(preds.permute(0, 2, 1), labels)
            elif isinstance(self.loss, torch.nn.BCEWithLogitsLoss):
                loss = self.loss(preds.squeeze(-1), labels.float())
            else:
                loss = self.loss(preds, labels)
            loss = self.apply_sample_weight(loss, labels_dict)
            loss *= self.weight

        return preds, loss

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
            values = class_dict[self.stream][self.label]
        except (KeyError, TypeError):
            raise ValueError(
                f"Label {self.label!r} for stream {self.stream!r} not found in class dict "
                f"{path} — drop weight_source and specify class weights manually "
                f"(cli.py:483-488 semantics)"
            ) from None
        if len(values) != len(self.class_names):
            raise ValueError(
                f"class dict {path} has {len(values)} weights for {self.stream}.{self.label} "
                f"but the task declares {len(self.class_names)} class_names (design §3.3)"
            )
        with torch.no_grad():
            self.loss.weight.copy_(torch.as_tensor(values, dtype=torch.float32))

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Publishes RAW logits in every non-training mode; conversion happens
        exactly once downstream in `get_output`.
        """
        assert self.net is not None, "forward before bind()"
        x = b.get(self.input_key)
        ctx = b.get(self.context) if self.context is not None else None
        # objects-stream (query-bank) heads have no pad mask
        mask = b.get(f"masks.{self.stream}") if self.has_pad_mask else None
        if mode & Mode.TRAINING:
            labels_dict = {self.stream: {self.label: b.get(self.label_key)}}
            pad_masks = {self.stream: mask} if self.has_pad_mask else None
            preds, loss = self.head_forward(x, labels_dict, pad_masks, context=ctx)
            return {self.pred_key: preds, self.loss_key: loss}
        preds, _ = self.head_forward(x, None, None, context=ctx)
        return {self.pred_key: preds}

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
        # has_pad_mask already implies sequence, so this covers all three cases:
        # global [] / no-pad-mask seq [] / padded seq [masks.<stream>]
        deps: list[str] = [f"masks.{self.stream}"] if self.has_pad_mask else []
        if self._emit_targets(mode):
            deps.append(self.label_key)
        return deps

    def get_output(self, b: Bundle, mode: Mode, run_name: str) -> list[OutputField]:
        """Global head: sigmoid (BCE) or softmax, one ``f4`` field per class (ONNX squeezes to
        rank-0 scalars). Sequence head: masked softmax, one ``f4`` field per class in H5 modes;
        ONNX instead emits a single int8 argmax field named ``pascal_case(name)``.
        """
        del run_name
        assert self.net is not None, "get_output before bind()"
        logits = b.get(self.pred_key)
        if not self.sequence:
            if isinstance(self.loss, torch.nn.BCEWithLogitsLoss):
                probs = torch.sigmoid(logits)
            else:
                assert logits.ndim == 2, "global classification head expects [B, C] logits"
                probs = torch.softmax(logits, dim=-1)
            # ONNX sink squeezes [1, C] per-class columns to rank-0 scalars; H5
            # sink packs the full [B] column, so only squeeze in ONNX mode
            squeeze_global = bool(mode & Mode.ONNX)
            fields = [
                OutputField(
                    h5_name=px,
                    dtype="f4",
                    axis="global",
                    final=True,
                    value=probs[..., c].squeeze() if squeeze_global else probs[..., c],
                )
                for c, px in enumerate(self.class_suffixes)
            ]
            if self._emit_targets(mode):
                fields.append(self._target_field(value=self._consumed_labels(b)))
            return fields
        mask = b.get(f"masks.{self.stream}") if self.has_pad_mask else None
        probs = masked_softmax(logits, mask.unsqueeze(-1) if mask is not None else None)
        if mode & Mode.ONNX:
            # zero-row append/strip -> [L] int8 argmax leaf under the pascal-case task name
            padded = torch.concatenate([probs, torch.zeros((1, 1, probs.shape[-1]))], dim=1)
            index = torch.argmax(padded, dim=-1)[:, :-1].squeeze(0).char()
            return [
                OutputField(
                    h5_name=None,
                    onnx_name=pascal_case(self.name),
                    dtype="int8",
                    axis="per_token",
                    final=True,
                    value=index,
                )
            ]
        # H5-only: the ONNX side of a seq head is the argmax leaf above, not these
        # per-class probs. Callers MUST select ONNX outputs by calling with
        # mode=Mode.ONNX (which returns before reaching here), not by scanning
        # resolved_onnx_name — these fields fall back to a (misleading) per-class
        # ONNX suffix since onnx_name=None.
        fields = [
            OutputField(
                h5_name=px,
                onnx_name=None,
                dtype="f4",
                axis="per_token",
                final=True,
                value=probs[..., c],
            )
            for c, px in enumerate(self.class_suffixes)
        ]
        if self._emit_targets(mode):
            fields.append(self._target_field(value=self._consumed_labels(b)))
        return fields

    def _target_field(self, value: Tensor | None = None) -> OutputField:
        """The target-label field: the consumed class label as an unprefixed
        ``target_{task}`` i4 column (labels are model-independent).
        """
        return OutputField(
            h5_name=f"target_{self.name}",
            onnx_name=None,
            dtype="i4",
            axis="per_token" if self.sequence else "global",
            final=True,
            prefix=False,
            value=value,
        )

    def _consumed_labels(self, b: Bundle) -> Tensor:
        """The class label exactly as the loss consumes it: post ``label_map``
        remap; for a padded seq head, padded and invalid (``-2``) positions
        read ``-1`` (mirrors `head_forward`).
        """
        labels = b.get(self.label_key)
        if self.label_map is not None:
            mapped = torch.clone(labels)
            for k, v in self.label_map.items():
                mapped[labels == k] = v
            labels = mapped
        if self.has_pad_mask:
            pad_mask = b.get(f"masks.{self.stream}")
            pad_mask = torch.masked_fill(pad_mask, labels == -2, True)
            labels = torch.masked_fill(labels, pad_mask, -1)
        return labels

    def get_output_manifest(self, mode: Mode, run_name: str) -> list[OutputField]:
        """The value-free field metadata mirroring `get_output` for `mode` (``value=None``)."""
        del run_name
        if not self.sequence:
            fields = [
                OutputField(h5_name=px, dtype="f4", axis="global", final=True)
                for px in self.class_suffixes
            ]
            if self._emit_targets(mode):
                fields.append(self._target_field())
            return fields
        if mode & Mode.ONNX:
            return [
                OutputField(
                    h5_name=None,
                    onnx_name=pascal_case(self.name),
                    dtype="int8",
                    axis="per_token",
                    final=True,
                )
            ]
        fields = [
            OutputField(h5_name=px, onnx_name=None, dtype="f4", axis="per_token", final=True)
            for px in self.class_suffixes
        ]
        if self._emit_targets(mode):
            fields.append(self._target_field())
        return fields
