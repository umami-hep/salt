"""Hungarian-matched MaskFormer loss GraphModule.

Given the decoder's object predictions and truth object labels, solves the
optimal 1-to-1 assignment of queries to truth objects (Hungarian matching)
and computes the matched classification/mask/regression loss components.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import (
    IO,
    OBJECT_STREAM,
    Mode,
    TensorSpec,
    sym_dim,
    unflatten_spec,
)
from salt.model.base import SaltModelModule
from salt.model.bind import ResolvedSchema
from salt.model.nn.maskformer_loss import MaskFormerLoss
from salt.model.nn.matcher import HungarianMatcher

__all__ = ["MaskFormerMatchedLoss"]


class MaskFormerMatchedLoss(SaltModelModule):
    """Hungarian-matched MaskFormer loss over the decoder's object predictions.

    A FIT|VAL-only module. Runs the matcher on the scaled object
    predictions/targets, then publishes the matcher-permuted predictions plus the
    truth labels as new ``matched.objects.*`` keys (no in-place permute) and emits
    ``losses.{object_class_ce, mask_dice, mask_focal, regression}`` for whichever
    components have a positive ``loss_weights`` entry.

    The regression component is the matched L1 over valid (non-null) objects.

    Parameters
    ----------
    num_classes : int
        The number of non-null object classes. The null/no-object class index is
        ``num_classes``. MUST equal the decoder's ``class_net.output_size - 1``.
    num_queries : int
        The number of object queries ``M``. Must equal the decoder's
        ``num_queries`` and the truth-object slot count. MUST be >= 1.
    loss_weights : Mapping[str, float]
        Per-component loss weights; keys among ``object_class_ce``, ``mask_dice``,
        ``mask_focal``, ``mask_ce``, ``regression``. A component is produced only
        when its weight is present and truthy. ``object_class_ce`` is always applied.
    matcher_weights : Mapping[str, float] | None, optional
        Per-component matcher cost weights, defaulting to ``loss_weights``.
    null_class_weight : float, optional
        The class-balance weight on the null category in the CE, by default 0.5.
    class_weights : list[float] | None, optional
        Optional per-class CE balance weights forwarded to the composed
        ``MaskFormerLoss``, by default None.
    input_stream : str, optional
        The decoder's object stream name, by default ``objects`` — the
        ``<input_stream>.{class_logits,class_probs,masks}`` keys it reads.

    Raises
    ------
    ConfigError
        On a non-positive ``num_classes`` / ``num_queries``, an unknown
        ``loss_weights`` / ``matcher_weights`` key, or an all-zero matcher-weight sum.
    """

    _KNOWN_COMPONENTS = ("object_class_ce", "mask_dice", "mask_focal", "mask_ce", "regression")

    def __init__(
        self,
        num_classes: int,
        num_queries: int,
        loss_weights: Mapping[str, float],
        matcher_weights: Mapping[str, float] | None = None,
        null_class_weight: float = 0.5,
        class_weights: list[float] | None = None,
        input_stream: str = "objects",
    ) -> None:
        super().__init__()
        for arg, value in (("num_classes", num_classes), ("num_queries", num_queries)):
            if value < 1:
                raise ConfigError(f"MaskFormerMatchedLoss: {arg} must be >= 1, got {value}")
        checked = {"loss_weights": loss_weights, "matcher_weights": matcher_weights or {}}
        for arg, weights in checked.items():
            if unknown := sorted(set(weights) - set(self._KNOWN_COMPONENTS)):
                raise ConfigError(
                    f"MaskFormerMatchedLoss: unknown {arg} keys {unknown} — known components "
                    f"are {list(self._KNOWN_COMPONENTS)}"
                )
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.input_stream = input_stream
        # the scaled object-regression (prediction, target) keys
        self._reg_keys = (f"preds.{input_stream}.regression", f"targets.{input_stream}.regression")

        self.loss_weights = {k: float(v) for k, v in dict(loss_weights).items()}
        mw = self.loss_weights if matcher_weights is None else dict(matcher_weights)
        self.matcher_weights = {k: float(v) for k, v in mw.items()}
        # get_batch_cost reads object_class_ce unconditionally — default it to the
        # loss weight if absent.
        self.matcher_weights.setdefault(
            "object_class_ce", self.loss_weights.get("object_class_ce", 1.0)
        )
        # an all-zero matcher cost is degenerate; the matcher itself does not re-check.
        if sum(self.matcher_weights.values()) == 0:
            raise ConfigError(
                "MaskFormerMatchedLoss: the matcher cost weights sum to 0 — at least one "
                f"matcher_weights entry must be positive (got {self.matcher_weights})"
            )

        # object_class_ce is always computed (so this is never empty); the others only
        # when their loss weight is truthy. regression is the matched extension.
        self.components: tuple[str, ...] = tuple(
            c for c in self._KNOWN_COMPONENTS if c == "object_class_ce" or self.loss_weights.get(c)
        )
        self.null_class_weight = float(null_class_weight)
        self.class_weights = list(class_weights) if class_weights is not None else None

        # composes the HungarianMatcher + empty_weight buffer + loss_labels/loss_masks
        self.v1_loss = MaskFormerLoss(
            num_classes=num_classes,
            num_objects=num_queries,
            loss_weights=self.loss_weights,
            matcher_weights=self.matcher_weights,
            null_class_weight=self.null_class_weight,
            class_weights=self.class_weights,
        )

    @property
    def matcher(self) -> HungarianMatcher:
        """The composed `HungarianMatcher` (owned by the loss)."""
        return self.v1_loss.matcher

    def declare_io(self, mode: Mode) -> IO:
        """Declare the object preds + truth labels -> ``matched.*``/``losses.*`` (FIT|VAL only)."""
        if not (mode & Mode.TRAINING):
            return IO(requires={}, produces={})
        s, lbl, mo = self.input_stream, f"labels.{OBJECT_STREAM}", f"matched.{OBJECT_STREAM}"
        n_cls, tok, emb = self.num_classes + 1, sym_dim("T", self.name), sym_dim("E", self.name)

        def _spec(*tail: int | str, dtype: str = "float32", **kw: Any) -> TensorSpec:
            shape = ("B", self.num_queries, *tail)
            return TensorSpec(shape=shape, dtype=dtype, modes=Mode.TRAINING, **kw)

        # insertion order is kept stable (it fixes the nested spec order)
        requires = {
            f"{s}.class_logits": _spec(n_cls),
            f"{s}.class_probs": _spec(n_cls),
            f"{s}.masks": _spec(tok),
            f"{lbl}.object_class": _spec(dtype="int64", kind="label"),
            f"{lbl}.masks": _spec(tok, dtype="bool", kind="label"),
        }
        produces = {
            f"{mo}.embed": _spec(emb),
            f"{mo}.class_logits": _spec(n_cls),
            f"{mo}.class_probs": _spec(n_cls),
            f"{mo}.masks": _spec(tok),
            f"{mo}.object_class": _spec(dtype="int64", kind="label"),
            f"{mo}.target_masks": _spec(tok, dtype="bool", kind="label"),
        }
        if "regression" in self.components:
            r = sym_dim("R", self.name)
            requires.update({key: _spec(r) for key in self._reg_keys})
            produces.update({f"{mo}.regression": _spec(r), f"{mo}.target_regression": _spec(r)})
        requires[f"{s}.embed"] = _spec(emb)
        for component in self.components:
            produces[f"losses.{component}"] = TensorSpec(shape=(), kind="loss", modes=Mode.TRAINING)
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def bind(self, schema: ResolvedSchema) -> None:
        """Validate the regression prediction/target widths agree; skips if either is unresolved."""
        if "regression" not in self.components or any(
            key not in schema.widths for key in self._reg_keys
        ):
            return
        pred_key, tgt_key = self._reg_keys
        if (wp := schema.width(pred_key)) != (wt := schema.width(tgt_key)):
            raise ConfigError(
                f"MaskFormerMatchedLoss {self.name!r}: regression prediction width {wp} != "
                f"target width {wt} ({pred_key!r} vs {tgt_key!r}) — the matched L1 is element-wise"
            )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Match queries to truth; emit matched preds (new ``matched.*`` keys) + loss components."""
        del mode
        s, mo = self.input_stream, f"matched.{OBJECT_STREAM}"
        class_logits, class_probs, masks, embed = (
            b.get(f"{s}.{leaf}") for leaf in ("class_logits", "class_probs", "masks", "embed")
        )
        object_class = b.get(f"labels.{OBJECT_STREAM}.object_class")
        target_masks = b.get(f"labels.{OBJECT_STREAM}.masks")

        # matcher cost stays in scaled space
        pred_for_match = {"class_logits": class_logits, "class_probs": class_probs, "masks": masks}
        tgt_for_match = {"object_class": object_class, "masks": target_masks.to(masks.dtype)}
        reg_pred = reg_tgt = None
        if "regression" in self.components:
            reg_pred, reg_tgt = b.get(self._reg_keys[0]), b.get(self._reg_keys[1])
            pred_for_match["regression"], tgt_for_match["regression"] = reg_pred, reg_tgt
        idx = self.matcher(pred_for_match, tgt_for_match)

        # advanced indexing returns fresh tensors, so matched.* never aliases the
        # decoder output (no in-place permute).
        m_class_logits, m_class_probs, m_masks, m_embed = (
            t[idx] for t in (class_logits, class_probs, masks, embed)
        )
        out: dict[str, Tensor] = {
            f"{mo}.embed": m_embed,
            f"{mo}.class_logits": m_class_logits,
            f"{mo}.class_probs": m_class_probs,
            f"{mo}.masks": m_masks,
            f"{mo}.object_class": object_class,
            f"{mo}.target_masks": target_masks,
        }
        permuted = {"class_logits": m_class_logits, "masks": m_masks}
        truth = {"object_class": object_class, "masks": target_masks}
        losses = self.v1_loss.loss_labels(permuted, truth)
        if any(self.loss_weights.get(c) for c in ("mask_dice", "mask_focal", "mask_ce")):
            losses.update(self.v1_loss.loss_masks(permuted, truth))

        for component in self.components:
            if component == "regression":
                assert reg_pred is not None
                assert reg_tgt is not None
                m_reg = reg_pred[idx]
                # matched L1 (mean over valid objects x R; preds already truth-aligned);
                # 0.0 if the batch has no valid (non-null) object
                valid = object_class != self.num_classes  # [B, M]
                reg_loss = (
                    torch.nn.functional.l1_loss(m_reg[valid], reg_tgt[valid])
                    if valid.any()
                    else m_reg.new_zeros(())
                )
                out[f"{mo}.regression"] = m_reg
                out[f"{mo}.target_regression"] = reg_tgt
                out["losses.regression"] = self.loss_weights["regression"] * reg_loss
            else:
                out[f"losses.{component}"] = self.loss_weights[component] * losses[component]
        return out
