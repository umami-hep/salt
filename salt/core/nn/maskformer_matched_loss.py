"""Hungarian-matched MaskFormer loss GraphModule.

Given the decoder's object predictions and truth object labels, solves the
optimal 1-to-1 assignment of queries to truth objects (Hungarian matching)
and computes the matched classification/mask/regression loss components.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.core.nn.bind import ResolvedSchema
from salt.core.nn.maskformer_loss import MaskFormerLoss
from salt.core.nn.matcher import HungarianMatcher

__all__ = ["MaskFormerMatchedLoss"]


_UNNAMED = "unnamed"
"""Placeholder instance name, assigned before compile."""

# bundle stream name for the reconstructed objects (matches the MaskDecoder
# out_stream and the MaskFormerTargets object stream).
_OBJECT_STREAM = "objects"


class MaskFormerMatchedLoss(nn.Module):
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
    num_objects : int
        The number of object queries ``M``. Must equal the decoder's
        ``num_objects`` and the truth-object slot count. MUST be >= 1.
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
        On a non-positive ``num_classes``, an unknown ``loss_weights`` key, an
        all-zero matcher-weight sum, or no requested loss component.
    """

    _KNOWN_COMPONENTS = ("object_class_ce", "mask_dice", "mask_focal", "mask_ce", "regression")

    def __init__(
        self,
        num_classes: int,
        num_objects: int,
        loss_weights: Mapping[str, float],
        matcher_weights: Mapping[str, float] | None = None,
        null_class_weight: float = 0.5,
        class_weights: list[float] | None = None,
        input_stream: str = "objects",
    ) -> None:
        super().__init__()
        self.name = _UNNAMED
        if num_classes < 1:
            raise ConfigError(
                f"MaskFormerMatchedLoss: num_classes (non-null classes) must be >= 1, got "
                f"{num_classes}"
            )
        if num_objects < 1:
            raise ConfigError(
                f"MaskFormerMatchedLoss: num_objects (query count M) must be >= 1, got "
                f"{num_objects}"
            )
        self.num_classes = num_classes
        self.num_objects = num_objects
        self.input_stream = input_stream

        self.loss_weights = {k: float(v) for k, v in dict(loss_weights).items()}
        if unknown := sorted(set(self.loss_weights) - set(self._KNOWN_COMPONENTS)):
            raise ConfigError(
                f"MaskFormerMatchedLoss: unknown loss_weights keys {unknown} — known components "
                f"are {list(self._KNOWN_COMPONENTS)} (v1 maskformer_loss.py loss_map + regression)"
            )
        mw = self.loss_weights if matcher_weights is None else dict(matcher_weights)
        self.matcher_weights = {k: float(v) for k, v in mw.items()}
        if unknown := sorted(set(self.matcher_weights) - set(self._KNOWN_COMPONENTS)):
            raise ConfigError(
                f"MaskFormerMatchedLoss: unknown matcher_weights keys {unknown} — known components "
                f"are {list(self._KNOWN_COMPONENTS)}"
            )
        # get_batch_cost reads object_class_ce unconditionally — default it to the
        # loss weight if absent.
        self.matcher_weights.setdefault(
            "object_class_ce", self.loss_weights.get("object_class_ce", 1.0)
        )
        # the matcher asserts the cost-weight sum is positive; raise a clear
        # ConfigError here instead of a bare AssertionError deeper in construction.
        if sum(self.matcher_weights.values()) == 0:
            raise ConfigError(
                "MaskFormerMatchedLoss: the matcher cost weights sum to 0 — at least one "
                f"matcher_weights entry must be positive (got {self.matcher_weights}; v1 "
                "HungarianMatcher asserts this, matcher.py:167)"
            )

        # object_class_ce is always computed; the others only when their loss
        # weight is truthy. regression is the matched extension.
        self.components: tuple[str, ...] = tuple(
            c for c in self._KNOWN_COMPONENTS if c == "object_class_ce" or self.loss_weights.get(c)
        )
        if not self.components:
            raise ConfigError(
                "MaskFormerMatchedLoss: no loss component requested — set at least "
                "object_class_ce in loss_weights"
            )

        self.null_class_weight = float(null_class_weight)
        self.class_weights = list(class_weights) if class_weights is not None else None

        # composes the HungarianMatcher + empty_weight buffer + loss_labels/loss_masks
        self.v1_loss = MaskFormerLoss(
            num_classes=num_classes,
            num_objects=num_objects,
            loss_weights=self.loss_weights,
            matcher_weights=self.matcher_weights,
            null_class_weight=self.null_class_weight,
            class_weights=self.class_weights,
        )

    @property
    def matcher(self) -> HungarianMatcher:
        """The composed `HungarianMatcher` (owned by the loss)."""
        return self.v1_loss.matcher

    def _reg_pred_key(self) -> str:
        """The scaled object-regression prediction key (``preds.<stream>.regression``)."""
        return f"preds.{self.input_stream}.regression"

    def _reg_tgt_key(self) -> str:
        """The scaled object-regression target key (``targets.<stream>.regression``)."""
        return f"targets.{self.input_stream}.regression"

    def declare_io(self, mode: Mode) -> IO:
        """Declare the object preds + truth labels -> ``matched.objects.*`` + ``losses.*``.

        FIT|VAL only — empty IO in TEST/ONNX (the matched loss is mode-inactive
        there). Requires the decoder's ``<stream>.{class_logits,class_probs,masks}``
        and the truth ``labels.objects.{object_class,masks}``; additionally the
        scaled ``preds.<stream>.regression`` + ``targets.<stream>.regression`` when a
        regression component is requested.
        """
        if not (mode & Mode.TRAINING):
            return IO(requires={}, produces={})

        m = self.num_objects
        tok = sym_dim("T", self.name)
        emb = sym_dim("E", self.name)
        n_classes = self.num_classes + 1
        f = Mode.TRAINING

        requires: dict[str, TensorSpec] = {
            f"{self.input_stream}.class_logits": TensorSpec(
                shape=("B", m, n_classes), dtype="float32", modes=f
            ),
            f"{self.input_stream}.class_probs": TensorSpec(
                shape=("B", m, n_classes), dtype="float32", modes=f
            ),
            f"{self.input_stream}.masks": TensorSpec(shape=("B", m, tok), dtype="float32", modes=f),
            f"labels.{_OBJECT_STREAM}.object_class": TensorSpec(
                shape=("B", m), dtype="int64", kind="label", modes=f
            ),
            f"labels.{_OBJECT_STREAM}.masks": TensorSpec(
                shape=("B", m, tok), dtype="bool", kind="label", modes=f
            ),
        }
        if "regression" in self.components:
            r = sym_dim("R", self.name)
            requires[self._reg_pred_key()] = TensorSpec(shape=("B", m, r), dtype="float32", modes=f)
            requires[self._reg_tgt_key()] = TensorSpec(shape=("B", m, r), dtype="float32", modes=f)

        produces: dict[str, TensorSpec] = {
            f"matched.{_OBJECT_STREAM}.embed": TensorSpec(
                shape=("B", m, emb), dtype="float32", modes=f
            ),
            f"matched.{_OBJECT_STREAM}.class_logits": TensorSpec(
                shape=("B", m, n_classes), dtype="float32", modes=f
            ),
            f"matched.{_OBJECT_STREAM}.class_probs": TensorSpec(
                shape=("B", m, n_classes), dtype="float32", modes=f
            ),
            f"matched.{_OBJECT_STREAM}.masks": TensorSpec(
                shape=("B", m, tok), dtype="float32", modes=f
            ),
            f"matched.{_OBJECT_STREAM}.object_class": TensorSpec(
                shape=("B", m), dtype="int64", kind="label", modes=f
            ),
            f"matched.{_OBJECT_STREAM}.target_masks": TensorSpec(
                shape=("B", m, tok), dtype="bool", kind="label", modes=f
            ),
        }
        requires[f"{self.input_stream}.embed"] = TensorSpec(
            shape=("B", m, emb), dtype="float32", modes=f
        )
        if "regression" in self.components:
            r = sym_dim("R", self.name)
            produces[f"matched.{_OBJECT_STREAM}.regression"] = TensorSpec(
                shape=("B", m, r), dtype="float32", modes=f
            )
            produces[f"matched.{_OBJECT_STREAM}.target_regression"] = TensorSpec(
                shape=("B", m, r), dtype="float32", modes=f
            )
        for component in self.components:
            produces[f"losses.{component}"] = TensorSpec(shape=(), kind="loss", modes=f)
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def bind(self, schema: ResolvedSchema) -> None:
        """Validate the regression prediction/target widths agree (element-wise L1).

        ``bind_all`` runs ``bind`` on every configured module regardless of the
        compiled mode, but this loss is FIT|VAL-only, so in a TEST/ONNX-only bind
        schema the regression pred/target keys are absent and their widths never
        resolve statically — skip validation when either key is missing from
        ``schema.widths``. A genuinely missing regression producer in a training
        plan still fails earlier as a planner `ConnectivityError`, so this guard
        cannot mask a real training-mode wiring bug.

        Raises
        ------
        ConfigError
            If a requested regression component has mismatched pred/target widths.
        """
        if "regression" not in self.components:
            return
        reg_pred_key, reg_tgt_key = self._reg_pred_key(), self._reg_tgt_key()
        if reg_pred_key not in schema.widths or reg_tgt_key not in schema.widths:
            return
        wp = schema.width(reg_pred_key)
        wt = schema.width(reg_tgt_key)
        if wp != wt:
            raise ConfigError(
                f"MaskFormerMatchedLoss {self.name!r}: regression prediction width {wp} != "
                f"target width {wt} ({reg_pred_key!r} vs {reg_tgt_key!r}) — "
                "the matched L1 is element-wise (v1 maskformer_loss.py regression cost)"
            )

    def _matched_regression_loss(
        self, reg_pred: Tensor, reg_tgt: Tensor, object_class: Tensor
    ) -> Tensor:
        """The matched object-regression L1 over valid (non-null) objects.

        The matcher-permuted regression predictions are aligned to the truth-order
        targets, and the L1 is averaged over the valid objects only
        (``object_class != num_classes``).

        Returns
        -------
        Tensor
            A scalar matched-regression L1 (0.0 if no valid object in the batch).
        """
        valid = object_class != self.num_classes  # [B, M]
        if not valid.any():
            return reg_pred.new_zeros(())
        # element-wise L1 over the valid objects' targets; the matcher-permuted preds
        # are already truth-aligned. mean over (valid objects x R), v1-style reduction.
        return torch.nn.functional.l1_loss(reg_pred[valid], reg_tgt[valid])

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Match queries to truth, then emit the matched predictions + the loss components.

        Solves the assignment on the scaled preds/targets, gathers the permuted
        predictions into new ``matched.*`` keys (no in-place permute), and computes
        the classification/mask losses via the composed v1 methods plus the matched
        regression L1.

        Returns
        -------
        dict[str, Tensor]
            The new ``matched.objects.*`` + ``losses.*`` keys only.
        """
        del mode
        class_logits = b.get(f"{self.input_stream}.class_logits")
        class_probs = b.get(f"{self.input_stream}.class_probs")
        masks = b.get(f"{self.input_stream}.masks")
        embed = b.get(f"{self.input_stream}.embed")
        object_class = b.get(f"labels.{_OBJECT_STREAM}.object_class")
        target_masks = b.get(f"labels.{_OBJECT_STREAM}.masks")

        # matcher cost stays in scaled space
        pred_for_match: dict[str, Tensor] = {
            "class_logits": class_logits,
            "class_probs": class_probs,
            "masks": masks,
        }
        tgt_for_match: dict[str, Tensor] = {
            "object_class": object_class,
            "masks": target_masks.to(masks.dtype),
        }
        reg_pred = reg_tgt = None
        if "regression" in self.components:
            reg_pred = b.get(self._reg_pred_key())
            reg_tgt = b.get(self._reg_tgt_key())
            pred_for_match["regression"] = reg_pred
            tgt_for_match["regression"] = reg_tgt

        idx = self.matcher(pred_for_match, tgt_for_match)

        # advanced indexing returns a fresh tensor, so matched.* never aliases the
        # decoder output (no in-place permute).
        m_class_logits = class_logits[idx]
        m_class_probs = class_probs[idx]
        m_masks = masks[idx]
        m_embed = embed[idx]

        out: dict[str, Tensor] = {
            f"matched.{_OBJECT_STREAM}.embed": m_embed,
            f"matched.{_OBJECT_STREAM}.class_logits": m_class_logits,
            f"matched.{_OBJECT_STREAM}.class_probs": m_class_probs,
            f"matched.{_OBJECT_STREAM}.masks": m_masks,
            f"matched.{_OBJECT_STREAM}.object_class": object_class,
            f"matched.{_OBJECT_STREAM}.target_masks": target_masks,
        }

        permuted_preds = {"objects": {"class_logits": m_class_logits, "masks": m_masks}}
        truth_labels = {"objects": {"object_class": object_class, "masks": target_masks}}
        losses: dict[str, Tensor] = {}
        losses.update(self.v1_loss.get_loss("labels", permuted_preds, truth_labels))
        if any(self.loss_weights.get(c) for c in ("mask_dice", "mask_focal", "mask_ce")):
            losses.update(self.v1_loss.get_loss("masks", permuted_preds, truth_labels))

        for component in self.components:
            if component == "regression":
                assert reg_pred is not None
                assert reg_tgt is not None
                m_reg = reg_pred[idx]
                reg_loss = self._matched_regression_loss(m_reg, reg_tgt, object_class)
                out[f"matched.{_OBJECT_STREAM}.regression"] = m_reg
                out[f"matched.{_OBJECT_STREAM}.target_regression"] = reg_tgt
                out["losses.regression"] = self.loss_weights["regression"] * reg_loss
            else:
                out[f"losses.{component}"] = losses[component]
        return out
