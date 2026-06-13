"""Config-constructed MaskFormer matched loss + Hungarian matcher (FD 1158-1170).

M5 sub-wave C (plan 10). The v2 spelling of v1's matcher-driven MaskFormer loss
(``salt.models.maskformer_loss.MaskFormerLoss.forward`` + ``salt.models.matcher.
HungarianMatcher``). A FIT|VAL-only `GraphModule` that, given the decoder's object
predictions and the truth object labels, solves the optimal 1-to-1 assignment of
queries to truth objects and emits the four MaskFormer loss components.

M2/M5 porting policy (plan 05): this COMPOSES the verbatim v1 building blocks — a
v1 ``MaskFormerLoss`` instance OWNS the ``HungarianMatcher`` (matcher.py), the
``empty_weight`` class-balance buffer, and the three loss methods
(``loss_labels``/``loss_masks``, maskformer_loss.py:154-227) — and reproduces v1's
matching + the three v1-decidable loss components (``object_class_ce``,
``mask_dice``, ``mask_focal``) BYTE-faithfully. Full code absorption is M7.

What is DELIBERATELY DIFFERENT from v1 (FD 1158-1170, the design's single-ownership
+ no-in-place-permute rules; the alignment change is the MF1b sign-off item):

- **NO in-place permute.** v1 mutates ``preds["objects"][k] = v[idx]`` in place
  (maskformer_loss.py:338-343), corrupting the prediction dict every other module
  shares. v2 gathers the matcher-permuted predictions into NEW ``matched.objects.*``
  bundle keys (write-once, design §2.1); the decoder's ``objects.*`` are never
  touched. `MaskformerMetrics` consumes ``matched.objects.*`` (its sub-wave-C wiring).
- **The object regression loss is MATCHED, not query-order.** v1's *effective*
  regression loss is QUERY-ORDER: the object-regression task runs in
  ``SaltModel.run_tasks`` with ``labels`` (saltmodel.py:218-219) and computes its L1
  there, aligning query-i to truth-object-i WITHOUT the matcher; the matcher only
  uses ``regression`` as a *cost* term, and ``"regression"`` is NOT in
  ``MaskFormerLoss.losses`` (= ``["labels", "masks"]``), so the matched loop never
  re-computes it. v2 makes the regression loss MATCHED (the matcher-permuted
  predictions vs the truth-order targets) and OWNED here — the single-ownership
  rule. This is a USER-VISIBLE alignment change (FD 1168-1170, §12): the MF1b human
  sign-off characterises it; the code gate (MF1a/this module) asserts the three
  mask/class components BYTE vs v1 and the matched-regression as a DESIGN-conformance
  property (no v1 byte reference exists for a matched object-regression loss).
- **NO double task execution.** v1 runs the object task twice (saltmodel.py:218-219
  for the loss + maskformer_loss.py:330 for the matcher cost). v2 runs the
  regression task ONCE; its scaled predictions/targets arrive here as the declared
  ``preds.objects.regression`` / ``targets.objects.regression`` keys.
- **NO ``aux_loss`` deep supervision** (maskformer_loss.py:306-322). PARKED (FD §10;
  shipped MaskFormer.yaml:39 sets ``aux_loss: false``) — re-deferred, not implemented.

Matching costs stay in SCALED space (v1 behaviour, FD 1166): the matcher reads the
scaled ``regression`` predictions/targets exactly as v1's
``get_batch_cost`` does (matcher.py:236-239).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.core.nn.bind import ResolvedSchema

# composed v1 loss + matcher (M2 porting policy, plan 05 — absorbed at M7)
from salt.models.maskformer_loss import MaskFormerLoss as V1MaskFormerLoss
from salt.models.matcher import HungarianMatcher as V1HungarianMatcher

__all__ = ["MaskFormerMatchedLoss"]

_UNNAMED = "unnamed"
"""Placeholder instance name — the config dict key is assigned before compile (design §2.2)."""

# the v2 bundle stream name for the reconstructed objects (matches the MaskDecoder
# out_stream and the MaskFormerTargets object stream — FD uses objects.* / labels.objects.*)
_OBJECT_STREAM = "objects"


class MaskFormerMatchedLoss(nn.Module):
    """Hungarian-matched MaskFormer loss over the decoder's object predictions (FD 1158-1170).

    A FIT|VAL-only `GraphModule`. It composes a v1 ``MaskFormerLoss`` (which holds
    the ``HungarianMatcher`` + the ``empty_weight`` buffer + the v1 ``loss_labels``/
    ``loss_masks`` methods), runs the matcher on the SCALED object predictions/targets
    (v1 ``get_batch_cost``, matcher.py:171-248), then:

    - publishes the matcher-permuted predictions + the truth labels as NEW
      ``matched.objects.*`` keys (no in-place permute — design §2.1); and
    - emits ``losses.{object_class_ce, mask_dice, mask_focal, regression}`` (only the
      components with a positive ``loss_weights`` entry are produced, v1
      maskformer_loss.py:221-226).

    The three mask/class components are computed by the composed v1 methods on the
    permuted predictions (byte-faithful v1). The regression component is the MATCHED
    L1 over valid (non-null) objects (the FD alignment change; no v1 byte reference).

    Lifecycle (design §2.3): ``__init__`` builds the composed v1 loss (the matcher +
    buffer are width-free), `declare_io` is static (FIT|VAL only), `bind` validates
    the regression prediction/target widths agree when a regression component is
    requested. forward runs the matcher + the loss methods and returns ONLY the new
    ``matched.objects.*`` + ``losses.*`` keys (write-once).

    Parameters
    ----------
    num_classes : int
        The number of NON-null object classes (v1 ``loss_config.num_classes``,
        MaskFormer.yaml:58). The null/no-object class index is ``num_classes`` (the
        sentinel the matcher + target masks use, matcher.py:209). MUST equal the
        decoder's ``class_net.output_size - 1``.
    num_objects : int
        The number of object queries ``M`` (v1 ``num_objects``, passed from the
        ``MaskDecoder`` to the loss, maskformer.py:74; MaskFormer.yaml:36). The
        matcher cost matrix is ``[B, M, M]`` (matcher.py:218), so ``M`` must equal
        the decoder's ``num_objects`` AND the truth-object slot count. MUST be >= 1.
    loss_weights : Mapping[str, float]
        Per-component LOSS weights (v1 ``loss_config.loss_weights``,
        MaskFormer.yaml:59-63): keys among ``object_class_ce``, ``mask_dice``,
        ``mask_focal``, ``mask_ce``, ``regression``. A component is produced only
        when its weight is present and truthy (v1 maskformer_loss.py:221-226). The
        ``object_class_ce`` weight is always applied (v1 always computes labels).
    matcher_weights : Mapping[str, float] | None, optional
        Per-component MATCHER cost weights (v1 ``matcher_weights``), defaulting to
        ``loss_weights`` (v1 maskformer_loss.py:144-145).
    null_class_weight : float, optional
        The class-balance weight on the null category in the CE
        (v1 ``null_class_weight``, maskformer_loss.py:130,135-142), by default 0.5.
    input_stream : str, optional
        The decoder's object stream name, by default ``objects`` — the
        ``<input_stream>.{class_logits,class_probs,masks}`` keys it reads.

    Raises
    ------
    ConfigError
        On a non-positive ``num_classes``, an unknown ``loss_weights`` key, an
        all-zero matcher-weight sum (the matcher asserts it positive,
        matcher.py:167), or no requested loss component.
    """

    # the loss components this module knows how to emit (v1 maskformer_loss.py loss_map
    # + the matched-regression extension); mask_ce is v1-supported but NOT in the shipped
    # MaskFormer.yaml weights, so it is emitted only if weighted.
    _KNOWN_COMPONENTS = ("object_class_ce", "mask_dice", "mask_focal", "mask_ce", "regression")

    def __init__(
        self,
        num_classes: int,
        num_objects: int,
        loss_weights: Mapping[str, float],
        matcher_weights: Mapping[str, float] | None = None,
        null_class_weight: float = 0.5,
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
        # v1 requires object_class_ce in the matcher cost (get_batch_cost reads it
        # unconditionally, matcher.py:222) — default it to the loss weight if absent.
        self.matcher_weights.setdefault(
            "object_class_ce", self.loss_weights.get("object_class_ce", 1.0)
        )
        # the composed v1 matcher asserts the cost-weight sum is positive
        # (matcher.py:167, a bare AssertionError); promote it to the loud config
        # surface so a zero-cost config fails as a ConfigError at construction.
        if sum(self.matcher_weights.values()) == 0:
            raise ConfigError(
                "MaskFormerMatchedLoss: the matcher cost weights sum to 0 — at least one "
                f"matcher_weights entry must be positive (got {self.matcher_weights}; v1 "
                "HungarianMatcher asserts this, matcher.py:167)"
            )

        # the produced loss components: object_class_ce is always computed (v1 always
        # runs loss_labels), the others only when their loss weight is truthy
        # (v1 maskformer_loss.py:221-226). regression is the matched extension.
        self.components: tuple[str, ...] = tuple(
            c for c in self._KNOWN_COMPONENTS if c == "object_class_ce" or self.loss_weights.get(c)
        )
        if not self.components:
            raise ConfigError(
                "MaskFormerMatchedLoss: no loss component requested — set at least "
                "object_class_ce in loss_weights"
            )

        self.null_class_weight = float(null_class_weight)

        # compose a v1 MaskFormerLoss: it owns the HungarianMatcher (matcher.py),
        # the empty_weight class-balance buffer, and the three loss methods. The
        # matcher asserts sum(matcher_weights) != 0 (matcher.py:167); v1 passes the
        # FULL loss_weights to MaskFormerLoss and forwards matcher_weights to the
        # matcher (maskformer_loss.py:148-152). ``losses=["labels", "masks"]`` is the
        # v1 default; we drive loss_labels/loss_masks directly so the list is unused.
        self.v1_loss = V1MaskFormerLoss(
            num_classes=num_classes,
            num_objects=num_objects,  # the matcher cost matrix is [B, M, M] (matcher.py:218)
            loss_weights=self.loss_weights,
            matcher_weights=self.matcher_weights,
            null_class_weight=self.null_class_weight,
        )

    @property
    def matcher(self) -> V1HungarianMatcher:
        """The composed v1 `HungarianMatcher` (owned by the v1 loss).

        Returns
        -------
        HungarianMatcher
            The matcher instance the forward drives.
        """
        return self.v1_loss.matcher

    def _reg_pred_key(self) -> str:
        """The scaled object-regression PREDICTION key (``preds.<stream>.regression``).

        Returns
        -------
        str
            The bundle key the object-regression task publishes.
        """
        return f"preds.{self.input_stream}.regression"

    def _reg_tgt_key(self) -> str:
        """The scaled object-regression TARGET key (``targets.<stream>.regression``).

        Returns
        -------
        str
            The bundle key the object-regression task publishes in FIT|VAL.
        """
        return f"targets.{self.input_stream}.regression"

    def declare_io(self, mode: Mode) -> IO:
        """Declare the object preds + truth labels -> ``matched.objects.*`` + ``losses.*``.

        FIT|VAL only (the matched loss is pruned from TEST/ONNX, FD 1159). Requires
        the decoder's ``<stream>.{class_logits,class_probs,masks}`` and the truth
        ``labels.objects.{object_class,masks}``; additionally the scaled
        ``preds.<stream>.regression`` + ``targets.<stream>.regression`` when a
        regression component is requested. Produces the matcher-permuted predictions
        + the truth labels as NEW ``matched.objects.*`` keys (consumed by
        `MaskformerMetrics`) and one scalar ``losses.<component>`` per requested
        component.

        Returns
        -------
        IO
            Empty in TEST/ONNX (the module is mode-inactive there).
        """
        if not (mode & Mode.TRAINING):
            return IO(requires={}, produces={})

        m = self.num_objects  # the query count M (concrete; matcher cost is [B, M, M])
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
            # the matcher-permuted predictions + the truth labels, for MaskformerMetrics
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
        # the decoder's embed feeds the permuted matched.embed (MaskformerMetrics reads
        # the matched query embeddings); only required when the embed is available.
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

        The query count ``M`` is config-known (``num_objects``, baked into the
        composed matcher at ``__init__``), so nothing structural binds here. When a
        regression component is requested, the scaled prediction width
        (``preds.<stream>.regression``) and target width
        (``targets.<stream>.regression``) must match — the matched L1 is element-wise
        (v1 ``batch_mae_loss`` cost, matcher.py:131).

        Raises
        ------
        ConfigError
            If a requested regression component has mismatched pred/target widths.
        """
        if "regression" in self.components:
            wp = schema.width(self._reg_pred_key())
            wt = schema.width(self._reg_tgt_key())
            if wp != wt:
                raise ConfigError(
                    f"MaskFormerMatchedLoss {self.name!r}: regression prediction width {wp} != "
                    f"target width {wt} ({self._reg_pred_key()!r} vs {self._reg_tgt_key()!r}) — "
                    "the matched L1 is element-wise (v1 maskformer_loss.py regression cost)"
                )

    def _matched_regression_loss(
        self, reg_pred: Tensor, reg_tgt: Tensor, object_class: Tensor
    ) -> Tensor:
        """The MATCHED object-regression L1 over valid (non-null) objects (FD 1168-1170).

        The FD alignment change: the matcher-permuted regression predictions are
        aligned to the truth-order targets, and the L1 is averaged over the VALID
        objects only (``object_class != num_classes``) — the same validity mask v1
        uses for the mask losses (maskformer_loss.py:215). No v1 byte reference: v1's
        regression loss is query-order (see module docstring); this is the design's
        matched alignment, gated as a design-conformance property.

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

        Reproduces v1's matched loop (maskformer_loss.py:334-347) WITHOUT the
        in-place permute: solve the assignment (matcher reads the SCALED preds/targets,
        v1 get_batch_cost), gather the permuted predictions into NEW ``matched.*`` keys,
        and compute the three v1-decidable losses (``object_class_ce``, ``mask_dice``,
        ``mask_focal``) via the composed v1 methods on the permuted predictions plus
        the MATCHED regression L1.

        Returns
        -------
        dict[str, Tensor]
            The new ``matched.objects.*`` + ``losses.*`` keys only (design §2.5).
        """
        del mode
        class_logits = b.get(f"{self.input_stream}.class_logits")
        class_probs = b.get(f"{self.input_stream}.class_probs")
        masks = b.get(f"{self.input_stream}.masks")
        embed = b.get(f"{self.input_stream}.embed")
        object_class = b.get(f"labels.{_OBJECT_STREAM}.object_class")
        target_masks = b.get(f"labels.{_OBJECT_STREAM}.masks")

        # the matcher cost dict (v1 keys: class_logits/class_probs/masks [+regression],
        # matcher.py:181-190). Costs stay in SCALED space (FD 1166).
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

        # solve the optimal assignment: idx = (batch_arange[B,1], tgt_idx[B,M]) — the
        # advanced-indexing tuple v1 applies as v[idx] (matcher.py:293-296).
        idx = self.matcher(pred_for_match, tgt_for_match)

        # permute the predictions into NEW tensors (NO in-place; v1 does
        # preds["objects"][k] = v[idx] in place, maskformer_loss.py:338-343). Advanced
        # indexing returns a fresh tensor, so matched.* never aliases the decoder output.
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

        # the three v1-decidable losses, on the PERMUTED predictions vs truth-order
        # labels, via the composed v1 methods (loss_labels/loss_masks already apply
        # the loss weights via weight_loss, maskformer_loss.py:252-254 -> 256-271).
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
