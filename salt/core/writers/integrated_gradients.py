"""The integrated-gradients attribution writer — eval-only (design §9.5 1719 / §10 1778).

M6 sub-wave E (plan 12; user-decided IN 2026-06-15). The v2 spelling of v1's
`IntegratedGradientWriter` (``salt/callbacks/integrated_gradients_writer.py``),
re-authored against the M4.5 unified-writer interface (``writers/base.py``;
sibling of `InputCopyWriter` / `PadMaskWriter` / `MaskFormerObjectWriter`) so
it produces integrated-gradient attribution columns in the eval H5 through the
ONE writer manifest — exactly the framework the M4.5 amendment built for "I
want a new column" custom writers (design §8).

EVAL-ONLY by construction (amendment §3, merge condition 3): the inherited
`Writer.onnx_outputs` returns ``[]`` — IG attributions are a TEST-time
diagnostic with no Athena consumer, the same eval-only direction as
`InputCopyWriter` / `PadMaskWriter`. A non-empty TEST `requires` keeps the
writer a legal eval-only writer (`WriterCallback._validate_writer_roles`: a
writer with a TEST role and no ONNX role is the default-legal shape).

What v1 did, and what v2 reproduces
-----------------------------------
v1 (``integrated_gradients_writer.py``) was a Lightning `Callback` that, on
``on_test_start``, wrapped the model in the external ``salt_attribution``
`SaltModelCaptumWrapper`, ran ``captum.attr.IntegratedGradients`` over the test
loader, and wrote a side-car ``{ckpt}__attributions_{sample}.h5`` with the
feature attributions, baselines and convergence deltas. The numerical core was
captum's Riemann-sum estimate of the path integral

    IG_i(x) = (x_i - b_i) * mean_{k=1..n_steps} dF/dx_i |_{x = b + (k/n) (x - b)}

(the *completeness*-satisfying integrated-gradient, Sundararajan et al. 2017).

v2 keeps that EXACT estimator but folds it into the writer's per-batch
``write`` on the unified framework, so the attributions land as columns IN the
eval H5 (one ``{run_name}_IG_{feature}`` f4 column per input feature) rather
than a separate captum-only side-car file — and with NO hard dependency on
captum / salt-attribution (neither is in the container; v1 raised `ImportError`
without them). The path integral is computed directly against a differentiable
forward closure (`forward_fn`) over a zero baseline (the v1 default baseline
shape; salt-attribution's entropy-baseline selection is an optional refinement
the writer does not require). The feature column NAMES come from the bind-time
`ResolvedSchema` feature order (``WriteCtx.feature_fields``) — resolved BY NAME,
never index arithmetic (the M3-review ergonomics fix the writer base documents).

The differentiable forward
--------------------------
A `Writer.write` receives the already-executed (no-grad) TEST `bundle`, which is
insufficient for a path integral — IG needs gradients of a scalar output w.r.t.
the input along the baseline->input path. The writer therefore takes a
``forward_fn`` callable bound at construction: ``forward_fn(x) -> [B]`` maps a
``inputs.<stream>``-shaped tensor to ONE scalar per jet (the attributed output,
e.g. the b-jet probability after softmax — v1's ``add_softmax`` + ``output_keys``
selection). The wiring supplies a closure over the live model
(``SaltModule.forward``); the IG1 gate / test fixture supplies a small
``nn.Module``. This keeps the writer self-contained, on the Writer interface,
and exercisable in-container without the external attribution stack.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import TensorSpec
from salt.core.outputs.writer_base import WriteCtx, Writer, WriterDeclareCtx

__all__ = ["IntegratedGradientWriter", "integrated_gradients"]

# v1 default suffix stem (the eval column tag). v1 wrote a side-car
# "..._attributions_{sample}.h5"; v2 writes per-feature columns tagged "IG".
_IG_SUFFIX = "IG"


def integrated_gradients(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    *,
    n_steps: int = 50,
    internal_batch_size: int | None = None,
) -> torch.Tensor:
    """The integrated-gradients path integral (Sundararajan et al. 2017; v1's captum core).

    Reproduces captum's ``IntegratedGradients`` Riemann estimate (the numerical
    core v1's `IntegratedGradientWriter` delegated to ``captum.attr``): for a
    scalar-valued ``forward_fn`` and a straight-line path from `baseline` to
    `inputs`,

        IG = (inputs - baseline) * mean_k dF/dx |_{x = baseline + (k/n)(inputs - baseline)}

    over ``n_steps`` interpolation points (``k = 1 .. n_steps``, the right
    Riemann rule captum uses). The result has the same shape as `inputs` and
    satisfies completeness (``IG.sum() ≈ F(inputs) - F(baseline)``) to the
    estimator's discretisation error.

    Parameters
    ----------
    forward_fn : Callable[[torch.Tensor], torch.Tensor]
        Differentiable map from an ``inputs``-shaped tensor to a ``[B]`` scalar
        per jet (the attributed output — v1's softmaxed ``output_keys``
        selection).
    inputs : torch.Tensor
        The input tensor to attribute, shape ``[B, ...]`` (``[B, F]`` for a
        vector stream, ``[B, T, F]`` for a sequence stream).
    baseline : torch.Tensor
        The reference input, broadcastable to `inputs` (v1's baseline; a zero
        tensor is the default reference).
    n_steps : int, optional
        Number of Riemann interpolation steps, by default 50 (v1's
        ``n_steps`` default).
    internal_batch_size : int | None, optional
        Maximum number of (jet x step) rows evaluated per forward — caps memory
        for large ``n_steps`` (v1's ``internal_batch_size``); ``None`` (default)
        evaluates all steps at once.

    Returns
    -------
    torch.Tensor
        The per-feature attribution, same shape and device as `inputs`.

    Raises
    ------
    ConfigError
        For ``n_steps < 1``, or a ``forward_fn`` that does not return one scalar
        per jet (shape ``[B]``).
    """
    if n_steps < 1:
        raise ConfigError(
            f"integrated_gradients: n_steps must be >= 1, got {n_steps} (v1 captum n_steps)"
        )
    inputs = inputs.detach()
    baseline = baseline.detach().to(inputs.dtype).expand_as(inputs)
    batch = inputs.shape[0]
    delta = inputs - baseline  # [B, ...]

    # right-Riemann interpolation coefficients k/n for k = 1..n_steps (captum's rule)
    alphas = torch.arange(1, n_steps + 1, device=inputs.device, dtype=inputs.dtype) / n_steps
    chunk = n_steps if internal_batch_size is None else max(1, internal_batch_size // batch)

    grad_sum = torch.zeros_like(inputs)
    for lo in range(0, n_steps, chunk):
        a = alphas[lo : lo + chunk]  # [S]
        # scaled inputs along the path: [S, B, ...] -> flatten step+jet to one batch
        scaled = baseline.unsqueeze(0) + a.view(-1, *([1] * inputs.ndim)) * delta.unsqueeze(0)
        flat = scaled.reshape(-1, *inputs.shape[1:]).requires_grad_(True)
        out = forward_fn(flat)  # [S*B]
        if out.ndim != 1 or out.shape[0] != flat.shape[0]:
            raise ConfigError(
                "IntegratedGradientWriter forward_fn must return one scalar per jet "
                f"(shape [{flat.shape[0]}]), got shape {tuple(out.shape)} — select a single "
                "output (v1's add_softmax + output_keys reduced the model to one scalar)"
            )
        total = out.sum()
        # a forward that does not depend on the input produces an output with no
        # grad_fn -> zero attribution (correct IG: an input with no influence
        # contributes nothing; completeness still holds since F is then constant).
        # allow_unused additionally tolerates a partial (some-features-unused)
        # dependency, returning a None grad for the unused span.
        if not total.requires_grad:
            continue
        (grads,) = torch.autograd.grad(total, flat, allow_unused=True)
        if grads is None:
            continue
        grad_sum += grads.reshape(a.shape[0], batch, *inputs.shape[1:]).sum(dim=0)

    return delta * (grad_sum / n_steps)


class IntegratedGradientWriter(Writer):
    """Integrated-gradient feature attributions as eval columns (design §9.5/§10; M4.5).

    The v2 unified-framework port of v1's `IntegratedGradientWriter`
    (``integrated_gradients_writer.py``). EVAL-ONLY: the inherited
    `Writer.onnx_outputs` returns ``[]`` (no ONNX role — attributions are a
    TEST diagnostic with no Athena consumer, the `InputCopyWriter` /
    `PadMaskWriter` direction). Per test batch it computes the
    integrated-gradients attribution of one scalar model output w.r.t. an input
    feature tensor and writes one ``{run_name}_IG_{feature}`` f4 column per input
    feature on that stream (the v1 per-feature attribution, now IN the eval H5
    rather than a captum-only side-car file).

    Integrating with `WriterCallback`: the writer participates exactly like the
    shipped writers — `requires` declares its consumed bundle key (the attributed
    ``inputs.<stream>``), so it joins the TEST demand without colliding with the
    `TaskWriter` (which consumes ``preds.*``), `InputCopyWriter` (``meta.rows``)
    or `PadMaskWriter` (``masks.*``); `columns` declares NEW ``IG_*`` columns on
    the stream (a name collision with another writer's column on the same stream
    is caught by `WriterCallback._merge_columns`).

    Parameters
    ----------
    forward_fn : Callable[[torch.Tensor], torch.Tensor]
        Differentiable map from an ``inputs.<stream>``-shaped tensor to a ``[B]``
        scalar per jet (the attributed output — v1's softmaxed ``output_keys``
        selection over the model). The production wiring binds a closure over
        ``SaltModule.forward``; the IG1 gate / fixture binds a small
        ``nn.Module``. Required: without it the writer has nothing to
        differentiate.
    stream : str, optional
        The input stream to attribute, by default ``"jets"`` (the v1 global
        object; a sequence stream like ``"tracks"`` is equally valid — the
        attribution carries the stream's token axis). The consumed bundle key
        is ``inputs.<stream>`` and the columns land on that stream's H5 group.
    n_steps : int, optional
        Riemann steps for the path integral, by default 50 (v1 default).
    internal_batch_size : int | None, optional
        Max (jet x step) rows per forward (caps memory for large ``n_steps``);
        v1's ``internal_batch_size``, by default None (all steps at once).
    suffix : str, optional
        The column-name stem, by default ``"IG"`` — columns are
        ``{run_name}_{suffix}_{feature}``.

    Raises
    ------
    ConfigError
        On a missing ``forward_fn`` (nothing to differentiate) or ``n_steps <
        1``.
    """

    def __init__(
        self,
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
        stream: str = "jets",
        n_steps: int = 50,
        internal_batch_size: int | None = None,
        suffix: str = _IG_SUFFIX,
        feature_names: list[str] | None = None,
    ) -> None:
        if forward_fn is None or not callable(forward_fn):
            raise ConfigError(
                "IntegratedGradientWriter: forward_fn is required — it is the differentiable "
                "scalar output the integrated gradients attribute (v1 wrapped the model in "
                "salt_attribution.SaltModelCaptumWrapper; the v2 writer takes the forward "
                "closure directly)"
            )
        if n_steps < 1:
            raise ConfigError(
                f"IntegratedGradientWriter: n_steps must be >= 1, got {n_steps} (v1 n_steps)"
            )
        self.forward_fn = forward_fn
        self.stream = str(stream)
        self.n_steps = int(n_steps)
        self.internal_batch_size = internal_batch_size
        self.suffix = str(suffix)
        # explicit feature-name override (a fixture / config without a resolved
        # ResolvedSchema feature order); else resolved from WriteCtx.feature_fields
        self.feature_names = list(feature_names) if feature_names is not None else None

    def _input_key(self) -> str:
        """The attributed input bundle key (``inputs.<stream>``).

        Returns
        -------
        str
            The model input the integrated gradients attribute.
        """
        return f"inputs.{self.stream}"

    def requires(self, ctx: WriterDeclareCtx) -> dict[str, TensorSpec]:
        """Declare the consumed input feature tensor (TEST demand).

        The attributed ``inputs.<stream>`` — a model-side leaf, so it anchors
        the TEST plan and keeps the input alive in the bundle the writer reads
        (design §8). Unconstrained spec: the input shape/dtype unify against the
        reader at bind. NON-EMPTY by design, so the writer is a legal eval-only
        writer (`WriterCallback._validate_writer_roles`).

        Returns
        -------
        dict[str, TensorSpec]
            The single ``inputs.<stream>`` requirement.
        """
        del ctx
        return {self._input_key(): TensorSpec(shape=None, dtype=None)}

    def _feature_names(self, ctx: WriteCtx) -> list[str] | None:
        """The attributed input's per-feature column stems (resolved BY NAME).

        Priority: an explicit ``feature_names`` ctor override (a fixture /
        config without a resolved schema), then the bind-time `ResolvedSchema`
        feature order (``WriteCtx.feature_fields["inputs.<stream>"]`` — the
        configured `Features` variable order) so columns are named after the
        real input variables (``..._IG_d0``), never index arithmetic (the
        M3-review ergonomics contract the writer base documents).

        Returns
        -------
        list[str] | None
            One stem per input feature, in feature order; ``None`` when no
            schema feature order is available (the caller falls back to
            positional ``f{i}`` stems against the live feature count).
        """
        if self.feature_names is not None:
            return list(self.feature_names)
        fields = ctx.feature_fields.get(self._input_key())
        if fields:
            return list(fields)
        return None

    @staticmethod
    def _declared_feature_count(ctx: WriteCtx) -> int | None:
        """The input's declared last-dim feature count, if the schema exposes one.

        The shipped `WriteCtx` carries feature NAMES in ``feature_fields`` (the
        primary path); a feature COUNT without names is not separately exposed,
        so this returns ``None`` and the caller raises the loud configuration
        error. The hook exists so a future `WriteCtx` carrying only counts can
        name positional columns without an interface change.

        Returns
        -------
        int | None
            Always ``None`` for the current `WriteCtx` surface.
        """
        del ctx
        return None

    def columns(self, ctx: WriteCtx) -> dict[str, np.dtype]:
        """Declare one ``{run_name}_{suffix}_{feature}`` f4 column per input feature.

        The feature names come from the explicit ``feature_names`` override or
        the bind-time schema feature order; absent both, positional ``f{i}``
        stems are derived from the input's declared last-dim feature count.
        Float columns are f2 under the half-precision sink policy, matching the
        shipped writers.

        Returns
        -------
        dict[str, np.dtype]
            ``{stream: structured dtype}`` — the IG attribution columns.

        Raises
        ------
        ConfigError
            When the input feature count cannot be resolved (no feature names
            and no declared feature count for the stream).
        """
        fmt = "f2" if ctx.precision == "half" else "f4"
        names = self._feature_names(ctx)
        if names is None:
            n_feats = self._declared_feature_count(ctx)
            if n_feats is None:
                raise ConfigError(
                    f"IntegratedGradientWriter {self.name!r}: cannot resolve the feature count "
                    f"for input {self._input_key()!r} — configure the stream's Features (so the "
                    "bind-time schema carries a feature order) or pass feature_names. The IG "
                    "columns are one per input variable (design §4.4 annotation)."
                )
            names = [f"f{i}" for i in range(n_feats)]
        descr = [(f"{ctx.run_name}_{self.suffix}_{name}", fmt) for name in names]
        if len({field for field, _ in descr}) != len(descr):
            raise ConfigError(
                f"IntegratedGradientWriter {self.name!r}: duplicate IG column names on stream "
                f"{self.stream!r}: {[f for f, _ in descr]} — two input features share a name"
            )
        return {self.stream: np.dtype(descr)}

    def column_manifest(self, ctx: WriterDeclareCtx, run_name: str) -> dict[str, list[str]]:
        """The statically-derivable eval columns (design §4.4 annotation surface).

        Returns
        -------
        dict[str, list[str]]
            ``{stream: [IG column names]}`` when the feature order is known
            statically; empty (file-dependent) otherwise.
        """
        del ctx
        # the declare ctx carries no schema feature names; the binding columns
        # come from WriteCtx.feature_fields (see `columns`). Annotated as
        # file/schema-dependent here (the InputCopyWriter precedent).
        del run_name
        return {}

    def write(self, bundle: Bundle, rows: slice) -> dict[str, np.ndarray]:
        """Compute this batch's integrated-gradient attributions.

        Reads the attributed ``inputs.<stream>`` from the executed bundle,
        attributes the configured scalar output via `integrated_gradients` over
        a zero baseline (v1's default reference), reduces a per-token sequence
        attribution to a per-jet, per-feature value (mean over valid tokens,
        the v1 per-jet attribution), and renders one structured f4 column per
        input feature.

        For a sequence stream (``[B, T, F]``) the attribution is masked-mean
        pooled over the stream's valid tokens (``masks.<stream>`` when present)
        so the eval column is a per-jet, per-feature attribution (v1 wrote
        per-jet rows); a vector stream (``[B, F]``) is rendered directly.

        Returns
        -------
        dict[str, np.ndarray]
            ``{stream: structured array}`` of per-jet IG attributions.

        Raises
        ------
        ConfigError
            For an unsupported input rank (not ``[B, F]`` / ``[B, T, F]``), a
            ``forward_fn`` returning a non-scalar output (via
            `integrated_gradients`), or a feature-count mismatch between the
            declared IG columns and the live input.
        """
        del rows
        ctx = self.ctx
        x = bundle.get(self._input_key())  # [B, F] or [B, T, F]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        baseline = torch.zeros_like(x)

        with torch.enable_grad():
            attr = integrated_gradients(
                self.forward_fn,
                x,
                baseline,
                n_steps=self.n_steps,
                internal_batch_size=self.internal_batch_size,
            )

        if attr.ndim == 3:  # [B, T, F] sequence stream -> masked-mean over tokens
            pad_key = f"masks.{self.stream}"
            if pad_key in bundle:
                pad = bundle.get(pad_key).to(attr.device)  # True = padded
                valid = (~pad).unsqueeze(-1).to(attr.dtype)  # [B, T, 1]
                counts = valid.sum(dim=1).clamp(min=1.0)
                attr = (attr * valid).sum(dim=1) / counts  # [B, F]
            else:
                attr = attr.mean(dim=1)  # [B, F]
        elif attr.ndim != 2:
            raise ConfigError(
                f"IntegratedGradientWriter {self.name!r}: input {self._input_key()!r} has rank "
                f"{attr.ndim} ([B, F] or [B, T, F] supported), shape {tuple(attr.shape)}"
            )

        dtype = self.columns(ctx)[self.stream]
        if attr.shape[1] != len(dtype.names):
            raise ConfigError(
                f"IntegratedGradientWriter {self.name!r}: input {self._input_key()!r} has "
                f"{attr.shape[1]} features but {len(dtype.names)} IG columns were declared — "
                "the bind-time Features order and the live input feature count disagree"
            )
        return {self.stream: u2s(attr.detach().cpu().float().numpy(), dtype)}
