"""M5 gates harness — R1-R4 (sub-wave A) + L1-L3 (sub-wave B) (plan 10; design §9.5).

Standalone gates, each a subcommand of ``python -m salt.core.gates_m5``, each
writing a machine-readable ``<gate>_report.json`` into ``--outdir`` and a
human-readable table to stdout, exiting non-zero on failure (the gates_m2/m3/m4
envelope). NO machine data path lives in this file (design §5 placeholder
policy): every gate generates its own dummy fixtures into ``--outdir`` from the
in-repo parity generators (`salt.tests.core.regression_fixture` + the gn2
fixture), and compares an INDEPENDENT v1 reference against the v2 module driven
through the REAL plan compiler / two-phase bind / executor.

R1-R4 cover the sub-wave-A regression family (`RegressionTaskModule`); L1 covers
the sub-wave-B `LossGLS` geometric-mean loss combination; L2 covers the
sub-wave-B `TransformerEncoder` ``norm_type: hybrid`` passthrough (the GN3V01
flagship encoder block); L3 covers the sub-wave-B `VectorConcat` + ``export.
inputs alias:`` (the GN3 ``global`` stream past the encoder).

Gate criteria (each justified in its ``run_r*`` docstring vs the design/v1 ref):

- **R1 RegressionTask forward+denorm parity** — the A2 core surface on
  ``regression.yaml``'s variants (norm_params scalar+vector, target_denominators
  ratio + custom_output_names, functional scaler per-token sequence). Two tiers
  of assertion, kept honestly distinct:
  (i) *v2-wiring self-consistency vs the composed v1 head* — the executor's
  output equals ``module.task(...)`` (the SAME composed-v1 object the executor
  invoked), which is byte-equal by construction and proves the Split/pool/
  executor key-routing doesn't corrupt I/O (the corruption hook gives it teeth);
  (ii) *genuine v1-vs-v2 cross-impl parity* — for the norm_params variant AND the
  functional-scaler per-token sequence variant, the executor's output is compared
  against an INDEPENDENT v1 ``RegressionTask`` (separately constructed from the
  same kwargs, its own `Dense`, ``load_state_dict``-copied weights — the R3
  dips-pooling pattern) to <= 1e-6. PLUS the ONNX denominator-∈-Features rule (a
  denominator absent from the input Features is a bind-time ConfigError; the
  TEST-vs-ONNX denominator source split, FD §3.3).
- **R2 GaussianRegressionTask parity** — mu/sigma head on
  ``regression_gaussian.yaml``: output_size == 2R, Gaussian NLL FIT loss, and
  the TEST de-scaling (means ‖ sqrt(softplus(var))·std). Self-consistency vs the
  composed v1 head PLUS a genuine cross-impl check against an INDEPENDENT v1
  ``GaussianRegressionTask`` (separately built + state-copied) to <= 1e-6. PLUS
  the Gaussian ONNX representation is EXERCISED (not merely asserted): a Mode.ONNX
  plan is compiled and run, and the ONNX ``[B, 2R]`` array equals the TEST
  ``[B, 2R]`` (the de-scaling math is mode-independent under norm_params), the
  second R columns strictly positive (stddev = sqrt(softplus(var))). This is a
  *design-conformance* check (NOT v1-byte-parity): v1 ``onnx/to_onnx.py`` has
  zero Gaussian handling and v1's (means, stds) is TEST-path-only, so there is no
  v1 ONNX golden to match. Gated at R=1 (v1's only correct multi-target gaussian
  de-scaling case — see ``run_r2``).
- **R3 sample_weight + NaN masking + encoder-less pooling parity** — the
  per-sample-weighted loss (``regression_weighted.yaml``: nonuniform + all-zero
  weights, multi-target expansion) and NaN-target masking
  (``nan_regression.yaml``: NaN targets masked to 0, torch.nanmean): the loss
  sub-checks assert v2-wiring self-consistency vs the composed v1 head (same
  mask→0, weight-expand, nanmean ORDERING, v1 task.py:412-437), and the
  nonuniform-weight variant ALSO asserts genuine cross-impl parity against an
  INDEPENDENT v1 ``RegressionTask`` (separately built + state-copied) to <= 1e-6;
  PLUS the encoder-less pooling forward parity on ``legacy/dips.yaml`` — the
  encoder-less `GlobalAttentionPooling`
  (``init_nets`` + ``pool_net``, no encoder, no ``masks.registers``; v1
  saltmodel.py:90-93,155-156) reproduces a standalone v1
  `GlobalAttentionPooling` BITWISE on the same ``seq.x``/``seq.mask`` with the
  v1 encoder-less pad dict ``{"seq": seq.mask}`` (pooling.py:53-63), and the
  compiled dips plan carries NO ``masks.registers`` key.
- **R4 MultiTarget processor row-replacement parity** — the conditional
  ``np.where`` replacement on ``regression_multi_target.yaml``'s two-rule
  custom_target chain matches v1's ``apply_multi_target_replacements``
  (``torch.where`` over a labels dict, datasets.py:695-739) BITWISE, including
  the sequential per-output running array (v1 in-place mutation).
- **L1 LossGLS geometric-mean parity + all-weights==1.0 guard** (sub-wave B) —
  a two-task encoder-less plan's ``loss.total`` (from the real
  compiler/bind/executor) equals an INDEPENDENT v1 ``ModelWrapper.total_loss``
  in GLS mode (a separately-constructed v1 wrapper, fed the SAME per-task
  losses) BITWISE — the ``math.prod`` then ``pow(·, 1/n)`` reduction
  (modelwrapper.py:194-196) — and differs from the per-task sum. The
  all-weights==1.0 guard raises a ``ConfigError`` on BOTH weight surfaces (the
  module's per-loss ``weights`` in ``LossGLS.__init__``; the task-side
  ``weight: float`` via ``LossGLS.check_task_weights`` — the v2 home of v1's
  ctor assert, modelwrapper.py:139-142) and stays silent at exactly 1.0. L1
  exercises ONLY loss combination + the weight guard — no ONNX claim (LossGLS is
  TRAINING-mode-only, inactive in TEST/ONNX, inheriting `LossSum.declare_io`).
- **L2 norm_type:hybrid passthrough parity** (sub-wave B) — a GN3V01-style
  ``norm_type: hybrid`` encoder block (norm -> embed -> concat -> encoder ->
  split -> pool -> head) from the real compiler/bind/executor produces
  ``encoded.seq`` equal to an INDEPENDENT v1 ``Transformer`` (same depth/widths/
  norm_type/attn+dense kwargs, ``load_state_dict``-copied weights, fed the SAME
  ``seq.x``/``seq.mask``) BITWISE; every composed v1 ``EncoderLayer`` reports
  ``norm_type == "hybrid"`` with forced ``do_qk_norm``/``do_v_norm`` and the
  depth-0 (``"pre"`` + Identity) vs depth>0 (``"none"`` + a real norm) residual
  split (transformer.py:350-356,421); hybrid differs from pre-norm; an unknown
  ``norm_type`` raises a ``ConfigError``. L2 asserts the FIT/TEST encoder forward
  only — no ONNX claim (the hybrid flag is construction-time wiring of the v1
  encoder; the encoder's ONNX path is norm_type-agnostic, gated by M2/M4).
- **L3 VectorConcat order + Dsum + export.inputs alias** (sub-wave B) — the GN3
  ``global`` stream past the encoder (design §6.6). VectorConcat has NO v1 class
  (v1 inlined the cat at saltmodel.py:175-177), so the anchors are
  design-conformance, not v1-byte-parity. Three checks, each with teeth:
  (i) *ORDER + Dsum (FIT)* — a GN3V01-style block (norm/norm_global -> embed ->
  concat -> encoder -> split -> pool -> VectorConcat[pooled.global,
  normed.global] -> head) from the real compiler/bind/executor yields
  ``vconcat.global`` EQUAL (BITWISE, ``torch.equal``) to the literal v1 line
  ``cat([pooled.global, normed.global], dim=-1)`` (`build_independent_v1_global_concat`)
  — pooled FIRST, global features LAST (design §6.6 1390-1391); its width is
  ``Dsum = width(pooled.global) + width(normed.global)`` (resolved at bind by
  `VectorConcat.derived_widths`), and the leading columns equal ``pooled.global``
  while the trailing columns equal ``normed.global`` (an ORDER-sensitive split a
  reversed concat would fail). (ii) *alias name-resolution (Mode.ONNX, RUN)* —
  the alias is an EXPORT concern (`OnnxAdapter`), so this compiles and RUNS a
  real Mode.ONNX plan with the GN3 alias (``inputs.global`` cloned from
  ``inputs.jets``, to_onnx.py:377-378): the IDENTITY case (``global``==``jets``
  columns) gathers None (a clone) and the ONNX output equals the all-valid
  TEST/eager forward; the NAME-GATHER case (``global`` = the jets columns in
  REVERSED order) resolves an ``index_select`` that genuinely reorders, and the
  aliased ``inputs.global`` fed to the concat equals the by-NAME gather of
  ``inputs.jets``. (iii) *loud surfaces* — a duplicate-input VectorConcat and a
  self-feeding ``out in inputs`` both raise ``ConfigError``; an alias naming a
  column absent from the source raises at adapter construction. L3 DOES make an
  ONNX claim and backs it by compiling + running ``Mode.ONNX`` (never asserting a
  TEST-mode array as an ONNX proxy).

Negative-control hooks (the pytest suite, ``test_gates_m5.py``): each ``run_*``
takes a python-only ``corruption`` keyword applied to the v2 OBSERVED values (or
the loss) before the parity comparison — the gate must FAIL while the surface
checks stay green. The gates_m2/m3/m4 pattern, never exposed on the CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from salt.core.data import MultiTarget
from salt.core.graph import Bundle, Executor, Mode
from salt.core.graph.errors import ConfigError
from salt.core.nn import (
    LossGLS,
    TransformerEncoder,
    VectorConcat,
    bind_all,
    materialise_all,
    resolve_bind_schema,
)
from salt.core.nn.tasks import RegressionTaskModule
from salt.core.onnx import OnnxAdapter
from salt.data.datasets import OPERATORS as V1_OPERATORS
from salt.models.pooling import GlobalAttentionPooling as V1GlobalAttentionPooling
from salt.tests.core.gn2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    make_gn2_batch,
    write_parity_norm_dict,
)
from salt.tests.core.regression_fixture import (
    GLOBAL_VARIABLES,
    HYBRID_ENC_DIM,
    build_dips_modules,
    build_gls_modules,
    build_hybrid_encoder_modules,
    build_independent_v1_global_concat,
    build_independent_v1_gls_total_loss,
    build_independent_v1_head,
    build_independent_v1_transformer,
    build_regression_modules,
    build_vector_concat_modules,
    compile_dips,
    compile_gls,
    compile_hybrid_encoder,
    compile_regression,
    compile_vector_concat,
    compile_vector_concat_onnx,
    make_dips_labels,
    make_gls_labels,
    make_regression_labels,
    write_vector_concat_norm_dict,
)
from salt.utils.scalers import RegressionTargetScaler

__all__ = [
    "PARITY_ATOL",
    "main",
    "run_l1",
    "run_l2",
    "run_l3",
    "run_r1",
    "run_r2",
    "run_r3",
    "run_r4",
]

PARITY_ATOL = 1e-6
"""Forward/de-scaling parity ceiling — the A2 forward-equivalence bound.

The v2 task path composes a FRESH v1 head and hands it single-stream dicts; the
math is verbatim v1 but executed through the v2 Split/pool/executor, so it is
mathematically equal, not guaranteed bitwise. The R-gates therefore assert
<= 1e-6 absolute on every pred and loss (the same bound the M2 gates use,
gates_m2.G3_ATOL0). R4 (the MultiTarget processor) is a pure numpy op with no
model in the loop, so it asserts BITWISE.
"""

B, T = 6, 10


# ---------------------------------------------------------------------------
# shared report helpers (the gates_m2/m3/m4 envelope, kept standalone per harness)
# ---------------------------------------------------------------------------


def _emit_report(report: dict[str, Any], outdir: Path, gate: str) -> Path:
    """Write the gate report JSON into ``outdir`` and return its path.

    Returns
    -------
    Path
        The written ``<gate>_report.json`` path.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{gate}_report.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return path


def _print_verdict(gate: str, passed: bool, criterion: str, report_path: Path) -> None:
    """Print the closing verdict block every gate ends with."""
    print("=" * 96)
    print(f"GATE {gate.upper()}: {'PASS' if passed else 'FAIL'} — criterion: {criterion}")
    print(f"report: {report_path}")


def _base_report(gate: str, passed: bool, criterion: str, config: dict[str, Any]) -> dict[str, Any]:
    """Build the common report envelope shared with the M2/M3/M4 gates.

    Returns
    -------
    dict[str, Any]
        Envelope with gate name, timestamp, verdict, criterion, config and
        environment fields; gate-specific sections are added by the caller.
    """
    return {
        "gate": gate,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "passed": passed,
        "criterion": criterion,
        "config": config,
        "environment": {"torch": torch.__version__, "device": "cpu"},
    }


def _print_checks(checks: dict[str, bool]) -> None:
    """Print a name/PASS-FAIL table for a checks mapping."""
    for name, ok in checks.items():
        print(f"  {name:<56} {'PASS' if ok else 'FAIL'}")


# ---------------------------------------------------------------------------
# fixture helpers (the parity scaffold — dummy data, no machine paths)
# ---------------------------------------------------------------------------


def _norm_dict(outdir: Path) -> Path:
    """Write the parity norm/class dicts into ``outdir``; return the norm-dict path.

    Returns
    -------
    Path
        The norm-dict YAML path (the encoder-less DiPS fixture needs it).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    nd, cd = outdir / "norm_dict.yaml", outdir / "class_dict.yaml"
    write_parity_norm_dict(nd, cd)
    return nd


def _build_and_bind(
    norm_dict: Path,
    task: RegressionTaskModule,
    targets: tuple[str, ...],
    denominators: tuple[str, ...] = (),
    weight: str | None = None,
    modes: Sequence[Mode] = (Mode.FIT, Mode.TEST),
) -> dict[Mode, Any]:
    """Build the encoder-less regression plan(s), bind + materialise the modules.

    The module dict is bound in place; `task` (the caller's reference into it)
    is the composed-v1-head handle used for the reference comparison.

    Returns
    -------
    dict[Mode, Any]
        ``{mode: compiled plan}`` — ready for `Executor`.
    """
    modules = build_regression_modules(norm_dict, task)
    plans = {
        mode: compile_regression(modules, mode, targets, denominators, weight) for mode in modes
    }
    bind_all(modules, resolve_bind_schema(list(plans.values())))
    materialise_all(modules)
    return plans


def _fit_bundle(
    targets: tuple[str, ...],
    denominators: tuple[str, ...] = (),
    *,
    seed: int = 11,
    weight: tuple[str, torch.Tensor] | None = None,
    poison_nan: bool = False,
) -> tuple[Bundle, dict[str, torch.Tensor]]:
    """A FIT-mode bundle: GN2 inputs + regression labels (+ optional weight/NaN).

    Returns
    -------
    tuple[Bundle, dict[str, torch.Tensor]]
        ``(bundle, labels)`` — the bundle for the executor and the flat label
        dict for the v1 reference call.
    """
    inputs, masks = make_gn2_batch(B, T)
    labels = make_regression_labels(B, targets, denominators, seed=seed)
    if poison_nan:
        poisoned = labels[f"labels.jets.{targets[0]}"].clone()
        poisoned[::2] = torch.nan
        labels[f"labels.jets.{targets[0]}"] = poisoned
    b = Bundle()
    for stream, x in inputs.items():
        b.set(f"inputs.{stream}", x)
    b.set("masks.tracks", masks["tracks"])
    for key, val in labels.items():
        b.set(key, val)
    if weight is not None:
        name, values = weight
        b.set(f"labels.jets.{name}", values)
        labels[f"labels.jets.{name}"] = values
    return b, labels


def _test_bundle(denominators: tuple[str, ...] = ()) -> tuple[Bundle, dict[str, torch.Tensor]]:
    """A TEST/ONNX-mode bundle: GN2 inputs + (TEST) the ratio denominator labels.

    Returns
    -------
    tuple[Bundle, dict[str, torch.Tensor]]
        ``(bundle, labels)``.
    """
    inputs, masks = make_gn2_batch(B, T)
    labels = make_regression_labels(B, (), denominators)
    b = Bundle()
    for stream, x in inputs.items():
        b.set(f"inputs.{stream}", x)
    b.set("masks.tracks", masks["tracks"])
    for key, val in labels.items():
        b.set(key, val)
    return b, labels


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max absolute difference, NaN-robust (masked positions are NaN both sides).

    Returns
    -------
    float
        ``max |a - b|`` over the non-NaN positions (0.0 if all-NaN).
    """
    diff = (a.double() - b.double()).abs()
    finite = diff[torch.isfinite(diff)]
    return float(finite.max().item()) if finite.numel() else 0.0


# ---------------------------------------------------------------------------
# R1 — RegressionTask forward + de-scaling parity
# ---------------------------------------------------------------------------


def run_r1(
    outdir: Path | str,
    *,
    corruption: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[int, dict[str, Any]]:
    """R1: the A2 RegressionTask family — FIT/TEST/ONNX parity vs v1.

    Covers the ``regression.yaml`` variants: vector ``norm_params`` (FIT preds
    raw/scaled + loss), ratio ``target_denominators`` + ``custom_output_names``
    with the mode-split de-scaling (TEST sources the denominator from the label
    group, ONNX gathers it BY NAME from the input Feature tensor — different
    source, different de-scaled values, FD §3.3), the ONNX denominator-∈-Features
    bind-time rule, AND the functional ``scaler`` per-token SEQUENCE path
    (``input="seq.x"``, ``sequence=True``: v1 ``run_inference`` de-scales
    ``preds[:, :, i]``, task.py:594-596).

    Two assertion tiers, kept honestly distinct:

    - ``*_self`` checks — v2-wiring self-consistency: the executor output equals
      ``module.task(...)`` (the SAME composed-v1 object the executor invoked).
      Byte-equal by construction; proves Split/pool/executor key-routing doesn't
      corrupt I/O (the corruption hook gives this teeth).
    - ``*_vs_independent_v1`` checks — genuine v1-vs-v2 cross-impl parity: the
      executor output is compared against an INDEPENDENT v1 ``RegressionTask``
      (`build_independent_v1_head`: separately constructed from the same kwargs,
      its own `Dense`, ``load_state_dict``-copied weights — the R3 dips-pooling
      pattern). Asserted on the norm_params variant AND the scaler sequence
      variant, the variants whose de-scaling math the wiring most affects.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)`` — 0 only if every check passed.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("R1 RegressionTask forward + de-scaling parity vs the composed v1 head")
    print("=" * 96)
    norm_dict = _norm_dict(outdir)
    checks: dict[str, bool] = {}
    diffs: dict[str, float] = {}

    # -- (a) vector norm_params: FIT preds raw/scaled + loss parity ------------
    targets = ("R10TruthLabel_R22v1_TruthJetMass", "R10TruthLabel_R22v1_TruthJetPt")
    task = RegressionTaskModule(
        stream="jets",
        targets=list(targets),
        input="pooled.global",
        norm_params={"mean": [1.0, 2.0], "std": [3.0, 4.0]},
        weight=0.5,
    )
    plans = _build_and_bind(norm_dict, task, targets, modes=(Mode.FIT,))
    b, labels = _fit_bundle(targets)
    out = Executor(plans[Mode.FIT]).run(b, debug=True)
    v2_pred = out.get("preds.jets.regression")
    if corruption is not None:
        v2_pred = corruption(v2_pred)
    v2_loss = out.get("losses.regression")
    pooled = out.get("pooled.global")
    tdict = {"jets": {t: labels[f"labels.jets.{t}"] for t in targets}}
    # tier 1: self-consistency vs the composed v1 head (the executor's own object)
    ref_pred, ref_loss = task.task(pooled, tdict, None, context=None)
    diffs["normparams_fit_pred_self"] = _max_abs(v2_pred, ref_pred)
    diffs["normparams_fit_loss_self"] = _max_abs(v2_loss, ref_loss)
    checks["normparams_fit_pred_self"] = diffs["normparams_fit_pred_self"] <= PARITY_ATOL
    checks["normparams_fit_loss_self"] = diffs["normparams_fit_loss_self"] <= PARITY_ATOL
    # tier 2: genuine cross-impl parity vs an INDEPENDENT v1 RegressionTask
    indep = build_independent_v1_head(task.task)
    ind_pred, ind_loss = indep(pooled, tdict, None, context=None)
    diffs["normparams_fit_pred_vs_independent_v1"] = _max_abs(v2_pred, ind_pred)
    diffs["normparams_fit_loss_vs_independent_v1"] = _max_abs(v2_loss, ind_loss)
    checks["normparams_fit_pred_vs_independent_v1"] = (
        diffs["normparams_fit_pred_vs_independent_v1"] <= PARITY_ATOL
    )
    checks["normparams_fit_loss_vs_independent_v1"] = (
        diffs["normparams_fit_loss_vs_independent_v1"] <= PARITY_ATOL
    )

    # -- (b) ratio denominator: TEST (label source) vs ONNX (input-feature) ----
    rtargets, denoms = ("HadronConeExclTruthLabelPt",), ("pt_btagJes",)
    rtask = RegressionTaskModule(
        stream="jets",
        targets=list(rtargets),
        input="pooled.global",
        target_denominators=list(denoms),
        custom_output_names="pt",
    )
    rplans = _build_and_bind(
        norm_dict, rtask, rtargets, denoms, modes=(Mode.FIT, Mode.TEST, Mode.ONNX)
    )
    # TEST: denominator from the label group
    bt, tlabels = _test_bundle(denoms)
    with torch.no_grad():
        bt = Executor(rplans[Mode.TEST]).run(bt)
    v2_test = bt.get("preds.jets.regression")
    pooled_t = bt.get("pooled.global")
    with torch.no_grad():
        raw_t, _ = rtask.task(pooled_t, {}, None, context=None)
        ref_test = rtask.task.run_inference(
            raw_t.clone(), labels={"jets": {"pt_btagJes": tlabels["labels.jets.pt_btagJes"]}}
        )
    diffs["ratio_test_descale"] = _max_abs(v2_test, ref_test)
    checks["ratio_test_descale_from_label"] = diffs["ratio_test_descale"] <= PARITY_ATOL
    # ONNX: denominator gathered by NAME from inputs.jets
    inputs, masks = make_gn2_batch(B, T)
    bo = Bundle()
    for stream, x in inputs.items():
        bo.set(f"inputs.{stream}", x)
    bo.set("masks.tracks", masks["tracks"])
    with torch.no_grad():
        bo = Executor(rplans[Mode.ONNX]).run(bo)
    v2_onnx = bo.get("preds.jets.regression")
    col = JET_VARIABLES.index("pt_btagJes")
    denom_from_input = inputs["jets"][..., col]
    pooled_o = bo.get("pooled.global")
    with torch.no_grad():
        raw_o, _ = rtask.task(pooled_o, {}, None, context=None)
        ref_onnx = rtask.task.run_inference(
            raw_o.clone(), labels={"jets": {"pt_btagJes": denom_from_input}}
        )
    diffs["ratio_onnx_descale"] = _max_abs(v2_onnx, ref_onnx)
    checks["ratio_onnx_descale_by_name"] = diffs["ratio_onnx_descale"] <= PARITY_ATOL
    checks["test_and_onnx_sources_differ"] = not torch.allclose(v2_test, v2_onnx, atol=PARITY_ATOL)

    # -- (c) the ONNX denominator-∈-Features bind-time rule --------------------
    bad = RegressionTaskModule(
        stream="jets",
        targets="t",
        input="pooled.global",
        target_denominators="not_a_feature",
    )
    bad.name = "regression"
    raised = False
    try:
        bad_modules = build_regression_modules(norm_dict, bad)
        bad_plan = compile_regression(bad_modules, Mode.ONNX, ("t",), ("not_a_feature",))
        bad.bind(resolve_bind_schema([bad_plan]))
    except ConfigError:
        raised = True
    checks["denominator_not_a_feature_is_bind_error"] = raised

    # -- (d) functional scaler — per-token SEQUENCE de-scaling -----------------
    # the scaler path is the ONLY de-scaling v1 runs on 3D preds[:, :, i]
    # (task.py:594-596); a per-token regression head (input="seq.x",
    # sequence=True) on legacy/dips-shaped tracks exercises it end-to-end and is
    # checked against an INDEPENDENT v1 RegressionTask carrying the same scaler.
    scaler_cfg = {"HadronConeExclTruthLabelPt": {"op": "log", "x_scale": 5}}
    starget = ("HadronConeExclTruthLabelPt",)
    stask = RegressionTaskModule(
        stream="tracks",
        targets=list(starget),
        input="seq.x",
        sequence=True,
        scaler=scaler_cfg,
        loss="MSELoss",
    )
    smodules = build_regression_modules(norm_dict, stask)
    sfit = compile_regression(smodules, Mode.FIT, starget, sequence=True)
    stest = compile_regression(smodules, Mode.TEST, starget, sequence=True)
    bind_all(smodules, resolve_bind_schema([sfit, stest]))
    materialise_all(smodules)
    sinputs, smasks = make_gn2_batch(B, T)
    slabels = make_regression_labels(B, starget, seq_len=T)
    sb = Bundle()
    for stream, x in sinputs.items():
        sb.set(f"inputs.{stream}", x)
    sb.set("masks.tracks", smasks["tracks"])
    for key, val in slabels.items():
        sb.set(key, val)
    sout = Executor(sfit).run(sb, debug=True)
    v2_spred = sout.get("preds.tracks.regression")
    v2_sloss = sout.get("losses.regression")
    spooled, smask = sout.get("seq.x"), sout.get("masks.tracks")
    stdict = {"tracks": {starget[0]: slabels[f"labels.tracks.{starget[0]}"]}}
    # FIT self-consistency + cross-impl parity (scaled space, no de-scale in FIT)
    _, sref_loss = stask.task(spooled, stdict, {"tracks": smask}, context=None)
    diffs["scaler_seq_fit_loss_self"] = _max_abs(v2_sloss, sref_loss)
    checks["scaler_seq_fit_loss_self"] = diffs["scaler_seq_fit_loss_self"] <= PARITY_ATOL
    # TEST: per-token de-scaling through the functional scaler (preds[:, :, i])
    stb = Bundle()
    for stream, x in sinputs.items():
        stb.set(f"inputs.{stream}", x)
    stb.set("masks.tracks", smasks["tracks"])
    with torch.no_grad():
        stb = Executor(stest).run(stb)
    v2_stest = stb.get("preds.tracks.regression")
    sind = build_independent_v1_head(stask.task, scaler=RegressionTargetScaler(scaler_cfg))
    st_pooled, st_mask = stb.get("seq.x"), stb.get("masks.tracks")
    with torch.no_grad():
        st_raw, _ = sind(st_pooled, {}, {"tracks": st_mask}, context=None)
        sref_test = sind.run_inference(st_raw.clone(), labels=None, pad_mask=st_mask)
    diffs["scaler_seq_test_descale_vs_independent_v1"] = _max_abs(v2_stest, sref_test)
    checks["scaler_seq_test_descale_vs_independent_v1"] = (
        diffs["scaler_seq_test_descale_vs_independent_v1"] <= PARITY_ATOL
    )
    checks["scaler_seq_pred_is_3d"] = v2_spred.dim() == 3

    passed = all(checks.values())
    criterion = (
        "RegressionTask FIT preds+loss (vector norm_params) and TEST/ONNX ratio de-scaling match "
        "the composed v1 head (self-consistency) AND an INDEPENDENT v1 RegressionTask (cross-impl) "
        "to <= 1e-6; the functional-scaler per-token SEQUENCE de-scaling (preds[:, :, i]) matches "
        "an independent v1 head; TEST and ONNX denominator sources differ (FD §3.3); a denominator "
        "absent from inputs.<stream> is a bind-time ConfigError"
    )
    report = _base_report(
        "r1_regression_parity",
        passed,
        criterion,
        {"atol": PARITY_ATOL, "B": B, "T": T, "corrupted_by_test_hook": corruption is not None},
    )
    report["checks"] = checks
    report["max_abs_diffs"] = diffs
    print(f"{'diff':<32}{'value':>14}")
    for name, val in diffs.items():
        print(f"{name:<32}{val:>14.3e}")
    _print_checks(checks)
    _print_verdict("r1", passed, criterion, _emit_report(report, outdir, "r1"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# R2 — GaussianRegressionTask parity + ONNX-stddev design-conformance
# ---------------------------------------------------------------------------


def run_r2(
    outdir: Path | str,
    *,
    corruption: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[int, dict[str, Any]]:
    """R2: GaussianRegressionTask parity (mu/sigma, NLL, stddev) + ONNX-stddev conformance.

    v1-decidable half: output_size == 2R, the FIT Gaussian NLL loss and the TEST
    de-scaling (means ‖ sqrt(softplus(var))·std) match BOTH the composed v1
    GaussianRegressionTask (self-consistency, the executor's own object) AND an
    INDEPENDENT v1 ``GaussianRegressionTask`` (`build_independent_v1_head`:
    separately constructed + state-copied — genuine cross-impl parity) to
    <= 1e-6. The published TEST array IS the v1 (means, stds) TUPLE
    re-concatenated to one [B, 2R] leaf (FD 1567-1568 one-array contract).

    ONNX representation EXERCISED (not merely asserted on a TEST array): a
    Mode.ONNX plan is compiled and RUN, and the ONNX [B, 2R] array equals the
    TEST [B, 2R] (under norm_params the de-scaling math is mode-independent, so
    ONNX shares the TEST stddev = sqrt(softplus(var)) computation), with the
    second R columns strictly positive. This is *design-conformance*, NOT
    v1-byte-parity: v1 ``onnx/to_onnx.py`` has zero Gaussian handling and v1's
    (means, stds) is TEST-path-only, so there is no v1 ONNX golden to match. The
    writer's split_scalars manifest carries R mean suffixes + R `_stddev`
    suffixes index-aligned with the array.

    R=1 ONLY: v1 ``GaussianRegressionTask.run_inference`` indexes variances as
    ``preds[:, i+1]`` (task.py:752-754), correct only for R=1 (the variance sits
    at column 1). For R>1 the variances live at columns R..2R-1, so v1's i+1
    indexing is WRONG — RegressionTaskModule composes v1 verbatim and inherits
    that bug. No shipped config uses a multi-target gaussian head, so this gate
    gates the single correct case (R=1); multi-target gaussian de-scaling is a
    KNOWN-BROKEN v1 path, untested both sides, deferred to the M7 v1-absorption.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("R2 GaussianRegressionTask parity (mu/sigma/NLL/stddev) + ONNX-stddev conformance")
    print("=" * 96)
    norm_dict = _norm_dict(outdir)
    checks: dict[str, bool] = {}
    diffs: dict[str, float] = {}
    targets = ("HadronConeExclTruthLabelPt",)
    task = RegressionTaskModule(
        stream="jets",
        targets=list(targets),
        input="pooled.global",
        gaussian=True,
        norm_params={"mean": 2.0, "std": 3.0},
        weight=0.5,
        loss="GaussianNLLLoss",
    )
    plans = _build_and_bind(norm_dict, task, targets, modes=(Mode.FIT, Mode.TEST, Mode.ONNX))
    checks["output_size_is_2R"] = task.task.net.output_size == 2 * len(targets)
    checks["output_suffixes_means_then_stddev"] = task.output_suffixes == (
        *targets,
        *(f"{t}_stddev" for t in targets),
    )

    # FIT: raw [B, 2R] preds + NLL loss parity (self + independent v1 cross-impl)
    b, labels = _fit_bundle(targets)
    out = Executor(plans[Mode.FIT]).run(b, debug=True)
    v2_pred = out.get("preds.jets.regression")
    v2_loss = out.get("losses.regression")
    pooled = out.get("pooled.global")
    tdict = {"jets": {targets[0]: labels[f"labels.jets.{targets[0]}"]}}
    ref_pred, ref_loss = task.task(pooled, tdict, None, context=None)
    indep = build_independent_v1_head(task.task)
    _, ind_loss = indep(pooled, tdict, None, context=None)
    if corruption is not None:
        v2_loss = corruption(v2_loss)
    diffs["gaussian_fit_pred_self"] = _max_abs(v2_pred, ref_pred)
    diffs["gaussian_fit_loss_self"] = _max_abs(v2_loss, ref_loss)
    diffs["gaussian_fit_loss_vs_independent_v1"] = _max_abs(v2_loss, ind_loss)
    checks["gaussian_fit_pred_self"] = diffs["gaussian_fit_pred_self"] <= PARITY_ATOL
    checks["gaussian_fit_loss_parity"] = diffs["gaussian_fit_loss_self"] <= PARITY_ATOL
    checks["gaussian_fit_loss_vs_independent_v1"] = (
        diffs["gaussian_fit_loss_vs_independent_v1"] <= PARITY_ATOL
    )

    # TEST: one [B, 2R] array == v1 (means, stds) re-concatenated (self + independent)
    bt, _ = _test_bundle()
    with torch.no_grad():
        bt = Executor(plans[Mode.TEST]).run(bt)
    v2_test = bt.get("preds.jets.regression")
    pooled_t = bt.get("pooled.global")
    with torch.no_grad():
        raw_t, _ = task.task(pooled_t, {}, None, context=None)
        ref_means, ref_stds = task.task.run_inference(raw_t.clone())
        ind_raw, _ = indep(pooled_t, {}, None, context=None)
        ind_means, ind_stds = indep.run_inference(ind_raw.clone())
    ref_concat = torch.cat([ref_means, ref_stds], dim=-1)
    ind_concat = torch.cat([ind_means, ind_stds], dim=-1)
    diffs["gaussian_test_one_array"] = _max_abs(v2_test, ref_concat)
    diffs["gaussian_test_vs_independent_v1"] = _max_abs(v2_test, ind_concat)
    checks["gaussian_test_one_array_parity"] = diffs["gaussian_test_one_array"] <= PARITY_ATOL
    checks["gaussian_test_vs_independent_v1"] = (
        diffs["gaussian_test_vs_independent_v1"] <= PARITY_ATOL
    )
    checks["test_array_is_2R_wide"] = tuple(v2_test.shape) == (B, 2 * len(targets))

    # ONNX EXERCISED: run the compiled Mode.ONNX plan, assert ONNX == TEST
    # [B, 2R] (norm_params de-scaling is mode-independent, so ONNX shares the
    # TEST stddev = sqrt(softplus(var)) math). NOT a v1-byte-parity gate — v1 has
    # no gaussian ONNX handling; this is design-conformance the gate now RUNS.
    inputs, masks = make_gn2_batch(B, T)
    bonnx = Bundle()
    for stream, x in inputs.items():
        bonnx.set(f"inputs.{stream}", x)
    bonnx.set("masks.tracks", masks["tracks"])
    with torch.no_grad():
        bonnx = Executor(plans[Mode.ONNX]).run(bonnx)
    v2_onnx = bonnx.get("preds.jets.regression")
    diffs["gaussian_onnx_eq_test"] = _max_abs(v2_onnx, v2_test)
    checks["gaussian_onnx_array_is_2R_wide"] = tuple(v2_onnx.shape) == (B, 2 * len(targets))
    checks["gaussian_onnx_equals_test"] = diffs["gaussian_onnx_eq_test"] <= PARITY_ATOL
    # design-conformance: the ONNX stddev half is strictly positive
    checks["onnx_stddev_design_conformance_positive"] = bool((v2_onnx[:, len(targets) :] > 0).all())

    passed = all(checks.values())
    criterion = (
        "GaussianRegressionTask output_size==2R (R=1, v1's only correct multi-target case); FIT "
        "NLL loss + TEST de-scaling (means ‖ sqrt(softplus(var))·std) match the composed v1 head "
        "(self-consistency) AND an INDEPENDENT v1 GaussianRegressionTask (cross-impl) to <= 1e-6; "
        "the published TEST array is the v1 (means, stds) re-concatenated to one [B, 2R] leaf; the "
        "Mode.ONNX plan is COMPILED+RUN and its [B, 2R] equals TEST with a strictly-positive "
        "stddev half — design-conformance (NO v1 ONNX golden — v1 onnx/to_onnx.py has zero "
        "gaussian handling)"
    )
    report = _base_report(
        "r2_gaussian_parity",
        passed,
        criterion,
        {
            "atol": PARITY_ATOL,
            "B": B,
            "R": len(targets),
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["max_abs_diffs"] = diffs
    report["onnx_stddev_is_design_conformance"] = (
        "the Mode.ONNX plan is compiled and RUN; ONNX [B, 2R] == TEST [B, 2R] because norm_params "
        "de-scaling is mode-independent, so ONNX shares the TEST stddev = sqrt(softplus(var)) "
        "math; the split_scalars manifest declares R means + R _stddev suffixes. NOT a "
        "v1-byte-parity gate: v1 has no gaussian ONNX handling (to_onnx.py grep empty)."
    )
    report["multi_target_gaussian_descale"] = (
        "R=1 ONLY: v1 GaussianRegressionTask.run_inference indexes variances as preds[:, i+1] "
        "(task.py:752-754), correct only for R=1. For R>1 variances live at columns R..2R-1, so "
        "v1's i+1 indexing is broken; RegressionTaskModule composes v1 verbatim and inherits it. "
        "No shipped config uses a multi-target gaussian head — the broken path is untested both "
        "sides, deferred to M7 v1-absorption."
    )
    print(f"{'diff':<32}{'value':>14}")
    for name, val in diffs.items():
        print(f"{name:<32}{val:>14.3e}")
    _print_checks(checks)
    _print_verdict("r2", passed, criterion, _emit_report(report, outdir, "r2"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# R3 — sample_weight + NaN-target masking parity
# ---------------------------------------------------------------------------


def run_r3(
    outdir: Path | str,
    *,
    corruption: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[int, dict[str, Any]]:
    """R3: sample_weight + NaN-target masking + encoder-less pooling parity.

    Adversarial coverage (the codex plan-review finding): NONUNIFORM and ALL-ZERO
    sample weights over TWO targets (the unsqueeze + expand, task.py:433-434),
    AND a NaN-target run (NaN masked to 0, reduced with torch.nanmean,
    task.py:412-437). Every case asserts the v2 scalar loss matches the composed
    v1 head (self-consistency) after the same mask→0, weight-expand, nanmean
    ORDERING to <= 1e-6, and that the NaN case did not poison the loss (finite).
    The nonuniform-weight variant ALSO asserts genuine cross-impl parity against
    an INDEPENDENT v1 ``RegressionTask`` (`build_independent_v1_head`: separately
    constructed + state-copied), so the sample-weight wiring is checked against a
    head v2 never touched, not only its own composed object.

    Encoder-less pooling forward parity (plan 10 R3 row; the primary CI smoke
    fixture ``legacy/dips.yaml``, test_pipeline.py:208,309,317): the dips plan
    (``init_nets`` + ``pool_net``, NO ``encoder:``) runs through the real
    compiler/bind/executor, and the executor's ``pooled.global`` is compared
    BITWISE against a STANDALONE v1 `GlobalAttentionPooling` — built fresh,
    loaded with the bound v2 pool's gate weights, fed the executor's
    ``seq.x``/``seq.mask`` with the v1 encoder-less pad dict ``{"seq": seq.mask}``
    (pooling.py:53-63). Also asserts the compiled dips plan carries NO
    ``masks.registers`` key (the optional require is dropped when no encoder
    produces it — the encoder-less pooling fix this gate guards; v1
    saltmodel.py:155-156). A pure pooling-math comparison with deterministic
    inputs, so BITWISE is required, not the <= 1e-6 fallback.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("R3 sample_weight + NaN-target masking + encoder-less dips pooling parity vs v1")
    print("=" * 96)
    norm_dict = _norm_dict(outdir)
    none_loss = {"class_path": "torch.nn.MSELoss", "init_args": {"reduction": "none"}}
    checks: dict[str, bool] = {}
    diffs: dict[str, float] = {}

    # -- sample_weight: nonuniform + all-zero, two targets --------------------
    targets = ("R10TruthLabel_R22v1_TruthJetMass", "R10TruthLabel_R22v1_TruthJetPt")
    weight_cases = (
        ("nonuniform", torch.tensor([1.0, 0.5, 2.0, 0.0, 1.5, 0.25])),
        ("zero", torch.zeros(B)),
    )
    for tag, w in weight_cases:
        task = RegressionTaskModule(
            stream="jets",
            targets=list(targets),
            input="pooled.global",
            sample_weight="w",
            norm_params={"mean": [1.0, 2.0], "std": [3.0, 4.0]},
            loss=none_loss,
        )
        plans = _build_and_bind(norm_dict, task, targets, weight="w", modes=(Mode.FIT,))
        b, labels = _fit_bundle(targets, weight=("w", w))
        out = Executor(plans[Mode.FIT]).run(b, debug=True)
        v2_loss = out.get("losses.regression")
        if corruption is not None and tag == "nonuniform":
            v2_loss = corruption(v2_loss)
        pooled = out.get("pooled.global")
        tdict = {"jets": {t: labels[f"labels.jets.{t}"] for t in targets}}
        tdict["jets"]["w"] = w
        # self-consistency vs the composed v1 head (the executor's own object)
        _, ref_loss = task.task(pooled, tdict, None, context=None)
        diff = _max_abs(v2_loss, ref_loss)
        diffs[f"sample_weight_{tag}_loss"] = diff
        checks[f"sample_weight_{tag}_loss_parity"] = diff <= PARITY_ATOL
        # the nonuniform variant ALSO gets genuine cross-impl parity vs an
        # INDEPENDENT v1 RegressionTask (separately built + state-copied)
        if tag == "nonuniform":
            indep = build_independent_v1_head(task.task)
            _, ind_loss = indep(pooled, tdict, None, context=None)
            diffs["sample_weight_nonuniform_loss_vs_independent_v1"] = _max_abs(v2_loss, ind_loss)
            checks["sample_weight_nonuniform_loss_vs_independent_v1"] = (
                diffs["sample_weight_nonuniform_loss_vs_independent_v1"] <= PARITY_ATOL
            )

    # -- NaN-target masking ----------------------------------------------------
    ntargets = ("HadronConeExclTruthLabelPt",)
    ntask = RegressionTaskModule(
        stream="jets",
        targets=list(ntargets),
        input="pooled.global",
        norm_params={"mean": 1.0, "std": 1.0},
        loss=none_loss,
    )
    nplans = _build_and_bind(norm_dict, ntask, ntargets, modes=(Mode.FIT,))
    nb, nlabels = _fit_bundle(ntargets, poison_nan=True)
    nout = Executor(nplans[Mode.FIT]).run(nb, debug=True)
    v2_nloss = nout.get("losses.regression")
    checks["nan_loss_is_finite"] = bool(torch.isfinite(v2_nloss))
    pooled_n = nout.get("pooled.global")
    tdict_n = {"jets": {ntargets[0]: nlabels[f"labels.jets.{ntargets[0]}"]}}
    _, ref_nloss = ntask.task(pooled_n, tdict_n, None, context=None)
    diffs["nan_masking_loss"] = _max_abs(v2_nloss, ref_nloss)
    checks["nan_masking_loss_parity"] = diffs["nan_masking_loss"] <= PARITY_ATOL

    # -- encoder-less pooling forward parity on legacy/dips.yaml ---------------
    dips = build_dips_modules(norm_dict)
    fit_plan = compile_dips(dips, Mode.FIT)
    bind_all(dips, resolve_bind_schema([fit_plan]))
    materialise_all(dips)
    # the encoder-less plan must NOT carry the (optional, encoder-produced)
    # registers mask — the pooling fix this gate guards (v1 saltmodel.py:155-156).
    # the optional require is dropped from the pool step when no producer exists.
    checks["dips_plan_has_no_registers"] = "masks.registers" not in fit_plan.step("pool").requires
    inputs, masks = make_gn2_batch(B, T)
    db = Bundle()
    for stream, x in inputs.items():
        db.set(f"inputs.{stream}", x)
    db.set("masks.tracks", masks["tracks"])
    for key, val in make_dips_labels(B).items():
        db.set(key, val)
    dout = Executor(fit_plan).run(db, debug=True)
    v2_pooled = dout.get("pooled.global")
    if corruption is not None:
        v2_pooled = corruption(v2_pooled)
    # standalone v1 reference: fresh GAP, the bound v2 gate weights, the same
    # seq.x / seq.mask, the v1 encoder-less pad dict {"seq": seq.mask}
    seq_x, seq_mask = dout.get("seq.x"), dout.get("seq.mask")
    pool_mod = dips["pool"]
    v1_pool = V1GlobalAttentionPooling(input_size=seq_x.shape[-1])
    v1_pool.load_state_dict(pool_mod.pool_net.state_dict())
    v1_pool.eval()
    with torch.no_grad():
        ref_pooled = v1_pool({"seq": seq_x}, pad_mask={"seq": seq_mask})
    diffs["dips_pool_forward"] = _max_abs(v2_pooled, ref_pooled)
    checks["dips_encoderless_pool_bitwise"] = torch.equal(v2_pooled, ref_pooled)
    checks["dips_encoderless_pool_parity"] = diffs["dips_pool_forward"] <= PARITY_ATOL

    passed = all(checks.values())
    criterion = (
        "per-sample-weighted loss (nonuniform + all-zero weights, two targets, the "
        "unsqueeze+expand) and NaN-target masking (mask→0, torch.nanmean) match the composed v1 "
        "nan_loss to <= 1e-6 with the same ordering; the NaN run stays finite (v1 "
        "task.py:412-437); the encoder-less legacy/dips.yaml pool reproduces a standalone v1 "
        "GlobalAttentionPooling BITWISE on the same seq.x/seq.mask with the v1 pad dict "
        "{'seq': seq.mask}, and the dips plan carries no masks.registers (v1 "
        "saltmodel.py:90-93,155-156, pooling.py:53-63)"
    )
    report = _base_report(
        "r3_weight_nan_parity",
        passed,
        criterion,
        {"atol": PARITY_ATOL, "B": B, "corrupted_by_test_hook": corruption is not None},
    )
    report["checks"] = checks
    report["max_abs_diffs"] = diffs
    print(f"{'diff':<32}{'value':>14}")
    for name, val in diffs.items():
        print(f"{name:<32}{val:>14.3e}")
    _print_checks(checks)
    _print_verdict("r3", passed, criterion, _emit_report(report, outdir, "r3"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# R4 — MultiTarget processor row-replacement parity
# ---------------------------------------------------------------------------


def run_r4(
    outdir: Path | str,
    *,
    corruption: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[int, dict[str, Any]]:
    """R4: MultiTarget conditional replacement BITWISE parity vs v1's torch.where.

    The shipped ``regression_multi_target.yaml`` shape: two rules write one
    ``custom_target`` (``ID==15 -> Pt``, ``ID!=15 -> pt``) over a single NaN-base
    running array (v1 in-place mutation, datasets.py:709-739). R4 reproduces v1's
    ``apply_multi_target_replacements`` (``torch.where`` over a labels dict) step
    by step and asserts the v2 numpy ``np.where`` output is BITWISE identical —
    a pure data op, no model, no float drift. Also covers the single-rule
    ``target:`` (raw-base replacement) case.

    Placeholder-dtype deviation (documented, not gated as a divergence): v1's
    ``inject_custom_target_placeholders`` (datasets.py:686-690) types the NaN base
    as ``batch[source].dtype`` verbatim; v2 ``MultiTarget.process`` promotes any
    sub-float32 source to >=float32 (``np.result_type(template.dtype, float32)``).
    For f4/f8 sources the two are byte-identical, and every shipped
    regression_multi_target.yaml source is f4 (HadronConeExclTruthLabelPt, pt), so
    this fixture is f4 and the BITWISE claim holds. The divergence is reachable
    only with an f2 source — which no shipped config has — and the float32
    promotion is intentional (a NaN-filled regression placeholder needs the range
    for log/ratio targets).

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("R4 MultiTarget processor conditional-replacement BITWISE parity vs v1 torch.where")
    print("=" * 96)
    checks: dict[str, bool] = {}
    rng = np.random.default_rng(7)
    n = 64
    flav = rng.integers(0, 16, size=n)
    pt_truth = rng.standard_normal(n).astype(np.float32)
    pt_reco = rng.standard_normal(n).astype(np.float32)
    raw_existing = rng.standard_normal(n).astype(np.float32)

    # -- (a) two-rule custom_target chain (the shipped config) -----------------
    mt = MultiTarget(
        replacements=[
            {
                "stream": "jets",
                "sel_label": "HadronConeExclTruthLabelID",
                "op": "==",
                "value": 15,
                "source": "Pt",
                "custom_target": "pt_label_handle",
            },
            {
                "stream": "jets",
                "sel_label": "HadronConeExclTruthLabelID",
                "op": "!=",
                "value": 15,
                "source": "pt",
                "custom_target": "pt_label_handle",
            },
        ]
    )
    mt.name = "multi_target"
    b = Bundle()
    b.set("labels.jets.HadronConeExclTruthLabelID", flav)
    b.set("labels.jets.Pt", pt_truth)
    b.set("labels.jets.pt", pt_reco)
    v2 = mt.process(b, np.s_[0:n], Mode.FIT)["labels.jets.pt_label_handle"]
    if corruption is not None:
        v2 = corruption(v2)
    # v1 reference: NaN-base placeholder, two sequential torch.where (the v1 loop
    # mutates labels[input][target] in place, datasets.py:735-739)
    running = torch.full((n,), float("nan"))
    running = torch.where(
        V1_OPERATORS["=="](torch.as_tensor(flav), 15), torch.as_tensor(pt_truth), running
    )
    running = torch.where(
        V1_OPERATORS["!="](torch.as_tensor(flav), 15), torch.as_tensor(pt_reco), running
    )
    ref = running.numpy()
    checks["custom_target_chain_bitwise"] = (
        np.array_equal(np.nan_to_num(v2), np.nan_to_num(ref))
        and (np.isnan(v2) == np.isnan(ref)).all()
    )
    checks["custom_target_chain_no_nan_left"] = not np.isnan(v2).any()

    # -- (b) single-rule target: (raw-base replacement) -----------------------
    mt2 = MultiTarget(
        replacements=[
            {
                "stream": "jets",
                "sel_label": "flav",
                "op": ">=",
                "value": 4,
                "source": "src",
                "target": "tgt",
            }
        ]
    )
    mt2.name = "multi_target"
    arr = np.zeros(n, dtype=[("tgt", "f4")])
    arr["tgt"] = raw_existing
    b2 = Bundle()
    b2.set("labels.jets.flav", flav)
    b2.set("labels.jets.src", pt_truth)
    b2.set("raw.jets", arr)
    v2b = mt2.process(b2, np.s_[0:n], Mode.FIT)["labels.jets.tgt"]
    refb = torch.where(
        V1_OPERATORS[">="](torch.as_tensor(flav), 4),
        torch.as_tensor(pt_truth),
        torch.as_tensor(raw_existing),
    ).numpy()
    checks["target_mode_raw_base_bitwise"] = np.array_equal(v2b, refb)

    passed = all(checks.values())
    criterion = (
        "MultiTarget np.where conditional replacement is BITWISE identical to v1's "
        "apply_multi_target_replacements (torch.where over a labels dict, datasets.py:695-739): "
        "the two-rule custom_target chain over a NaN-base running array AND the single-rule "
        "target: raw-base replacement"
    )
    report = _base_report(
        "r4_multitarget_parity",
        passed,
        criterion,
        {"n_rows": n, "corrupted_by_test_hook": corruption is not None},
    )
    report["checks"] = checks
    _print_checks(checks)
    _print_verdict("r4", passed, criterion, _emit_report(report, outdir, "r4"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# L1 — LossGLS geometric-mean parity + all-weights==1.0 guard (sub-wave B)
# ---------------------------------------------------------------------------


def run_l1(
    outdir: Path | str,
    *,
    corruption: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[int, dict[str, Any]]:
    """L1: `LossGLS` geometric-mean parity + the all-weights==1.0 guard vs v1.

    `LossGLS` is the v2 home of v1's ``loss_mode == 'GLS'`` reduction (a
    ``loss_mode`` string in v1, not a class — modelwrapper.py:89,136,139-142,
    194-196), gating the entire GN3 port (14 configs untrainable without it,
    plan 10 sub-wave B). It is a NetModule SIBLING of `LossSum` (subclassing it
    to share ``losses.**`` framework narrowing + ``declare_io``); the ONLY
    behavioural change is the combination rule and the weight guard.

    Two assertions, both with teeth:

    - **Geometric-mean parity vs an INDEPENDENT v1 reference.** A two-task
      encoder-less DiPS plan (jet classifier + jet regression, both weight 1.0)
      runs through the REAL compiler / two-phase bind / executor, yielding
      ``loss.total`` AND the per-task ``losses.*``. The reference is v1's ACTUAL
      ``ModelWrapper.total_loss`` in GLS mode, taken off a SEPARATELY-constructed
      v1 ``ModelWrapper`` (`build_independent_v1_gls_total_loss`) and fed the
      SAME per-task loss dict — so the comparison isolates the combination math
      (``math.prod`` then ``pow(·, 1/n)``, modelwrapper.py:194-196), not a
      re-implementation in the gate. BITWISE-first (``torch.equal``): the v2
      ``forward`` issues the identical ``math.prod`` + ``torch.pow`` ops on the
      identical inputs, so byte-equality is expected; the <= 1e-6 fallback is
      recorded but should never be the reason the check passes. A control proves
      it is a real geometric mean and not a sum (``loss.total`` differs from the
      per-task sum).

    - **The all-weights==1.0 guard raises loudly (both weight surfaces).** GLS is
      only valid when no task is pre-scaled (a weighted task loss ``w*L`` rescales
      the mean by ``w^(1/n)`` — silent divergence). v2 guards BOTH weight
      surfaces: (a) the module's own per-loss ``weights`` multiplier — rejected
      in ``LossGLS.__init__``; (b) the task-side ``weight: float`` applied inside
      each head — checked by ``LossGLS.check_task_weights(modules)`` (the v2 home
      of v1's ctor assert, modelwrapper.py:139-142). The gate asserts each raises
      a ``ConfigError`` for a non-1.0 weight AND stays silent at exactly 1.0.

    Negative control (``test_gates_m5.py``): the ``corruption`` hook perturbs a
    per-task loss before the parity comparison — the parity check must FAIL while
    the guard checks (independent of the loss values) stay green.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)`` — 0 only if every check passed.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("L1 LossGLS geometric-mean parity + all-weights==1.0 guard vs v1 modelwrapper.py")
    print("=" * 96)
    norm_dict = _norm_dict(outdir)
    checks: dict[str, bool] = {}
    diffs: dict[str, float] = {}

    # -- (a) geometric-mean parity vs an INDEPENDENT v1 ModelWrapper.total_loss -
    modules = build_gls_modules(norm_dict)
    fit_plan = compile_gls(modules, Mode.FIT)
    bind_all(modules, resolve_bind_schema([fit_plan]))
    materialise_all(modules)
    loss_mod = modules["loss"]
    assert isinstance(loss_mod, LossGLS)
    loss_keys = list(loss_mod._loss_keys)  # noqa: SLF001 - gate inspects the narrowed keys
    checks["gls_has_two_task_losses"] = len(loss_keys) >= 2

    inputs, masks = make_gn2_batch(B, T)
    b = Bundle()
    for stream, x in inputs.items():
        b.set(f"inputs.{stream}", x)
    b.set("masks.tracks", masks["tracks"])
    for key, val in make_gls_labels(B).items():
        b.set(key, val)
    out = Executor(fit_plan).run(b, debug=True)
    v2_total = out.get("loss.total")
    # the per-task losses the executor produced (the v1 reference's input)
    per_task = {key.split(".", 1)[1]: out.get(key) for key in loss_keys}
    # the negative control perturbs ONLY the v2 OBSERVED total (the v1 reference
    # below is fed the genuine per-task losses), so a corrupted combination math
    # diverges from v1 — the parity check must then FAIL
    if corruption is not None:
        v2_total = corruption(v2_total)

    # INDEPENDENT v1: a fresh ModelWrapper's real total_loss (GLS mode), fed the
    # SAME per-task loss dict — pure combination-math parity (modelwrapper.py:194-196)
    v1_total = build_independent_v1_gls_total_loss(outdir)(per_task)
    diffs["gls_total_vs_independent_v1"] = _max_abs(v2_total, v1_total)
    checks["gls_total_bitwise_vs_independent_v1"] = torch.equal(v2_total, v1_total)
    checks["gls_total_parity_vs_independent_v1"] = (
        diffs["gls_total_vs_independent_v1"] <= PARITY_ATOL
    )
    # control: GLS is a geometric mean, NOT a sum (would diverge for >1 task)
    v2_sum = sum(per_task.values())
    checks["gls_is_not_the_sum"] = not torch.allclose(v2_total, v2_sum, atol=PARITY_ATOL)

    # -- (b) the all-weights==1.0 guard, both surfaces --------------------------
    # surface (a): the module's per-loss weights multiplier — rejected in __init__
    raised_module_weight = False
    try:
        LossGLS(losses=["jets_classification", "jets_regression"], weights={"jets_regression": 2.0})
    except ConfigError:
        raised_module_weight = True
    checks["module_weight_not_one_raises"] = raised_module_weight
    # at exactly 1.0 the module weight is accepted (no false positive)
    accepted_unit_module_weight = True
    try:
        LossGLS(losses=["a", "b"], weights={"a": 1.0, "b": 1.0})
    except ConfigError:
        accepted_unit_module_weight = False
    checks["module_weight_one_accepted"] = accepted_unit_module_weight

    # surface (b): the task-side weight: float — check_task_weights (the v1 guard)
    weighted_modules = build_gls_modules(norm_dict, weights={"jets_regression": 2.0})
    raised_task_weight = False
    try:
        LossGLS.check_task_weights(weighted_modules)
    except ConfigError:
        raised_task_weight = True
    checks["task_weight_not_one_raises"] = raised_task_weight
    # all task weights 1.0 (the GLS domain) passes the guard cleanly
    unit_task_weight_ok = True
    try:
        LossGLS.check_task_weights(modules)  # the weight-1.0 fixture from (a)
    except ConfigError:
        unit_task_weight_ok = False
    checks["task_weight_one_accepted"] = unit_task_weight_ok

    passed = all(checks.values())
    criterion = (
        "the two-task encoder-less LossGLS plan's loss.total equals an INDEPENDENT v1 "
        "ModelWrapper.total_loss (loss_mode='GLS') fed the same per-task losses BITWISE "
        "(math.prod then pow(.,1/n), modelwrapper.py:194-196) and differs from the sum; the "
        "all-weights==1.0 guard raises a ConfigError for a non-1.0 weight on BOTH the module's "
        "per-loss weights surface (LossGLS.__init__) and the task-side weight surface "
        "(LossGLS.check_task_weights — the v2 home of v1 modelwrapper.py:139-142), and is silent "
        "at exactly 1.0"
    )
    report = _base_report(
        "l1_gls_parity_and_weight_guard",
        passed,
        criterion,
        {
            "atol": PARITY_ATOL,
            "B": B,
            "loss_keys": loss_keys,
            "n_losses": len(loss_keys),
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["max_abs_diffs"] = diffs
    report["v1_reference"] = (
        "salt.modelwrapper.ModelWrapper.total_loss with loss_mode='GLS' on a "
        "separately-constructed v1 wrapper (build_independent_v1_gls_total_loss)"
    )
    print(f"{'diff':<40}{'value':>14}")
    for name, val in diffs.items():
        print(f"{name:<40}{val:>14.3e}")
    _print_checks(checks)
    _print_verdict("l1", passed, criterion, _emit_report(report, outdir, "l1"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# L2 — norm_type: hybrid passthrough numerical parity (sub-wave B)
# ---------------------------------------------------------------------------


def run_l2(
    outdir: Path | str,
    *,
    corruption: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[int, dict[str, Any]]:
    """L2: `TransformerEncoder` ``norm_type: hybrid`` passthrough parity vs v1.

    The GN3V01 flagship (+ GN3_Hybrid, GN3EPCLV01) encoder is ``norm_type:
    hybrid`` (plan 10 sub-wave B); the v2 `TransformerEncoder` wrapper now
    threads the flag through to every composed v1 ``EncoderLayer``. All the
    hybrid placement logic — forced ``do_qk_norm``/``do_v_norm`` on the
    `Attention`, the depth-0 (``"pre"`` + Identity) vs depth>0 (``"none"`` + a
    real norm) residual split, and the pre-FFN norm in ``forward`` — lives in
    the v1 layer (transformer.py:350-356,421); the wrapper only passes the
    string. L2 proves that passthrough is numerically faithful.

    Assertions, all with teeth:

    - **Encoder forward parity vs an INDEPENDENT v1 `Transformer`.** A
      GN3V01-style block (norm -> embed -> concat -> encoder[hybrid] -> split ->
      pool -> head) runs through the REAL compiler / two-phase bind / executor,
      yielding the encoder output ``encoded.seq`` and its inputs ``seq.x`` /
      ``seq.mask``. The reference is a SEPARATELY-constructed v1 ``Transformer``
      (`build_independent_v1_transformer`: fresh, same depth / widths /
      ``norm_type`` / attention+dense kwargs, ``load_state_dict``-copied
      weights), fed the SAME ``seq.x`` / ``seq.mask`` on the v1 ``{"seq": x}``
      dict contract — so the comparison isolates the hybrid-norm math, not the
      wrapper's own internal object. BITWISE-first (``torch.equal``): the v2
      wrapper hands its composed v1 ``Transformer`` the same dict the
      independent ref gets, so byte-equality is expected; the <= 1e-6 fallback
      is recorded but should never be the reason the check passes.

    - **The hybrid flag actually reached the v1 layers.** Every composed
      `EncoderLayer` reports ``norm_type == "hybrid"`` and its `Attention`
      forced ``do_qk_norm`` / ``do_v_norm`` (transformer.py:350-352); depth 0
      uses an Identity layer-norm with a ``"pre"`` residual while depth>0 uses a
      real layer-norm with a ``"none"`` residual (transformer.py:353-356). This
      proves the passthrough is the v1 hybrid wiring, not a silent ``"pre"``
      fallback.

    - **Hybrid differs from pre-norm (MATCHED-INIT control).** A
      ``norm_type="pre"`` encoder block of the SAME shape is built, then loaded
      with the HYBRID encoder's weights (``load_state_dict(..., strict=False)``):
      the pre-norm encoder's parameters are a strict subset of the hybrid one's
      (the extra hybrid keys are only the forced QK/V norms) with matching
      shapes, so EVERY pre-norm weight is copied from the hybrid encoder. The
      two paths then differ ONLY by norm PLACEMENT (pre vs hybrid) plus the
      hybrid-only forced QK/V norms — both purely the hybrid wiring — so the
      checks ``matched_init_no_missing_pre_keys`` /
      ``matched_init_unexpected_are_qk_v_norms`` pin the copy, and the SAME
      ``seq.x`` / ``seq.mask`` through it yields a different ``encoded.seq``.
      This is a true matched-init A/B: the difference is attributable to the
      placement, not to two differently-seeded encoders.

    - **The wrapper rejects an unknown ``norm_type`` loudly.** ``norm_type:
      "none"`` (a v1 residual-only mode with no shipped v2 config) raises a
      ``ConfigError`` at construction.

    L2 exercises the FIT/TEST encoder forward only — no ONNX claim. The hybrid
    passthrough is purely a construction-time wiring of the v1 encoder; the
    encoder's own ONNX path (the deterministic torch-math backend +
    `Split`/`Concat` offsets) is already gated by the M2/M4 ONNX gates and is
    norm_type-agnostic, so L2 makes no ONNX assertion here.

    Negative control (``test_gates_m5.py``): the ``corruption`` hook perturbs
    the v2 OBSERVED ``encoded.seq`` before the parity comparison — the parity
    check must FAIL while the flag-reached-the-layers and reject-unknown checks
    (independent of the encoder values) stay green.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)`` — 0 only if every check passed.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("L2 TransformerEncoder norm_type:hybrid passthrough parity vs v1 transformer.py")
    print("=" * 96)
    norm_dict = _norm_dict(outdir)
    checks: dict[str, bool] = {}
    diffs: dict[str, float] = {}

    # -- (a) encoder forward parity vs an INDEPENDENT v1 Transformer -----------
    modules = build_hybrid_encoder_modules(norm_dict, norm_type="hybrid")
    fit_plan = compile_hybrid_encoder(modules, Mode.FIT)
    bind_all(modules, resolve_bind_schema([fit_plan]))
    materialise_all(modules)
    inputs, masks = make_gn2_batch(B, T)
    b = Bundle()
    for stream, x in inputs.items():
        b.set(f"inputs.{stream}", x)
    b.set("masks.tracks", masks["tracks"])
    for key, val in make_dips_labels(B).items():
        b.set(key, val)
    out = Executor(fit_plan).run(b, debug=True)
    v2_enc = out.get("encoded.seq")
    seq_x, seq_mask = out.get("seq.x"), out.get("seq.mask")
    if corruption is not None:
        v2_enc = corruption(v2_enc)
    # INDEPENDENT v1: fresh Transformer, same kwargs + copied weights, fed the
    # SAME seq.x/seq.mask on the v1 {"seq": x} dict contract (transformer.py:684).
    encoder_mod = modules["encoder"]
    v1_enc = build_independent_v1_transformer(encoder_mod)
    with torch.no_grad():
        ref_enc, _ = v1_enc({"seq": seq_x.clone()}, pad_mask={"seq": seq_mask.clone()})
    diffs["hybrid_encoded_vs_independent_v1"] = _max_abs(v2_enc, ref_enc)
    checks["hybrid_encoded_bitwise_vs_independent_v1"] = torch.equal(v2_enc, ref_enc)
    checks["hybrid_encoded_parity_vs_independent_v1"] = (
        diffs["hybrid_encoded_vs_independent_v1"] <= PARITY_ATOL
    )

    # -- (b) the hybrid flag actually reached every composed v1 EncoderLayer ---
    layers = encoder_mod.encoder.layers
    all_hybrid = all(layer.norm_type == "hybrid" for layer in layers)
    all_qk_v_forced = all(layer.attn.fn.do_qk_norm and layer.attn.fn.do_v_norm for layer in layers)
    # depth 0: residual "pre" + Identity layer-norm; depth>0: "none" + a real norm
    depth0 = layers[0]
    depth0_ok = depth0.attn.norm_type == "pre" and type(depth0.norm).__name__ == "Identity"
    deeper_ok = all(
        layer.attn.norm_type == "none" and type(layer.norm).__name__ != "Identity"
        for layer in layers[1:]
    )
    checks["all_layers_hybrid"] = all_hybrid
    checks["all_layers_force_qk_v_norm"] = all_qk_v_forced
    checks["depth0_pre_residual_identity_norm"] = depth0_ok
    checks["deeper_layers_none_residual_real_norm"] = deeper_ok

    # -- (c) control: MATCHED-INIT placement A/B -------------------------------
    # A pre-norm encoder block built with the SAME shape, then loaded with the
    # hybrid encoder's weights (strict=False): the pre-norm encoder's 28 params
    # are a SUBSET of the hybrid encoder's 37 (the 9 extra are the forced
    # do_qk/do_v norms) and every shared key has a matching shape, so this copies
    # ALL of the pre-norm encoder's weights from the hybrid one. The two paths
    # now differ ONLY by norm PLACEMENT (pre vs hybrid) plus the hybrid-only
    # forced QK/V norms — i.e. purely the hybrid wiring, not a wholesale random
    # reinit. A non-trivial difference therefore proves the gate is sensitive to
    # the norm placement itself, not just to two differently-seeded encoders.
    pre_modules = build_hybrid_encoder_modules(norm_dict, norm_type="pre")
    pre_plan = compile_hybrid_encoder(pre_modules, Mode.FIT)
    bind_all(pre_modules, resolve_bind_schema([pre_plan]))
    materialise_all(pre_modules)
    pre_encoder = pre_modules["encoder"].encoder
    hybrid_sd = encoder_mod.encoder.state_dict()
    missing, unexpected = pre_encoder.load_state_dict(hybrid_sd, strict=False)
    # matched-init invariant: every pre-norm param is filled from the hybrid one
    # (no missing keys); the only un-copied hybrid keys are the forced QK/V norms.
    checks["matched_init_no_missing_pre_keys"] = len(missing) == 0
    checks["matched_init_unexpected_are_qk_v_norms"] = all(
        k.endswith(("q_norm.weight", "k_norm.weight", "v_norm.weight")) for k in unexpected
    )
    pre_encoder.eval()
    with torch.no_grad():
        pre_enc, _ = pre_encoder({"seq": seq_x.clone()}, pad_mask={"seq": seq_mask.clone()})
    checks["hybrid_differs_from_pre"] = not torch.allclose(ref_enc, pre_enc, atol=PARITY_ATOL)

    # -- (d) an unknown norm_type is a loud construction error -----------------
    raised_bad_norm_type = False
    try:
        TransformerEncoder(
            dim=HYBRID_ENC_DIM, num_layers=1, attention={"num_heads": 2}, norm_type="none"
        )
    except ConfigError:
        raised_bad_norm_type = True
    checks["unknown_norm_type_raises"] = raised_bad_norm_type

    passed = all(checks.values())
    criterion = (
        "the GN3V01-style norm_type=hybrid encoder block's encoded.seq (from the real compiler/"
        "bind/executor) equals an INDEPENDENT v1 Transformer (same kwargs + copied weights, fed "
        "the same seq.x/seq.mask) BITWISE; every composed v1 EncoderLayer reports norm_type=hybrid "
        "with forced do_qk_norm/do_v_norm and the depth-0 (pre+Identity) vs depth>0 (none+real "
        "norm) residual split (transformer.py:350-356,421); a MATCHED-INIT pre-norm encoder "
        "(hybrid weights load_state_dict-copied, strict=False) differs from the hybrid output — "
        "isolating the difference to norm placement; and an unknown norm_type raises a ConfigError"
    )
    report = _base_report(
        "l2_hybrid_norm_passthrough",
        passed,
        criterion,
        {
            "atol": PARITY_ATOL,
            "B": B,
            "T": T,
            "num_layers": len(layers),
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["max_abs_diffs"] = diffs
    report["v1_reference"] = (
        "salt.models.Transformer built with norm_type='hybrid' (same depth/widths/attn+dense "
        "kwargs) + load_state_dict-copied weights (build_independent_v1_transformer); the hybrid "
        "placement lives in v1 EncoderLayer (transformer.py:350-356,421)"
    )
    report["no_onnx_claim"] = (
        "L2 asserts the FIT/TEST encoder forward only. The hybrid passthrough is construction-time "
        "wiring of the composed v1 encoder; the encoder's ONNX path (torch-math backend + "
        "Split/Concat offsets) is norm_type-agnostic and already gated by the M2/M4 ONNX gates."
    )
    print(f"{'diff':<44}{'value':>14}")
    for name, val in diffs.items():
        print(f"{name:<44}{val:>14.3e}")
    _print_checks(checks)
    _print_verdict("l2", passed, criterion, _emit_report(report, outdir, "l2"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# L3 — VectorConcat order + Dsum + export.inputs alias (sub-wave B)
# ---------------------------------------------------------------------------


def run_l3(
    outdir: Path | str,
    *,
    corruption: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[int, dict[str, Any]]:
    """L3: `VectorConcat` order + ``Dsum`` + ``export.inputs alias:`` conformance.

    The GN3 ``global`` stream past the encoder (design §6.6). v1 has NO
    VectorConcat class — it inlines ``global_rep = cat([global_rep,
    global_feats], dim=-1)`` in ``SaltModel.forward`` (saltmodel.py:175-177) —
    so the anchors are DESIGN-CONFORMANCE, not v1-byte-parity (plan 10 Risks: the
    highest-uncertainty M5 item). Three checks, each with teeth:

    - **Concat ORDER + ``Dsum`` (FIT).** A GN3V01-style block (norm +
      norm_global -> track_embed -> concat -> encoder -> split -> pool ->
      VectorConcat[pooled.global, normed.global] -> head) runs through the REAL
      compiler / two-phase bind / executor. The VectorConcat output
      ``vconcat.global`` equals the LITERAL v1 line ``cat([pooled.global,
      normed.global], dim=-1)`` (`build_independent_v1_global_concat`, fed the
      SAME pooled rep + normalised global features the executor produced) —
      BITWISE (``torch.equal``; the v2 module issues the identical ``torch.cat``,
      so byte-equality is expected, the <= 1e-6 fallback recorded but unused).
      The width is ``Dsum = width(pooled.global) + width(normed.global)``
      (resolved at bind by `VectorConcat.derived_widths`, design §6.6 "Dsum
      unified at bind"). The ORDER is checked structurally: the LEADING columns
      equal ``pooled.global`` (pooled FIRST) and the TRAILING columns equal
      ``normed.global`` (global features LAST, design §6.6 1390-1391) — a
      reversed concat would fail this split even at the same Dsum.

    - **Alias name-resolution (Mode.ONNX — COMPILED + RUN).** The ONNX alias is
      an EXPORT concern (`OnnxAdapter`, design §6.6/§7): Athena feeds ONE jet
      tensor that v1 clones into ``global`` (to_onnx.py:377-378). This compiles a
      REAL ``Mode.ONNX`` plan with the GN3 alias (``inputs.global`` aliased from
      ``inputs.jets``) and RUNS the adapter — it never asserts a TEST-mode array
      as an ONNX proxy. The IDENTITY case (``global`` declares the jets columns)
      resolves a None gather (a clone) and the adapter's ONNX output equals the
      all-valid eager forward on the same jets/tracks (a single full-length jet,
      all-valid pad mask — the v1 export contract). The NAME-GATHER case
      (``global`` = the jets columns REVERSED) resolves an ``index_select`` whose
      indices are the by-NAME positions ([1, 0] for two reversed columns), and
      the aliased ``inputs.global`` actually fed into the concat equals
      ``inputs.jets`` gathered by those indices — proving the gather reorders
      columns by name, not by position.

    - **Loud construction/bind surfaces.** A duplicate-input VectorConcat and a
      self-feeding ``out in inputs`` VectorConcat both raise ``ConfigError`` at
      construction; an alias naming a column ABSENT from its source raises a
      ``ConfigError`` at adapter construction (`OnnxAdapter._resolve_alias_gather`).

    L3 DOES make an ONNX claim and backs it by compiling + running ``Mode.ONNX``.

    Negative control (``test_gates_m5.py``): the ``corruption`` hook perturbs the
    v2 OBSERVED ``vconcat.global`` before the parity comparison — the order/Dsum
    parity must FAIL while the alias-run and loud-surface checks (independent of
    the FIT concat values) stay green.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)`` — 0 only if every check passed.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("L3 VectorConcat order + Dsum + export.inputs alias conformance (design §6.6)")
    print("=" * 96)
    nd, cd = outdir / "norm_dict.yaml", outdir / "class_dict.yaml"
    outdir.mkdir(parents=True, exist_ok=True)
    write_vector_concat_norm_dict(nd, cd)
    checks: dict[str, bool] = {}
    diffs: dict[str, float] = {}

    # -- (a) FIT: order + Dsum vs the literal v1 cat line ----------------------
    modules = build_vector_concat_modules(nd)
    fit_plan = compile_vector_concat(modules, Mode.FIT)
    test_plan = compile_vector_concat(modules, Mode.TEST)
    # the ONNX plan participates in the SHARED schema so the alias source widths
    # resolve once (the adapter is built from the SAME bound modules below)
    onnx_plan, onnx_export, onnx_fields = compile_vector_concat_onnx(modules)
    schema = resolve_bind_schema([fit_plan, test_plan, onnx_plan])
    bind_all(modules, schema)
    materialise_all(modules)

    inputs, masks = make_gn2_batch(B, T)
    global_feats = torch.randn(B, len(GLOBAL_VARIABLES), generator=torch.Generator().manual_seed(7))
    b = Bundle()
    for stream, x in inputs.items():
        b.set(f"inputs.{stream}", x)
    b.set("masks.tracks", masks["tracks"])
    b.set("inputs.global", global_feats)
    for key, val in make_dips_labels(B).items():
        b.set(key, val)
    out = Executor(fit_plan).run(b, debug=True)
    pooled = out.get("pooled.global")
    normed_global = out.get("normed.global")
    v2_vconcat = out.get("vconcat.global")
    if corruption is not None:
        v2_vconcat = corruption(v2_vconcat)
    # v1 reference: the LITERAL saltmodel.py:175-177 cat, fed the SAME inputs
    v1_vconcat = build_independent_v1_global_concat(pooled, normed_global)
    diffs["vconcat_vs_v1_cat"] = _max_abs(v2_vconcat, v1_vconcat)
    checks["vconcat_bitwise_vs_v1_cat"] = torch.equal(v2_vconcat, v1_vconcat)
    checks["vconcat_parity_vs_v1_cat"] = diffs["vconcat_vs_v1_cat"] <= PARITY_ATOL
    d_pool, d_glob = pooled.shape[-1], normed_global.shape[-1]
    checks["dsum_is_input_width_sum"] = (
        schema.width("vconcat.global") == d_pool + d_glob == v2_vconcat.shape[-1]
    )
    # ORDER: pooled FIRST, global features LAST (design §6.6 1390-1391)
    checks["order_pooled_first"] = torch.equal(v2_vconcat[:, :d_pool], pooled)
    checks["order_global_last"] = torch.equal(v2_vconcat[:, d_pool:], normed_global)

    # -- (b) Mode.ONNX (COMPILED + RUN): alias identity + name-gather ----------
    jets = torch.randn(1, len(JET_VARIABLES), generator=torch.Generator().manual_seed(3))
    tracks = torch.randn(5, len(TRACK_VARIABLES), generator=torch.Generator().manual_seed(4))
    # IDENTITY: global declares the jets columns -> None gather (a clone). The
    # adapter's ONNX output must equal the all-valid eager TEST forward on the
    # same jets/tracks (the v1 export contract: a single full-length jet, an
    # all-valid pad mask, to_onnx.py:386-390).
    id_adapter = OnnxAdapter(onnx_plan, onnx_export, onnx_fields)
    # inspect the resolved alias gather (the export internal): None == identity
    # clone (test_onnx_adapter.py precedent for the same conformance check)
    checks["identity_alias_is_clone"] = id_adapter._alias_gathers == [None]  # noqa: SLF001 - alias-gather conformance probe
    with torch.no_grad():
        onnx_out = id_adapter(jets.clone(), tracks.clone())
    # the eager TEST forward on the same single jet (batched) — all tracks valid
    eager_b = Bundle()
    eager_b.set("inputs.jets", jets.clone())
    eager_b.set("inputs.tracks", tracks.unsqueeze(0).clone())
    eager_b.set("masks.tracks", torch.zeros(1, tracks.shape[0], dtype=torch.bool))
    eager_b.set("inputs.global", jets.clone())  # identity alias clones jets
    eager_out = Executor(test_plan).run(eager_b)
    eager_probs = eager_out.get("preds.jets.jets_classification")
    onnx_probs = torch.stack([o.reshape(-1)[0] for o in onnx_out])
    diffs["onnx_identity_vs_eager"] = _max_abs(onnx_probs, eager_probs.reshape(-1))
    checks["onnx_identity_runs_and_matches_eager"] = diffs["onnx_identity_vs_eager"] <= PARITY_ATOL
    # NAME-GATHER: global = the jets columns REVERSED. A SEPARATE module set so
    # the bound-once buffers are not reused across alias layouts.
    rev_cols = list(reversed(JET_VARIABLES))
    gather_modules = build_vector_concat_modules(nd)
    g_fit = compile_vector_concat(gather_modules, Mode.FIT)
    g_test = compile_vector_concat(gather_modules, Mode.TEST)
    g_onnx, g_export, g_fields = compile_vector_concat_onnx(
        gather_modules, alias_variables=rev_cols
    )
    bind_all(gather_modules, resolve_bind_schema([g_fit, g_test, g_onnx]))
    materialise_all(gather_modules)
    gather_adapter = OnnxAdapter(g_onnx, g_export, g_fields)
    (gather_index,) = gather_adapter._alias_gathers  # noqa: SLF001 - alias-gather conformance probe
    expected_index = [list(JET_VARIABLES).index(name) for name in rev_cols]
    checks["name_gather_index_resolves_by_name"] = (
        gather_index is not None and gather_index.tolist() == expected_index
    )
    # the aliased inputs.global fed to the concat == inputs.jets gathered by name
    with torch.no_grad():
        _ = gather_adapter(jets.clone(), tracks.clone())  # runs the gather in-trace
    gathered = jets.index_select(-1, gather_index)
    by_name = jets[:, expected_index]
    checks["name_gather_reorders_columns"] = torch.equal(gathered, by_name)
    # the gather is a genuine REORDER (reversed != identity for >=2 distinct cols)
    checks["name_gather_is_not_identity"] = not torch.equal(gathered, jets)

    # -- (c) loud construction / bind surfaces ---------------------------------
    raised_dup = False
    try:
        VectorConcat(inputs=["pooled.global", "pooled.global"], out="vconcat.global")
    except ConfigError:
        raised_dup = True
    checks["duplicate_inputs_raise"] = raised_dup
    raised_self = False
    try:
        VectorConcat(inputs=["pooled.global", "vconcat.global"], out="vconcat.global")
    except ConfigError:
        raised_self = True
    checks["self_feed_out_in_inputs_raises"] = raised_self
    # an alias whose TARGET (inputs.global) declares a column FOREIGN to the
    # source (inputs.jets) is a construction-time ConfigError in
    # OnnxAdapter._resolve_alias_gather (the by-name gather cannot resolve it).
    # Reuse the already-bound identity ONNX plan/export and feed a DOCTORED
    # feature_fields whose global column list contains a foreign name — the check
    # reads feature_fields, isolating the alias gather from the norm materialise.
    raised_missing_col = False
    doctored_fields = {**onnx_fields, "inputs.global": (JET_VARIABLES[0], "not_a_jet_column")}
    try:
        OnnxAdapter(onnx_plan, onnx_export, doctored_fields)
    except ConfigError:
        raised_missing_col = True
    checks["alias_missing_column_raises"] = raised_missing_col

    passed = all(checks.values())
    criterion = (
        "VectorConcat output (real compiler/bind/executor) equals the literal v1 "
        "cat([pooled.global, normed.global]) BITWISE — pooled first, global features last "
        "(design §6.6 1390-1391) — with Dsum = sum of input widths; the export.inputs alias "
        "compiles+runs a Mode.ONNX plan where the IDENTITY case clones (output == eager) and "
        "the REVERSED case resolves a by-NAME index_select that reorders columns; and duplicate/"
        "self-feed/missing-alias-column all raise ConfigError"
    )
    report = _base_report(
        "l3_vector_concat_alias",
        passed,
        criterion,
        {
            "atol": PARITY_ATOL,
            "B": B,
            "T": T,
            "d_pooled": d_pool,
            "d_global": d_glob,
            "dsum": d_pool + d_glob,
            "alias_identity_columns": list(JET_VARIABLES),
            "alias_name_gather_columns": rev_cols,
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["max_abs_diffs"] = diffs
    report["v1_reference"] = (
        "no v1 class — the literal cat([global_rep, global_feats], dim=-1) at "
        "saltmodel.py:175-177 (build_independent_v1_global_concat); the ONNX alias clones one "
        "Athena jet tensor into the global port (to_onnx.py:377-378), reproduced by OnnxAdapter"
    )
    report["onnx_claim"] = (
        "L3 compiles AND runs a Mode.ONNX plan through OnnxAdapter for BOTH the identity-clone "
        "and the by-name column-gather alias cases — it never asserts a TEST-mode array as an "
        "ONNX proxy (sub-wave-A gate-quality lesson)."
    )
    print(f"{'diff':<44}{'value':>14}")
    for name, val in diffs.items():
        print(f"{name:<44}{val:>14.3e}")
    _print_checks(checks)
    _print_verdict("l3", passed, criterion, _emit_report(report, outdir, "l3"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the gate subcommand parser (R1-R4, L1-L3).

    Returns
    -------
    argparse.ArgumentParser
        Parser with the ``r1``/``r2``/``r3``/``r4``/``l1``/``l2``/``l3``
        subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="python -m salt.core.gates_m5", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="gate", required=True)
    helps = {
        "r1": "RegressionTask forward + de-scaling parity (norm_params/ratio/scaler, mode-split)",
        "r2": "GaussianRegressionTask parity (mu/sigma/NLL/stddev) + ONNX-stddev conformance",
        "r3": "sample_weight + NaN-target masking loss parity",
        "r4": "MultiTarget processor conditional-replacement bitwise parity",
        "l1": "LossGLS geometric-mean parity + all-weights==1.0 guard (sub-wave B)",
        "l2": "TransformerEncoder norm_type:hybrid passthrough parity (sub-wave B)",
        "l3": "VectorConcat order + Dsum + export.inputs alias conformance (sub-wave B)",
    }
    for gate, help_text in helps.items():
        p = sub.add_parser(gate, help=help_text)
        p.add_argument("--outdir", type=Path, required=True, help="report output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run an M5 regression-family gate from the command line.

    Returns
    -------
    int
        0 if the gate passed, 1 otherwise.
    """
    args = _build_parser().parse_args(argv)
    runner = {
        "r1": run_r1,
        "r2": run_r2,
        "r3": run_r3,
        "r4": run_r4,
        "l1": run_l1,
        "l2": run_l2,
        "l3": run_l3,
    }[args.gate]
    code, _ = runner(args.outdir)
    return code


if __name__ == "__main__":
    sys.exit(main())
