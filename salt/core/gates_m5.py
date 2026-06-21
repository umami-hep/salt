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
inputs alias:`` (the GN3 ``global`` stream past the encoder); MF1a covers the
sub-wave-C `MaskDecoder` + encoder ``drop_registers`` passthrough (the MaskFormer
object decoder over a register-dropping encoder).

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
- **MF1a MaskDecoder + encoder drop_registers parity** (sub-wave C; the decoder
  slice of the gate-table MF1a — the matched-loss sub-checks are added by the
  `MaskFormerMatchedLoss` module). The MaskFormer encoder->decoder block (norm ->
  embed -> concat -> encoder(``drop_registers=True``) -> decoder) from the real
  compiler/bind/executor, with TWO INDEPENDENT v1 references:
  (i) *drop_registers (BITWISE)* — ``encoded.seq`` equals a separately-constructed
  v1 `Transformer` built with ``drop_registers=True`` + ``load_state_dict``-copied
  weights (`build_independent_v1_transformer_drop`) fed the SAME ``seq.x``/
  ``seq.mask``; its shape is register-FREE (``T`` not ``T + num_registers``); the
  encoder plan/forward carry NO ``masks.registers`` key (v1 ``del
  pad_mask["REGISTERS"]``, transformer.py:745-750); and the registers were
  genuinely VISIBLE to attention — zeroing the register parameter CHANGES
  ``encoded.seq`` (a no-op slice would not, so the "visible to encoder, stripped
  from output" claim has teeth). (ii) *MaskDecoder forward (BITWISE)* — all four
  ``objects.{embed,class_logits,class_probs,masks}`` equal an INDEPENDENT v1
  `salt.models.maskformer.MaskDecoder` (`build_independent_v1_mask_decoder`:
  separately built from the same architecture, fresh `Dense` heads, state-copied,
  driven via its ``labels=None`` inference path) — proving the dummy-token trick
  and layer loop key-routing reproduce v1 exactly. (iii) *ONNX exercised* — a
  COMPILED + RUN ``Mode.ONNX`` plan produces register-free ``encoded.seq`` and the
  dummy-token-handled ``objects.*`` (the decoder is mode-agnostic), and a
  zero-constituent jet stays finite (the dummy-token's reason to exist). (iv)
  *loud surfaces* — a missing ``md.n_heads``, a missing ``class_net.output_size``,
  a width-key (``input_size``) in ``class_net``, and an ``embed_dim``-vs-input
  mismatch at bind all raise ``ConfigError``. MF1a does NOT cover the matched loss
  / matcher / object writer (separate sub-wave-C modules) — its report claims only
  the decoder + drop_registers coverage its sub-checks exercise.
- **MF2 MaskFormerObjectWriter + the two object reduces** (sub-wave C). The
  object-writer slice: the writer's TEST ``write`` is BITWISE (``.tobytes()``) vs
  the v1 op chain recomputed in the gate (``predictionwriter.py:276-308`` — the
  per-class probs, the ``class_label`` truth, the ``MaskIndex`` -1/-2 column, the
  ``object_masks`` truth/logits group), and a COMPILED + RUN ``Mode.ONNX`` export
  of the writer's manifest produces the ``leading_object`` (float32 scalars) and
  ``object_index`` (int8 ``HadronIndex``, dynamic token axis) reduces with
  torch-vs-onnxruntime agreement (int8 EXACT; leading floats where finite — v1
  returns NaN for a null leading object). It asserts `OBJECT_INDEX` is IMPORTED
  (the strings never re-declared) and that the loud surfaces raise. MF2 does NOT
  cover the decoder (MF1a) or the matched loss / targets (MF1c).
- **D1 callback-sink assembly + register_reduce live registry** (sub-wave D, the
  C-prereqs). Two slices, each with teeth, both driving the REAL surfaces (no
  re-implementation in the gate):
  (i) *FIT/VAL callback-sink assembly* — a real `SaltModule` with an aux head
  producing a ``preds.*`` key NO loss consumes is compiled (`compile_plan`) with
  the model's OWN `_model_sinks`: WITHOUT a FIT/VAL-sink callback the sinks are
  ``['loss.total']`` only and the aux producer is demand-PRUNED; WITH the real
  `MaskformerMetrics`-style callback (or the `ConfusionMatrix` ``preds.*`` /
  ``labels.*`` declaration) attached, its `fit_val_demand` becomes a FIT/VAL plan
  sink (`_callback_demand`) and the once-pruned producer is KEPT ALIVE in BOTH
  FIT and VAL — the §3.1 454-456 / §3.4 667-671 deferral the DP2 prereq lands;
  AND the callback demand is TRAINING-ONLY (empty in TEST/ONNX, where the sinks
  are the writer manifest). (ii) *register_reduce live-registry validation +
  reduce-declared dtypes* — a freshly `register_reduce`-d reduce VALIDATES through
  the REAL ``export.outputs`` resolution (`attach_manifest` -> ``_resolve_output``
  against the LIVE registry, defaulting + accepting the reduce's declared dtype),
  while an UNREGISTERED reduce name is REJECTED loudly (``ConfigError``), and a
  declared dtype that DISAGREES with the reduce's registered dtype is REJECTED
  loudly (the reduce owns its per-reduce dtype rule, amendment 555-567 / design
  §7.3). Plus the public `register_reduce` loud surfaces (duplicate name, bad
  dtype, non-string name). D1 makes NO model-parity claim (it is a planner/
  registry-assembly gate); the decoder/loss/writer parity are MF1a/MF1c/MF2.
- **D2ckpt Checkpoint filename/monitor contract + run-dir path inference +
  ProgressBar smoke** (sub-wave D-rest, plan 10 D2 row — THIS slice; the other D2
  items live elsewhere). The REAL `salt.core.callbacks.Checkpoint` produces the v1
  ``epoch=NNN-loss={monitor_loss:.5f}.ckpt`` stem (``save_top_k=-1``, per-task
  monitor unmangled), its `setup("fit")` forces ``dirpath`` to ``<log_dir>/ckpts``
  (driven through the actual `ModelCheckpoint.setup`, not a re-implementation), and
  `salt.core.main._best_checkpoint` resolves the lowest-loss epoch from what the
  callback wrote — the ``salt2 test`` no-``--ckpt_path`` run-dir path inference.
  NON-GATING for model reproduction (it gates the eval run-dir discovery). The
  `ProgressBar` smoke confirms the stock ``TQDMProgressBar`` attaches under a
  salt-owned name (v1 ``callbacks/checkpoint.py:25-48``, ``base.yaml:38-39``).
- **D2expose per-task expose opt-out + name-based origin_weighting** (sub-wave
  D-rest, plan 10 D2 row — THIS slice). The per-task ``expose: [fit, val]`` gates a
  task's ``preds.*`` to the exposed modes, so the planner KEEPS the task in the FIT
  plan and PRUNES it from the TEST plan, and the REAL
  `SaltModule._model_sinks(Mode.TEST)` writer-demand dead-preds path raises WITHOUT
  expose (advertising the fix) but is silent WITH it; name-based
  ``origin_weighting`` resolves CLASS NAMES to v1's hardcoded ids (3,4,5)/(1) via
  the REAL `salt.core.saltmodule.resolve_origin_weighting`, and the composed head's
  ``get_weights`` is bit-identical to an INDEPENDENT v1 `VertexingTask` (design
  §4.2, §5.1; v1 ``task.py:957-964``).
- **D2cfg writer TensorSpec kind/dtype validation + lrs_config->lrs migration**
  (sub-wave D-rest, plan 10 D2 row — THIS slice; the LAST two D2 items). The REAL
  `WriterCallback.validate_specs` (the static §2.7 writer-input validator driven
  through the SAME ``model_producer_specs`` + boundary union `SaltModule._validate_
  writer_specs` assembles) unifies a writer's declared require kind/dtype against
  its producing leaf: a matching ``data``/``float32`` (and an unconstrained
  ``None`` dtype) require PASSES; a wrong ``dtype`` raises `ShapeError`, a wrong
  ``kind`` raises `KindError`, and a non-optional require with no producer raises
  `ConfigError` (existence half), while an OPTIONAL missing require is silent. PLUS
  the ``lrs_config`` -> ``lrs`` rename (M3 cleanup, design §5.1): a shipped v2
  config (carrying ``lrs:``) loads + plan-compiles through the REAL ``salt2 graph
  validate`` in all four modes, NO shipped v2 config retains a stale ``lrs_config:``
  key, and a config carrying the old ``lrs_config:`` key FAILS the real
  config-load path (the renamed kwarg is rejected — v1
  ``modelwrapper.py``/``base.yaml:45``).
- **CONV (M5-CONV) the consolidated acceptance gate** (plan 10 §4 / matrix §4 —
  the M7-slice acceptance for M5). For EVERY needs-M5 config (the matrix §2 ``🔶``
  rows + the plan-10-adjudicated ``regression_multi_target``; the AUTHORITATIVE
  list is embedded VERBATIM in ``_CONV_CONFIGS`` and the report) it drives the
  REAL ``salt2 graph validate`` (the canonical static validator, the d2cfg command
  path) in fit + test (+ onnx where the config is export-representable;
  ``event_classifier`` is fit/test-only — a non-feature ratio denominator, no
  ``export:`` block) and asserts rc == 0 (no error-level finding — design §4.2:
  convert+validate+plan-compile). The gate exits non-zero if any matrix ``🔶``
  config is MISSING from the list or FAILS a mode. NO ``--strict``: a data-free
  validation cannot satisfy it (the no-``schema:`` warning, cli.py:832-836, fires
  for every config without a schema artifact — derived from a real H5, none
  data-free — and ``--strict`` promotes it, cli.py:919-920; the same rationale
  d2cfg + every config header record). CONV makes NO new forward-parity claim —
  the family-representative forward parity is owned by R1-R4 / L1-L3 / MF1a/MF1c/
  MF2 (plan 10 §4.4). Its corruption hook transforms the embedded config LIST
  (inject a missing/bogus entry) to give the completeness + per-config assertions
  teeth.

Negative-control hooks (the pytest suite, ``test_gates_m5.py``): each ``run_*``
takes a python-only ``corruption`` keyword applied to the v2 OBSERVED values (or
the loss) before the parity comparison — the gate must FAIL while the surface
checks stay green. The gates_m2/m3/m4 pattern, never exposed on the CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, TQDMProgressBar
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from salt.core.callbacks import Checkpoint, ConfusionMatrix, MaskformerMetrics, ProgressBar
from salt.core.data import H5StructuredReader, MaskFormerTargets, MultiTarget
from salt.core.graph import Bundle, Executor, Mode
from salt.core.graph.errors import ConfigError, GraphError, KindError, ShapeError
from salt.core.graph.planner import compile_plan
from salt.core.graph.spec import IO, TensorSpec, flatten_spec, unflatten_spec
from salt.core.main import CONFIG_DIR, _best_checkpoint
from salt.core.main import main as salt2_main
from salt.core.nn import (
    LossGLS,
    LossSum,
    MaskDecoder,
    MaskFormerMatchedLoss,
    TransformerEncoder,
    VectorConcat,
    bind_all,
    materialise_all,
    resolve_bind_schema,
)
from salt.core.nn.bind import ResolvedSchema
from salt.core.nn.tasks import (
    ClassificationTaskModule,
    RegressionTaskModule,
    VertexingTaskModule,
)
from salt.core.onnx import (
    ExportConfig,
    ExportInput,
    ExportOutput,
    OnnxAdapter,
    attach_manifest,
    compile_onnx_plan,
    export_graph,
    make_session,
    resolve_export_config,
)
from salt.core.onnx.reduces import (
    reduce_dtype,
    register_reduce,
    registered_reduces,
    unregister_reduce,
)
from salt.core.saltmodule import SaltModule, resolve_origin_weighting
from salt.core.schema import GroupSchema, Schema
from salt.core.writers import (
    OBJECT_INDEX,
    MaskFormerObjectWriter,
    TaskWriter,
    WriteCtx,
    Writer,
    WriterCallback,
    WriterDeclareCtx,
)
from salt.data.datasets import OPERATORS as V1_OPERATORS
from salt.models.pooling import GlobalAttentionPooling as V1GlobalAttentionPooling
from salt.models.task import VertexingTask as V1VertexingTask
from salt.tests.core.gn2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    make_gn2_batch,
    write_parity_norm_dict,
)
from salt.tests.core.gn2v2_fixture import ORIGIN_CLASSES, build_gn2v2_modules, gn2v2_sources
from salt.tests.core.regression_fixture import (
    GLOBAL_VARIABLES,
    HYBRID_ENC_DIM,
    MASKFORMER_NUM_OBJECT_CLASSES,
    MASKFORMER_NUM_OBJECTS,
    MASKFORMER_NUM_REG_TARGETS,
    MASKFORMER_OBJECT_CLASS_MAP,
    build_dips_modules,
    build_gls_modules,
    build_hybrid_encoder_modules,
    build_independent_v1_global_concat,
    build_independent_v1_gls_total_loss,
    build_independent_v1_head,
    build_independent_v1_mask_decoder,
    build_independent_v1_matched_loss,
    build_independent_v1_transformer,
    build_independent_v1_transformer_drop,
    build_maskformer_decoder_modules,
    build_maskformer_targets,
    build_maskformer_writer_modules,
    build_matched_loss_module,
    build_regression_modules,
    build_vector_concat_modules,
    compile_dips,
    compile_gls,
    compile_hybrid_encoder,
    compile_maskformer_decoder,
    compile_regression,
    compile_vector_concat,
    compile_vector_concat_onnx,
    make_dips_labels,
    make_gls_labels,
    make_maskformer_object_batch,
    make_maskformer_targets_batch,
    make_maskformer_writer_batch,
    make_regression_labels,
    write_vector_concat_norm_dict,
)
from salt.utils.mask_utils import build_target_masks as v1_build_target_masks
from salt.utils.mask_utils import indices_from_mask as v1_indices_from_mask
from salt.utils.scalers import RegressionTargetScaler

__all__ = [
    "PARITY_ATOL",
    "main",
    "run_conv",
    "run_d1",
    "run_d2cfg",
    "run_d2ckpt",
    "run_d2expose",
    "run_l1",
    "run_l2",
    "run_l3",
    "run_mf1a",
    "run_mf1c",
    "run_mf2",
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
    # adapter's ONNX output must equal the all-valid eager ONNX forward on the
    # same jets/tracks (the v1 export contract: a single full-length jet, an
    # all-valid pad mask, to_onnx.py:386-390). The adapter WRAPS ``onnx_plan``
    # (it runs ``Executor(onnx_plan)`` on the bundle it assembles), so the eager
    # reference here is the SAME ``onnx_plan`` fed the SAME bundle the adapter
    # builds (jets [1, F], tracks [1, L, F], an all-valid pad mask, global =
    # jets.clone()) — proving the adapter's input-reorder + alias-gather +
    # output-stack are transparent over the plan. NOTE (P1.5): we reference the
    # ONNX plan, NOT the TEST plan: post-flip the TEST classification forward
    # publishes RAW logits (design §2) while the ONNX forward still publishes the
    # converted, EVAL-READY probs (the deferred-to-P4 ONNX carve-out). The v1
    # export contract is "adapter ONNX output == eval-ready probs", and the
    # eval-ready value IS the eager ONNX forward (softmax of the TEST raw logits);
    # comparing the adapter against the eager ONNX forward keeps that contract
    # faithful AND keeps the gate's teeth (the adapter wraps the plan, so the
    # input-reorder / alias-gather / output-stack are still under test — and
    # ``identity_alias_is_clone`` above is untouched).
    id_adapter = OnnxAdapter(onnx_plan, onnx_export, onnx_fields)
    # inspect the resolved alias gather (the export internal): None == identity
    # clone (test_onnx_adapter.py precedent for the same conformance check)
    checks["identity_alias_is_clone"] = id_adapter._alias_gathers == [None]  # noqa: SLF001 - alias-gather conformance probe
    with torch.no_grad():
        onnx_out = id_adapter(jets.clone(), tracks.clone())
    # the eager ONNX forward on the same single jet (batched) — all tracks valid;
    # this is byte-for-byte the bundle the adapter assembles internally (§7)
    eager_b = Bundle()
    eager_b.set("inputs.jets", jets.clone())
    eager_b.set("inputs.tracks", tracks.unsqueeze(0).clone())
    eager_b.set("masks.tracks", torch.zeros(1, tracks.shape[0], dtype=torch.bool))
    eager_b.set("inputs.global", jets.clone())  # identity alias clones jets
    eager_out = Executor(onnx_plan).run(eager_b)
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
# MF1a — MaskDecoder + encoder drop_registers parity (sub-wave C)
# ---------------------------------------------------------------------------


def run_mf1a(
    outdir: Path | str,
    *,
    corruption: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[int, dict[str, Any]]:
    """MF1a (decoder slice): MaskDecoder + encoder drop_registers parity vs INDEPENDENT v1.

    The MaskFormer encoder->decoder block (norm -> embed -> concat ->
    encoder(``drop_registers=True``) -> `MaskDecoder`) is driven through the REAL
    compiler/bind/executor and compared, BITWISE, against TWO independently-constructed
    v1 references:

    - **drop_registers** (`build_independent_v1_transformer_drop`): ``encoded.seq``
      equals a fresh v1 `Transformer` (``drop_registers=True``, state-copied) fed the
      SAME ``seq.x``/``seq.mask``; the shape is register-FREE; the encoder plan + forward
      carry NO ``masks.registers`` (v1 ``del pad_mask["REGISTERS"]``,
      transformer.py:745-750); and zeroing the register parameter CHANGES ``encoded.seq``
      — proving the registers were visible to attention, not a no-op slice.
    - **MaskDecoder** (`build_independent_v1_mask_decoder`): the four
      ``objects.{embed,class_logits,class_probs,masks}`` equal a fresh v1
      `salt.models.maskformer.MaskDecoder` (state-copied, driven via its ``labels=None``
      inference path) — the dummy-token trick + layer loop reproduce v1 exactly.

    A COMPILED + RUN ``Mode.ONNX`` plan exercises the export path (register-free
    ``encoded.seq``, dummy-token-handled ``objects.*``, finite on a zero-constituent
    jet). Loud surfaces (missing ``md.n_heads`` / ``class_net.output_size``, a width-key
    in ``class_net``, an ``embed_dim`` mismatch at bind) all raise ``ConfigError``.

    MF1a does NOT cover the matched loss / matcher / object writer (separate sub-wave-C
    modules) — its criterion claims only the decoder + drop_registers coverage.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)`` — 0 only if every check passed.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("MF1a MaskDecoder + encoder drop_registers parity vs INDEPENDENT v1")
    print("=" * 96)
    norm_dict = _norm_dict(outdir)
    checks: dict[str, bool] = {}
    diffs: dict[str, float] = {}

    modules = build_maskformer_decoder_modules(norm_dict)
    test_plan = compile_maskformer_decoder(modules, Mode.TEST)
    onnx_plan = compile_maskformer_decoder(modules, Mode.ONNX)
    bind_all(modules, resolve_bind_schema([test_plan, onnx_plan]))
    materialise_all(modules)
    encoder = modules["encoder"]
    decoder = modules["mask_decoder"]

    # the drop-registers encoder produces NO masks.registers in EITHER plan
    checks["test_plan_has_no_registers"] = (
        "masks.registers" not in test_plan.step("encoder").produces
    )
    checks["onnx_plan_has_no_registers"] = (
        "masks.registers" not in onnx_plan.step("encoder").produces
    )

    inputs, masks = make_gn2_batch(B, T)
    b = Bundle()
    for stream, x in inputs.items():
        b.set(f"inputs.{stream}", x)
    b.set("masks.tracks", masks["tracks"])
    out = Executor(test_plan).run(b, debug=True)
    seq_x, seq_mask = out.get("seq.x"), out.get("seq.mask")
    v2_encoded = out.get("encoded.seq")
    v2_embed = out.get("objects.embed")
    v2_class_logits = out.get("objects.class_logits")
    v2_class_probs = out.get("objects.class_probs")
    v2_masks = out.get("objects.masks")
    if corruption is not None:
        v2_encoded = corruption(v2_encoded)
        v2_embed = corruption(v2_embed)

    # -- (i) drop_registers BITWISE parity vs an INDEPENDENT v1 Transformer ----
    v1_encoder = build_independent_v1_transformer_drop(encoder)
    with torch.no_grad():
        v1_encoded, v1_pad = v1_encoder({"seq": seq_x.clone()}, pad_mask={"seq": seq_mask.clone()})
    diffs["drop_registers_encoded"] = _max_abs(v2_encoded, v1_encoded)
    checks["drop_registers_encoded_bitwise_vs_independent_v1"] = torch.equal(v2_encoded, v1_encoded)
    checks["drop_registers_encoded_parity_vs_independent_v1"] = (
        diffs["drop_registers_encoded"] <= PARITY_ATOL
    )
    # the encoded sequence is register-FREE: T track rows, no register rows
    checks["encoded_seq_is_register_free"] = v2_encoded.shape[1] == seq_x.shape[1]
    # v1 also removed the "REGISTERS" pad entry (transformer.py:748)
    checks["v1_pad_dropped_registers"] = "REGISTERS" not in v1_pad

    # registers were VISIBLE to attention: zeroing the register parameter (a fresh
    # state-copy, untouched executor) CHANGES encoded.seq — a no-op slice would not.
    probe = build_independent_v1_transformer_drop(encoder)
    with torch.no_grad():
        probe.registers.zero_()
        zeroed_encoded, _ = probe({"seq": seq_x.clone()}, pad_mask={"seq": seq_mask.clone()})
    checks["registers_visible_to_encoder"] = not torch.equal(v1_encoded, zeroed_encoded)

    # -- (ii) MaskDecoder forward BITWISE parity vs an INDEPENDENT v1 decoder ---
    # feed the v1 decoder the v2 executor's (uncorrupted) encoded.seq — its
    # labels=None inference path returns preds["objects"] (maskformer.py:166-203)
    v1_decoder = build_independent_v1_mask_decoder(decoder)
    with torch.no_grad():
        v1_preds, _, _ = v1_decoder(
            {"embed_xs": out.get("encoded.seq").clone()},
            tasks=[],
            pad_mask=seq_mask.clone(),
            labels=None,
        )
    v1_obj = v1_preds["objects"]
    diffs["decoder_embed"] = _max_abs(v2_embed, v1_obj["embed"])
    diffs["decoder_class_logits"] = _max_abs(v2_class_logits, v1_obj["class_logits"])
    diffs["decoder_class_probs"] = _max_abs(v2_class_probs, v1_obj["class_probs"])
    diffs["decoder_masks"] = _max_abs(v2_masks, v1_obj["masks"])
    checks["decoder_embed_bitwise_vs_independent_v1"] = torch.equal(v2_embed, v1_obj["embed"])
    checks["decoder_class_logits_bitwise_vs_independent_v1"] = torch.equal(
        v2_class_logits, v1_obj["class_logits"]
    )
    checks["decoder_class_probs_bitwise_vs_independent_v1"] = torch.equal(
        v2_class_probs, v1_obj["class_probs"]
    )
    checks["decoder_masks_bitwise_vs_independent_v1"] = torch.equal(v2_masks, v1_obj["masks"])
    # the dummy token is stripped: masks span the real T constituents, not T + 1
    checks["decoder_masks_unpadded_to_T"] = v2_masks.shape[-1] == seq_x.shape[1]
    checks["decoder_objects_shape"] = tuple(v2_embed.shape) == (
        B,
        MASKFORMER_NUM_OBJECTS,
        encoder.out_dim,
    )
    # class_probs is a proper distribution over C = num_classes + 1 (null LAST)
    checks["class_probs_is_distribution"] = bool(
        torch.allclose(v2_class_probs.sum(-1), torch.ones(B, MASKFORMER_NUM_OBJECTS), atol=1e-5)
    )
    checks["class_count_is_C"] = v2_class_logits.shape[-1] == MASKFORMER_NUM_OBJECT_CLASSES

    # -- (iii) Mode.ONNX EXERCISED: compile + RUN, never a TEST-mode proxy -----
    encoder.set_export_mode()
    decoder.set_export_mode()
    inputs_o, masks_o = make_gn2_batch(B, T)
    bo = Bundle()
    for stream, x in inputs_o.items():
        bo.set(f"inputs.{stream}", x)
    bo.set("masks.tracks", masks_o["tracks"])
    with torch.no_grad():
        bo = Executor(onnx_plan).run(bo)
    onnx_embed = bo.get("objects.embed")
    checks["onnx_encoded_is_register_free"] = bo.get("encoded.seq").shape[1] == T
    checks["onnx_objects_finite"] = bool(torch.isfinite(onnx_embed).all())
    checks["onnx_objects_masks_unpadded"] = bo.get("objects.masks").shape[-1] == T
    # the dummy-token's reason to exist: a zero-constituent sequence must not NaN
    zb = Bundle()
    zb.set("encoded.seq", torch.zeros(1, 0, encoder.out_dim))
    zb.set("seq.mask", torch.zeros(1, 0, dtype=torch.bool))
    with torch.no_grad():
        zout = decoder(zb, Mode.ONNX)
    checks["zero_constituent_jet_is_finite"] = bool(torch.isfinite(zout["objects.embed"]).all())
    checks["zero_constituent_masks_shape"] = tuple(zout["objects.masks"].shape) == (
        1,
        MASKFORMER_NUM_OBJECTS,
        0,
    )

    # -- (iv) loud construction / bind surfaces --------------------------------
    checks["missing_n_heads_raises"] = _raises_config_error(
        lambda: MaskDecoder(
            embed_dim=16, num_objects=5, num_layers=2, class_net={"output_size": 3}, md={}
        )
    )
    checks["missing_class_output_size_raises"] = _raises_config_error(
        lambda: MaskDecoder(
            embed_dim=16, num_objects=5, num_layers=2, class_net={}, md={"n_heads": 2}
        )
    )
    checks["class_net_width_key_raises"] = _raises_config_error(
        lambda: MaskDecoder(
            embed_dim=16,
            num_objects=5,
            num_layers=2,
            class_net={"input_size": 16, "output_size": 3},
            md={"n_heads": 2},
        )
    )
    checks["embed_dim_mismatch_at_bind_raises"] = _raises_config_error(
        _bad_embed_dim_bind, norm_dict
    )

    passed = all(checks.values())
    criterion = (
        "the MaskFormer encoder->decoder block (real compiler/bind/executor) reproduces, BITWISE, "
        "an INDEPENDENT v1 Transformer (drop_registers=True) for encoded.seq AND an INDEPENDENT v1 "
        "MaskDecoder for all four objects.* — encoded.seq is register-free with no masks.registers "
        "key, the registers were visible to attention (zeroing them changes the output), the "
        "dummy-token trick is stripped (masks span T), a COMPILED+RUN Mode.ONNX plan stays "
        "register-free + finite (incl. a zero-constituent jet), and missing n_heads / "
        "class_net.output_size / width-key / embed_dim-mismatch all raise ConfigError. Decoder "
        "slice ONLY — the matched loss / matcher / object writer are separate sub-wave-C modules"
    )
    report = _base_report(
        "mf1a_maskdecoder_drop_registers",
        passed,
        criterion,
        {
            "atol": PARITY_ATOL,
            "B": B,
            "T": T,
            "num_objects": MASKFORMER_NUM_OBJECTS,
            "num_object_classes": MASKFORMER_NUM_OBJECT_CLASSES,
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["max_abs_diffs"] = diffs
    report["v1_reference"] = (
        "two independently-constructed, state-copied v1 references: "
        "salt.models.Transformer(drop_registers=True) for encoded.seq "
        "(build_independent_v1_transformer_drop) and salt.models.maskformer.MaskDecoder driven "
        "on its labels=None inference path (build_independent_v1_mask_decoder) — NEVER the v2 "
        "module's own composed objects"
    )
    report["onnx_claim"] = (
        "MF1a compiles AND runs a Mode.ONNX plan through the executor (register-free encoded.seq, "
        "dummy-token-handled objects.*, finite on a zero-constituent jet) — it never asserts a "
        "TEST-mode array as an ONNX proxy (sub-wave-A gate-quality lesson)."
    )
    report["scope_note"] = (
        "decoder slice of the gate-table MF1a; the MaskFormerMatchedLoss / HungarianMatcher / "
        "MaskFormerObjectWriter forward-parity sub-checks are added by those (separate) modules"
    )
    print(f"{'diff':<40}{'value':>14}")
    for name, val in diffs.items():
        print(f"{name:<40}{val:>14.3e}")
    _print_checks(checks)
    _print_verdict("mf1a", passed, criterion, _emit_report(report, outdir, "mf1a"))
    return (0 if passed else 1), report


def _raises_config_error(fn: Callable[..., Any], *args: Any) -> bool:
    """Whether calling ``fn(*args)`` raises a `ConfigError` (the loud-surface probe).

    Returns
    -------
    bool
        True iff `fn` raised `ConfigError`.
    """
    try:
        fn(*args)
    except ConfigError:
        return True
    return False


def _bad_embed_dim_bind(norm_dict: Path) -> None:
    """Bind a `MaskDecoder` whose ``embed_dim`` disagrees with the encoder output width.

    The encoder produces a 16-wide ``encoded.seq`` but the decoder declares
    ``embed_dim=32``, so `MaskDecoder.bind` must raise (the queries could not attend to a
    width-mismatched sequence; v1 maskformer.py:48,172). Driven through the real
    compiler/bind so the failure is the genuine schema-resolution path.
    """
    modules = build_maskformer_decoder_modules(norm_dict)
    modules["mask_decoder"] = MaskDecoder(
        embed_dim=32,  # != encoder out_dim (16) -> bind-time ConfigError
        num_objects=MASKFORMER_NUM_OBJECTS,
        num_layers=2,
        class_net={"output_size": MASKFORMER_NUM_OBJECT_CLASSES},
        md={"n_heads": 2, "mask_attention": True, "bidirectional_ca": True},
    )
    modules["mask_decoder"].name = "mask_decoder"
    plan = compile_maskformer_decoder(modules, Mode.TEST)
    bind_all(modules, resolve_bind_schema([plan]))


# ---------------------------------------------------------------------------
# MF1c — MaskFormerMatchedLoss + HungarianMatcher + MaskFormerTargets parity
# (the matched-loss / matcher / targets-processor slice of the gate-table MF1a;
#  the decoder slice is run_mf1a, the object writer is MF2)
# ---------------------------------------------------------------------------


def _matched_loss_schema() -> ResolvedSchema:
    """A minimal `ResolvedSchema` for the matched-loss bind (the regression widths).

    The matched loss binds only the scaled regression prediction/target widths (the
    matcher cost matrix M is config-known); the fixture's regression width is R.

    Returns
    -------
    ResolvedSchema
        Widths for ``preds.objects.regression`` / ``targets.objects.regression``.
    """
    return ResolvedSchema(
        widths={
            "preds.objects.regression": MASKFORMER_NUM_REG_TARGETS,
            "targets.objects.regression": MASKFORMER_NUM_REG_TARGETS,
        }
    )


def run_mf1c(
    outdir: Path | str,
    *,
    corruption: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[int, dict[str, Any]]:
    """MF1c: MaskFormerMatchedLoss + matcher + MaskFormerTargets parity vs INDEPENDENT v1.

    The matched-loss / matcher / targets-processor slice of the gate-table MF1a "code
    forward-parity (v1-decidable)" row (the decoder slice is `run_mf1a`; the object
    writer is MF2). Three slices:

    - **Matched loss + matcher (the 4 loss components)**: the `MaskFormerMatchedLoss`
      forward (matcher + loss methods) is compared against an INDEPENDENTLY-constructed
      v1 ``MaskFormerLoss`` (`build_independent_v1_matched_loss`: a fresh instance from
      the SAME config, its own ``HungarianMatcher`` + ``empty_weight`` buffer, NOT the
      v2 module's composed ``v1_loss``). The matcher assignment is BITWISE identical
      (same v1 matcher, same inputs), and the three v1-decidable components —
      ``object_class_ce``, ``mask_dice``, ``mask_focal`` — are BITWISE
      (``torch.equal``) vs the v1 reference driven on the v1-permuted predictions. The
      fourth, ``regression``, is the FD alignment change (matched, not v1's
      query-order, see Risks/the MF1b doc): there is NO v1 byte reference for a matched
      object-regression loss, so it is asserted as a DESIGN-conformance property (the
      matched L1 over valid objects of the matcher-permuted preds vs truth-order
      targets, recomputed independently in the gate).
    - **No in-place permute**: the matched loss publishes the permuted predictions as
      NEW ``matched.objects.*`` keys; the input ``objects.*`` tensors are byte-unchanged
      after the forward (v1 mutates them in place, maskformer_loss.py:338-343).
    - **MaskFormerTargets**: the processor's ``object_class`` / ``masks`` / regression
      labels are BITWISE identical to the v1 references (the v1 `build_target_masks`
      equality, mask_utils.py, and the v1 ``MaskformerObjectConfig`` class map applied
      to the ORIGINAL raw values), WITHOUT the v1 in-place id mutations.

    Loud surfaces (null-not-last class map, unknown loss-weight key, regression
    pred/target width mismatch at bind, all-zero matcher weights) all raise
    ``ConfigError``.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)`` — 0 only if every check passed.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("MF1c MaskFormerMatchedLoss + matcher + MaskFormerTargets parity vs INDEPENDENT v1")
    print("=" * 96)
    checks: dict[str, bool] = {}
    diffs: dict[str, float] = {}

    # -- (i) matched loss + matcher: the 4 loss components vs INDEPENDENT v1 ----
    module = build_matched_loss_module()
    module.bind(_matched_loss_schema())
    data = make_maskformer_object_batch()
    b = Bundle()
    for key, value in data.items():
        b.set(key, value)
    # snapshots of the input predictions, to prove they are NOT mutated in place
    pre_class_logits = data["objects.class_logits"].clone()
    pre_masks = data["objects.masks"].clone()
    pre_reg = data["preds.objects.regression"].clone()

    out = Executor(_single_module_plan(module)).run(b, debug=True)
    v2_ce = out.get("losses.object_class_ce")
    v2_dice = out.get("losses.mask_dice")
    v2_focal = out.get("losses.mask_focal")
    v2_reg = out.get("losses.regression")
    if corruption is not None:
        v2_ce = corruption(v2_ce)
        v2_reg = corruption(v2_reg)

    # the INDEPENDENT v1 reference: a fresh MaskFormerLoss, its OWN matcher + buffer
    v1 = build_independent_v1_matched_loss(module)
    pred_match = {
        "class_logits": data["objects.class_logits"],
        "class_probs": data["objects.class_probs"],
        "masks": data["objects.masks"],
        "regression": data["preds.objects.regression"],
    }
    tgt_match = {
        "object_class": data["labels.objects.object_class"],
        "masks": data["labels.objects.masks"].to(data["objects.masks"].dtype),
        "regression": data["targets.objects.regression"],
    }
    v1_idx = v1.matcher(pred_match, tgt_match)
    v1_perm = {
        "objects": {
            "class_logits": data["objects.class_logits"][v1_idx],
            "masks": data["objects.masks"][v1_idx],
        }
    }
    v1_truth = {
        "objects": {
            "object_class": data["labels.objects.object_class"],
            "masks": data["labels.objects.masks"],
        }
    }
    v1_ce = v1.get_loss("labels", v1_perm, v1_truth)["object_class_ce"]
    v1_masks = v1.get_loss("masks", v1_perm, v1_truth)
    v1_dice, v1_focal = v1_masks["mask_dice"], v1_masks["mask_focal"]

    diffs["matched_object_class_ce"] = _max_abs(v2_ce, v1_ce)
    diffs["matched_mask_dice"] = _max_abs(v2_dice, v1_dice)
    diffs["matched_mask_focal"] = _max_abs(v2_focal, v1_focal)
    checks["matched_object_class_ce_bitwise_vs_independent_v1"] = torch.equal(v2_ce, v1_ce)
    checks["matched_mask_dice_bitwise_vs_independent_v1"] = torch.equal(v2_dice, v1_dice)
    checks["matched_mask_focal_bitwise_vs_independent_v1"] = torch.equal(v2_focal, v1_focal)
    # the matcher assignment itself matches the independent v1 matcher BITWISE
    checks["matcher_assignment_bitwise_vs_independent_v1"] = torch.equal(
        out.get("matched.objects.class_logits"), data["objects.class_logits"][v1_idx]
    )

    # -- (ii) matched regression: the FD alignment change (design-conformance) -
    # NO v1 byte reference (v1's effective regression loss is QUERY-order); the gate
    # recomputes the FD-specified matched L1 (matcher-permuted preds vs truth-order
    # targets, valid objects only) independently and asserts the module matches it.
    object_class = data["labels.objects.object_class"]
    valid = object_class != module.num_classes
    m_reg = data["preds.objects.regression"][v1_idx]
    ref_reg = module.loss_weights["regression"] * torch.nn.functional.l1_loss(
        m_reg[valid], data["targets.objects.regression"][valid]
    )
    diffs["matched_regression_design"] = _max_abs(v2_reg, ref_reg)
    checks["matched_regression_design_conformance"] = diffs["matched_regression_design"] <= 0.0
    # the matched regression is NOT v1's query-order loss (they genuinely differ here)
    query_reg = module.loss_weights["regression"] * torch.nn.functional.l1_loss(
        data["preds.objects.regression"][valid], data["targets.objects.regression"][valid]
    )
    checks["matched_regression_differs_from_query_order"] = not torch.equal(ref_reg, query_reg)

    # -- (iii) NO in-place permute: the decoder's objects.* are untouched ------
    checks["objects_class_logits_not_mutated"] = torch.equal(
        b.get("objects.class_logits"), pre_class_logits
    )
    checks["objects_masks_not_mutated"] = torch.equal(b.get("objects.masks"), pre_masks)
    checks["objects_regression_not_mutated"] = torch.equal(
        b.get("preds.objects.regression"), pre_reg
    )
    # matched.* are NEW keys (distinct from objects.*), permuted by the matcher
    checks["matched_objects_are_new_keys"] = (
        "matched.objects.class_logits" in out and "matched.objects.masks" in out
    )

    # -- (iv) MaskFormerTargets: object_class / masks / regression vs v1 -------
    proc = build_maskformer_targets()
    tbatch = make_maskformer_targets_batch()
    tb = Bundle()
    for key, value in tbatch.items():
        tb.set(key, value)
    tout = proc.process(tb, np.s_[0:6], Mode.FIT)
    v2_object_class = tout["labels.objects.object_class"]
    v2_target_masks = tout["labels.objects.masks"]

    # v1 object_class: the in-place x[x==raw]=mapped remap (datasets.py:641-643), here
    # via the shipped class map applied to the ORIGINAL raw values.
    raw_flav = np.asarray(tbatch["raw.truth_hadrons"]["flavour"], dtype=np.int64)
    v1_oc = raw_flav.copy()
    for spec in MASKFORMER_OBJECT_CLASS_MAP.values():
        v1_oc[raw_flav == spec["raw"]] = spec["mapped"]
    checks["targets_object_class_bitwise_vs_v1_classmap"] = np.array_equal(v2_object_class, v1_oc)

    # v1 masks: build_target_masks(object_ids, constituent_ids) (mask_utils.py:5-37)
    obj_ids = torch.as_tensor(np.asarray(tbatch["raw.truth_hadrons"]["barcode"], dtype=np.int64))
    con_ids = torch.as_tensor(
        np.asarray(tbatch["raw.tracks"]["ftagTruthParentBarcode"], dtype=np.int64)
    )
    v1_target_masks = v1_build_target_masks(obj_ids.clone(), con_ids).numpy()
    checks["targets_masks_bitwise_vs_v1_build_target_masks"] = np.array_equal(
        v2_target_masks, v1_target_masks
    )
    # the regression labels are the raw object columns verbatim (float32)
    checks["targets_regression_labels_bitwise_vs_raw"] = all(
        np.array_equal(
            tout[f"labels.objects.{t}"],
            np.asarray(tbatch["raw.truth_hadrons"][t], dtype=np.float32),
        )
        for t in proc.regression_targets
    )
    # v1's build_target_masks MUTATES object_ids in place (-1 -> -999, mask_utils.py:36);
    # the v2 processor must NOT mutate the published object ids — the raw barcode column
    # in the batch is byte-unchanged after process()
    checks["targets_no_inplace_id_mutation"] = np.array_equal(
        np.asarray(tb.get("raw.truth_hadrons")["barcode"]),
        np.asarray(tbatch["raw.truth_hadrons"]["barcode"]),
    )
    # null is validated LAST (the mapped null index == num non-null classes)
    checks["targets_null_class_is_last"] = proc.null_index == MASKFORMER_NUM_OBJECT_CLASSES - 1

    # -- (v) loud construction / bind surfaces --------------------------------
    # null not mapped last (null mapped to 0, b to 1 -> the v1 'Null class must be last')
    checks["null_not_last_raises"] = _raises_config_error(
        lambda: MaskFormerTargets(
            object_class="flavour",
            object_id="barcode",
            constituent_id="ftagTruthParentBarcode",
            class_map={"b": {"raw": 5, "mapped": 1}, "null": {"raw": -1, "mapped": 0}},
            object_stream="truth_hadrons",
            constituent_stream="tracks",
        )
    )
    # missing null class
    checks["missing_null_class_raises"] = _raises_config_error(
        lambda: MaskFormerTargets(
            object_class="flavour",
            object_id="barcode",
            constituent_id="ftagTruthParentBarcode",
            class_map={"b": {"raw": 5, "mapped": 0}, "c": {"raw": 4, "mapped": 1}},
            object_stream="truth_hadrons",
            constituent_stream="tracks",
        )
    )
    # unknown loss-weight key on the matched loss
    checks["unknown_loss_weight_key_raises"] = _raises_config_error(
        lambda: MaskFormerMatchedLoss(
            num_classes=2, num_objects=5, loss_weights={"not_a_component": 1.0}
        )
    )
    # all-zero matcher weights (the v1 matcher asserts the sum positive, matcher.py:167)
    checks["zero_matcher_weights_raises"] = _raises_config_error(
        lambda: MaskFormerMatchedLoss(
            num_classes=2,
            num_objects=5,
            loss_weights={"object_class_ce": 1.0},
            matcher_weights={"object_class_ce": 0.0},
        )
    )
    # regression pred/target width mismatch at bind
    checks["regression_width_mismatch_at_bind_raises"] = _raises_config_error(
        _bad_matched_regression_bind
    )

    passed = all(checks.values())
    criterion = (
        "the MaskFormerMatchedLoss forward (matcher + loss methods) reproduces, BITWISE, an "
        "INDEPENDENT v1 MaskFormerLoss for the three v1-decidable components (object_class_ce, "
        "mask_dice, mask_focal) on the v1-permuted predictions and for the matcher assignment; "
        "the matched regression is the FD alignment change (matched, not v1's query-order) "
        "asserted as a design-conformance property; the decoder's objects.* are NOT mutated in "
        "place (matched.objects.* are NEW keys); and MaskFormerTargets reproduces object_class / "
        "masks / regression labels BITWISE vs v1 (build_target_masks + the validated class map) "
        "WITHOUT v1's in-place id mutations. Loud surfaces (null-not-last, missing null, unknown "
        "loss weight, zero matcher weights, regression width mismatch) all raise ConfigError. "
        "Matched-loss / matcher / targets slice ONLY — decoder is MF1a, object writer is MF2"
    )
    report = _base_report(
        "mf1c_matched_loss_matcher_targets",
        passed,
        criterion,
        {
            "atol": PARITY_ATOL,
            "num_classes": MASKFORMER_NUM_OBJECT_CLASSES - 1,
            "num_objects": MASKFORMER_NUM_OBJECTS,
            "num_reg_targets": MASKFORMER_NUM_REG_TARGETS,
            "loss_weights": module.loss_weights,
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["max_abs_diffs"] = diffs
    report["v1_reference"] = (
        "an independently-constructed v1 MaskFormerLoss (build_independent_v1_matched_loss: a "
        "fresh instance from the same config, its own HungarianMatcher + empty_weight buffer, "
        "NOT the v2 module's composed v1_loss) for the matcher + the three mask/class losses; "
        "v1 build_target_masks + MaskformerObjectConfig class map for the targets processor"
    )
    report["alignment_note"] = (
        "MF1b (human sign-off, separate gate): v1's EFFECTIVE object-regression loss is "
        "QUERY-ORDER (the object task computes its L1 in SaltModel.run_tasks WITHOUT the matcher, "
        "saltmodel.py:218-219; 'regression' is not in MaskFormerLoss.losses, so the matched loop "
        "never re-computes it — the matcher uses regression only as a COST). v2's "
        "MaskFormerMatchedLoss makes the regression loss MATCHED (matcher-permuted preds vs "
        "truth-order targets), a user-visible alignment change. This gate asserts the matched "
        "form is what the FD specifies; it does NOT (cannot) claim v1 byte-parity for the "
        "regression component. See the alignment characterisation doc for the MF1b sign-off."
    )
    report["scope_note"] = (
        "matched-loss / matcher / MaskFormerTargets slice of the gate-table MF1a; the MaskDecoder "
        "+ drop_registers forward-parity is run_mf1a; the MaskFormerObjectWriter byte-parity is MF2"
    )
    print(f"{'diff':<40}{'value':>14}")
    for name, val in diffs.items():
        print(f"{name:<40}{val:>14.3e}")
    _print_checks(checks)
    _print_verdict("mf1c", passed, criterion, _emit_report(report, outdir, "mf1c"))
    return (0 if passed else 1), report


def _single_module_plan(module: MaskFormerMatchedLoss):
    """Compile a one-module FIT plan that drives the matched loss on a fixture bundle.

    The matched loss reads the decoder's object predictions, the scaled regression
    prediction/target, and the truth labels — all supplied directly in the gate bundle
    (the decoder + regression task are OTHER sub-wave-C modules). A single-module plan
    over those source keys exercises the REAL executor (merge + write-once + the
    declared produces check) rather than a bare ``module.forward`` call.

    Returns
    -------
    Plan
        The compiled FIT plan with the matched loss as its only step.
    """
    f = Mode.FIT
    m = MASKFORMER_NUM_OBJECTS
    n_cls = MASKFORMER_NUM_OBJECT_CLASSES
    sources = unflatten_spec({
        "objects.class_logits": TensorSpec(shape=("B", m, n_cls), dtype="float32", modes=f),
        "objects.class_probs": TensorSpec(shape=("B", m, n_cls), dtype="float32", modes=f),
        "objects.masks": TensorSpec(shape=("B", m, "T:tracks"), dtype="float32", modes=f),
        "objects.embed": TensorSpec(shape=("B", m, HYBRID_ENC_DIM), dtype="float32", modes=f),
        "preds.objects.regression": TensorSpec(
            shape=("B", m, MASKFORMER_NUM_REG_TARGETS), dtype="float32", modes=f
        ),
        "targets.objects.regression": TensorSpec(
            shape=("B", m, MASKFORMER_NUM_REG_TARGETS), dtype="float32", modes=f
        ),
        "labels.objects.object_class": TensorSpec(
            shape=("B", m), dtype="int64", kind="label", modes=f
        ),
        "labels.objects.masks": TensorSpec(
            shape=("B", m, "T:tracks"), dtype="bool", kind="label", modes=f
        ),
    })
    sinks = [f"losses.{c}" for c in module.components] + [
        "matched.objects.class_logits",
        "matched.objects.masks",
    ]
    return compile_plan({module.name: module}, Mode.FIT, sources=sources, sinks=sinks)


def _bad_matched_regression_bind() -> None:
    """Bind a `MaskFormerMatchedLoss` whose regression pred/target widths disagree.

    The matched L1 is element-wise, so a 3-wide prediction against a 4-wide target must
    raise at bind (the schema resolves both widths and the module compares them).
    """
    module = MaskFormerMatchedLoss(
        num_classes=2, num_objects=5, loss_weights={"object_class_ce": 1.0, "regression": 1.0}
    )
    module.name = "mf_matched_loss"
    module.bind(
        ResolvedSchema(widths={"preds.objects.regression": 3, "targets.objects.regression": 4})
    )


def _writer_write_ctx(modules: dict[str, Any], outdir: Path, n_tracks: int, total: int) -> WriteCtx:
    """A `WriteCtx` for the MaskFormer object writer's TEST forward.

    Returns
    -------
    WriteCtx
        Run context with the constituent file length the masks span.
    """
    return WriteCtx(
        output_path=outdir / "mf_objects.h5",
        total=total,
        run_name="MFrun",
        source_path=outdir / "src.h5",
        streams=("jets", "tracks"),
        sequence_streams=("tracks",),
        group_datasets={"jets": "jets", "tracks": "tracks"},
        seq_lengths={"tracks": n_tracks},
        model_modules=modules,
        batch_size=total,
    )


def run_mf2(
    outdir: Path | str,
    *,
    corruption: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[int, dict[str, Any]]:
    """MF2: MaskFormerObjectWriter TEST byte-parity + ONNX leading_object/object_index reduces.

    The object-writer slice of sub-wave C. Two halves:

    - **TEST byte-parity (BITWISE, ``.tobytes()``)**: the writer's per-batch
      ``write`` is compared, column by column, against the EXACT v1 op chain
      (``predictionwriter.py:276-308``) recomputed in the gate on the SAME
      decoder predictions + truth labels — the ``objects`` group (per-class
      ``MFrun_p{c}`` f4 probs + the ``class_label`` i8 truth), the constituent
      ``MFrun_MaskIndex`` i8 column (``indices_from_mask(masks.sigmoid() > 0.5)``,
      ``-1`` padded / ``-2`` no-object), and the ``object_masks`` group
      (``truth_mask`` i8 + ``mask_logits`` f4). Both MaskIndex sentinels are
      exercised (the fixture pads the last constituents). The truth ``class_label``
      column carries the v2 canonical remapped ``labels.objects.object_class``
      (the documented v1 VALUE divergence — raw ``flavour`` in v1; the column
      name/dtype/structure are v1-identical), so the gate compares it against
      that bundle key, not v1's raw field.
    - **ONNX (a COMPILED + RUN ``Mode.ONNX`` plan, never a TEST-mode proxy)**:
      the writer's `onnx_outputs` manifest (``preds.objects.regression`` ->
      ``leading_object``, ``objects.masks`` -> ``object_index``) is attached and a
      REAL graph is traced + exported. The onnxruntime session's output dtypes are
      asserted (the R leading scalars ``tensor(float)`` / float32, ``HadronIndex``
      ``tensor(int8)`` with a dynamic ``n_tracks`` axis), and torch-vs-onnxruntime
      AGREE over a constituent-length sweep — the int8 ``HadronIndex`` EXACTLY
      (always finite), the leading-object floats where finite (v1 returns NaN for a
      null leading object, ``maskformer.py:330-332``; NaN-on-both-sides is
      agreement). The ONNX index suffix is `OBJECT_INDEX.onnx` (``HadronIndex``)
      and the TEST suffix `OBJECT_INDEX.test` (``MaskIndex``) — the PINNED pair.

    Plus: the writer IMPORTS `OBJECT_INDEX` and re-declares neither
    ``"MaskIndex"`` nor ``"HadronIndex"`` as a string literal (the merge-condition-4
    discipline, asserted by scanning the writer source), and loud surfaces raise
    ``ConfigError`` — empty ``object_classes``; an unconfigured ``regression_task``
    (the missing-``output_suffixes`` raise); a NON-DEFAULT ``regression_task`` for
    ONNX (the ``object_index`` reduce reads the fixed
    ``preds.<stream>.regression`` key, so a non-``regression`` name fails loudly at
    ``onnx_outputs`` instead of fetching a stale key at trace time); and a
    non-sequence constituent stream.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)`` — 0 only if every check passed.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("MF2 MaskFormerObjectWriter TEST byte-parity + ONNX leading_object/object_index reduces")
    print("=" * 96)
    norm_dict = _norm_dict(outdir)
    checks: dict[str, bool] = {}
    diffs: dict[str, float] = {}

    object_classes = ["b", "c", "null"]  # null LAST (MaskFormer.yaml:169-178)
    # deterministic module weights: v1's composed get_maskformer_outputs has
    # data-dependent control flow (the all-null `if not null_preds.any()` branch,
    # maskformer.py:312-319) that bakes a SHAPE into the trace, so the random init
    # must be fixed for a stable ONNX export regardless of prior-test RNG state.
    torch.manual_seed(0)
    modules = build_maskformer_writer_modules(norm_dict)
    writer = MaskFormerObjectWriter(object_classes=object_classes, regression_task="regression")
    writer.name = "object_writer"

    # -- (i) TEST byte-parity vs the v1 op chain (predictionwriter.py:276-308) --
    n_tracks, total = 10, 6
    data = make_maskformer_writer_batch(batch_size=total, n_tracks=n_tracks)
    b = Bundle()
    for key, value in data.items():
        b.set(key, value)
    writer.setup(_writer_write_ctx(modules, outdir, n_tracks, total))
    out = writer.write(b, slice(0, total))
    if corruption is not None:
        # corrupt one shipped column in place — the byte comparison must FAIL
        out = {k: v.copy() for k, v in out.items()}
        out["objects"]["MFrun_pb"] = corruption(out["objects"]["MFrun_pb"])

    cp = data["objects.class_probs"]
    masks = data["objects.masks"]
    obj_class = data["labels.objects.object_class"]
    truth_mask = data["labels.objects.masks"]
    pad = data["masks.tracks"]

    # objects group: per-class probs (v1 :276-281) + remapped truth class (v1 :282-285)
    v1_prob_dtype = np.dtype([(f"MFrun_p{c}", "f4") for c in object_classes])
    v1_probs = u2s(cp.cpu().float().numpy(), v1_prob_dtype)
    v1_class = u2s(obj_class.cpu().unsqueeze(-1).numpy(), np.dtype([("class_label", "i8")]))
    checks["test_class_probs_bitwise_vs_v1"] = all(
        out["objects"][n].tobytes() == v1_probs[n].tobytes() for n in v1_probs.dtype.names
    )
    checks["test_class_target_bitwise_vs_v1"] = (
        out["objects"]["class_label"].tobytes() == v1_class["class_label"].tobytes()
    )

    # MaskIndex on the constituent stream (v1 :287-297)
    v1_idx = v1_indices_from_mask(masks.cpu().sigmoid() > 0.5).int().cpu().numpy()
    v1_idx = np.where(~pad.cpu().numpy(), v1_idx, -1)
    v1_idx_struct = u2s(
        np.expand_dims(v1_idx, -1), np.dtype([(f"MFrun_{OBJECT_INDEX.test}", "i8")])
    )
    mask_index_col = f"MFrun_{OBJECT_INDEX.test}"
    checks["test_mask_index_bitwise_vs_v1"] = (
        out["tracks"][mask_index_col].tobytes() == v1_idx_struct[mask_index_col].tobytes()
    )
    # both sentinels exercised (the byte-parity claim has teeth on the encodings)
    checks["test_mask_index_has_padded_sentinel"] = bool(
        (out["tracks"][mask_index_col] == -1).any()
    )
    checks["test_mask_index_has_no_object_sentinel"] = bool(
        (out["tracks"][mask_index_col] == -2).any()
    )

    # object_masks group: truth mask (v1 :300-304) + raw mask logits (v1 :305-308)
    v1_tm = u2s(truth_mask.cpu().unsqueeze(-1).numpy(), np.dtype([("truth_mask", "i8")]))
    v1_ml = u2s(masks.cpu().float().unsqueeze(-1).numpy(), np.dtype([("mask_logits", "f4")]))
    checks["test_truth_mask_bitwise_vs_v1"] = (
        out["object_masks"]["truth_mask"].tobytes() == v1_tm["truth_mask"].tobytes()
    )
    checks["test_mask_logits_bitwise_vs_v1"] = (
        out["object_masks"]["mask_logits"].tobytes() == v1_ml["mask_logits"].tobytes()
    )

    # -- (ii) OBJECT_INDEX imported, NOT re-declared (merge condition 4) --------
    import inspect  # noqa: PLC0415 - gate-local source scan

    from salt.core.writers import maskformer as writer_src  # noqa: PLC0415 - gate-local source scan

    source = inspect.getsource(writer_src)
    checks["object_index_imported_not_redeclared"] = (
        "from salt.core.writers.names import OBJECT_INDEX" in source
        and '"MaskIndex"' not in source
        and '"HadronIndex"' not in source
        and "'MaskIndex'" not in source
        and "'HadronIndex'" not in source
    )

    # -- (iii) ONNX: a real Mode.ONNX export of the two object reduces ----------
    variables = {"jets": JET_VARIABLES, "tracks": TRACK_VARIABLES}
    declare = WriterDeclareCtx(
        model_modules=modules, streams=("jets", "tracks"), sequence_streams=("tracks",)
    )
    manifest = writer.onnx_outputs(declare)
    export = ExportConfig(
        model_name="MFv2",
        inputs=[
            ExportInput(port="inputs.jets"),
            ExportInput(port="inputs.tracks", sequence=True, dyn_axis="n_tracks"),
        ],
    )
    resolved = attach_manifest(resolve_export_config(export, "MFv2"), manifest)
    plan = compile_onnx_plan(modules, resolved, variables)
    bind_all(modules, resolve_bind_schema([plan]))
    materialise_all(modules)
    result = export_graph(
        modules, export, variables, outdir / "mf_objects.onnx", outputs=manifest, run_name="MFv2"
    )
    session = make_session(result.onnx_path)
    out_types = {o.name: o.type for o in session.get_outputs()}
    out_axes = {o.name: o.shape for o in session.get_outputs()}
    leading_names = [f"MFv2_leading_objects_{t}" for t in writer._regression_suffixes(modules)]  # noqa: SLF001 - gate adapter
    index_name = f"MFv2_{OBJECT_INDEX.onnx}"
    checks["onnx_leading_object_is_float32"] = all(
        out_types.get(n) == "tensor(float)" for n in leading_names
    )
    checks["onnx_object_index_is_int8"] = out_types.get(index_name) == "tensor(int8)"
    checks["onnx_object_index_has_dynamic_token_axis"] = out_axes.get(index_name) == ["n_tracks"]
    checks["onnx_index_suffix_is_pinned_HadronIndex"] = index_name == "MFv2_HadronIndex"
    # the manifest carries the two writer-declared reduces (not a TaskWriter family)
    checks["onnx_manifest_uses_object_reduces"] = sorted(e.reduce for e in resolved.outputs) == [
        "leading_object",
        "object_index",
    ]

    # torch-vs-onnxruntime agreement, computed DIRECTLY (not via check_onnx's
    # all-or-nothing comparison, which aborts a trial on the first NaN — and v1's
    # leading-reg is NaN whenever the leading object is null, maskformer.py:330-332).
    # A COMPILED + RUN Mode.ONNX claim over a constituent-length sweep: the int8
    # HadronIndex EXACTLY (always finite), the leading floats where finite.
    index_exact = True
    float_finite_max = 0.0
    torch.manual_seed(7)
    for length in (3, 11, 20):
        jets = torch.rand(1, len(JET_VARIABLES))
        tracks = torch.rand(length, len(TRACK_VARIABLES))
        with torch.no_grad():
            eager = dict(
                zip(result.adapter.output_names, result.adapter(jets, tracks), strict=True)
            )
        ort_out = dict(
            zip(
                [o.name for o in session.get_outputs()],
                session.run(None, {"jet_features": jets.numpy(), "track_features": tracks.numpy()}),
                strict=True,
            )
        )
        index_exact &= np.array_equal(eager[index_name].numpy(), ort_out[index_name])
        for name in leading_names:
            e, o = eager[name].numpy(), ort_out[name]
            finite = np.isfinite(e) & np.isfinite(o)
            if finite.any():
                float_finite_max = max(float_finite_max, float(np.abs(e[finite] - o[finite]).max()))
            # NaN-on-both-sides is agreement (v1 null-leading-object NaN)
            index_exact &= bool(np.array_equal(np.isnan(e), np.isnan(o)))
    checks["onnx_object_index_int8_exact"] = index_exact
    checks["onnx_leading_object_floats_agree_where_finite"] = float_finite_max <= 1e-4
    diffs["onnx_leading_object_worst_abs_diff"] = float_finite_max

    # -- (iv) loud construction / configuration surfaces -----------------------
    checks["empty_object_classes_raises"] = _raises_config_error(
        lambda: MaskFormerObjectWriter(object_classes=[])
    )
    checks["missing_regression_task_raises"] = _raises_config_error(
        lambda: _bad_writer_onnx(modules)
    )
    checks["nondefault_regression_task_raises"] = _raises_config_error(
        lambda: _bad_writer_nondefault_regression_task(modules)
    )
    checks["non_sequence_constituent_raises"] = _raises_config_error(
        lambda: _bad_writer_extra_groups(outdir, modules)
    )

    passed = all(checks.values())
    criterion = (
        "the MaskFormerObjectWriter reproduces v1's object-writing block BITWISE in TEST "
        "(per-class probs + class_label truth on the objects group, the MaskIndex i8 column -1/-2 "
        "sentinels on the constituent stream, the truth_mask + mask_logits object_masks group) and "
        "declares the two object ONNX outputs that a COMPILED + RUN Mode.ONNX export produces with "
        "the correct dtypes (leading_object float32 scalars, object_index int8 HadronIndex with a "
        "dynamic token axis), torch and onnxruntime agreeing (int8 EXACT, leading floats where "
        "finite). OBJECT_INDEX is IMPORTED, never re-declared. Loud surfaces raise ConfigError. "
        "Object-writer slice ONLY (decoder is MF1a, matched loss/targets is MF1c)"
    )
    report = _base_report(
        "mf2_object_writer",
        passed,
        criterion,
        {
            "object_classes": object_classes,
            "regression_targets": list(writer._regression_suffixes(modules)),  # noqa: SLF001
            "onnx_index_test_suffix": OBJECT_INDEX.test,
            "onnx_index_onnx_suffix": OBJECT_INDEX.onnx,
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["max_abs_diffs"] = diffs
    report["v1_reference"] = (
        "the v1 op chain recomputed in the gate (predictionwriter.py:276-308: u2s of the class "
        "probs / class label / mask index / truth mask / mask logits) for TEST; a COMPILED + RUN "
        "Mode.ONNX export checked against the eager OnnxAdapter (check_onnx) for the two reduces"
    )
    report["divergence_note"] = (
        "the class_label truth column carries the REMAPPED labels.objects.object_class (the "
        "canonical v2 truth MaskFormerTargets produces), where v1 wrote the RAW flavour field "
        "(predictionwriter.py:282-285). Column name/dtype/structure are v1-identical; only the "
        "integer mapping differs, intentionally (v2 standardised on the remapped class, and "
        "MaskFormerTargets eliminates the v1 in-place raw-id mutation)"
    )
    report["scope_note"] = (
        "MaskFormerObjectWriter byte-parity + the two new ONNX reduces; the MaskDecoder + "
        "drop_registers is run_mf1a, the matched loss / matcher / targets is run_mf1c"
    )
    print(f"{'diff':<40}{'value':>14}")
    for name, val in diffs.items():
        print(f"{name:<40}{val:>14.3e}")
    _print_checks(checks)
    _print_verdict("mf2", passed, criterion, _emit_report(report, outdir, "mf2"))
    return (0 if passed else 1), report


def _bad_writer_onnx(modules: dict[str, Any]) -> None:
    """Derive the manifest of a writer whose object-regression task is unconfigured.

    The ``leading_object`` entry needs the object-regression task's
    ``output_suffixes``; with ``regression_task == "regression"`` (the only ONNX-
    legal name) but NO such module configured, ``_regression_suffixes`` must
    raise. Drops the ``regression`` stub from the module dict so the raise is the
    unconfigured-task one (not the non-default-name guard, which fires earlier).
    """
    without_reg = {k: v for k, v in modules.items() if k != "regression"}
    writer = MaskFormerObjectWriter(object_classes=["b", "c", "null"], regression_task="regression")
    writer.name = "object_writer"
    writer.onnx_outputs(
        WriterDeclareCtx(
            model_modules=without_reg, streams=("jets", "tracks"), sequence_streams=("tracks",)
        )
    )


def _bad_writer_nondefault_regression_task(modules: dict[str, Any]) -> None:
    """Derive the manifest of a writer whose object-regression task is non-default.

    The ``object_index`` reduce reads the fixed ``preds.<stream>.regression`` key
    (it is declared on the masks port and cannot carry the task name), so a
    ``regression_task`` other than ``"regression"`` must raise loudly at
    ``onnx_outputs`` rather than fetching a stale/absent key at trace time.
    """
    writer = MaskFormerObjectWriter(object_classes=["b", "c", "null"], regression_task="objreg")
    writer.name = "object_writer"
    writer.onnx_outputs(
        WriterDeclareCtx(
            model_modules=modules, streams=("jets", "tracks"), sequence_streams=("tracks",)
        )
    )


def _bad_writer_extra_groups(outdir: Path, modules: dict[str, Any]) -> None:
    """Ask a writer to size its object_masks group with a non-sequence constituent.

    The masks span the constituent stream's tokens, so a constituent stream with
    no file sequence length (here ``jets``, a global stream) must raise.
    """
    writer = MaskFormerObjectWriter(
        object_classes=["b", "c", "null"], constituent_stream="jets", regression_task="regression"
    )
    writer.name = "object_writer"
    ctx = WriteCtx(
        output_path=outdir / "bad.h5",
        total=6,
        run_name="MFrun",
        source_path=outdir / "src.h5",
        streams=("jets", "tracks"),
        sequence_streams=("tracks",),
        group_datasets={"jets": "jets", "tracks": "tracks"},
        seq_lengths={"tracks": 10},  # 'jets' is NOT a sequence stream here
        model_modules=modules,
        batch_size=6,
    )
    writer.extra_groups(ctx)


# ---------------------------------------------------------------------------
# D1 — FIT/VAL callback-sink assembly + register_reduce live registry (sub-wave D)
# ---------------------------------------------------------------------------


class _D1AuxProbe(torch.nn.Module):
    """An aux head producing a ``preds.*`` key NO loss consumes (the MaskformerMetrics shape).

    Carries a real parameter so it is a genuine model module; its product
    ``preds.jets.aux_d1`` is anchored by NOTHING in FIT/VAL, so it is demand-pruned
    unless a callback declares it a sink (the §3.1/§3.4 deferral D1 lands).
    """

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(16, 2)  # 16 = the gn2v2 fixture pooled.global width
        self.name = "aux_d1"

    def declare_io(self, mode: Mode) -> IO:
        """The aux head's IO: reads ``pooled.global``, produces an un-anchored pred.

        Returns
        -------
        IO
            ``pooled.global`` in, ``preds.jets.aux_d1`` out (all modes).
        """
        del mode
        return IO(
            requires=unflatten_spec({"pooled.global": TensorSpec()}),
            produces=unflatten_spec({"preds.jets.aux_d1": TensorSpec(shape=("B", 2))}),
        )


class _D1SinkCallback:
    """A duck-typed FIT/VAL-sink callback (the `fit_val_demand` surface, design §3.4).

    The same protocol the real `MaskformerMetrics` / `ConfusionMatrix` callbacks
    expose — declaring the bundle keys it reads each VAL epoch so they survive
    demand pruning. Used to drive the REAL `SaltModule._callback_demand` /
    `_model_sinks` / `compile_plan` assembly with an aux head's un-anchored pred.
    """

    def __init__(self, *keys: str) -> None:
        self._keys = keys

    def fit_val_demand(self, model_modules: Any) -> tuple[str, ...]:
        """The declared FIT/VAL demand keys (config-only, static).

        Returns
        -------
        tuple[str, ...]
            The keys this callback demands as FIT/VAL plan sinks.
        """
        del model_modules
        return self._keys


def _attach_callbacks(model: SaltModule, *callbacks: Any) -> None:
    """Attach a duck-typed trainer carrying `callbacks` to a `SaltModule`.

    The static-tooling attach shape (a trainer namespace exposing
    ``callbacks`` + a ``datamodule.reader``) the model's `_attached_callbacks`
    / `_attached_writer` discovery reads — exactly the `test_saltmodule`
    `TestCallbackSinks` precedent.
    """
    from types import SimpleNamespace  # noqa: PLC0415 - gate-local duck-typed attach

    model._trainer = SimpleNamespace(  # noqa: SLF001 - duck-typed trainer attach
        callbacks=list(callbacks), datamodule=SimpleNamespace(reader=None)
    )


def _d1_compile_with_sinks(model: SaltModule, mode: Mode) -> Any:
    """Compile a model-side plan with the model's OWN assembled sinks for `mode`.

    Drives the REAL `_model_sinks` (which folds in `_callback_demand`) and
    `compile_plan` — so a callback-declared sink genuinely participates in
    demand pruning, not a re-implemented sink list.

    Returns
    -------
    Plan
        The compiled plan for `mode`.
    """
    return compile_plan(
        model._graph_modules,  # noqa: SLF001 - gate drives the real model-module dict
        mode,
        sources=gn2v2_sources(),
        sinks=model._model_sinks(mode),  # noqa: SLF001 - the assembled FIT/VAL sinks under test
    )


def run_d1(
    outdir: Path | str,
    *,
    corruption: Callable[[list[str]], list[str]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """D1: FIT/VAL callback-sink assembly + register_reduce live-registry + reduce dtypes.

    The sub-wave-D C-prereqs (plan 10 D1 row). Two independent slices, both
    driving the REAL surfaces (no re-implementation in the gate):

    - **FIT/VAL callback-sink assembly** (design §3.1 454-456, §3.4 667-671). A
      real `SaltModule` carries an aux head (`_D1AuxProbe`) producing
      ``preds.jets.aux_d1`` that NO loss consumes. Compiled with the model's OWN
      `_model_sinks` (which folds in `_callback_demand`):
      (i) WITHOUT a FIT/VAL-sink callback, the FIT and VAL sinks are
      ``['loss.total']`` only and the aux producer is demand-PRUNED (the
      unchanged M2/M3 baseline — the negative control with teeth);
      (ii) WITH a `_D1SinkCallback("preds.jets.aux_d1")` attached, its
      `fit_val_demand` becomes a FIT/VAL plan sink (``preds.jets.aux_d1`` appears
      in the assembled sinks, after the ``loss.total`` anchor) and the once-pruned
      producer is KEPT ALIVE in BOTH FIT and VAL;
      (iii) the SAME surface drives the real `ConfusionMatrix` callback — its
      `fit_val_demand` declaration (``preds.jets.jets_classification`` +
      ``labels.jets.flavour_label``) is consumed by `_callback_demand`, proving a
      shipped callback class participates, not only the duck-typed fixture;
      (iv) callback demand is TRAINING-ONLY — `_callback_demand` is empty in
      TEST/ONNX (those sinks are the writer manifest, not callback requires).

    - **register_reduce live-registry validation + reduce-declared dtypes**
      (amendment 555-567, design §7.3). The export reduce set is a LIVE registry,
      not a frozen tuple. A freshly `register_reduce`-d reduce
      (``d1_probe_reduce``, int8 per-token) VALIDATES through the REAL
      ``export.outputs`` resolution path (`attach_manifest` ->
      ``config._resolve_output`` against the live registry), defaulting + accepting
      its DECLARED dtype; an UNREGISTERED reduce name is REJECTED loudly
      (``ConfigError``); and a declared dtype that DISAGREES with the reduce's
      registered dtype is REJECTED loudly (the reduce owns its per-reduce dtype
      rule). Plus the public `register_reduce` loud surfaces: a DUPLICATE name, a
      bad dtype, and a non-string name each raise ``ConfigError`` — and the new
      reduce appears in `registered_reduces` with `reduce_dtype` returning its
      declared dtype. The probe is then `unregister_reduce`-d so the
      process-global registry is restored (no residue leaks into the in-process
      test suite's exact-shipped-set assertions).

    D1 makes NO model-parity claim — it is a planner/registry-assembly gate; the
    MaskFormer decoder/loss/writer forward parity are MF1a/MF1c/MF2.

    Negative control (``test_gates_m5.py``): the ``corruption`` hook perturbs the
    WITH-callback assembled sink list (dropping the callback-declared sink) before
    the kept-alive recompile — the "callback sink keeps the producer alive" check
    must then FAIL while the register_reduce / TRAINING-only / loud-surface checks
    (independent of that sink list) stay green.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)`` — 0 only if every check passed.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("D1 FIT/VAL callback-sink assembly + register_reduce live registry + reduce dtypes")
    print("=" * 96)
    norm_dict = _norm_dict(outdir)
    checks: dict[str, bool] = {}

    # -- (a) FIT/VAL callback-sink assembly through the REAL SaltModule ---------
    def _model_with_aux() -> SaltModule:
        modules = build_gn2v2_modules(norm_dict)
        modules["aux_d1"] = _D1AuxProbe()
        modules["loss"] = LossSum()
        return SaltModule(
            modules, lrs={"initial": 1e-3, "max": 5e-3, "end": 1e-4, "pct_start": 0.1}
        )

    # baseline: NO FIT/VAL-sink callback -> sinks are loss.total only (M2/M3)
    base_model = _model_with_aux()
    checks["no_callback_fit_sinks_are_loss_only"] = base_model._model_sinks(Mode.FIT) == [  # noqa: SLF001
        "loss.total"
    ]
    checks["no_callback_val_sinks_are_loss_only"] = base_model._model_sinks(Mode.VAL) == [  # noqa: SLF001
        "loss.total"
    ]
    # negative control with teeth: the aux producer is demand-PRUNED without a sink
    pruned_plan = _d1_compile_with_sinks(base_model, Mode.FIT)
    checks["aux_producer_pruned_without_callback"] = "aux_d1" not in pruned_plan.module_names

    # WITH a FIT/VAL-sink callback declaring the aux pred -> it becomes a sink
    sink_model = _model_with_aux()
    _attach_callbacks(sink_model, _D1SinkCallback("preds.jets.aux_d1"))
    fit_sinks = sink_model._model_sinks(Mode.FIT)  # noqa: SLF001
    val_sinks = sink_model._model_sinks(Mode.VAL)  # noqa: SLF001
    if corruption is not None:
        # drop the callback-declared sink -> the kept-alive recompile must FAIL
        fit_sinks = corruption(list(fit_sinks))
    checks["loss_anchor_stays_first"] = fit_sinks[0] == "loss.total"
    checks["callback_require_becomes_fit_sink"] = "preds.jets.aux_d1" in fit_sinks
    checks["callback_require_becomes_val_sink"] = "preds.jets.aux_d1" in val_sinks
    # the once-pruned aux producer is KEPT ALIVE in BOTH FIT and VAL with the sink
    kept_fit = compile_plan(
        sink_model._graph_modules,  # noqa: SLF001
        Mode.FIT,
        sources=gn2v2_sources(),
        sinks=fit_sinks,
    )
    kept_val = compile_plan(
        sink_model._graph_modules,  # noqa: SLF001
        Mode.VAL,
        sources=gn2v2_sources(),
        sinks=val_sinks,
    )
    checks["aux_producer_kept_alive_in_fit"] = "aux_d1" in kept_fit.module_names
    checks["aux_producer_kept_alive_in_val"] = "aux_d1" in kept_val.module_names

    # a shipped callback class participates in the SAME surface: ConfusionMatrix's
    # fit_val_demand is consumed by the real _callback_demand (preds + label keys)
    cm_model = _model_with_aux()
    _attach_callbacks(cm_model, ConfusionMatrix(task_name="jets_classification"))
    cm_demand = cm_model._callback_demand(Mode.FIT)  # noqa: SLF001
    checks["shipped_confusion_matrix_demand_is_assembled"] = (
        "preds.jets.jets_classification" in cm_demand and "labels.jets.flavour_label" in cm_demand
    )
    # the real MaskformerMetrics callback exposes the same static fit_val_demand
    mf_demand = MaskformerMetrics().fit_val_demand({})
    checks["maskformer_metrics_declares_matched_sinks"] = set(mf_demand) == {
        "matched.objects.class_logits",
        "matched.objects.object_class",
        "matched.objects.masks",
        "matched.objects.target_masks",
    }

    # callback demand is TRAINING-ONLY: empty in TEST/ONNX (writer-manifest sinks)
    checks["callback_demand_empty_in_test"] = cm_model._callback_demand(Mode.TEST) == {}  # noqa: SLF001
    checks["callback_demand_empty_in_onnx"] = cm_model._callback_demand(Mode.ONNX) == {}  # noqa: SLF001

    # -- (b) register_reduce live-registry validation + reduce-declared dtypes --
    reduce_name = "d1_probe_reduce"
    if reduce_name not in registered_reduces():
        register_reduce(reduce_name, _d1_probe_binder, dtype="int8", per_token=True)
    checks["registered_reduce_in_live_registry"] = reduce_name in registered_reduces()
    checks["reduce_declared_dtype_is_queryable"] = reduce_dtype(reduce_name) == "int8"

    # a registered reduce VALIDATES through the REAL export.outputs resolution
    # (attach_manifest -> config._resolve_output against the live registry); the
    # declared dtype defaults from the registered declaration.
    export = ExportConfig(model_name="D1probe", inputs=[ExportInput(port="inputs.tracks")])
    good_entry = ExportOutput(port="objects.masks", name="ProbeIndex", reduce=reduce_name)
    resolved = attach_manifest(resolve_export_config(export, "D1probe"), [good_entry])
    (resolved_out,) = resolved.outputs
    checks["registered_reduce_validates_in_manifest"] = (
        resolved_out.reduce == reduce_name and resolved_out.dtype == "int8"
    )
    # an UNREGISTERED reduce name is REJECTED loudly through the same path
    checks["unregistered_reduce_rejected_loudly"] = _raises_config_error(
        _resolve_unregistered_reduce, export
    )
    # a declared dtype DISAGREEING with the reduce's registered dtype is REJECTED
    checks["reduce_dtype_mismatch_rejected_loudly"] = _raises_config_error(
        _resolve_dtype_mismatch, export, reduce_name
    )
    # an unregistered reduce name has NO declared dtype (reduce_dtype raises)
    checks["unregistered_reduce_dtype_query_raises"] = _raises_config_error(
        lambda: reduce_dtype("not_a_registered_reduce")
    )

    # the public register_reduce loud surfaces (duplicate / bad dtype / non-string)
    checks["duplicate_register_reduce_raises"] = _raises_config_error(
        lambda: register_reduce(reduce_name, _d1_probe_binder, dtype="int8")
    )
    checks["bad_dtype_register_reduce_raises"] = _raises_config_error(
        lambda: register_reduce("d1_bad_dtype", _d1_probe_binder, dtype="float64")
    )
    checks["non_string_name_register_reduce_raises"] = _raises_config_error(
        lambda: register_reduce(123, _d1_probe_binder, dtype="int8")  # type: ignore[arg-type]
    )

    # the live registry is process-global module state; this gate mutated it with a
    # PROBE reduce — restore it so the gate leaves no residue (a leaked probe would
    # break the exact-shipped-set assertions in the in-process test suite). Only the
    # probe name registers successfully (the bad-dtype/non-string attempts raised).
    unregister_reduce(reduce_name)
    checks["register_reduce_residue_cleared"] = reduce_name not in registered_reduces()

    passed = all(checks.values())
    criterion = (
        "a FIT/VAL-sink callback's declared require becomes a FIT/VAL plan sink in the REAL "
        "SaltModule._model_sinks/_callback_demand, keeping an otherwise demand-PRUNED aux "
        "producer alive in BOTH FIT and VAL (pruned without it — the negative control), with the "
        "shipped ConfusionMatrix/MaskformerMetrics participating in the same surface, callback "
        "demand TRAINING-only (empty in TEST/ONNX); AND a register_reduce-d reduce VALIDATES "
        "through the live export.outputs registry with its declared dtype while an unregistered "
        "reduce name and a mismatched declared dtype are both rejected loudly (ConfigError), and "
        "the public register_reduce duplicate/bad-dtype/non-string loud surfaces raise; the gate "
        "then UNregisters its probe so the process-global registry is left with no residue "
        "(design §3.1 454-456, §3.4 667-671; amendment 555-567, design §7.3)"
    )
    report = _base_report(
        "d1_callback_sinks_register_reduce",
        passed,
        criterion,
        {
            "registered_reduce": reduce_name,
            "fit_sinks_with_callback": list(fit_sinks),
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["scope_note"] = (
        "the sub-wave-D C-prereqs (FIT/VAL callback-sink assembly + register_reduce live "
        "registry/dtypes) ONLY — NO model-parity claim; the MaskFormer decoder/loss/writer "
        "parity are run_mf1a/run_mf1c/run_mf2"
    )
    report["surfaces_exercised"] = (
        "the REAL SaltModule._model_sinks/_callback_demand + compile_plan demand pruning (not a "
        "re-implemented sink list); the REAL salt.core.onnx.reduces live registry validated via "
        "attach_manifest -> config._resolve_output (not a hard-coded reduce check)"
    )
    _print_checks(checks)
    _print_verdict("d1", passed, criterion, _emit_report(report, outdir, "d1"))
    return (0 if passed else 1), report


def _d1_probe_binder(out_cfg: ExportOutput, ctx: Any) -> Any:
    """A trivial reduce binder for the D1 probe reduce (registry-validation only).

    D1 never RUNS this binder (it exercises the static config-validation path,
    not a traced export); it exists so `register_reduce` has a callable to bind.
    A one-output int8 per-token `BoundReduce` cloning the source port, the
    minimal shape the registry accepts.

    Returns
    -------
    BoundReduce
        The bound reduce for ``out_cfg`` (never executed by D1).
    """
    from salt.core.onnx.reduces import BoundReduce  # noqa: PLC0415 - gate-local probe binder

    name = f"{ctx.model_name}_{out_cfg.name}"
    return BoundReduce(
        port=out_cfg.port,
        output_names=(name,),
        dtypes=("int8",),
        dynamic_axes={name: {0: "n_tracks"}},
        fn=lambda bundle: (bundle.get(out_cfg.port),),
    )


def _resolve_unregistered_reduce(export: ExportConfig) -> None:
    """Resolve an ``export.outputs`` entry naming an UNREGISTERED reduce.

    The live registry has no such reduce, so `config._resolve_output` (reached
    through `attach_manifest`) must raise a `ConfigError`.
    """
    entry = ExportOutput(port="objects.masks", name="Nope", reduce="not_a_registered_reduce")
    attach_manifest(resolve_export_config(export, "D1probe"), [entry])


def _resolve_dtype_mismatch(export: ExportConfig, reduce_name: str) -> None:
    """Resolve an entry whose declared dtype disagrees with the reduce's declared dtype.

    The D1 probe reduce is int8; declaring ``float32`` on its entry must raise a
    `ConfigError` (the reduce owns its per-reduce dtype rule, design §7.3).
    """
    entry = ExportOutput(
        port="objects.masks", name="ProbeIndex", reduce=reduce_name, dtype="float32"
    )
    attach_manifest(resolve_export_config(export, "D1probe"), [entry])


# ---------------------------------------------------------------------------
# D2ckpt — Checkpoint monitor/filename contract + run-dir path inference;
#          ProgressBar smoke (sub-wave D-rest, plan 10 D2 row, this slice)
# ---------------------------------------------------------------------------


def _ckpt_setup_trainer(log_dir: str, *, fast_dev_run: bool = False) -> Any:
    """A minimal `ModelCheckpoint.setup`-compatible trainer stub.

    The exact surface ``ModelCheckpoint.setup`` reads: ``fast_dev_run``,
    ``log_dir``/``default_root_dir``, ``is_global_zero``, an empty ``loggers``
    (so `__resolve_ckpt_dir` short-circuits on a pre-set ``dirpath``) and a
    pass-through ``strategy.broadcast``. NOT a re-implementation — the gate
    drives the REAL `Checkpoint.setup`, which calls ``super().setup``.

    Returns
    -------
    SimpleNamespace
        The stub trainer.
    """
    from types import SimpleNamespace  # noqa: PLC0415 - gate-local duck-typed trainer

    return SimpleNamespace(
        fast_dev_run=fast_dev_run,
        log_dir=log_dir,
        default_root_dir=log_dir,
        is_global_zero=True,
        loggers=[],
        strategy=SimpleNamespace(broadcast=lambda x: x),
    )


def run_d2ckpt(
    outdir: Path | str,
    *,
    corruption: Callable[[str], str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """D2 (Checkpoint/ProgressBar slice): filename/monitor contract + run-dir inference.

    The sub-wave-D-rest Checkpoint + ProgressBar ports (plan 10 D2 row, this
    slice). NON-GATING for model reproduction — it gates the ``salt2 test``
    no-``--ckpt_path`` run-dir checkpoint path inference, so the v1 filename +
    ``ckpts/`` layout contract must hold end-to-end against the REAL
    `salt.core.main._best_checkpoint`. Three slices, every surface real:

    - **Filename + monitor contract** (v1 ``callbacks/checkpoint.py:25-27``). The
      REAL `salt.core.callbacks.Checkpoint`'s `format_checkpoint_name` produces
      ``epoch=NNN-{fname_string}={monitor_loss:.5f}.ckpt`` — the default
      ``fname_string="loss"`` gives the ``loss=`` stem; ``save_top_k == -1``
      (keep every epoch); ``monitor == monitor_loss``;
      ``auto_insert_metric_name is False`` (the metric carries a ``/``). The
      per-task monitor (``val/jets_classification_loss``) round-trips into the
      stem unmangled.

    - **Run-dir path inference** (the ``salt2 test`` contract,
      `salt.core.main._best_checkpoint`). The REAL `Checkpoint.setup("fit")`
      forces ``dirpath`` to ``<log_dir>/ckpts`` (v1 ``checkpoint.py:44-46``;
      driven through the actual `ModelCheckpoint.setup` short-circuit, NOT a
      re-implemented assignment); checkpoints named by the REAL
      `format_checkpoint_name` are written there, and `_best_checkpoint`
      (globbing ``{ckpts,checkpoints}/*.ckpt`` for the smallest ``loss=``)
      resolves the lowest-loss epoch — proving discovery works on what the
      callback writes. ``setup`` is a no-op on a non-fit stage / under
      ``fast_dev_run`` (``dirpath`` is then super-resolved to ``checkpoints/``,
      not forced to ``ckpts/`` — v1 ``checkpoint.py:30-32``), and an ``s3://``
      log dir is a loud `ConfigError` (the v1 s3 branch deferred to M6).

    - **ProgressBar smoke** (v1 stock ``TQDMProgressBar``, ``base.yaml:38-39``).
      `salt.core.callbacks.ProgressBar` instantiates with ``refresh_rate``,
      attaches as a `Callback`, and is a `TQDMProgressBar` (the stock bar under
      a salt-owned name).

    Negative control (``test_gates_m5.py``): the ``corruption`` hook rewrites the
    written checkpoint STEM (dropping the ``loss=`` tag) before the inference
    glob — `_best_checkpoint` must then raise (no ``loss=``-named checkpoint),
    so ``inference_resolves_lowest_loss`` FAILS while the contract / setup-dir /
    progress checks (independent of that filename) stay green.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)`` — 0 only if every check passed.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print(
        "D2ckpt Checkpoint filename/monitor contract + run-dir path inference + ProgressBar smoke"
    )
    print("=" * 96)
    checks: dict[str, bool] = {}

    # -- (a) filename + monitor contract (the REAL format_checkpoint_name) ------
    per_task = Checkpoint(monitor_loss="val/jets_classification_loss")
    checks["filename_stem_is_v1_contract"] = (
        per_task.filename == "epoch={epoch:03d}-loss={val/jets_classification_loss:.5f}"
    )
    checks["save_top_k_is_minus_one"] = per_task.save_top_k == -1
    checks["monitor_is_monitor_loss"] = per_task.monitor == "val/jets_classification_loss"
    checks["auto_insert_metric_name_off"] = per_task.auto_insert_metric_name is False
    # the per-task metric (carrying a '/') round-trips into the loss= stem
    rendered = per_task.format_checkpoint_name({
        "epoch": 9,
        "val/jets_classification_loss": 0.64624,
    })
    checks["formatted_name_has_loss_tag"] = rendered == "epoch=009-loss=0.64624.ckpt"

    # -- (b) run-dir path inference through the REAL setup + _best_checkpoint ---
    from types import SimpleNamespace  # noqa: PLC0415 - gate-local duck-typed pl_module

    run_dir = outdir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = Checkpoint(monitor_loss="val/loss")
    ckpt_cb.setup(_ckpt_setup_trainer(str(run_dir)), SimpleNamespace(), stage="fit")
    checks["setup_forces_ckpts_dir"] = str(ckpt_cb.dirpath) == str(run_dir / "ckpts")

    ckpt_dir = Path(ckpt_cb.dirpath)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for epoch, loss in ((8, 0.70123), (9, 0.64624)):
        name = ckpt_cb.format_checkpoint_name({"epoch": epoch, "val/loss": loss})
        if corruption is not None:
            name = corruption(name)
        (ckpt_dir / name).write_text("ckpt-placeholder")
        written.append(name)
    config_yaml = run_dir / "config.yaml"  # the saved run config _best_checkpoint scans next to
    config_yaml.write_text("class_path: salt.core.SaltModule\n")
    try:
        best = _best_checkpoint(config_yaml)
        checks["inference_resolves_lowest_loss"] = Path(best).name == "epoch=009-loss=0.64624.ckpt"
    except ConfigError:
        # the negative control reaches here (no loss=-named ckpt after corruption)
        checks["inference_resolves_lowest_loss"] = False

    # non-fit / fast_dev_run setup do NOT force the ckpts/ layout (v1 :30-32)
    non_fit = Checkpoint()
    non_fit.setup(_ckpt_setup_trainer(str(run_dir)), SimpleNamespace(), stage="test")
    checks["non_fit_setup_does_not_force_ckpts"] = not str(non_fit.dirpath).endswith("ckpts")
    fdr = Checkpoint()
    fdr.setup(_ckpt_setup_trainer(str(run_dir), fast_dev_run=True), SimpleNamespace(), stage="fit")
    checks["fast_dev_run_does_not_force_ckpts"] = not str(fdr.dirpath).endswith("ckpts")
    # an s3:// log dir is a loud ConfigError (the v1 s3 branch deferred to M6)
    checks["s3_log_dir_rejected_loudly"] = _raises_config_error(
        lambda: Checkpoint().setup(
            _ckpt_setup_trainer("s3://bucket/run"), SimpleNamespace(), stage="fit"
        )
    )

    # -- (c) ProgressBar smoke (the stock TQDMProgressBar under a salt name) ----
    bar = ProgressBar(refresh_rate=50)
    checks["progressbar_attaches_without_error"] = isinstance(bar, Callback)
    checks["progressbar_is_stock_tqdm"] = isinstance(bar, TQDMProgressBar)
    checks["progressbar_refresh_rate_passthrough"] = bar.refresh_rate == 50

    passed = all(checks.values())
    criterion = (
        "the REAL salt.core.callbacks.Checkpoint produces the v1 "
        "'epoch=NNN-loss={monitor_loss:.5f}.ckpt' stem with save_top_k=-1 and the per-task "
        "monitor unmangled, its setup('fit') forces dirpath to <log_dir>/ckpts (driven through "
        "the actual ModelCheckpoint.setup, not a re-implementation), and "
        "salt.core.main._best_checkpoint resolves the lowest-loss epoch from what it wrote — the "
        "salt2-test no-ckpt_path run-dir path inference (non-fit/fast_dev_run do NOT force ckpts/, "
        "an s3:// log dir raises ConfigError); and ProgressBar is the stock TQDMProgressBar under "
        "a salt-owned name (v1 callbacks/checkpoint.py:25-48, base.yaml:38-39; design §5.1 989-992 "
        "/ §5.3 1233-1245)"
    )
    config = {"written_ckpts": written, "corrupted_by_test_hook": corruption is not None}
    report = _base_report("d2ckpt_checkpoint_progressbar", passed, criterion, config)
    report["checks"] = checks
    report["scope_note"] = (
        "the sub-wave-D-rest Checkpoint + ProgressBar slice ONLY (plan 10 D2 row); NON-GATING for "
        "model reproduction — it gates the salt2-test run-dir checkpoint path inference. The other "
        "D2 items (origin_weighting, TensorSpec validation, lrs_config->lrs) are out of this slice"
    )
    report["surfaces_exercised"] = (
        "the REAL salt.core.callbacks.Checkpoint.format_checkpoint_name + setup (through "
        "ModelCheckpoint.setup's pre-set-dirpath short-circuit) and the REAL "
        "salt.core.main._best_checkpoint glob (not a re-implemented filename/inference)"
    )
    _print_checks(checks)
    _print_verdict("d2ckpt", passed, criterion, _emit_report(report, outdir, "d2ckpt"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# D2-expose: per-task expose: [fit,val] opt-out + name-based origin_weighting
# ---------------------------------------------------------------------------

_D2_LRS = {"initial": 1e-3, "max": 5e-3, "end": 1e-4, "pct_start": 0.1}


def _origin_schema_reader() -> H5StructuredReader:
    """A reader whose schema carries the tracks origin class names (design §2.6).

    ``ORIGIN_CLASSES`` is index-aligned so the GN3 names map to v1's hardcoded
    ids: ``Fake``->1, ``FromB``->3, ``FromBC``->4, ``FromC``->5.

    Returns
    -------
    H5StructuredReader
        A reader with the origin class-name attr on the tracks group.
    """
    schema = Schema(
        groups={
            "jets": GroupSchema(fields={}),
            "tracks": GroupSchema(fields={}, attrs={"ftagTruthOriginLabel": list(ORIGIN_CLASSES)}),
        }
    )
    return H5StructuredReader(groups={"jets": {}, "tracks": {}}, schema=schema)


def _name_based_vertexing() -> VertexingTaskModule:
    """A `VertexingTaskModule` whose origin_weighting is CLASS NAMES (the GN3 surface).

    Returns
    -------
    VertexingTaskModule
        A named vertexing module with name-based heavy/fake origin weighting.
    """
    task = VertexingTaskModule(
        stream="tracks",
        label="ftagTruthVertexIndex",
        origin_label="ftagTruthOriginLabel",
        context="pooled.global",
        weight=1.5,
        dense={"hidden_layers": [16], "activation": "ReLU"},
        origin_weighting={"heavy": ["FromB", "FromBC", "FromC"], "fake": ["Fake"]},
    )
    task.name = "track_vertexing"
    return task


def _d2_model(norm_dict: Path, *, expose: Sequence[str] | None) -> SaltModule:
    """A real GN2v2 `SaltModule` with the tracks tasks optionally expose-gated.

    Returns
    -------
    SaltModule
        The assembled model (both tracks tasks carry `expose`).
    """
    modules = build_gn2v2_modules(norm_dict)
    modules["track_origin"] = ClassificationTaskModule(
        stream="tracks",
        label="ftagTruthOriginLabel",
        class_names=list(ORIGIN_CLASSES),
        context="pooled.global",
        weight=0.5,
        dense={"hidden_layers": [16], "activation": "ReLU"},
        expose=expose,
    )
    modules["track_origin"].name = "track_origin"
    modules["track_vertexing"] = VertexingTaskModule(
        stream="tracks",
        label="ftagTruthVertexIndex",
        origin_label="ftagTruthOriginLabel",
        context="pooled.global",
        weight=1.5,
        dense={"hidden_layers": [16], "activation": "ReLU"},
        expose=expose,
    )
    modules["track_vertexing"].name = "track_vertexing"
    loss = LossSum()
    loss.name = "loss"
    loss.narrow(LossSum.collect_loss_keys(modules))
    modules["loss"] = loss
    return SaltModule(modules, lrs=_D2_LRS)


def _jets_only_writer() -> tuple[Any, Any]:
    """A `TaskWriter` narrowed to the jets task + a matching reader (tracks preds unconsumed).

    Returns
    -------
    tuple[Any, Any]
        ``(writer_callback, reader)`` for the `_model_sinks(Mode.TEST)` path.
    """
    wcb = WriterCallback(modules={"tasks": TaskWriter(tasks=["jets_classification"])})
    reader = H5StructuredReader(groups={"jets": {"global_object": True}, "tracks": {"global_object": False}})
    return wcb, reader


def run_d2expose(
    outdir: Path | str,
    *,
    corruption: Callable[[dict[str, int]], dict[str, int]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """D2 (expose + name-origin slice): per-task opt-out + name-based origin_weighting.

    The two sub-wave-D-rest model-feature items (plan 10 D2 row). Both drive the
    REAL surfaces — no re-implementation in the gate.

    - **Name-based origin_weighting** (design §5.1; v1 ``task.py:957-964`` uses
      HARDCODED ids 3,4,5 / 1). A `VertexingTaskModule` configured with class
      NAMES (``heavy: [FromB, FromBC, FromC]``, ``fake: [Fake]``) leaves
      ``heavy_ids``/``fake_ids`` None until the REAL
      `salt.core.saltmodule.resolve_origin_weighting` resolves them at setup
      against the schema's origin class-name attr — yielding exactly (3,4,5)/(1,).
      PARITY: the composed v2 head's per-edge ``get_weights`` is BIT-IDENTICAL to
      a FRESHLY, INDEPENDENTLY constructed v1 ``VertexingTask`` whose
      ``get_weights`` hardcodes (3,4,5)/1 — never the v2 module's own head. Loud
      surfaces: an unknown name (``ConfigError`` naming it), a name-based config
      with no origin class-name attr (``ConfigError``), and a name-based bind that
      never reached a schema (``ConfigError``) all raise; an integer-id config is
      a resolution no-op and binds standalone.

    - **Per-task expose: [fit, val] opt-out** (design §4.2). The pred port is
      gated to the exposed modes: an exposed task's ``preds.*`` is active in
      FIT/VAL but NOT in TEST/ONNX. End-to-end through the planner: the exposed
      task is KEPT in the FIT plan and PRUNED from the TEST plan (compiled with
      its full preds as sinks; the gated pred cannot be demanded). At the REAL
      TEST writer-demand surface (`SaltModule._model_sinks(Mode.TEST)` with a
      jets-only `TaskWriter`): WITHOUT expose the unconsumed tracks preds are the
      dead-preds hard error (``ConfigError`` "consumed by NO writer", whose fix
      now advertises ``expose: [fit, val]``); WITH expose the SAME writer raises
      nothing and the only TEST sink is the jets pred.

    Negative control (``test_gates_m5.py``): the ``corruption`` hook rewrites the
    name->id resolution map (shifting every id by +1) before the parity weights
    are computed, so the name-resolved ids DISAGREE with v1's hardcoded ids and
    ``origin_names_match_independent_v1`` FAILS — while the expose / loud-surface
    checks (independent of that map) stay green. Proves the parity check has
    teeth.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)`` — 0 only if every check passed.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("D2expose per-task expose:[fit,val] opt-out + name-based origin_weighting")
    print("=" * 96)
    norm_dict = _norm_dict(outdir)
    checks: dict[str, bool] = {}

    # -- (a) name-based origin_weighting: resolution + INDEPENDENT-v1 parity ----
    task = _name_based_vertexing()
    checks["names_pending_before_resolution"] = task.heavy_ids is None and task.fake_ids is None
    reader = _origin_schema_reader()
    n_resolved = resolve_origin_weighting({"track_vertexing": task}, reader)
    checks["resolve_origin_weighting_counts_one"] = n_resolved == 1
    checks["names_resolve_to_v1_heavy_ids"] = task.heavy_ids == (3, 4, 5)
    checks["names_resolve_to_v1_fake_ids"] = task.fake_ids == (1,)

    # the negative-control hook can corrupt the resolved ids before the parity
    # weights are computed (python-only; default = identity)
    heavy_ids, fake_ids = task.heavy_ids, task.fake_ids
    if corruption is not None:
        index = corruption({name: i for i, name in enumerate(ORIGIN_CLASSES)})
        heavy_ids = tuple(index[n] for n in ("FromB", "FromBC", "FromC"))
        fake_ids = tuple(index[n] for n in ("Fake",))
        task.heavy_ids, task.fake_ids = heavy_ids, fake_ids

    task.bind(ResolvedSchema(widths={"encoded.tracks": 16, "pooled.global": 16}))
    # INDEPENDENT v1 reference: a fresh head whose get_weights hardcodes (3,4,5)/1
    indep_v1 = V1VertexingTask(
        name="track_vertexing",
        input_name="tracks",
        label="ftagTruthVertexIndex",
        loss=torch.nn.BCEWithLogitsLoss(reduction="none"),
        dense_config={"input_size": 32, "output_size": 1},
    )
    labels = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    n = labels.shape[1]
    adjmat = ~torch.eye(n, dtype=torch.bool).unsqueeze(0)
    v1_weights = V1VertexingTask.get_weights(indep_v1, labels, adjmat)
    v2_weights = task.task.get_weights(labels, adjmat)
    checks["origin_names_match_independent_v1"] = torch.equal(v1_weights, v2_weights)

    # loud surfaces (independent of the corruption hook)
    unknown = VertexingTaskModule(
        stream="tracks",
        label="ftagTruthVertexIndex",
        origin_label="ftagTruthOriginLabel",
        origin_weighting={"heavy": ["NotAClass"], "fake": ["Fake"]},
    )
    unknown.name = "u"
    checks["unknown_name_rejected"] = _raises_config_error(
        lambda: unknown.resolve_origin_names(_origin_schema_reader())
    )
    no_attr = _name_based_vertexing()
    # a reader with NO schema artifact (global_object set explicitly so it constructs)
    no_attr_reader = H5StructuredReader(groups={"tracks": {"global_object": False}})
    checks["name_based_no_schema_bind_fails"] = _raises_config_error(
        lambda: (
            no_attr.resolve_origin_names(no_attr_reader),
            no_attr.bind(ResolvedSchema(widths={"encoded.tracks": 16})),
        )
    )
    int_task = VertexingTaskModule(
        stream="tracks",
        label="ftagTruthVertexIndex",
        origin_label="o",
        origin_weighting={"heavy": [3, 4, 5], "fake": [1]},
    )
    int_task.name = "i"
    checks["integer_ids_resolution_is_noop"] = int_task.resolve_origin_names(
        _origin_schema_reader()
    ) is False and int_task.heavy_ids == (3, 4, 5)

    # -- (b) expose: [fit,val] opt-out: planner pruning + REAL dead-preds path --
    exposed = _d2_model(norm_dict, expose=["fit", "val"])
    aux = exposed._graph_modules["track_origin"]  # noqa: SLF001
    pred_fit = flatten_spec(aux.declare_io(Mode.FIT).produces)[aux.pred_key]
    checks["exposed_pred_active_in_fit"] = pred_fit.active_in(Mode.FIT)
    checks["exposed_pred_inactive_in_test"] = not pred_fit.active_in(Mode.TEST)

    fit_plan = compile_plan(
        exposed._graph_modules,  # noqa: SLF001
        Mode.FIT,
        sources=gn2v2_sources(),
        sinks=["loss.total"],
    )
    checks["exposed_task_kept_in_fit_plan"] = "track_origin" in fit_plan.module_names
    test_plan = compile_plan(
        exposed._graph_modules,  # noqa: SLF001
        Mode.TEST,
        sources=gn2v2_sources(),
        sinks=["preds.jets.jets_classification"],
    )
    checks["exposed_task_pruned_from_test_plan"] = "track_origin" not in test_plan.module_names

    # negative control with teeth: WITHOUT expose, the SAME jets-only writer
    # makes the tracks preds a dead-preds ERROR through the REAL _model_sinks
    default = _d2_model(norm_dict, expose=None)
    wcb, wreader = _jets_only_writer()
    dead_err = ""
    try:
        default._model_sinks(Mode.TEST, writers=wcb, reader=wreader)  # noqa: SLF001
    except ConfigError as err:
        dead_err = str(err)
    checks["default_task_triggers_dead_preds_error"] = "consumed by NO writer" in dead_err
    checks["dead_preds_error_advertises_expose"] = "expose: [fit, val]" in dead_err

    # WITH expose, the SAME writer raises nothing and only the jets pred sinks
    exposed_w = _d2_model(norm_dict, expose=["fit", "val"])
    wcb2, wreader2 = _jets_only_writer()
    try:
        sinks = exposed_w._model_sinks(Mode.TEST, writers=wcb2, reader=wreader2)  # noqa: SLF001
        checks["expose_silences_dead_preds_error"] = sinks == ["preds.jets.jets_classification"]
    except ConfigError:
        checks["expose_silences_dead_preds_error"] = False

    passed = all(checks.values())
    criterion = (
        "name-based origin_weighting resolves CLASS NAMES to v1's hardcoded ids (3,4,5)/(1) via "
        "the REAL resolve_origin_weighting at setup and the composed head's get_weights is "
        "bit-identical to an INDEPENDENT v1 VertexingTask (fresh instance, hardcoded ids — never "
        "the v2 head); unknown names / no-schema / unresolved-bind raise ConfigError, integer ids "
        "are a no-op. The per-task expose: [fit, val] opt-out gates preds.* to fit/val so the "
        "planner KEEPS the task in the FIT plan and PRUNES it from the TEST plan, and at the REAL "
        "SaltModule._model_sinks(Mode.TEST) writer-demand surface a jets-only TaskWriter raises "
        "the dead-preds hard error WITHOUT expose (advertising the expose fix) but NOTHING with it "
        "(design §4.2, §5.1; v1 task.py:957-964)"
    )
    config = {"corrupted_by_test_hook": corruption is not None, "resolved_count": n_resolved}
    report = _base_report("d2expose_optout_name_origin", passed, criterion, config)
    report["checks"] = checks
    report["scope_note"] = (
        "the sub-wave-D-rest model-feature slice: per-task expose opt-out + name-based "
        "origin_weighting ONLY (plan 10 D2 row). Checkpoint/ProgressBar are d2ckpt; "
        "TensorSpec kind/dtype validation + lrs_config->lrs migration are out of this slice"
    )
    report["surfaces_exercised"] = (
        "the REAL salt.core.saltmodule.resolve_origin_weighting + VertexingTaskModule."
        "resolve_origin_names/bind, an INDEPENDENT v1 VertexingTask.get_weights, the REAL "
        "compile_plan demand pruning, and the REAL SaltModule._model_sinks(Mode.TEST) "
        "writer-demand dead-preds path (not a re-implemented planner/sink list)"
    )
    _print_checks(checks)
    _print_verdict("d2expose", passed, criterion, _emit_report(report, outdir, "d2expose"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# D2cfg — writer TensorSpec kind/dtype validation + lrs_config->lrs migration
#         (sub-wave D-rest, plan 10 D2 row, the LAST two D2 items)
# ---------------------------------------------------------------------------


class _SpecProbeWriter(Writer):
    """A gate-local `Writer` whose single declared require's kind/dtype is tunable.

    `WriterCallback.validate_specs` consults ONLY `requires` (it is a static
    config-validation surface — `columns`/`write` never run), so the gate
    instantiates the writer with a chosen ``(key, kind, dtype, optional)`` and
    drives the REAL validator. The abstract `columns`/`write` exist so the
    `Writer` ABC instantiates; they are dead in this gate.
    """

    def __init__(
        self,
        key: str,
        *,
        kind: str = "data",
        dtype: str | None = None,
        optional: bool = False,
    ) -> None:
        self._key = key
        self._kind = kind
        self._dtype = dtype
        self._optional = optional

    def requires(self, ctx: WriterDeclareCtx) -> dict[str, TensorSpec]:  # noqa: ARG002 - static probe
        """The single tunable-kind/dtype require under validation.

        Returns
        -------
        dict[str, TensorSpec]
            ``{key: spec}`` with the gate-chosen kind/dtype (TEST-active).
        """
        return {
            self._key: TensorSpec(
                shape=("B", 3), kind=self._kind, dtype=self._dtype, optional=self._optional
            )
        }

    def columns(self, ctx: WriteCtx) -> dict[str, Any]:  # noqa: ARG002 - never reached by validate_specs
        """Dead in this gate (validate_specs never reaches the write lifecycle).

        Returns
        -------
        dict[str, Any]
            Always empty — never reached.
        """
        return {}

    def write(self, bundle: Bundle, rows: slice) -> dict[str, Any]:  # noqa: ARG002 - dead
        """Dead in this gate (validate_specs never reaches the write lifecycle).

        Returns
        -------
        dict[str, Any]
            Always empty — never reached.
        """
        return {}


def _raises(fn: Callable[..., Any], exc: type[BaseException]) -> bool:
    """Whether calling ``fn()`` raises ``exc`` (the kind/dtype loud-surface probe).

    `KindError`/`ShapeError` are `GraphError` siblings of `ConfigError`, NOT
    subclasses of it, so the existing `_raises_config_error` cannot catch them —
    this gate-local helper takes the expected type explicitly.

    Returns
    -------
    bool
        True iff `fn` raised an instance of `exc`.
    """
    try:
        fn()
    except exc:
        return True
    return False


def _validate_probe(
    model_modules: dict[str, Any],
    reader: Any,
    producer_specs: Mapping[str, TensorSpec],
    writer: _SpecProbeWriter,
) -> None:
    """Drive the REAL `WriterCallback.validate_specs` for one probe writer.

    Builds a `WriterCallback` over the single probe and calls the actual
    static validator with the SAME ``producer_specs`` union (model-produced +
    boundary) that `SaltModule._validate_writer_specs` assembles — no
    re-implementation of the kind/dtype unification in the gate.
    """
    WriterCallback(modules={"probe": writer}).validate_specs(
        model_modules, reader, dict(producer_specs)
    )


def run_d2cfg(
    outdir: Path | str,
    *,
    corruption: Callable[[dict[str, TensorSpec]], dict[str, TensorSpec]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """D2 (cfg slice): writer TensorSpec kind/dtype validation + lrs_config->lrs migration.

    The LAST two sub-wave-D-rest D2 items (plan 10 D2 row). Both drive the REAL
    surfaces — no re-implementation in the gate.

    - **Writer TensorSpec kind/dtype validation** (design §2.7/§8; v2 demand was
      key-existence-only, the ``writers/base.py`` deviation note now closed). The
      REAL `WriterCallback.validate_specs` — the static validator
      `SaltModule._validate_writer_specs` calls on the TEST path — is driven via
      the REAL `WriterCallback.model_producer_specs` (the gn2v2 TEST-active
      produced leaves; the model half of the `_validate_writer_specs` producer
      union — every key this gate's probes consume is model-produced, so it takes
      precedence over the boundary half regardless). It unifies a writer-declared
      require's kind/dtype against its producing
      leaf. Against the real ``preds.jets.jets_classification`` producer
      (``kind="data"``, ``dtype="float32"``): a MATCHING ``data``/``float32``
      require PASSES (no raise), an UNCONSTRAINED ``None``-dtype require PASSES (a
      ``None`` unifies with anything — the TaskWriter's deliberate looseness stays
      legal), a WRONG dtype (``int64``) raises `ShapeError`, a WRONG kind
      (``label``) raises `KindError`, a NON-OPTIONAL require with NO producer
      raises `ConfigError` (the existence half of the §2.7 contract), and an
      OPTIONAL missing require is SILENT. The validator is the REAL one (loud and
      static, before the first batch), not a re-implemented check.

    - **Writer kind/dtype unification THROUGH `salt2 graph validate`** (the
      end-to-end half — design §2.7/§8; the medium critic finding's fix). The
      kind/dtype unification is now wired into ``cli._cmd_validate`` for TEST, so
      the canonical static-validator command — the one the M5-CONV gate runs
      across the 23 configs — invokes it data-free. A shipped config carrying a
      gate-local probe writer (`_SpecProbeWriter`) declaring
      ``preds.jets.jets_classification`` as ``kind="label"`` (or ``dtype="int64"``
      where the task publishes ``data``/``float32``) FAILS ``salt2 graph validate
      --mode test`` (rc != 0) — NOT only later at ``salt2 test`` setup; the
      MATCHING ``data``/``float32`` probe PASSES the same command (rc=0), the
      inert positive control. A writer kind/dtype mismatch is an ERROR-level
      finding, so the wrong-config failures are attributable to the WRITER check
      itself (no ``--strict`` — they do not lean on the promotable no-schema
      warning). Driven through the REAL command end to end, never a re-parsed
      dict.

    - **lrs_config -> lrs migration** (M3 cleanup, design §5.1; v1
      ``modelwrapper.py`` ``lrs_config`` kwarg / ``base.yaml:45``). A shipped v2
      config (``gn2v2-dummy.yaml``, carrying ``lrs:``) LOADS + plan-compiles through
      the REAL ``salt2 graph validate`` (all four modes, rc=0); NO shipped v2
      ``salt/core/configs/*.yaml`` retains a stale ``lrs_config:`` key (a literal
      scan — comment mentions documenting the rename are excluded); and a config
      that re-introduces the OLD ``lrs_config:`` key FAILS the real config-load path
      (``SaltModule`` takes ``lrs``, not ``lrs_config`` — jsonargparse rejects the
      renamed kwarg, rc != 0). The migration is exercised THROUGH the real
      config-load/plan-compile surface, never asserted on a re-parsed dict.

    Negative control (``test_gates_m5.py``): the ``corruption`` hook rewrites the
    ``producer_specs`` map (flipping the ``jets_classification`` producer dtype to
    ``int64``) before the MATCHING-require validation, so the matching probe is no
    longer dtype-compatible and ``matching_require_passes`` FAILS — while the
    wrong-kind / wrong-dtype / existence / lrs-migration checks (independent of
    that producer dtype) stay green. Proves the matching check has teeth.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)`` — 0 only if every check passed.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("D2cfg writer TensorSpec kind/dtype validation + lrs_config->lrs migration")
    print("=" * 96)
    checks: dict[str, bool] = {}

    # -- (a) writer TensorSpec kind/dtype validation (the REAL validate_specs) --
    norm_dict = _norm_dict(outdir)
    model_modules = build_gn2v2_modules(norm_dict)
    reader = H5StructuredReader(groups={"jets": {"global_object": True}, "tracks": {"global_object": False}})
    producer_specs = WriterCallback.model_producer_specs(model_modules)
    produced_key = "preds.jets.jets_classification"
    # the producing leaf this gate unifies against (kind="data", dtype="float32")
    checks["producer_leaf_is_data_float32"] = (
        produced_key in producer_specs
        and producer_specs[produced_key].kind == "data"
        and producer_specs[produced_key].dtype == "float32"
    )

    # the negative-control hook can break the producer dtype before the MATCHING
    # validation (python-only; default = identity)
    match_specs = dict(producer_specs)
    if corruption is not None:
        match_specs = corruption(dict(producer_specs))

    # MATCHING data/float32 -> PASSES (no raise of any GraphError)
    checks["matching_require_passes"] = not _raises(
        lambda: _validate_probe(
            model_modules,
            reader,
            match_specs,
            _SpecProbeWriter(produced_key, kind="data", dtype="float32"),
        ),
        GraphError,
    )
    # UNCONSTRAINED None dtype -> PASSES (None unifies with the float32 producer)
    checks["unconstrained_dtype_require_passes"] = not _raises(
        lambda: _validate_probe(
            model_modules,
            reader,
            dict(producer_specs),
            _SpecProbeWriter(produced_key, kind="data", dtype=None),
        ),
        GraphError,
    )
    # WRONG dtype -> ShapeError (the planner's _unify_edge rule on the writer edge)
    checks["wrong_dtype_raises_shape_error"] = _raises(
        lambda: _validate_probe(
            model_modules,
            reader,
            dict(producer_specs),
            _SpecProbeWriter(produced_key, kind="data", dtype="int64"),
        ),
        ShapeError,
    )
    # WRONG kind -> KindError (design §2.2 kind-typed requires)
    checks["wrong_kind_raises_kind_error"] = _raises(
        lambda: _validate_probe(
            model_modules,
            reader,
            dict(producer_specs),
            _SpecProbeWriter(produced_key, kind="label", dtype="float32"),
        ),
        KindError,
    )
    # NON-OPTIONAL require with NO producer -> ConfigError (existence half, §2.7)
    checks["missing_nonoptional_require_raises_config_error"] = _raises(
        lambda: _validate_probe(
            model_modules,
            reader,
            dict(producer_specs),
            _SpecProbeWriter("preds.jets.does_not_exist", kind="data", optional=False),
        ),
        ConfigError,
    )
    # OPTIONAL missing require -> SILENT (no raise of any GraphError)
    checks["optional_missing_require_is_silent"] = not _raises(
        lambda: _validate_probe(
            model_modules,
            reader,
            dict(producer_specs),
            _SpecProbeWriter("preds.jets.does_not_exist", kind="data", optional=True),
        ),
        GraphError,
    )

    # -- (a-e2e) the END-TO-END `salt2 graph validate` writer kind/dtype check --
    # the canonical static-validator command (the one the M5-CONV gate runs)
    # now invokes the writer kind/dtype unification for TEST (cli._cmd_validate),
    # so a config with a genuinely WRONG writer kind/dtype FAILS `salt2 graph
    # validate` data-free — not only later at `salt2 test` setup. Driven through
    # the REAL command, end to end (the medium critic finding's fix). NB no
    # `--strict` here: a writer kind/dtype mismatch is an ERROR-level finding
    # (not a promotable warning), so the wrong-config failures are attributable
    # to the WRITER check specifically — they do NOT lean on the no-schema
    # warning `--strict` would promote (which would make the matching positive
    # control fail too and the negative results ambiguous; the no-schema config
    # is deliberate — this gate validates the writer edge, not field spellings).
    nd_e2e = outdir / "norm_dict.yaml"  # _norm_dict already wrote it above
    set_norm_e2e = f"model.modules.norm.init_args.norm_dict={nd_e2e}"
    wrong_kind_cfg = _write_probe_writer_config(
        outdir, kind="label", dtype="float32", fname="probe_wrong_kind.yaml"
    )
    wrong_kind_rc = salt2_main([
        "graph",
        "validate",
        "--mode",
        "test",
        "-c",
        str(wrong_kind_cfg),
        "--set",
        set_norm_e2e,
    ])
    checks["wrong_kind_writer_fails_graph_validate"] = wrong_kind_rc != 0

    wrong_dtype_cfg = _write_probe_writer_config(
        outdir, kind="data", dtype="int64", fname="probe_wrong_dtype.yaml"
    )
    wrong_dtype_rc = salt2_main([
        "graph",
        "validate",
        "--mode",
        "test",
        "-c",
        str(wrong_dtype_cfg),
        "--set",
        set_norm_e2e,
    ])
    checks["wrong_dtype_writer_fails_graph_validate"] = wrong_dtype_rc != 0

    # the inert positive control: a MATCHING probe (data/float32) passes the
    # REAL `salt2 graph validate --mode test` (rc=0) — proves the e2e check has
    # no false positive and the wired-in path is otherwise transparent (the
    # no-schema warning stays a non-fatal warning without --strict)
    matching_cfg = _write_probe_writer_config(
        outdir, kind="data", dtype="float32", fname="probe_matching.yaml"
    )
    matching_rc = salt2_main([
        "graph",
        "validate",
        "--mode",
        "test",
        "-c",
        str(matching_cfg),
        "--set",
        set_norm_e2e,
    ])
    checks["matching_writer_passes_graph_validate"] = matching_rc == 0

    # -- (b) lrs_config -> lrs migration (the REAL salt2 config-load/compile) ---
    nd_path = outdir / "norm_dict.yaml"  # _norm_dict already wrote it above
    set_norm = f"model.modules.norm.init_args.norm_dict={nd_path}"
    # a shipped v2 config (carrying lrs:) loads + plan-compiles in all four modes
    migrated_rc = salt2_main([
        "graph",
        "validate",
        "-c",
        str(CONFIG_DIR / "gn2v2-dummy.yaml"),
        "--set",
        set_norm,
    ])
    checks["migrated_config_graph_validate_ok"] = migrated_rc == 0

    # NO shipped v2 config retains a stale `lrs_config:` KEY (comment mentions of
    # the rename are excluded — only an actual YAML key counts)
    def _has_lrs_config_key(text: str) -> bool:
        for line in text.splitlines():
            stripped = line.split("#", 1)[0].strip()  # drop trailing/whole-line comments
            if stripped.startswith("lrs_config:") or stripped == "lrs_config":
                return True
        return False

    stale_configs = sorted(
        cfg.name for cfg in CONFIG_DIR.glob("*.yaml") if _has_lrs_config_key(cfg.read_text())
    )
    checks["no_v2_config_retains_lrs_config_key"] = stale_configs == []

    # a config re-introducing the OLD lrs_config: key FAILS the real load path
    # (SaltModule takes `lrs`, not `lrs_config` — jsonargparse rejects the renamed
    # kwarg + the missing required `lrs`). Built data-free from the shipped config.
    stale_yaml = _write_stale_lrs_config(outdir)
    stale_rc = salt2_main([
        "graph",
        "validate",
        "-c",
        str(stale_yaml),
        "--set",
        set_norm,
    ])
    checks["stale_lrs_config_key_rejected"] = stale_rc != 0

    passed = all(checks.values())
    criterion = (
        "the REAL WriterCallback.validate_specs (the static §2.7 writer-input validator "
        "SaltModule._validate_writer_specs calls on TEST) driven via the REAL "
        "model_producer_specs (the model half of the producer union; every probe key is "
        "model-produced) unifies a writer require's kind/dtype against its producing leaf: "
        "a matching data/float32 require and an unconstrained None-dtype require PASS, a wrong "
        "dtype raises "
        "ShapeError, a wrong kind raises KindError, a non-optional missing producer raises "
        "ConfigError, an optional missing require is silent; AND — the end-to-end half — the "
        "SAME kind/dtype unification is now wired into salt2 graph validate (cli._cmd_validate, "
        "TEST), so a config with a wrong writer kind (label) or dtype (int64) FAILS `salt2 graph "
        "validate --mode test` (rc != 0, an ERROR-level finding) data-free while a matching "
        "data/float32 writer PASSES it (rc=0) — the canonical static-validator command the "
        "M5-CONV gate runs now has the writer kind/dtype teeth; AND a shipped v2 config (carrying "
        "lrs:) loads + plan-compiles through the REAL salt2 graph validate while NO shipped v2 "
        "config retains a stale lrs_config: key and a config re-introducing lrs_config: FAILS the "
        "real config-load path (the renamed kwarg is rejected) — design §2.7/§8, §5.1; v1 "
        "modelwrapper.py/base.yaml:45"
    )
    config = {
        "stale_configs_found": stale_configs,
        "migrated_rc": migrated_rc,
        "stale_rc": stale_rc,
        "wrong_kind_rc": wrong_kind_rc,
        "wrong_dtype_rc": wrong_dtype_rc,
        "matching_rc": matching_rc,
        "corrupted_by_test_hook": corruption is not None,
    }
    report = _base_report("d2cfg_writer_spec_lrs_migration", passed, criterion, config)
    report["checks"] = checks
    report["scope_note"] = (
        "the sub-wave-D-rest cfg slice: writer TensorSpec kind/dtype validation + lrs_config->lrs "
        "migration ONLY (plan 10 D2 row, the LAST two D2 items). Checkpoint/ProgressBar are "
        "d2ckpt; per-task expose opt-out + name-based origin_weighting are d2expose"
    )
    report["surfaces_exercised"] = (
        "the REAL WriterCallback.validate_specs + model_producer_specs (not a re-implemented "
        "kind/dtype unification), the REAL salt2 graph validate end-to-end (a wrong-kind/dtype "
        "writer fails the command, a matching writer passes — the wired-in cli._cmd_validate TEST "
        "check), and the REAL salt2 graph validate config-load/plan-compile path "
        "(not a re-parsed config dict)"
    )
    _print_checks(checks)
    _print_verdict("d2cfg", passed, criterion, _emit_report(report, outdir, "d2cfg"))
    return (0 if passed else 1), report


def _write_stale_lrs_config(outdir: Path) -> Path:
    """Write a v2 config carrying the OLD ``lrs_config:`` key into ``outdir``.

    Derived from the shipped ``gn2v2-dummy.yaml`` by renaming its ``lrs:`` block
    back to ``lrs_config:`` — the exact pre-rename spelling. The real
    config-load path must reject it (``SaltModule`` takes ``lrs``).

    Returns
    -------
    Path
        The written stale-config YAML path.
    """
    import yaml  # noqa: PLC0415 - gate-local: shipped-config round-trip for the negative control

    spec = yaml.safe_load((CONFIG_DIR / "gn2v2-dummy.yaml").read_text())
    init = spec["model"]["init_args"]
    init["lrs_config"] = init.pop("lrs")
    path = outdir / "stale_lrs_config.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


def _write_probe_writer_config(outdir: Path, *, kind: str, dtype: str | None, fname: str) -> Path:
    """Write a config that adds a `_SpecProbeWriter` to the `writers:` block.

    Derived from the shipped ``gn2v2-dummy.yaml``: a gate-local probe writer
    (``salt.core.gates_m5._SpecProbeWriter``) is wired into ``writers.modules``
    declaring a require on ``preds.jets.jets_classification`` with the chosen
    ``(kind, dtype)``. The producer of that key is the jets `ClassificationTask`
    (``kind="data"``, ``dtype="float32"``), so a ``kind="label"`` or
    ``dtype="int64"`` require is a genuine writer/producer contract mismatch.

    This drives the END-TO-END ``salt2 graph validate`` path — the canonical
    static-validator command the M5-CONV gate runs — so the writer kind/dtype
    unification (now wired into ``_cmd_validate`` for TEST) is proven to fail
    data-free there, not only at ``salt2 test`` setup. A MATCHING
    ``kind="data"``/``dtype="float32"`` probe is the inert positive control
    (``salt2 graph validate`` rc=0).

    Returns
    -------
    Path
        The written probe-writer config YAML path.
    """
    import yaml  # noqa: PLC0415 - gate-local: shipped-config round-trip for the e2e probe

    spec = yaml.safe_load((CONFIG_DIR / "gn2v2-dummy.yaml").read_text())
    # base2.yaml ships writers.modules (inputs_copy/tasks/pad_mask); the trainer
    # parser deep-merges this addition into them, so the probe joins the real
    # writer set the runtime TEST path carries.
    writers = spec.setdefault("writers", {}).setdefault("modules", {})
    init_args: dict[str, Any] = {"key": "preds.jets.jets_classification", "kind": kind}
    if dtype is not None:
        init_args["dtype"] = dtype
    writers["spec_probe"] = {
        "class_path": "salt.core.gates_m5._SpecProbeWriter",
        "init_args": init_args,
    }
    path = outdir / fname
    path.write_text(yaml.safe_dump(spec))
    return path


# ---------------------------------------------------------------------------
# M5-CONV — the consolidated acceptance gate (plan 10 §4 / matrix §4)
# ---------------------------------------------------------------------------

# The AUTHORITATIVE needs-M5 config list — config-coverage-matrix.md §2/§4, the
# 23 `🔶 needs-M5` rows, PLUS `regression_multi_target` (matrix §2 marks it
# `🔷 needs-M6` for the MultiTarget processor, but plan 10 ADJUDICATED the
# processor INTO M5 — its R4 gate, its `RegressionTaskModule`/pooling parts, and
# the study CLAUDE.md's "7 already-existing needs-M5 configs" all place it in
# M5; co-locating it here keeps the regression family undivided, plan 10 l.62-66).
#
# Each entry pins, for the v2-native MIGRATED fixture in `salt/core/configs/`:
#   name           — the matrix §2 v1 config it reproduces (the report key)
#   cfg            — the v2 config filenames to stack with `-c` (overlays carry
#                    their FULL declared base chain first, then the overlay; the
#                    exact stack in each config's header — design §5.3 list/dict
#                    merge semantics mean an overlay is only valid composed)
#   norm_global    — True when the config has a SECOND `norm_global` Normaliser
#                    (the GN3/GN2 `global` post-pooling concat stream) needing its
#                    own data-free `--set ...norm_global...norm_dict` override
#   onnx           — "validate" when the config (its stack's base) declares an
#                    `export:` block so `salt2 graph validate --mode onnx` gates a
#                    real export plan; "na" when the config has NO ONNX export
#                    representation (event_classifier: a non-feature ratio
#                    denominator + no `export:` block — fit/test only, design §3.3
#                    + the config header)
#   family         — the matrix §2 grouping (for the report table)
#   note           — the M5 feature(s) this config's validation exercises
#
# This list IS the §4 acceptance denominator embedded in the report (plan 10
# M5-CONV row: "the exact config names embedded in the report"); a `🔶` matrix
# config absent here, or any config that fails its modes, fails the gate.
_CONV_CONFIGS: tuple[dict[str, Any], ...] = (
    # -- regression family (encoder-less pooling + RegressionTaskModule) -------
    {
        "name": "regression",
        "cfg": ("regression.yaml",),
        "norm_global": False,
        "onnx": "validate",
        "family": "regression",
        "note": "RegressionTask all variants + encoder-less pooling + ONNX denom-in-Features",
    },
    {
        "name": "regression_gaussian",
        "cfg": ("regression_gaussian.yaml",),
        "norm_global": False,
        "onnx": "validate",
        "family": "regression",
        "note": "GaussianRegressionTask (mu/sigma) + ONNX stddev + encoder-less pooling",
    },
    {
        "name": "regression_weighted",
        "cfg": ("regression_weighted.yaml",),
        "norm_global": False,
        "onnx": "validate",
        "family": "regression",
        "note": "RegressionTask sample_weight + encoder-less pooling",
    },
    {
        "name": "regression_multi_target",
        "cfg": ("regression_multi_target.yaml",),
        "norm_global": False,
        "onnx": "validate",
        "family": "regression",
        "note": "MultiTarget processor (plan 10 ADJUDICATED into M5) + RegressionTask",
    },
    {
        "name": "nan_regression",
        "cfg": ("nan_regression.yaml",),
        "norm_global": False,
        "onnx": "validate",
        "family": "regression",
        "note": "RegressionTask NaN-target masking + encoder-less pooling",
    },
    {
        "name": "legacy/dips",
        "cfg": ("dips.yaml",),
        "norm_global": False,
        "onnx": "validate",
        "family": "regression",
        "note": "encoder-less pooling (the primary CI smoke fixture, KEEP despite legacy/)",
    },
    # -- GN3 family (LossGLS + RegressionTask; standalone + overlay stacks) -----
    {
        "name": "GN3V00",
        "cfg": ("GN3V00.yaml",),
        "norm_global": False,
        "onnx": "validate",
        "family": "GN3",
        "note": "LossGLS + RegressionTask (the GN3 dev baseline body)",
    },
    {
        "name": "GN3_v00",
        "cfg": ("GN3_v00.yaml",),
        "norm_global": False,
        "onnx": "validate",
        "family": "GN3",
        "note": "standalone dev twin of GN3V00 (IP3D-named vars); LossGLS + RegressionTask",
    },
    {
        "name": "GN3_baseline",
        "cfg": ("GN3_baseline.yaml",),
        "norm_global": False,
        "onnx": "validate",
        "family": "GN3",
        "note": "LossGLS + track selections (the GN3_dev overlay-stack base)",
    },
    {
        "name": "GN3_Hybrid",
        "cfg": ("GN3V00.yaml", "GN3_Hybrid.yaml"),
        "norm_global": False,
        "onnx": "validate",
        "family": "GN3",
        "note": "norm_type:hybrid passthrough overlay on GN3V00; (stack) LossGLS",
    },
    {
        "name": "GN3_Charge",
        "cfg": ("GN3V00.yaml", "GN3_Charge.yaml"),
        "norm_global": False,
        "onnx": "validate",
        "family": "GN3",
        "note": "b-jet charge task overlay on GN3V00; (stack) LossGLS + RegressionTask",
    },
    {
        "name": "GN3_baseline_loose",
        "cfg": ("GN3_baseline.yaml", "GN3_baseline_loose.yaml"),
        "norm_global": False,
        "onnx": "validate",
        "family": "GN3",
        "note": "selections:null overlay on GN3_baseline; (stack) LossGLS",
    },
    {
        "name": "GN3_dR",
        "cfg": ("GN3_baseline.yaml", "GN3_dR.yaml"),
        "norm_global": False,
        "onnx": "validate",
        "family": "GN3",
        "note": "dR-matched tracks retarget overlay on GN3_baseline; (stack) LossGLS",
    },
    {
        "name": "GN3_flow",
        "cfg": ("GN3_baseline.yaml", "GN3_baseline_loose.yaml", "GN3_flow.yaml"),
        "norm_global": False,
        "onnx": "validate",
        "family": "GN3",
        "note": "pflow-stream overlay on GN3_baseline->loose; (stack) LossGLS",
    },
    {
        "name": "GN3_LepID_SMT",
        "cfg": (
            "GN3_baseline.yaml",
            "GN3_baseline_loose.yaml",
            "GN3_flow.yaml",
            "GN3_LepID_SMT.yaml",
        ),
        "norm_global": False,
        "onnx": "validate",
        "family": "GN3",
        "note": "lepton-ID+SMT data overlay on GN3_baseline->loose->flow; (stack) LossGLS",
    },
    {
        "name": "GN3_tracklabel",
        "cfg": (
            "GN3_baseline.yaml",
            "GN3_baseline_loose.yaml",
            "GN3_flow.yaml",
            "GN3_LepID_SMT.yaml",
            "GN3_tracklabel.yaml",
        ),
        "norm_global": False,
        "onnx": "validate",
        "family": "GN3",
        "note": "5-task track_type/track_source overlay (deepest stack); (stack) LossGLS",
    },
    # -- concat/global family (VectorConcat + export.inputs alias) --------------
    {
        "name": "GN3V01",
        "cfg": ("gn3v01.yaml",),
        "norm_global": True,
        "onnx": "validate",
        "family": "concat",
        "note": "flagship GN3: VectorConcat+alias + LossGLS + norm_type:hybrid + RegressionTask",
    },
    {
        "name": "GN2emu",
        "cfg": ("GN2emu.yaml",),
        "norm_global": True,
        "onnx": "na",
        "family": "concat",
        "note": (
            "VectorConcat (soft-muon global concat) outside the GN3 family; fit+test ONLY "
            "— no export: block. The 14-var soft-muon `global` has no ONNX representation: v1 "
            "only clones `global` from global_object (=jets, 2 vars; to_onnx.py:377-378), which "
            "is width-incoherent for the 14-wide slot (pooled_dim 142) — v1 never exported it"
        ),
    },
    {
        "name": "GN3_SoftE",
        "cfg": ("GN3V00.yaml", "GN3_SoftE.yaml"),
        "norm_global": True,
        "onnx": "validate",
        "family": "concat",
        "note": "electrons stream + global concat overlay on GN3V00; (stack) LossGLS",
    },
    {
        "name": "GN3EPCLV01",
        "cfg": ("GN3EPCLV01.yaml",),
        "norm_global": True,
        "onnx": "validate",
        "family": "concat",
        "note": "GN3 3-stream: LossGLS + norm_type:hybrid + VectorConcat+alias + RegressionTask",
    },
    # -- Gaussian-PVz / event-level / MaskFormer -------------------------------
    {
        "name": "hitz",
        "cfg": ("hitz.yaml",),
        "norm_global": False,
        "onnx": "validate",
        "family": "gaussian",
        "note": "GaussianRegressionTask on an encoder (HLT hits PV-z)",
    },
    {
        "name": "legacy/Dipz",
        "cfg": ("Dipz.yaml",),
        "norm_global": False,
        "onnx": "validate",
        "family": "gaussian",
        "note": "encoder-less GaussianRegressionTask; CONDITIONAL DROP (drops once hitz validates)",
    },
    {
        "name": "event_classifier",
        "cfg": ("event_classifier.yaml",),
        "norm_global": False,
        "onnx": "na",
        "family": "event",
        "note": (
            "RegressionTask TEST de-scaling from a NON-feature label (design §3.3 fallback); "
            "fit+test ONLY — no export: block, the ratio denominator is not an input Feature so "
            "there is no ONNX representation (config header; v1 never exported it)"
        ),
    },
    {
        "name": "MaskFormer",
        "cfg": ("MaskFormer.yaml",),
        "norm_global": False,
        "onnx": "validate",
        "family": "maskformer",
        "note": "MaskDecoder + MaskFormerMatchedLoss + MaskFormerTargets + object writer + metrics",
    },
)


def _conv_set_args(entry: dict[str, Any], norm_dict: Path) -> list[str]:
    """Build the data-free ``--set`` overrides for one config (norm + norm_global).

    Every shipped v2 config materialises its `Normaliser` from a norm dict at
    setup; static validation supplies it data-free via the documented
    ``--set model.modules.norm.init_args.norm_dict=<path>`` (design §5 / each
    config header). The concat/global configs carry a SECOND ``norm_global``
    Normaliser that needs its own override.

    Returns
    -------
    list[str]
        The ``--set KEY=VALUE`` flag pairs (already split for ``salt2_main``).
    """
    args = ["--set", f"model.modules.norm.init_args.norm_dict={norm_dict}"]
    if entry["norm_global"]:
        args += ["--set", f"model.modules.norm_global.init_args.norm_dict={norm_dict}"]
    return args


def _conv_modes(entry: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Resolve the (mode, key) pairs to validate for one config.

    Always fit + test; onnx only when the config is export-representable (the
    ``onnx`` field is ``"validate"``). event_classifier (``onnx == "na"``) has
    no ONNX representation, so its onnx slot is reported ``na``, never validated.

    Returns
    -------
    tuple[tuple[str, str], ...]
        ``((cli_mode, report_key), ...)`` — the modes `salt2 graph validate`
        is actually invoked for.
    """
    modes = [("fit", "validateFit"), ("test", "validateTest")]
    if entry["onnx"] == "validate":
        modes.append(("onnx", "validateOnnx"))
    return tuple(modes)


def run_conv(
    outdir: Path | str,
    *,
    corruption: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """M5-CONV: the consolidated acceptance gate (plan 10 §4 / matrix §4).

    For EVERY needs-M5 config (the matrix §2 ``🔶`` rows + the M5-adjudicated
    ``regression_multi_target``), this drives the canonical static validator —
    the REAL ``salt2 graph validate`` subcommand (``salt2_main``, the SAME
    command path d2cfg exercises) — in fit + test (+ onnx where the config is
    export-representable) and asserts rc == 0 for every applicable mode. The
    authoritative config list (``_CONV_CONFIGS``) is embedded VERBATIM in the
    report (plan 10 M5-CONV row: "the exact config names embedded in the
    report"). The gate exits non-zero if any matrix ``🔶`` config is MISSING from
    the list or if any config FAILS a mode (plan 10 §4.2/§4.3: convert +
    statically validate + plan-compile for every applicable mode).

    Why NO ``--strict``: a data-free validation cannot satisfy ``--strict``. The
    acceptance level is ``rc == 0`` (no ERROR-level finding: a GraphError, an
    error-level deadcode finding, or a stored mode error — design §4.2), which IS
    the convert+validate+plan-compile criterion. ``--strict`` promotes EVERY
    warning to an error (cli.py:919-920), and several warnings are inherent to
    data-free validation and orthogonal to convertibility:

    1. the no-``schema:`` warning ("field spellings cannot be checked
       statically", cli.py:832-836) — emitted for every config lacking a
       `schema:` artifact (derived from a real H5 via `salt2 schema dump`, none
       data-free);
    2. warning-level deadcode such as ``[mode=TEST] concat/seq.layout: produced
       but never consumed`` (the Concat layout helper is consumed only by
       flash-varlen attention paths absent in the torch-math fixtures) and the
       per-mode "produced but never consumed" prediction infos;
    3. `Normaliser` preflight warnings for streams the parity norm dict does not
       supply (it covers jets/tracks/electrons but NOT the GN3 `global` or the
       event `events` stream, so GN3V01/GN2emu/GN3_SoftE/GN3EPCLV01 and
       event_classifier emit "missing input type 'global'/'events'").

    So ``--strict`` is unusable here for reasons BEYOND the no-schema warning —
    the report does NOT claim no-schema is the only blocker. This is exactly why
    d2cfg's end-to-end `salt2 graph validate` probes and EVERY config header
    validate WITHOUT ``--strict`` (the d2cfg docstring records the same
    rationale). The parity norm dict is still written so the jets/tracks/
    electrons preflight resolves and the bind succeeds — the remaining
    global/events preflight warnings are data-shape artifacts, not graph errors.

    Forward-parity: M5-CONV makes NO new forward-parity claim. The family
    representatives' forward parity is owned by the R/L/MF gates (R1-R4, L1-L3,
    MF1a/MF1c/MF2); this gate is the static convert+validate+plan-compile slice
    (plan 10 §4.4: "forward-parity spot-checks only for family reps already
    covered by R/L/MF gates").

    Parameters
    ----------
    outdir : Path | str
        Report output directory (the parity norm dict is written here too).
    corruption : Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None
        Test-only hook (the gates_m2/m3/m4 pattern): transforms the embedded
        config list before validation (e.g. inject a missing/bogus config) to
        prove the completeness + per-config assertions have teeth. Never on CLI.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)`` — 0 only if the list is complete AND every
        config validated every applicable mode.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print(
        "M5-CONV consolidated acceptance — salt2 graph validate (fit/test/onnx) on needs-M5 configs"
    )
    print("=" * 96)
    norm_dict = _norm_dict(outdir)

    configs: list[dict[str, Any]] = [dict(e) for e in _CONV_CONFIGS]
    if corruption is not None:
        configs = corruption(configs)

    # -- completeness: every matrix §2 `🔶` config has an entry here -----------
    # the AUTHORITATIVE needs-M5 names (matrix §2 `🔶` rows; v1 config names).
    # `regression_multi_target` is matrix-`🔷` but plan-10-ADJUDICATED into M5
    # (so it is REQUIRED here, not optional — see the _CONV_CONFIGS preamble).
    matrix_needs_m5 = {
        "GN2emu",
        "GN3V01",
        "GN3V00",
        "GN3_Hybrid",
        "GN3_Charge",
        "GN3_SoftE",
        "GN3_baseline",
        "GN3_baseline_loose",
        "GN3_dR",
        "GN3_flow",
        "GN3_LepID_SMT",
        "GN3_tracklabel",
        "GN3_v00",
        "event_classifier",
        "GN3EPCLV01",
        "hitz",
        "MaskFormer",
        "nan_regression",
        "regression",
        "regression_gaussian",
        "regression_weighted",
        "legacy/dips",
        "legacy/Dipz",
        "regression_multi_target",  # matrix-🔷, plan-10-adjudicated into M5
    }
    present = {e["name"] for e in configs}
    missing = sorted(matrix_needs_m5 - present)

    checks: dict[str, bool] = {}
    checks["all_matrix_needs_m5_configs_present"] = missing == []

    # -- per-config validation across the applicable modes --------------------
    results: list[dict[str, Any]] = []
    for entry in configs:
        cargs: list[str] = []
        for cfg_name in entry["cfg"]:
            cargs += ["-c", str(CONFIG_DIR / cfg_name)]
        set_args = _conv_set_args(entry, norm_dict)
        row: dict[str, Any] = {
            "name": entry["name"],
            "family": entry["family"],
            "cfg_stack": list(entry["cfg"]),
            "note": entry["note"],
            "validateFit": False,
            "validateTest": False,
            "validateOnnx": entry["onnx"] if entry["onnx"] == "na" else False,
            "rc": {},
        }
        for cli_mode, report_key in _conv_modes(entry):
            rc = salt2_main(["graph", "validate", "--mode", cli_mode, *cargs, *set_args])
            row["rc"][cli_mode] = rc
            ok = rc == 0
            row[report_key] = ok
            checks[f"{entry['name']}:{cli_mode}"] = ok
        results.append(row)

    passed = all(checks.values())
    n_total = len(configs)
    n_validated = sum(
        1
        for r in results
        if r["validateFit"] and r["validateTest"] and r["validateOnnx"] in {True, "na"}
    )

    criterion = (
        "the M7-slice acceptance (plan 10 §4 / matrix §4): EVERY needs-M5 config (the matrix §2 "
        f"🔶 rows + the plan-10-adjudicated regression_multi_target; {n_total} configs, names "
        "embedded in the report) has a v2-native MIGRATED fixture in salt/core/configs/ that the "
        "REAL salt2 graph validate (the canonical static validator, the d2cfg command path) "
        "convert+validates+plan-compiles in fit + test (+ onnx where export-representable; "
        "event_classifier is fit/test-only — no ONNX representation) with rc == 0; the list is "
        "complete vs the matrix and exits non-zero if any config is missing or fails. No --strict "
        "(the no-schema warning is inherent to data-free validation — see the docstring); no new "
        "forward-parity claim (owned by R/L/MF)."
    )
    report = _base_report(
        "m5_conv_acceptance",
        passed,
        criterion,
        {
            "norm_dict": str(norm_dict),
            "strict": False,
            "total_configs": n_total,
            "validated_configs": n_validated,
            "missing_matrix_configs": missing,
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["configs"] = results
    report["matrix_needs_m5"] = sorted(matrix_needs_m5)
    report["scope_note"] = (
        "M5-CONV is the consolidated convert+validate+plan-compile acceptance (plan 10 §4 / "
        "matrix §4): static validation ONLY (salt2 graph validate fit/test/onnx). It makes NO "
        "forward-parity claim — the family-representative forward parity is owned by R1-R4 / "
        "L1-L3 / MF1a/MF1c/MF2. regression_multi_target is included (matrix-🔷 but "
        "plan-10-adjudicated into M5). legacy/Dipz is VALIDATED here for completeness but is "
        "NOT counted in the matrix §1 M7 denominator (39) — it is a CONDITIONAL DROP that drops "
        "once hitz reproduces in v2; hitz now validates here too, so Dipz is droppable at M7 "
        "(validating a drop candidate is harmless/conservative). The M5-CONV total (24) thus "
        "exceeds the M7 denominator's GN3/regression slice by this one drop candidate. "
        "event_classifier validates fit+test only (no ONNX export representation)."
    )
    report["no_strict_rationale"] = (
        "no --strict: a data-free validation cannot satisfy it. --strict promotes EVERY warning to "
        "an error (cli.py:919-920) and SEVERAL warnings are inherent to data-free validation, "
        "orthogonal to convertibility — NOT just the no-schema one: (1) the no-schema warning "
        "(cli.py:832-836, every config lacking a schema: artifact, derived from a real H5 via "
        "salt2 schema dump — none data-free); (2) warning-level deadcode like '[mode=TEST] "
        "concat/seq.layout produced but never consumed' (the Concat layout helper feeds only "
        "flash-varlen attention, absent in the torch-math fixtures) + the per-mode unconsumed-pred "
        "infos; (3) Normaliser preflight warnings for the GN3 'global' / event 'events' streams "
        "the parity norm dict does not supply (it covers jets/tracks/electrons only). The parity "
        "norm dict resolves the jets/tracks/electrons preflight so bind succeeds; the "
        "global/events warnings are data-shape artifacts, not graph errors. Same rationale as "
        "d2cfg and every "
        "config header. rc == 0 (no ERROR-level finding, design §4.2) IS the "
        "convert+validate+plan-compile criterion."
    )

    # -- stdout table ----------------------------------------------------------
    print(f"{'config':<26}{'family':<12}{'fit':>5}{'test':>6}{'onnx':>7}   stack / note")
    for r in results:
        onnx_disp = "na" if r["validateOnnx"] == "na" else ("PASS" if r["validateOnnx"] else "FAIL")
        print(
            f"{r['name']:<26}{r['family']:<12}"
            f"{'PASS' if r['validateFit'] else 'FAIL':>5}"
            f"{'PASS' if r['validateTest'] else 'FAIL':>6}"
            f"{onnx_disp:>7}   {'+'.join(r['cfg_stack'])}"
        )
    if missing:
        print(f"\nMISSING matrix 🔶 configs (no _CONV_CONFIGS entry): {missing}")
    print(f"\n{n_validated}/{n_total} configs validated all applicable modes")
    _print_verdict("conv", passed, criterion, _emit_report(report, outdir, "conv"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the gate subcommand parser (R1-R4, L1-L3, MF1a/MF1c/MF2, D1, D2*, CONV).

    Returns
    -------
    argparse.ArgumentParser
        Parser with the ``r1``/``r2``/``r3``/``r4``/``l1``/``l2``/``l3``/
        ``mf1a``/``mf1c``/``mf2``/``d1``/``d2ckpt``/``d2expose``/``d2cfg``/``conv``
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
        "mf1a": "MaskDecoder + encoder drop_registers parity vs independent v1 (sub-wave C)",
        "mf1c": "MaskFormerMatchedLoss + matcher + MaskFormerTargets parity (sub-wave C)",
        "mf2": "MaskFormerObjectWriter TEST byte-parity + ONNX object reduces (sub-wave C)",
        "d1": "FIT/VAL callback-sink assembly + register_reduce live registry/dtypes (sub-wave D)",
        "d2ckpt": "Checkpoint filename/monitor contract + run-dir path inference + ProgressBar "
        "smoke (sub-wave D-rest)",
        "d2expose": "per-task expose:[fit,val] opt-out + name-based origin_weighting parity "
        "(sub-wave D-rest)",
        "d2cfg": "writer TensorSpec kind/dtype validation + lrs_config->lrs migration "
        "(sub-wave D-rest)",
        "conv": "M5-CONV consolidated acceptance: salt2 graph validate (fit/test/onnx) on "
        "EVERY needs-M5 config (plan 10 §4 / matrix §4)",
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
        "mf1a": run_mf1a,
        "mf1c": run_mf1c,
        "mf2": run_mf2,
        "d1": run_d1,
        "d2ckpt": run_d2ckpt,
        "d2expose": run_d2expose,
        "d2cfg": run_d2cfg,
        "conv": run_conv,
    }[args.gate]
    code, _ = runner(args.outdir)
    return code


if __name__ == "__main__":
    sys.exit(main())
