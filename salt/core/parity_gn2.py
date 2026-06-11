"""GN2 v1 <-> v2 forward-parity gate (plan 04, stage 3).

Instantiates the small v1 GN2 `ModelWrapper` from the stage-1 recipe
(`salt.tests.core.gn2_fixture` — deterministic dummy data with a zero-track
jet, plus a parity norm dict with DISTINCT per-variable constants so
norm-constant order/scale wiring is excitable), builds the v2 module graph
by WRAPPING the same live
``nn.Module`` instances (`salt.core.nn.from_v1` — weights identical by
construction, zero copies), compiles a TEST plan with the M1 planner,
executes it through the M1 `Executor` in debug mode (undeclared bundle reads
and in-place tensor mutations raise), and compares EVERY
``preds.<stream>.<task>`` leaf of the executed bundle against the v1
``ModelWrapper.forward`` output on an identically cloned batch — plus the
bonus intermediates ``encoded.seq`` <-> ``embed_xs`` and ``pooled.global``
<-> ``global_rep``.

Because the two sides share instances and op order, the PASS criterion is
BITWISE equality (``torch.equal``; the stage-1 probe confirmed CPU bitwise
repeatability with ``attn_type="torch-math"``). Any difference is a WIRING
bug — concat order, context prepend, pad-mask polarity / dict order,
padded-position handling, task-input assembly — exactly the failure class
this gate exists to catch before M2 training. Multi-stream concat order and
per-stream mask dict order are only genuinely exercised by the two-stream
run (``--with-electrons``, GN2e-style second sequence stream); the default
GN2 fixture has a single sequence stream, where any concat order is
trivially the identity. A non-bitwise leaf can pass
ONLY with a written justification registered in `JUSTIFIED_NONBITWISE`
(printed in the report) AND a max abs diff <= 1e-6; the criterion is never
silently downgraded. No leaf currently needs (or has) a justification.

Shapes are compared FIRST: for the vertexing edge output an ``E`` mismatch
means a pad-mask wiring bug, and a value diff on mismatched shapes would be
meaningless.

Artifacts written into ``--outdir``: ``parity_report.json`` (machine-readable
report including the executed plan steps and the plan hash — proof the v2
side ran through the Executor) plus the dummy ``norm_dict.yaml`` /
``class_dict.yaml`` the v1 model was built from.

Usage (the experiment ``do_run`` payload)::

    python -m salt.core.parity_gn2 --outdir /path/to/outputs \
        [--batch-size N] [--n-tracks T] [--seed S] [--model-seed S]

Exit code 0 only if every compared leaf passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from salt.core.graph import Bundle, Executor, GraphModule, Mode, compile_plan
from salt.core.nn import from_v1, v1_sinks, v1_sources
from salt.tests.core.gn2_fixture import build_test_gn2, make_gn2_batch, v1_forward

__all__ = ["JUSTIFIED_NONBITWISE", "LeafComparison", "compare_leaf", "main", "run_parity"]

BITWISE_FALLBACK_ATOL = 1e-6
"""Ceiling for a JUSTIFIED non-bitwise leaf — never applied without a justification."""

JUSTIFIED_NONBITWISE: dict[str, str] = {}
"""Per-leaf written justifications for accepting a non-bitwise (<= atol) diff.

Empty by design: both sides share the same nn.Module instances and op order,
so every leaf is expected bitwise-identical (stage-2 confirmed 0.0 on all
leaves and intermediates). Any future entry must name the leaf key and spell
out the EVIDENCED op-reassociation that makes bitwise unattainable; the
justification is printed in the report. Do not use this to paper over diffs.
"""

# Fixture model dims (small test scale; GN2 nominal is embed 256 / out 128).
_EMBED_DIM = 16
_OUT_DIM = 16
_NUM_LAYERS = 2
_NUM_HEADS = 2

_N_ELECTRONS = 6
"""Electron positions per jet in the two-stream (``--with-electrons``) run.

Deliberately different from the default ``n_tracks`` so the two streams have
distinguishable token widths in the sequence layout.
"""


@dataclass
class LeafComparison:
    """One compared output leaf: a v2 bundle key vs its v1 forward reference."""

    key: str
    v1_ref: str
    ref_shape: tuple[int, ...]
    got_shape: tuple[int, ...]
    ref_dtype: str
    got_dtype: str
    max_abs_diff: float | None
    bitwise: bool
    passed: bool
    note: str = ""


def compare_leaf(key: str, v1_ref: str, ref: Tensor, got: Tensor) -> LeafComparison:
    """Compare one v2 bundle leaf against its v1 reference tensor.

    Shape is checked first (an ``E`` mismatch on the vertexing edge output is
    a pad-mask wiring bug and makes a value diff meaningless), then dtype,
    then bitwise equality. A non-bitwise leaf passes only with a registered
    `JUSTIFIED_NONBITWISE` entry and a diff <= `BITWISE_FALLBACK_ATOL`.

    Parameters
    ----------
    key : str
        The v2 bundle key (report row identifier).
    v1_ref : str
        Human-readable name of the v1 preds entry compared against.
    ref : Tensor
        The v1 reference tensor.
    got : Tensor
        The executed v2 bundle tensor.

    Returns
    -------
    LeafComparison
        The comparison record for the report.
    """
    base: dict[str, Any] = {
        "key": key,
        "v1_ref": v1_ref,
        "ref_shape": tuple(ref.shape),
        "got_shape": tuple(got.shape),
        "ref_dtype": str(ref.dtype),
        "got_dtype": str(got.dtype),
    }
    if ref.shape != got.shape:
        return LeafComparison(
            **base,
            max_abs_diff=None,
            bitwise=False,
            passed=False,
            note="SHAPE mismatch — for edge outputs an E mismatch means a pad-mask wiring bug",
        )
    if ref.dtype != got.dtype:
        return LeafComparison(
            **base, max_abs_diff=None, bitwise=False, passed=False, note="dtype mismatch"
        )
    bitwise = torch.equal(ref, got)
    max_abs = 0.0 if bitwise else float((ref - got).abs().max().item())
    if bitwise:
        return LeafComparison(**base, max_abs_diff=max_abs, bitwise=True, passed=True)
    justification = JUSTIFIED_NONBITWISE.get(key)
    if justification is not None and max_abs <= BITWISE_FALLBACK_ATOL:
        return LeafComparison(
            **base,
            max_abs_diff=max_abs,
            bitwise=False,
            passed=True,
            note=f"non-bitwise but justified (<= {BITWISE_FALLBACK_ATOL:g}): {justification}",
        )
    return LeafComparison(
        **base, max_abs_diff=max_abs, bitwise=False, passed=False, note="values differ"
    )


def run_parity(
    outdir: Path | str,
    *,
    batch_size: int = 6,
    n_tracks: int = 10,
    p_valid: float = 0.6,
    seed: int = 123,
    model_seed: int = 42,
    with_electrons: bool = False,
    modules_hook: Callable[[dict[str, GraphModule]], dict[str, GraphModule]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run the full parity gate: build, execute both sides, compare, report.

    Parameters
    ----------
    outdir : Path | str
        Output directory for ``parity_report.json`` and the dummy norm dict.
    batch_size : int, optional
        Jets per batch, by default 6.
    n_tracks : int, optional
        Track positions per jet (>= 2 so padded positions exist and
        mask-polarity bugs are excitable), by default 10.
    p_valid : float, optional
        Probability a track position is valid, by default 0.6.
    seed : int, optional
        Batch seed (local generator), by default 123.
    model_seed : int, optional
        ``torch.manual_seed`` for v1 parameter init, by default 42.
    with_electrons : bool, optional
        Add a GN2e-style second sequence stream (`_N_ELECTRONS` positions per
        jet), by default False. Only this variant genuinely exercises
        multi-stream concat order and per-stream mask dict order.
    modules_hook : Callable, optional
        TEST-ONLY hook: receives the `from_v1` module dict and may replace
        entries with deliberately mis-wired variants (the negative controls
        in ``test_parity_gn2.py`` prove the gate FAILS on them). Instance
        names are re-asserted to the dict keys after the hook. Never used by
        the CLI.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)`` — exit code 0 only if every compared leaf
        passed; the report is what was written to ``parity_report.json``.

    Raises
    ------
    ValueError
        If `batch_size` < 1 or `n_tracks` < 2.
    RuntimeError
        If the fixture batch lacks a zero-track jet (``batch_size >= 2`` is
        expected to dedicate jet 1 to the all-padded production edge case).
    """
    if batch_size < 1:
        raise ValueError("run_parity: batch_size must be >= 1")
    if n_tracks < 2:
        raise ValueError("run_parity: n_tracks must be >= 2 (need real padded positions)")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # (a)+(b) deterministic dummy norm dict + small v1 GN2 (stage-1 recipe:
    # torch-math attention, explicit class_names, eval mode) and a batch with
    # real padding from a local generator.
    v1 = build_test_gn2(
        outdir,
        embed_dim=_EMBED_DIM,
        out_dim=_OUT_DIM,
        num_layers=_NUM_LAYERS,
        num_heads=_NUM_HEADS,
        seed=model_seed,
        with_electrons=with_electrons,
    )
    inputs, pad_masks = make_gn2_batch(
        batch_size=batch_size,
        n_tracks=n_tracks,
        p_valid=p_valid,
        seed=seed,
        n_electrons=_N_ELECTRONS if with_electrons else 0,
    )
    # Per-stream valid-constituent counts go into the report; the batch must
    # contain a ZERO-track jet (production edge case the gate must cover —
    # stage-4 critic finding) whenever the batch is big enough to hold one.
    valid_counts = {k: (~mask).sum(dim=-1).tolist() for k, mask in pad_masks.items()}
    if batch_size >= 2 and 0 not in valid_counts["tracks"]:
        raise RuntimeError("run_parity: fixture batch must contain a zero-track jet")

    # v1 reference forward: clones its dicts internally (InputNorm rebinds the
    # inputs dict's keys; the encoder inserts a "REGISTERS" pad-mask key),
    # runs under no_grad with labels=None.
    ref = v1_forward(v1, inputs, pad_masks)

    # (c) v2 graph wrapping the SAME instances; compile + validate the TEST plan.
    modules = from_v1(v1)
    if modules_hook is not None:
        modules = modules_hook(modules)
        for key, module in modules.items():
            module.name = key  # re-assert instance name == dict key (design §2.2)
    sinks = v1_sinks(v1)
    plan = compile_plan(modules, Mode.TEST, sources=v1_sources(v1), sinks=sinks)

    # (d) v2 forward on an identically CLONED batch through the M1 Executor.
    # debug=True: undeclared reads raise UndeclaredAccessError, in-place
    # mutation of bundle tensors raises MutationError — mechanical proof of
    # the wiring discipline.
    bundle = Bundle({
        "inputs": {k: v.clone() for k, v in inputs.items()},
        "masks": {k: v.clone() for k, v in pad_masks.items()},
    })
    with torch.no_grad():
        Executor(plan).run(bundle, debug=True)

    # (e) compare every preds.<stream>.<task> leaf + bonus intermediates.
    leaves = []
    for sink in sinks:
        _, stream, task = sink.split(".")
        leaves.append(
            compare_leaf(sink, f'preds["{stream}"]["{task}"]', ref[stream][task], bundle.get(sink))
        )
    intermediates = [
        compare_leaf(
            "encoded.seq", 'preds["embed_xs"]', ref["embed_xs"], bundle.get("encoded.seq")
        ),
        compare_leaf(
            "pooled.global", 'preds["global_rep"]', ref["global_rep"], bundle.get("pooled.global")
        ),
    ]
    passed = all(c.passed for c in (*leaves, *intermediates))

    report: dict[str, Any] = {
        "gate": "gn2_forward_parity",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "passed": passed,
        "criterion": (
            "bitwise torch.equal on every compared leaf; a non-bitwise leaf may pass only with "
            f"a registered written justification AND max abs diff <= {BITWISE_FALLBACK_ATOL:g}"
        ),
        "justified_nonbitwise": dict(JUSTIFIED_NONBITWISE),
        "config": {
            "batch_size": batch_size,
            "n_tracks": n_tracks,
            "p_valid": p_valid,
            "batch_seed": seed,
            "model_seed": model_seed,
            "with_electrons": with_electrons,
            "n_electrons": _N_ELECTRONS if with_electrons else 0,
            "valid_counts": valid_counts,
            "embed_dim": _EMBED_DIM,
            "out_dim": _OUT_DIM,
            "num_layers": _NUM_LAYERS,
            "num_heads": _NUM_HEADS,
            "attn_type": "torch-math",
            "device": "cpu",
            "torch_version": torch.__version__,
            "miswired_by_test_hook": modules_hook is not None,
        },
        "plan": {
            "mode": plan.mode.name,
            "plan_hash": plan.plan_hash,
            "module_names": list(plan.module_names),
            "steps": [
                {"name": s.name, "requires": sorted(s.requires), "produces": sorted(s.produces)}
                for s in plan.steps
            ],
            "executed_bundle_keys": sorted(bundle.keys()),
            "executor_debug": True,
        },
        "leaves": [asdict(c) for c in leaves],
        "intermediates": [asdict(c) for c in intermediates],
    }
    report_path = outdir / "parity_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    _print_report(report, report_path)
    return (0 if passed else 1), report


def _fmt_row(record: dict[str, Any]) -> str:
    """Format one comparison record as a fixed-width table row.

    Returns
    -------
    str
        The formatted row.
    """
    if record["ref_shape"] == record["got_shape"]:
        shape = str(tuple(record["got_shape"]))
    else:
        shape = f"ref{tuple(record['ref_shape'])} != got{tuple(record['got_shape'])}"
    dtype = record["got_dtype"].removeprefix("torch.")
    diff = "n/a" if record["max_abs_diff"] is None else f"{record['max_abs_diff']:.3e}"
    bitwise = "yes" if record["bitwise"] else "NO"
    status = "PASS" if record["passed"] else "FAIL"
    return f"{record['key']:<34}{shape:<24}{dtype:<10}{diff:>12}{bitwise:>9}{status:>7}"


def _print_report(report: dict[str, Any], report_path: Path) -> None:
    """Print the human-readable parity table and executed-plan proof to stdout."""
    width = 96
    cfg = report["config"]
    plan = report["plan"]
    print("=" * width)
    print("GN2 v1 <-> v2 forward-parity gate (plan 04) — weight-sharing, bitwise criterion")
    print("=" * width)
    print(
        f"config: B={cfg['batch_size']} T={cfg['n_tracks']} p_valid={cfg['p_valid']} "
        f"batch_seed={cfg['batch_seed']} model_seed={cfg['model_seed']} "
        f"electrons={cfg['n_electrons'] if cfg['with_electrons'] else 'off'} | "
        f"embed={cfg['embed_dim']} out={cfg['out_dim']} layers={cfg['num_layers']} "
        f"heads={cfg['num_heads']} attn={cfg['attn_type']} torch={cfg['torch_version']}"
    )
    print(f"valid counts per stream (0 = all-padded edge case): {cfg['valid_counts']}")
    print(f"plan: mode={plan['mode']} hash={plan['plan_hash']}")
    print("executed v2 plan steps (M1 Executor, debug=True), in execution order:")
    for i, step in enumerate(plan["steps"], 1):
        print(f"  {i}. {step['name']:<22} -> {', '.join(step['produces'])}")
    keys = plan["executed_bundle_keys"]
    print(f"executed bundle keys ({len(keys)}): {', '.join(keys)}")
    print("-" * width)
    header = f"{'leaf (v2 bundle key)':<34}{'shape':<24}{'dtype':<10}{'max|diff|':>12}"
    print(header + f"{'bitwise':>9}{'status':>7}")
    print("-" * width)
    notes: list[tuple[str, str]] = []
    for record in report["leaves"]:
        print(_fmt_row(record))
        if record["note"]:
            notes.append((record["key"], record["note"]))
    print(f"{'-- bonus intermediates --':<34}")
    for record in report["intermediates"]:
        print(_fmt_row(record))
        if record["note"]:
            notes.append((record["key"], record["note"]))
    for key, note in notes:
        print(f"note [{key}]: {note}")
    print("=" * width)
    verdict = "PASS" if report["passed"] else "FAIL"
    print(f"PARITY GATE: {verdict} — criterion: {report['criterion']}")
    print(f"report: {report_path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the GN2 forward-parity gate from the command line.

    Returns
    -------
    int
        0 if every compared leaf passed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        prog="python -m salt.core.parity_gn2", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--outdir", type=Path, required=True, help="report output directory")
    parser.add_argument("--batch-size", type=int, default=6, help="jets per batch")
    parser.add_argument("--n-tracks", type=int, default=10, help="track positions per jet")
    parser.add_argument(
        "--seed", type=int, default=123, help="batch seed (model init uses --model-seed)"
    )
    parser.add_argument(
        "--model-seed", type=int, default=42, help="torch.manual_seed for v1 parameter init"
    )
    parser.add_argument(
        "--with-electrons",
        action="store_true",
        help="add a GN2e-style second sequence stream (exercises multi-stream concat order)",
    )
    args = parser.parse_args(argv)
    code, _report = run_parity(
        args.outdir,
        batch_size=args.batch_size,
        n_tracks=args.n_tracks,
        seed=args.seed,
        model_seed=args.model_seed,
        with_electrons=args.with_electrons,
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
