"""Toy end-to-end demo of the M1 graph kernel (plan 03 stage F, design §9.5 M1 gate 3).

Drives the full static loop on the toy graph from
``salt/tests/core/configs/toy.yaml`` (toy fixtures only — no physics):
validate every primary mode (design §4.1), write the fit/test plan tables
(§4.4), write the per-mode dead-output report (§4.2), render the graph
(§4.3), then actually execute the FIT and TEST plans on random source
tensors via the executor (§3.2) under debug read tracking.

Artifacts written into ``--outdir``: ``plan_fit.txt``, ``plan_test.txt``,
``deadcode.txt``, ``graph.dot`` (always) plus the rendered ``graph.svg`` +
``graph.pdf``, produced via the ``dot`` binary baked into the salt container.

Usage (the experiment ``do_run`` payload)::

    python -m salt.core.demo_m1 --outdir /path/to/outputs [--config cfg.yaml]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from salt.core.cli import load_config
from salt.core.cli import main as cli_main
from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import GraphError
from salt.core.graph.executor import Executor
from salt.core.graph.planner import Plan, compile_plan
from salt.core.graph.spec import Mode, TensorSpec

__all__ = ["main"]

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "tests" / "core" / "configs" / "toy.yaml"


def _run_captured(argv: list[str]) -> tuple[int, str]:
    """Run a ``salt2`` CLI command in-process, capturing stdout.

    Returns
    -------
    tuple[int, str]
        The exit code and the captured stdout text.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = cli_main(argv)
    return code, buffer.getvalue()


def _random_value(spec: TensorSpec, batch_size: int, gen: torch.Generator) -> torch.Tensor:
    """Build one random source tensor matching a spec (symbolic dims -> batch_size).

    Returns
    -------
    torch.Tensor
        Random floats, small random ints, or all-False bools per the dtype.
    """
    shape = tuple(batch_size if isinstance(dim, str) else dim for dim in (spec.shape or ("B",)))
    dtype = getattr(torch, spec.dtype or "float32")
    if dtype == torch.bool:
        return torch.zeros(shape, dtype=torch.bool)
    if dtype.is_floating_point:
        return torch.randn(shape, generator=gen, dtype=dtype)
    return torch.randint(0, 2, shape, generator=gen, dtype=dtype)


def _random_sources(plan: Plan, batch_size: int, seed: int) -> Bundle:
    """Seed a bundle with random tensors for every non-optional plan source.

    Returns
    -------
    Bundle
        The input bundle the caller hands to `Executor.run` (design §3.2).
    """
    gen = torch.Generator().manual_seed(seed)
    bundle = Bundle()
    for key, spec in plan.sources.items():
        if not spec.optional:
            bundle.set(key, _random_value(spec, batch_size, gen))
    return bundle


def _execute(config_path: Path, batch_size: int, seed: int) -> None:
    """Compile and execute the FIT and TEST plans on random source tensors.

    `GraphError` from compilation or (debug) execution propagates to `main`.
    """
    cfg = load_config(config_path)
    for mode in (Mode.FIT, Mode.TEST):
        plan = compile_plan(cfg.modules, mode, cfg.sources, schema=cfg.schema, sinks=cfg.sinks)
        bundle = _random_sources(plan, batch_size, seed)
        Executor(plan).run(bundle, debug=True)
        print(
            f"[demo] executed {mode.name} plan: steps={', '.join(plan.module_names)}; "
            f"bundle keys={', '.join(bundle.keys())}"
        )
        if "losses.total" in bundle:
            print(f"[demo]   losses.total = {bundle.get('losses.total').item():.4f}")
        if "preds.x" in bundle:
            print(f"[demo]   preds.x shape = {tuple(bundle.get('preds.x').shape)}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the toy end-to-end demo and write its artifacts (M1 gate 3).

    Returns
    -------
    int
        Process exit code: 0 on success, 1 on kernel/config errors.
    """
    parser = argparse.ArgumentParser(
        prog="python -m salt.core.demo_m1", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--outdir", type=Path, required=True, help="artifact output directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="M1 graph config YAML")
    parser.add_argument("--batch-size", type=int, default=4, help="toy batch size")
    parser.add_argument("--seed", type=int, default=0, help="random seed for source tensors")
    args = parser.parse_args(argv)
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    config = str(args.config)
    print(f"[demo] config: {config}")

    # 1. validate every primary mode (design §4.1) — errors are fatal
    if (code := cli_main(["graph", "validate", "-c", config])) != 0:
        return code

    # 2. plan tables for fit and test (design §4.4)
    for mode_name in ("fit", "test"):
        code, text = _run_captured(["graph", "plan", "-c", config, "--mode", mode_name])
        if code != 0:
            return code
        path = outdir / f"plan_{mode_name}.txt"
        path.write_text(text)
        print(f"[demo] wrote {path}")

    # 3. per-mode dead-output report (design §4.2)
    code, text = _run_captured(["graph", "deadcode", "-c", config])
    if code != 0:
        return code
    path = outdir / "deadcode.txt"
    path.write_text(text)
    print(f"[demo] wrote {path}")

    # 4. graph plot (design §4.3): DOT always + SVG/PDF rendered via the dot binary
    plot_args = ["graph", "plot", "-c", config, "--mode", "fit", "-o", str(outdir / "graph.svg")]
    if (code := cli_main(plot_args)) != 0:
        return code

    # 5. execute the fit and test plans on random tensors (design §3.2)
    try:
        _execute(args.config, args.batch_size, args.seed)
    except GraphError as err:
        print(f"salt.core.graph.{type(err).__name__}: {err}", file=sys.stderr)
        return 1
    print(f"[demo] done — artifacts in {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
