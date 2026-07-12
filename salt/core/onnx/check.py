"""Torch-vs-ONNX agreement checker.

The eager `OnnxAdapter` (the SAME module instances the trace saw, torch-math
forced) is the reference; onnxruntime (CPUExecutionProvider) evaluates the
exported graph on identical random inputs; outputs are addressed BY NAME from
the export config.

Default bars: float outputs at ``rtol=atol=1e-4`` with no-NaN and no-exact-zero
asserts, int8 aux outputs exact. The sweep covers every sequence length
0..max_length-1 (including the zero-token edge case) times `trials` random draws.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from salt.core.onnx.adapter import OnnxAdapter
from salt.core.onnx.config import stream_of_input_port

__all__ = ["CheckResult", "check_onnx", "compare_once", "make_session"]


@dataclass
class CheckResult:
    """Aggregated checker outcome over the sweep.

    `worst_abs_diff` is per output name (floats only — int8 outputs are
    exact-or-fail); `failures` carries one message per failed case;
    `int8_distinct` records the distinct values each int8 output took over the
    sweep — non-degeneracy evidence: a collapsed argmax/all-one-cluster
    union-find output would compare exactly while proving nothing.
    """

    passed: bool
    n_cases: int
    worst_abs_diff: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    int8_distinct: dict[str, list[int]] = field(default_factory=dict)


def make_session(onnx_path: str | Path):
    """Build the CPU onnxruntime session.

    Returns
    -------
    onnxruntime.InferenceSession
        Session on the CPUExecutionProvider with warning-level logging
        suppressed (unoptimised-subgraph noise).
    """
    import onnxruntime as ort  # noqa: PLC0415 - heavy import, checker-only

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    return ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"], sess_options=sess_options
    )


def _draw_inputs(
    adapter: OnnxAdapter, lengths: Mapping[str, int], gen: torch.Generator
) -> tuple[torch.Tensor, ...]:
    """Draw one random input tuple: globals ``[1, F]``, sequences ``[L, F]``."""
    drawn: list[torch.Tensor] = []
    for entry in adapter._positional:  # noqa: SLF001 - same-package checker
        width = len(adapter._field_list(entry.port))  # noqa: SLF001 - same-package checker
        if entry.sequence:
            length = lengths[stream_of_input_port(entry.port)]
            drawn.append(torch.rand(length, width, generator=gen))
        else:
            drawn.append(torch.rand(1, width, generator=gen))
    return tuple(drawn)


def compare_once(
    adapter: OnnxAdapter,
    session,
    lengths: Mapping[str, int],
    gen: torch.Generator,
    *,
    float_rtol: float = 1e-4,
    float_atol: float = 1e-4,
    forbid_zeros: bool = True,
    int8_distinct: MutableMapping[str, set[int]] | None = None,
) -> dict[str, float]:
    """Compare eager-adapter vs ONNX outputs for one random case, BY NAME.

    Parameters
    ----------
    adapter : OnnxAdapter
        The eager reference (torch-math forced at construction).
    session : onnxruntime.InferenceSession
        The exported model's session.
    lengths : Mapping[str, int]
        Per-sequence-stream token counts for this case.
    gen : torch.Generator
        Input RNG.
    float_rtol : float, optional
        Relative tolerance for float outputs, by default 1e-4.
    float_atol : float, optional
        Absolute tolerance for float outputs, by default 1e-4.
    forbid_zeros : bool, optional
        Assert float outputs contain no exact zeros (a dead-output canary),
        by default True.
    int8_distinct : MutableMapping[str, set[int]] | None, optional
        When given, the distinct values of each int8 ONNX output are
        accumulated into it (per output name) BEFORE the exactness assert,
        by default None.

    Returns
    -------
    dict[str, float]
        Max abs diff per float output name. Any mismatch, NaN, or
        exact-zero float output raises `AssertionError` (from the
        comparison asserts), naming the output and the failing lengths.
    """
    inputs = _draw_inputs(adapter, lengths, gen)
    with torch.no_grad():
        torch_outputs = adapter(*inputs)
    ort_inputs = {
        name: tensor.numpy() for name, tensor in zip(adapter.input_names, inputs, strict=True)
    }
    ort_names = [out.name for out in session.get_outputs()]
    ort_outputs = dict(zip(ort_names, session.run(None, ort_inputs), strict=True))
    where = f"lengths={dict(lengths)}"
    worst: dict[str, float] = {}
    for name, dtype, ref in zip(
        adapter.output_names, adapter.output_dtypes, torch_outputs, strict=True
    ):
        assert name in ort_outputs, f"ONNX model has no output {name!r} ({sorted(ort_outputs)})"
        got = ort_outputs[name]
        ref_np = ref.detach().numpy()
        if dtype == "int8":
            assert got.dtype == np.int8, f"{name!r}: ONNX dtype {got.dtype}, expected int8"
            if int8_distinct is not None:
                int8_distinct.setdefault(name, set()).update(int(v) for v in np.unique(got))
            assert np.array_equal(ref_np, got), (
                f"int8 output {name!r} mismatch at {where}: torch={ref_np.tolist()} "
                f"onnx={got.tolist()}"
            )
            continue
        assert not np.isnan(ref_np).any(), f"{name!r}: NaN in torch output at {where}"
        assert not np.isnan(got).any(), f"{name!r}: NaN in ONNX output at {where}"
        if forbid_zeros:
            assert not (ref_np == 0).any(), f"{name!r}: exact zero in torch output at {where}"
            assert not (got == 0).any(), f"{name!r}: exact zero in ONNX output at {where}"
        np.testing.assert_allclose(
            ref_np,
            got,
            rtol=float_rtol,
            atol=float_atol,
            err_msg=f"torch vs ONNX mismatch for output {name!r} at {where}",
        )
        # a per-token float output is EMPTY at the L=0 sweep sample (zero-token jet);
        # np.max over a zero-size array raises, so this is guarded (the diff IS 0.0
        # when there are no elements to differ).
        diff = np.abs(ref_np.astype(np.float64) - got.astype(np.float64))
        worst[name] = 0.0 if diff.size == 0 else float(np.max(diff))
    return worst


def check_onnx(
    adapter: OnnxAdapter,
    onnx_path: str | Path,
    *,
    max_length: int = 40,
    trials: int = 10,
    float_rtol: float = 1e-4,
    float_atol: float = 1e-4,
    forbid_zeros: bool = True,
    lengths_grid: Iterable[Mapping[str, int]] | None = None,
    seed: int = 42,
    fail_fast: bool = False,
) -> CheckResult:
    """Sweep the torch-vs-ONNX comparison.

    By default every sequence stream is swept TOGETHER over
    ``L = 0..max_length-1`` (including the zero-token edge case) with
    `trials` random draws each. `lengths_grid` overrides the sweep with
    explicit per-stream length combinations.

    Returns
    -------
    CheckResult
        Aggregated verdict; `passed` is False when any case failed
        (failures are collected unless `fail_fast`).

    Raises
    ------
    AssertionError
        Only with ``fail_fast=True`` — the first failing case propagates.
    """
    session = make_session(onnx_path)
    seq_streams = [
        stream_of_input_port(entry.port)
        for entry in adapter._positional  # noqa: SLF001 - same-package checker
        if entry.sequence
    ]
    if lengths_grid is None:
        cases: list[Mapping[str, int]] = [
            dict.fromkeys(seq_streams, length) for length in range(max_length)
        ]
    else:
        cases = [dict(entry) for entry in lengths_grid]
    gen = torch.Generator().manual_seed(seed)
    result = CheckResult(passed=True, n_cases=0)
    distinct: dict[str, set[int]] = {}
    for lengths in cases:
        for _ in range(trials):
            result.n_cases += 1
            try:
                worst = compare_once(
                    adapter,
                    session,
                    lengths,
                    gen,
                    float_rtol=float_rtol,
                    float_atol=float_atol,
                    forbid_zeros=forbid_zeros,
                    int8_distinct=distinct,
                )
            except AssertionError as err:
                if fail_fast:
                    raise
                result.passed = False
                # np.testing.assert_allclose messages BEGIN with a newline — take
                # the first NON-empty line so failures are never blank
                message = next((line for line in str(err).splitlines() if line.strip()), str(err))
                result.failures.append(message)
                continue
            for name, diff in worst.items():
                if diff > result.worst_abs_diff.get(name, 0.0):
                    result.worst_abs_diff[name] = diff
    result.int8_distinct = {name: sorted(values) for name, values in distinct.items()}
    return result
