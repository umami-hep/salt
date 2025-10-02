"""Benchmarking utilities for PyTorch models."""

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, dtype
from torch.utils import benchmark


def time_forward(
    fn: Callable[..., Any],
    *args: Any,
    repeats: int = 10,
    block_time: float = 0.0,
    desc: str = "",
    verbose: bool = True,
    amp: bool = False,
    amp_dtype: dtype = torch.float16,
    **kwargs: Any,
) -> tuple[benchmark.Timer, benchmark.Measurement]:
    """Benchmark the forward pass of an arbitrary function using ``torch.utils.benchmark``.

    Parameters
    ----------
    fn : Callable[..., Any]
        The function to benchmark (typically the model's forward or a wrapper).
    *args : Any
        Positional arguments passed to ``fn``.
    repeats : int, optional
        Number of timed runs when ``block_time == 0``. Default is ``10``.
    block_time : float, optional
        If ``> 0``, run for (approximately) this many seconds using
        :meth:`benchmark.Timer.blocked_autorange`; otherwise use
        :meth:`benchmark.Timer.timeit` with ``repeats``. Default is ``0.0``.
    desc : str, optional
        Description printed before results when ``verbose`` is ``True``. Default is ``""``.
    verbose : bool, optional
        If ``True``, print the measured results. Default is ``True``.
    amp : bool, optional
        If ``True``, enable automatic mixed precision (CUDA autocast). Default is ``False``.
    amp_dtype : dtype, optional
        Data type used for autocast when ``amp`` is enabled. Default is ``torch.float16``.
    **kwargs : Any
        Keyword arguments passed to ``fn``.

    Returns
    -------
    tuple[benchmark.Timer, benchmark.Measurement]
        The constructed :class:`benchmark.Timer` and the resulting
        :class:`benchmark.Measurement` from the run.
    """
    if verbose:
        print(desc, " - Forward pass")

    def fn_with_amp(*_args: Any, **_kwargs: Any) -> None:
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*_args, **_kwargs)

    t = benchmark.Timer(
        stmt="fn_with_amp(*args, **kwargs)",
        globals={"fn_with_amp": fn_with_amp, "args": args, "kwargs": kwargs},
        num_threads=torch.get_num_threads(),
    )

    m = t.blocked_autorange(min_run_time=block_time) if block_time > 0 else t.timeit(repeats)

    if verbose:
        print(m)
    return t, m


def time_backward(
    fn: Callable[..., Any],
    *args: Any,
    repeats: int = 10,
    block_time: float = 0.0,
    desc: str = "",
    verbose: bool = True,
    amp: bool = False,
    amp_dtype: dtype = torch.float16,
    **kwargs: Any,
) -> tuple[benchmark.Timer, benchmark.Measurement]:
    """Benchmark the backward pass (autograd) using ``torch.utils.benchmark``.

    The function ``fn`` is first run once (under optional autocast) to produce an output
    tensor ``y``. A random gradient with the same shape as ``y`` is then used for the
    backward pass timing.

    Parameters
    ----------
    fn : Callable[..., Any]
        The function to benchmark. It must return either:
        - a tensor,
        - a tuple whose first element is a tensor, or
        - a dict whose first value is a tensor.
    *args : Any
        Positional arguments passed to ``fn``.
    repeats : int, optional
        Number of timed runs when ``block_time == 0``. Default is ``10``.
    block_time : float, optional
        If ``> 0``, run for (approximately) this many seconds using
        :meth:`benchmark.Timer.blocked_autorange`; otherwise use
        :meth:`benchmark.Timer.timeit` with ``repeats``. Default is ``0.0``.
    desc : str, optional
        Description printed before results when ``verbose`` is ``True``. Default is ``""``.
    verbose : bool, optional
        If ``True``, print the measured results. Default is ``True``.
    amp : bool, optional
        If ``True``, enable automatic mixed precision (CUDA autocast). Default is ``False``.
    amp_dtype : dtype, optional
        Data type used for autocast when ``amp`` is enabled. Default is ``torch.float16``.
    **kwargs : Any
        Keyword arguments passed to ``fn``.

    Returns
    -------
    tuple[benchmark.Timer, benchmark.Measurement]
        The constructed :class:`benchmark.Timer` and the resulting
        :class:`benchmark.Measurement` from the run.

    Raises
    ------
    TypeError
        If ``fn`` does not produce a tensor, a tuple with a tensor as the first item,
        or a dict with a tensor as the first value.
    """
    if verbose:
        print(desc, " - Backward pass")

    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
        y = fn(*args, **kwargs)
        if isinstance(y, tuple):
            y = y[0]
        elif isinstance(y, dict):
            y = next(iter(y.values()))

    if not isinstance(y, Tensor):
        raise TypeError(
            "time_backward expected `fn` to return a Tensor, a tuple(Tensor, ...), "
            "or a dict[str, Tensor]."
        )

    grad = torch.randn_like(y)

    def bwd(*b_args: Any, y: Tensor, grad: Tensor) -> None:
        # Clear gradients on any tensor positional arguments
        for x in b_args:
            if isinstance(x, Tensor):
                x.grad = None
        # Backpropagate a synthetic gradient to measure backward time
        y.backward(grad, retain_graph=True)

    t = benchmark.Timer(
        stmt="f(*args, y=y, grad=grad)",
        globals={"f": bwd, "args": args, "y": y, "grad": grad},
        num_threads=torch.get_num_threads(),
    )

    m = t.blocked_autorange(min_run_time=block_time) if block_time > 0 else t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_gpu_memory(
    fn: Callable[..., Any],
    *args: Any,
    amp: bool = False,
    amp_dtype: dtype = torch.float16,
    **kwargs: Any,
) -> float:
    """Measure the peak GPU memory (in GiB) used while running a function.

    Parameters
    ----------
    fn : Callable[..., Any]
        The function to execute for memory measurement.
    *args : Any
        Positional arguments passed to ``fn``.
    amp : bool, optional
        If ``True``, enable automatic mixed precision (CUDA autocast). Default is ``False``.
    amp_dtype : dtype, optional
        Data type used for autocast when ``amp`` is enabled. Default is ``torch.float16``.
    **kwargs : Any
        Keyword arguments passed to ``fn``.

    Returns
    -------
    float
        Peak allocated GPU memory in gibibytes (GiB).
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    mem_gib = torch.cuda.max_memory_allocated() / (2**30)  # bytes -> GiB
    torch.cuda.empty_cache()
    return mem_gib
