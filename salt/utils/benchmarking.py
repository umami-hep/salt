"""Benchmarking utilities for Pytorch models."""

from collections.abc import Callable

import torch
from torch import Tensor, dtype
from torch.utils import benchmark


def time_forward(
    fn: Callable,
    *args,
    repeats: int = 10,
    block_time: float = 0.0,
    desc: str = "",
    verbose: bool = True,
    amp: bool = False,
    amp_dtype: dtype = torch.float16,
    **kwargs,
) -> tuple:
    """Use Pytorch Benchmark on the forward pass of an arbitrary function.

    Parameters
    ----------
    fn : function
        The function to benchmark.
    args : list
        The args to the function.
    repeats : int
        Number of times to repeat the benchmark.
    block_time : float
        Instead of repeats, run the benchmark for a fixed amount of time.
    desc : str
        Description of the benchmark.
    verbose : bool
        Whether to print the benchmark results.
    amp : bool
        Whether to use automatic mixed precision.
    amp_dtype : torch.dtype
        The dtype to use for automatic mixed precision.
    kwargs : dict
        Additional keyword arguments to pass to the function.
    """
    if verbose:
        print(desc, " - Foward pass")

    # Define the automatic mixed precision wrapper
    def fn_with_amp(*args, **kwargs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*args, **kwargs)

    # Create the benchmark timer
    t = benchmark.Timer(
        stmt="fn_with_amp(*args, **kwargs)",
        globals={"fn_with_amp": fn_with_amp, "args": args, "kwargs": kwargs},
        num_threads=torch.get_num_threads(),
    )

    # Run the benchmark
    m = t.blocked_autorange(min_run_time=block_time) if block_time > 0 else t.timeit(repeats)

    if verbose:
        print(m)
    return t, m


def time_backward(
    fn: Callable,
    *args,
    repeats: int = 10,
    block_time: float = 0.0,
    desc: str = "",
    verbose: bool = True,
    amp: bool = False,
    amp_dtype: dtype = torch.float16,
    **kwargs,
) -> tuple:
    """Use Pytorch Benchmark on the backward pass of an arbitrary function.

    Parameters
    ----------
    fn : function
        The function to benchmark.
    args : list
        The args to the function.
    repeats : int
        Number of times to repeat the benchmark.
    block_time : float
        Instead of repeats, run the benchmark for a fixed amount of time.
    desc : str
        Description of the benchmark.
    verbose : bool
        Whether to print the benchmark results.
    amp : bool
        Whether to use automatic mixed precision.
    amp_dtype : torch.dtype
        The dtype to use for automatic mixed precision.
    kwargs : dict
        Additional keyword arguments to pass to the function.
    """
    if verbose:
        print(desc, " - Backward pass")

    # Run in forward to get the output so we can backpropagate
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
        y = fn(*args, **kwargs)
        if type(y) is tuple:
            y = y[0]
        elif type(y) is dict:
            y = next(iter(y.values()))

    # Generate a random gradient
    grad = torch.randn_like(y)

    # Define the backward function
    def bwd(*args, y, grad):
        for x in args:  # Turn off gradients for all args
            if isinstance(x, Tensor):
                x.grad = None
        y.backward(grad, retain_graph=True)

    # Create the benchmark timer
    t = benchmark.Timer(
        stmt="f(*args, y=y, grad=grad)",
        globals={"f": bwd, "args": args, "y": y, "grad": grad},
        num_threads=torch.get_num_threads(),
    )

    # Run the benchmark
    m = t.blocked_autorange(min_run_time=block_time) if block_time > 0 else t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_gpu_memory(
    fn: Callable,
    *args,
    amp: bool = False,
    amp_dtype: dtype = torch.float16,
    **kwargs,
) -> tuple:
    """Calculate the maximum GPU memory used by a function.

    Parameters
    ----------
    fn : function
        The function to benchmark.
    args : list
        The args to the function.
    amp : bool
        Whether to use automatic mixed precision.
    amp_dtype : torch.dtype
        The dtype to use for automatic mixed precision.
    kwargs : dict
        Additional keyword arguments to pass to the function.
    """
    # Clear the cache and reset memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Run the function
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    # Calculate the max memory used in GB
    mem = torch.cuda.max_memory_allocated() / ((2**20) * 1000)
    torch.cuda.empty_cache()
    return mem
