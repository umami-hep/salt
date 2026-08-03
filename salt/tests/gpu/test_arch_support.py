"""Does the installed torch actually serve the GPU it is running on?

`torch.cuda.is_available()` is NOT the question. A torch built for an older CUDA
toolkit reports a newer card as available and then fails every kernel launch with
``no kernel image is available for execution``. Measured on an RTX 5060 Ti
(sm_120) inside salt's cu126 container:

    torch.cuda.is_available()  -> True
    torch.cuda.get_arch_list() -> sm_50 … sm_90     (no sm_120, no PTX)
    a 512x512 matmul           -> CUDA error: no kernel image is available

That matters beyond one dev box: `salt/tests/conftest.py` gates the integration
suite on `is_available()`, so such a machine un-skips all 207 integration tests
and then fails every one of them.

No single machine can cover every generation, so this module does two things:
tests that interrogate WHATEVER GPU is present and assert the image serves it,
and `requires_sm`-gated tests that skip elsewhere. Coverage accumulates across
CI (A100), lxplus (V100/T4) and dev boxes (Blackwell).
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu_arch


# --- helpers ---------------------------------------------------------------


def gpu_capability() -> tuple[int, int] | None:
    """The present device's compute capability, or None when there is no GPU."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(0)


def _sm_tag(cap: tuple[int, int]) -> str:
    return f"sm_{cap[0]}{cap[1]}"


def requires_gpu(func):
    """Skip with a stated reason when no CUDA device is present."""
    return pytest.mark.skipif(gpu_capability() is None, reason="no CUDA device")(func)


def requires_sm(major: int, minor: int, who: str):
    """Skip unless the present device is exactly this generation.

    A skipped arch test must say WHICH arch it wanted and what it found, so a
    machine that silently tests nothing is visible in the report.
    """
    cap = gpu_capability()
    if cap is None:
        reason = f"no CUDA device (test targets {who}, sm_{major}{minor})"
    else:
        reason = f"device is {_sm_tag(cap)}, test targets {who} (sm_{major}{minor})"
    return pytest.mark.skipif(cap != (major, minor), reason=reason)


def _arch_list() -> list[str]:
    """Compiled arch flags, preferring the driverless-safe private call.

    `torch.cuda.get_arch_list()` short-circuits to [] when `is_available()` is
    False, which makes it useless in a driverless build sandbox. The private
    `_cuda_getArchFlags` reads the compiled-in flags and needs no device.
    """
    try:
        flags = torch._C._cuda_getArchFlags()  # noqa: SLF001 - no public equivalent
    except Exception:  # noqa: BLE001 - fall back to the public API
        flags = None
    if flags:
        return flags.split()
    return list(torch.cuda.get_arch_list())


# --- the assertions that would have caught the cu126-on-Blackwell failure ---


@requires_gpu
def test_torch_serves_this_device_architecture():
    """The device's capability is covered by compiled SASS or a PTX fallback."""
    cap = gpu_capability()
    assert cap is not None
    tag, archs = _sm_tag(cap), _arch_list()

    has_sass = any(a.startswith(tag) for a in archs)
    # PTX is forward-compatible: compute_XX can JIT onto a newer device.
    has_ptx = any(a.startswith("compute_") for a in archs)

    assert has_sass or has_ptx, (
        f"torch {torch.__version__} (CUDA {torch.version.cuda}) has no kernels for "
        f"{torch.cuda.get_device_name(0)} ({tag}). Compiled for: {archs}. "
        "Every kernel launch on this device will fail with 'no kernel image is "
        "available for execution' even though torch.cuda.is_available() is True. "
        "Fix: install a torch built against a CUDA toolkit that targets this arch."
    )


@requires_gpu
def test_a_real_kernel_launches_and_computes_correctly():
    """A capability query is not proof — launch a kernel and check the numbers."""
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    got = (a @ b).cpu()
    want = a.cpu() @ b.cpu()
    # Loose: TF32 matmul is on by default on Ampere+, so this is not fp32-exact.
    torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-2)


@requires_gpu
def test_flash_attn_is_either_working_or_cleanly_absent():
    """flash-attn must not be importable-but-broken — that degrades silently.

    Salt falls back to torch-math when the import fails, which is fine. What is
    not fine is a flash-attn compiled for a different arch: it imports cleanly
    and dies at launch, so a run looks healthy while attention is broken.
    """
    try:
        import flash_attn  # noqa: F401
        from flash_attn import flash_attn_func
    except Exception as exc:  # noqa: BLE001 - clean absence is an accepted state
        pytest.skip(f"flash-attn not installed — salt uses torch-math attention ({exc})")

    cap = gpu_capability()
    assert cap is not None
    if cap[0] < 8:
        pytest.fail(
            f"flash-attn imported on {_sm_tag(cap)}, but flash-attn 2.x requires "
            "Ampere (sm_80) or newer. It cannot work here and will fail at launch."
        )

    q = torch.randn(1, 8, 4, 32, device="cuda", dtype=torch.float16)
    try:
        out = flash_attn_func(q, q, q)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"flash-attn is installed but its kernels do not run on {_sm_tag(cap)}: "
            f"{type(exc).__name__}: {exc}. It was built for a different arch — salt "
            "will not fall back, it will crash mid-training."
        )
    assert out.shape == q.shape


@requires_gpu
def test_arch_coverage_is_reported():
    """Always-passing inventory line, so every run records what it proved."""
    cap = gpu_capability()
    assert cap is not None
    print(
        f"\ndevice={torch.cuda.get_device_name(0)} {_sm_tag(cap)} | "
        f"torch={torch.__version__} cuda={torch.version.cuda} | "
        f"compiled_archs={_arch_list()}"
    )


# --- per-generation gates -------------------------------------------------
#
# These skip everywhere except their own arch. None of them can run on a single
# machine; the point is that whichever machine DOES have the card exercises its
# path, and the skip reason names what was missing everywhere else.


@requires_sm(7, 0, "V100")
def test_volta_has_no_flash_attn_and_says_so():
    """sm_70: flash-attn 2.x is Ampere+, so torch-math is the only correct path."""
    try:
        from flash_attn import flash_attn_func  # noqa: F401
    except Exception:  # noqa: BLE001 - the expected, correct state on Volta
        return
    pytest.fail("flash-attn is importable on sm_70; flash-attn 2.x cannot support Volta")


@requires_sm(7, 5, "T4")
def test_turing_has_no_flash_attn_and_says_so():
    """sm_75: same as Volta — Turing predates flash-attn 2.x's Ampere floor."""
    try:
        from flash_attn import flash_attn_func  # noqa: F401
    except Exception:  # noqa: BLE001 - the expected, correct state on Turing
        return
    pytest.fail("flash-attn is importable on sm_75; flash-attn 2.x cannot support Turing")


@requires_sm(8, 0, "A100")
def test_ampere_runs_flash_attn():
    """sm_80: the CI runner's arch and salt's production training target."""
    from flash_attn import flash_attn_func

    q = torch.randn(1, 8, 4, 32, device="cuda", dtype=torch.float16)
    assert flash_attn_func(q, q, q).shape == q.shape


@requires_sm(9, 0, "H100/GH200")
def test_hopper_runs_flash_attn():
    """sm_90."""
    from flash_attn import flash_attn_func

    q = torch.randn(1, 8, 4, 32, device="cuda", dtype=torch.float16)
    assert flash_attn_func(q, q, q).shape == q.shape


@requires_sm(12, 0, "Blackwell")
def test_blackwell_runs_flash_attn():
    """sm_120 — the arch that exposed all of this, and which CI cannot reach."""
    from flash_attn import flash_attn_func

    q = torch.randn(1, 8, 4, 32, device="cuda", dtype=torch.float16)
    assert flash_attn_func(q, q, q).shape == q.shape
