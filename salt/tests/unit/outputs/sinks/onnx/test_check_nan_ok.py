"""`nan_ok` threading through `OutputField` -> `OnnxExportLeaf` ->
`OnnxExportSink.nan_ok_outputs()` -> `OnnxAdapter` -> `check_onnx`/`compare_once`
(plan 08 Q7 (b)): NaN is a declared semantic for some outputs (MaskFormer
leading-object / lead-vertex scalars for a jet with no qualifying object), and
those are compared ``equal_nan`` instead of tripping the NaN canary.
"""

from __future__ import annotations

import types
from collections.abc import Mapping

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from salt.graph.spec import Mode
from salt.outputs import OnnxExportSink, OutputField
from salt.outputs.sinks.onnx import check as check_module
from salt.outputs.sinks.onnx.check import check_onnx, compare_once
from salt.outputs.sinks.onnx.config import ExportInput

pytestmark = pytest.mark.cpu_always

_JETS_FIELDS = ("pt", "eta")
_TRACKS_FIELDS = ("pt", "eta", "phi")


class _StubAdapter(nn.Module):
    """A stub exposing exactly what `compare_once`/`check_onnx` read from an
    `OnnxAdapter`: no bundle/executor machinery, just the naming + forward
    surface. `_positional` carries one global port (``inputs.jets``) and one
    sequence port (``inputs.tracks``), mirroring a real adapter's ctor, so
    `_draw_inputs`'s per-length sweep exercises the real code path; every
    test output here is a GLOBAL scalar, so `forward` ignores the drawn
    tensors' values (and the sequence length) and returns fixed outputs.
    """

    def __init__(
        self,
        output_names: list[str],
        output_dtypes: list[str],
        nan_ok_outputs: frozenset[str],
        torch_outputs: tuple[Tensor, ...],
    ) -> None:
        super().__init__()
        self._positional = [
            ExportInput(port="inputs.jets", name="jets_features"),
            ExportInput(port="inputs.tracks", name="tracks_features", sequence=True),
        ]
        self.input_names = [str(entry.name) for entry in self._positional]
        self.output_names = output_names
        self.output_dtypes = output_dtypes
        self.nan_ok_outputs = nan_ok_outputs
        self._torch_outputs = torch_outputs

    def _field_list(self, port: str) -> tuple[str, ...]:
        return _TRACKS_FIELDS if port == "inputs.tracks" else _JETS_FIELDS

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        """Ignores `inputs` — every test output here is a global scalar."""
        del inputs
        return self._torch_outputs


class _StubOutput:
    """Mirrors an onnxruntime `NodeArg` — just the `.name` the checker reads."""

    def __init__(self, name: str) -> None:
        self.name = name


class _StubSession:
    """Mirrors `onnxruntime.InferenceSession`'s two read call sites."""

    def __init__(self, output_names: list[str], outputs: list[np.ndarray]) -> None:
        self._output_names = output_names
        self._outputs = outputs

    def get_outputs(self) -> list[_StubOutput]:
        return [_StubOutput(name) for name in self._output_names]

    def run(self, _output_names: None, _feeds: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        del _output_names, _feeds
        return self._outputs


def _gen() -> torch.Generator:
    return torch.Generator().manual_seed(0)


# -- (a)/(b)/(c)/(d): compare_once, one case at a time -----------------------


def test_nan_ok_output_nan_in_both_torch_and_ort_passes_and_worst_is_zero():
    """A `nan_ok` output NaN in BOTH torch and ORT is not an error — it's the
    declared semantic — and its worst-diff entry excludes the NaN.
    """
    adapter = _StubAdapter(
        output_names=["X_a", "X_b"],
        output_dtypes=["float32", "float32"],
        nan_ok_outputs=frozenset({"X_a"}),
        torch_outputs=(
            torch.tensor([[float("nan")]], dtype=torch.float32),
            torch.tensor([[2.5]], dtype=torch.float32),
        ),
    )
    session = _StubSession(
        ["X_a", "X_b"],
        [
            np.array([[np.nan]], dtype=np.float32),
            np.array([[2.5]], dtype=np.float32),
        ],
    )
    worst = compare_once(adapter, session, {"tracks": 3}, _gen(), nan_ok=frozenset({"X_a"}))
    assert worst["X_a"] == 0.0
    assert worst["X_b"] == 0.0


def test_same_nan_not_declared_nan_ok_fails_the_canary():
    """The identical NaN, without `nan_ok`, still trips the NaN assert."""
    adapter = _StubAdapter(
        output_names=["X_a"],
        output_dtypes=["float32"],
        nan_ok_outputs=frozenset(),
        torch_outputs=(torch.tensor([[float("nan")]], dtype=torch.float32),),
    )
    session = _StubSession(["X_a"], [np.array([[np.nan]], dtype=np.float32)])
    with pytest.raises(AssertionError, match="NaN in torch output"):
        compare_once(adapter, session, {"tracks": 3}, _gen(), nan_ok=frozenset())


def test_nan_ok_output_nan_in_torch_but_finite_in_ort_is_a_mismatch():
    """`nan_ok` skips the NaN canary, not the actual torch-vs-ORT comparison —
    a NaN/finite divergence is still a real mismatch.
    """
    adapter = _StubAdapter(
        output_names=["X_a"],
        output_dtypes=["float32"],
        nan_ok_outputs=frozenset({"X_a"}),
        torch_outputs=(torch.tensor([[float("nan")]], dtype=torch.float32),),
    )
    session = _StubSession(["X_a"], [np.array([[3.0]], dtype=np.float32)])
    with pytest.raises(AssertionError, match="mismatch"):
        compare_once(adapter, session, {"tracks": 3}, _gen(), nan_ok=frozenset({"X_a"}))


def test_nan_ok_output_finite_exact_zero_still_trips_forbid_zeros():
    """`nan_ok` only exempts NaN — a finite exact zero is still the dead-output
    canary and still fails.
    """
    adapter = _StubAdapter(
        output_names=["X_a"],
        output_dtypes=["float32"],
        nan_ok_outputs=frozenset({"X_a"}),
        torch_outputs=(torch.tensor([[0.0]], dtype=torch.float32),),
    )
    session = _StubSession(["X_a"], [np.array([[0.0]], dtype=np.float32)])
    with pytest.raises(AssertionError, match="exact zero"):
        compare_once(adapter, session, {"tracks": 3}, _gen(), nan_ok=frozenset({"X_a"}))


# -- (e): check_onnx end-to-end, adapter's own nan_ok_outputs threaded -------


def _monkeypatch_session(monkeypatch: pytest.MonkeyPatch, session: _StubSession) -> None:
    monkeypatch.setattr(check_module, "make_session", lambda _path: session)


def test_check_onnx_passes_with_the_nan_ok_stub(monkeypatch: pytest.MonkeyPatch):
    adapter = _StubAdapter(
        output_names=["X_a"],
        output_dtypes=["float32"],
        nan_ok_outputs=frozenset({"X_a"}),
        torch_outputs=(torch.tensor([[float("nan")]], dtype=torch.float32),),
    )
    session = _StubSession(["X_a"], [np.array([[np.nan]], dtype=np.float32)])
    _monkeypatch_session(monkeypatch, session)
    result = check_onnx(
        adapter,
        "unused.onnx",
        lengths_grid=[{"tracks": 0}, {"tracks": 3}],
        trials=1,
    )
    assert result.passed is True


def test_check_onnx_fails_with_the_strict_stub(monkeypatch: pytest.MonkeyPatch):
    """The identical NaN, on an adapter that declares NO `nan_ok` outputs, fails."""
    adapter = _StubAdapter(
        output_names=["X_a"],
        output_dtypes=["float32"],
        nan_ok_outputs=frozenset(),
        torch_outputs=(torch.tensor([[float("nan")]], dtype=torch.float32),),
    )
    session = _StubSession(["X_a"], [np.array([[np.nan]], dtype=np.float32)])
    _monkeypatch_session(monkeypatch, session)
    result = check_onnx(
        adapter,
        "unused.onnx",
        lengths_grid=[{"tracks": 0}, {"tracks": 3}],
        trials=1,
    )
    assert result.passed is False


# -- (f): manifest threading — OutputField.nan_ok -> OnnxExportSink.nan_ok_outputs() ---


def test_sink_threads_nan_ok_from_the_bound_producer_manifest():
    """Mirrors `test_sink_consumes.py`'s `_StubProducer` binding pattern: a
    producer's `manifest_fields(mode)` OutputField.nan_ok flows through to
    `OnnxExportSink.nan_ok_outputs()`, one flag per Athena suffix.
    """
    producer = types.SimpleNamespace(
        manifest_fields=lambda mode: [
            ("outputs.jets.a", OutputField(h5_name=None, onnx_name="a", nan_ok=True)),
            ("outputs.jets.b", OutputField(h5_name=None, onnx_name="b")),
        ]
        if mode & Mode.ONNX
        else [],
    )
    sink = OnnxExportSink(model_name="X")
    sink.bind_model_modules({"p": producer})
    assert sink.nan_ok_outputs() == {"X_a"}
    assert sink.output_names() == ["X_a", "X_b"]
