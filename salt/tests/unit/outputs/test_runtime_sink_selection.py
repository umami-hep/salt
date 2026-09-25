"""ORDER-INDEPENDENT TEST sink selection + persistence discrimination."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from salt.graph.spec import Mode
from salt.main import (
    _has_callback_persistence_sink,
    _is_persistence_sink,
)
from salt.model.modules.losses import LossSum
from salt.outputs import (
    H5OutputSink,
    OnnxExportSink,
)
from salt.model.saltmodule import SaltModule
from salt.model.sink_prep import select_test_sink
from salt.outputs.sinks.sink import is_test_persistence_sink
from salt.tests._fixtures.gn2v2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    write_parity_norm_dict,
)
from salt.tests._fixtures.gn2v2_fixture import build_gn2v2_modules

_ONNX_CLASS_PATH = "salt.outputs.OnnxExportSink"
_H5_CLASS_PATH = "salt.outputs.H5OutputSink"

_LRS = {"initial": 1e-3, "max": 5e-3, "end": 1e-4, "pct_start": 0.1}


def _h5_sink() -> H5OutputSink:
    """The H5 persistence sink (`is_test_sink()` is resolution-free — always True)."""
    return H5OutputSink()


def _onnx_sink() -> OnnxExportSink:
    """An ONNX-only sink: ``declare_io(Mode.TEST)`` is empty -> NOT a test persistence sink."""
    return OnnxExportSink(model_name="GN2v2")


# the is_test_sink() discriminator (the clean, declare_io-derived signal)


def test_h5_sink_is_a_test_sink_onnx_sink_is_not():
    """`H5OutputSink.is_test_sink()` is True; `OnnxExportSink.is_test_sink()` is False."""
    assert _h5_sink().is_test_sink() is True
    assert _onnx_sink().is_test_sink() is False


def test_is_test_persistence_sink_helper_defaults_true_for_duck_typed():
    """`is_test_persistence_sink` treats a sink WITHOUT ``is_test_sink`` as a test sink."""
    assert is_test_persistence_sink(_h5_sink()) is True
    assert is_test_persistence_sink(_onnx_sink()) is False
    # a duck-typed object with writer_demand but no is_test_sink -> treated as a sink
    duck = SimpleNamespace(writer_demand=lambda *_a, **_k: {})
    assert is_test_persistence_sink(duck) is True


# select_test_sink — ORDER-INDEPENDENT


@pytest.mark.parametrize("onnx_first", [True, False], ids=["onnx_first", "h5_first"])
def test_attached_writer_picks_h5_sink_regardless_of_order(onnx_first):
    """`select_test_sink` selects the H5 sink for TEST in BOTH callback orders."""
    h5, onnx = _h5_sink(), _onnx_sink()
    callbacks = [onnx, h5] if onnx_first else [h5, onnx]
    assert select_test_sink(callbacks) is h5


def test_attached_writer_none_when_only_onnx_sink_present():
    """An ONNX-only sink alone is NOT selected by `select_test_sink` (no TEST persistence)."""
    assert select_test_sink([_onnx_sink()]) is None


# the real runtime compile_mode(Mode.TEST) path — no dead-preds crash, both orders


@pytest.fixture(scope="module")
def cutover_data(tmp_path_factory):
    """A dummy H5 + norm dict + schema for a real ``model.setup('test')`` (CPU)."""
    from salt.data import (  # noqa: PLC0415
        Features,
        SaltDataModule,
        H5StructuredReader,
        Labels,
    )
    from salt.schema import dump_schema, save_schema  # noqa: PLC0415
    from salt.testing.inputs import write_dummy_file  # noqa: PLC0415

    base = tmp_path_factory.mktemp("runtime_sink_b2")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    h5_path = base / "pp_output_test.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return SimpleNamespace(
        Features=Features,
        SaltDataModule=SaltDataModule,
        H5StructuredReader=H5StructuredReader,
        Labels=Labels,
        nd=nd_path,
        h5=h5_path,
        schema=schema_path,
    )


def _cutover_model(nd_path: Path) -> SaltModule:
    """A GN2v2 SaltModule with a section RunTaskOutput over the jet/track heads
    (the eval-H5 producers).
    """
    # PLC2701: _parse_expose is private but the cleanest way to set expose_modes
    from salt.model.modules.tasks import _parse_expose  # noqa: PLC0415, PLC2701 - test-local
    from salt.outputs.run_task_output import RunTaskOutput  # noqa: PLC0415 - test-local

    modules = build_gn2v2_modules(nd_path)
    # opt track_vertexing OUT of TEST eval (its VertexIndex is a deferred reduce
    # with no producer here — a dead pred otherwise), mirroring the cutover config.
    modules["track_vertexing"].expose_modes = _parse_expose(["fit", "val"], "VertexingTaskModule")
    modules["loss"] = LossSum()
    return SaltModule(
        modules,
        lrs=_LRS,
        outputs={"run_tasks": RunTaskOutput(tasks=["jets_classification", "track_origin"])},
    )


@pytest.mark.parametrize("onnx_first", [True, False], ids=["onnx_first", "h5_first"])
def test_setup_test_does_not_crash_regardless_of_order(cutover_data, onnx_first):
    """``model.setup('test')`` (the real ``compile_mode(Mode.TEST)``) succeeds in BOTH orders."""
    d = cutover_data
    dm = d.SaltDataModule(
        {
            "reader": d.H5StructuredReader(groups={"jets": {}, "tracks": {}}, schema=d.schema),
            "features": d.Features(
                variables={"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}
            ),
            "labels": d.Labels(),
        },
        batch_size=100,
        num_workers=0,
        train_file=d.h5,
        val_file=d.h5,
        test_file=d.h5,
    )
    h5, onnx = _h5_sink(), _onnx_sink()
    callbacks = [onnx, h5] if onnx_first else [h5, onnx]
    model = _cutover_model(d.nd)
    # attach a trainer carrying the callbacks (selection source) + the real
    # the trainer supplies the callbacks (selection source) and the real datamodule
    # (boundary source) that ``setup('test')`` reads; the callbacks must be in place
    # BEFORE sink_demand — it routes through the same select_test_sink selection.
    model._trainer = SimpleNamespace(callbacks=callbacks, datamodule=dm)  # noqa: SLF001
    dm.set_sinks(model.sink_demand())  # the model boundary demand (TEST sinks)
    dm.setup("test")  # build the test dataset (the boundary source compile_mode reads)

    # the load-bearing call: order-dependent selection would raise ConfigError
    # (dead preds) when onnx_first; it must compile the TEST plan with the H5
    # sink folded in BOTH orders.
    model.setup("test")

    test_plan = model.plans[Mode.TEST]
    assert "h5_output" in test_plan.module_names  # H5 sink folded as a node
    assert "onnx_export" not in test_plan.module_names  # ONNX-only sink pruned in TEST
    assert "run_tasks" in test_plan.module_names


# main._is_persistence_sink / _has_callback_persistence_sink (writer-less check)


def test_is_persistence_sink_false_for_onnx_only_sink():
    """`_is_persistence_sink(OnnxExportSink)` is False (no TEST persistence)."""
    assert _is_persistence_sink(_ONNX_CLASS_PATH) is False
    assert _is_persistence_sink(_H5_CLASS_PATH) is True


@pytest.mark.parametrize("onnx_first", [True, False], ids=["onnx_first", "h5_first"])
def test_has_callback_persistence_sink_ignores_onnx_only(onnx_first):
    """`_has_callback_persistence_sink` is True iff a REAL test sink is present, any order."""
    onnx_entry = {"class_path": _ONNX_CLASS_PATH}
    h5_entry = {"class_path": _H5_CLASS_PATH}
    both = (
        {"onnx_export": onnx_entry, "h5_output": h5_entry}
        if onnx_first
        else {"h5_output": h5_entry, "onnx_export": onnx_entry}
    )
    assert _has_callback_persistence_sink(both) is True
    assert _has_callback_persistence_sink({"onnx_export": onnx_entry}) is False
