"""W5.1 integration gate — H5 auto-collect == the explicit-columns cutover output.

Plan 31 W5.1 gate (a): on the SAME synthetic GN2v2 model + producers + source H5,
the AUTO-COLLECT `H5OutputSink` (omitted ``outputs:``) writes a byte/field-equal
eval H5 to the EXPLICIT-columns form (the reference `gn2v2-dummy-cutover.yaml`
wiring). This proves auto-collect reproduces the reference exactly — same groups,
columns, ORDER, dtypes, and values (COMPLETE column-coverage, the
`test_outputs_h5_parity.py` pattern: ints/bools exact, floats atol 1e-6).

Both runs share one trained checkpoint; the only difference under test is whether
the H5 sink's columns are hand-listed or auto-discovered from the producers.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from lightning import Trainer

from salt.core import SaltModule
from salt.core.data import Features, H5StructuredReader, Labels
from salt.core.data.datamodule import GraphDataModule
from salt.core.graph.spec import Mode
from salt.core.main import main
from salt.core.outputs import ClassProbs, H5OutputSink, OutputColumn, SeqClassProbs
from salt.core.schema import dump_schema, save_schema
from salt.tests._fixtures.gn2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    write_parity_norm_dict,
)
from salt.tests._fixtures.gn2v2_fixture import ORIGIN_CLASSES, build_gn2v2_modules
from salt.utils.inputs import write_dummy_file

RUN_NAME = "GN2v2_dummy"
N_TEST = 200
_FLOAT_TOL = 1e-6
JET_SUFFIXES = ["pb", "pc", "pu"]
ORIGIN_SUFFIXES = [f"p{c}" for c in ORIGIN_CLASSES]


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("auto_h5")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    h5_path = base / "pp_output_test_ttbar.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


@pytest.fixture(scope="module")
def ckpt(data, tmp_path_factory) -> Path:
    fit_dir = tmp_path_factory.mktemp("auto_h5_fit")
    from salt.core.main import CONFIG_DIR

    rc = main([
        "fit", "--config", str(CONFIG_DIR / "gn2v2-dummy.yaml"),
        f"--data.train_file={data['h5']}", f"--data.val_file={data['h5']}",
        f"--data.modules.reader.init_args.schema={data['schema']}",
        f"--model.modules.norm.init_args.norm_dict={data['nd']}",
        "--trainer.accelerator=cpu", "--trainer.logger=false", "--callbacks.progress=null",
        f"--trainer.default_root_dir={fit_dir}", "--trainer.max_epochs=1",
        "--trainer.limit_train_batches=2", "--trainer.limit_val_batches=2",
        "--trainer.num_sanity_val_steps=0",
    ])
    assert rc == 0
    ckpts = sorted(fit_dir.rglob("*.ckpt"))
    assert ckpts
    return ckpts[0]


def _model(data) -> SaltModule:
    modules = build_gn2v2_modules(data["nd"])
    modules["track_vertexing"].expose_modes = Mode.FIT | Mode.VAL
    modules["jet_probs"] = ClassProbs(task="jets_classification", stream="jets")
    modules["track_origin_probs"] = SeqClassProbs(task="track_origin", stream="tracks")
    model = SaltModule(modules, lrs={"initial": 1e-7, "max": 1e-3, "end": 1e-5, "pct_start": 0.01})
    model.name = RUN_NAME
    return model


def _dm(data) -> GraphDataModule:
    return GraphDataModule(
        modules={
            "reader": H5StructuredReader(
                groups={"jets": {"global_object": True}, "tracks": {"global_object": False}},
                schema=str(data["schema"]),
            ),
            "features": Features(
                variables={"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}
            ),
            "labels": Labels(dtype_policy="int64-for-int"),
        },
        test_file=str(data["h5"]), batch_size=100, num_workers=0, num_test=N_TEST,
        pin_memory=False, persistent_workers=False,
    )


def _run(data, ckpt, sink, tmp_path_factory, tag) -> Path:
    out = tmp_path_factory.mktemp(tag) / f"{tag}.h5"
    sink.output = str(out)
    trainer = Trainer(
        accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
        enable_model_summary=False, callbacks=[sink],
    )
    trainer.test(_model(data), datamodule=_dm(data), ckpt_path=str(ckpt))
    assert out.exists()
    return out


@pytest.fixture(scope="module")
def explicit_h5(data, ckpt, tmp_path_factory) -> Path:
    sink = H5OutputSink(
        outputs=[
            OutputColumn(key="outputs.jets.jets_classification", suffixes=JET_SUFFIXES),
            OutputColumn(key="outputs.tracks.track_origin", suffixes=ORIGIN_SUFFIXES),
        ],
        copy_inputs={"jets": [], "tracks": []},
        write_pad_mask=["tracks"],
        output="unused",
    )
    return _run(data, ckpt, sink, tmp_path_factory, "explicit")


@pytest.fixture(scope="module")
def auto_h5(data, ckpt, tmp_path_factory) -> Path:
    sink = H5OutputSink(  # AUTO-COLLECT: no `outputs:`
        copy_inputs={"jets": [], "tracks": []},
        write_pad_mask=["tracks"],
        output="unused",
    )
    return _run(data, ckpt, sink, tmp_path_factory, "auto")


def test_auto_collect_h5_equals_explicit(explicit_h5, auto_h5):
    """Auto-collect H5 == explicit-columns H5: same groups, columns, order, dtypes, values."""
    diffs: list[str] = []
    with h5py.File(explicit_h5) as a, h5py.File(auto_h5) as b:
        assert set(a.keys()) == set(b.keys()), f"groups differ: {set(a)} vs {set(b)}"
        for group in a:
            want, got = a[group][:], b[group][:]
            if list(want.dtype.names) != list(got.dtype.names):
                diffs.append(
                    f"{group}: column set/order mismatch\n  explicit: {list(want.dtype.names)}"
                    f"\n  auto    : {list(got.dtype.names)}"
                )
                continue
            for col in want.dtype.names:
                w, g = want[col], got[col]
                if w.dtype != g.dtype:
                    diffs.append(f"{group}.{col}: dtype {w.dtype} != {g.dtype}")
                elif np.issubdtype(w.dtype, np.floating):
                    if not np.allclose(w, g, rtol=0.0, atol=_FLOAT_TOL, equal_nan=True):
                        diffs.append(f"{group}.{col}: float values differ > {_FLOAT_TOL}")
                elif not np.array_equal(w, g):
                    diffs.append(f"{group}.{col}: int/bool values differ")
    assert not diffs, "AUTO==EXPLICIT H5 PARITY FAILED:\n" + "\n".join(diffs)
