"""P1 SEMANTIC H5 PARITY GATE for the producers -> ``H5OutputWriter`` chain (plan 01, P1).

The MVP gate (design §4b, §4a SEMANTIC H5 parity): on the SAME synthetic GN2-like
model + the SAME source H5, run BOTH eval sinks and assert per-column array
equality between their eval H5 files —

- **(a) the M4.5 ``WriterCallback``** (the P1 oracle, ``salt/core/writers``) via
  the proven ``salt2 test`` CLI surface (`run_test_cli`, the M3 end-to-end
  harness), and
- **(b) the new producers -> ``H5OutputWriter``** via a programmatic
  ``Trainer.test`` (the CLI builds only the legacy ``WriterCallback`` from
  ``writers:``, so the new sink is driven through the public Lightning API: the
  classification producers are added to the model module dict — demand-pruned
  from FIT/VAL so the trained checkpoint loads unchanged, design §4 risk 4 — and
  ``H5OutputWriter`` is attached as the callback).

Parity is SEMANTIC (design §4a), NOT raw byte-compare: same groups, columns,
dtypes, shapes; values EXACT for int/bool columns, ``<=1e-6`` for floats. The
comparison is SCOPED to the families P1 ships — classification (``jets`` 3-class
probs + ``tracks`` 8-class origin probs) + the input copies + the pad mask. The
GN2v2 fixture also carries a ``track_vertexing`` head whose eval column
(``VertexIndex``) is a DEFERRED family (P2, vertexing); those columns are
EXCLUDED from the comparison and recorded in ``DEFERRED_COLUMNS`` (logged, not
silently dropped — design §4b).

TRANSITIONAL-STATE NOTE (the ``preds.*`` already-converted truth — verified in
source, ``salt/core/nn/tasks.py:1228,1632``): the FULL redesign (design §2,
"removes the TEST/ONNX mode-branches inside ``task.forward``") moves the eval
conversion (softmax / de-scale / union-find) OUT of the task and INTO the
producer, so the task would publish RAW logits and the producer would convert
once. That task change is a LATER step of the redesign and is **not yet in the
tree**: the M4.5 ``ClassificationTaskModule.forward`` still runs
``run_inference`` in TEST, so ``preds.<stream>.<task>`` in the executed TEST
bundle is ALREADY the eval-ready (softmaxed) value. A conversion producer
(`ClassProbs` / `SeqClassProbs`) reading that leaf would DOUBLE-convert. So this
end-to-end gate wires the IDENTITY ``TaskOutput`` producer (the P0 passthrough
clone) onto each task's published ``preds.*`` — the producer -> ``outputs.*`` ->
``H5OutputWriter`` SINK chain serialises EXACTLY the values the M4.5 ``get_h5``
serialises, so byte/array parity is the honest contract here. The conversion
ops' MATH (softmax / masked-softmax / de-scale) is the SEPARATE concern proven
bitwise against the task ``run_inference`` oracle in
``test_outputs_producers.py`` (recon PR2). Once the task-forward conversion is
removed (a later redesign step) this gate swaps the identity producer for the
conversion producer with no change to the SINK or the parity assertion.

If parity cannot be reached the assertion reports the exact column with expected
vs got — never weaken the tolerance to pass.
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
from salt.core.main import CONFIG_DIR, main
from salt.core.outputs import H5OutputWriter, OutputColumn, TaskOutput
from salt.core.schema import dump_schema, save_schema
from salt.tests.core.gn2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    write_parity_norm_dict,
)
from salt.tests.core.gn2v2_fixture import ORIGIN_CLASSES, build_gn2v2_modules
from salt.utils.inputs import write_dummy_file

DUMMY_CFG = CONFIG_DIR / "gn2v2-dummy.yaml"
RUN_NAME = "GN2v2_dummy"  # the dummy config's `name:`
N_TEST = 300  # data.num_test for both sinks
L_FILE = 40  # write_dummy_file sequence length
_FLOAT_TOL = 1e-6

# The P1 classification families compared for parity. Suffixes match the M4.5
# task.output_names contract verbatim: jets -> Flavours[c].px (pb/pc/pu),
# tracks -> p{origin} for the 8 origin classes (tasks.py:1233-1255).
JET_SUFFIXES = ["pb", "pc", "pu"]
ORIGIN_SUFFIXES = [f"p{c}" for c in ORIGIN_CLASSES]

# Vertexing (track_vertexing -> bare `VertexIndex` i8) is a DEFERRED family (P2);
# EXCLUDE it from the parity comparison and record it (design §4b — log, never
# silently drop a column from the oracle file).
DEFERRED_COLUMNS = {"tracks": ["VertexIndex"]}


# ---------------------------------------------------------------------------
# fixtures: one synthetic source H5 + one trained GN2v2 checkpoint, shared by
# BOTH eval paths so the only difference under test is the sink.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("h5_parity")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    # exactly four underscore parts -> the v1 sample heuristic yields 'ttbar'
    h5_path = base / "pp_output_test_ttbar.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


def _overrides(data) -> list[str]:
    return [
        f"--data.modules.reader.init_args.schema={data['schema']}",
        f"--model.modules.norm.init_args.norm_dict={data['nd']}",
        "--trainer.accelerator=cpu",
        "--trainer.logger=false",
        "--callbacks.progress=null",
    ]


@pytest.fixture(scope="module")
def ckpt(data, tmp_path_factory) -> Path:
    """Train the GN2v2 dummy for one short epoch (shared by both eval sinks).

    Returns
    -------
    Path
        The trained checkpoint.
    """
    fit_dir = tmp_path_factory.mktemp("h5_parity_fit")
    rc = main([
        "fit",
        "--config",
        str(DUMMY_CFG),
        f"--data.train_file={data['h5']}",
        f"--data.val_file={data['h5']}",
        *_overrides(data),
        f"--trainer.default_root_dir={fit_dir}",
        "--trainer.max_epochs=1",
        "--trainer.limit_train_batches=2",
        "--trainer.limit_val_batches=2",
        "--trainer.num_sanity_val_steps=0",
        "--trainer.log_every_n_steps=1",
    ])
    assert rc == 0
    ckpts = sorted(fit_dir.rglob("*.ckpt"))
    assert ckpts, f"no checkpoint under {fit_dir}"
    return ckpts[0]


# ---------------------------------------------------------------------------
# path (a): the M4.5 WriterCallback oracle (proven salt2 test CLI harness)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def oracle_h5(data, ckpt, tmp_path_factory) -> Path:
    """Eval H5 from the M4.5 ``WriterCallback`` (the P1 oracle, design §4a).

    Returns
    -------
    Path
        The oracle eval H5 file.
    """
    out = tmp_path_factory.mktemp("oracle") / "oracle.h5"
    rc = main([
        "test",
        "--config",
        str(DUMMY_CFG),
        f"--data.test_file={data['h5']}",
        f"--ckpt_path={ckpt}",
        f"--data.num_test={N_TEST}",
        f"--trainer.default_root_dir={data['dir']}",
        f"--writers.output={out}",
        *_overrides(data),
    ])
    assert rc == 0
    assert out.exists()
    return out


# ---------------------------------------------------------------------------
# path (b): the producers -> H5OutputWriter chain (programmatic Trainer.test)
# ---------------------------------------------------------------------------


def _model_with_producers(data) -> SaltModule:
    """Build the GN2v2 model and ADD the P1 output producers.

    The producers are TEST-only by demand (design §4 risk 4): pruned from FIT/VAL
    so the trained checkpoint — which never saw them — loads unchanged. They are
    the IDENTITY ``TaskOutput`` (P0 passthrough clone) reading each task's
    published ``preds.*``; see the module docstring's TRANSITIONAL-STATE NOTE for
    why an identity (not conversion) producer is correct here — the M4.5 task
    still converts in ``forward``, so ``preds.*`` is already the eval-ready value,
    and the producer's job is to hand it to the SINK without re-converting. The
    conversion ops' math is proven separately in ``test_outputs_producers.py``.

    track_vertexing is left WITHOUT a producer (deferred P2), and opted OUT of
    the TEST eval path (expose:[fit,val]) so its ``preds.*`` port is FIT/VAL-only
    and the dead-preds gate does not require a sink for it (design §4.2/§4 risk
    5). The training checkpoint is unaffected — FIT/VAL keep the head.

    Returns
    -------
    SaltModule
        The GN2v2 model with identity output producers added.
    """
    modules = build_gn2v2_modules(data["nd"])
    modules["track_vertexing"].expose_modes = Mode.FIT | Mode.VAL
    modules["jet_probs"] = TaskOutput(task="jets_classification", stream="jets")
    modules["track_origin_probs"] = TaskOutput(task="track_origin", stream="tracks")
    model = SaltModule(modules, lrs={"initial": 1e-7, "max": 1e-3, "end": 1e-5, "pct_start": 0.01})
    model.name = RUN_NAME
    return model


def _h5_output_writer(out: Path) -> H5OutputWriter:
    """The P1 sink mirroring the M4.5 inputs_copy -> tasks -> pad_mask layout.

    copy_inputs copies ALL source fields for each stream (the v1 InputCopyWriter
    default — every dtype field rides along, including labels/truth), in the file
    field order; the output columns reproduce the classification prob columns;
    write_pad_mask adds the boolean tracks mask. The column ORDER (copies, then
    probs, then mask) matches the oracle group layout.

    Returns
    -------
    H5OutputWriter
        The configured P1 sink.
    """
    return H5OutputWriter(
        outputs=[
            OutputColumn(key="outputs.jets.jets_classification", suffixes=JET_SUFFIXES),
            OutputColumn(key="outputs.tracks.track_origin", suffixes=ORIGIN_SUFFIXES),
        ],
        copy_inputs={"jets": [], "tracks": []},  # [] -> all source fields (v1 default)
        write_pad_mask=["tracks"],
        output=str(out),
    )


@pytest.fixture(scope="module")
def p1_h5(data, ckpt, tmp_path_factory) -> Path:
    """Eval H5 from the producers -> ``H5OutputWriter`` chain (the P1 path).

    Returns
    -------
    Path
        The P1 eval H5 file.
    """
    out = tmp_path_factory.mktemp("p1") / "p1.h5"
    model = _model_with_producers(data)
    dm = GraphDataModule(
        modules={
            "reader": H5StructuredReader(
                groups={
                    "jets": {"global_object": True},
                    "tracks": {"global_object": False},
                },
                schema=str(data["schema"]),
            ),
            "features": Features(
                variables={"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}
            ),
            "labels": Labels(dtype_policy="int64-for-int"),
        },
        test_file=str(data["h5"]),
        batch_size=100,
        num_workers=0,
        num_test=N_TEST,
        pin_memory=False,
        persistent_workers=False,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[_h5_output_writer(out)],
    )
    trainer.test(model, datamodule=dm, ckpt_path=str(ckpt))
    assert out.exists()
    return out


# ---------------------------------------------------------------------------
# the semantic-H5-parity comparison (design §4a)
# ---------------------------------------------------------------------------


def _drop_deferred(names: list[str], stream: str) -> list[str]:
    """Names in `stream` with the DEFERRED (P2) columns removed.

    Returns
    -------
    list[str]
        The names with the deferred columns excluded, order preserved.
    """
    deferred = set(DEFERRED_COLUMNS.get(stream, ()))
    return [n for n in names if n not in deferred]


def _compare_column(group: str, col: str, want: np.ndarray, got: np.ndarray) -> str | None:
    """Compare one column; return a diff message on mismatch, else None.

    Returns
    -------
    str | None
        A human-readable diff message, or ``None`` when the columns match.
    """
    if want.dtype != got.dtype:
        return f"{group}.{col}: dtype {want.dtype} (oracle) != {got.dtype} (p1)"
    if want.shape != got.shape:
        return f"{group}.{col}: shape {want.shape} (oracle) != {got.shape} (p1)"
    if np.issubdtype(want.dtype, np.floating):
        if not np.allclose(want, got, rtol=0.0, atol=_FLOAT_TOL, equal_nan=True):
            bad = int(np.argmax(np.abs(want.ravel() - got.ravel())))
            return (
                f"{group}.{col}: float values differ beyond atol={_FLOAT_TOL} — first worst "
                f"at flat idx {bad}: oracle={want.ravel()[bad]!r} p1={got.ravel()[bad]!r}"
            )
    elif not np.array_equal(want, got):
        bad = int(np.argmax(want.ravel() != got.ravel()))
        return (
            f"{group}.{col}: int/bool values differ (exact required) — first at flat idx "
            f"{bad}: oracle={want.ravel()[bad]!r} p1={got.ravel()[bad]!r}"
        )
    return None


class TestH5OutputWriterParity:
    def test_deferred_columns_present_in_oracle(self, oracle_h5):
        """Sanity: the DEFERRED columns DO exist in the oracle (we exclude, not miss)."""
        with h5py.File(oracle_h5) as f:
            for stream, cols in DEFERRED_COLUMNS.items():
                names = set(f[stream].dtype.names)
                missing = [c for c in cols if c not in names]
                assert not missing, f"deferred columns {missing} absent from oracle {stream!r}"

    def test_groups_match(self, oracle_h5, p1_h5):
        """Both sinks write the same H5 groups."""
        with h5py.File(oracle_h5) as a, h5py.File(p1_h5) as b:
            assert set(a.keys()) == set(b.keys())

    def test_semantic_h5_parity(self, oracle_h5, p1_h5):
        """Per-column array equality (ints/bools exact, floats <=1e-6), deferred excluded."""
        diffs: list[str] = []
        with h5py.File(oracle_h5) as a, h5py.File(p1_h5) as b:
            for group in a:
                oracle = a[group][:]
                p1 = b[group][:]
                # the compared column set = oracle columns minus the deferred (P2) ones
                want_cols = _drop_deferred(list(oracle.dtype.names), group)
                got_cols = list(p1.dtype.names)
                # the P1 file must carry EXACTLY the compared (non-deferred) columns,
                # in the SAME order (column order is part of the byte schema)
                if want_cols != got_cols:
                    diffs.append(
                        f"{group}: column set/order mismatch\n  oracle (minus deferred): "
                        f"{want_cols}\n  p1                     : {got_cols}"
                    )
                    continue
                if oracle.shape != p1.shape:
                    diffs.append(f"{group}: shape {oracle.shape} (oracle) != {p1.shape} (p1)")
                    continue
                diffs.extend(
                    msg
                    for col in want_cols
                    if (msg := _compare_column(group, col, oracle[col], p1[col])) is not None
                )
        assert not diffs, "SEMANTIC H5 PARITY FAILED:\n" + "\n".join(diffs)

    def test_probs_are_softmaxed(self, p1_h5):
        """The P1 prob columns are probabilities (sum ~1), not raw logits."""
        with h5py.File(p1_h5) as f:
            jets = f["jets"][:]
            tracks = f["tracks"][:]
            valid = ~tracks["mask"]
        jet_cols = [f"{RUN_NAME}_{s}" for s in JET_SUFFIXES]
        prob_sum = sum(jets[c].astype("f8") for c in jet_cols)
        assert np.allclose(prob_sum, 1.0, atol=1e-3)
        # padded track positions read 0.0 (masked softmax), valid sum to ~1
        origin_cols = [f"{RUN_NAME}_{s}" for s in ORIGIN_SUFFIXES]
        origin_sum = sum(tracks[c].astype("f8") for c in origin_cols)
        assert np.allclose(origin_sum[valid], 1.0, atol=1e-3)
        assert np.allclose(origin_sum[~valid], 0.0, atol=1e-6)
