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

P1.5 FLIP DONE FOR CLASSIFICATION (the conversion producers are now LOAD-BEARING
end-to-end — verified in source, ``salt/core/nn/tasks.py``
``ClassificationTaskModule.forward``/``get_h5``): the P1.5 step (design §2,
"removes the TEST/ONNX mode-branches inside ``task.forward``") has been applied
to the CLASSIFICATION family. The classification ``forward`` now publishes RAW
logits in TEST (the softmax was removed), so ``preds.<stream>.<task>`` in the
executed TEST bundle is the RAW logits. This gate therefore wires the REAL
conversion producers — `ClassProbs` (``jets_classification`` -> global softmax)
and `SeqClassProbs` (``track_origin`` -> masked softmax) — which convert ONCE on
the new path. The M4.5 oracle path stays LIVE and ALSO converts once, because
the conversion relocated INTO ``ClassificationTaskModule.get_h5`` (which the M4.5
``TaskWriter`` calls): ``get_h5`` now ``run_inference``-s the raw leaf before
packing, so the oracle eval H5 is byte-identical to pre-flip. Conversion happens
in EXACTLY ONE place per path (producer for the new path, ``get_h5`` for the M4.5
path; never both on the same leaf) — no double-conversion. The conversion ops'
MATH is independently proven bitwise against the task ``run_inference`` oracle in
``test_outputs_producers.py`` (recon PR2). Regression/vertexing are NOT yet
flipped (P2): the GN2v2 ``track_vertexing`` head's eval column (``VertexIndex``)
stays a DEFERRED family and is excluded from the comparison (``DEFERRED_COLUMNS``).

If parity cannot be reached the assertion reports the exact column with expected
vs got — never weaken the tolerance to pass.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from lightning import Trainer

from salt.core import SaltModule
from salt.core.data import Features, H5StructuredReader, Labels
from salt.core.data.datamodule import GraphDataModule
from salt.core.graph.spec import Mode
from salt.core.main import CONFIG_DIR, main
from salt.core.outputs import ClassProbs, H5OutputWriter, OutputColumn, SeqClassProbs
from salt.core.schema import dump_schema, save_schema
from salt.tests.core.gn2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    write_parity_norm_dict,
)
from salt.tests.core.gn2v2_fixture import ORIGIN_CLASSES, build_gn2v2_modules
from salt.utils.inputs import write_dummy_file

DUMMY_CFG = CONFIG_DIR / "gn2v2-dummy.yaml"
CUTOVER_CFG = CONFIG_DIR / "gn2v2-dummy-cutover.yaml"  # the P1.5 live cutover
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
    """Build the GN2v2 model and ADD the REAL P1 classification conversion producers.

    Since the P1.5 flip the classification task ``forward`` publishes RAW logits
    in TEST (design §2), so the producers must do the eval conversion: `ClassProbs`
    (``jets_classification`` -> global softmax) and `SeqClassProbs`
    (``track_origin`` -> masked softmax), the SAME math the M4.5 oracle now runs
    in ``get_h5``. Each converts ONCE — the new path's single conversion site (the
    M4.5 oracle's is ``get_h5``; never both on the same leaf, no double-convert).

    The producers are TEST-only by demand (design §4 risk 4): pruned from FIT/VAL
    so the trained checkpoint — which never saw them — loads unchanged.

    track_vertexing is left WITHOUT a producer (deferred P2), and opted OUT of
    the TEST eval path (expose:[fit,val]) so its ``preds.*`` port is FIT/VAL-only
    and the dead-preds gate does not require a sink for it (design §4.2/§4 risk
    5). The training checkpoint is unaffected — FIT/VAL keep the head. This is the
    same wiring the ``gn2v2-dummy-cutover.yaml`` config encodes for the live CLI
    path.

    Returns
    -------
    SaltModule
        The GN2v2 model with the real classification conversion producers added.
    """
    modules = build_gn2v2_modules(data["nd"])
    modules["track_vertexing"].expose_modes = Mode.FIT | Mode.VAL
    modules["jet_probs"] = ClassProbs(task="jets_classification", stream="jets")
    modules["track_origin_probs"] = SeqClassProbs(task="track_origin", stream="tracks")
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

    def test_probs_are_softmaxed_not_double_converted(self, p1_h5):
        """The P1 prob columns are probabilities (sum ~1) — converted EXACTLY ONCE.

        The double-conversion guard (design §4 "double-conversion"): if the new
        path converted twice (e.g. the task forward still softmaxed AND the
        producer softmaxed) the columns would be softmax(softmax(logits)) — still
        in [0, 1] and summing to 1 per row, but a DIFFERENT distribution. The
        semantic-H5-parity test already pins the exact values vs the M4.5 oracle
        (which converts once in get_h5), so a double-convert would FAIL parity.
        This test additionally pins the basic "is a distribution" invariant
        (sum ~1, not raw logits) on both the global and the masked-softmax
        sequence head.
        """
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


class TestCutoverConfig:
    """The ``gn2v2-dummy-cutover.yaml`` config is the live wiring source of truth.

    It encodes EXACTLY the wiring the programmatic gate (`_model_with_producers`
    + `_h5_output_writer`) drives: the classification conversion producers
    (`ClassProbs`/`SeqClassProbs`) in ``model.modules``, ``track_vertexing``
    opted out of TEST (``expose: [fit, val]``), the M4.5 ``writers:`` nulled, and
    the `H5OutputWriter` as a ``callbacks:`` sink.

    Three levels of proof: (1) the config FITS end-to-end through the real CLI
    (``main(['fit', ...])`` instantiates the model + the producers + the sink;
    the producers are demand-pruned from FIT and the training path is unchanged),
    (2) the config CONTENT is asserted by parsing the YAML (the exact module
    classes / null-merge / callback class names), and (3) the config TESTS
    end-to-end through ``salt2 test`` and the resulting eval H5 matches the M4.5
    oracle (`TestCutoverCliE2E`). The ``test`` subcommand's writer-less refusal
    now recognises the callbacks-level `H5OutputWriter` sink (main.py
    ``_has_callback_persistence_sink``), so the LIVE cutover eval path is proven
    through the real CLI — closing the prior "live cutover proven only
    programmatically" gap (the programmatic `p1_h5` gate's wiring mirrors this
    config exactly, and the CLI path now corroborates it).
    """

    def test_cutover_config_fits_end_to_end(self, data, tmp_path_factory):
        """The cutover config instantiates + fits (producers pruned from FIT)."""
        fit_dir = tmp_path_factory.mktemp("cutover_fit")
        rc = main([
            "fit",
            "--config",
            str(DUMMY_CFG),
            "--config",
            str(CUTOVER_CFG),
            f"--data.train_file={data['h5']}",
            f"--data.val_file={data['h5']}",
            *_overrides(data),
            f"--trainer.default_root_dir={fit_dir}",
            "--trainer.max_epochs=1",
            "--trainer.limit_train_batches=2",
            "--trainer.limit_val_batches=2",
            "--trainer.num_sanity_val_steps=0",
        ])
        assert rc == 0

    def test_cutover_config_content(self):
        """The cutover YAML wires the producers, nulls M4.5 writers, adds the sink."""
        cfg = yaml.safe_load(CUTOVER_CFG.read_text())
        mods = cfg["model"]["modules"]
        # the classification conversion producers (real ops, NOT identity)
        assert mods["jet_probs"]["class_path"] == "salt.core.outputs.ClassProbs"
        assert mods["jet_probs"]["init_args"]["task"] == "jets_classification"
        assert mods["track_origin_probs"]["class_path"] == "salt.core.outputs.SeqClassProbs"
        assert mods["track_origin_probs"]["init_args"]["task"] == "track_origin"
        # track_vertexing opted out of TEST (deferred P2)
        assert mods["track_vertexing"]["init_args"]["expose"] == ["fit", "val"]
        # M4.5 writers nulled (null-merge deletes the WriterCallback)
        assert cfg["writers"]["modules"] == {
            "inputs_copy": None,
            "tasks": None,
            "pad_mask": None,
        }
        # H5OutputWriter is the live TEST sink (a callbacks: entry)
        h5 = cfg["callbacks"]["h5_output"]
        assert h5["class_path"] == "salt.core.outputs.H5OutputWriter"
        out_keys = {o["key"] for o in h5["init_args"]["outputs"]}
        assert out_keys == {
            "outputs.jets.jets_classification",
            "outputs.tracks.track_origin",
        }
        assert h5["init_args"]["write_pad_mask"] == ["tracks"]


# ---------------------------------------------------------------------------
# path (c): the LIVE cutover through the real ``salt2 test`` CLI (closes the
# "live cutover proven only programmatically" gap — the cutover config drives
# the producers -> H5OutputWriter chain end-to-end, the same way a user runs it)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cutover_cli_h5(data, ckpt, tmp_path_factory) -> Path:
    """Eval H5 from ``salt2 test`` on the gn2v2-dummy + cutover configs (the CLI path).

    Runs the LIVE cutover exactly as a user would — ``salt2 test --config
    gn2v2-dummy.yaml --config gn2v2-dummy-cutover.yaml --ckpt_path <trained>`` —
    so the callbacks-level `H5OutputWriter` persistence sink is recognised by the
    test-stage writer-less check (main.py ``_has_callback_persistence_sink``), the
    classification conversion producers run on the flipped (RAW-logits) TEST
    forward, and the eval H5 is written by the real CLI. The output path is
    overridden onto a tmp file so the assertion can read it back.

    Returns
    -------
    Path
        The cutover CLI eval H5 file.
    """
    out = tmp_path_factory.mktemp("cutover_cli") / "cutover_cli.h5"
    rc = main([
        "test",
        "--config",
        str(DUMMY_CFG),
        "--config",
        str(CUTOVER_CFG),
        f"--data.test_file={data['h5']}",
        f"--ckpt_path={ckpt}",
        f"--data.num_test={N_TEST}",
        f"--trainer.default_root_dir={data['dir']}",
        f"--callbacks.h5_output.init_args.output={out}",
        *_overrides(data),
    ])
    assert rc == 0, "salt2 test on the cutover config must run end-to-end (FIX 2)"
    assert out.exists(), f"the cutover CLI wrote no eval H5 at {out}"
    return out


class TestCutoverCliE2E:
    """The cutover config drives the new path through ``salt2 test`` end-to-end.

    This is G1 driven via the REAL CLI (not only ``Trainer.test``): the
    callbacks-level `H5OutputWriter` sink is now accepted by the test-stage check
    (FIX 2), so ``salt2 test`` on ``gn2v2-dummy.yaml`` + ``gn2v2-dummy-cutover.yaml``
    runs producers -> sink end-to-end and the eval H5 must match the M4.5 oracle
    at SEMANTIC parity (ints exact, floats <=1e-6), deferred (P2 vertexing)
    columns excluded — the same contract as the programmatic `p1_h5` gate.
    """

    def test_cli_writes_eval_h5(self, cutover_cli_h5):
        """``salt2 test`` on the cutover config writes a non-empty eval H5."""
        with h5py.File(cutover_cli_h5) as f:
            assert set(f.keys()) >= {"jets", "tracks"}
            assert f["jets"].shape[0] == N_TEST

    def test_cli_groups_match_oracle(self, oracle_h5, cutover_cli_h5):
        """The CLI cutover eval H5 writes the same groups as the M4.5 oracle."""
        with h5py.File(oracle_h5) as a, h5py.File(cutover_cli_h5) as b:
            assert set(a.keys()) == set(b.keys())

    def test_cli_semantic_h5_parity(self, oracle_h5, cutover_cli_h5):
        """Per-column array equality vs the M4.5 oracle (deferred P2 columns excluded).

        Proves the LIVE cutover (driven by the real ``salt2 test`` CLI) reproduces
        the M4.5 ``WriterCallback`` eval H5 — closing the gap where the cutover was
        previously proven only via the programmatic ``Trainer.test`` (`p1_h5`).
        """
        diffs: list[str] = []
        with h5py.File(oracle_h5) as a, h5py.File(cutover_cli_h5) as b:
            for group in a:
                oracle = a[group][:]
                cli = b[group][:]
                want_cols = _drop_deferred(list(oracle.dtype.names), group)
                got_cols = list(cli.dtype.names)
                if want_cols != got_cols:
                    diffs.append(
                        f"{group}: column set/order mismatch\n  oracle (minus deferred): "
                        f"{want_cols}\n  cli                    : {got_cols}"
                    )
                    continue
                if oracle.shape != cli.shape:
                    diffs.append(f"{group}: shape {oracle.shape} (oracle) != {cli.shape} (cli)")
                    continue
                diffs.extend(
                    msg
                    for col in want_cols
                    if (msg := _compare_column(group, col, oracle[col], cli[col])) is not None
                )
        assert not diffs, "CUTOVER CLI SEMANTIC H5 PARITY FAILED:\n" + "\n".join(diffs)
