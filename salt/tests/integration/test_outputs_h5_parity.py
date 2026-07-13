"""P1 SEMANTIC H5 PARITY GATE for the producers -> ``H5OutputWriter`` chain (plan 01, P1)."""

from __future__ import annotations

import shutil
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
from salt.tests._fixtures.gn2v2_fixture import ORIGIN_CLASSES, build_gn2v2_modules
from salt.tests._fixtures.gn2v2_fixture import JET_VARIABLES, TRACK_VARIABLES
from salt.core.testing.inputs import write_dummy_file

# frozen oracle fixture directory
# Generated once (commit a9e2ac2) from the WriterCallback path on the synthetic
# GN2v2 dummy model + data; never regenerated automatically — see provenance.json.
FIXTURE_DIR = Path(__file__).parent.parent / "_fixtures" / "gn2v2_dummy_oracle"

DUMMY_CFG = CONFIG_DIR / "gn2v2-dummy.yaml"
RUN_NAME = "GN2v2_dummy"  # the dummy config's `name:`
N_TEST = 300  # data.num_test for both sinks
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


# fixtures: frozen data + checkpoint (loaded from the committed fixture dir);
# the WriterCallback oracle H5 is also frozen — see FIXTURE_DIR/provenance.json.
# Only the P1 path (path b) and the cutover CLI path (path c) run live.


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    """Regenerate the synthetic source H5 deterministically; return its paths."""
    nd = FIXTURE_DIR / "norm_dict.yaml"
    h5 = tmp_path_factory.mktemp("data") / "data.h5"
    write_dummy_file(h5, nd)  # deterministic (default_rng(42)) → matches the frozen ckpt/oracle inputs
    return {
        "dir": FIXTURE_DIR,
        "h5": h5,
        "nd": nd,
        "schema": FIXTURE_DIR / "schema.yaml",
    }


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
    """Return a byte-identical tmp copy of the frozen GN2v2 dummy checkpoint."""
    src = FIXTURE_DIR / "ckpt.ckpt"
    assert src.exists(), f"frozen ckpt not found at {src} — re-run the oracle generator"
    dst = tmp_path_factory.mktemp("ckpt") / "ckpt.ckpt"
    shutil.copyfile(src, dst)
    return dst


# path (a): the frozen M4.5 WriterCallback oracle (committed static fixture)


@pytest.fixture(scope="module")
def oracle_h5() -> Path:
    """Return the frozen WriterCallback oracle H5 from the fixture directory."""
    oracle_path = FIXTURE_DIR / "oracle.h5"
    assert oracle_path.exists(), (
        f"frozen oracle not found at {oracle_path} — re-run the oracle generator"
    )
    return oracle_path


# path (b): the producers -> H5OutputWriter chain (programmatic Trainer.test)


def _model_with_producers(data) -> SaltModule:
    """Build the GN2v2 model and ADD the REAL P1 classification conversion producers."""
    modules = build_gn2v2_modules(data["nd"])
    modules["track_vertexing"].expose_modes = Mode.FIT | Mode.VAL
    modules["jet_probs"] = ClassProbs(task="jets_classification", stream="jets")
    modules["track_origin_probs"] = SeqClassProbs(task="track_origin", stream="tracks")
    model = SaltModule(modules, lrs={"initial": 1e-7, "max": 1e-3, "end": 1e-5, "pct_start": 0.01})
    model.name = RUN_NAME
    return model


def _h5_output_writer(out: Path) -> H5OutputWriter:
    """The P1 sink mirroring the M4.5 inputs_copy -> tasks -> pad_mask layout."""
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
    """Eval H5 from the producers -> ``H5OutputWriter`` chain (the P1 path)."""
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


# the semantic-H5-parity comparison (design §4a)


def _drop_deferred(names: list[str], stream: str) -> list[str]:
    """Names in `stream` with the DEFERRED (P2) columns removed."""
    deferred = set(DEFERRED_COLUMNS.get(stream, ()))
    return [n for n in names if n not in deferred]


def _compare_column(group: str, col: str, want: np.ndarray, got: np.ndarray) -> str | None:
    """Compare one column; return a diff message on mismatch, else None."""
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
        """The P1 prob columns are probabilities (sum ~1) — converted EXACTLY ONCE."""
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
    """The ``gn2v2-dummy-cutover.yaml`` config is the live wiring source of truth."""

    def test_cutover_config_fits_end_to_end(self, data, tmp_path_factory):
        """The migrated DUMMY_CFG instantiates + fits (producers pruned from FIT)."""
        fit_dir = tmp_path_factory.mktemp("cutover_fit")
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
        ])
        assert rc == 0


# path (c): the LIVE cutover through the real ``salt2 test`` CLI (closes the
# "live cutover proven only programmatically" gap — the cutover config drives
# the producers -> H5OutputWriter chain end-to-end, the same way a user runs it)


@pytest.fixture(scope="module")
def cutover_cli_h5(data, ckpt, tmp_path_factory) -> Path:
    """Eval H5 from ``salt2 test`` on the migrated gn2v2-dummy.yaml (the CLI path)."""
    out = tmp_path_factory.mktemp("cutover_cli") / "cutover_cli.h5"
    root = tmp_path_factory.mktemp("cutover_cli_root")
    rc = main([
        "test",
        "--config",
        str(DUMMY_CFG),
        f"--data.test_file={data['h5']}",
        f"--ckpt_path={ckpt}",
        f"--data.num_test={N_TEST}",
        f"--trainer.default_root_dir={root}",
        f"--callbacks.h5_output.init_args.output={out}",
        *_overrides(data),
    ])
    assert rc == 0, "salt2 test on the migrated DUMMY_CFG must run end-to-end"
    assert out.exists(), f"the CLI wrote no eval H5 at {out}"
    return out


class TestCutoverCliE2E:
    """The migrated gn2v2-dummy.yaml drives the new path through ``salt2 test`` end-to-end."""

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
        """Per-column array equality vs the M4.5 oracle (deferred P2 columns excluded)."""
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
