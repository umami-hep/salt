"""End-to-end eval-H5 check for the base gn2v2-dummy.yaml — implicit H5 sink.

Plan 50 Phase B: gn2v2-dummy.yaml migrated OFF the explicit-sink ``outputs:``
OutputColumn table onto the ``outputs:`` section path (two mode-split
RunTaskOutput writers). No config declares H5OutputSink/OnnxExportSink — the
``salt2 test`` command wires the H5 sink over the section. This test fits +
evaluates the shipped config end-to-end through the real CLI (proving the
implicit-sink runtime path) and asserts the eval H5's TASK columns against the
committed Phase-A golden (``gn2v2-dummy.json``) — a PARSED-JSON contract, not a
text oracle.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from salt.core.main import CONFIG_DIR, main
from salt.core.schema import dump_schema, save_schema
from salt.core.testing.inputs import write_dummy_file
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict

DUMMY_CFG = CONFIG_DIR / "gn2v2-dummy.yaml"
GOLDEN = Path(__file__).resolve().parents[1] / "_fixtures/output_goldens/gn2v2-dummy.json"
RUN_NAME = "GN2v2_dummy"  # the dummy config's `name:`
N_TEST = 300


def _golden_task_columns() -> dict[str, list[str]]:
    """Per-stream ordered flat TASK column names from the committed golden."""
    golden = json.loads(GOLDEN.read_text())
    per_stream: dict[str, list[str]] = {}
    for col in golden["h5"]["columns"]:
        per_stream.setdefault(col["stream"], []).extend(col["column_names"])
    return per_stream


def _expected_full_columns(src_cols: dict[str, list[str]]) -> dict[str, list[str]]:
    """The FULL ordered per-stream H5 column contract for the Phase-B gate.

    Byte-identical columns = input-copy source columns FIRST (in source-file
    order; an empty golden ``copy_inputs`` means the v1 copy-ALL default, so
    every source field is copied), then the task columns in golden order, then
    the trailing pad-mask column. Asserting the H5 dtype.names EQUAL this list
    (not merely contain it) enforces "no ADDED columns" — the exact Phase-C
    label-leak the gate forbids from landing early.
    """
    h5 = json.loads(GOLDEN.read_text())["h5"]
    tasks: dict[str, list[str]] = {}
    for col in h5["columns"]:
        tasks.setdefault(col["stream"], []).extend(col["column_names"])
    copy_cfg = h5["copy_inputs"]
    pad_streams = set(h5["write_pad_mask"])
    per_stream: dict[str, list[str]] = {}
    for stream, file_fields in src_cols.items():
        # input copies: explicit golden subset, else copy-all (source order)
        cols = list(copy_cfg.get(stream) or file_fields)
        cols += tasks.get(stream, [])  # task columns in golden order
        if stream in pad_streams:
            cols.append("mask")  # pad mask written last
        per_stream[stream] = cols
    return per_stream


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    """Synthetic norm dict + dummy H5 + schema artifact (all generated live)."""
    base = tmp_path_factory.mktemp("h5_writer_data")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
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
    """A live 1-epoch fit of the shipped gn2v2-dummy.yaml (also proves it fits)."""
    fit_dir = tmp_path_factory.mktemp("h5_writer_fit")
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
    assert rc == 0, "salt2 fit on the shipped gn2v2-dummy.yaml must run end-to-end"
    ckpts = sorted(fit_dir.rglob("*.ckpt"))
    assert ckpts, f"no checkpoint under {fit_dir}"
    return ckpts[0]


@pytest.fixture(scope="module")
def cli_h5(data, ckpt) -> Path:
    """Eval H5 from ``salt2 test`` — the H5 sink is IMPLICIT (wired by the command).

    No ``--callbacks.h5_output`` override: the sink is injected over the outputs:
    section and writes to the default ``{ckpt_dir}/{ckpt_stem}__test_{sample}.h5``.
    """
    rc = main([
        "test",
        "--config",
        str(DUMMY_CFG),
        f"--data.test_file={data['h5']}",
        f"--ckpt_path={ckpt}",
        f"--data.num_test={N_TEST}",
        f"--trainer.default_root_dir={data['dir']}",
        *_overrides(data),
    ])
    assert rc == 0, "salt2 test on the shipped gn2v2-dummy.yaml must run end-to-end"
    evals = sorted(ckpt.parent.glob("*__test_*.h5"))
    assert evals, f"the implicit H5 sink wrote no eval H5 next to {ckpt}"
    return evals[-1]


class TestImplicitSinkCliE2E:
    """The shipped gn2v2-dummy.yaml drives the implicit H5 sink end-to-end."""

    def test_cli_writes_eval_h5(self, cli_h5):
        """``salt2 test`` writes a non-empty eval H5 with the reader-stream groups."""
        with h5py.File(cli_h5) as f:
            assert set(f.keys()) >= {"jets", "tracks"}
            assert f["jets"].shape[0] == N_TEST

    def test_task_columns_match_golden(self, data, cli_h5):
        """The eval H5's columns EQUAL the committed Phase-A golden, per stream, in order.

        Exact-list equality (not membership) is the Phase-B gate: byte-identical
        columns with NO new label columns yet. Any ADDED column — a Phase-C
        label-emission leak, or an accidental extra leaf — fails here.
        """
        with h5py.File(data["h5"]) as src:
            src_cols = {"jets": list(src["jets"].dtype.names), "tracks": list(src["tracks"].dtype.names)}
        expected = _expected_full_columns(src_cols)
        with h5py.File(cli_h5) as f:
            present = {"jets": list(f["jets"].dtype.names), "tracks": list(f["tracks"].dtype.names)}
        assert present.keys() == expected.keys(), (
            f"H5 streams {sorted(present)} != golden streams {sorted(expected)}"
        )
        for stream, cols in expected.items():
            assert present[stream] == cols, (
                f"{stream} columns diverge from golden (added/removed/reordered): "
                f"got {present[stream]}, golden {cols}"
            )

    def test_probs_are_softmaxed_not_double_converted(self, cli_h5):
        """The prob columns are probabilities (sum ~1) — converted EXACTLY ONCE."""
        expected = _golden_task_columns()
        with h5py.File(cli_h5) as f:
            jets = f["jets"][:]
            tracks = f["tracks"][:]
            valid = ~tracks["mask"]
        jet_cols = expected["jets"]
        prob_sum = sum(jets[c].astype("f8") for c in jet_cols)
        assert np.allclose(prob_sum, 1.0, atol=1e-3)
        # padded track positions read 0.0 (masked softmax), valid sum to ~1
        origin_cols = expected["tracks"]
        origin_sum = sum(tracks[c].astype("f8") for c in origin_cols)
        assert np.allclose(origin_sum[valid], 1.0, atol=1e-3)
        assert np.allclose(origin_sum[~valid], 0.0, atol=1e-6)
