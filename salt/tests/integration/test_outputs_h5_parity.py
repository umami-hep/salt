"""End-to-end eval-H5 checks for the EXPLICIT-outputs H5 sink path (gn2v2-dummy.yaml).

Historical note (plan 47): this file was the frozen-oracle byte-parity gate
diffing the sink's eval H5 against a committed ``gn2v2_dummy_oracle`` fixture
(the legacy ``WriterCallback`` output). The v1 writers path is deleted and the
byte-for-byte parity was proven and CLOSED at the v1 pin ``29c67a1`` (see the
parity-closure section of ``salt/core/README.md`` for the closure evidence and
the regeneration recipe). What remains are the v2-only checks: the shipped
``gn2v2-dummy.yaml`` (explicit sink ``outputs:`` tables — the complement of the
``outputs:``-section path gated by ``test_outputs_section``) fits and
evaluates end-to-end through the real CLI, and the eval H5's contents are
asserted from first principles.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from salt.core.main import CONFIG_DIR, main
from salt.core.schema import dump_schema, save_schema
from salt.core.testing.inputs import write_dummy_file
from salt.tests._fixtures.gn2v2_fixture import ORIGIN_CLASSES, write_parity_norm_dict

DUMMY_CFG = CONFIG_DIR / "gn2v2-dummy.yaml"
RUN_NAME = "GN2v2_dummy"  # the dummy config's `name:`
N_TEST = 300  # data.num_test

JET_SUFFIXES = ["pb", "pc", "pu"]
ORIGIN_SUFFIXES = [f"p{c}" for c in ORIGIN_CLASSES]


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
def cli_h5(data, ckpt, tmp_path_factory) -> Path:
    """Eval H5 from ``salt2 test`` on the shipped gn2v2-dummy.yaml (explicit tables)."""
    out = tmp_path_factory.mktemp("h5_writer_cli") / "eval.h5"
    root = tmp_path_factory.mktemp("h5_writer_cli_root")
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
    assert rc == 0, "salt2 test on the shipped gn2v2-dummy.yaml must run end-to-end"
    assert out.exists(), f"the CLI wrote no eval H5 at {out}"
    return out


class TestExplicitSinkCliE2E:
    """The shipped gn2v2-dummy.yaml drives the explicit-tables H5 sink end-to-end."""

    def test_cli_writes_eval_h5(self, cli_h5):
        """``salt2 test`` writes a non-empty eval H5 with the reader-stream groups."""
        with h5py.File(cli_h5) as f:
            assert set(f.keys()) >= {"jets", "tracks"}
            assert f["jets"].shape[0] == N_TEST

    def test_task_columns_present(self, cli_h5):
        """The explicit outputs: tables mint the jet + origin prob columns (f4)."""
        with h5py.File(cli_h5) as f:
            jets, tracks = f["jets"].dtype, f["tracks"].dtype
        for s in JET_SUFFIXES:
            col = f"{RUN_NAME}_{s}"
            assert col in jets.names, f"missing jet prob column {col}"
            assert np.issubdtype(jets[col], np.floating)
        for s in ORIGIN_SUFFIXES:
            col = f"{RUN_NAME}_{s}"
            assert col in tracks.names, f"missing origin prob column {col}"
            assert np.issubdtype(tracks[col], np.floating)

    def test_probs_are_softmaxed_not_double_converted(self, cli_h5):
        """The prob columns are probabilities (sum ~1) — converted EXACTLY ONCE."""
        with h5py.File(cli_h5) as f:
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
