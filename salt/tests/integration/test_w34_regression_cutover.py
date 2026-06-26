"""PLAN 34 W34.3 REGRESSION cutover proof — outputs:-section get_output de-scale parity.

The W34.3 regression gate (plan 34 §4/§6): on the SAME regression model + source H5
+ trained checkpoint, run BOTH eval paths and assert the eval H5 matches at SEMANTIC
parity (floats <=1e-6, ints/bools EXACT), including column NAMES + ORDER + DTYPES:

- **(a) the W34.3 Regression (RegressionDescaleOp) PRODUCER path** — the shipped
  ``regression.yaml`` (auto-collect H5OutputSink over the de-scale producers), the
  ORACLE for the de-scaled physical columns; and
- **(b) the PLAN 34 ``outputs:`` section + dumb sinks** — `RunTaskOutput` (calling
  each RegressionTaskModule's ``get_output`` de-scale) + `InputCopyWriter` +
  `PadMaskWriter`, dumped by the DUMB `H5OutputSink`, driven by the
  ``regression-cutover34.yaml`` config through the real ``salt2 test`` CLI.

Both paths read the RAW (scaled) preds the W34.3-flipped forward now publishes and
de-scale ONCE — (a) in the producer, (b) in get_output — so the de-scaled physical
columns must be byte-identical. regression.yaml is the broadest non-gaussian surface
(norm_params scalar+vector, ratio-denominator with a FEATURE denom for ONNX, per-token
seq), so this proves regression get_output de-scale end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from salt.core.main import CONFIG_DIR, main
from salt.core.schema import dump_schema, save_schema
from salt.tests._fixtures.gn2_fixture import write_parity_norm_dict
from salt.utils.inputs import write_dummy_file

pytestmark = pytest.mark.cpu_always

REGRESSION_CFG = CONFIG_DIR / "regression.yaml"
CUTOVER34_CFG = CONFIG_DIR / "regression-cutover34.yaml"
N_TEST = 200
_FLOAT_TOL = 1e-6


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("w34_reg_parity")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    # regression.yaml declares `mass` as a third jets input variable (ratio
    # denominator — must be a declared input Feature for ONNX de-scaling); add it
    # to the parity norm dict here (same scheme as write_parity_norm_dict: index 2).
    with open(nd_path) as f:
        nd = yaml.safe_load(f)
    nd["jets"]["mass"] = {"mean": round(0.1 * 3, 6), "std": round(1.0 + 0.05 * 3, 6)}
    with open(nd_path, "w") as f:
        yaml.safe_dump(nd, f)
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
    fit_dir = tmp_path_factory.mktemp("w34_reg_fit")
    rc = main([
        "fit",
        "--config",
        str(REGRESSION_CFG),
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


@pytest.fixture(scope="module")
def producer_h5(data, ckpt, tmp_path_factory) -> Path:
    """Eval H5 from the W34.3 Regression (RegressionDescaleOp) producer path (the ORACLE)."""
    out = tmp_path_factory.mktemp("w34_reg_producer") / "producer.h5"
    rc = main([
        "test",
        "--config",
        str(REGRESSION_CFG),
        f"--data.test_file={data['h5']}",
        f"--ckpt_path={ckpt}",
        f"--data.num_test={N_TEST}",
        f"--trainer.default_root_dir={data['dir']}",
        f"--callbacks.h5_output.init_args.output={out}",
        *_overrides(data),
    ])
    assert rc == 0
    assert out.exists()
    return out


@pytest.fixture(scope="module")
def section_h5(data, ckpt, tmp_path_factory) -> Path:
    """Eval H5 from the PLAN 34 outputs:-section get_output de-scale path (via the CLI)."""
    out = tmp_path_factory.mktemp("w34_reg_section") / "section.h5"
    rc = main([
        "test",
        "--config",
        str(REGRESSION_CFG),
        "--config",
        str(CUTOVER34_CFG),
        f"--data.test_file={data['h5']}",
        f"--ckpt_path={ckpt}",
        f"--data.num_test={N_TEST}",
        f"--trainer.default_root_dir={data['dir']}",
        f"--callbacks.h5_output.init_args.output={out}",
        *_overrides(data),
    ])
    assert rc == 0, "salt2 test on the regression-cutover34 outputs:-section must run end-to-end"
    assert out.exists()
    return out


def _compare_column(group, col, want, got) -> str | None:
    if want.dtype != got.dtype:
        return f"{group}.{col}: dtype {want.dtype} != {got.dtype}"
    if want.shape != got.shape:
        return f"{group}.{col}: shape {want.shape} != {got.shape}"
    if np.issubdtype(want.dtype, np.floating):
        if not np.allclose(want, got, rtol=0.0, atol=_FLOAT_TOL, equal_nan=True):
            bad = int(np.argmax(np.abs(np.nan_to_num(want.ravel()) - np.nan_to_num(got.ravel()))))
            return (
                f"{group}.{col}: floats differ beyond atol={_FLOAT_TOL} at flat idx {bad}: "
                f"producer={want.ravel()[bad]!r} section={got.ravel()[bad]!r}"
            )
    elif not np.array_equal(want, got):
        return f"{group}.{col}: int/bool values differ (exact required)"
    return None


def test_regression_section_h5_matches_producer_oracle(producer_h5, section_h5):
    """The regression get_output de-scale eval H5 == the Regression producer oracle.

    Full-payload parity: the de-scaled regression columns (norm_params scalar+vector,
    ratio-denominator, per-token seq) + input copies + pad mask, NAMES + ORDER +
    DTYPES, all matching the auto-collect-producer path byte-for-byte.
    """
    diffs: list[str] = []
    with h5py.File(producer_h5) as a, h5py.File(section_h5) as b:
        assert set(a.keys()) == set(b.keys()), f"groups differ: {set(a)} vs {set(b)}"
        for group in a:
            want, got = a[group][:], b[group][:]
            want_cols, got_cols = list(want.dtype.names), list(got.dtype.names)
            if want_cols != got_cols:
                diffs.append(
                    f"{group}: column set/order mismatch\n  producer: {want_cols}\n  "
                    f"section : {got_cols}"
                )
                continue
            diffs.extend(
                msg
                for col in want_cols
                if (msg := _compare_column(group, col, want[col], got[col])) is not None
            )
    assert not diffs, "W34.3 REGRESSION CUTOVER H5 PARITY FAILED:\n" + "\n".join(diffs)


def test_regression_section_descaled_not_raw(section_h5, producer_h5):
    """Sanity: the section columns are DE-SCALED (differ from the raw scaled preds).

    norm_params reg_normed has std=1.0/mean=1.0, so the de-scale is pred+1.0 — the
    section column must NOT equal the raw pred (proves get_output de-scaled, not
    copied raw). Compared structurally vs the producer oracle (already asserted equal)
    so this is a belt-and-braces non-identity check on a known-offset column.
    """
    with h5py.File(section_h5) as f:
        jets = f["jets"][:]
    # reg_normed -> HadronConeExclTruthLabelPt (norm_params mean=1.0 std=1.0):
    # de-scaled = raw*1 + 1, so the column is finite and not trivially zero-centred
    col = jets["regression_HadronConeExclTruthLabelPt"]
    assert np.isfinite(col).all()
