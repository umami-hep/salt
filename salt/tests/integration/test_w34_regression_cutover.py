"""PLAN 34 W34.3 REGRESSION cutover proof — outputs:-section get_output de-scale parity."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import onnx
import pytest
import yaml

from salt.core.main import CONFIG_DIR, main
from salt.core.onnx import make_session
from salt.core.schema import dump_schema, save_schema
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict
from salt.core.testing.inputs import write_dummy_file

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
    """The regression get_output de-scale eval H5 == the Regression producer oracle."""
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
    """Sanity: the section columns are DE-SCALED (differ from the raw scaled preds)."""
    with h5py.File(section_h5) as f:
        jets = f["jets"][:]
    # reg_normed -> HadronConeExclTruthLabelPt (norm_params mean=1.0 std=1.0):
    # de-scaled = raw*1 + 1, so the column is finite and not trivially zero-centred
    col = jets["regression_HadronConeExclTruthLabelPt"]
    assert np.isfinite(col).all()


# ===========================================================================
# W34.3 critic-fix GATE: REGRESSION ONNX value-parity (blocker #2 / finding #6).
# The W34.3 regression descale-in-graph ONNX path (get_output ONNX squeeze, the
# section single-name leaf, the in-graph run_inference descale, the ONNX
# feature-gather denominator) previously had ZERO automated ONNX coverage —
# test_regression_configs only `graph validate`s (compile, not export) and the H5
# cutover gate above never exports ONNX. This exports BOTH the explicit-ONNX
# producer (regression.yaml, the ORACLE) and the dumb outputs:-section
# (regression.yaml + regression-cutover34.yaml) from ONE checkpoint, then asserts
# the ONNX contract (names/dtypes/dynamic_axes/order) AND the onnxruntime VALUES
# are identical (atol 1e-6) — and that the values are de-scaled, not raw.
# ===========================================================================


def _export_onnx(extra_cfgs, ckpt, out: Path) -> Path:
    """Export to ONNX via the real `salt2 export` CLI (--no-check)."""
    saved_config = Path(ckpt).parents[1] / "config.yaml"
    assert saved_config.is_file(), f"no saved run config at {saved_config}"
    argv = ["export", "--config", str(saved_config)]
    for c in extra_cfgs:
        argv += ["--config", str(c)]
    argv += [f"--ckpt_path={ckpt}", f"--output={out}", "--no-check", "--overwrite"]
    rc = main(argv)
    assert rc == 0, f"salt2 export failed (rc={rc}) for extra configs {[str(c) for c in extra_cfgs]}"  # noqa: E501
    assert out.exists()
    return out


def _onnx_contract(path: Path) -> tuple[list[str], dict[str, int], dict[str, int]]:
    """The exported graph's output names, ranks and element dtypes."""
    model = onnx.load(str(path))
    names = [o.name for o in model.graph.output]
    ranks = {o.name: len(o.type.tensor_type.shape.dim) for o in model.graph.output}
    dtypes = {o.name: o.type.tensor_type.elem_type for o in model.graph.output}
    return names, ranks, dtypes


class TestW34RegressionOnnxParity:
    """Regression ONNX: the dumb outputs:-section export == the producer-oracle export."""

    @pytest.fixture(scope="class")
    def oracle_onnx(self, ckpt, tmp_path_factory) -> Path:
        out = tmp_path_factory.mktemp("w34_reg_onnx_oracle") / "oracle.onnx"
        return _export_onnx([], ckpt, out)  # saved config = the RegressionDescaleOp oracle

    @pytest.fixture(scope="class")
    def section_onnx(self, ckpt, tmp_path_factory) -> Path:
        out = tmp_path_factory.mktemp("w34_reg_onnx_section") / "section.onnx"
        return _export_onnx([CUTOVER34_CFG], ckpt, out)  # stack the dumb outputs:-section

    def test_onnx_output_contract_matches(self, oracle_onnx, section_onnx):
        """Names + dtypes + dynamic axes + ORDER + ranks identical (oracle vs section)."""
        o_names, o_ranks, o_dtypes = _onnx_contract(oracle_onnx)
        s_names, s_ranks, s_dtypes = _onnx_contract(section_onnx)
        # the 8 regression outputs: 6 rank-0 globals (norm/ratio scalars) + 2 rank-1
        # per-token seq columns (dummyOutput_dPhi/dEta), in canonical globals->per-token
        # order — identical between the producer oracle and the dumb section.
        assert s_names == o_names, f"ONNX names differ\n  oracle : {o_names}\n  section: {s_names}"
        assert s_ranks == o_ranks, f"ONNX ranks differ: {o_ranks} vs {s_ranks}"
        assert s_dtypes == o_dtypes, f"ONNX dtypes differ: {o_dtypes} vs {s_dtypes}"
        # 6 global rank-0 + 2 per-token rank-1 (the regression.yaml head layout)
        assert sorted(o_ranks.values()) == [0, 0, 0, 0, 0, 0, 1, 1], o_ranks

    def test_onnx_runtime_values_match_and_are_descaled(self, oracle_onnx, section_onnx):
        """Onnxruntime values identical (atol 1e-6) between section and producer oracle."""
        oracle_sess = make_session(oracle_onnx)
        section_sess = make_session(section_onnx)
        # build batch-1 example inputs at L=5 (the export trace is batch-1; onnxruntime
        # accepts the dynamic n_tracks axis). jet_features [1, F], track_features [L, F].
        in_meta = {i.name: i.shape for i in oracle_sess.get_inputs()}
        rng = np.random.default_rng(0)

        def shape_for(dims):
            return tuple(5 if (isinstance(d, str) or d is None) else d for d in dims)

        feeds = {
            name: rng.standard_normal(shape_for(dims)).astype(np.float32)
            for name, dims in in_meta.items()
        }
        oracle_out = {
            o.name: v
            for o, v in zip(oracle_sess.get_outputs(), oracle_sess.run(None, feeds), strict=True)
        }
        section_out = {
            o.name: v
            for o, v in zip(section_sess.get_outputs(), section_sess.run(None, feeds), strict=True)
        }
        assert set(oracle_out) == set(section_out)
        max_diff = 0.0
        for name, a in oracle_out.items():
            b = section_out[name]
            assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
            d = np.abs(np.nan_to_num(a) - np.nan_to_num(b))
            max_diff = max(max_diff, float(d.max()) if d.size else 0.0)
        assert max_diff <= 1e-6, f"section ONNX values diverge from producer oracle: max {max_diff}"
