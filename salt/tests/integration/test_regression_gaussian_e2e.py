"""End-to-end gate for the shipped ``regression_gaussian.yaml`` — fit, eval H5, ONNX.

Historical note: this file was the gaussian CUTOVER A/B gate, diffing
the explicit gaussian ``Regression``-producer eval path against the
``outputs:``-section path via a ``regression_gaussian-cutover34.yaml`` overlay.
The shipped ``regression_gaussian.yaml`` has since been migrated onto the
section + dumb sinks natively, which made the overlay a no-op and the A/B
comparison degenerate (both legs ran the identical path). The comparison seam
is retired (section==producer parity was proven while both paths existed — see
the parity-closure section of ``docs/architecture.md``); what remains is the
live single-leg coverage: the shipped config trains, evaluates and exports
through the real CLI, with the gaussian doubled columns and the ONNX contract
asserted from first principles. ``--no-check`` on export: the torch-vs-ONNX
sweep checker has no gaussian handling (orthogonal to the contract checks).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import onnx
import pytest

from salt.main import CONFIG_DIR, main
from salt.outputs.sinks.onnx import make_session
from salt.schema import dump_schema, save_schema
from salt.testing.inputs import write_dummy_file
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict

pytestmark = pytest.mark.cpu_always

GAUSSIAN_CFG = CONFIG_DIR / "regression_gaussian.yaml"
N_TEST = 200


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("gauss_e2e")
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
    fit_dir = tmp_path_factory.mktemp("gauss_e2e_fit")
    rc = main([
        "fit",
        "--config",
        str(GAUSSIAN_CFG),
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
def eval_h5(data, ckpt) -> Path:
    """Eval H5 from the ``outputs:``-section gaussian get_output path (via the CLI).

    The H5 sink is IMPLICIT (wired by the command) — no
    ``--callbacks.h5_output`` override; read the default-templated eval H5.
    """
    rc = main([
        "test",
        "--config",
        str(GAUSSIAN_CFG),
        f"--data.test_file={data['h5']}",
        f"--ckpt_path={ckpt}",
        f"--data.num_test={N_TEST}",
        f"--trainer.default_root_dir={data['dir']}",
        *_overrides(data),
    ])
    assert rc == 0, "salt test on the shipped regression_gaussian.yaml must run end-to-end"
    evals = sorted(ckpt.parent.glob("*__test_*.h5"))
    assert evals, f"the implicit H5 sink wrote no eval H5 next to {ckpt}"
    return evals[-1]


def test_gaussian_eval_h5_has_stddev_columns(eval_h5):
    """The eval H5 carries the gaussian doubled columns (mean + _stddev), both streams."""
    with h5py.File(eval_h5) as f:
        jet_cols = set(f["jets"].dtype.names)
        track_cols = set(f["tracks"].dtype.names)
        n_rows = f["jets"].shape[0]
    assert n_rows == N_TEST
    assert any(c.endswith("_stddev") for c in jet_cols), f"no gaussian stddev in jets: {jet_cols}"
    assert any(
        c.endswith("_stddev") for c in track_cols
    ), f"no gaussian stddev in tracks: {track_cols}"


# ONNX: the gaussian descale-in-graph export path (get_output ONNX squeeze + the
# section single-name leaf naming + the in-graph means‖softplus-stddev concat)
# through the real `salt export` CLI.


class TestGaussianOnnxContract:
    """The shipped regression_gaussian.yaml exports a well-formed ONNX contract."""

    @pytest.fixture(scope="class")
    def exported(self, ckpt, tmp_path_factory) -> Path:
        out = tmp_path_factory.mktemp("gauss_e2e_onnx") / "gaussian.onnx"
        saved_config = Path(ckpt).parents[1] / "config.yaml"
        assert saved_config.is_file(), f"no saved run config at {saved_config}"
        rc = main([
            "export",
            "--config",
            str(saved_config),
            f"--ckpt_path={ckpt}",
            f"--output={out}",
            "--no-check",
            "--overwrite",
        ])
        assert rc == 0, f"salt export failed (rc={rc})"
        assert out.exists()
        return out

    def test_onnx_output_ranks_global_vs_per_token(self, exported):
        """2 rank-0 globals (mean + _stddev of the global head) + 2 rank-1 per-token."""
        model = onnx.load(str(exported))
        ranks = {o.name: len(o.type.tensor_type.shape.dim) for o in model.graph.output}
        assert sorted(ranks.values()) == [0, 0, 1, 1], ranks

    def test_onnx_session_runs(self, exported):
        """The exported graph runs in onnxruntime on batch-1 inputs (L=5 tokens)."""
        sess = make_session(exported)
        in_meta = {i.name: i.shape for i in sess.get_inputs()}
        rng = np.random.default_rng(0)

        def shape_for(dims):
            return tuple(5 if (isinstance(d, str) or d is None) else d for d in dims)

        feeds = {
            name: rng.standard_normal(shape_for(dims)).astype(np.float32)
            for name, dims in in_meta.items()
        }
        out = {o.name: v for o, v in zip(sess.get_outputs(), sess.run(None, feeds), strict=True)}
        assert len(out) == 4, f"expected the 4 gaussian outputs, got {sorted(out)}"
