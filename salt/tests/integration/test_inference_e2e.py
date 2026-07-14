"""Plan 50 Phase D gate — ``salt2 inference`` end-to-end on the dummy fixture.

The three-part gate (plan 50 §Phase D):

(a) on a labelled file, the inference H5 columns correspond ONE-TO-ONE to the
    export-mode `OutputField` selection and the ONNX tuple order/names
    (anchored on the committed ``gn2v2-dummy-cutover34`` golden);
(b) the H5 values match the ONNXRuntime outputs of the exported model on the
    same per-jet inputs (the ``check_onnx`` comparison style + tolerance);
(c) on a label-stripped copy of the same file the command runs green and
    produces identical prediction columns.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from numpy.lib.recfunctions import repack_fields

from salt.core.main import CONFIG_DIR, main
from salt.core.schema import dump_schema, save_schema
from salt.core.testing.inputs import write_dummy_file
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict

DUMMY_CFG = CONFIG_DIR / "gn2v2-dummy.yaml"
CUTOVER34_CFG = CONFIG_DIR / "gn2v2-dummy-cutover34.yaml"
GOLDEN = Path(__file__).resolve().parents[1] / "_fixtures/output_goldens/gn2v2-dummy-cutover34.json"

RUN_NAME = "GN2v2_dummy"  # the config `name:` — the H5 column prefix
MODEL_NAME = "GN2v2dummy"  # export.model_name — the ONNX tuple prefix
N_TEST = 96

# label columns physically removed for the stripped copy (the
# test_label_stripped_read.py field set)
LABEL_FIELDS = {
    "flavour_label",
    "HadronConeExclTruthLabelID",
    "HadronGhostInitialTruthLabelPdgId",
    "ftagTruthOriginLabel",
    "ftagTruthTypeLabel",
    "ftagTruthVertexIndex",
    "ftagTruthParentBarcode",
}

pytestmark = pytest.mark.cpu_always


def _golden_onnx() -> dict:
    """The committed export-selection contract for the cutover34 stack."""  # noqa: DOC201
    return json.loads(GOLDEN.read_text())["onnx"]


def _strip_labels(src: Path, dst: Path) -> None:
    """Copy ``src`` dropping every LABEL_FIELDS column from the structured datasets."""
    with h5py.File(src) as fin, h5py.File(dst, "w") as fout:
        for name, ds in fin.items():
            arr = ds[:]
            keep = [f for f in arr.dtype.names if f not in LABEL_FIELDS]
            out = fout.create_dataset(name, data=repack_fields(arr[keep]))
            for k, v in ds.attrs.items():
                if k not in LABEL_FIELDS:
                    out.attrs[k] = v


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    """Labelled dummy file + label-stripped copy + per-file schema artifacts."""
    base = tmp_path_factory.mktemp("inference_e2e")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    labelled = base / "pp_output_test_ttbar.h5"
    write_dummy_file(labelled, nd_path)
    stripped = base / "pp_output_test_stripped.h5"
    _strip_labels(labelled, stripped)
    schemas: dict[str, Path] = {}
    for key, path in (("labelled", labelled), ("stripped", stripped)):
        schemas[key] = base / f"{key}_schema.yaml"
        save_schema(dump_schema(path), schemas[key])
    return {
        "dir": base,
        "labelled": labelled,
        "stripped": stripped,
        "schemas": schemas,
        "nd": nd_path,
    }


def _fit_overrides(data) -> list[str]:
    return [
        f"--data.modules.reader.init_args.schema={data['schemas']['labelled']}",
        f"--model.modules.norm.init_args.norm_dict={data['nd']}",
        "--trainer.accelerator=cpu",
        "--trainer.logger=false",
        "--callbacks.progress=null",
    ]


@pytest.fixture(scope="module")
def ckpt(data, tmp_path_factory) -> Path:
    """A 1-epoch checkpoint on the base dummy config (the test_outputs_section recipe)."""
    fit_dir = tmp_path_factory.mktemp("inference_fit")
    rc = main([
        "fit",
        "--config",
        str(DUMMY_CFG),
        f"--data.train_file={data['labelled']}",
        f"--data.val_file={data['labelled']}",
        *_fit_overrides(data),
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


def _inference_args(data, ckpt: Path, file_key: str, out: Path) -> list[str]:
    """The ``salt2 inference`` argv for one input file (cutover34 stack)."""  # noqa: DOC201
    return [
        "inference",
        "-c",
        str(DUMMY_CFG),
        "-c",
        str(CUTOVER34_CFG),
        f"--ckpt_path={ckpt}",
        f"--data.test_file={data[file_key]}",
        f"--output={out}",
        "--set",
        f"data.modules.reader.init_args.schema={data['schemas'][file_key]}",
        "--set",
        f"model.modules.norm.init_args.norm_dict={data['nd']}",
        "--set",
        f"data.num_test={N_TEST}",
    ]


@pytest.fixture(scope="module")
def inference_h5(data, ckpt) -> Path:
    """The inference H5 on the LABELLED file."""
    out = data["dir"] / "inference_labelled.h5"
    rc = main(_inference_args(data, ckpt, "labelled", out))
    assert rc == 0, "salt2 inference must run green on the labelled file"
    assert out.is_file(), "salt2 inference wrote no H5"
    return out


@pytest.fixture(scope="module")
def inference_h5_stripped(data, ckpt) -> Path:
    """The inference H5 on the LABEL-STRIPPED copy (gate c: runs green)."""
    out = data["dir"] / "inference_stripped.h5"
    rc = main(_inference_args(data, ckpt, "stripped", out))
    assert rc == 0, "salt2 inference must run green on a label-stripped file"
    assert out.is_file()
    return out


class TestExportSelectionColumns:
    """Gate (a): H5 columns == the export-mode OutputField selection, 1:1 with the ONNX tuple."""

    def test_task_columns_are_the_onnx_tuple(self, inference_h5):
        """Every golden ONNX output has exactly one H5 column (same suffix, right
        stream/group, per-group order == tuple order), and no other prefixed
        task column exists.
        """
        golden = _golden_onnx()
        manifest = golden["task_manifest_onnx"]
        assert [f"{MODEL_NAME}_{m['resolved_onnx_name']}" for m in manifest if m["axis"] == "global"] + [
            f"{MODEL_NAME}_{m['resolved_onnx_name']}" for m in manifest if m["axis"] == "per_token"
        ] == golden["output_names"], "golden self-consistency (globals-then-per-token tuple)"
        expected: dict[str, list[str]] = {}
        for entry in manifest:
            stream = entry["leaf_key"].split(".")[1]
            name = f"{RUN_NAME}_{entry['resolved_onnx_name']}" if entry["prefix"] else entry["resolved_onnx_name"]
            expected.setdefault(stream, []).append(name)
        with h5py.File(inference_h5) as f:
            groups = {g: list(f[g].dtype.names) for g in f}
        for stream, cols in expected.items():
            present = [c for c in groups[stream] if c in set(cols)]
            assert present == cols, (
                f"{stream}: export-selection columns {present} != golden tuple order {cols}"
            )
            # 1:1 — no OTHER run-prefixed task column may exist (the export set
            # is STRICT; e.g. no per-class track_origin probs, no target_*)
            extras = [
                c
                for c in groups[stream]
                if (c.startswith(f"{RUN_NAME}_") or c.startswith("target_")) and c not in set(cols)
            ]
            assert not extras, f"{stream}: columns beyond the export selection: {extras}"

    def test_dtypes_match_onnx_tuple_dtypes(self, inference_h5):
        """f4 columns for float32 tuple entries, integer for int8."""
        golden = _golden_onnx()
        with h5py.File(inference_h5) as f:
            dtypes = {g: f[g].dtype for g in f}
        for entry in golden["task_manifest_onnx"]:
            stream = entry["leaf_key"].split(".")[1]
            col = f"{RUN_NAME}_{entry['resolved_onnx_name']}" if entry["prefix"] else entry["resolved_onnx_name"]
            if entry["onnx_dtype"] == "int8":
                assert np.issubdtype(dtypes[stream][col], np.integer), col
            else:
                assert np.issubdtype(dtypes[stream][col], np.floating), col

    def test_mode_gated_copy_and_mask_columns(self, inference_h5, data):
        """The default-modes InputCopyWriter/PadMaskWriter run under inference
        (they declare export implicitly): source copies precede the task
        columns and the tracks 'mask' column is last.
        """
        with h5py.File(data["labelled"]) as src:
            src_jets = list(src["jets"].dtype.names)
        with h5py.File(inference_h5) as f:
            jets = list(f["jets"].dtype.names)
            tracks = list(f["tracks"].dtype.names)
        assert jets[: len(src_jets)] == src_jets, "input copies must lead the jets group"
        assert tracks[-1] == "mask", "the pad-mask column must be last in tracks"


class TestValuesMatchOnnxRuntime:
    """Gate (b): H5 values == ONNXRuntime outputs on the same per-jet inputs."""

    @pytest.fixture(scope="class")
    def onnx_path(self, data, ckpt, tmp_path_factory) -> Path:
        """Export the SAME config stack + checkpoint to ONNX (checker skipped —
        this class IS the comparison).
        """
        from salt.core.onnx.export import main as export_main

        out = tmp_path_factory.mktemp("inference_onnx") / "network.onnx"
        rc = export_main([
            f"--ckpt_path={ckpt}",
            "-c",
            str(DUMMY_CFG),
            "-c",
            str(CUTOVER34_CFG),
            f"--output={out}",
            "--no-check",
            "--set",
            f"data.modules.reader.init_args.schema={data['schemas']['labelled']}",
            "--set",
            f"model.modules.norm.init_args.norm_dict={data['nd']}",
        ])
        assert rc == 0, "salt2 export must succeed on the inference config stack"
        return out

    def test_h5_values_equal_onnxruntime(self, data, inference_h5, onnx_path):
        """Per jet: run the exported ONNX on the file's valid tokens (Athena
        convention) and compare against the H5 — floats at the check_onnx
        tolerance (1e-4), int8 exact; padded H5 positions read 0.
        """
        from salt.core.onnx.check import make_session

        cfg = yaml.safe_load(DUMMY_CFG.read_text())
        variables = cfg["data"]["modules"]["features"]["init_args"]["variables"]
        with h5py.File(data["labelled"]) as f:
            jets_src = f["jets"][:N_TEST]
            tracks_src = f["tracks"][:N_TEST]
        jet_feats = np.stack([jets_src[v] for v in variables["jets"]], -1).astype(np.float32)
        trk_feats = np.stack([tracks_src[v] for v in variables["tracks"]], -1).astype(np.float32)
        valid = tracks_src["valid"].astype(bool)
        # fixture sanity: valid tokens are LEADING (the reader/pad layout the
        # H5 per-token placement relies on)
        assert (np.sort(valid, axis=-1)[:, ::-1] == valid).all()
        with h5py.File(inference_h5) as f:
            jets_out = f["jets"][:N_TEST]
            tracks_out = f["tracks"][:N_TEST]
        golden = _golden_onnx()
        session = make_session(onnx_path)
        ort_names = [o.name for o in session.get_outputs()]
        assert ort_names == golden["output_names"], "exported tuple != golden tuple"
        n_mismatch_checked = 0
        for i in range(N_TEST):
            ort_out = dict(
                zip(
                    ort_names,
                    session.run(
                        None,
                        {
                            "jet_features": jet_feats[i : i + 1],
                            "track_features": trk_feats[i][valid[i]],
                        },
                    ),
                    strict=True,
                )
            )
            for entry in golden["task_manifest_onnx"]:
                stream = entry["leaf_key"].split(".")[1]
                col = (
                    f"{RUN_NAME}_{entry['resolved_onnx_name']}"
                    if entry["prefix"]
                    else entry["resolved_onnx_name"]
                )
                ref = ort_out[f"{MODEL_NAME}_{entry['resolved_onnx_name']}"]
                if entry["axis"] == "global":
                    got = np.float64(jets_out[col][i]) if stream == "jets" else None
                    assert got is not None, f"global export leaf on unexpected stream {stream}"
                    np.testing.assert_allclose(
                        got, np.ravel(ref)[0], rtol=1e-4, atol=1e-4, err_msg=f"{col} jet {i}"
                    )
                else:
                    n_valid = int(valid[i].sum())
                    got_tokens = tracks_out[col][i]
                    if entry["onnx_dtype"] == "int8":
                        np.testing.assert_array_equal(
                            got_tokens[:n_valid], ref, err_msg=f"{col} jet {i}"
                        )
                    else:
                        np.testing.assert_allclose(
                            got_tokens[:n_valid], ref, rtol=1e-4, atol=1e-4,
                            err_msg=f"{col} jet {i}",
                        )
                    assert (got_tokens[n_valid:] == 0).all(), f"{col} jet {i}: pad not zero"
                    n_mismatch_checked += 1
        assert n_mismatch_checked > 0, "no per-token comparison ran (degenerate fixture)"


class TestLabelStripped:
    """Gate (c): label-stripped copy runs green with identical prediction columns."""

    def test_stripped_file_really_lacks_label_fields(self, data):
        """Fixture sanity: no LABEL_FIELDS column survives in the stripped copy."""
        with h5py.File(data["stripped"]) as f:
            for name in f:
                assert not set(f[name].dtype.names) & LABEL_FIELDS, name

    def test_prediction_columns_identical(self, inference_h5, inference_h5_stripped):
        """Every export-selection column (and the pad mask) is bit-identical
        between the labelled and stripped runs — labels contribute nothing.
        """
        golden = _golden_onnx()
        with h5py.File(inference_h5) as fa, h5py.File(inference_h5_stripped) as fb:
            for entry in golden["task_manifest_onnx"]:
                stream = entry["leaf_key"].split(".")[1]
                col = (
                    f"{RUN_NAME}_{entry['resolved_onnx_name']}"
                    if entry["prefix"]
                    else entry["resolved_onnx_name"]
                )
                a, b = fa[stream][col][:], fb[stream][col][:]
                assert np.array_equal(a, b), f"{stream}/{col} differs on the stripped file"
            assert np.array_equal(fa["tracks"]["mask"][:], fb["tracks"]["mask"][:])

    def test_stripped_h5_carries_no_label_columns(self, inference_h5_stripped):
        """The stripped-run H5 carries no label copy and no target_* column."""
        with h5py.File(inference_h5_stripped) as f:
            for group in f:
                names = set(f[group].dtype.names)
                assert not names & LABEL_FIELDS, f"label columns leaked into {group}"
                assert not {n for n in names if n.startswith("target_")}, group
