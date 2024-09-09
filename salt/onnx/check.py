import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm

from salt.models.task import mask_fill_flattened
from salt.utils.inputs import (
    inputs_sep_with_pad_multi_sequece,
)
from salt.utils.union_find import get_node_assignment_jit

torch.manual_seed(42)


def compare_output(
    pt_model,
    onnx_session,
    include_aux,
    seq_names_salt,
    seq_names_onnx,
    n_seq=40,
):
    n_batch = 1

    jets, sequences, pad_masks = inputs_sep_with_pad_multi_sequece(
        n_batch,
        [n_seq for seqn in seq_names_salt],
        pt_model.input_dims["jets"],
        [pt_model.input_dims[seqn] for seqn in seq_names_salt],
        p_valid=1,
    )

    inputs_pytorch = {seqn: seq for seq, seqn in zip(sequences, seq_names_salt, strict=False)}
    inputs_pytorch["jets"] = jets

    masks_pytorch = {seqn: mask for mask, seqn in zip(pad_masks, seq_names_salt, strict=False)}

    outputs_pytorch = pt_model(inputs_pytorch, masks_pytorch)[0]
    if "jets" in outputs_pytorch:
        out = list(outputs_pytorch["jets"].values())[0]
        out = pt_model.model.tasks[0].get_onnx(out)
        global_pred_pytorch = [p.detach().numpy() for p in out]
    else:
        global_pred_pytorch = []

    inputs_onnx = {"jet_features": jets.numpy()}
    for seq, seqn in zip(sequences, seq_names_onnx, strict=False):
        inputs_onnx[seqn] = seq.squeeze(0).numpy()

    outputs_onnx = onnx_session.run(None, inputs_onnx)

    # test jet classification
    pred_onnx_jc = outputs_onnx[: len(global_pred_pytorch)]

    np.testing.assert_allclose(
        global_pred_pytorch,
        pred_onnx_jc,
        rtol=1e-04,
        atol=1e-04,
        err_msg="Torch vs ONNX check failed for global task",
    )

    assert not np.isnan(np.array(pred_onnx_jc)).any()  # non nans
    assert not (np.array(pred_onnx_jc) == 0).any()  # no trivial zeros

    # test track origin
    if include_aux:
        if n_seq == 0:
            return

        pred_pytorch_origin = (
            torch.argmax(outputs_pytorch["tracks"]["track_origin"], dim=-1).detach().numpy()
        )
        pred_onnx_origin = outputs_onnx[
            len(global_pred_pytorch) : len(global_pred_pytorch) + len(pred_pytorch_origin)
        ][0]

        np.testing.assert_allclose(
            pred_pytorch_origin.squeeze(),
            pred_onnx_origin,
            rtol=1e-06,
            atol=1e-06,
            err_msg="Torch vs ONNX check failed for track origin",
        )

    # test vertexing
    if include_aux and "track_vertexing" in outputs_pytorch["tracks"]:
        pred_pytorch_scores = outputs_pytorch["tracks"]["track_vertexing"].detach()
        pred_pytorch_indices = get_node_assignment_jit(pred_pytorch_scores, pad_masks[0])
        pred_pytorch_vtx = mask_fill_flattened(pred_pytorch_indices, pad_masks[0])

        pred_onnx_vtx = outputs_onnx[-1]
        np.testing.assert_allclose(
            pred_pytorch_vtx.squeeze(),
            pred_onnx_vtx,
            rtol=1e-06,
            atol=1e-06,
            err_msg="Torch vs ONNX check failed for vertexing",
        )


def compare_outputs(pt_model, onnx_path, include_aux, seq_names_salt, seq_names_onnx):
    print("\n" + "-" * 100)
    print("Validating ONNX model...")

    sess_options = ort.SessionOptions()
    # suppress warnings due to unoptimized subgraphs - https://github.com/microsoft/onnxruntime/issues/14694
    sess_options.log_severity_level = 3
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"], sess_options=sess_options
    )
    for n_track in tqdm(range(40), leave=False):
        for _ in range(10):
            compare_output(
                pt_model,
                session,
                include_aux,
                seq_names_salt,
                seq_names_onnx,
                n_track,
            )

    print(
        "Success! Pytorch and ONNX models are consistent, but you should verify this in"
        " Athena.\nFor more info see: https://ftag-salt.docs.cern.ch/export/#athena-validation"
    )
    print("-" * 100)
