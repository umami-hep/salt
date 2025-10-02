from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm

from salt.models.task import mask_fill_flattened
from salt.utils.get_structured_input_dict import get_structured_input_dict
from salt.utils.inputs import (
    inputs_sep_with_pad_multi_sequece,
)
from salt.utils.union_find import get_node_assignment_jit

torch.manual_seed(42)


def compare_output(
    pt_model: torch.nn.Module,
    onnx_session: ort.InferenceSession,
    global_object: str,
    seq_names_salt: Sequence[str],
    seq_names_onnx: Sequence[str],
    variable_map: Mapping[str, list[str]],
    tasks_to_output: Sequence[str],
    n_seq: int = 40,
) -> None:
    """Compare PyTorch vs ONNX outputs for a single synthetic batch/config."""
    # Generate a single batch of synthetic inputs (jets + constituent sequences)
    n_batch = 1
    jets, sequences, pad_masks = inputs_sep_with_pad_multi_sequece(
        n_batch,
        [n_seq for _ in seq_names_salt],
        pt_model.input_dims[global_object],
        [pt_model.input_dims[seqn] for seqn in seq_names_salt],
        p_valid=1,
    )

    # Build input dict for PyTorch
    inputs_pytorch = {seqn: seq for seq, seqn in zip(sequences, seq_names_salt, strict=False)}
    inputs_pytorch[global_object] = jets

    # Build padding mask dict for PyTorch
    masks_pytorch = {seqn: mask for mask, seqn in zip(pad_masks, seq_names_salt, strict=False)}

    # Convert inputs into structured form for tasks (e.g., label preparation)
    structured_input_dict = get_structured_input_dict(inputs_pytorch, variable_map, global_object)

    # Run forward pass of the PyTorch model
    outputs_pytorch = pt_model(inputs_pytorch, masks_pytorch)[0]

    # Collect predictions from PyTorch global tasks
    if global_object in outputs_pytorch:
        global_pred_pytorch = []
        global_tasks = [t for t in pt_model.model.tasks if t.input_name == global_object]

        for i, out in enumerate(list(outputs_pytorch[global_object].values())):
            if global_tasks[i].name not in tasks_to_output:
                continue
            # Convert PyTorch task output to ONNX-compatible representation
            onnx_out = global_tasks[i].get_onnx(out, labels=structured_input_dict)
            global_pred_pytorch += [p.detach().numpy() for p in onnx_out]
    else:
        global_pred_pytorch = []

    # Build ONNX input dict (names differ slightly from PyTorch)
    inputs_onnx = {f"{global_object.removesuffix('s')}_features": jets.numpy()}
    for seq, seqn in zip(sequences, seq_names_onnx, strict=False):
        inputs_onnx[seqn] = seq.squeeze(0).numpy()

    # Run ONNX model inference
    outputs_onnx = onnx_session.run(None, inputs_onnx)

    # test jet classification
    global_pred_onnx = outputs_onnx[: len(global_pred_pytorch)]
    assert not np.isnan(np.array(global_pred_pytorch)).any()
    assert not np.isnan(np.array(global_pred_onnx)).any()
    assert not (np.array(global_pred_pytorch) == 0).any()
    assert not (np.array(global_pred_onnx) == 0).any()
    np.testing.assert_allclose(
        global_pred_pytorch,
        global_pred_onnx,
        rtol=1e-04,
        atol=1e-04,
        err_msg="Torch vs ONNX check failed for global task",
    )

    # Stop here if no sequences were generated
    if n_seq == 0:
        return

    # Test track origin
    if "track_origin" in tasks_to_output:
        pred_pytorch_origin = (
            torch.argmax(outputs_pytorch["tracks"]["track_origin"], dim=-1).detach().numpy()
        )
        onnx_index = tasks_to_output.index("track_origin") - len(tasks_to_output)
        pred_onnx_origin = outputs_onnx[onnx_index]
        assert (
            len(pred_onnx_origin.shape) == 1
        ), "ONNX output for track origin should be a single tensor"
        np.testing.assert_allclose(
            pred_pytorch_origin.squeeze(),
            pred_onnx_origin,
            rtol=1e-06,
            atol=1e-06,
            err_msg="Torch vs ONNX check failed for track origin",
        )

    # Test track vertexing
    if "track_vertexing" in tasks_to_output:
        # Get vertex assignment from PyTorch output
        pred_pytorch_scores = outputs_pytorch["tracks"]["track_vertexing"].detach()
        pred_pytorch_indices = get_node_assignment_jit(pred_pytorch_scores, pad_masks[0])
        pred_pytorch_vtx = mask_fill_flattened(pred_pytorch_indices, pad_masks[0])

        onnx_index = tasks_to_output.index("track_vertexing") - len(tasks_to_output)
        pred_onnx_vtx = outputs_onnx[onnx_index]
        np.testing.assert_allclose(
            pred_pytorch_vtx.squeeze(),
            pred_onnx_vtx,
            rtol=1e-06,
            atol=1e-06,
            err_msg="Torch vs ONNX check failed for vertexing",
        )

    # Test track type
    if "track_type" in tasks_to_output:
        pred_pytorch_type = (
            torch.argmax(outputs_pytorch["tracks"]["track_type"], dim=-1).detach().numpy()
        )
        onnx_index = tasks_to_output.index("track_type") - len(tasks_to_output)
        pred_onnx_type = outputs_onnx[onnx_index]

        assert (
            len(pred_onnx_type.shape) == 1
        ), "ONNX output for track origin should be a single tensor"
        np.testing.assert_allclose(
            pred_pytorch_type.squeeze(),
            pred_onnx_type,
            rtol=1e-06,
            atol=1e-06,
            err_msg="Torch vs ONNX check failed for track type",
        )


def compare_outputs(
    pt_model: torch.nn.Module,
    onnx_path: str | Path,
    global_object: str,
    seq_names_salt: Sequence[str],
    seq_names_onnx: Sequence[str],
    variable_map: Mapping[str, list[str]],
    tasks_to_output: Sequence[str],
) -> None:
    """Validate that PyTorch and ONNX models match across many synthetic cases."""
    print("\n" + "-" * 100)
    print("Validating ONNX model...")

    # Create ONNX Runtime session with reduced logging
    sess_options = ort.SessionOptions()

    # suppress warnings due to unoptimized subgraphs - https://github.com/microsoft/onnxruntime/issues/14694
    sess_options.log_severity_level = 3
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"], sess_options=sess_options
    )

    # Loop over a range of sequence lengths (0-39) and multiple trials
    for n_track in tqdm(range(40), leave=False):
        for _ in range(10):
            compare_output(
                pt_model,
                session,
                global_object,
                seq_names_salt,
                seq_names_onnx,
                variable_map,
                tasks_to_output,
                n_track,
            )

    # Report final success message
    print(
        "Success! Pytorch and ONNX models are consistent, but you should verify this in"
        " Athena.\nFor more info see: https://ftag-salt.docs.cern.ch/export/#athena-validation"
    )
    print("-" * 100)
