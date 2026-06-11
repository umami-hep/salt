"""Small deterministic v1 GN2 builder for the forward-parity gate (plan 04).

Shared by the wrapper unit tests (stage 2) and the parity harness (stage 3).
Construction kwargs follow the stage-1 recipe exactly — every non-obvious
kwarg is load-bearing and commented with its v1 citation.

Determinism notes (recipe §3):

- Importing ``salt.utils.inputs`` seeds torch to 42 as an import side effect
  (inputs.py:134-135) — `build_test_gn2` therefore re-seeds explicitly AFTER
  imports, before parameter init.
- ``attn_type="torch-math"`` picks the deterministic SDPBackend.MATH kernel
  at CONSTRUCTION (attention.py:212-220); it matches v1 test/ONNX semantics
  (modelwrapper.py:331-335) and avoids the flash-varlen packing path even on
  a CUDA+flash machine.
- The batch is generated from a local ``torch.Generator`` (no global-RNG
  call-order dependence) with real padding (mask-polarity bugs must be
  excitable), one jet with exactly one valid track, and one jet with ZERO
  valid tracks (production edge case: pooling's all-padded path and the
  zero-edge vertexing contribution).
- The norm dict is written by `write_parity_norm_dict` with DISTINCT
  per-variable constants — salt's ``write_dummy_norm_dict`` (mean=1/std=1
  for every variable) made per-variable constant misordering invisible
  (stage-4 critic finding: rolled constants gave a false PASS).
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch import Tensor, nn

from salt.models import SaltModel, Transformer
from salt.models.pooling import GlobalAttentionPooling
from salt.models.task import ClassificationTask, VertexingTask
from salt.modelwrapper import ModelWrapper

__all__ = [
    "ELECTRON_VARIABLES",
    "JET_VARIABLES",
    "TRACK_VARIABLES",
    "build_test_gn2",
    "make_gn2_batch",
    "v1_forward",
    "write_parity_norm_dict",
]

JET_VARIABLES = ["pt_btagJes", "eta_btagJes"]  # GN2.yaml:90-92

TRACK_VARIABLES = [  # the 19 GN2 track variables, GN2.yaml:93-112
    "d0",
    "z0SinTheta",
    "dphi",
    "deta",
    "qOverP",
    "IP3D_signed_d0_significance",
    "IP3D_signed_z0_significance",
    "phiUncertainty",
    "thetaUncertainty",
    "qOverPUncertainty",
    "numberOfPixelHits",
    "numberOfSCTHits",
    "numberOfInnermostPixelLayerHits",
    "numberOfNextToInnermostPixelLayerHits",
    "numberOfInnermostPixelLayerSharedHits",
    "numberOfInnermostPixelLayerSplitHits",
    "numberOfPixelSharedHits",
    "numberOfPixelSplitHits",
    "numberOfSCTSharedHits",
]

ELECTRON_VARIABLES = [  # GN2e-style second sequence stream (subset of inputs.py ELECTRON_VARS)
    "pt",
    "ptfrac",
    "ptrel",
    "dr",
    "abs_eta",
]


def write_parity_norm_dict(nd_path: Path, cd_path: Path) -> None:
    """Write the parity norm/class dicts with DISTINCT per-variable constants.

    salt's ``write_dummy_norm_dict`` (inputs.py:348-357) gives EVERY variable
    ``mean=1.0, std=1.0`` — under uniform constants, rolling the means/stds
    across variables is the identity, so a per-variable field-order mismatch
    between the norm dict and the input columns passes the gate silently
    (stage-4 critic finding: demonstrated false PASS), and ``std=1.0`` hides
    any scale-wiring difference. Distinct constants make both excitable:

    - ``mean_i = 0.1 * (i + 1)`` — nonzero for every variable, so skipping
      the normaliser entirely stays caught (negative control 2);
    - ``std_i = 1.0 + 0.05 * (i + 1)`` — non-unit, so scale wiring is
      exercised, and bounded away from 0 (InputNorm rejects zero stds);

    indexed by the variable's position in the stream's variable list.

    Parameters
    ----------
    nd_path : Path
        Output path for the normalisation dictionary YAML.
    cd_path : Path
        Output path for the class dictionary YAML (same content as
        ``write_dummy_norm_dict``'s non-GN3 class dict).
    """
    sd = {
        stream: {
            v: {"mean": round(0.1 * (i + 1), 6), "std": round(1.0 + 0.05 * (i + 1), 6)}
            for i, v in enumerate(variables)
        }
        for stream, variables in (
            ("jets", JET_VARIABLES),
            ("tracks", TRACK_VARIABLES),
            ("electrons", ELECTRON_VARIABLES),
        )
    }
    with open(nd_path, "w") as file:
        yaml.dump(sd, file, sort_keys=False)

    cd = {
        "jets": {
            "HadronConeExclTruthLabelID": [1.0, 2.0, 2.0, 2.0],
            "flavour_label": [1.0, 2.0, 2.0, 2.0],
        },
        "tracks": {"ftagTruthOriginLabel": [4.2, 73.7, 1.0, 17.5, 12.3, 12.5, 141.7, 22.3]},
    }
    with open(cd_path, "w") as file:
        yaml.dump(cd, file, sort_keys=False)


def build_test_gn2(
    norm_dir: Path | str,
    embed_dim: int = 16,
    out_dim: int = 16,
    num_layers: int = 2,
    num_heads: int = 2,
    seed: int = 42,
    with_electrons: bool = False,
) -> ModelWrapper:
    """Construct a small v1 GN2 `ModelWrapper` directly (no CLI), in eval mode.

    Parameters
    ----------
    norm_dir : Path | str
        Writable directory for the parity norm/class dicts. Constants are
        DISTINCT per variable with nonzero means (`write_parity_norm_dict`),
        so missing-norm, constant-order, and scale wiring are all caught.
    embed_dim : int, optional
        Encoder embedding width, by default 16 (GN2 nominal: 256).
    out_dim : int, optional
        Encoder output width, by default 16 (GN2 nominal: 128).
    num_layers : int, optional
        Encoder layers, by default 2.
    num_heads : int, optional
        Attention heads, by default 2.
    seed : int, optional
        ``torch.manual_seed`` applied right before construction (parameter
        init), by default 42.
    with_electrons : bool, optional
        Add a second sequence stream (GN2e-style ``electrons`` init net,
        same recipe as ``tracks``), by default False. Two streams make the
        multi-stream Concat order / per-stream mask dict order genuinely
        excitable — with a single stream they are vacuous (stage-4 critic
        finding: a reversed single-stream Concat passes trivially).

    Returns
    -------
    ModelWrapper
        The constructed v1 model, switched to ``eval()``.
    """
    norm_dir = Path(norm_dir)
    nd_path = norm_dir / "norm_dict.yaml"
    cd_path = norm_dir / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)

    variables = {"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}
    if with_electrons:
        variables["electrons"] = list(ELECTRON_VARIABLES)

    # Seed AFTER imports: importing salt.utils.inputs already seeded torch
    # to 42 (inputs.py:134-135); this makes param init reproducible per call.
    torch.manual_seed(seed)

    init_nets = [
        {
            "input_name": "tracks",
            "dense_config": {
                # input_size inferred: 19 track + 2 jet vars (initnet.py:54-59)
                "output_size": embed_dim,
                "hidden_layers": [embed_dim],
                "activation": "ReLU",
            },
            # variables + global_object are normally CLI-injected
            # (cli.py:227-229) — direct construction must supply them.
            "variables": variables,
            "global_object": "jets",
        }
    ]
    if with_electrons:
        # Second sequence stream AFTER tracks: init_nets order defines the
        # v1 concat order (saltmodel.py:128-129, transformer.py:684-686).
        init_nets.append({
            "input_name": "electrons",
            "dense_config": {
                # input_size inferred: 5 electron + 2 jet vars (initnet.py:54-59)
                "output_size": embed_dim,
                "hidden_layers": [embed_dim],
                "activation": "ReLU",
            },
            "variables": variables,
            "global_object": "jets",
        })
    encoder = Transformer(
        num_layers=num_layers,
        embed_dim=embed_dim,
        out_dim=out_dim,
        norm="LayerNorm",
        # torch-math: deterministic CPU/GPU kernel, matches v1 test/ONNX
        # semantics (modelwrapper.py:331-335); no flash fallback ambiguity.
        attn_type="torch-math",
        do_final_norm=True,
        dense_kwargs={"activation": "ReLU", "gated": False},
        # attn_kwargs is REQUIRED: transformer.py:600-601 writes into it.
        attn_kwargs={"num_heads": num_heads},
        # num_registers defaults to 1 (transformer.py:558-559) — keep it.
    )
    tasks = nn.ModuleList([
        ClassificationTask(
            name="jets_classification",
            input_name="jets",
            label="flavour_label",
            # class_names REQUIRED: the CLASS_NAMES fallback has no
            # "flavour_label" key (task.py:130-131).
            class_names=["bjets", "cjets", "ujets"],
            loss=nn.CrossEntropyLoss(),
            dense_config={
                "input_size": out_dim,
                "output_size": 3,
                "hidden_layers": [out_dim],
                "activation": "ReLU",
            },
        ),
        ClassificationTask(
            name="track_origin",
            input_name="tracks",
            label="ftagTruthOriginLabel",
            # class_names=None is OK here: CLASS_NAMES["ftagTruthOriginLabel"]
            # has exactly 8 entries.
            weight=0.5,
            loss=nn.CrossEntropyLoss(),
            dense_config={
                "input_size": out_dim,
                "output_size": 8,
                "hidden_layers": [out_dim],
                "activation": "ReLU",
                "context_size": out_dim,
            },
        ),
        VertexingTask(
            name="track_vertexing",
            input_name="tracks",
            label="ftagTruthVertexIndex",
            weight=1.5,
            loss=nn.BCEWithLogitsLoss(reduction="none"),
            dense_config={
                "input_size": 2 * out_dim,  # pair concat (task.py:894-896)
                "output_size": 1,
                "hidden_layers": [out_dim],
                "activation": "ReLU",
                "context_size": out_dim,  # context width is out_dim (task.py:884-891)
            },
        ),
    ])
    model = SaltModel(
        init_nets=init_nets,
        tasks=tasks,
        encoder=encoder,
        pool_net=GlobalAttentionPooling(input_size=out_dim),
    )
    # ModelWrapper.__init__ is LOAD-BEARING: it monkey-patches
    # task.global_object / task.model_name onto every task
    # (modelwrapper.py:111-114) — run_tasks routing reads them
    # (saltmodel.py:216).
    wrapper = ModelWrapper(
        model=model,
        lrs_config={"initial": 1e-7, "max": 5e-4, "end": 1e-5, "pct_start": 0.01},
        global_object="jets",
        norm_config={
            "norm_dict": str(nd_path),
            "variables": variables,
            "global_object": "jets",
            "input_map": {k: k for k in variables},
        },
    )
    wrapper.eval()
    return wrapper


def make_gn2_batch(
    batch_size: int = 6,
    n_tracks: int = 10,
    p_valid: float = 0.6,
    seed: int = 123,
    n_electrons: int = 0,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Build a deterministic GN2 batch with real padding.

    Mirrors the v1 dataset contract (datasets.py:435-441, 448-559):
    ``inputs = {"jets": [B, 2] f32, "tracks": [B, T, 19] f32}``,
    ``pad_masks = {"tracks": [B, T] bool}`` with True = padded
    (datasets.py:523), padded input positions ZEROED (datasets.py:524),
    valid tracks first, jet 0 with EXACTLY one valid track (recipe edge
    case), and — for ``batch_size >= 2`` — jet 1 with ZERO valid tracks
    (production edge case: pooling's all-padded ONNX branch, pooling.py:59-63,
    and a zero-edge per-jet vertexing contribution; verified bitwise-safe by
    the stage-4 critic probe).

    With ``n_electrons > 0`` a second sequence stream is added AFTER tracks
    (matching the two-stream fixture's init_nets order — the batch dict
    order must match it, since v1's encoder concatenates dict values in
    insertion order, transformer.py:684-686): ``inputs["electrons"]``
    ``[B, T_e, 5]`` and ``pad_masks["electrons"]``, with jet 2 (when present)
    forced to ZERO electrons. Electron tensors are drawn AFTER all track
    draws, so the tracks/jets content is unchanged vs ``n_electrons=0``.

    Parameters
    ----------
    batch_size : int, optional
        Number of jets, by default 6.
    n_tracks : int, optional
        Track positions per jet, by default 10.
    p_valid : float, optional
        Probability a position is valid, by default 0.6 — so padded
        positions exist and mask-polarity bugs are excitable.
    seed : int, optional
        Seed for the local generator, by default 123.
    n_electrons : int, optional
        Electron positions per jet (0 disables the stream), by default 0.

    Returns
    -------
    tuple[dict[str, Tensor], dict[str, Tensor]]
        ``(inputs, pad_masks)``. Callers must CLONE both dicts (and never
        share them between two forwards): v1's InputNorm rebinds the inputs
        dict's keys (inputnorm.py:103-106) and the encoder ADDS a
        ``"REGISTERS"`` key to the pad-mask dict (transformer.py:777,785).
    """
    gen = torch.Generator().manual_seed(seed)
    jets = torch.randn(batch_size, len(JET_VARIABLES), generator=gen)
    tracks = torch.randn(batch_size, n_tracks, len(TRACK_VARIABLES), generator=gen)
    mask = torch.rand(batch_size, n_tracks, generator=gen) >= p_valid  # True = padded
    mask = torch.sort(mask.to(torch.uint8), dim=-1).values.bool()  # valid first, like v1 dumps
    mask[:, 0] = False  # >=1 valid track per jet by default (inputs.py:296-297 analogue)
    mask[0, 1:] = True  # jet 0: exactly one valid track
    if batch_size >= 2:
        mask[1, :] = True  # jet 1: ZERO valid tracks (production edge case)
    tracks[mask] = 0.0  # padded positions zeroed (datasets.py:524)
    inputs = {"jets": jets, "tracks": tracks}
    pad_masks = {"tracks": mask}
    if n_electrons > 0:
        electrons = torch.randn(batch_size, n_electrons, len(ELECTRON_VARIABLES), generator=gen)
        emask = torch.rand(batch_size, n_electrons, generator=gen) >= p_valid  # True = padded
        emask = torch.sort(emask.to(torch.uint8), dim=-1).values.bool()  # valid first
        if batch_size >= 3:
            emask[2, :] = True  # jet 2: ZERO electrons (typical for real jets)
        electrons[emask] = 0.0  # padded positions zeroed (datasets.py:524)
        inputs["electrons"] = electrons
        pad_masks["electrons"] = emask
    return inputs, pad_masks


def v1_forward(
    wrapper: ModelWrapper,
    inputs: dict[str, Tensor],
    pad_masks: dict[str, Tensor],
) -> dict:
    """Run the v1 reference forward on CLONED dicts (inference, no labels).

    Equivalent to v1's test_step path (modelwrapper.py:318-338) for a
    torch-math model: ``wrapper(inputs, pad_masks, None)`` under
    ``no_grad``. Fresh dicts with cloned tensors per call — see
    `make_gn2_batch` for why sharing dicts between forwards is wrong.

    Returns
    -------
    dict
        The v1 preds dict: ``embed_xs``, ``global_rep``, and the nested
        ``{stream: {task: raw_output}}`` leaves.
    """
    with torch.no_grad():
        preds, _loss = wrapper(
            {k: v.clone() for k, v in inputs.items()},
            {k: v.clone() for k, v in pad_masks.items()},
            None,
        )
    return preds
