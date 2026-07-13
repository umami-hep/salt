"""Small deterministic v1 GN2 builder for the forward-parity gate (plan 04).

DEL-1: the v1-free constants/helpers (JET_VARIABLES, TRACK_VARIABLES,
ELECTRON_VARIABLES, write_parity_norm_dict, make_gn2_batch) moved to
``gn2v2_fixture.py`` and are re-imported here so the remaining v1-only
consumers (parity_gn2, test_wrappers, ...) keep working until this file is
deleted with the rest of the v1 tree.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from salt.models import SaltModel, Transformer
from salt.models.pooling import GlobalAttentionPooling
from salt.models.task import ClassificationTask, VertexingTask
from salt.modelwrapper import ModelWrapper
from salt.tests._fixtures.gn2v2_fixture import (
    ELECTRON_VARIABLES,
    JET_VARIABLES,
    TRACK_VARIABLES,
    make_gn2_batch,
    write_parity_norm_dict,
)

__all__ = [
    "ELECTRON_VARIABLES",
    "JET_VARIABLES",
    "TRACK_VARIABLES",
    "build_test_gn2",
    "make_gn2_batch",
    "v1_forward",
    "write_parity_norm_dict",
]


def build_test_gn2(
    norm_dir: Path | str,
    embed_dim: int = 16,
    out_dim: int = 16,
    num_layers: int = 2,
    num_heads: int = 2,
    seed: int = 42,
    with_electrons: bool = False,
    variables: dict[str, list[str]] | None = None,
    norm_dict: Path | str | None = None,
) -> ModelWrapper:
    """Construct a small v1 GN2 `ModelWrapper` directly (no CLI), in eval mode."""
    norm_dir = Path(norm_dir)
    nd_path = norm_dir / "norm_dict.yaml"
    cd_path = norm_dir / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    if norm_dict is not None:
        nd_path = Path(norm_dict)

    if variables is None:
        variables = {"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}
        if with_electrons:
            variables["electrons"] = list(ELECTRON_VARIABLES)
    else:
        variables = {stream: list(names) for stream, names in variables.items()}

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


def v1_forward(
    wrapper: ModelWrapper,
    inputs: dict[str, Tensor],
    pad_masks: dict[str, Tensor],
) -> dict:
    """Run the v1 reference forward on CLONED dicts (inference, no labels)."""
    with torch.no_grad():
        preds, _loss = wrapper(
            {k: v.clone() for k, v in inputs.items()},
            {k: v.clone() for k, v in pad_masks.items()},
            None,
        )
    return preds
