"""M4 gates harness — O1-O5 for the ONNX-export milestone (plan 07, stage C; design §9.5 M4).

Five standalone gates, each a subcommand of ``python -m salt.tests.integration.gates_m4``,
each writing a machine-readable ``<gate>_report.json`` into ``--outdir`` and
a human-readable table to stdout, exiting non-zero on failure. NO machine
data path appears anywhere: the export surface needs no data (design §7.1 —
boundary specs derive from config), so every gate builds its fixtures (the
plan-04/05 weight-matched GN2 pair) inside ``--outdir`` from the in-repo
generators.

Gate criteria (each derived/justified in its ``run_o*`` docstring):

- **O1 torch-vs-ONNX sweep** — the v2 export of the weight-matched GN2
  fixture agrees with the eager v2 adapter (torch-math, CPU) over the full
  v1 sweep L=0..39 x trials INCLUDING L=0, at the 1e-6 gate bar (v1 ships
  1e-4 on global floats, ``check.py:81-92``); int8 aux outputs exact.
- **O2 v1-vs-v2 export identity** — the v1 in-process export path
  (`ONNXModel` + ``add_metadata``, ``to_onnx.py``) and ``salt2 export``'s
  programmatic core, run on the SAME weights, produce ONNX models with
  identical input/output names, graph dims/dynamic axes, Athena-visible
  metadata, and IDENTICAL outputs under onnxruntime on identical inputs
  over the sweep (bitwise target; non-bitwise floats justified <= 1e-6,
  printed, never silent — the W1 discipline).
- **O3 two dynamic axes** — a two-stream (tracks + electrons) model with a
  per-electron output exports and agrees with torch over an
  (L_trk, L_el) grid including zero on EACH axis and (0, 0) — the design
  risk-7 release blocker; the gate also cross-checks against the v1
  two-stream wrapper's eager forward and records the adjudicated Split
  mechanism (`Concat` ``seq.offsets`` -> `Split` ``index_select``) from the
  compiled plan + the class docstring evidence.
- **O4 vertexing in-graph union-find** — triple identity over the sweep:
  v1-ONNX VertexIndex == v2-ONNX VertexIndex == the v1 export wrapper's
  eager torch chain (``get_node_assignment_jit`` + ``mask_fill_flattened``,
  ``to_onnx.py:426-432``), int8-exact, with a non-degeneracy bar (the sweep
  must produce multi-vertex assignments, or all-zero outputs would match
  vacuously).
- **O5 negative controls + metadata** — the machinery must FAIL when it
  should: an (asymmetric) weight perturbation after export fails the
  checker and the restored weights pass again; ``export.model_name`` with
  ``_``/``-`` is rejected by a named `ConfigError` before any file is
  written; the ``gnn_config`` envelope byte-reproduces v1 (13-key ordered
  prefix + single additive trailing ``plan_hash``; Athena-visible subset
  byte-equal; ``salt_export_hash`` is the only justified per-key exception
  — environment-dependent, the v1 exporter would CRASH where v2 degrades).

Negative-control hooks (the pytest suite, ``test_gates_m4.py``): `run_o2`
and `run_o4` take a python-only ``corruption`` keyword (applied to the v2
ONNX outputs before comparison) so the tests can prove the identity
comparisons fail on a real difference — the gates_m2/m3 pattern, never
exposed on the CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from salt.core.graph.errors import ConfigError
from salt.core.nn import (
    Concat,
    GlobalAttentionPooling,
    Normaliser,
    Split,
    StreamEmbed,
    TransformerEncoder,
    bind_all,
    map_v1_state_dict,
    resolve_bind_schema,
)
from salt.core.nn.tasks import ClassificationTaskModule, VertexingTaskModule
from salt.core.onnx import (
    ExportConfig,
    ExportInput,
    ExportOutput,
    attach_manifest,
    check_onnx,
    compile_onnx_plan,
    export_graph,
    make_session,
    resolve_export_config,
)
from salt.core.writers import TaskWriter, WriterDeclareCtx
from salt.models.task import mask_fill_flattened
from salt.models.transformer import change_attn_backends
from salt.onnx.to_onnx import ONNXModel, add_metadata, get_default_onnx_feature_map
from salt.tests._fixtures.gn2_fixture import (
    ELECTRON_VARIABLES,
    JET_VARIABLES,
    TRACK_VARIABLES,
    build_test_gn2,
)
from salt.tests._fixtures.gn2v2_fixture import ORIGIN_CLASSES, build_gn2v2_modules
from salt.utils.union_find import get_node_assignment_jit

__all__ = [
    "ATHENA_SUBSET_KEYS",
    "FLOAT_ATOL",
    "MODEL_NAME",
    "RUN_NAME",
    "V1_GNN_KEYS",
    "gn2_export_config",
    "main",
    "run_o1",
    "run_o2",
    "run_o3",
    "run_o4",
    "run_o5",
    "two_stream_export_config",
    "writer_manifest",
]

FLOAT_ATOL = 1e-6
"""The gate-level float bar for torch-vs-ONNX and v1-vs-v2 comparisons.

v1's shipped checker uses ``rtol=atol=1e-4`` on global floats and 1e-6 on
aux floats (``check.py:81-92,103-150``); the gates tighten the float bar to
1e-6 everywhere — the M4 recipe spike measured 5.96e-8 worst-case over the
full L=0..39 sweep on the GN2 fixture, so 1e-6 is >16x headroom while still
being far below any physics-relevant difference. int8 outputs are NEVER
tolerant (exact ports of v1's argmax/union-find chains).
"""

O2_JUSTIFICATION = (
    "the two ONNX graphs are traced from DIFFERENT module decompositions of the same math "
    "(v1 boolean-index stream gather + in-encoder registers vs v2 Concat/Split index_select, "
    "identical weights): op order/fusion differences make bitwise non-contractual; bounded by "
    "the parity_gn2 bitwise EAGER parity + the <=6e-8 trace drift each side (gate O1)"
)
"""Printed justification for any non-bitwise float in the O2/O4 identity sweeps."""

HASH_JUSTIFICATION = (
    "salt_export_hash is environment-dependent (ftag get_git_hash on the salt tree): the v1 "
    "exporter CRASHES when git state is unreadable (e.g. a container binding only the "
    "worktree) where v2 degrades to None with a warning — the gate patches the v1 path when "
    "needed (reported as v1_git_hash_patched) and does not require hash equality (stage-A "
    "adjudication, plan 07)"
)
"""The single justified per-key ``gnn_config`` difference (gate O5)."""

MODEL_NAME = "GN2v2"
"""The Athena-facing export name used by every gate (no ``_``/``-``)."""

RUN_NAME = "GN2_v2"
"""The run ``name:`` — deliberately carries an underscore so the default
``model_name`` derivation (strip ``_``/``-``, ``to_onnx.py:687``) and the
"run names are unrestricted" rule (design §7 "Naming") are exercised."""

V1_GNN_KEYS = (
    "ckpt_path",
    "layers",
    "nodes",
    "config.yaml",
    "metadata.yaml",
    "salt_export_hash",
    "onnx_model_version",
    "output_names",
    "model_name",
    "inputs",
    "input_sequences",
    "combine_outputs",
    "rename_outputs",
)
"""The v1 ``gnn_config`` top-level key ORDER (``to_onnx.py:806-846``, verified live)."""

ATHENA_SUBSET_KEYS = (
    "onnx_model_version",
    "output_names",
    "model_name",
    "inputs",
    "input_sequences",
    "combine_outputs",
    "rename_outputs",
)
"""The Athena-parsed ``gnn_config`` subset (design §7.5): the keys the
FlavorTagInference loader consumes — byte-reproducible regardless of the
legitimately-differing bookkeeping keys (paths, embedded configs, hashes)."""

_LRS = {"initial": 1e-7, "max": 1e-3, "end": 1e-5, "pct_start": 0.01}


# ---------------------------------------------------------------------------
# shared report helpers (the gates_m2/m3 envelope, kept standalone per harness)
# ---------------------------------------------------------------------------


def _emit_report(report: dict[str, Any], outdir: Path, gate: str) -> Path:
    """Write the gate report JSON into ``outdir`` and return its path.

    Returns
    -------
    Path
        The written ``<gate>_report.json`` path.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{gate}_report.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return path


def _print_verdict(gate: str, passed: bool, criterion: str, report_path: Path) -> None:
    """Print the closing verdict block every gate ends with."""
    print("=" * 96)
    print(f"GATE {gate.upper()}: {'PASS' if passed else 'FAIL'} — criterion: {criterion}")
    print(f"report: {report_path}")


def _base_report(gate: str, passed: bool, criterion: str, config: dict[str, Any]) -> dict[str, Any]:
    """Build the common report envelope shared by all five gates.

    Returns
    -------
    dict[str, Any]
        Envelope with gate name, timestamp, verdict, criterion, config and
        environment fields; gate-specific sections are added by the caller.
    """
    import onnx  # noqa: PLC0415 - heavy import, version recording only
    import onnxruntime  # noqa: PLC0415 - heavy import, version recording only

    return {
        "gate": gate,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "passed": passed,
        "criterion": criterion,
        "config": config,
        "environment": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
            "device": "cpu",
        },
    }


def _print_checks(checks: Mapping[str, bool]) -> None:
    """Print a name/PASS-FAIL table for a checks mapping."""
    for name, ok in checks.items():
        print(f"  {name:<48} {'PASS' if ok else 'FAIL'}")


# ---------------------------------------------------------------------------
# fixtures: the weight-matched GN2 pair + both export paths
# ---------------------------------------------------------------------------


def gn2_export_config(**overrides: Any) -> ExportConfig:
    """The GN2 fixture ``export:`` block (mirrors ``configs/gn2v2-dummy.yaml``).

    Since M4.5 the block carries the export-only half — the output
    manifest derives from the writers (`writer_manifest`), exactly as the
    shipped configs do.

    Parameters
    ----------
    **overrides : Any
        Attribute overrides applied to the fresh config (test/control use,
        e.g. ``model_name="GN2_v2"`` for the O5 rejection control).

    Returns
    -------
    ExportConfig
        A fresh, unresolved export config.
    """
    cfg = ExportConfig(
        model_name=MODEL_NAME,
        inputs=[
            ExportInput(port="inputs.jets", name="jet_features"),
            ExportInput(
                port="inputs.tracks", name="track_features", sequence=True, dyn_axis="n_tracks"
            ),
        ],
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def writer_manifest(
    modules: Mapping[str, Any],
    streams: Sequence[str],
    sequence_streams: Sequence[str],
) -> list[ExportOutput]:
    """The writer-derived export manifest — the M4.5 mechanism under gate.

    A default `TaskWriter` (the ``base2.yaml`` shipped writer, ``onnx:
    true``) declares the fixture's export entries from the SAME suffix
    helpers that name the eval columns; O2's identity bar then proves the
    derived manifest reproduces v1's hand-built output list bitwise — the
    re-sourcing the M4.5 re-run exists to gate.

    Returns
    -------
    list[ExportOutput]
        The assembled manifest in the v1 output order (global-stream
        entries first, then sequence-stream aux entries).
    """
    writer = TaskWriter()
    writer.name = "tasks"
    ctx = WriterDeclareCtx(
        model_modules=dict(modules),
        streams=tuple(streams),
        sequence_streams=tuple(sequence_streams),
    )
    return writer.onnx_outputs(ctx)


def _gn2_variables() -> dict[str, list[str]]:
    """The GN2 fixture per-stream variable lists.

    Returns
    -------
    dict[str, list[str]]
        Fresh copies of the fixture jet/track variable lists.
    """
    return {"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}


def _decollapse_v1_heads(
    v1, variables: Mapping[str, Sequence[str]], *, length: int = 20, seed: int = 99
) -> None:
    """Center every v1 task head so the int8 outputs have discriminating power.

    An UNTRAINED fixture's per-class logit offsets dominate the per-sample
    variation: every ``TrackOrigin`` argmax lands in one class, and —
    measured here — EVERY edge score sigmoids above the union-find merge
    threshold, so the whole sweep's ``VertexIndex`` is all-zeros (one
    cluster) and the int8 identity comparisons would match VACUOUSLY. The
    gates_m3 ``_decollapse_heads`` finding (M3-review fix), extended to the
    vertexing head: subtract each head's mean output over one probe batch
    from its final-layer bias — argmax then follows per-sample
    fluctuations, and the edge scores straddle the union-find threshold
    (multi-vertex assignments appear; the O2/O4 non-degeneracy bars assert
    it). Applied to the v1 wrapper BEFORE the `map_v1_state_dict` transfer,
    so both sides stay weight-identical and every parity bar is unchanged.

    Parameters
    ----------
    v1 : ModelWrapper
        The fixture wrapper (modified in place).
    variables : Mapping[str, Sequence[str]]
        Per-stream variable lists (probe-batch widths).
    length : int, optional
        Probe sequence length per non-global stream, by default 20.
    seed : int, optional
        Probe-batch RNG seed, by default 99.
    """
    gen = torch.Generator().manual_seed(seed)
    inputs: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    for stream, names in variables.items():
        if stream == "jets":
            inputs[stream] = torch.rand(1, len(names), generator=gen)
        else:
            inputs[stream] = torch.rand(1, length, len(names), generator=gen)
            masks[stream] = torch.zeros((1, length), dtype=torch.bool)
    with torch.no_grad():
        preds, _ = v1(
            {key: tensor.clone() for key, tensor in inputs.items()},
            {key: tensor.clone() for key, tensor in masks.items()},
            None,
        )
        for task in v1.model.tasks:
            out = preds[task.input_name][task.name]
            offset = out.reshape(-1, out.shape[-1]).mean(0)  # [C] logits / [1] edge score
            final = [m for m in task.modules() if isinstance(m, nn.Linear)][-1]
            final.bias.sub_(offset)


def _export_v1_onnx(
    v1,
    variables: Mapping[str, Sequence[str]],
    norm_dict: Path,
    workdir: Path,
    *,
    config_path: Path,
    config_payload: Mapping[str, Any],
    ckpt_path: Path,
    model_name: str = MODEL_NAME,
) -> SimpleNamespace:
    """Export through the REAL v1 path in-process (no CLI/run dir needed).

    The recipe driver: wrap the fixture's `SaltModel` in v1's `ONNXModel`
    (feature map from ``get_default_onnx_feature_map`` — the substring
    if-ladder being replaced), strict-load the v1 state dict, force
    torch-math on BOTH sides (``to_onnx.py:670,700`` — the Athena-agreement
    requirement), trace via lightning ``to_onnx`` at v1's exact call
    (``opset_version=20, dynamo=False``, ``to_onnx.py:714-721``), then run
    v1's ``add_metadata`` against a synthesized run dir.

    ``get_git_hash`` fallback: v1 CRASHES when git state is unreadable
    (e.g. a container binding only the worktree, whose ``.git`` file points
    outside the bind). The gate retries with the v1 module's hash patched
    to None and reports it — the comparison never requires hash equality
    (`HASH_JUSTIFICATION`).

    Returns
    -------
    SimpleNamespace
        ``onnx_path`` / ``om`` (the eager v1 export wrapper — the O4 torch
        reference) / ``feature_map`` / ``git_hash_patched``.
    """
    import salt.onnx.to_onnx as v1_export_module  # noqa: PLC0415 - patch target for the fallback

    feature_map = get_default_onnx_feature_map("r22default", list(variables), "jets")
    om = ONNXModel(
        onnx_feature_map=feature_map,
        variable_map={k: list(v) for k, v in variables.items()},
        name=model_name,
        tasks_to_output=["jets_classification", "track_origin", "track_vertexing"],
        model=v1.model,
        lrs_config=dict(_LRS),
        global_object="jets",
        norm_config={
            "norm_dict": str(norm_dict),
            "variables": {k: list(v) for k, v in variables.items()},
            "global_object": "jets",
            "input_map": {k: k for k in variables},
        },
    )
    om.load_state_dict(v1.state_dict())
    om.eval()
    om.float()
    change_attn_backends(om.model, "torch-math")  # to_onnx.py:700
    onnx_path = workdir / "v1_network.onnx"
    om.to_onnx(
        onnx_path,
        opset_version=20,
        input_names=om.input_names,
        output_names=om.output_names,
        dynamic_axes=om.dynamic_axes,
        dynamo=False,
    )
    metadata_kwargs: dict[str, Any] = {
        "config_path": config_path,
        "config": dict(config_payload),
        "ckpt_path": ckpt_path,
        "onnx_path": onnx_path,
        "model_name": om.name,
        "output_names": om.output_names,
        "onnx_feature_map": feature_map,
        "combine_outputs": [],
        "rename_outputs": {},
    }
    git_hash_patched = False
    try:
        add_metadata(**metadata_kwargs)
    except Exception as err:  # noqa: BLE001 - any git/env failure; the retry re-raises real bugs
        original = v1_export_module.get_git_hash
        v1_export_module.get_git_hash = lambda path: None  # noqa: ARG005 - v1 signature
        try:
            add_metadata(**metadata_kwargs)
        finally:
            v1_export_module.get_git_hash = original
        git_hash_patched = True
        print(f"note: v1 get_git_hash unavailable ({err}); patched to None ({HASH_JUSTIFICATION})")
    return SimpleNamespace(
        onnx_path=onnx_path, om=om, feature_map=feature_map, git_hash_patched=git_hash_patched
    )


def _build_gn2_pair(workdir: Path, *, with_v1_export: bool = True) -> SimpleNamespace:
    """Build the weight-matched GN2 pair and export it through both paths.

    Construction (the M2/M3 pattern): the small v1 GN2 (`build_test_gn2`,
    torch-math, 16-dim widths), the config-built v2 module dict bound
    against the compiled ONNX plan, then the strict `map_v1_state_dict`
    transfer — so both exports carry IDENTICAL weights. The metadata inputs
    (ckpt path, config payload, ``metadata.yaml`` content) are deliberately
    SHARED between the two exporters so the O5 per-key comparison isolates
    exporter behaviour: in production the embedded ``config.yaml`` payloads
    legitimately differ (each export embeds its own run's config) — that
    difference carries no exporter information.

    Returns
    -------
    SimpleNamespace
        ``v1`` / ``modules`` / ``variables`` / ``v2`` (`ExportResult`) /
        ``norm_dict`` / ``ckpt_path`` / ``config_payload`` and, with
        `with_v1_export`, ``v1_onnx`` / ``v1_om`` / ``v1_git_hash_patched``.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    fixture_dir = workdir / "v1_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    v1 = build_test_gn2(fixture_dir)
    norm_dict = fixture_dir / "norm_dict.yaml"
    variables = _gn2_variables()
    # BEFORE the transfer, so both sides stay weight-identical (docstring)
    _decollapse_v1_heads(v1, variables)

    modules = build_gn2v2_modules(norm_dict)
    manifest = writer_manifest(modules, ("jets", "tracks"), ("tracks",))
    resolved = attach_manifest(resolve_export_config(gn2_export_config(), RUN_NAME), manifest)
    plan = compile_onnx_plan(modules, resolved, variables)
    bind_all(modules, resolve_bind_schema([plan]))
    nn.ModuleDict(modules).load_state_dict(map_v1_state_dict(v1.state_dict(), modules))

    run_dir = workdir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "name": RUN_NAME,
        "data": {
            "variables": {k: list(v) for k, v in variables.items()},
            "global_object": "jets",
        },
    }
    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as fh:
        yaml.dump(config_payload, fh, sort_keys=False)
    (run_dir / "metadata.yaml").write_text("{}\n")  # v1 hard-requires the file (to_onnx.py:808)
    ckpt_path = run_dir / "weights.ckpt"  # recorded in metadata only — never read

    v2 = export_graph(
        modules,
        gn2_export_config(),
        variables,
        workdir / "v2_network.onnx",
        outputs=manifest,  # writer-derived (M4.5) — the re-sourcing under gate
        run_name=RUN_NAME,
        config=config_payload,
        run_metadata={},  # == the synthesized metadata.yaml content read by v1
        ckpt_path=ckpt_path,
    )
    pair = SimpleNamespace(
        v1=v1,
        modules=modules,
        variables=variables,
        v2=v2,
        norm_dict=norm_dict,
        ckpt_path=ckpt_path,
        config_payload=config_payload,
        v1_onnx=None,
        v1_om=None,
        v1_git_hash_patched=False,
    )
    if with_v1_export:
        v1_export = _export_v1_onnx(
            v1,
            variables,
            norm_dict,
            workdir,
            config_path=config_path,
            config_payload=config_payload,
            ckpt_path=ckpt_path,
        )
        pair.v1_onnx = v1_export.onnx_path
        pair.v1_om = v1_export.om
        pair.v1_git_hash_patched = v1_export.git_hash_patched
    return pair


# ---------------------------------------------------------------------------
# ONNX-file introspection + the v1-vs-v2 identity sweep (O2/O4 core)
# ---------------------------------------------------------------------------


def _graph_io(onnx_path: Path) -> dict[str, Any]:
    """Read the graph-level IO contract from a saved ONNX file.

    Returns
    -------
    dict[str, Any]
        ``inputs`` / ``outputs`` (name -> dim list, where dynamic dims
        appear as their ``dim_param`` axis names — the saved form of the
        ``dynamic_axes`` export argument) and the ``doc_string``.
    """
    import onnx  # noqa: PLC0415 - heavy import, gate-only

    model = onnx.load(str(onnx_path))

    def dims(values) -> dict[str, list[Any]]:
        out: dict[str, list[Any]] = {}
        for value in values:
            shape = value.type.tensor_type.shape
            out[value.name] = [d.dim_param or int(d.dim_value) for d in shape.dim]
        return out

    return {
        "inputs": dims(model.graph.input),
        "outputs": dims(model.graph.output),
        "doc_string": model.doc_string,
    }


def _read_gnn_config(onnx_path: Path) -> tuple[dict[str, Any], str]:
    """Read the parsed ``gnn_config`` payload + doc string from an ONNX file.

    Returns
    -------
    tuple[dict[str, Any], str]
        ``(payload, doc_string)`` — the payload preserves the stored key
        order (``json.loads`` keeps insertion order).

    Raises
    ------
    KeyError
        When the file carries no ``gnn_config`` metadata key.
    """
    import onnx  # noqa: PLC0415 - heavy import, gate-only

    model = onnx.load(str(onnx_path))
    for prop in model.metadata_props:
        if prop.key == "gnn_config":
            return json.loads(prop.value), model.doc_string
    raise KeyError(f"{onnx_path} carries no 'gnn_config' metadata key")


def _read_gnn_config_raw(onnx_path: Path) -> str:
    """Read the RAW stored ``gnn_config`` metadata string (no JSON round-trip).

    The O5 raw-byte comparison operates on this — a parse-then-redump check
    would mask non-semantic byte differences in the stored string.

    Returns
    -------
    str
        The stored metadata value, byte-for-byte.

    Raises
    ------
    KeyError
        When the file carries no ``gnn_config`` metadata key.
    """
    import onnx  # noqa: PLC0415 - heavy import, gate-only

    model = onnx.load(str(onnx_path))
    for prop in model.metadata_props:
        if prop.key == "gnn_config":
            return prop.value
    raise KeyError(f"{onnx_path} carries no 'gnn_config' metadata key")


def _athena_subset_json(payload: Mapping[str, Any]) -> str:
    """Serialise the Athena-parsed metadata subset for byte comparison.

    Returns
    -------
    str
        ``json.dumps`` of the `ATHENA_SUBSET_KEYS` entries in fixed order.
    """
    return json.dumps({key: payload.get(key) for key in ATHENA_SUBSET_KEYS})


def _run_by_name(session, feed: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run an ORT session and return outputs keyed BY NAME (design §7.6).

    Returns
    -------
    dict[str, np.ndarray]
        Output name -> array.
    """
    names = [output.name for output in session.get_outputs()]
    return dict(zip(names, session.run(None, dict(feed)), strict=True))


def _draw_gn2_feed(
    variables: Mapping[str, Sequence[str]], length: int, gen: torch.Generator
) -> dict[str, np.ndarray]:
    """Draw one random GN2 input feed: jets ``[1, J]``, tracks ``[L, F]``.

    Returns
    -------
    dict[str, np.ndarray]
        ONNX-input-name -> float32 array (the v1 Athena input contract).
    """
    return {
        "jet_features": torch.rand(1, len(variables["jets"]), generator=gen).numpy(),
        "track_features": torch.rand(length, len(variables["tracks"]), generator=gen).numpy(),
    }


def _identity_sweep(
    v1_session,
    v2_session,
    variables: Mapping[str, Sequence[str]],
    output_dtypes: Mapping[str, str],
    *,
    max_length: int,
    trials: int,
    seed: int,
    float_atol: float,
    corruption: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Compare two exported models output-by-output on identical inputs.

    Bitwise-first (the W1 discipline): a non-bitwise float output is
    JUSTIFIED within `float_atol` (`O2_JUSTIFICATION` — printed by the
    caller, never silent); int8 outputs and shapes/dtypes are never
    tolerant. NaNs fail outright (the v1 canary, ``check.py:84-85``).

    Parameters
    ----------
    v1_session, v2_session : onnxruntime.InferenceSession
        The two exported models.
    variables : Mapping[str, Sequence[str]]
        Per-stream variable lists (input widths).
    output_dtypes : Mapping[str, str]
        Output name -> declared dtype (``float32``/``int8``).
    max_length : int
        Sweep lengths ``0..max_length-1`` (incl. the zero-token case).
    trials : int
        Random draws per length.
    seed : int
        Input RNG seed.
    float_atol : float
        Justified-fallback bound for non-bitwise floats.
    corruption : Callable | None, optional
        TEST-ONLY hook applied to the v2 outputs dict before comparison
        (negative control), never exposed on the CLI, by default None.

    Returns
    -------
    dict[str, Any]
        ``passed`` + per-output records + truncated failure list.
    """
    gen = torch.Generator().manual_seed(seed)
    per_output: dict[str, dict[str, Any]] = {
        name: {
            "dtype": dtype,
            "n_cases": 0,
            "n_bitwise": 0,
            "n_justified": 0,
            "n_failed": 0,
            "worst_abs_diff": 0.0,
        }
        for name, dtype in output_dtypes.items()
    }
    distinct: dict[str, set[int]] = {
        name: set() for name, dtype in output_dtypes.items() if dtype == "int8"
    }
    failures: list[str] = []
    n_cases = 0
    for length in range(max_length):
        for _ in range(trials):
            n_cases += 1
            feed = _draw_gn2_feed(variables, length, gen)
            got_v1 = _run_by_name(v1_session, feed)
            got_v2 = _run_by_name(v2_session, feed)
            if corruption is not None:
                got_v2 = corruption(got_v2)
            for name, record in per_output.items():
                a, b = got_v1[name], got_v2[name]
                record["n_cases"] += 1
                if name in distinct:
                    distinct[name].update(int(v) for v in np.unique(a))
                where = f"L={length}"
                if a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes():
                    record["n_bitwise"] += 1
                    continue
                if record["dtype"] == "int8" or a.shape != b.shape or a.dtype != b.dtype:
                    record["n_failed"] += 1
                    failures.append(
                        f"{name} at {where}: v1 {a.dtype}{a.shape} vs v2 {b.dtype}{b.shape} "
                        "differ (int8/shape/dtype outputs are never tolerant)"
                    )
                    continue
                a64, b64 = a.astype(np.float64), b.astype(np.float64)
                if np.isnan(a64).any() or np.isnan(b64).any():
                    record["n_failed"] += 1
                    failures.append(f"{name} at {where}: NaN output")
                    continue
                diff = float(np.max(np.abs(a64 - b64))) if a64.size else 0.0
                record["worst_abs_diff"] = max(record["worst_abs_diff"], diff)
                if diff <= float_atol:
                    record["n_justified"] += 1
                else:
                    record["n_failed"] += 1
                    failures.append(f"{name} at {where}: max|diff|={diff:.3e} > {float_atol:g}")
    for name, values in distinct.items():
        per_output[name]["distinct_values"] = sorted(values)
    return {
        "passed": not failures,
        "n_cases": n_cases,
        "per_output": per_output,
        "n_failures": len(failures),
        "failures": failures[:50],
        "corrupted_by_test_hook": corruption is not None,
    }


def _print_identity(sweep: dict[str, Any], float_atol: float) -> None:
    """Print the per-output identity table + the justification block."""
    print(f"{'output':<24}{'dtype':<10}{'cases':>7}{'bitwise':>9}{'justified':>11}{'worst':>12}")
    print("-" * 96)
    for name, record in sweep["per_output"].items():
        if record["n_failed"]:
            status = "FAIL"
        elif record["n_justified"]:
            status = "JUSTIFIED"
        else:
            status = "BITWISE"
        print(
            f"{name:<24}{record['dtype']:<10}{record['n_cases']:>7}{record['n_bitwise']:>9}"
            f"{record['n_justified']:>11}{record['worst_abs_diff']:>12.3e}  {status}"
        )
    justified = {n: r for n, r in sweep["per_output"].items() if r["n_justified"]}
    if justified:
        print(f"justified non-bitwise float outputs ({len(justified)}; never silent):")
        for name in justified:
            print(f"  {name}: <= {float_atol:g} — {O2_JUSTIFICATION}")
    for failure in sweep["failures"][:10]:
        print(f"  FAIL: {failure}")
    if sweep["n_failures"] > 10:
        print(f"  ... and {sweep['n_failures'] - 10} more failures")


# ---------------------------------------------------------------------------
# O1 — v2-torch vs v2-ONNX over the full v1 sweep
# ---------------------------------------------------------------------------


def run_o1(
    outdir: Path | str,
    *,
    trials: int = 10,
    max_length: int = 40,
    float_atol: float = FLOAT_ATOL,
) -> tuple[int, dict[str, Any]]:
    """O1: v2-torch vs v2-ONNX <= 1e-6 over L=0..39 incl. L=0 (design §9.5 M4 gate a).

    Construction: the weight-matched GN2 fixture exported through the
    ``salt2 export`` programmatic core (`export_graph`: ONNX plan from
    config-derived sources, `OnnxAdapter` trace at opset 20, torch-math
    forced via the ``set_export_mode`` protocol), then the shipped checker
    (`check_onnx`) sweeps every sequence length ``0..max_length-1`` times
    `trials` random draws — v1's exact loop (``check.py:176-178``)
    INCLUDING the zero-token edge case (pooling zero-row pad + union-find
    fake-track workarounds in-graph).

    PASS requires every case to agree at ``rtol=atol=float_atol`` (1e-6 —
    see `FLOAT_ATOL` for the derivation vs v1's shipped 1e-4 bar), int8 aux
    outputs exact, no NaNs, no exact-zero floats (the v1 dead-output
    canary), and the case count to equal the full sweep.

    Parameters
    ----------
    outdir : Path | str
        Report + artifact output directory.
    trials : int, optional
        Random draws per length, by default 10 (v1 ``check.py:177``).
    max_length : int, optional
        Sweep lengths ``0..max_length-1``, by default 40 (v1
        ``check.py:176``).
    float_atol : float, optional
        Float rtol AND atol, by default `FLOAT_ATOL`.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)`` — 0 only if every check passed.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("=" * 96)
    print("O1 torch-vs-ONNX sweep — eager v2 adapter vs onnxruntime, full v1 sweep incl. L=0")
    print("=" * 96)
    pair = _build_gn2_pair(outdir / "export", with_v1_export=False)
    result = check_onnx(
        pair.v2.adapter,
        pair.v2.onnx_path,
        max_length=max_length,
        trials=trials,
        float_rtol=float_atol,
        float_atol=float_atol,
    )
    checks = {
        "sweep_passed": result.passed,
        "case_count_is_full_sweep": result.n_cases == max_length * trials,
        "zero_token_length_swept": max_length >= 1,  # L=0 is case 0 by construction
        "float_diffs_within_bar": all(d <= float_atol for d in result.worst_abs_diff.values()),
    }
    passed = all(checks.values())
    criterion = (
        f"v2-torch (eager OnnxAdapter, torch-math) vs v2-ONNX (onnxruntime CPU) at "
        f"rtol=atol={float_atol:g} over L=0..{max_length - 1} x {trials} trials incl. L=0 "
        "(v1 ships 1e-4: check.py:81-92; spike headroom ~6e-8); int8 aux outputs exact; "
        "no NaN; no exact-zero floats"
    )
    report = _base_report(
        "o1_torch_vs_onnx",
        passed,
        criterion,
        {
            "trials": trials,
            "max_length": max_length,
            "float_atol": float_atol,
            "onnx_path": str(pair.v2.onnx_path),
            "output_names": pair.v2.adapter.output_names,
            "output_dtypes": pair.v2.adapter.output_dtypes,
            "plan_hash": pair.v2.plan.plan_hash,
        },
    )
    report["checks"] = checks
    report["n_cases"] = result.n_cases
    report["worst_abs_diff"] = result.worst_abs_diff
    report["n_failures"] = len(result.failures)
    report["failures"] = result.failures[:50]

    print(f"cases: {result.n_cases} (L=0..{max_length - 1} x {trials} trials)")
    for name, dtype in zip(
        pair.v2.adapter.output_names, pair.v2.adapter.output_dtypes, strict=True
    ):
        if dtype == "int8":
            print(f"  {name:<24} exact (int8)")
        else:
            print(f"  {name:<24} worst abs diff {result.worst_abs_diff.get(name, 0.0):.3e}")
    for failure in result.failures[:10]:
        print(f"  FAIL: {failure}")
    _print_checks(checks)
    _print_verdict("o1", passed, criterion, _emit_report(report, outdir, "o1"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# O2 — v1 export path vs v2 export path: output identity + contract equality
# ---------------------------------------------------------------------------


def run_o2(
    outdir: Path | str,
    *,
    trials: int = 3,
    max_length: int = 40,
    float_atol: float = FLOAT_ATOL,
    corruption: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """O2: v1-ONNX vs v2-ONNX identity on the same weights (design §9.5 M4 gate b).

    Construction: ONE set of weights (the GN2 fixture), exported through
    BOTH paths — v1's `ONNXModel` + ``add_metadata`` in-process (the exact
    CLI flow minus checkpoint plumbing, `_export_v1_onnx`) and the v2
    ``salt2 export`` programmatic core — then both ``.onnx`` files are
    evaluated under onnxruntime on IDENTICAL inputs over the v1 sweep.

    PASS requires: identical input/output names (session order AND the
    adapter's generated lists), identical graph dims incl. dynamic-axis
    names (the saved ``dim_param`` form of ``dynamic_axes``), identical
    ``doc_string``, byte-equal Athena-visible metadata subset
    (`ATHENA_SUBSET_KEYS`), and per-case outputs bitwise (int8 always;
    floats bitwise or justified <= `float_atol`, printed — the bitwise
    target with the W1 fallback discipline; see `O2_JUSTIFICATION` for why
    bitwise is not contractual across two different traced graph
    topologies). Each int8 output must take >= 2 distinct values over the
    sweep (`_decollapse_v1_heads` — an untrained fixture's collapsed
    argmax/all-merged union-find would make the exact comparisons
    vacuous).

    Parameters
    ----------
    outdir : Path | str
        Report + artifact output directory.
    trials : int, optional
        Random draws per length, by default 3 (two ORT runs per case).
    max_length : int, optional
        Sweep lengths ``0..max_length-1``, by default 40.
    float_atol : float, optional
        Justified-fallback bound, by default `FLOAT_ATOL`.
    corruption : Callable | None, optional
        TEST-ONLY hook applied to the v2 outputs before comparison
        (negative control — the gate must FAIL), never exposed on the CLI,
        by default None.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("=" * 96)
    print("O2 v1-vs-v2 export identity — same weights through both export paths, ORT vs ORT")
    print("=" * 96)
    pair = _build_gn2_pair(outdir / "export", with_v1_export=True)
    assert pair.v1_onnx is not None
    v1_session = make_session(pair.v1_onnx)
    v2_session = make_session(pair.v2.onnx_path)
    v1_io = _graph_io(pair.v1_onnx)
    v2_io = _graph_io(pair.v2.onnx_path)
    v1_meta, v1_doc = _read_gnn_config(pair.v1_onnx)
    v2_meta, v2_doc = _read_gnn_config(pair.v2.onnx_path)

    v1_inputs = [item.name for item in v1_session.get_inputs()]
    v2_inputs = [item.name for item in v2_session.get_inputs()]
    v1_outputs = [item.name for item in v1_session.get_outputs()]
    v2_outputs = [item.name for item in v2_session.get_outputs()]
    output_dtypes = dict(
        zip(pair.v2.adapter.output_names, pair.v2.adapter.output_dtypes, strict=True)
    )
    sweep = _identity_sweep(
        v1_session,
        v2_session,
        pair.variables,
        output_dtypes,
        max_length=max_length,
        trials=trials,
        seed=42,
        float_atol=float_atol,
        corruption=corruption,
    )
    checks = {
        "input_names_equal": v1_inputs == v2_inputs == pair.v2.adapter.input_names,
        "output_names_equal": v1_outputs == v2_outputs == pair.v2.adapter.output_names,
        "graph_dims_equal": v1_io["inputs"] == v2_io["inputs"]
        and v1_io["outputs"] == v2_io["outputs"],
        "doc_strings_equal": v1_doc == v2_doc == MODEL_NAME,
        "athena_metadata_subset_equal": _athena_subset_json(v1_meta)
        == _athena_subset_json(v2_meta),
        "outputs_identical_over_sweep": sweep["passed"],
        "case_count_is_full_sweep": sweep["n_cases"] == max_length * trials,
        # >= 2 distinct values per int8 output over the sweep, or the exact
        # comparisons prove nothing (_decollapse_v1_heads)
        "int8_outputs_nondegenerate": all(
            len(record.get("distinct_values", ())) >= 2
            for record in sweep["per_output"].values()
            if record["dtype"] == "int8"
        ),
    }
    passed = all(checks.values())
    criterion = (
        "identical input/output names, graph dims + dynamic-axis names, doc_string and "
        "Athena-visible gnn_config subset between the v1 and v2 exports of the SAME weights; "
        f"outputs over L=0..{max_length - 1} x {trials} trials: int8 bitwise, floats bitwise "
        f"or justified <= {float_atol:g} (printed, never silent); int8 outputs non-degenerate "
        "(>= 2 distinct values over the sweep — _decollapse_v1_heads)"
    )
    report = _base_report(
        "o2_v1_vs_v2_identity",
        passed,
        criterion,
        {
            "trials": trials,
            "max_length": max_length,
            "float_atol": float_atol,
            "v1_onnx": str(pair.v1_onnx),
            "v2_onnx": str(pair.v2.onnx_path),
            "v1_git_hash_patched": pair.v1_git_hash_patched,
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["identity"] = sweep
    report["io_contract"] = {
        "input_names": {"v1": v1_inputs, "v2": v2_inputs},
        "output_names": {"v1": v1_outputs, "v2": v2_outputs},
        "graph_dims": {"v1": v1_io, "v2": v2_io},
    }
    report["athena_metadata_subset"] = {
        "v1": _athena_subset_json(v1_meta),
        "v2": _athena_subset_json(v2_meta),
    }
    report["justification"] = O2_JUSTIFICATION

    print(f"v1 export: {pair.v1_onnx}")
    print(f"v2 export: {pair.v2.onnx_path}")
    print(f"inputs:  v1={v1_inputs} v2={v2_inputs}")
    print(f"outputs: v1={v1_outputs} v2={v2_outputs}")
    print(f"graph dims v1: in={v1_io['inputs']} out={v1_io['outputs']}")
    print(f"graph dims v2: in={v2_io['inputs']} out={v2_io['outputs']}")
    if not checks["athena_metadata_subset_equal"]:
        print(f"metadata subset v1: {_athena_subset_json(v1_meta)}")
        print(f"metadata subset v2: {_athena_subset_json(v2_meta)}")
    _print_identity(sweep, float_atol)
    _print_checks(checks)
    _print_verdict("o2", passed, criterion, _emit_report(report, outdir, "o2"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# O3 — two independent dynamic sequence axes (design risk 7)
# ---------------------------------------------------------------------------


def two_stream_export_config() -> ExportConfig:
    """The two-stream (tracks + electrons) ``export:`` block for gate O3.

    Export-only half (M4.5) — the manifest derives from `writer_manifest`,
    where the per-electron ``ElectronOrigin`` entry (the default
    snake-to-Pascal naming of the ``electron_origin`` task) is
    load-bearing: electrons are the SECOND stream in the concat layout, so
    producing it forces a `Split` slice at a DYNAMIC offset — the exact
    operation design risk 7 is about (a track-only output would leave the
    offset-dependent slice untraced and the gate vacuous).

    Returns
    -------
    ExportConfig
        A fresh, unresolved export config.
    """
    return ExportConfig(
        model_name=MODEL_NAME,
        inputs=[
            ExportInput(port="inputs.jets", name="jet_features"),
            ExportInput(
                port="inputs.tracks", name="track_features", sequence=True, dyn_axis="n_tracks"
            ),
            ExportInput(
                port="inputs.electrons",
                name="electron_features",
                sequence=True,
                dyn_axis="n_electrons",
            ),
        ],
    )


def _build_two_stream(workdir: Path) -> SimpleNamespace:
    """Build the two-stream pair: v1 GN2e fixture + a mirrored v2 export.

    The v2 module dict mirrors ``build_test_gn2(with_electrons=True)``
    module-for-module (concat order tracks-then-electrons = v1 init_nets
    order) and strict-transfers the v1 trunk weights via
    `map_v1_state_dict`. The v2-only ``electron_origin`` head has no v1
    counterpart — the load runs ``strict=False`` and the gate ASSERTS the
    missing keys are exactly that head (everything else weight-matched, so
    the v1 cross-check stays exact); its bind-time init is seeded.

    Returns
    -------
    SimpleNamespace
        ``v1`` / ``modules`` / ``variables`` / ``result`` (`ExportResult`)
        / ``plan`` / ``missing_keys``.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    fixture_dir = workdir / "v1_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    v1 = build_test_gn2(fixture_dir, with_electrons=True)
    norm_dict = fixture_dir / "norm_dict.yaml"
    variables = {
        "jets": list(JET_VARIABLES),
        "tracks": list(TRACK_VARIABLES),
        "electrons": list(ELECTRON_VARIABLES),
    }
    # BEFORE the transfer, so both sides stay weight-identical (docstring)
    _decollapse_v1_heads(v1, variables)
    dense = {"hidden_layers": [16], "activation": "ReLU"}
    torch.manual_seed(42)
    modules: dict[str, Any] = {
        "norm": Normaliser(
            norm_dict=norm_dict, streams=["jets", "tracks", "electrons"], global_object="jets"
        ),
        "track_embed": StreamEmbed(
            stream="tracks", out_dim=16, dense=dense, context=["normed.jets"]
        ),
        "electron_embed": StreamEmbed(
            stream="electrons", out_dim=16, dense=dense, context=["normed.jets"]
        ),
        # concat order tracks-then-electrons == the v1 fixture's init_nets
        # order (gn2_fixture.py — the v1 concat order contract)
        "concat": Concat(streams=["tracks", "electrons"]),
        "encoder": TransformerEncoder(
            dim=16,
            num_layers=2,
            out_dim=16,
            attention={"num_heads": 2, "attn_type": "torch-math"},
            dense={"activation": "ReLU", "gated": False},
        ),
        "split": Split(streams=["tracks", "electrons"]),
        "pool": GlobalAttentionPooling(input="encoded.seq", out="pooled.global"),
        "jets_classification": ClassificationTaskModule(
            stream="jets",
            label="flavour_label",
            class_names=["bjets", "cjets", "ujets"],
            input="pooled.global",
            dense=dense,
        ),
        "track_origin": ClassificationTaskModule(
            stream="tracks",
            label="ftagTruthOriginLabel",
            class_names=list(ORIGIN_CLASSES),
            context="pooled.global",
            weight=0.5,
            dense=dense,
        ),
        "track_vertexing": VertexingTaskModule(
            stream="tracks",
            label="ftagTruthVertexIndex",
            origin_label="ftagTruthOriginLabel",
            context="pooled.global",
            weight=1.5,
            dense=dense,
        ),
        # v2-only second-axis head (LAST in dict order, so the v1 task
        # indices 0..2 keep their default association) — see the docstring
        "electron_origin": ClassificationTaskModule(
            stream="electrons",
            label="ftagTruthOriginLabel",
            class_names=[f"e{i}" for i in range(4)],
            context="pooled.global",
            dense=dense,
        ),
    }
    for name, module in modules.items():
        module.name = name

    export_cfg = two_stream_export_config()
    manifest = writer_manifest(modules, ("jets", "tracks", "electrons"), ("tracks", "electrons"))
    resolved = attach_manifest(resolve_export_config(export_cfg, RUN_NAME), manifest)
    plan = compile_onnx_plan(modules, resolved, variables)
    torch.manual_seed(42)  # deterministic bind-time init for the v2-only head
    bind_all(modules, resolve_bind_schema([plan]))
    holder = nn.ModuleDict(modules)
    mapped = map_v1_state_dict(v1.state_dict(), modules)
    missing, unexpected = holder.load_state_dict(mapped, strict=False)
    assert not unexpected, f"unexpected keys in the v1->v2 transfer: {unexpected}"
    assert all(key.startswith("electron_origin.") for key in missing), (
        f"only the v2-only electron head may be unmatched, got missing={list(missing)}"
    )
    result = export_graph(
        modules,
        export_cfg,
        variables,
        workdir / "two_stream.onnx",
        outputs=manifest,
        run_name=RUN_NAME,
    )
    return SimpleNamespace(
        v1=v1,
        modules=modules,
        variables=variables,
        result=result,
        plan=result.plan,
        missing_keys=list(missing),
    )


def _v1_cross_check(
    two: SimpleNamespace, session, grid: Sequence[Mapping[str, int]], float_atol: float
) -> dict[str, Any]:
    """Cross-check the two-axis ONNX export against the v1 EAGER forward.

    Eager-vs-traced agreement alone (the `check_onnx` sweep) cannot expose
    a Split mis-slice shared by BOTH v2 paths; the v1 wrapper computes the
    stream gather with v1's independent boolean-index mechanism, so
    agreement here pins the traced slices to v1 semantics. Per grid point
    (one deterministic draw): jets probabilities (v1 softmax) <= 1e-6 vs
    the ONNX scalars; ``TrackOrigin`` (v1 raw-logit argmax — invariant
    under the v2 softmax conversion) and ``VertexIndex`` (the v1 chain on
    the v1 edge scores) int8-exact. The v2-only ``ElectronOrigin`` head has
    no v1 counterpart and is excluded (covered by the eager-vs-traced
    sweep).

    Returns
    -------
    dict[str, Any]
        ``passed`` + per-point records + the worst probability diff.
    """
    gen = torch.Generator().manual_seed(11)
    records: list[dict[str, Any]] = []
    worst_prob = 0.0
    origin_distinct: set[int] = set()
    vertex_distinct: set[int] = set()
    for lengths in grid:
        n_trk, n_el = int(lengths["tracks"]), int(lengths["electrons"])
        jets = torch.rand(1, len(JET_VARIABLES), generator=gen)
        tracks = torch.rand(n_trk, len(TRACK_VARIABLES), generator=gen)
        electrons = torch.rand(n_el, len(ELECTRON_VARIABLES), generator=gen)
        mask = torch.zeros((1, n_trk), dtype=torch.bool)
        with torch.no_grad():
            preds, _ = two.v1(
                {
                    "jets": jets.clone(),
                    "tracks": tracks.unsqueeze(0).clone(),
                    "electrons": electrons.unsqueeze(0).clone(),
                },
                {"tracks": mask.clone(), "electrons": torch.zeros((1, n_el), dtype=torch.bool)},
                None,
            )
        outs = _run_by_name(
            session,
            {
                "jet_features": jets.numpy(),
                "track_features": tracks.numpy(),
                "electron_features": electrons.numpy(),
            },
        )
        p_v1 = torch.softmax(preds["jets"]["jets_classification"], dim=-1).numpy().ravel()
        p_onnx = np.array(
            [outs[f"{MODEL_NAME}_{suffix}"] for suffix in ("pb", "pc", "pu")], dtype=np.float64
        ).ravel()
        prob_diff = float(np.max(np.abs(p_v1.astype(np.float64) - p_onnx)))
        worst_prob = max(worst_prob, prob_diff)
        origin_v1 = torch.argmax(preds["tracks"]["track_origin"], dim=-1).squeeze(0).char().numpy()
        scores = preds["tracks"]["track_vertexing"]
        vertex_v1 = (
            mask_fill_flattened(get_node_assignment_jit(scores, mask), mask)
            .reshape(-1)
            .char()
            .numpy()
        )
        origin_distinct.update(int(v) for v in np.unique(origin_v1))
        vertex_distinct.update(int(v) for v in np.unique(vertex_v1))
        record = {
            "lengths": dict(lengths),
            "prob_diff": prob_diff,
            "prob_ok": prob_diff <= float_atol and not np.isnan(p_onnx).any(),
            "origin_ok": bool(np.array_equal(origin_v1, outs[f"{MODEL_NAME}_TrackOrigin"])),
            "vertex_ok": bool(np.array_equal(vertex_v1, outs[f"{MODEL_NAME}_VertexIndex"])),
        }
        records.append(record)
    passed = all(r["prob_ok"] and r["origin_ok"] and r["vertex_ok"] for r in records)
    return {
        "passed": passed,
        "worst_prob_diff": worst_prob,
        "n_points": len(records),
        "points": records,
        "origin_distinct": sorted(origin_distinct),
        "vertex_distinct": sorted(vertex_distinct),
        # >= 2 origin classes and a multi-vertex assignment somewhere on the
        # grid, or the int8 comparisons prove nothing (_decollapse_v1_heads)
        "aux_nondegenerate": len(origin_distinct) >= 2
        and any(value > 0 for value in vertex_distinct),
    }


def run_o3(
    outdir: Path | str,
    *,
    trials: int = 3,
    track_lengths: Sequence[int] = (0, 1, 2, 7, 21, 39),
    electron_lengths: Sequence[int] = (0, 1, 4, 11),
    float_atol: float = FLOAT_ATOL,
) -> tuple[int, dict[str, Any]]:
    """O3: two independent dynamic axes export + agree on the full grid (gate c, risk 7).

    Construction: the two-stream fixture (`_build_two_stream` — v1 GN2e
    trunk weights, tracks + electrons, a per-ELECTRON output so the
    dynamic-offset `Split` slice is in the graph) exported through the v2
    core, then THREE independent agreements:

    1. eager-v2-torch vs v2-ONNX (`check_onnx`) over the full
       ``(L_trk, L_el)`` grid INCLUDING zero on each axis and ``(0, 0)``,
       at the 1e-6 bar;
    2. the v1 wrapper's eager forward vs the ONNX outputs per grid point
       (`_v1_cross_check` — kills any mis-slice shared by both v2 paths);
    3. the per-stream outputs track DIFFERENT axes at an asymmetric point
       (``TrackOrigin`` is ``(L_trk,)``, ``ElectronOrigin`` ``(L_el,)``).

    The gate also RECORDS the adjudicated risk-7 mechanism: the compiled
    ONNX plan must show `Concat` producing and `Split` consuming the
    ``seq.offsets`` boundary tensor (mechanism (A) of the recipe spike;
    the eager Python-int slicing is provably wrong under tracing with two
    dynamic axes — evidence preserved in the `Split` class docstring,
    embedded in the report).

    Parameters
    ----------
    outdir : Path | str
        Report + artifact output directory.
    trials : int, optional
        Random draws per grid point for the eager-vs-traced sweep, by
        default 3.
    track_lengths : Sequence[int], optional
        Track-axis grid values (must include 0), by default
        ``(0, 1, 2, 7, 21, 39)``.
    electron_lengths : Sequence[int], optional
        Electron-axis grid values (must include 0), by default
        ``(0, 1, 4, 11)``.
    float_atol : float, optional
        Float bar for both float comparisons, by default `FLOAT_ATOL`.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("=" * 96)
    print("O3 two dynamic axes — (L_trk, L_el) grid incl. zeros; the design risk-7 blocker")
    print("=" * 96)
    two = _build_two_stream(outdir / "export")
    grid = [
        {"tracks": n_trk, "electrons": n_el} for n_trk in track_lengths for n_el in electron_lengths
    ]
    sweep = check_onnx(
        two.result.adapter,
        two.result.onnx_path,
        trials=trials,
        float_rtol=float_atol,
        float_atol=float_atol,
        lengths_grid=grid,
    )
    session = make_session(two.result.onnx_path)
    cross = _v1_cross_check(two, session, grid, float_atol)

    # independent-axes shape probe at an asymmetric point
    gen = torch.Generator().manual_seed(5)
    probe = _run_by_name(
        session,
        {
            "jet_features": torch.rand(1, len(JET_VARIABLES), generator=gen).numpy(),
            "track_features": torch.rand(5, len(TRACK_VARIABLES), generator=gen).numpy(),
            "electron_features": torch.rand(2, len(ELECTRON_VARIABLES), generator=gen).numpy(),
        },
    )
    shapes_ok = (
        probe[f"{MODEL_NAME}_TrackOrigin"].shape == (5,)
        and probe[f"{MODEL_NAME}_ElectronOrigin"].shape == (2,)
        and probe[f"{MODEL_NAME}_VertexIndex"].shape == (5,)
    )
    concat_offsets = any(
        "seq.offsets" in step.produces for step in two.plan.steps if step.name == "concat"
    )
    split_offsets = any(
        "seq.offsets" in step.requires for step in two.plan.steps if step.name == "split"
    )
    # EVERY int8 output must take >= 2 distinct values over the traced
    # sweep — incl. the v2-only ElectronOrigin, which `_v1_cross_check`
    # cannot cover (no v1 counterpart): a collapsed argmax would make its
    # int8-exact comparison vacuous (M4-review fix; `CheckResult.int8_distinct`)
    int8_names = [
        name
        for name, dtype in zip(
            two.result.adapter.output_names, two.result.adapter.output_dtypes, strict=True
        )
        if dtype == "int8"
    ]
    checks = {
        "grid_includes_zero_on_each_axis": 0 in track_lengths and 0 in electron_lengths,
        "grid_includes_double_zero": any(
            point["tracks"] == 0 and point["electrons"] == 0 for point in grid
        ),
        "sweep_passed": sweep.passed,
        "case_count_is_full_grid": sweep.n_cases == len(grid) * trials,
        "v1_cross_check_passed": cross["passed"],
        "aux_outputs_nondegenerate": cross["aux_nondegenerate"],
        "int8_outputs_nondegenerate_in_sweep": all(
            len(sweep.int8_distinct.get(name, [])) >= 2 for name in int8_names
        ),
        "independent_axes_shapes": shapes_ok,
        "concat_publishes_seq_offsets_in_onnx_mode": concat_offsets,
        "split_consumes_seq_offsets_in_onnx_mode": split_offsets,
    }
    passed = all(checks.values())
    criterion = (
        f"two-stream export agrees with eager v2 torch at {float_atol:g} AND with the v1 "
        "eager forward (probs <= bar; TrackOrigin/VertexIndex int8-exact) over the full "
        "(L_trk, L_el) grid incl. zero on each axis and (0,0); every int8 output — incl. the "
        "v2-only ElectronOrigin — takes >= 2 distinct values over the sweep (non-degeneracy); "
        "per-stream outputs track independent dynamic axes; the ONNX plan carries the "
        "adjudicated Concat seq.offsets -> Split index_select mechanism (design risk 7)"
    )
    report = _base_report(
        "o3_two_dynamic_axes",
        passed,
        criterion,
        {
            "trials": trials,
            "track_lengths": list(track_lengths),
            "electron_lengths": list(electron_lengths),
            "float_atol": float_atol,
            "onnx_path": str(two.result.onnx_path),
            "v2_only_head_missing_keys": two.missing_keys,
        },
    )
    report["checks"] = checks
    report["n_cases"] = sweep.n_cases
    report["worst_abs_diff"] = sweep.worst_abs_diff
    report["int8_distinct"] = sweep.int8_distinct
    report["failures"] = sweep.failures[:50]
    report["v1_cross_check"] = cross
    report["split_decision_evidence"] = Split.__doc__

    print(f"grid: {len(grid)} points x {trials} trials = {sweep.n_cases} cases")
    for name, diff in sorted(sweep.worst_abs_diff.items()):
        print(f"  {name:<26} worst abs diff {diff:.3e}")
    for failure in sweep.failures[:10]:
        print(f"  FAIL: {failure}")
    print(
        f"v1 cross-check: {cross['n_points']} points, worst prob diff "
        f"{cross['worst_prob_diff']:.3e}, passed={cross['passed']}"
    )
    print("risk-7 mechanism (recorded from the compiled plan + the Split docstring):")
    print(f"  concat ONNX-mode produces seq.offsets: {concat_offsets}")
    print(f"  split ONNX-mode requires seq.offsets:  {split_offsets}")
    _print_checks(checks)
    _print_verdict("o3", passed, criterion, _emit_report(report, outdir, "o3"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# O4 — vertexing: in-graph union-find triple identity
# ---------------------------------------------------------------------------


def run_o4(
    outdir: Path | str,
    *,
    trials: int = 3,
    max_length: int = 40,
    corruption: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[int, dict[str, Any]]:
    """O4: VertexIndex triple identity — the union-find-in-graph proof (gate d).

    The raw edge scores are weight-identical by construction (gate O2 on
    the same fixture), so the union-find OUTPUT must be int-exact: per
    sweep case, v1-ONNX ``VertexIndex`` == v2-ONNX ``VertexIndex`` == the
    v1 export wrapper's EAGER torch chain (`ONNXModel.forward`'s
    ``get_node_assignment_jit`` + ``mask_fill_flattened`` + ``.char()``,
    ``to_onnx.py:426-432`` — the same ``@torch.jit.script`` function the
    trace inlines, fake-pad-track workaround inside,
    ``union_find.py:151-153``). Includes L=0 and L=1 (zero-edge cases).

    Non-degeneracy bar: the sweep must produce at least one multi-vertex
    assignment (an index > 0) — all-zero outputs (every track in one
    cluster) would match vacuously.

    Parameters
    ----------
    outdir : Path | str
        Report + artifact output directory.
    trials : int, optional
        Random draws per length, by default 3.
    max_length : int, optional
        Sweep lengths ``0..max_length-1``, by default 40.
    corruption : Callable | None, optional
        TEST-ONLY hook applied to the v2 ``VertexIndex`` array before
        comparison (negative control), never exposed on the CLI, by
        default None.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("=" * 96)
    print("O4 vertexing union-find — v1-ONNX == v2-ONNX == v1-torch chain, int8-exact sweep")
    print("=" * 96)
    pair = _build_gn2_pair(outdir / "export", with_v1_export=True)
    assert pair.v1_onnx is not None
    assert pair.v1_om is not None
    v1_session = make_session(pair.v1_onnx)
    v2_session = make_session(pair.v2.onnx_path)
    name = f"{MODEL_NAME}_VertexIndex"
    torch_index = pair.v1_om.output_names.index(name)

    gen = torch.Generator().manual_seed(42)
    n_cases = 0
    mismatches: list[str] = []
    nonzero_assignment_seen = False
    for length in range(max_length):
        for _ in range(trials):
            n_cases += 1
            feed = _draw_gn2_feed(pair.variables, length, gen)
            vertex_v1 = _run_by_name(v1_session, feed)[name]
            vertex_v2 = _run_by_name(v2_session, feed)[name]
            if corruption is not None:
                vertex_v2 = corruption(vertex_v2)
            with torch.no_grad():
                torch_outputs = pair.v1_om(
                    torch.from_numpy(feed["jet_features"]).clone(),
                    torch.from_numpy(feed["track_features"]).clone(),
                )
            vertex_torch = torch_outputs[torch_index].numpy()
            if (vertex_torch > 0).any():
                nonzero_assignment_seen = True
            where = f"L={length}"
            for label, got in (("v1-onnx", vertex_v1), ("v2-onnx", vertex_v2)):
                if got.dtype != np.int8 or got.shape != (length,):
                    mismatches.append(f"{label} at {where}: dtype/shape {got.dtype}{got.shape}")
            if not (
                np.array_equal(vertex_v1, vertex_v2) and np.array_equal(vertex_torch, vertex_v2)
            ):
                mismatches.append(
                    f"{where}: torch={vertex_torch.tolist()} v1={vertex_v1.tolist()} "
                    f"v2={vertex_v2.tolist()}"
                )
    checks = {
        "all_triple_exact": not mismatches,
        "case_count_is_full_sweep": n_cases == max_length * trials,
        "zero_edge_lengths_swept": max_length >= 2,  # L=0 and L=1 have no track pairs
        "vertex_assignments_nondegenerate": nonzero_assignment_seen,
    }
    passed = all(checks.values())
    criterion = (
        f"VertexIndex int8-exact across v1-ONNX, v2-ONNX and the v1 eager torch chain "
        f"(to_onnx.py:426-432) for every case of L=0..{max_length - 1} x {trials} trials "
        "(incl. the zero-edge L=0/L=1 cases), with at least one multi-vertex assignment in "
        "the sweep (non-degeneracy)"
    )
    report = _base_report(
        "o4_vertexing_union_find",
        passed,
        criterion,
        {
            "trials": trials,
            "max_length": max_length,
            "v1_onnx": str(pair.v1_onnx),
            "v2_onnx": str(pair.v2.onnx_path),
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["n_cases"] = n_cases
    report["n_mismatches"] = len(mismatches)
    report["mismatches"] = mismatches[:20]

    print(f"cases: {n_cases}; mismatches: {len(mismatches)}")
    for mismatch in mismatches[:10]:
        print(f"  MISMATCH {mismatch}")
    _print_checks(checks)
    _print_verdict("o4", passed, criterion, _emit_report(report, outdir, "o4"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# O5 — negative controls + gnn_config metadata vs v1
# ---------------------------------------------------------------------------


def _preview(value: Any, limit: int = 96) -> str:
    """Truncate a JSON-serialised value for the per-key report table.

    Returns
    -------
    str
        The (possibly truncated) JSON string.
    """
    text = json.dumps(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def run_o5(
    outdir: Path | str,
    *,
    check_trials: int = 2,
    check_length: int = 11,
    float_atol: float = FLOAT_ATOL,
) -> tuple[int, dict[str, Any]]:
    """O5: negative controls + ``gnn_config`` byte-comparison vs v1 (design §7.5).

    Three control families, all of which must behave:

    1. **Corrupted weights fail the checker** (proves teeth): after the v2
       export, ONE element of the jets-classification output-layer bias is
       perturbed (+1.0) on the EAGER side — `check_onnx` must FAIL, and
       must PASS again after the weights are restored (locality). The
       perturbation is deliberately ASYMMETRIC: a uniform shift of a
       first-layer weight cancels through the softmax to ~1e-8 and
       silently passes (observed in stage A — documented in
       ``test_onnx_export.py``).
    2. **Invalid ``export.model_name`` rejected**: ``GN2_v2`` must raise a
       `ConfigError` naming the offending value and the rule (the §4.1
       error-quality bar: actionable, cites ``to_onnx.py:169-170``), from
       both `resolve_export_config` and the full `export_graph` path —
       BEFORE any file is written. The default derivation must strip the
       run name (``GN2_v2`` -> ``GN2v2``, ``to_onnx.py:687``).
    3. **Metadata byte-comparison**: on shared metadata inputs (see
       `_build_gn2_pair`), the v2 ``gnn_config`` must carry the 13 v1 keys
       as an ordered PREFIX with the single additive trailing
       ``plan_hash`` (design §7.5); the Athena-visible subset
       (`ATHENA_SUBSET_KEYS`) must be byte-equal; every v1 key must be
       byte-equal with ``salt_export_hash`` the only justified exception
       (`HASH_JUSTIFICATION` — reported, never silent); and the RAW stored
       v2 string must be a literal byte prefix-extension of v1's (v1 minus
       its closing brace + exactly the ``plan_hash`` entry — strictly
       stronger than the per-key parse-redump checks). Whole-envelope
       byte-equality is impossible by design: the additive ``plan_hash``
       key (and, in production, the embedded run configs) legitimately
       differ — the adjudication is recorded here.

    Parameters
    ----------
    outdir : Path | str
        Report + artifact output directory.
    check_trials : int, optional
        Trials for the corrupted/restored checker runs, by default 2.
    check_length : int, optional
        Sequence length for the corrupted/restored checker runs, by
        default 11.
    float_atol : float, optional
        Checker float bar, by default `FLOAT_ATOL`.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)`` — 0 only if every control behaved.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("=" * 96)
    print("O5 negative controls + metadata — corrupted weights, name rule, gnn_config vs v1")
    print("=" * 96)
    pair = _build_gn2_pair(outdir / "export", with_v1_export=True)

    # control (1): corrupted weights must FAIL the checker, restored must PASS
    head_bias = list(pair.modules["jets_classification"].task.parameters())[-1]
    original = head_bias.detach().clone()
    with torch.no_grad():
        head_bias[0] += 1.0
    corrupted = check_onnx(
        pair.v2.adapter,
        pair.v2.onnx_path,
        trials=check_trials,
        float_rtol=float_atol,
        float_atol=float_atol,
        lengths_grid=[{"tracks": check_length}],
    )
    with torch.no_grad():
        head_bias.copy_(original)
    restored = check_onnx(
        pair.v2.adapter,
        pair.v2.onnx_path,
        trials=check_trials,
        float_rtol=float_atol,
        float_atol=float_atol,
        lengths_grid=[{"tracks": check_length}],
    )
    control_weights = {
        "corrupted_check_failed": not corrupted.passed,
        "corrupted_failures_reported": len(corrupted.failures) > 0,
        # every failure message must carry CONTENT (value vs NaN vs zero) —
        # assert_allclose strings begin with a newline, and the old
        # first-line truncation blanked them (M4-review fix, check_onnx)
        "corrupted_failures_nonempty": all(f.strip() for f in corrupted.failures),
        "restored_check_passed": restored.passed,
    }

    # control (2): the model-name rule (export-time only, design §7 "Naming")
    error_message = ""
    try:
        resolve_export_config(gn2_export_config(model_name="GN2_v2"), RUN_NAME)
    except ConfigError as err:
        error_message = str(err)
    bad_path = outdir / "rejected.onnx"
    export_rejected = False
    try:
        export_graph(
            pair.modules,
            gn2_export_config(model_name="GN2_v2"),
            pair.variables,
            bad_path,
            outputs=writer_manifest(pair.modules, ("jets", "tracks"), ("tracks",)),
            run_name=RUN_NAME,
        )
    except ConfigError:
        export_rejected = True
    control_name = {
        "config_error_raised": bool(error_message),
        "message_names_offender": "GN2_v2" in error_message,
        "message_states_rule": "underscores or dashes" in error_message,
        "message_cites_v1_rule": "to_onnx.py:169-170" in error_message,
        "export_graph_rejects_before_writing": export_rejected and not bad_path.exists(),
        "default_name_strips_run_name": (
            resolve_export_config(gn2_export_config(model_name=None), RUN_NAME).model_name
            == MODEL_NAME
        ),
    }

    # control (3): gnn_config vs v1 (shared metadata inputs — _build_gn2_pair)
    assert pair.v1_onnx is not None
    v1_meta, v1_doc = _read_gnn_config(pair.v1_onnx)
    v2_meta, v2_doc = _read_gnn_config(pair.v2.onnx_path)
    key_records: list[dict[str, Any]] = []
    for key in V1_GNN_KEYS:
        in_both = key in v1_meta and key in v2_meta
        equal = in_both and json.dumps(v1_meta[key]) == json.dumps(v2_meta[key])
        justified = (not equal) and key == "salt_export_hash"
        key_records.append({
            "key": key,
            "equal": bool(equal),
            "justified": justified,
            "note": HASH_JUSTIFICATION if justified else "",
            "v1": _preview(v1_meta.get(key)),
            "v2": _preview(v2_meta.get(key)),
        })
    # raw STORED bytes (no JSON round-trip): the v2 string must be a literal
    # byte prefix-extension of v1's — v1's full payload minus its closing
    # brace, then exactly the additive plan_hash entry (M4-review fix: the
    # per-key parse-redump check above would mask non-semantic byte
    # differences). salt_export_hash is value-normalised first when it is
    # the (single, justified) differing key — HASH_JUSTIFICATION.
    v1_raw = _read_gnn_config_raw(pair.v1_onnx)
    v2_raw = _read_gnn_config_raw(pair.v2.onnx_path)
    hash_normalised = json.dumps(v1_meta.get("salt_export_hash")) != json.dumps(
        v2_meta.get("salt_export_hash")
    )
    v1_raw_cmp = v1_raw
    if hash_normalised:
        v1_raw_cmp = v1_raw.replace(
            f'"salt_export_hash": {json.dumps(v1_meta.get("salt_export_hash"))}',
            f'"salt_export_hash": {json.dumps(v2_meta.get("salt_export_hash"))}',
            1,
        )
    expected_v2_raw = v1_raw_cmp[:-1] + f', "plan_hash": {json.dumps(pair.v2.plan.plan_hash)}}}'
    raw_bytes_ok = v1_raw_cmp.endswith("}") and v2_raw == expected_v2_raw
    metadata_checks = {
        "v1_envelope_is_the_13_key_set": list(v1_meta) == list(V1_GNN_KEYS),
        "v2_keys_are_v1_prefix_plus_plan_hash": list(v2_meta) == [*V1_GNN_KEYS, "plan_hash"],
        "athena_subset_byte_equal": _athena_subset_json(v1_meta) == _athena_subset_json(v2_meta),
        "all_v1_keys_equal_or_justified": all(r["equal"] or r["justified"] for r in key_records),
        "raw_bytes_v1_prefix_plus_plan_hash": raw_bytes_ok,
        "doc_strings_equal": v1_doc == v2_doc == MODEL_NAME,
        "plan_hash_recorded": v2_meta.get("plan_hash") == pair.v2.plan.plan_hash,
    }

    passed = (
        all(control_weights.values())
        and all(control_name.values())
        and all(metadata_checks.values())
    )
    criterion = (
        "corrupted weights (asymmetric output-bias element) FAIL the checker with NON-EMPTY "
        "failure messages and the restored weights PASS again; export.model_name 'GN2_v2' "
        "rejected by a named ConfigError citing the v1 rule before any file is written (run "
        "names stay unrestricted); gnn_config: v1 13-key envelope reproduced as an ordered "
        "prefix + single additive trailing plan_hash (verified on the RAW stored bytes), "
        "Athena-visible subset byte-equal, every v1 key byte-equal with salt_export_hash the "
        "only justified exception (reported)"
    )
    report = _base_report(
        "o5_negative_controls_metadata",
        passed,
        criterion,
        {
            "check_trials": check_trials,
            "check_length": check_length,
            "float_atol": float_atol,
            "v1_onnx": str(pair.v1_onnx),
            "v2_onnx": str(pair.v2.onnx_path),
            "v1_git_hash_patched": pair.v1_git_hash_patched,
        },
    )
    report["control_corrupted_weights"] = dict(control_weights)
    report["corrupted_check_failures"] = corrupted.failures[:10]
    report["control_model_name"] = dict(control_name)
    report["model_name_error"] = error_message
    report["metadata_checks"] = metadata_checks
    report["metadata_keys"] = key_records
    report["raw_byte_comparison"] = {
        "v1_bytes": len(v1_raw),
        "v2_bytes": len(v2_raw),
        "salt_export_hash_normalised": hash_normalised,
        "v2_equals_v1_prefix_plus_plan_hash_entry": raw_bytes_ok,
    }
    report["metadata_adjudication"] = (
        "whole-envelope byte-equality is impossible by design (additive plan_hash; production "
        "config.yaml payloads differ per run) — the gate asserts the v1-keys-as-ordered-prefix "
        "+ Athena-subset byte-equality form (plan 07 stage-A adjudication)"
    )

    print("control (1): corrupted weights")
    _print_checks(control_weights)
    print("control (2): model-name rule")
    print(f"  error: {error_message}")
    _print_checks(control_name)
    print("control (3): gnn_config vs v1")
    print(f"{'key':<22}{'status':<12}v1 / v2")
    print("-" * 96)
    for record in key_records:
        if record["equal"]:
            status = "EQUAL"
        elif record["justified"]:
            status = "JUSTIFIED"
        else:
            status = "FAIL"
        print(f"{record['key']:<22}{status:<12}{record['v1']} / {record['v2']}")
        if record["note"]:
            print(f"{'':<34}note: {record['note']}")
    _print_checks(metadata_checks)
    _print_verdict("o5", passed, criterion, _emit_report(report, outdir, "o5"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Add the arguments shared by every gate subcommand."""
    parser.add_argument("--outdir", type=Path, required=True, help="report output directory")


def _csv_ints(value: str) -> list[int]:
    """Split a comma-separated CLI value into ints.

    Returns
    -------
    list[int]
        The parsed values.
    """
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _build_parser() -> argparse.ArgumentParser:
    """Build the gate subcommand parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with ``o1``-``o5`` subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="python -m salt.tests.integration.gates_m4", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="gate", required=True)

    o1 = sub.add_parser("o1", help="v2-torch vs v2-ONNX <= 1e-6 over the full L sweep incl. L=0")
    _add_common(o1)
    o1.add_argument("--trials", type=int, default=10, help="draws per length (v1 check.py:177)")
    o1.add_argument("--max-length", type=int, default=40, help="sweep L=0..N-1 (check.py:176)")
    o1.add_argument("--float-atol", type=float, default=FLOAT_ATOL)

    o2 = sub.add_parser("o2", help="v1-ONNX vs v2-ONNX output identity + IO/metadata contract")
    _add_common(o2)
    o2.add_argument("--trials", type=int, default=3)
    o2.add_argument("--max-length", type=int, default=40)
    o2.add_argument("--float-atol", type=float, default=FLOAT_ATOL)

    o3 = sub.add_parser("o3", help="two dynamic axes: (L_trk, L_el) grid incl. zeros (risk 7)")
    _add_common(o3)
    o3.add_argument("--trials", type=int, default=3)
    o3.add_argument("--track-lengths", type=_csv_ints, default=[0, 1, 2, 7, 21, 39])
    o3.add_argument("--electron-lengths", type=_csv_ints, default=[0, 1, 4, 11])
    o3.add_argument("--float-atol", type=float, default=FLOAT_ATOL)

    o4 = sub.add_parser("o4", help="VertexIndex triple identity (union-find in-graph proof)")
    _add_common(o4)
    o4.add_argument("--trials", type=int, default=3)
    o4.add_argument("--max-length", type=int, default=40)

    o5 = sub.add_parser("o5", help="negative controls + gnn_config metadata vs v1")
    _add_common(o5)
    o5.add_argument("--check-trials", type=int, default=2)
    o5.add_argument("--check-length", type=int, default=11)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one M4 gate from the command line.

    Returns
    -------
    int
        0 if the gate passed, 1 otherwise.
    """
    args = _build_parser().parse_args(argv)
    if args.gate == "o1":
        code, _ = run_o1(
            args.outdir, trials=args.trials, max_length=args.max_length, float_atol=args.float_atol
        )
    elif args.gate == "o2":
        code, _ = run_o2(
            args.outdir, trials=args.trials, max_length=args.max_length, float_atol=args.float_atol
        )
    elif args.gate == "o3":
        code, _ = run_o3(
            args.outdir,
            trials=args.trials,
            track_lengths=args.track_lengths,
            electron_lengths=args.electron_lengths,
            float_atol=args.float_atol,
        )
    elif args.gate == "o4":
        code, _ = run_o4(args.outdir, trials=args.trials, max_length=args.max_length)
    else:
        code, _ = run_o5(
            args.outdir, check_trials=args.check_trials, check_length=args.check_length
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
