"""``salt2 convert-config`` — translate a legacy v1 salt YAML config to the v2
``salt.core`` schema (M7 sub-wave W1; plan 13 W1b; matrix §4 / FD §9.3).

The converter is the automated counterpart of the hand-written v2 fixtures in
``salt/core/configs/`` — for every shipped v1 config there is a v2-native twin,
and this module reproduces that mapping mechanically. The output need NOT be
byte-identical to the fixture, but it MUST be SEMANTICALLY EQUIVALENT: it
validates ``salt2 graph validate`` (data-free, non-strict — see ``_self_validate``
for why ``--strict`` cannot be satisfied without a real norm_dict + schema
artifact) and plan-compiles to the same graph (same modules, same plan hash where
deterministic, same eval columns + ONNX manifest).

Behaviour (FD §9.3):

1. Resolve the DECLARED v1 stack (a base config + zero-or-more overlay files,
   deep-merged left-to-right exactly as ``salt fit`` / jsonargparse would —
   ``salt/configs/base.yaml`` is auto-prepended). v1 list-replacement task
   semantics are honoured: a later file's ``tasks`` list REPLACES the earlier
   one (jsonargparse list semantics).
2. Run the data-free half of the v1 ``before_instantiate_classes`` injections
   (the parts that do NOT need a train file): collect labels, default
   ``global_object``, resolve ``class_names`` via the v1 ``CLASS_NAMES`` lookup
   (``utils/class_names.py``) + the loss/output-size width, etc.
3. Translate the resolved v1 namespace into the v2 modules dict + data modules
   + writers + export block.
4. HARD-ERROR (never approximate, FD §10) on bit-rotted / unconvertible idioms:
   the removed v1 classes (``TransformerEncoder`` / ``ScaledDotProductAttention``
   / ``TransformerCrossAttentionEncoder``), ``salt.models.R21Xbb``,
   ``merge_dict`` with a task on the merged stream, ``aux_loss: true`` on a
   MaskDecoder, and any construct it cannot prove equivalent — each with a
   ``TODO``-flagged message identifying the config as a drop.
5. Run ``salt2 graph validate`` on the converter's own output (the FD §9.3 final
   self-check), supplying ``norm_dict`` data-free. The check is NON-strict: a
   data-free validation cannot satisfy ``--strict`` (which promotes the inherent
   no-schema / missing-input-type-preflight warnings to errors); ``_self_validate``
   documents this and the rc==0 graph/connectivity pass IS the criterion.

The one v1 idiom with NO data-free target is the GLOBAL ``flavour_label``
classification head's ``class_names``: v1 autodiscovers them from the train
file's H5 attrs (``utils/cli.py:get_object_class_names``). The converter cannot
read attrs without data, so it requires either an explicit ``class_names`` on
the head, an explicit ``--class-names`` mapping, or a ``--train-file`` to read
the attrs; absent all three it HARD-ERRORS with an actionable message rather
than guessing (FD §10 "never approximate").
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from salt.core.graph.errors import GraphError

__all__ = ["ConvertError", "convert_config", "convert_stack", "main"]


# ---------------------------------------------------------------------------
# v1 reference data (mirrors of the live v1 lookups so the converter is itself
# self-contained — it reads YAML, never imports v1)
# ---------------------------------------------------------------------------

# salt/utils/class_names.py — the v1 label -> ordered class-name lookup the v1
# ClassificationTask falls back to when class_names is unset (task.py:130-131).
_V1_CLASS_NAMES: dict[str, list[str]] = {
    "ftagTruthOriginLabel": [
        "Pileup",
        "Fake",
        "Primary",
        "FromB",
        "FromBC",
        "FromC",
        "FromTau",
        "OtherSecondary",
    ],
    "ftagTruthTypeLabel": ["NoTruth", "Other", "Pion", "Kaon", "Electron", "Muon"],
    "ftagTruthSourceLabel": [
        "NoTruth",
        "NotSecondary",
        "HadronicInteraction",
        "StrangeMesonDecay",
        "StrangeBaryonDecay",
        "GammaConversion",
        "Other",
    ],
}

# salt/models/__init__.py __all__ at 67d496e — the classes still importable from
# salt.models. A salt.models.<X> class_path naming anything else is bit-rot.
_V1_MODELS_EXPORTS = frozenset({
    "ClassificationTask",
    "Dense",
    "EdgeConstructor",
    "FeaturewiseTransformation",
    "GaussianRegressionTask",
    "GlobalAttentionPooling",
    "InitNet",
    "InputNorm",
    "MaskFormerLoss",
    "NodeQueryGAP",
    "Pooling",
    "PositionalEncoder",
    "R21Xbb",
    "RegressionTask",
    "RegressionTaskBase",
    "SaltModel",
    "TaskBase",
    "Transformer",
    "TransformerV2",
    "VertexingTask",
})

# The three v1 classes REMOVED from salt.models since (configs referencing them
# fail jsonargparse on v1 main today) — FD §10 / matrix §5.1 bit-rot markers.
_REMOVED_V1_CLASSES = frozenset({
    "TransformerEncoder",
    "ScaledDotProductAttention",
    "TransformerCrossAttentionEncoder",
})


class ConvertError(GraphError):
    """A v1 config cannot be converted — bit-rotted, parked, or unprovable.

    A `GraphError` subclass so ``salt2``'s top-level handler prints it as the
    clean one-block ``salt.core.convert.ConvertError: ...`` form (no traceback),
    exactly like the graph-tooling config errors.
    """


# ---------------------------------------------------------------------------
# stack resolution (v1 deep-merge semantics, FD §6.1)
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge `overlay` onto `base` with v1 jsonargparse semantics.

    Dicts merge key-by-key (later wins); every other leaf — INCLUDING lists —
    is replaced wholesale. The list-replacement rule is load-bearing: a v1
    overlay's ``tasks`` list REPLACES the base's task list (matrix §4.1), and
    so does ``variables`` / ``selections`` etc.

    Returns
    -------
    dict[str, Any]
        A new merged dict (inputs are not mutated).
    """
    out = copy.deepcopy(base)
    for key, val in overlay.items():
        if isinstance(val, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], val)  # type: ignore[arg-type]
        else:
            out[key] = copy.deepcopy(val)
    return out


_ANCHOR_RE = re.compile(r"&([A-Za-z0-9_-]+)\b")
_ALIAS_RE = re.compile(r"\*([A-Za-z0-9_-]+)\b")


def _uniquify_anchors(text: str) -> str:
    """Rewrite REDEFINED YAML anchors to be unique (each alias -> nearest prior).

    Some shipped v1 configs redefine an anchor (``hitz.yaml`` declares
    ``&out_dim`` twice) — illegal YAML that strict parsers reject, but which v1
    ships. Pass over the text line-by-line: each ``&name`` (2nd+ occurrence) is
    renamed ``name__dupN`` and every following ``*name`` is repointed to the most
    recent definition, exactly reproducing the "nearest prior anchor" resolution
    PyYAML would use if the anchors were distinct.

    Returns
    -------
    str
        The de-duplicated YAML text.
    """
    seen: dict[str, int] = {}
    current: dict[str, str] = {}  # base name -> current (possibly renamed) anchor
    out_lines: list[str] = []
    for line in text.splitlines():
        # repoint aliases on this line to the current binding FIRST (an alias
        # uses the latest anchor seen above it)
        def _repoint(m: re.Match[str]) -> str:
            base = m.group(1)
            return "*" + current.get(base, base)

        line = _ALIAS_RE.sub(_repoint, line)  # noqa: PLW2901 - intentional rebind

        def _rename(m: re.Match[str]) -> str:
            base = m.group(1)
            seen[base] = seen.get(base, 0) + 1
            new = base if seen[base] == 1 else f"{base}__dup{seen[base]}"
            current[base] = new
            return "&" + new

        line = _ANCHOR_RE.sub(_rename, line)  # noqa: PLW2901 - intentional rebind
        out_lines.append(line)
    return "\n".join(out_lines)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML config file into a plain dict.

    PyYAML ``safe_load`` is tried first; on a duplicate-anchor `ComposerError`
    (the malformed-but-shipped ``hitz.yaml``, two ``&out_dim`` anchors) the text
    is de-duplicated (`_uniquify_anchors`, nearest-prior resolution) and retried.

    Returns
    -------
    dict[str, Any]
        The parsed mapping (empty dict for an empty file).

    Raises
    ------
    ConvertError
        When the file does not parse to a mapping.
    """
    text = path.read_text()
    try:
        raw = yaml.safe_load(text)
    except yaml.composer.ComposerError:
        raw = yaml.safe_load(_uniquify_anchors(text))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConvertError(f"{path}: top-level YAML is not a mapping (got {type(raw).__name__})")
    return dict(raw)


def _find_base_yaml(first: Path) -> Path | None:
    """Locate the v1 auto-loaded ``base.yaml`` for `first` (the v1 default config).

    v1 ALWAYS loads ``salt/configs/base.yaml`` for every ``salt fit``/``test``,
    regardless of the user config's subdirectory (``main.py``: the package
    ``configs/`` dir). So a ``configs/legacy/dips.yaml`` is auto-loaded WITH
    ``configs/base.yaml`` (one dir up), not ``configs/legacy/base.yaml``. Walk
    up from the config dir to the nearest ``configs`` ancestor and use its
    ``base.yaml``; fall back to the config's own-dir ``base.yaml``.

    Returns
    -------
    Path | None
        The base.yaml path, or None when none exists.
    """
    for parent in (first.parent, *first.parents):
        if parent.name == "configs":
            candidate = parent / "base.yaml"
            return candidate if candidate.is_file() else None
    sibling = first.parent / "base.yaml"
    return sibling if sibling.is_file() else None


def resolve_stack(paths: Sequence[Path], *, with_base: bool = True) -> dict[str, Any]:
    """Resolve a v1 declared config stack into one merged namespace.

    `paths` are the user-declared configs in stack order (base first, most
    specific last). When `with_base` is set the auto-loaded ``base.yaml`` (the
    package ``configs/base.yaml`` — the v1 ``salt fit`` default-config
    mechanism, ``main.py``) is prepended if found.

    Returns
    -------
    dict[str, Any]
        The deep-merged v1 config namespace.

    Raises
    ------
    ConvertError
        When a path is missing.
    """
    if not paths:
        raise ConvertError("convert-config: at least one v1 config path is required")
    files: list[Path] = []
    if with_base:
        base = _find_base_yaml(paths[0])
        # avoid a double-merge when base.yaml is itself the declared first config
        if base is not None and base.resolve() != paths[0].resolve():
            files.append(base)
    files.extend(paths)
    merged: dict[str, Any] = {}
    for path in files:
        if not path.is_file():
            raise ConvertError(f"convert-config: config not found: {path}")
        merged = _deep_merge(merged, _load_yaml(path))
    return merged


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _class_tail(class_path: str | None) -> str:
    """The final dotted component of a ``class_path`` (e.g. ``Transformer``).

    Returns
    -------
    str
        The class name, or ``""`` for a falsy input.
    """
    return class_path.rsplit(".", 1)[-1] if class_path else ""


def _embed_name(stream: str) -> str:
    """The StreamEmbed instance name for `stream` — the v2-fixture convention.

    The fixtures name the embed ``<stream singular>_embed`` (``tracks`` ->
    ``track_embed``, ``electrons`` -> ``electron_embed``, ``flow`` ->
    ``flow_embed``), so the converter strips ONE trailing ``s``. This is an
    instance NAME only (it does not change the graph wiring — the stream still
    feeds the encoder by its real name) but it makes the converter's plan hash
    match the hand-written fixture for the single-track-stream configs.

    Returns
    -------
    str
        ``<stream without a trailing 's'>_embed``.
    """
    return f"{stream.removesuffix('s')}_embed"


def _check_bit_rot(class_path: str | None, *, where: str, cfg_name: str) -> None:
    """Hard-error if `class_path` names a removed/parked v1 class (FD §10).

    Raises
    ------
    ConvertError
        For the three removed encoder/attention classes, ``salt.models.R21Xbb``,
        or any other unknown ``salt.models.*`` name (bit-rot).
    """
    tail = _class_tail(class_path)
    if tail in _REMOVED_V1_CLASSES:
        raise ConvertError(
            f"TODO(drop): config {cfg_name!r} references the REMOVED v1 class "
            f"salt.models.{tail} at {where} — this class no longer exists in salt.models on v1 "
            "main, so the config is bit-rotted and cannot run on v1 (matrix §5.1). DROP it "
            "(GN2_open_data / GN2_tracks_neutral_CA|SA_aux / GN2_dR / SubjetXbb) or rewrite the "
            "encoder against salt.models.Transformer (the v2 TransformerEncoder)."
        )
    if tail == "R21Xbb":
        raise ConvertError(
            f"TODO(drop): config {cfg_name!r} uses salt.models.R21Xbb at {where} — the R21-era "
            "Xbb MLP is PARKED (design §10) and has no v2 module. DROP it (legacy/Baseline_Xbb / "
            "legacy/SubjetXbb)."
        )
    if class_path and class_path.startswith("salt.models.") and tail not in _V1_MODELS_EXPORTS:
        raise ConvertError(
            f"TODO(drop): config {cfg_name!r} references unknown class salt.models.{tail} at "
            f"{where} — not importable from salt.models on v1 main (bit-rot). DROP or rewrite."
        )


def _as_list(value: Any) -> list[Any]:
    """Coerce a scalar-or-list into a list (None -> []).

    Returns
    -------
    list[Any]
        ``[]`` for None, the value itself for a list/tuple, else ``[value]``.
    """
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def _loss_block(v1_loss: Any) -> Any:
    """Translate a v1 ``loss:`` value to the v2 task ``loss:`` surface.

    v1 accepts a bare ``torch.nn`` class name (``MSELoss``), a class-path
    string (``torch.nn.CrossEntropyLoss``), or a ``{class_path, init_args}``
    block. v2 accepts the same forms — the only normalisation is keeping the
    block intact (so ``BCEWithLogitsLoss(reduction=none)`` survives).

    Returns
    -------
    Any
        The v2 loss value (string or ``{class_path, init_args}`` mapping), or
        None to fall through to the v2 task default.

    Raises
    ------
    ConvertError
        On a loss spec that is neither a string nor a mapping.
    """
    if v1_loss is None:
        return None
    if isinstance(v1_loss, str):
        return _qualify_loss(v1_loss)
    if isinstance(v1_loss, Mapping):
        block = dict(v1_loss)
        if isinstance(block.get("class_path"), str):
            block["class_path"] = _qualify_loss(block["class_path"])
        return block
    raise ConvertError(f"unsupported loss spec: {v1_loss!r}")


def _qualify_loss(name: str) -> str:
    """Qualify a bare ``torch.nn`` loss name (v1 accepts ``MSELoss``).

    v1 resolves a bare loss class name against ``torch.nn`` (and a class-path is
    passed through); v2's loss adapter wants a dot-import path for the block
    ``class_path``, so a bare ``MSELoss`` -> ``torch.nn.MSELoss``.

    Returns
    -------
    str
        The dotted loss class path (unchanged when already dotted).
    """
    return name if "." in name else f"torch.nn.{name}"


# ---------------------------------------------------------------------------
# task conversion
# ---------------------------------------------------------------------------


def _resolve_class_names(
    task: Mapping[str, Any],
    *,
    global_object: str,
    cfg_name: str,
    train_attr_names: Mapping[str, list[str]] | None,
) -> list[str]:
    """Resolve the v2-required ``class_names`` for a classification task.

    Priority (mirrors v1 ClassificationTask + the CLI autodiscovery):
      1. explicit ``class_names`` in the v1 task config;
      2. the v1 ``CLASS_NAMES[label]`` lookup (track_origin/type/source);
      3. for the GLOBAL ``flavour_label`` head only: H5 attrs read from a
         supplied ``--train-file`` (the v1 autodiscovery, cli.py:436-443).

    Returns
    -------
    list[str]
        The ordered class names.

    Raises
    ------
    ConvertError
        When the global ``flavour_label`` head's names cannot be resolved
        data-free (no explicit list, no train-file attrs) — never guessed.
    """
    init = task["init_args"]
    explicit = init.get("class_names")
    if explicit:
        return list(explicit)
    label = init.get("label")
    if label in _V1_CLASS_NAMES:
        return list(_V1_CLASS_NAMES[label])
    # the global flavour head: v1 reads names from the train file's h5 attrs
    name = init.get("name")
    if name == f"{global_object}_classification" and label == "flavour_label":
        if train_attr_names and label in train_attr_names:
            return list(train_attr_names[label])
        raise ConvertError(
            f"TODO(class_names): config {cfg_name!r} task {name!r} relies on v1 H5-attr "
            "autodiscovery for its flavour class_names (cli.py:436-443) — unrecoverable from the "
            "config alone. Supply them via an explicit `class_names:` on the v1 head, or pass "
            "`--train-file <train.h5>` so the converter can read the "
            f"'{label}' attr of the '{global_object}' group, or `--class-names "
            f"{name}=bjets,cjets,...`."
        )
    raise ConvertError(
        f"TODO(class_names): config {cfg_name!r} classification task {name!r} (label {label!r}) "
        "has no class_names and no v1 CLASS_NAMES fallback — specify class_names explicitly."
    )


def _convert_task(
    task: Mapping[str, Any],
    *,
    global_object: str,
    cfg_name: str,
    pooled_key: str,
    context_key: str,
    has_encoder: bool,
    train_attr_names: Mapping[str, list[str]] | None,
) -> tuple[str, dict[str, Any]]:
    """Convert one v1 task module into a (instance-name, v2 module) pair.

    The v1 ``name``/``input_name`` couple becomes the v2 instance name + the
    ``stream`` + ``input``/``context`` wiring. A GLOBAL task (``input_name`` ==
    ``global_object``) reads the pooled rep as ``input``; a per-token task on a
    track stream reads ``encoded.<stream>`` (the default), with the pooled rep
    as ``context`` ONLY when the v1 head declared ``context_size`` (the v1 Dense
    consumes context only then — dense.py:91-94; the encoder-less no-context-size
    track heads receive but DROP the passed context).

    Returns
    -------
    tuple[str, dict[str, Any]]
        The instance name and the v2 module dict.

    Raises
    ------
    ConvertError
        On an unrecognised task class or unconvertible regression idiom.
    """
    init = dict(task["init_args"])
    cls = _class_tail(task["class_path"])
    name = init["name"]
    input_name = init["input_name"]
    dense_cfg = dict(init.get("dense_config") or {})
    # v2 Dense kwargs are the v1 dense_config MINUS the width keys (inferred at
    # bind, design §2.3): input_size/output_size/context_size are dropped.
    dense = {
        k: v
        for k, v in dense_cfg.items()
        if k not in {"input_size", "output_size", "context_size", "mup"}
    }
    is_global = input_name == global_object
    module: dict[str, Any] = {"stream": input_name}

    if is_global:
        module["input"] = pooled_key
    else:
        # per-token head on a constituent stream. With an encoder the v2 default
        # input is encoded.<stream> (the Split output); WITHOUT one (encoder-less
        # DiPS/regression body) it reads embed.<stream> (the StreamEmbed output,
        # rank-3 [B, T, D]) directly — there is no encoded.* key. An explicit
        # input forces sequence inference off, so set sequence:true to keep the
        # per-token semantics.
        if not has_encoder:
            module["input"] = f"embed.{input_name}"
            module["sequence"] = True
        # the pooled rep as context — ONLY when the v1 head declared context_size.
        # v1 SaltModel.run_tasks (saltmodel.py:221-222) ALWAYS PASSES the pooled
        # global_rep as `context` to every per-token (non-global, non-objects) task,
        # but the v1 Dense head only CONSUMES it when context_size is truthy
        # (dense.py:91-94: `if self.context_size: x = attach_context(x, context)`).
        # With no context_size the context is silently dropped (and the net width
        # input_size+context_size has no room for it), so injecting `context:
        # pooled.global` here would be UNFAITHFUL — it would wire/consume a context
        # the v1 head never used. The encoder-less no_global_object track heads
        # (regression / regression_weighted / nan_regression) declare no context_size,
        # and their hand-written v2 fixtures correctly omit context.
        if "context_size" in dense_cfg:
            module["context"] = context_key

    weight = init.get("weight")

    if cls == "ClassificationTask":
        module["label"] = init["label"]
        module["class_names"] = _resolve_class_names(
            task,
            global_object=global_object,
            cfg_name=cfg_name,
            train_attr_names=train_attr_names,
        )
        if init.get("label_map") is not None:
            module["label_map"] = init["label_map"]
        if init.get("use_class_dict"):
            # v1 use_class_dict -> v2 weight_source override at fit; the converter
            # emits an explicit null (the fixtures' pattern) so the head reads its
            # weights from the override, never silently unweighted.
            module["weight_source"] = None
        loss = _loss_block(init.get("loss"))
        if loss is not None:
            module["loss"] = loss
        if weight is not None:
            module["weight"] = weight
        if dense:
            module["dense"] = dense
        class_path = "salt.core.nn.tasks.ClassificationTaskModule"

    elif cls == "VertexingTask":
        module["label"] = init["label"]
        # v1 derives the origin key by VertexIndex->OriginLabel string-replace
        # (task.py:937); v2 makes it an explicit declared dependency and REQUIRES
        # the vertex label to contain 'VertexIndex' (the v2 VertexingTaskModule
        # invariant, tasks.py:702-707). A non-standard vertex label (e.g. the
        # tutorial config's `truth_vertex_idx`) has no derivable OriginLabel key,
        # so the converter cannot prove equivalence -> HARD-ERROR with an
        # actionable TODO (FD §10 "never approximate") rather than emit output
        # that fails opaquely at v2 validate time.
        if "VertexIndex" not in init["label"]:
            raise ConvertError(
                f"TODO(manual): config {cfg_name!r} VertexingTask label {init['label']!r} does "
                "not contain 'VertexIndex' — v2's VertexingTaskModule derives the origin-label "
                "column from label.replace('VertexIndex','OriginLabel') (tasks.py:702-707) and "
                "cannot map a non-standard vertex label. Rename the label to the production "
                "'*VertexIndex' scheme (with a matching '*OriginLabel') or drop the vertexing head."
            )
        module["origin_label"] = init["label"].replace("VertexIndex", "OriginLabel")
        # FD §8 / AMD: the converter sets the v1-compat bare-VertexIndex column
        # name (prefix flip is the post-deletion W4 change).
        module["prefix_vertex_column"] = False
        loss = _loss_block(init.get("loss"))
        if loss is not None:
            module["loss"] = loss
        if weight is not None:
            module["weight"] = weight
        if dense:
            module["dense"] = dense
        class_path = "salt.core.nn.tasks.VertexingTaskModule"

    elif cls in {"RegressionTask", "GaussianRegressionTask"}:
        # a pooled global head is a scalar; a per-object (sequence) regression
        # head on a non-global stream (e.g. MaskFormer objects) is retargeted to
        # sequence semantics by the caller (convert_stack MaskFormer path).
        module["targets"] = init["targets"]
        if init.get("target_denominators") is not None:
            module["target_denominators"] = init["target_denominators"]
        if init.get("norm_params") is not None:
            module["norm_params"] = init["norm_params"]
        if init.get("custom_output_names") is not None:
            module["custom_output_names"] = init["custom_output_names"]
        if init.get("sample_weight") is not None:
            module["sample_weight"] = init["sample_weight"]
        scaler = _convert_scaler(init.get("scaler"), cfg_name=cfg_name)
        if scaler is not None:
            module["scaler"] = scaler
        if cls == "GaussianRegressionTask":
            module["gaussian"] = True
        loss = _loss_block(init.get("loss"))
        if loss is not None:
            module["loss"] = loss
        if weight is not None:
            module["weight"] = weight
        if dense:
            module["dense"] = dense
        class_path = "salt.core.nn.tasks.RegressionTaskModule"
    else:
        raise ConvertError(
            f"TODO(task): config {cfg_name!r} task {name!r} has unsupported class "
            f"{task['class_path']!r} — only Classification/Vertexing/Regression heads convert."
        )

    return name, {"class_path": class_path, "init_args": module}


def _convert_scaler(scaler: Any, *, cfg_name: str) -> dict[str, dict[str, Any]] | None:
    """Translate a v1 RegressionTargetScaler block to the v2 per-target scaler.

    v1 spells it ``{class_path: salt.utils.scalers.RegressionTargetScaler,
    init_args: {scales: {<target>: {op, ...}}}}``; v2 inlines the per-target
    ``{<target>: {op, ...}}`` dict directly under ``scaler:``.

    Returns
    -------
    dict[str, dict[str, Any]] | None
        The per-target scaler dict, or None when no scaler is configured.

    Raises
    ------
    ConvertError
        On a RegressionTargetScaler block with no ``scales`` or an
        unrecognised scaler spec.
    """
    if scaler is None:
        return None
    if isinstance(scaler, Mapping) and "class_path" in scaler:
        init = scaler.get("init_args") or {}
        scales = init.get("scales")
        if scales is None:
            raise ConvertError(f"config {cfg_name!r}: RegressionTargetScaler has no 'scales'")
        return {k: dict(v) for k, v in scales.items()}
    if isinstance(scaler, Mapping):  # already the inline per-target form
        return {k: dict(v) for k, v in scaler.items()}
    raise ConvertError(f"config {cfg_name!r}: unsupported scaler spec {scaler!r}")


# ---------------------------------------------------------------------------
# data block conversion
# ---------------------------------------------------------------------------


def _convert_data(
    v1: Mapping[str, Any],
    *,
    cfg_name: str,
) -> dict[str, Any]:
    """Translate the v1 ``data:`` block into the v2 Reader/Features/Labels modules.

    v1 ``variables`` / ``input_map`` / ``num_inputs`` / ``selections`` /
    ``labeller_config`` / ``mf_config`` map onto the v2 data PROCESSORS:
      - ``variables`` -> ``Features.variables`` (the one place column order lives);
      - ``input_map`` -> per-group ``dataset:`` on the reader;
      - ``num_inputs`` -> per-group ``truncate:``;
      - ``selections`` -> the reader's ``selections:`` (ftag cut strings);
      - ``labeller_config`` -> the ``Labels`` processor labeller init_args;
      - ``mf_config`` -> a ``MaskFormerTargets`` processor (handled separately);
      - ``multi_target`` -> a ``MultiTarget`` processor (the conditional
        target-replacement rules, datasets.py:695-739).

    Returns
    -------
    dict[str, Any]
        The v2 ``data:`` block.

    Raises
    ------
    ConvertError
        When ``data.variables`` is empty (nothing to read), or a multi_target
        rule is unconvertible (FD §10).
    """
    variables = dict(v1.get("variables") or {})
    if not variables:
        raise ConvertError(f"config {cfg_name!r}: data.variables is empty — nothing to read")
    input_map = v1.get("input_map") or {}
    num_inputs = v1.get("num_inputs") or {}
    global_object = v1.get("global_object", "jets")

    # the model-visible streams are the variables keys MINUS edge-feature pseudo
    # streams (_edge_features_*); 'global' is a model stream too.
    streams = [s for s in variables if not s.startswith("_edge_features_")]

    groups: dict[str, Any] = {}
    for stream in streams:
        gc: dict[str, Any] = {}
        # global_object: the global object (+ a 'global' vector stream) read as [B, F]
        gc["global_object"] = stream in {global_object, "global"}
        if stream in input_map and input_map[stream] != stream:
            gc["dataset"] = input_map[stream]
        if stream in num_inputs:
            gc["truncate"] = num_inputs[stream]
        groups[stream] = gc

    reader_init: dict[str, Any] = {"groups": groups}
    selections = v1.get("selections")
    if selections:
        # v1 selection strings pass straight through (ftag Cuts); a None/empty
        # value (GN3_baseline_loose) means "no selections" -> drop the key.
        reader_init["selections"] = {k: list(v) for k, v in selections.items() if v}

    features_vars = {s: list(variables[s]) for s in streams}

    labels_init: dict[str, Any] = {"dtype_policy": "int64-for-int"}
    labeller = v1.get("labeller_config")
    if labeller and labeller.get("use_labeller"):
        labels_init["use_labeller"] = True
        labels_init["require_labels"] = bool(labeller.get("require_labels", True))
        if labeller.get("class_names"):
            labels_init["class_names"] = list(labeller["class_names"])

    data: dict[str, Any] = {}
    for key in ("batch_size", "num_workers"):
        if v1.get(key) is not None:
            data[key] = v1[key]
    data["modules"] = {
        "reader": {"class_path": "salt.core.data.H5StructuredReader", "init_args": reader_init},
        "features": {
            "class_path": "salt.core.data.Features",
            "init_args": {"variables": features_vars},
        },
        "labels": {"class_path": "salt.core.data.Labels", "init_args": labels_init},
    }
    # v1 top-level data.multi_target -> the MultiTarget processor (the conditional
    # target-replacement rules; the produced custom_target/target is a TRAINING-gated
    # label the regression head consumes). Concrete produce beats the Labels wildcard
    # (planner rule (a)), so it OWNS the built label key — write-once is preserved.
    multi_target = v1.get("multi_target")
    if multi_target:
        data["modules"]["multi_target"] = _multi_target_processor(
            _as_list(multi_target), cfg_name=cfg_name
        )
    return data


def _read_train_attr_names(train_file: Path | None, global_object: str) -> dict[str, list[str]]:
    """Read the global-object class-name attrs from a train file (v1 autodiscovery).

    Mirrors ``utils/cli.py:get_object_class_names`` (cli.py:436-443): open the
    first wildcard match and read ``f[group].attrs[label]``. Only used to fill
    the GLOBAL ``flavour_label`` head's class_names when no explicit list is
    given; returns ``{}`` when no train file is supplied.

    Returns
    -------
    dict[str, list[str]]
        ``{label: [names...]}`` for whatever class attrs the group carries.

    Raises
    ------
    ConvertError
        When the train-file glob matches no file.
    """
    if train_file is None:
        return {}
    import h5py  # noqa: PLC0415 - only when a train file is actually supplied

    matches = sorted(train_file.parent.glob(train_file.name))
    if not matches:
        raise ConvertError(f"--train-file: no file matches {train_file}")
    out: dict[str, list[str]] = {}
    with h5py.File(matches[0]) as f:
        if global_object not in f:
            return {}
        for label, val in f[global_object].attrs.items():
            arr = _attr_to_names(val)
            if arr is not None:
                out[label] = arr
    return out


def _attr_to_names(val: Any) -> list[str] | None:
    """Coerce an H5 array attr to a list of strings, or None if not array-like.

    Returns
    -------
    list[str] | None
        The string list, or None for a scalar/non-iterable attr.
    """
    try:
        return [str(x) for x in val]
    except TypeError:  # not an array attr
        return None


# ---------------------------------------------------------------------------
# model block conversion
# ---------------------------------------------------------------------------


def _convert_encoder(enc: Mapping[str, Any], *, cfg_name: str) -> dict[str, Any]:
    """Translate the v1 ``encoder`` (salt.models.Transformer) to TransformerEncoder.

    v1 ``embed_dim``/``out_dim``/``num_layers``/``norm``/``norm_type``/
    ``num_registers``/``mup`` map onto the v2 args; ``dense_kwargs`` ->
    ``dense:`` and ``attn_kwargs`` + ``attn_type`` -> ``attention:`` (with
    ``num_heads`` REQUIRED). ``do_final_norm`` is dropped (the v2 encoder always
    final-norms). Edge ports (``edge_embed_dim``/``update_edges``) are wired by
    the edge path, not here.

    Returns
    -------
    dict[str, Any]
        The v2 TransformerEncoder module dict (without edge ports).

    Raises
    ------
    ConvertError
        On a bit-rotted encoder class or a missing num_heads.
    """
    _check_bit_rot(enc.get("class_path"), where="encoder", cfg_name=cfg_name)
    if _class_tail(enc.get("class_path")) != "Transformer":
        raise ConvertError(
            f"TODO(drop): config {cfg_name!r} encoder is {enc.get('class_path')!r}, not "
            "salt.models.Transformer — only the modern Transformer encoder converts (bit-rot)."
        )
    init = enc.get("init_args") or {}
    embed_dim = init.get("embed_dim")
    if embed_dim is None:
        raise ConvertError(f"config {cfg_name!r}: encoder has no embed_dim")
    attn_kwargs = dict(init.get("attn_kwargs") or {})
    if "num_heads" not in attn_kwargs:
        raise ConvertError(f"config {cfg_name!r}: encoder attn_kwargs has no num_heads")
    attn_type = init.get("attn_type", "torch-math")
    # v1 flash-varlen is a cluster runtime concern; the v2 fixtures pin
    # torch-math for static/CPU validation (flash-varlen rides M6 cluster). The
    # converter preserves the v1 backend so a cluster fit reproduces flash; CPU
    # validation can override via --set.
    attention = {"num_heads": attn_kwargs.pop("num_heads"), "attn_type": attn_type}
    attention.update(attn_kwargs)  # dropout etc.

    out: dict[str, Any] = {
        "dim": embed_dim,
        "num_layers": init.get("num_layers", 4),
        "attention": attention,
    }
    if init.get("out_dim") is not None:
        out["out_dim"] = init["out_dim"]
    dense = dict(init.get("dense_kwargs") or {})
    if dense:
        out["dense"] = dense
    if init.get("norm") is not None:
        out["norm"] = init["norm"]
    if init.get("norm_type") is not None:
        out["norm_type"] = init["norm_type"]
    if init.get("num_registers") is not None:
        out["num_registers"] = init["num_registers"]
    if init.get("drop_registers"):
        out["drop_registers"] = True
    if init.get("mup"):
        out["mup"] = True
    return out


def _mf_object_task(
    tasks: Sequence[Mapping[str, Any]], *, cfg_name: str
) -> Mapping[str, Any] | None:
    """The v1 object-stream (``input_name: objects``) regression task, or None.

    Returns
    -------
    Mapping[str, Any] | None
        The single object-regression task, or None when the config has none.

    Raises
    ------
    ConvertError
        When more than one object-stream task is declared (unsupported).
    """
    objs = [t for t in tasks if t["init_args"].get("input_name") == "objects"]
    if not objs:
        return None
    if len(objs) > 1:
        raise ConvertError(f"config {cfg_name!r}: more than one objects-stream task is unsupported")
    return objs[0]


def _convert_maskformer(
    modules: dict[str, Any],
    *,
    mask_decoder: Mapping[str, Any],
    embed_dim: int,
    cfg_name: str,
    object_task: Mapping[str, Any] | None,
    data_block: dict[str, Any],
) -> None:
    """Add the MaskFormer bundle (decoder + object regression + matched loss).

    Emits, IN the v2-fixture declaration order (the topological tie-break,
    design §3.1): the ``MaskDecoder`` (v1 ``mask_decoder``), then the
    object-regression head (v1 ``input_name: objects`` RegressionTask, retargeted
    to read ``objects.embed`` + publish scaled targets), then the
    ``MaskFormerMatchedLoss`` (v1 ``loss_config`` num_classes + loss_weights).
    The matched-loss owns the object-class CE + mask + matched-regression losses.

    Raises
    ------
    ConvertError
        On ``aux_loss: true`` (the §10 deep-supervision parking, FD §10).
    """
    md_init = mask_decoder.get("init_args") or {}
    if md_init.get("aux_loss"):
        raise ConvertError(
            f"TODO(drop): config {cfg_name!r} MaskDecoder sets aux_loss: true — deep-supervision "
            "auxiliary losses are PARKED (design §10; FD §10) and have no v2 module. Set "
            "aux_loss: false or drop the config."
        )
    md_config = dict(md_init.get("md_config") or {})
    class_net = md_init.get("class_net") or {}
    class_out = (class_net.get("init_args") or {}).get("output_size")
    decoder: dict[str, Any] = {
        "input": "encoded.seq",
        "embed_dim": embed_dim,
        "num_objects": md_init["num_objects"],
        "num_layers": md_init["num_layers"],
        "md": {
            "n_heads": md_config.get("n_heads", 8),
            "mask_attention": md_config.get("mask_attention", True),
            "bidirectional_ca": md_config.get("bidirectional_ca", True),
        },
        "class_net": {"output_size": class_out},
        "mask_net": {},
    }
    modules["mask_decoder"] = {
        "class_path": "salt.core.nn.MaskDecoder",
        "init_args": decoder,
    }
    # the object-regression head (decoder-adjacent, before the matched loss)
    if object_task is not None:
        name, mod = _convert_task(
            object_task,
            global_object="<none>",  # objects is never the global object
            cfg_name=cfg_name,
            pooled_key="pooled.global",
            context_key="pooled.global",
            has_encoder=True,  # the decoder requires an encoder; input overridden below
            train_attr_names=None,
        )
        mod["init_args"]["stream"] = "objects"
        mod["init_args"]["input"] = "objects.embed"
        mod["init_args"]["sequence"] = True
        mod["init_args"]["publish_targets"] = True
        mod["init_args"]["expose"] = ["fit", "val", "onnx"]
        mod["init_args"].pop("context", None)  # per-object head: no pooled context
        modules[name] = mod
        tgt = _as_list(object_task["init_args"]["targets"])
        data_block["modules"]["mf_targets"]["init_args"]["regression_targets"] = tgt
    loss_config = dict(md_init.get("loss_config") or {})
    matched: dict[str, Any] = {
        "num_classes": loss_config.get("num_classes", 2),
        "num_objects": md_init["num_objects"],
    }
    if loss_config.get("loss_weights"):
        matched["loss_weights"] = dict(loss_config["loss_weights"])
    modules["mf_matched_loss"] = {
        "class_path": "salt.core.nn.MaskFormerMatchedLoss",
        "init_args": matched,
    }


def _mf_targets_processor(mf_config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the v2 ``MaskFormerTargets`` data processor from v1 ``mf_config``.

    Returns
    -------
    dict[str, Any]
        The ``{class_path, init_args}`` MaskFormerTargets module.
    """
    obj = mf_config.get("object") or {}
    con = mf_config.get("constituent") or {}
    class_map: dict[str, Any] = {}
    for name, spec in (obj.get("object_classes") or {}).items():
        key = "null" if name is None else str(name)
        class_map[key] = {"raw": spec["raw"], "mapped": spec["mapped"]}
    init: dict[str, Any] = {
        # v1 mf_config.object.name — the READ truth-hadron group (e.g.
        # truth_hadrons), NOT the 'objects' decoder-query alias
        "object_stream": obj["name"],
        "object_class": obj["class_label"],
        "object_id": obj["id_label"],
        "constituent_stream": con["name"],
        "constituent_id": con["id_label"],
        "class_map": class_map,
    }
    return {"class_path": "salt.core.data.MaskFormerTargets", "init_args": init}


# v1 multi_target rule key -> v2 MultiTarget replacement-rule key. The v1 spelling
# (datasets.py:709-716) differs from the v2 processor's surface (processors.py:586-595):
# v1 'input_name' is the stream and v1 'opp' is the operator; every other key (sel_label,
# value, source, custom_target/target) is named identically.
_MULTI_TARGET_RULE_KEYS: dict[str, str] = {
    "input_name": "stream",
    "opp": "op",
    "sel_label": "sel_label",
    "value": "value",
    "source": "source",
    "custom_target": "custom_target",
    "target": "target",
}


def _multi_target_processor(
    multi_target: Sequence[Mapping[str, Any]], *, cfg_name: str
) -> dict[str, Any]:
    """Build the v2 ``MultiTarget`` data processor from a v1 ``data.multi_target`` block.

    v1's ``multi_target`` (``datasets.py:90/123``; setup ``237-248``; placeholder
    injection ``648-693``; sequential mutation ``695-739``) is a list of conditional
    target-replacement rules applied IN ORDER over a running per-output array
    (``torch.where(op(sel, value), source, running)``). Each rule names a stream
    (``input_name``), a selection label compared by an operator (``opp``) against a
    literal ``value``, the ``source`` label written where the condition holds, and
    exactly one of ``custom_target`` (create a NaN-base new label) or ``target``
    (replace an existing label). The v2 ``MultiTarget`` processor
    (``salt.core.data.MultiTarget``, processors.py:550-719) reproduces this exactly;
    the only rename is ``input_name`` -> ``stream`` and ``opp`` -> ``op``.

    Returns
    -------
    dict[str, Any]
        The ``{class_path, init_args}`` MultiTarget module (module key ``multi_target``).

    Raises
    ------
    ConvertError
        On a rule that is not a mapping, carries an unknown key (FD §10 — never
        silently drop a v1 directive), or sets neither/both of
        ``custom_target``/``target``.
    """
    rules: list[dict[str, Any]] = []
    for raw in multi_target:
        if not isinstance(raw, Mapping):
            raise ConvertError(
                f"config {cfg_name!r}: data.multi_target entry is not a mapping (got {raw!r})"
            )
        unknown = [k for k in raw if k not in _MULTI_TARGET_RULE_KEYS]
        if unknown:
            raise ConvertError(
                f"TODO(multi_target): config {cfg_name!r} data.multi_target rule has unknown "
                f"key(s) {unknown} — not in the v1 rule surface "
                f"{sorted(_MULTI_TARGET_RULE_KEYS)} (datasets.py:709-716). The converter must not "
                "silently drop a v1 directive (FD §10); extend the translation or fix the config."
            )
        has_custom = raw.get("custom_target") is not None
        has_target = raw.get("target") is not None
        if has_custom == has_target:
            raise ConvertError(
                f"TODO(multi_target): config {cfg_name!r} data.multi_target rule must set exactly "
                f"one of 'custom_target' (create a new label) or 'target' (replace an existing "
                f"label) (v1 datasets.py:237-248); got {dict(raw)!r}."
            )
        rule = {_MULTI_TARGET_RULE_KEYS[k]: v for k, v in raw.items()}
        rules.append(rule)
    return {
        "class_path": "salt.core.data.MultiTarget",
        "init_args": {"replacements": rules},
    }


def _convert_edges(
    modules: dict[str, Any],
    encoder: dict[str, Any],
    *,
    edge_constructors: Sequence[Mapping[str, Any]],
    edge_init_nets: Sequence[Mapping[str, Any]],
    enc_init: Mapping[str, Any],
    cfg_name: str,
) -> None:
    """Wire the v1 edge path (edge_constructors + edge_init_nets) onto v2 modules.

    v1 ``data.edge_constructors`` -> an ``EdgeFeatures`` module; v1
    ``edge_init_nets`` -> an ``EdgeEmbed`` module; the encoder gains the
    ``edges``/``edge_embed_dim``/``update_edges`` ports (GN2XE). EdgeAttention
    forces torch-math (the v2 encoder validator rejects flash-varlen + edges).

    Raises
    ------
    ConvertError
        On more than one edge stream (v1 supports exactly one).
    """
    if len(edge_constructors) != 1 or len(edge_init_nets) != 1:
        raise ConvertError(
            f"config {cfg_name!r}: exactly one edge_constructor + one edge_init_net are supported "
            "(v1 saltmodel.py:70); got "
            f"{len(edge_constructors)} / {len(edge_init_nets)}."
        )
    ec = edge_constructors[0]
    stream = ec["input_name"]
    modules["edge_features"] = {
        "class_path": "salt.core.nn.EdgeFeatures",
        "init_args": {"stream": stream, "features": list(ec["edge_features"])},
    }
    ein = edge_init_nets[0]
    ein_dense = dict(ein.get("dense_config") or {})
    edge_out = ein_dense.get("output_size")
    edge_dense = {
        k: v for k, v in ein_dense.items() if k not in {"input_size", "output_size", "context_size"}
    }
    modules["edge_embed"] = {
        "class_path": "salt.core.nn.EdgeEmbed",
        "init_args": {"stream": stream, "out_dim": edge_out, "dense": edge_dense},
    }
    encoder["edges"] = f"edges.{stream}_emb"
    if enc_init.get("edge_embed_dim") is not None:
        encoder["edge_embed_dim"] = enc_init["edge_embed_dim"]
    if enc_init.get("update_edges"):
        encoder["update_edges"] = True
    # EdgeAttention backend forces torch-math (v2 validator; ED2 rule b)
    encoder["attention"]["attn_type"] = "torch-math"


def convert_stack(
    v1: Mapping[str, Any],
    *,
    cfg_name: str,
    train_attr_names: Mapping[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Translate a RESOLVED v1 config namespace into a v2 salt.core config.

    Returns
    -------
    dict[str, Any]
        The v2 config dict (``name``/``data``/``model``/``writers``/``export``/
        ``trainer``), ready to dump + validate.

    Raises
    ------
    ConvertError
        On any bit-rotted / unconvertible idiom (FD §10).
    """
    model_block = v1.get("model") or {}
    inner = model_block.get("model") or {}
    _check_bit_rot(inner.get("class_path"), where="model", cfg_name=cfg_name)
    inner_init = inner.get("init_args") or {}

    # --- guards on parked / deferred idioms (FD §10) -------------------------
    if inner_init.get("featurewise_nets"):
        raise ConvertError(
            f"TODO(defer): config {cfg_name!r} uses featurewise_nets (FiLM) — parameterised/"
            "featurewise training is DEFERRED past M7 (plan 13 W-FILM; matrix §5.2.10 zero shipped "
            "configs). No v2 target yet."
        )
    merge_dict = inner_init.get("merge_dict")
    data = v1.get("data") or {}
    global_object = data.get("global_object", "jets")
    has_global_stream = "global" in (data.get("variables") or {}) or "global" in (
        data.get("input_map") or {}
    )

    init_nets = list(inner_init.get("init_nets") or [])
    edge_constructors = list(data.get("edge_constructors") or [])
    edge_init_nets = list(inner_init.get("edge_init_nets") or [])
    encoder_block = inner_init.get("encoder")
    pool_block = inner_init.get("pool_net")
    mask_decoder = inner_init.get("mask_decoder")
    tasks = list(inner_init.get("tasks", {}).get("init_args", {}).get("modules", []))

    if merge_dict:
        merged_streams = set(merge_dict.keys())
        for t in tasks:
            if t.get("init_args", {}).get("input_name") in merged_streams:
                raise ConvertError(
                    f"TODO(drop): config {cfg_name!r} has a task on a merge_dict-merged stream "
                    f"({t['init_args']['input_name']!r}) — v2 has no merged-stream task path "
                    "(FD §10). Drop or restructure."
                )

    # --- data block ----------------------------------------------------------
    data_block = _convert_data(data, cfg_name=cfg_name)
    # MaskFormer object stream: the truth-hadron group is read as a stream + an
    # mf_targets processor; the v1 'objects' alias is the decoder query bank.
    mf_config = data.get("mf_config")
    if mask_decoder is not None:
        if mf_config is None:
            raise ConvertError(f"config {cfg_name!r}: mask_decoder requires data.mf_config")
        data_block["modules"]["mf_targets"] = _mf_targets_processor(mf_config)
        # num_objects == the decoder query bank (sets the produced label-mask M
        # dimension; v2 fixture mf_targets.num_objects, MaskFormer.yaml)
        md_objects = (mask_decoder.get("init_args") or {}).get("num_objects")
        if md_objects is not None:
            data_block["modules"]["mf_targets"]["init_args"]["num_objects"] = md_objects
        # ensure the object (truth_hadron) group is read
        obj_name = (mf_config.get("object") or {}).get("name")
        if obj_name and obj_name not in data_block["modules"]["reader"]["init_args"]["groups"]:
            data_block["modules"]["reader"]["init_args"]["groups"][obj_name] = {
                "global_object": False
            }

    # --- model modules -------------------------------------------------------
    modules: dict[str, Any] = {}
    norm_streams = [s for s in (data.get("variables") or {}) if not s.startswith("_edge_features_")]
    modules["norm"] = {
        "class_path": "salt.core.nn.Normaliser",
        "init_args": {
            "streams": [s for s in norm_streams if s != "global"],
            "global_object": global_object,
        },
    }
    if has_global_stream:
        modules["norm_global"] = {
            "class_path": "salt.core.nn.Normaliser",
            "init_args": {"streams": ["global"], "global_object": "global"},
        }

    # stream embeds (v1 init_nets -> StreamEmbed) — skip the global-object init_net
    # if it appears (v1 attaches it as context, not its own embed for the encoder
    # streams); skip the 'global' vector stream (it feeds VectorConcat, not embed).
    embed_streams: list[str] = []
    for net in init_nets:
        stream = net["input_name"]
        # the 'global' vector stream (GN3) feeds VectorConcat, not an embed —
        # skip it (a global-object init_net WITHOUT an encoder is the DL1 rank-2
        # MLP and IS embedded — its rank-2-ness comes from the reader's
        # global_object: true flag, NOT a model-side flag).
        if stream == "global":
            continue
        is_global_mlp = stream == global_object and encoder_block is None
        if stream == global_object and not is_global_mlp:
            # the global object under an encoder is attached as CONTEXT to the
            # constituent embeds, not embedded itself (v1 attach_global) — skip.
            continue
        dense_cfg = dict(net.get("dense_config") or {})
        out_dim = dense_cfg.get("output_size")
        dense = {
            k: v
            for k, v in dense_cfg.items()
            if k not in {"input_size", "output_size", "context_size", "mup"}
        }
        embed: dict[str, Any] = {"stream": stream, "out_dim": out_dim, "dense": dense}
        attaches_global = net.get("attach_global", True) and global_object in norm_streams
        if not is_global_mlp and attaches_global:
            embed["context"] = [f"normed.{global_object}"]
        if dense_cfg.get("mup"):
            embed["mup"] = True
        # NOTE: StreamEmbed carries NO rank flag — it infers rank-2 vs rank-3 from
        # its bound input (the DL1 global-object MLP gets [B, F] because the reader
        # group is global_object: true). No model-side vector flag is emitted.
        modules[_embed_name(stream)] = {
            "class_path": "salt.core.nn.StreamEmbed",
            "init_args": embed,
        }
        embed_streams.append(stream)

    # edge path (before concat so EdgeFeatures/EdgeEmbed precede the encoder)
    encoder: dict[str, Any] | None = None
    if encoder_block is not None:
        encoder = _convert_encoder(encoder_block, cfg_name=cfg_name)

    if edge_constructors or edge_init_nets:
        if encoder is None:
            raise ConvertError(f"config {cfg_name!r}: edge features require an encoder")
        _convert_edges(
            modules,
            encoder,
            edge_constructors=edge_constructors,
            edge_init_nets=edge_init_nets,
            enc_init=(encoder_block or {}).get("init_args") or {},
            cfg_name=cfg_name,
        )

    # concat over the embedded (non-global) streams, in v1 init_net order
    concat_streams = [s for s in embed_streams if s not in {global_object, "global"}]
    has_encoder = encoder is not None
    has_split = False
    if concat_streams:
        modules["concat"] = {
            "class_path": "salt.core.nn.Concat",
            "init_args": {"streams": concat_streams},
        }

    if has_encoder:
        modules["encoder"] = {"class_path": "salt.core.nn.TransformerEncoder", "init_args": encoder}
        if mask_decoder is not None:
            embed_dim = (encoder_block.get("init_args") or {}).get("embed_dim")
            # the decoder, then the object-regression head, then the matched loss
            # — the v2-fixture declaration order (the topological tie-break,
            # design §3.1), so the converter's plan hash matches the fixture.
            _convert_maskformer(
                modules,
                mask_decoder=mask_decoder,
                embed_dim=embed_dim,
                cfg_name=cfg_name,
                object_task=_mf_object_task(tasks, cfg_name=cfg_name),
                data_block=data_block,
            )
        # split the encoded seq back per-stream for per-token track heads
        track_task_streams = [
            t["init_args"]["input_name"]
            for t in tasks
            if t["init_args"]["input_name"] in concat_streams
        ]
        split_streams = list(dict.fromkeys(track_task_streams))
        if split_streams:
            modules["split"] = {
                "class_path": "salt.core.nn.Split",
                "init_args": {"streams": split_streams},
            }
            has_split = True

    # pooling
    pool_input = "encoded.seq" if has_encoder else ("seq.x" if concat_streams else None)
    if pool_block is not None:
        if pool_input is None:
            raise ConvertError(f"config {cfg_name!r}: pool_net set but no encoder/concat to pool")
        modules["pool"] = {
            "class_path": "salt.core.nn.GlobalAttentionPooling",
            "init_args": {"input": pool_input, "out": "pooled.global"},
        }

    # the global head's pooled-rep key: pooled.global when a pool exists, else
    # the rank-2 vector embed read directly (the DL1 jets-only MLP — v1 SaltModel
    # asserts pool_net is None for a single global init_net, saltmodel.py:90-93).
    vector_mlp = pool_block is None and not has_encoder and global_object in embed_streams
    pooled_key = f"embed.{global_object}" if vector_mlp else "pooled.global"
    # global stream (VectorConcat) — the pooled rep augmented with global feats
    if has_global_stream:
        modules["vconcat"] = {
            "class_path": "salt.core.nn.VectorConcat",
            "init_args": {"inputs": ["pooled.global", "normed.global"], "out": "vconcat.global"},
        }
        pooled_key = "vconcat.global"

    # tasks (in v1 declaration order); the MaskFormer object-regression head is
    # already emitted by _convert_maskformer (decoder-adjacent) — skip it here.
    for task in tasks:
        if mask_decoder is not None and task["init_args"].get("input_name") == "objects":
            continue
        _check_bit_rot(task.get("class_path"), where="task", cfg_name=cfg_name)
        name, mod = _convert_task(
            task,
            global_object=global_object,
            cfg_name=cfg_name,
            pooled_key=pooled_key,
            context_key=pooled_key,
            has_encoder=has_encoder,
            train_attr_names=train_attr_names,
        )
        modules[name] = mod

    # loss aggregator (v1 loss_mode: wsum -> LossSum, GLS -> LossGLS)
    loss_mode = model_block.get("loss_mode", "wsum")
    loss_cls = "salt.core.nn.LossGLS" if loss_mode == "GLS" else "salt.core.nn.LossSum"
    modules["loss"] = {"class_path": loss_cls}

    # --- assemble model block ------------------------------------------------
    model_init: dict[str, Any] = {}
    lrs = model_block.get("lrs_config")
    if lrs is not None:
        # drop last_epoch (a v1 scheduler internal not in the v2 lrs surface)
        model_init["lrs"] = {k: v for k, v in lrs.items() if k != "last_epoch"}
    optimizer = model_block.get("optimizer", "AdamW")
    model_init["optimizer"] = optimizer
    mup_config = model_block.get("mup_config")
    if mup_config:
        mup_block: dict[str, Any] = {}
        if mup_config.get("shape_path"):
            mup_block["shape_path"] = mup_config["shape_path"]
        # v2 apply_to = the explicit module-name list (the muP-parametrised
        # modules); derived from the mup:true modules we emitted.
        apply_to = [m for m in modules if modules[m].get("init_args", {}).get("mup")]
        mup_block["apply_to"] = apply_to
        model_init["mup"] = mup_block
    model_init["modules"] = modules

    out: dict[str, Any] = {
        "name": v1.get("name", "salt"),
        "data": data_block,
        "model": {"class_path": "salt.core.SaltModule", "init_args": model_init},
    }

    # --- writers (regression-only / object overrides) ------------------------
    writers = _convert_writers(
        tasks,
        mask_decoder=mask_decoder,
        mf_config=mf_config,
        has_split=has_split,
        global_object=global_object,
    )
    if writers is not None:
        out["writers"] = writers

    # --- callbacks (the salt metric callbacks; base2.yaml owns the stock ones)-
    callbacks = _convert_callbacks(v1.get("trainer") or {})
    if callbacks:
        out["callbacks"] = callbacks

    # --- export block --------------------------------------------------------
    out["export"] = _convert_export(
        v1,
        norm_streams=norm_streams,
        global_object=global_object,
        has_global_stream=has_global_stream,
    )

    trainer = _convert_trainer(v1.get("trainer") or {})
    if trainer:
        out["trainer"] = trainer
    return out


# ---------------------------------------------------------------------------
# writers + export + trainer
# ---------------------------------------------------------------------------


def _onnx_collision_tasks(
    tasks: Sequence[Mapping[str, Any]], *, global_object: str
) -> list[str] | None:
    """The narrowed ``onnx_tasks`` when aux heads on >1 stream collide, else None.

    The v1 ONNX output suffix for an aux head is keyed by TASK TYPE, not by
    stream (track_vertexing -> ``VertexIndex``, classification -> ``TrackOrigin``
    etc; to_onnx.py:282-292) — so the SAME aux head on two constituent streams
    (tracks + electrons in GN2emu) collides in the flat ONNX namespace. v1's
    ``tasks_to_output`` CLI arg chose which to export and is UNRECOVERABLE from a
    config (AMD 462-463), so the converter falls back to the v1 DEFAULT surface:
    the global-object tasks + the FIRST non-global constituent stream's heads.

    Returns
    -------
    list[str] | None
        The narrowed ONNX task-name list, or None when no collision exists (the
        default TaskWriter "every task" surface is then safe).
    """
    aux_streams = [
        t["init_args"]["input_name"] for t in tasks if t["init_args"]["input_name"] != global_object
    ]
    distinct = list(dict.fromkeys(aux_streams))
    if len(distinct) <= 1:
        return None  # at most one constituent stream -> no cross-stream collision
    primary = distinct[0]  # v1 init_net order: tracks first
    keep = {global_object, primary}
    return [t["init_args"]["name"] for t in tasks if t["init_args"]["input_name"] in keep]


def _convert_writers(
    tasks: Sequence[Mapping[str, Any]],
    *,
    mask_decoder: Mapping[str, Any] | None,
    mf_config: Mapping[str, Any] | None,
    has_split: bool,
    global_object: str,
) -> dict[str, Any] | None:
    """Compute a v2 ``writers:`` override, or None to use the base2 defaults.

    The base2.yaml defaults (InputCopyWriter -> TaskWriter -> PadMaskWriter)
    cover the common case. Overrides:
      - no per-token (sequence) task -> drop the PadMaskWriter (pad_mask: null);
      - MaskFormer -> narrow the TaskWriter to the jet/track heads + add the
        MaskFormerObjectWriter (the object head is owned by it, not TaskWriter);
      - aux heads on >1 constituent stream collide on the ONNX suffix
        (track_vertexing + electron_vertexing both -> ``_VertexIndex``): narrow
        ``onnx_tasks`` to the global + the FIRST constituent stream (the v1
        default export surface — ``tasks_to_output`` was a CLI arg, unrecoverable
        from a config; AMD lines 462-463/535-537). Eval keeps every head (H5
        groups disambiguate); only ONNX narrows.

    Returns
    -------
    dict[str, Any] | None
        The writers block, or None when the defaults suffice.
    """
    onnx_tasks = _onnx_collision_tasks(tasks, global_object=global_object)
    if onnx_tasks is not None and mask_decoder is None:
        modules: dict[str, Any] = {
            "tasks": {
                "class_path": "salt.core.writers.TaskWriter",
                "init_args": {"onnx_tasks": onnx_tasks},
            }
        }
        if not has_split:
            modules["pad_mask"] = None
        return {"modules": modules}
    if mask_decoder is not None and mf_config is not None:
        obj = mf_config.get("object") or {}
        class_names: list[str] = [
            "null" if name is None else str(name) for name in (obj.get("object_classes") or {})
        ]
        task_names = [
            t["init_args"]["name"] for t in tasks if t["init_args"]["input_name"] != "objects"
        ]
        return {
            "modules": {
                "inputs_copy": {"class_path": "salt.core.writers.InputCopyWriter"},
                "tasks": {
                    "class_path": "salt.core.writers.TaskWriter",
                    "init_args": {"tasks": task_names},
                },
                "pad_mask": {"class_path": "salt.core.writers.PadMaskWriter"},
                "object_writer": {
                    "class_path": "salt.core.writers.MaskFormerObjectWriter",
                    "init_args": {
                        "object_classes": class_names,
                        "object_stream": "objects",
                        "constituent_stream": (mf_config.get("constituent") or {}).get("name"),
                        "regression_task": "regression",
                    },
                },
            }
        }
    if not has_split:
        # no sequence task -> no pad mask to write (regression-only / DL1)
        return {"modules": {"pad_mask": None}}
    return None


def _convert_export(
    v1: Mapping[str, Any],
    *,
    norm_streams: Sequence[str],
    global_object: str,
    has_global_stream: bool,
) -> dict[str, Any]:
    """Build the v2 ``export:`` block (model_name + inputs + global alias).

    ``model_name`` reproduces the v1 default: the run name sanitised
    (``replace('_','').replace('-','')`` — to_onnx.py:687). One ``inputs`` entry
    per encoder/global stream: the global object is a ``[1, F]`` vector, every
    other stream is a ``[L, F]`` sequence (``name_athena_out`` =
    ``<stream minus 's'>_features``, dyn_axis ``n_<stream>`` — to_onnx.py
    get_default_onnx_feature_map). A ``global`` stream is an ALIAS of the global
    object's tensor (no v1 export.outputs — the manifest derives from writers).

    Returns
    -------
    dict[str, Any]
        The export block.
    """
    name = str(v1.get("name", "salt"))
    model_name = name.replace("_", "").replace("-", "")
    inputs: list[dict[str, Any]] = []
    for stream in norm_streams:
        if stream == "global":
            continue
        feat = f"{stream.removesuffix('s')}_features"
        if stream == global_object:
            inputs.append({"port": f"inputs.{stream}", "name": feat})
        else:
            inputs.append({
                "port": f"inputs.{stream}",
                "name": feat,
                "sequence": True,
                "dyn_axis": f"n_{stream}",
            })
    if has_global_stream:
        # the GN3 global alias: one Athena (jet) tensor cloned into the global
        # port (design §6.6/§7; v1 to_onnx.py:377-378)
        inputs.append({"port": "inputs.global", "alias": f"inputs.{global_object}"})
    return {"model_name": model_name, "inputs": inputs}


# v1 salt.callbacks.<X> -> the v2 dict-keyed callback. The stock training/eval
# callbacks (Checkpoint, ProgressBar, LearningRateMonitor, ModelSummary) +
# PredictionWriter are base2.yaml/writers defaults, so only the v1 METRIC
# callbacks (consumed as FIT/VAL graph sinks) need translating.
_V1_METRIC_CALLBACKS: dict[str, tuple[str, str]] = {
    "MaskformerMetrics": ("maskformer_metrics", "salt.core.callbacks.MaskformerMetrics"),
    "ConfusionMatrixCallback": ("confusion_matrix", "salt.core.callbacks.ConfusionMatrix"),
}


def _convert_callbacks(trainer: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the v1 metric callbacks to the v2 dict-keyed ``callbacks:`` block.

    Only the salt METRIC callbacks (MaskformerMetrics / ConfusionMatrixCallback)
    are carried — they are FIT/VAL graph sinks that hold their matched/preds
    producers alive under demand-pruning (so the converted plan matches the
    fixture). The stock Checkpoint/ProgressBar/LR/Summary/PredictionWriter are
    base2.yaml/writers defaults and are skipped.

    Returns
    -------
    dict[str, Any]
        The ``callbacks:`` dict (empty when the config has no metric callbacks).
    """
    out: dict[str, Any] = {}
    for cb in trainer.get("callbacks") or []:
        tail = _class_tail(cb.get("class_path")) if isinstance(cb, Mapping) else ""
        if tail in _V1_METRIC_CALLBACKS:
            key, class_path = _V1_METRIC_CALLBACKS[tail]
            entry: dict[str, Any] = {"class_path": class_path}
            if cb.get("init_args"):
                entry["init_args"] = dict(cb["init_args"])
            out[key] = entry
    return out


def _convert_trainer(trainer: Mapping[str, Any]) -> dict[str, Any]:
    """Carry the safe subset of the v1 ``trainer:`` block.

    Only ``max_epochs`` is carried (accelerator/devices/precision/logger are
    machine/cluster concerns handled by base2.yaml + cluster overrides; v1
    flash/16-mixed defaults do not run on CPU validation). Precision is forced
    to ``32-true`` (the v2 fixtures' CPU-validatable default; cluster overrides
    flip it).

    Returns
    -------
    dict[str, Any]
        The trimmed trainer block.
    """
    out: dict[str, Any] = {}
    if trainer.get("max_epochs") is not None:
        out["max_epochs"] = trainer["max_epochs"]
    out["precision"] = "32-true"
    return out


# ---------------------------------------------------------------------------
# top-level driver + CLI
# ---------------------------------------------------------------------------


def convert_config(
    paths: Sequence[Path],
    *,
    train_file: Path | None = None,
    class_names: Mapping[str, list[str]] | None = None,
    with_base: bool = True,
) -> dict[str, Any]:
    """Resolve + convert a v1 config stack to the v2 schema (no file I/O).

    Propagates `ConvertError` from `resolve_stack` / `convert_stack` /
    `_read_train_attr_names` on any unconvertible idiom (FD §10).

    Returns
    -------
    dict[str, Any]
        The v2 config dict.
    """
    v1 = resolve_stack(paths, with_base=with_base)
    cfg_name = str(v1.get("name") or paths[-1].stem)
    global_object = (v1.get("data") or {}).get("global_object", "jets")
    attr_names: dict[str, list[str]] = {}
    attr_names.update(_read_train_attr_names(train_file, global_object))
    # an explicit --class-names mapping (head=...) overrides — surfaced as a
    # flavour_label attr so _resolve_class_names finds it. The mapping is keyed
    # by task name; we fold the single global flavour head into the attr name.
    explicit = dict(class_names or {})
    if explicit:
        # the global flavour head reads attr 'flavour_label'; map by that key
        for head_name, names in explicit.items():
            if head_name == f"{global_object}_classification":
                attr_names["flavour_label"] = list(names)
    return convert_stack(v1, cfg_name=cfg_name, train_attr_names=attr_names or None)


def _yaml_dump(cfg: Mapping[str, Any]) -> str:
    """Dump a v2 config dict to YAML (block style, key order preserved).

    Returns
    -------
    str
        The YAML text with a provenance header.
    """
    body = yaml.safe_dump(dict(cfg), sort_keys=False, default_flow_style=False, width=100)
    header = (
        "# AUTO-GENERATED by `salt2 convert-config` (salt.core.convert; M7 W1).\n"
        "# A v1->v2 translation of a legacy salt config. Validated (data-free,\n"
        "# non-strict `salt2 graph validate`) at conversion. norm_dict / class\n"
        "# weights / shape_path / data paths remain REQUIRED overrides (design §5\n"
        "# placeholder policy).\n"
    )
    return header + body


def main(argv: Sequence[str] | None = None) -> int:
    """``salt2 convert-config`` entry point (dispatched from salt.core.main).

    Returns
    -------
    int
        0 on success, 1 on a convert/validate error.
    """
    parser = argparse.ArgumentParser(
        prog="salt2 convert-config",
        description="translate a legacy v1 salt YAML config to the v2 salt.core schema (M7 W1)",
    )
    parser.add_argument(
        "configs",
        nargs="+",
        type=Path,
        help="v1 config stack (base first, overlays after — deep-merged like `salt fit`)",
    )
    parser.add_argument("-o", "--output", required=True, type=Path, help="output v2 config path")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=None,
        help="optional train H5 (wildcards ok) to read the global flavour head's class_names from "
        "its h5 attrs (v1 autodiscovery, cli.py:436-443) when not given explicitly",
    )
    parser.add_argument(
        "--class-names",
        action="append",
        default=[],
        metavar="HEAD=a,b,c",
        help="explicit class names for a head, e.g. --class-names jets_classification=bjets,cjets,"
        "ujets,taujets (repeatable). Overrides h5-attr autodiscovery.",
    )
    parser.add_argument(
        "--no-base",
        action="store_true",
        help="do not auto-prepend base.yaml from the first config's directory",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip the final `salt2 graph validate` self-check (data-free, non-strict; debug only)",
    )
    args = parser.parse_args(argv)

    class_names: dict[str, list[str]] = {}
    for spec in args.class_names:
        if "=" not in spec:
            print(f"convert-config: bad --class-names {spec!r} (want HEAD=a,b,c)", file=sys.stderr)
            return 2
        head, names = spec.split("=", 1)
        class_names[head.strip()] = [n.strip() for n in names.split(",") if n.strip()]

    try:
        cfg = convert_config(
            args.configs,
            train_file=args.train_file,
            class_names=class_names or None,
            with_base=not args.no_base,
        )
    except ConvertError as err:
        print(f"salt.core.convert.ConvertError: {err}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_yaml_dump(cfg))
    print(f"convert-config: wrote {args.output}")

    if args.no_validate:
        return 0
    return _self_validate(cfg, args.output)


def _self_validate(cfg: Mapping[str, Any], output: Path) -> int:
    """Run ``salt2 graph validate`` on the converter's output (FD §9.3 self-check).

    Supplies each `Normaliser` a real minimal norm_dict (covering the config's
    streams) written next to the output, so the data-free preflight passes — the
    M5/M6-CONV convention (gates_m6.py:135-139: a real parity norm dict, NO
    ``--strict`` because multi-stream configs emit benign "missing input type"
    preflight warnings without data). The check still catches every genuine
    graph/connectivity error.

    Returns
    -------
    int
        0 when the converted config plan-compiles cleanly in every mode, 1
        otherwise.
    """
    from salt.core import cli as graph_cli  # noqa: PLC0415 - heavy graph import, CLI-time only

    streams: list[str] = []
    for mod in cfg["model"]["init_args"]["modules"].values():
        if mod.get("class_path", "").endswith("Normaliser"):
            streams.extend(mod.get("init_args", {}).get("streams", []))
    norm_path = output.parent / f"{output.stem}__validation_norm_dict.yaml"
    norm_path.write_text(yaml.safe_dump({s: {} for s in dict.fromkeys(streams)}))

    set_overrides: list[str] = []
    for mod_name, mod in cfg["model"]["init_args"]["modules"].items():
        if mod.get("class_path", "").endswith("Normaliser"):
            set_overrides += [
                "--set",
                f"model.modules.{mod_name}.init_args.norm_dict={norm_path}",
            ]
    rc = graph_cli.main(["graph", "validate", "-c", str(output), *set_overrides])
    if rc != 0:
        print(
            f"convert-config: output FAILED `salt2 graph validate` (rc={rc}) — the conversion is "
            "not semantically valid (FD §9.3 self-check)",
            file=sys.stderr,
        )
        return 1
    print(
        "convert-config: output passed `salt2 graph validate` (a full --strict pass needs the "
        "real norm_dict + a schema artifact for the data sample)"
    )
    return 0
