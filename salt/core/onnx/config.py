"""The export-only ``export:`` config block + the writer-derived output manifest.

(design §5.1, §7; M4.5 unified-manifest amendment.)

Plain dataclasses parsed by jsonargparse (registered on the `Salt2CLI`
parser as ``--export``, so the block round-trips through saved run configs
and the run-free parse surface), plus the two resolution steps:

- `resolve_export_config` — validates/defaults the EXPORT-ONLY half
  (``model_name``, ``inputs`` incl. ``alias:``/``dyn_axis``,
  ``track_selection``, the ``rename:``/``combine:`` post-processing
  declarations). Declaring ``export.outputs`` is a HARD ERROR since M4.5:
  the output manifest derives from the writers (``writers.modules``), the
  single output manifest for eval AND export.
- `attach_manifest` — attaches the writer-derived `ExportOutput` list to a
  resolved config: per-entry validation/defaulting, ``rename:``
  application, ``combine:`` validation, and the flat-namespace uniqueness
  backstop. (`WriterCallback.onnx_manifest` runs the richer
  writer-attributed collision check before this.)

Validation timing (design §7 "Naming"): the no-``_``/no-``-`` restriction on
`ExportConfig.model_name` applies ONLY when the ONNX plan is compiled or
``salt2 export`` runs — never at fit time. Run names like ``GN2_v2`` train,
eval and write columns exactly as today; the default export name is the run
name with ``_``/``-`` stripped, byte-reproducing v1
(``to_onnx.py:687``: ``config['name'].replace('_','').replace('-','')``).

This module's BODY is deliberately torch-free: `salt.core.main` imports
`ExportConfig` from it at CLI startup to register the parser argument, and the
writer base imports `ExportOutput` (amendment merge condition 2 — writers
return M4's type directly, no parallel manifest type) — neither path needs the
torch-importing reduce registry at parse time. (A bare ``import
salt.core.onnx.config`` does pull torch transitively via the
``salt.core.onnx`` package ``__init__``, which imports the torch-using
adapter; the torch-free property here is this module's own body plus the
deferred-import discipline below, not the whole import path.)

Reduce validation is LIVE since M5: ``KNOWN_REDUCES`` / ``PER_TOKEN_REDUCES``
are no longer frozen tuples but module attributes resolved on access from the
live registry in `salt.core.onnx.reduces` (the M5 ``register_reduce`` surface,
amendment 555-567). `_resolve_output` defaults + validates each manifest
entry's dtype from the reduce's DECLARED dtype rather than hard-coding the
per-reduce rules. The torch-free seam is preserved by a DEFERRED import: the
torch-importing registry is loaded only inside the export-only validation path
(`_resolve_output`, `combine_insertion_index`, the lazy attribute lookup),
never at fit-time parse — `salt.core.main`'s CLI-startup import of this module
touches none of it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import split_key

__all__ = [
    "KNOWN_REDUCES",  # noqa: F822 - PEP 562 module __getattr__ (live registry view)
    "PER_TOKEN_REDUCES",  # noqa: F822 - PEP 562 module __getattr__ (live registry view)
    "TRACK_SELECTIONS",
    "ExportCombine",
    "ExportConfig",
    "ExportInput",
    "ExportOutput",
    "attach_manifest",
    "combine_insertion_index",
    "default_athena_name",
    "manifest_table",
    "ordered_output_names",
    "resolve_export_config",
    "sanitised_model_name",
    "stream_of_input_port",
    "validate_model_name",
]

TRACK_SELECTIONS = (
    "all",
    "ip3d",
    "dipsLoose202102",
    "r22default",
    "r22loose",
    "dipsTightUpgrade",
    "dipsLooseUpgrade",
)
"""Track selections mirroring the Athena-side loader (v1 to_onnx.py:24-35,
https://gitlab.cern.ch/atlas/athena/-/blob/main/PhysicsAnalysis/JetTagging/FlavorTagInference/Root/TracksLoader.cxx)."""


def _live_known_reduces() -> tuple[str, ...]:
    """The registered reduce names, queried from the live registry (deferred import).

    The M5 replacement for the frozen ``KNOWN_REDUCES`` tuple: `register_reduce`
    is the single owner of the set. Deferred so this torch-free module never
    imports the torch-importing registry at parse time (it loads only when an
    export block is actually resolved).

    Returns
    -------
    tuple[str, ...]
        Sorted registered reduce names.
    """
    from salt.core.onnx.reduces import registered_reduces  # noqa: PLC0415 - deferred torch seam

    return registered_reduces()


def _live_per_token_reduces() -> tuple[str, ...]:
    """The registered per-token reduce names, from the live registry (deferred import).

    The M5 replacement for the frozen ``PER_TOKEN_REDUCES`` tuple; consumed by
    `combine_insertion_index` (combines insert before the first per-token entry,
    amendment merge condition 5).

    Returns
    -------
    tuple[str, ...]
        Sorted per-token reduce names.
    """
    from salt.core.onnx.reduces import per_token_reduces  # noqa: PLC0415 - deferred torch seam

    return per_token_reduces()


def __getattr__(name: str) -> tuple[str, ...]:
    """Resolve the live ``KNOWN_REDUCES`` / ``PER_TOKEN_REDUCES`` attributes (PEP 562).

    These were frozen tuples through M4.5; since M5 they are LIVE views of the
    `salt.core.onnx.reduces` registry, resolved on attribute access (so
    ``from salt.core.onnx.config import KNOWN_REDUCES`` and ``config.KNOWN_REDUCES``
    keep working, now returning the registry's current contents). Accessing them
    triggers the deferred registry import — i.e. only when something actually
    reads the reduce set, never at this module's own import.

    Returns
    -------
    tuple[str, ...]
        The requested live tuple.

    Raises
    ------
    AttributeError
        For any other attribute (the normal module-attribute protocol).
    """
    if name == "KNOWN_REDUCES":
        return _live_known_reduces()
    if name == "PER_TOKEN_REDUCES":
        return _live_per_token_reduces()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass
class ExportInput:
    """One ONNX graph input (or ``alias:`` pseudo-input) — design §5.1/§7.

    Parameters
    ----------
    port : str
        The bundle port the tensor feeds, e.g. ``inputs.tracks``. Must be a
        two-component ``inputs.<stream>`` key.
    name : str | None, optional
        The ONNX graph input name (the Athena-facing tensor name), by
        default ``<stream minus trailing 's'>_features`` — v1's
        ``name_athena_out`` rule (``to_onnx.py:508,518``). Forbidden on
        ``alias:`` entries (they consume no positional input).
    sequence : bool, optional
        Whether the tensor is a variable-length sequence fed as ``[L, F]``
        without a batch dim (v1 contract, ``to_onnx.py:374``), by default
        False (a global ``[1, F]`` vector).
    dyn_axis : str | None, optional
        The ONNX dynamic-axis name of the sequence dim, by default
        ``n_<stream>`` (v1 ``athena_num_name``, ``to_onnx.py:519``).
        Sequence entries only.
    alias : str | None, optional
        Source port for alias entries (one Athena tensor feeding multiple
        bundle ports — the GN3 ``global`` stream, design §6.6/§7): the
        adapter binds `port` from the alias source's tensor instead of a
        positional input (``to_onnx.py:377-378`` semantics), by default
        None.
    athena_name : str | None, optional
        The ``gnn_config`` metadata input name Athena maps collections
        with (v1 ``name_athena_in``), by default derived exactly as v1's
        ``get_default_onnx_feature_map`` (``to_onnx.py:475-553``) — see
        `default_athena_name`.
    """

    port: str
    name: str | None = None
    sequence: bool = False
    dyn_axis: str | None = None
    alias: str | None = None
    athena_name: str | None = None


@dataclass
class ExportOutput:
    """One ONNX graph output group — design §5.1/§7.3; WRITER-declared since M4.5.

    Instances are returned by `salt.core.writers.Writer.onnx_outputs`
    (amendment merge condition 2: writers return THIS type directly) and
    assembled into the export manifest by ``WriterCallback.onnx_manifest``
    — they are no longer config-parsed (`resolve_export_config` hard-errors
    on a config-declared ``export.outputs``).

    Parameters
    ----------
    port : str
        The bundle port the reduce consumes (an ONNX-plan sink), e.g.
        ``preds.jets.jets_classification``. Any bundle key is legal (e.g.
        ``objects.masks``), not only ``preds.*``.
    name : str | None, optional
        Single-output suffix (``argmax``/``vertex_union_find`` reduces);
        the full ONNX name is ``{model_name}_{name}``. Exclusive with
        `names`.
    names : list[str] | None, optional
        Per-class scalar suffixes for the ``split_scalars`` reduce (e.g.
        ``[pb, pc, pu]`` -> ``GN2v2_pb`` ...). Exclusive with `name`.
    dtype : str | None, optional
        Output dtype, by default the reduce's native dtype (``float32``
        for ``split_scalars``, ``int8`` for the aux reduces — v1
        ``.char()``, ``to_onnx.py:422,432``).
    reduce : str | None, optional
        Registry key from `KNOWN_REDUCES`, by default ``split_scalars``
        when `names` is given; REQUIRED with `name` (the aux reduces are
        never implicit).
    """

    port: str
    name: str | None = None
    names: list[str] | None = None
    dtype: str | None = None
    reduce: str | None = None


@dataclass
class ExportCombine:
    """One manifest post-processing combine: a NEW output from existing ones.

    The v2 spelling of v1's ``--combine_outputs`` (``to_onnx.py:556-600``):
    an Athena-presentation concern with no eval analogue, kept in the
    ``export:`` block as manifest post-processing (amendment §7 cost 3 /
    merge condition 5). The combined value is
    ``sum(scale * output(suffix))`` over `inputs`, computed INSIDE the
    traced graph from the already-reduced outputs (v1 ``to_onnx.py:
    404-412``); the combined output is float32, global, no dynamic axis.

    Parameters
    ----------
    name : str
        The new output's suffix (full name ``{model_name}_{name}``).
    inputs : dict[str, float]
        Source suffix -> scale, in combination order. Every source must be
        an existing GLOBAL float manifest suffix (``split_scalars``
        entries, after ``rename:`` — the v1 existence check,
        ``to_onnx.py:273-279``).
    """

    name: str
    inputs: dict[str, float] = field(default_factory=dict)


@dataclass
class ExportConfig:
    """The top-level ``export:`` block (design §5.1), consumed by ``salt2 export``.

    Since M4.5 the block carries the EXPORT-ONLY half of the contract:
    inputs, the Athena model name, and the ``rename:``/``combine:``
    manifest post-processing. The output manifest itself derives from the
    writers (``writers.modules`` — the single output manifest for eval AND
    export); `outputs` is the assembled-manifest carrier filled by
    `attach_manifest`, and DECLARING it in a config is a hard error.

    Parameters
    ----------
    model_name : str | None, optional
        The Athena-facing model name (output prefix + ``doc_string``); no
        ``_``/``-`` allowed (v1 ``to_onnx.py:169-170``), by default the run
        ``name`` with ``_``/``-`` stripped (``to_onnx.py:687``).
    track_selection : str, optional
        Athena-side track selection used in the default metadata input
        names (`default_athena_name`), by default ``r22default``.
    inputs : list[ExportInput], optional
        ONNX graph inputs, in positional order.
    outputs : list[ExportOutput], optional
        The ASSEMBLED writer-derived manifest (`attach_manifest`). Never
        config-declared: `resolve_export_config` raises the M4.5 migration
        error on a non-empty parsed value.
    rename : dict[str, str], optional
        Manifest suffix renames ``old -> new``, applied BEFORE `combine`
        (v1 ``--rename`` semantics, existence-checked —
        ``to_onnx.py:263-270``).
    combine : list[ExportCombine], optional
        Combined outputs, inserted after the global entries and BEFORE the
        first per-token aux entry (`combine_insertion_index` — the v1
        order, amendment merge condition 5).
    """

    model_name: str | None = None
    track_selection: str = "r22default"
    inputs: list[ExportInput] = field(default_factory=list)
    outputs: list[ExportOutput] = field(default_factory=list)
    rename: dict[str, str] = field(default_factory=dict)
    combine: list[ExportCombine] = field(default_factory=list)


def sanitised_model_name(run_name: str) -> str:
    """The default export name: the run name with ``_``/``-`` stripped.

    Byte-reproduces v1's default (``to_onnx.py:687``).

    Returns
    -------
    str
        The sanitised name.
    """
    return run_name.replace("_", "").replace("-", "")


def validate_model_name(name: str) -> str:
    """Validate the Athena-facing model name (v1 ``to_onnx.py:169-170``).

    Called ONLY when compiling the ONNX plan / running ``salt2 export`` —
    never at fit time (design §7 "Naming").

    Returns
    -------
    str
        The validated name, unchanged.

    Raises
    ------
    ConfigError
        On an empty name or one containing ``_``/``-``.
    """
    if not name:
        raise ConfigError("export.model_name must be a non-empty string (design §7)")
    if "_" in name or "-" in name:
        raise ConfigError(
            f"export.model_name {name!r} must not contain underscores or dashes "
            "(Athena naming restriction, v1 to_onnx.py:169-170) — the run 'name:' is "
            "unrestricted; only the export name is validated (design §7)"
        )
    return name


def stream_of_input_port(port: str) -> str:
    """Extract the stream name from an ``inputs.<stream>`` export port.

    Returns
    -------
    str
        The stream name.

    Raises
    ------
    ConfigError
        If the port is not a two-component ``inputs.<stream>`` key.
    """
    try:
        parts = split_key(port)
    except (TypeError, ValueError) as err:
        raise ConfigError(f"export input port {port!r} is not a valid dotted key: {err}") from err
    if len(parts) != 2 or parts[0] != "inputs":
        raise ConfigError(
            f"export input port {port!r} must be a dataset-produced 'inputs.<stream>' key "
            "(design §7: every export.inputs port is a dataset-produced key)"
        )
    return parts[1]


def default_athena_name(stream: str, sequence: bool, track_selection: str) -> str:
    """Derive the v1 metadata input name for a stream (``to_onnx.py:475-553``).

    Reproduces ``get_default_onnx_feature_map`` exactly: globals map to
    ``<stream minus 's'>_var``; ``tracks``/``flows`` substreams to
    ``<base>_<track_selection>_sd0sort``; ``electrons`` to
    ``<stream>_r22default``; the literal ``flow`` back-compat stream to
    ``flows_<track_selection>_sd0sort``; any other sequence to
    ``<stream>_var``.

    Returns
    -------
    str
        The Athena-side input name for the ``gnn_config`` metadata.
    """
    if not sequence:
        return f"{stream.removesuffix('s')}_var"  # to_onnx.py:507
    if stream == "flow":  # back-compat flow/flows naming (to_onnx.py:531-541)
        return f"flows_{track_selection}_sd0sort"
    if "tracks" in stream or "flows" in stream:
        base = stream.split("_", maxsplit=1)[0]
        return f"{base}_{track_selection}_sd0sort"  # to_onnx.py:517
    if "electrons" in stream:
        return f"{stream}_r22default"  # to_onnx.py:525
    return f"{stream}_var"  # to_onnx.py:546


def resolve_export_config(export: ExportConfig, run_name: str) -> ExportConfig:
    """Validate the EXPORT-ONLY half of the ``export:`` block and fill its defaults.

    The returned config has `model_name` resolved+validated and, per input,
    `name`/`dyn_axis`/`athena_name` filled (see the field docs for the v1
    rules each default reproduces). `outputs` is NOT handled here — the
    manifest derives from the writers and is attached by `attach_manifest`;
    a non-empty parsed value is the M4.5 migration hard error. Called only
    on the export path — fit never validates (design §7).

    Returns
    -------
    ExportConfig
        A resolved copy of the export-only half (``outputs == []``; the
        input object is not mutated).

    Raises
    ------
    ConfigError
        On a declared ``export.outputs`` section (removed at M4.5 — the
        §4.1-bar pointer at ``writers:``), or any malformed entry
        (messages name the offending entry and the rule it breaks).
    """
    if export.outputs:
        raise ConfigError(
            "export.outputs was REMOVED by the M4.5 unified-manifest amendment — the ONNX "
            "output manifest now derives from the writers (the same declarations that name "
            "the eval columns), so the two can never drift.\n"
            "  fix: delete the export.outputs section. The shipped writers already declare "
            "the standard surface (writers.modules.tasks: global classification -> per-class "
            "scalars, sequence classification -> argmax int8, vertexing -> union-find int8); "
            "narrow with TaskWriter onnx/onnx_streams/onnx_tasks, rename with onnx_names, "
            "post-process with export.rename/export.combine, and inspect the assembled "
            "manifest with `salt2 export --manifest` or `salt2 graph resolve` "
            "(amendment-unified-writers.md §2, §6)"
        )
    model_name = validate_model_name(export.model_name or sanitised_model_name(run_name))
    if export.track_selection not in TRACK_SELECTIONS:
        raise ConfigError(
            f"export.track_selection {export.track_selection!r} is not a known Athena track "
            f"selection — choose from {list(TRACK_SELECTIONS)} (v1 to_onnx.py:24-35)"
        )
    if not export.inputs:
        raise ConfigError("export.inputs must declare at least one input (design §5.1)")
    inputs = [_resolve_input(entry, export.track_selection) for entry in export.inputs]
    _check_input_uniqueness(inputs)
    for old, new in export.rename.items():
        if not isinstance(old, str) or not isinstance(new, str) or not old or not new:
            raise ConfigError(
                f"export.rename entries must map a non-empty old suffix to a non-empty new "
                f"suffix, got {old!r}: {new!r} (v1 --rename semantics, to_onnx.py:263-270)"
            )
    for entry in export.combine:
        if not entry.name:
            raise ConfigError("export.combine entries must set a non-empty 'name' suffix")
        if not entry.inputs:
            raise ConfigError(
                f"export.combine {entry.name!r}: 'inputs' must map at least one source "
                "suffix to a scale (v1 parse_output_combination, to_onnx.py:556-600)"
            )
    return ExportConfig(
        model_name=model_name,
        track_selection=export.track_selection,
        inputs=inputs,
        outputs=[],
        rename=dict(export.rename),
        combine=[ExportCombine(name=c.name, inputs=dict(c.inputs)) for c in export.combine],
    )


def attach_manifest(export: ExportConfig, outputs: Sequence[ExportOutput]) -> ExportConfig:
    """Attach the writer-derived output manifest to a RESOLVED export config.

    Per entry: validation + ``reduce``/``dtype`` defaulting (unchanged M4
    rules). Then the post-processing declared in the export block, in v1
    order (``to_onnx.py:243-307``): ``rename:`` applied in place
    (existence-checked), ``combine:`` validated against the renamed GLOBAL
    float suffixes. Uniqueness (ports + flat suffix namespace, combines
    included) is re-checked as the backstop behind the writer-attributed
    collision check in ``WriterCallback.onnx_manifest``.

    Parameters
    ----------
    export : ExportConfig
        A `resolve_export_config` result (``model_name`` validated).
    outputs : Sequence[ExportOutput]
        The assembled writer manifest, in manifest order.

    Returns
    -------
    ExportConfig
        A copy carrying the resolved manifest.

    Raises
    ------
    ConfigError
        On an empty manifest, a malformed entry, a ``rename:`` of a
        missing suffix, a ``combine:`` referencing a non-global/missing
        suffix, or a name collision.
    """
    if export.model_name is None:
        raise ConfigError("attach_manifest needs a resolved export config (model_name set)")
    if not outputs:
        raise ConfigError(
            "the configured writers declare no ONNX outputs — the export manifest derives "
            "from writers.modules (M4.5 unified manifest); check that at least one writer "
            "participates (TaskWriter default onnx: true, or a custom onnx_outputs override; "
            "amendment §2)"
        )
    resolved = [_resolve_output(entry) for entry in outputs]
    resolved = _apply_rename(resolved, export.rename)
    _check_output_uniqueness(resolved)
    _check_combines(resolved, export.combine)
    return replace(export, outputs=resolved)


def _apply_rename(outputs: list[ExportOutput], rename: dict[str, str]) -> list[ExportOutput]:
    """Apply ``export.rename`` suffix renames to the manifest (v1 semantics).

    Every old suffix must exist (v1's existence assert,
    ``to_onnx.py:268``); renames run BEFORE combines so combine inputs can
    reference renamed suffixes (v1 order, ``to_onnx.py:263-279``).

    Returns
    -------
    list[ExportOutput]
        The manifest with renamed entries (input list not mutated).

    Raises
    ------
    ConfigError
        When an old suffix matches no manifest entry.
    """
    if not rename:
        return outputs
    out = list(outputs)
    for old, new in rename.items():
        hit = False
        for i, entry in enumerate(out):
            if entry.name == old:
                out[i] = replace(entry, name=new)
                hit = True
            elif entry.names is not None and old in entry.names:
                out[i] = replace(
                    entry, names=[new if suffix == old else suffix for suffix in entry.names]
                )
                hit = True
        if not hit:
            known = [s for entry in out for s in (entry.names or [entry.name])]
            raise ConfigError(
                f"export.rename: suffix {old!r} matches no manifest output — the "
                f"writer-derived manifest carries {known} (v1 existence check, "
                "to_onnx.py:268; inspect with `salt2 export --manifest`)"
            )
    return out


def _check_combines(outputs: list[ExportOutput], combines: list[ExportCombine]) -> None:
    """Validate ``export.combine`` entries against the (renamed) manifest.

    Combine inputs must be GLOBAL float suffixes (``split_scalars``
    entries) — the v1 contract: combined values are linear combinations of
    the global outputs, checked before the aux entries are appended
    (``to_onnx.py:273-279``). Combined names must not collide with the
    manifest or each other.

    Raises
    ------
    ConfigError
        Naming the offending combine entry and the rule it breaks.
    """
    global_suffixes = {
        suffix for entry in outputs if entry.names is not None for suffix in entry.names
    }
    taken = {str(s) for entry in outputs for s in (entry.names or [entry.name])}
    for entry in combines:
        if missing := sorted(set(entry.inputs) - global_suffixes):
            raise ConfigError(
                f"export.combine {entry.name!r}: input suffix(es) {missing} are not global "
                f"float outputs of the manifest — combines draw from the split_scalars "
                f"suffixes {sorted(global_suffixes)} after rename (v1 contract, "
                "to_onnx.py:273-279)"
            )
        if entry.name in taken:
            raise ConfigError(
                f"export.combine {entry.name!r}: the combined suffix collides with an "
                "existing output name — pick a fresh suffix"
            )
        taken.add(entry.name)


def combine_insertion_index(outputs: Sequence[ExportOutput]) -> int:
    """Where combined outputs insert into the manifest order (merge condition 5).

    The v1 rule (``to_onnx.py:258-292``): combined outputs are appended
    after the global entries (renames included) but BEFORE the sequence-aux
    entries — i.e. immediately before the first `PER_TOKEN_REDUCES` entry,
    or at the end when there is none. Append-at-end would diverge for any
    model carrying both combines and aux outputs (the O2 combine+aux
    fixture exercises exactly this).

    Returns
    -------
    int
        The entry index combines insert at (en bloc, declaration order).
    """
    per_token = _live_per_token_reduces()
    for i, entry in enumerate(outputs):
        if entry.reduce in per_token:
            return i
    return len(outputs)


def ordered_output_names(export: ExportConfig) -> list[tuple[str, str, str]]:
    """The final flat ONNX output list — names/dtypes/sources in traced order.

    The single ordering authority shared by the adapter
    (``output_names``/``output_dtypes``/``forward``), the metadata
    ``output_names`` list, and the `manifest_table` rendering: per-entry
    suffixes prefixed with ``{model_name}_``, combines inserted at
    `combine_insertion_index`.

    Parameters
    ----------
    export : ExportConfig
        A manifest-attached resolved config (`attach_manifest` output).

    Returns
    -------
    list[tuple[str, str, str]]
        ``(full output name, dtype, source)`` triples, where source is the
        entry's ``{reduce} {port}`` or ``combine(...)`` description.
    """
    prefix = str(export.model_name)
    rows: list[tuple[str, str, str]] = []
    for entry in export.outputs:
        rows.extend(
            (f"{prefix}_{suffix}", str(entry.dtype), f"{entry.reduce} {entry.port}")
            for suffix in (entry.names if entry.names is not None else [entry.name])
        )
    at = sum(
        len(entry.names) if entry.names is not None else 1
        for entry in export.outputs[: combine_insertion_index(export.outputs)]
    )
    combined = [
        (
            f"{prefix}_{c.name}",
            "float32",
            "combine(" + " + ".join(f"{scale:g}*{s}" for s, scale in c.inputs.items()) + ")",
        )
        for c in export.combine
    ]
    return rows[:at] + combined + rows[at:]


def manifest_table(export: ExportConfig) -> str:
    """Render the assembled output manifest (``salt2 export --manifest``, plan_onnx.txt).

    Parameters
    ----------
    export : ExportConfig
        A manifest-attached resolved config.

    Returns
    -------
    str
        One row per flat ONNX output: name, dtype, source (reduce + port,
        or the combine expression) — the generated-artifact answer to
        "what exactly does Athena see" (amendment §7 cost 2 mitigation).
    """
    rows = ordered_output_names(export)
    width = max(len(name) for name, _, _ in rows)
    lines = [f"ONNX output manifest (writer-derived, model_name={export.model_name}):"]
    lines += [f"  {name:<{width}}  {dtype:<7}  {source}" for name, dtype, source in rows]
    return "\n".join(lines)


def _resolve_input(entry: ExportInput, track_selection: str) -> ExportInput:
    """Validate one input entry and fill its defaults.

    Returns
    -------
    ExportInput
        A resolved copy.

    Raises
    ------
    ConfigError
        On alias/sequence/name rule violations.
    """
    stream = stream_of_input_port(entry.port)
    if entry.alias is not None:
        # alias entries consume NO positional input (to_onnx.py:377-378): no
        # graph tensor name, no dynamic axis, and — M4 scope — no sequences
        # (the GN3 global stream is a [B, F] vector, design §6.6)
        stream_of_input_port(entry.alias)
        if entry.name is not None:
            raise ConfigError(
                f"export input {entry.port!r}: alias entries bind from {entry.alias!r} and "
                "consume no positional ONNX input — drop 'name' (design §7 alias semantics)"
            )
        if entry.sequence or entry.dyn_axis is not None:
            raise ConfigError(
                f"export input {entry.port!r}: sequence alias entries are not supported in M4 "
                "(the alias mechanism serves the GN3 global [B, F] vector, design §6.6/§7)"
            )
        return replace(entry, athena_name=None)
    if entry.dyn_axis is not None and not entry.sequence:
        raise ConfigError(
            f"export input {entry.port!r} sets dyn_axis={entry.dyn_axis!r} but is not a "
            "sequence — dynamic axes apply to sequence inputs only (v1 to_onnx.py:319-323)"
        )
    return replace(
        entry,
        name=entry.name or f"{stream.removesuffix('s')}_features",
        dyn_axis=(entry.dyn_axis or f"n_{stream}") if entry.sequence else None,
        athena_name=entry.athena_name
        or default_athena_name(stream, entry.sequence, track_selection),
    )


def _resolve_output(entry: ExportOutput) -> ExportOutput:
    """Validate one output entry and fill its defaults.

    Returns
    -------
    ExportOutput
        A resolved copy.

    Raises
    ------
    ConfigError
        On name/names/reduce/dtype rule violations.
    """
    try:
        split_key(entry.port)
    except (TypeError, ValueError) as err:
        raise ConfigError(
            f"export output port {entry.port!r} is not a valid dotted key: {err}"
        ) from err
    # the live registry is the single owner of the reduce set + per-reduce dtype
    # rules (M5 register_reduce surface); deferred import keeps this module
    # torch-free at parse time (only export-block resolution loads it)
    from salt.core.onnx.reduces import reduce_spec, registered_reduces  # noqa: PLC0415 - torch seam

    if (entry.name is None) == (entry.names is None):
        raise ConfigError(
            f"export output {entry.port!r} must set exactly one of 'name' (single-output "
            "reduces) or 'names' (per-class scalars) (design §5.1)"
        )
    if entry.names is not None and (not entry.names or len(set(entry.names)) != len(entry.names)):
        raise ConfigError(
            f"export output {entry.port!r}: 'names' must be a non-empty list without "
            f"duplicates, got {entry.names!r}"
        )
    # resolve the reduce key: plural-names entries default to split_scalars (the
    # only names-consuming reduce); single-name entries must name one explicitly
    # (aux reduces are never implicit, design §7.3)
    if entry.names is not None:
        reduce = entry.reduce or "split_scalars"
    else:
        reduce = entry.reduce
        if reduce is None:
            aux = [k for k in registered_reduces() if not reduce_spec(k).expects_names]
            raise ConfigError(
                f"export output {entry.port!r}: single-name outputs must set an explicit "
                f"'reduce' from {aux} — aux reduces are never implicit (design §7.3)"
            )
    try:
        spec = reduce_spec(reduce)
    except ConfigError:
        raise ConfigError(
            f"export output {entry.port!r}: unknown reduce {reduce!r} — registry: "
            f"{list(registered_reduces())} (design §7.3; register via "
            "salt.core.onnx.reduces.register_reduce)"
        ) from None
    # the reduce's name/names arity must match how the entry was declared
    if entry.names is not None and not spec.expects_names:
        raise ConfigError(
            f"export output {entry.port!r}: 'names' implies the split_scalars reduce, "
            f"got reduce={entry.reduce!r} (design §7.3)"
        )
    if entry.name is not None and spec.expects_names:
        raise ConfigError(
            f"export output {entry.port!r}: {reduce} emits per-class scalars — use "
            "'names' (design §5.1)"
        )
    # default + validate the dtype from the reduce's DECLARED dtype (the live
    # registry owns the per-reduce rule — v1 .char() int8 for aux reduces,
    # to_onnx.py:422,432; float32 per-class probabilities for split_scalars)
    dtype = entry.dtype or spec.dtype
    if dtype != spec.dtype:
        raise ConfigError(
            f"export output {entry.port!r}: the {reduce!r} reduce emits {spec.dtype}, got "
            f"dtype={entry.dtype!r} (the reduce's declared output dtype, register_reduce)"
        )
    names = list(entry.names) if entry.names else None
    return replace(entry, names=names, reduce=reduce, dtype=dtype)


def _check_input_uniqueness(inputs: list[ExportInput]) -> None:
    """Reject duplicate input ports or graph tensor names.

    Raises
    ------
    ConfigError
        Naming the duplicate.
    """
    seen_ports: set[str] = set()
    seen_names: set[str] = set()
    for entry in inputs:
        if entry.port in seen_ports:
            raise ConfigError(f"export.inputs declares port {entry.port!r} twice")
        seen_ports.add(entry.port)
        if entry.name is not None:
            if entry.name in seen_names:
                raise ConfigError(f"export.inputs declares input name {entry.name!r} twice")
            seen_names.add(entry.name)
    for entry in inputs:
        if entry.alias is not None and entry.alias not in seen_ports:
            raise ConfigError(
                f"export input {entry.port!r}: alias source {entry.alias!r} is not another "
                "export input port — the alias binds from a declared input's tensor (design §7)"
            )


def _check_output_uniqueness(outputs: list[ExportOutput]) -> None:
    """Reject duplicate output ports or name suffixes.

    Raises
    ------
    ConfigError
        Naming the duplicate.
    """
    seen_ports: set[str] = set()
    seen_names: set[str] = set()
    for entry in outputs:
        if entry.port in seen_ports:
            raise ConfigError(f"the assembled export manifest declares port {entry.port!r} twice")
        seen_ports.add(entry.port)
        for suffix in entry.names if entry.names is not None else [entry.name]:
            if suffix in seen_names:
                raise ConfigError(
                    f"the assembled export manifest declares output name {suffix!r} twice "
                    "(flat ONNX namespace — rename one side via TaskWriter onnx_names or "
                    "export.rename, amendment §5 rule 4)"
                )
            seen_names.add(str(suffix))
