"""The ``export:`` config block (dataclasses) plus manifest resolution; the
module body stays torch-free (reduce-registry lookups are deferred imports).
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
"""Track selections accepted by the Athena-side loader."""


def _live_known_reduces() -> tuple[str, ...]:
    """Registered reduce names, from the live registry (deferred import keeps this module torch-free)."""
    from salt.core.onnx.reduces import registered_reduces  # noqa: PLC0415 - deferred torch seam

    return registered_reduces()


def _live_per_token_reduces() -> tuple[str, ...]:
    """Registered per-token reduce names, from the live registry (deferred import)."""
    from salt.core.onnx.reduces import per_token_reduces  # noqa: PLC0415 - deferred torch seam

    return per_token_reduces()


def __getattr__(name: str) -> tuple[str, ...]:
    """Resolve the live ``KNOWN_REDUCES``/``PER_TOKEN_REDUCES`` attributes (PEP 562).

    These are live views of the `salt.core.onnx.reduces` registry, resolved on attribute access —
    accessing them triggers the deferred registry import, never at this module's own import.
    """
    if name == "KNOWN_REDUCES":
        return _live_known_reduces()
    if name == "PER_TOKEN_REDUCES":
        return _live_per_token_reduces()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass
class ExportInput:
    """One ONNX graph input (or ``alias:`` pseudo-input).

    Parameters
    ----------
    port : str
        The bundle port the tensor feeds, e.g. ``inputs.tracks``. Must be a
        two-component ``inputs.<stream>`` key.
    name : str | None, optional
        The ONNX graph input name, by default ``<stream minus trailing 's'>_features``.
        Forbidden on ``alias:`` entries (they consume no positional input).
    sequence : bool, optional
        Whether the tensor is a variable-length sequence fed as ``[L, F]`` without a
        batch dim, by default False (a global ``[1, F]`` vector).
    dyn_axis : str | None, optional
        The ONNX dynamic-axis name of the sequence dim, by default ``n_<stream>``.
        Sequence entries only.
    alias : str | None, optional
        Source port for alias entries (one Athena tensor feeding multiple bundle
        ports): the adapter binds `port` from the alias source's tensor instead of a
        positional input, by default None.
    athena_name : str | None, optional
        The ``gnn_config`` metadata input name Athena maps collections with, by
        default derived by `default_athena_name`.
    """

    port: str
    name: str | None = None
    sequence: bool = False
    dyn_axis: str | None = None
    alias: str | None = None
    athena_name: str | None = None


@dataclass
class ExportOutput:
    """One ONNX graph output group; writer-declared.

    Instances are returned by `salt.core.writers.Writer.onnx_outputs` and assembled
    into the export manifest by ``WriterCallback.onnx_manifest`` — they are never
    config-parsed (`resolve_export_config` hard-errors on a config-declared
    ``export.outputs``).

    Parameters
    ----------
    port : str
        The bundle port the reduce consumes (an ONNX-plan sink), e.g.
        ``preds.jets.jets_classification``. Any bundle key is legal, not only ``preds.*``.
    name : str | None, optional
        Single-output suffix; the full ONNX name is ``{model_name}_{name}``.
        Exclusive with `names`.
    names : list[str] | None, optional
        Per-class scalar suffixes for the ``split_scalars`` reduce (e.g.
        ``[pb, pc, pu]`` -> ``GN2v2_pb`` ...). Exclusive with `name`.
    dtype : str | None, optional
        Output dtype, by default the reduce's native dtype (``float32`` for
        ``split_scalars``, ``int8`` for the aux reduces).
    reduce : str | None, optional
        Registry key from `KNOWN_REDUCES`, by default ``split_scalars`` when `names`
        is given; REQUIRED with `name` (the aux reduces are never implicit).
    """

    port: str
    name: str | None = None
    names: list[str] | None = None
    dtype: str | None = None
    reduce: str | None = None


@dataclass
class ExportCombine:
    """One manifest post-processing combine: a NEW output from existing ones.

    The combined value is ``sum(scale * output(suffix))`` over `inputs`, computed
    INSIDE the traced graph from the already-reduced outputs; the combined output is
    float32, global, no dynamic axis.

    Parameters
    ----------
    name : str
        The new output's suffix (full name ``{model_name}_{name}``).
    inputs : dict[str, float]
        Source suffix -> scale, in combination order. Every source must be an
        existing GLOBAL float manifest suffix (``split_scalars`` entries, after
        ``rename:``).
    """

    name: str
    inputs: dict[str, float] = field(default_factory=dict)


@dataclass
class ExportConfig:
    """The top-level ``export:`` block, consumed by ``salt2 export``.

    The block carries the EXPORT-ONLY half of the contract: inputs, the Athena model
    name, and the ``rename:``/``combine:`` manifest post-processing. The output
    manifest itself derives from the writers; `outputs` is the assembled-manifest
    carrier filled by `attach_manifest`, and DECLARING it in a config is a hard error.

    Parameters
    ----------
    model_name : str | None, optional
        The Athena-facing model name (output prefix + ``doc_string``); no
        ``_``/``-`` allowed, by default the run ``name`` with ``_``/``-`` stripped.
    track_selection : str, optional
        Athena-side track selection used in the default metadata input names
        (`default_athena_name`), by default ``r22default``.
    inputs : list[ExportInput], optional
        ONNX graph inputs, in positional order.
    outputs : list[ExportOutput], optional
        The ASSEMBLED writer-derived manifest (`attach_manifest`). Never
        config-declared: `resolve_export_config` raises on a non-empty parsed value.
    rename : dict[str, str], optional
        Manifest suffix renames ``old -> new``, applied BEFORE `combine`.
    combine : list[ExportCombine], optional
        Combined outputs, inserted after the global entries and BEFORE the first
        per-token aux entry (`combine_insertion_index`).
    """

    model_name: str | None = None
    track_selection: str = "r22default"
    inputs: list[ExportInput] = field(default_factory=list)
    outputs: list[ExportOutput] = field(default_factory=list)
    rename: dict[str, str] = field(default_factory=dict)
    combine: list[ExportCombine] = field(default_factory=list)


def sanitised_model_name(run_name: str) -> str:
    """The default export name: the run name with ``_``/``-`` stripped."""
    return run_name.replace("_", "").replace("-", "")


def validate_model_name(name: str) -> str:
    """Validate the Athena-facing model name.

    Called ONLY when compiling the ONNX plan / running ``salt2 export`` — never at
    fit time.

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
    """Derive the Athena ``gnn_config`` metadata input name for a stream.

    Globals map to ``<stream minus 's'>_var``; ``tracks``/``flows`` substreams to
    ``<base>_<track_selection>_sd0sort``; ``electrons`` to ``<stream>_r22default``;
    the literal ``flow`` back-compat stream to ``flows_<track_selection>_sd0sort``;
    any other sequence to ``<stream>_var``.
    """
    if not sequence:
        return f"{stream.removesuffix('s')}_var"
    if stream == "flow":  # back-compat flow/flows naming
        return f"flows_{track_selection}_sd0sort"
    if "tracks" in stream or "flows" in stream:
        base = stream.split("_", maxsplit=1)[0]
        return f"{base}_{track_selection}_sd0sort"
    if "electrons" in stream:
        return f"{stream}_r22default"
    return f"{stream}_var"


def resolve_export_config(export: ExportConfig, run_name: str) -> ExportConfig:
    """Validate the EXPORT-ONLY half of the ``export:`` block and fill its defaults.

    `outputs` is NOT handled here — the manifest derives from the writers and is
    attached by `attach_manifest`; a non-empty parsed value is a hard error. Called
    only on the export path — fit never validates.

    Returns
    -------
    ExportConfig
        A resolved copy of the export-only half (``outputs == []``; the input object
        is not mutated).

    Raises
    ------
    ConfigError
        On a declared ``export.outputs`` section, or any malformed entry (messages
        name the offending entry and the rule it breaks).
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

    Per entry: validation + ``reduce``/``dtype`` defaulting. Then the post-processing
    declared in the export block: ``rename:`` applied in place (existence-checked),
    ``combine:`` validated against the renamed GLOBAL float suffixes. Uniqueness
    (ports + flat suffix namespace, combines included) is re-checked as the backstop
    behind the writer-attributed collision check in ``WriterCallback.onnx_manifest``.

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
        On an empty manifest, a malformed entry, a ``rename:`` of a missing suffix, a
        ``combine:`` referencing a non-global/missing suffix, or a name collision.
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
    """Apply ``export.rename`` suffix renames to the manifest.

    Every old suffix must exist; renames run BEFORE combines so combine inputs can
    reference renamed suffixes.

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

    Combine inputs must be GLOBAL float suffixes (``split_scalars`` entries) —
    combined values are linear combinations of the global outputs, checked before
    the aux entries are appended. Combined names must not collide with the manifest
    or each other.

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
    """Where combined outputs insert into the manifest order.

    Combined outputs are appended after the global entries (renames included) but
    BEFORE the sequence-aux entries — i.e. immediately before the first
    `PER_TOKEN_REDUCES` entry, or at the end when there is none.

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
    ``output_names`` list, and the `manifest_table` rendering: per-entry suffixes
    prefixed with ``{model_name}_``, combines inserted at `combine_insertion_index`.

    Returns
    -------
    list[tuple[str, str, str]]
        ``(full output name, dtype, source)`` triples, where source is the entry's
        ``{reduce} {port}`` or ``combine(...)`` description.
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

    Returns
    -------
    str
        One row per flat ONNX output: name, dtype, source (reduce + port, or the
        combine expression).
    """
    rows = ordered_output_names(export)
    width = max(len(name) for name, _, _ in rows)
    lines = [f"ONNX output manifest (writer-derived, model_name={export.model_name}):"]
    lines += [f"  {name:<{width}}  {dtype:<7}  {source}" for name, dtype, source in rows]
    return "\n".join(lines)


def _resolve_input(entry: ExportInput, track_selection: str) -> ExportInput:
    """Validate one input entry and fill its defaults.

    Raises
    ------
    ConfigError
        On alias/sequence/name rule violations.
    """
    stream = stream_of_input_port(entry.port)
    if entry.alias is not None:
        # alias entries consume NO positional input: no graph tensor name, no dynamic
        # axis, no sequences (the GN3 global stream is a [B, F] vector)
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
    # the live registry owns the reduce set + per-reduce dtype rules; deferred import
    # keeps this module torch-free at parse time (only export-block resolution loads it)
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
    # plural-names entries default to split_scalars (the only names-consuming reduce);
    # single-name entries must name a reduce explicitly (aux reduces are never implicit)
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
    # default + validate dtype from the reduce's declared dtype (int8 for aux
    # reduces, float32 for split_scalars per-class probabilities)
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
