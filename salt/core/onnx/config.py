"""The declarative ``export:`` config block for ``salt2 export`` (design §5.1, §7).

Three plain dataclasses parsed by jsonargparse (registered on the `Salt2CLI`
parser as ``--export``, so the block round-trips through saved run configs
and the run-free parse surface), plus the resolution/validation step that
turns the parsed block into a fully-defaulted `ExportConfig`.

Validation timing (design §7 "Naming"): the no-``_``/no-``-`` restriction on
`ExportConfig.model_name` applies ONLY when the ONNX plan is compiled or
``salt2 export`` runs — never at fit time. Run names like ``GN2_v2`` train,
eval and write columns exactly as today; the default export name is the run
name with ``_``/``-`` stripped, byte-reproducing v1
(``to_onnx.py:687``: ``config['name'].replace('_','').replace('-','')``).

This module is deliberately torch-free: `salt.core.main` imports it at CLI
startup to register the parser argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import split_key

__all__ = [
    "KNOWN_REDUCES",
    "TRACK_SELECTIONS",
    "ExportConfig",
    "ExportInput",
    "ExportOutput",
    "default_athena_name",
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

KNOWN_REDUCES = ("split_scalars", "argmax", "vertex_union_find")
"""The shipped reduce registry keys (design §7.3; MaskFormer reduces are M5)."""


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
    """One ONNX graph output group — design §5.1/§7.3.

    Parameters
    ----------
    port : str
        The bundle port the reduce consumes (an ONNX-plan sink), e.g.
        ``preds.jets.jets_classification``.
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
class ExportConfig:
    """The top-level ``export:`` block (design §5.1), consumed by ``salt2 export``.

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
        ONNX graph outputs, in output order.
    """

    model_name: str | None = None
    track_selection: str = "r22default"
    inputs: list[ExportInput] = field(default_factory=list)
    outputs: list[ExportOutput] = field(default_factory=list)


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
    """Validate the parsed ``export:`` block and fill every default.

    The returned config has `model_name` resolved+validated and, per input,
    `name`/`dyn_axis`/`athena_name` filled (see the field docs for the v1
    rules each default reproduces); per output, `reduce`/`dtype` filled.
    Called only on the export path — fit never validates (design §7).

    Returns
    -------
    ExportConfig
        A fully-resolved copy (the input object is not mutated).

    Raises
    ------
    ConfigError
        On any malformed entry (messages name the offending entry and the
        rule it breaks).
    """
    model_name = validate_model_name(export.model_name or sanitised_model_name(run_name))
    if export.track_selection not in TRACK_SELECTIONS:
        raise ConfigError(
            f"export.track_selection {export.track_selection!r} is not a known Athena track "
            f"selection — choose from {list(TRACK_SELECTIONS)} (v1 to_onnx.py:24-35)"
        )
    if not export.inputs:
        raise ConfigError("export.inputs must declare at least one input (design §5.1)")
    if not export.outputs:
        raise ConfigError("export.outputs must declare at least one output (design §5.1)")
    inputs = [_resolve_input(entry, export.track_selection) for entry in export.inputs]
    _check_input_uniqueness(inputs)
    outputs = [_resolve_output(entry) for entry in export.outputs]
    _check_output_uniqueness(outputs)
    return ExportConfig(
        model_name=model_name,
        track_selection=export.track_selection,
        inputs=inputs,
        outputs=outputs,
    )


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
    if (entry.name is None) == (entry.names is None):
        raise ConfigError(
            f"export output {entry.port!r} must set exactly one of 'name' (single-output "
            "reduces) or 'names' (per-class scalars) (design §5.1)"
        )
    if entry.names is not None:
        if not entry.names or len(set(entry.names)) != len(entry.names):
            raise ConfigError(
                f"export output {entry.port!r}: 'names' must be a non-empty list without "
                f"duplicates, got {entry.names!r}"
            )
        reduce = entry.reduce or "split_scalars"
        if reduce != "split_scalars":
            raise ConfigError(
                f"export output {entry.port!r}: 'names' implies the split_scalars reduce, "
                f"got reduce={entry.reduce!r} (design §7.3)"
            )
        dtype = entry.dtype or "float32"
        if dtype != "float32":
            raise ConfigError(
                f"export output {entry.port!r}: split_scalars emits float32 scalars, got "
                f"dtype={entry.dtype!r} (v1 per-class probability outputs)"
            )
    else:
        reduce = entry.reduce
        if reduce is None:
            raise ConfigError(
                f"export output {entry.port!r}: single-name outputs must set an explicit "
                f"'reduce' from {[k for k in KNOWN_REDUCES if k != 'split_scalars']} — aux "
                "reduces are never implicit (design §7.3)"
            )
        if reduce == "split_scalars":
            raise ConfigError(
                f"export output {entry.port!r}: split_scalars emits per-class scalars — use "
                "'names' (design §5.1)"
            )
        dtype = entry.dtype or "int8"
        if dtype != "int8":
            raise ConfigError(
                f"export output {entry.port!r}: the {reduce!r} reduce emits int8 "
                f"(v1 .char(), to_onnx.py:422,432), got dtype={entry.dtype!r}"
            )
    if reduce not in KNOWN_REDUCES:
        raise ConfigError(
            f"export output {entry.port!r}: unknown reduce {reduce!r} — registry: "
            f"{list(KNOWN_REDUCES)} (design §7.3; MaskFormer reduces land in M5)"
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
            raise ConfigError(f"export.outputs declares port {entry.port!r} twice")
        seen_ports.add(entry.port)
        for suffix in entry.names if entry.names is not None else [entry.name]:
            if suffix in seen_names:
                raise ConfigError(f"export.outputs declares output name {suffix!r} twice")
            seen_names.add(str(suffix))
