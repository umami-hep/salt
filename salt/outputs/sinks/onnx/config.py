"""The ``export:`` config block (dataclasses) plus export-input resolution; the
module body stays torch-free (reduce-registry lookups are deferred imports).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from salt.graph.errors import ConfigError
from salt.graph.spec import split_key

__all__ = [
    "KNOWN_REDUCES",  # noqa: F822 - PEP 562 module __getattr__ (live registry view)
    "PER_TOKEN_REDUCES",  # noqa: F822 - PEP 562 module __getattr__ (live registry view)
    "TRACK_SELECTIONS",
    "ExportCombine",
    "ExportConfig",
    "ExportInput",
    "ExportOutput",
    "default_athena_name",
    "reject_declared_outputs",
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
    """Registered reduce names, from the live registry (the deferred import keeps this
    module torch-free).
    """
    from salt.outputs.sinks.onnx.reduces import registered_reduces

    return registered_reduces()


def _live_per_token_reduces() -> tuple[str, ...]:
    """Registered per-token reduce names, from the live registry (deferred import)."""
    from salt.outputs.sinks.onnx.reduces import per_token_reduces

    return per_token_reduces()


def __getattr__(name: str) -> tuple[str, ...]:
    """Resolve the live ``KNOWN_REDUCES``/``PER_TOKEN_REDUCES`` attributes (PEP 562).

    These are live views of the `salt.outputs.sinks.onnx.reduces` registry,
    resolved on attribute access — accessing them triggers the deferred registry
    import, never at this module's own import.
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

    @classmethod
    def coerce(cls, obj: ExportInput | Mapping[str, Any]) -> ExportInput:
        """Build from a dataclass or a plain config mapping."""
        if isinstance(obj, ExportInput):
            return obj
        return cls(**dict(obj))


@dataclass
class ExportOutput:
    """One ONNX output group for the custom-reduce binder protocol.

    The parameter object `salt.outputs.sinks.onnx.reduces.bind_reduce` (the public
    ``register_reduce`` extension surface) consumes; never config-parsed
    (`resolve_export_config` hard-errors on a config-declared
    ``export.outputs`` — the live output manifest is the folded
    `salt.outputs.OnnxExportSink`).

    Parameters
    ----------
    port : str
        The bundle port the reduce consumes (an ONNX-plan sink), e.g.
        ``preds.jets.jets_classification``. Any bundle key is legal, not only ``preds.*``.
    name : str | None, optional
        Single-output suffix; the full ONNX name is ``{model_name}_{name}``.
        Exclusive with `names`.
    names : list[str] | None, optional
        Per-class scalar suffixes (e.g. ``[pb, pc, pu]`` -> ``GN2v2_pb`` ...).
        Exclusive with `name`.
    dtype : str | None, optional
        Output dtype, by default the reduce's declared dtype.
    reduce : str | None, optional
        Registry key from `KNOWN_REDUCES` (the live ``register_reduce`` registry).
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

    @classmethod
    def coerce(cls, obj: ExportCombine | Mapping[str, Any]) -> ExportCombine:
        """Build from a dataclass or a plain config mapping."""
        if isinstance(obj, ExportCombine):
            return obj
        return cls(**dict(obj))


@dataclass
class ExportConfig:
    """The ONNX export contract, consumed by ``salt export``.

    Carries the EXPORT-ONLY half: inputs, the Athena model name, and the
    ``rename:``/``combine:`` manifest post-processing. Its config home is the
    `salt.outputs.OnnxExportSink` (``OnnxExportSink.export_config`` assembles and
    resolves one). The output manifest itself derives from the sink's collected
    leaves; DECLARING ``outputs`` in a config is a hard error.

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
        RETIRED manifest carrier. Never config-declared: `resolve_export_config`
        raises on a non-empty parsed value.
    rename : dict[str, str], optional
        Manifest suffix renames ``old -> new``, applied BEFORE `combine`.
    combine : list[ExportCombine], optional
        Combined outputs, inserted after the global entries and BEFORE the first
        per-token aux entry.
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

    Called ONLY when compiling the ONNX plan / running ``salt export`` — never at
    fit time.

    Raises
    ------
    ConfigError
        On an empty name or one containing ``_``/``-``.
    """
    if not name:
        raise ConfigError("export.model_name must be a non-empty string")
    if "_" in name or "-" in name:
        raise ConfigError(
            f"export.model_name {name!r} must not contain underscores or dashes "
            "(Athena naming restriction, v1 to_onnx.py:169-170) — the run 'name:' is "
            "unrestricted; only the export name is validated"
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
            "(every export.inputs port is a dataset-produced key)"
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


def reject_declared_outputs(outputs: Sequence[ExportOutput]) -> None:
    """Refuse a config-declared ``export.outputs`` section; raises `ConfigError`
    pointing at the `salt.outputs.OnnxExportSink` manifest instead.
    """
    if not outputs:
        return
    raise ConfigError(
        "export.outputs was REMOVED — the ONNX output manifest is declared by an "
        "OnnxExportSink naming the conversion outputs.* leaves.\n"
        "  fix: delete the export.outputs section; declare the conversion nodes + the "
        "OnnxExportSink instead, post-process with the sink's rename:, and inspect the "
        "assembled manifest with `salt export --manifest`"
    )


def resolve_export_config(export: ExportConfig, run_name: str) -> ExportConfig:
    """Validate the EXPORT-ONLY half of the export contract and fill its defaults.

    `outputs` is NOT handled here — the manifest derives from the folded
    `OnnxExportSink`; a non-empty parsed value is a hard error. Called
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
    reject_declared_outputs(export.outputs)
    model_name = validate_model_name(export.model_name or sanitised_model_name(run_name))
    if export.track_selection not in TRACK_SELECTIONS:
        raise ConfigError(
            f"export.track_selection {export.track_selection!r} is not a known Athena track "
            f"selection — choose from {list(TRACK_SELECTIONS)} (v1 to_onnx.py:24-35)"
        )
    if not export.inputs:
        raise ConfigError(
            "the ONNX export contract declares no input — set `inputs:` on the "
            "OnnxExportSink (outputs.<sink key>.init_args.inputs), a list of "
            "{port, name, sequence, dyn_axis} entries naming the Athena input tensors "
            "in positional order. A sink-only override file stacked as a second `-c` "
            "completes a run config trained without one."
        )
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


def _resolve_input(entry: ExportInput, track_selection: str) -> ExportInput:
    """Validate one input entry and fill its defaults; raises `ConfigError`
    on alias/sequence/name rule violations.
    """
    stream = stream_of_input_port(entry.port)
    if entry.alias is not None:
        # alias entries consume NO positional input: no graph tensor name, no dynamic
        # axis, no sequences (the GN3 global stream is a [B, F] vector)
        stream_of_input_port(entry.alias)
        if entry.name is not None:
            raise ConfigError(
                f"export input {entry.port!r}: alias entries bind from {entry.alias!r} and "
                "consume no positional ONNX input — drop 'name'"
            )
        if entry.sequence or entry.dyn_axis is not None:
            raise ConfigError(
                f"export input {entry.port!r}: sequence alias entries are not supported "
                "(the alias mechanism serves the GN3 global [B, F] vector)"
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


def _check_input_uniqueness(inputs: list[ExportInput]) -> None:
    """Reject duplicate input ports or graph tensor names; raises
    `ConfigError` naming the duplicate.
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
                "export input port — the alias binds from a declared input's tensor"
            )
