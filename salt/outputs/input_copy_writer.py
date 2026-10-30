"""`InputCopyWriter` — the ``outputs:`` section input-copy writer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import Tensor

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.outputs.output_schema import OutputField
from salt.outputs.run_task_output import OutputSectionWriter

_NEEDS_FIELD_LIST = (
    "InputCopyWriter needs `streams:` (copy every input field of each listed stream, "
    "e.g. `streams: [jets, tracks]`) and/or an explicit, non-empty field list per stream "
    "(`variables: {jets: [pt, eta]}`)"
)


class InputCopyWriter(OutputSectionWriter):
    """The ``outputs:`` section input-copy writer.

    Declares the copied source fields as ``labels.<stream>.<var>`` requires —
    the `Labels` disk-field producer serves them with its ``dtype_policy``
    (ints -> int64 by default, floats at file dtype; ``dtype_policy: file``
    keeps on-disk widths) — and forwards them to ``outputs.<stream>.<var>``.
    Never opens or indexes a source file. Global-stream copies work on every
    reader; sequence-stream copies written by `H5OutputSink` need a reader
    with an h5py-openable source (``reader.h5_source``) for the file token
    length — the same as per-token task outputs and pad masks. Column dtype is
    the bundle leaf's dtype (`OutputField.dtype=None`).

    The copied streams are `streams` in listed order when given, otherwise the
    keys of `variables`. A stream with a `variables` entry copies exactly
    those fields; a stream without one copies EVERY field of the source H5
    stream (file order) when the TEST reader has an h5py-openable source,
    else every field of its ``inputs.<stream>`` leaf (the data pipeline's
    `Features` variable list) — bound by `bind_stream_variables` before the
    writer's leaves are read. The values (and dtypes) always come from
    ``labels.<stream>.<var>``.

    Parameters
    ----------
    streams : Sequence[str] | None, optional
        Streams to copy, in column-group order; a stream without a
        `variables` entry copies every source-H5 field (else every
        ``inputs.<stream>`` field). ``None``
        (default) = the keys of `variables`. Non-empty, no duplicates.
    variables : Mapping[str, Sequence[str]] | None, optional
        Per-stream field list to copy, in the order each stream's fields are
        written; an explicit narrowing (non-empty, no duplicates). With
        `streams` given, every key must be one of `streams`.
    modes : Sequence[str] | None, optional
        The modes this writer runs in (``["test", "export"]`` subset; ``None``
        = both). Input copies are eval-H5 only regardless — `declare_io`
        yields nothing outside TEST.

    Raises
    ------
    ConfigError
        When neither `streams` nor `variables` is given, on an empty or
        duplicate-containing `streams` list or field list, or on a
        `variables` key outside `streams`.
    """

    name = "inputs_copy"
    """The section instance name (overridable by the config dict key)."""

    def __init__(
        self,
        streams: Sequence[str] | None = None,
        variables: Mapping[str, Sequence[str]] | None = None,
        modes: Sequence[str] | None = None,
    ) -> None:
        super().__init__(modes=modes)
        self.name = type(self).name
        if isinstance(streams, str):
            raise ConfigError(
                f"InputCopyWriter `streams:` must be a list of stream names, got the bare "
                f"string {streams!r} (did you mean `streams: [{streams}]`?)"
            )
        explicit = {s: list(v) for s, v in (variables or {}).items()}
        if not streams and not explicit:
            raise ConfigError(_NEEDS_FIELD_LIST)
        for stream, fields in explicit.items():
            if not fields:
                raise ConfigError(f"{_NEEDS_FIELD_LIST} (stream {stream!r} is empty)")
            if len(set(fields)) != len(fields):
                dup = sorted({f for f in fields if fields.count(f) > 1})
                raise ConfigError(f"{_NEEDS_FIELD_LIST} (stream {stream!r} repeats {dup})")
        if streams is None:
            self.streams: tuple[str, ...] = tuple(explicit)
        else:
            self.streams = tuple(streams)
            if not self.streams:
                raise ConfigError(
                    "InputCopyWriter `streams: []` is empty — list the streams to copy, or "
                    "omit `streams:` to copy the `variables:` streams"
                )
            for stream in self.streams:
                if self.streams.count(stream) > 1:
                    raise ConfigError(f"InputCopyWriter `streams:` repeats stream {stream!r}")
            for stream in explicit:
                if stream not in self.streams:
                    raise ConfigError(
                        f"InputCopyWriter `variables:` stream {stream!r} is not in `streams:` "
                        f"{list(self.streams)} — add it to `streams:` or drop its entry"
                    )
        # the resolved per-stream field lists in `streams` order; a streams-form
        # stream (no `variables:` entry) appears once `bind_stream_variables` fills it
        self.variables: dict[str, list[str]] = {
            s: explicit[s] for s in self.streams if s in explicit
        }

    def _unresolved(self) -> list[str]:
        """The copied streams with no field list yet (streams form, not yet bound)."""
        return [s for s in self.streams if s not in self.variables]

    def _require_resolved(self) -> None:
        """Raise `ConfigError` while any copied stream still lacks a field list."""
        if pending := self._unresolved():
            raise ConfigError(
                f"InputCopyWriter {self.name!r}: stream(s) {pending} have no field list yet — "
                "give each a `variables:` entry, or let the CLI bind it from the data "
                "pipeline's `inputs.<stream>` fields (`bind_stream_variables`)"
            )

    def bind_stream_variables(self, stream_fields: Mapping[str, Sequence[str]]) -> None:
        """Fill every streams-form stream's field list (source-H5 or ``inputs.<stream>`` fields).

        Explicit `variables` entries are never overridden and an already bound
        stream is left as is (idempotent).

        Parameters
        ----------
        stream_fields : Mapping[str, Sequence[str]]
            Stream name -> the field list to copy (the source-H5 stream's
            fields, else the declared fields of its ``inputs.<stream>`` leaf).

        Raises
        ------
        ConfigError
            When a pending stream is missing from `stream_fields` or has an
            empty field list there.
        """
        bound: dict[str, list[str]] = {}
        for stream in self._unresolved():
            fields = list(stream_fields.get(stream) or ())
            if not fields:
                raise ConfigError(
                    f"InputCopyWriter {self.name!r}: `streams:` lists {stream!r} without a "
                    f"`variables:` entry, but the data pipeline produces no `inputs.{stream}` "
                    f"field list (available: {sorted(stream_fields)}) — give it an explicit "
                    "`variables:` list"
                )
            bound[stream] = fields
        resolved = {**self.variables, **bound}
        self.variables = {s: resolved[s] for s in self.streams}

    def output_key(self, stream: str, var: str) -> str:
        """The ``outputs.<stream>.<var>`` leaf for one copied field."""
        return f"outputs.{stream}.{var}"

    def copy_leaves(self, mode: Mode) -> list[tuple[str, str]]:
        """The ``(stream, var)`` pairs copied in `mode` (empty when not `runs_in_mode`)."""
        if not self.runs_in_mode(mode):
            return []
        self._require_resolved()
        return [(s, v) for s, vs in self.variables.items() for v in vs]

    def declare_io(self, mode: Mode) -> IO:
        """Require ``labels.<stream>.<var>`` -> produce ``outputs.<stream>.<var>`` in TEST.

        Copies are eval-H5 only: export/ONNX has no dataset to demand from
        (``salt inference`` fills the copy columns itself), so any mode other
        than TEST declares empty IO.
        """
        if not self.runs_in_mode(mode) or not (mode & Mode.TEST):
            return IO(requires={}, produces={})
        pairs = self.copy_leaves(mode)
        req = {
            f"labels.{s}.{v}": TensorSpec(shape=None, dtype=None, kind="label") for s, v in pairs
        }
        prod = {
            self.output_key(s, v): TensorSpec(shape=None, dtype=None, kind="data") for s, v in pairs
        }
        return IO(requires=unflatten_spec(req), produces=unflatten_spec(prod))

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Clone each demanded ``labels.<stream>.<var>`` leaf into ``outputs.<stream>.<var>``."""
        del mode
        self._require_resolved()
        return {
            self.output_key(s, v): b.get(f"labels.{s}.{v}").clone()
            for s, vs in self.variables.items()
            for v in vs
        }

    def manifest_fields(self, mode: Mode) -> list[tuple[str, OutputField]]:
        """The copied columns' manifest entries, bare-named (`prefix=False`), `dtype=None`."""
        if not self.runs_in_mode(mode):
            return []
        self._require_resolved()
        return [
            (self.output_key(s, v), OutputField(h5_name=v, dtype=None, prefix=False, final=True))
            for s, vs in self.variables.items()
            for v in vs
        ]
