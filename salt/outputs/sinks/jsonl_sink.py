"""`JSONLOutputSink` — the worked example of a custom output format."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import IO, Mode, TensorSpec, flatten_spec, unflatten_spec
from salt.outputs.output_schema import OutputColumn
from salt.outputs.sinks.sink import RuntimeSink, SinkContext, collect_manifest_fields

__all__ = ["JSONLOutputSink"]

DEFAULT_OUTPUT = "{ckpt_dir}/{ckpt_stem}__test_{sample}.jsonl"
"""Output template — the eval-H5 name with a ``.jsonl`` suffix."""


def _jsonable(value: Any) -> Any:
    """Convert one numpy scalar/array to JSON, mapping NaN and +/-inf to ``null``.

    JSON has no NaN literal. ``json.dumps`` emits the non-standard ``NaN`` /
    ``Infinity`` tokens by default, which most parsers reject; this maps every
    non-finite float to ``None`` so the file is strict JSON, and the sink then
    dumps with ``allow_nan=False`` so any leak is a loud error, not silent
    corruption.
    """
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    return float(value)


class JSONLOutputSink(RuntimeSink):
    """Write the eval columns as newline-delimited JSON, one object per row.

    A deliberately minimal reference implementation of the `RuntimeSink`
    contract — the answer to "how do I write salt's outputs in some other
    format?". It is a real, tested sink, not pseudo-code: it resolves its
    column schema from the same bound ``outputs:`` section the H5 sink uses
    (so the two never disagree on names), demands exactly those leaves
    through `declare_io`, and streams one JSON object per row.

    It is an AUXILIARY sink: `is_test_sink` returns False so `H5OutputSink`
    stays the single TEST persistence sink that anchors the graph's boundary
    demand. Wire this alongside the implicit H5 sink in ``callbacks:`` and
    you get both files from one ``salt test``. Because its columns are
    derived from the same section, every leaf it reads is one the H5 sink
    already demanded — it never widens the plan.

    Output shape: one line per row (jet/event), each an object keyed by the
    eval column name. A single-suffix column yields a scalar; a multi-suffix
    column (e.g. the per-class probabilities) yields one key per class, matching
    the eval H5's flat column names exactly. A per-token leaf yields a nested
    list over the model's sequence positions (NOT re-expanded to the source
    file's padded length — the H5 sink's zero-padding exists to satisfy a fixed
    H5 dataset shape, which JSONL does not have).

    Parameters
    ----------
    columns : Sequence[str] | None, optional
        Flat eval column names to keep (e.g. ``["GN2_pb", "GN2_pc"]``), by
        default None = every column the section mints. Names not minted by the
        section are a `ConfigError` at run setup, so a typo fails loudly.
    output : str, optional
        Output path template accepting ``{ckpt_dir}`` / ``{ckpt_stem}`` /
        ``{sample}``, by default `DEFAULT_OUTPUT` — the eval H5's name with a
        ``.jsonl`` suffix, so the two land side by side.
    overwrite : bool, optional
        Whether to truncate an existing file, by default True. False refuses
        to clobber (raises `ConfigError`), which is the safer setting when the
        template is not checkpoint-unique.
    modes : Sequence[str] | None, optional
        Which planner modes to run in. REQUIRED when this sink is declared in
        the ``outputs:`` section: an auxiliary sink rides alongside the
        primary H5 sink rather than replacing it, so it states when it runs
        rather than inheriting a default. By default None (= ``[test]``).
    consumes : Sequence[str] | None, optional
        fnmatch patterns over the ``outputs.*`` leaf key narrowing the columns
        taken from the section, by default None (every column it mints).
        Complementary to `columns`, which selects by FLAT COLUMN NAME.

    Attributes
    ----------
    name : str
        The graph-node instance name (overridable by the section dict key).

    Notes
    -----
    Nothing is validated in the constructor. `ConfigError` is raised later, at
    run setup, when no ``outputs:`` section is bound, when `columns` names a
    column the section does not mint, when the context carries no checkpoint
    path, or
    when the target exists and `overwrite` is False.

    Examples
    --------
    Declare it in the ``outputs:`` section alongside the writers. The writers
    keep their declaration order (it is the eval-H5 column order); a sink is
    excluded from that ordering, so where it sits in the section is free::

        outputs:
          inputs_copy: {class_path: salt.outputs.InputCopyWriter, ...}
          run_tasks: {class_path: salt.outputs.RunTaskOutput, ...}
          jsonl:
            class_path: salt.outputs.JSONLOutputSink
            init_args:
              modes: [test]
              columns: [GN2_pb, GN2_pc, GN2_pu]

    ``salt test`` still writes its eval H5 — this sink is auxiliary and does
    not displace the H5 persistence sink.
    """

    name: str = "jsonl_output"
    """The graph-node instance name (overridable by the section dict key)."""

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        output: str = DEFAULT_OUTPUT,
        overwrite: bool = True,
        modes: Sequence[str] | None = None,
        consumes: Sequence[str] | None = None,
    ) -> None:
        super().__init__(modes=modes, consumes=consumes)
        self.columns = tuple(columns) if columns is not None else None
        self.output = output
        self.overwrite = overwrite
        self._output_section: Mapping[str, Any] | None = None
        # per-run state, reset at open_schema
        self._handle: TextIO | None = None
        self._run_name = "salt"
        self._resolved: tuple[OutputColumn, ...] | None = None
        self._rows_written = 0
        self.output_path: Path | None = None

    # -- section binding ---------------------------------------------------

    def bind_output_section(self, section: Mapping[str, Any]) -> None:
        """Capture the ``outputs:`` section this sink derives its columns from.

        Called by `SaltModule` / the CLI on every attached callback exposing
        this method, before any ``declare_io`` / ``writer_demand`` resolution.
        """
        self._output_section = section
        self._invalidate_manifest()

    def _invalidate_manifest(self) -> None:
        """Drop the cached column table after a manifest source rebinds."""
        self._resolved = None

    def _section_columns(self) -> tuple[OutputColumn, ...]:
        """Every TEST `OutputColumn` the bound section mints, in section order.

        Same walk (and therefore the same names and order) as the H5 sink's
        resolution: each writer's ``manifest_fields(Mode.TEST)`` contributes
        one column per ``outputs.*`` leaf, its suffixes in field order,
        narrowed by ``consumes:``.
        """
        if not self._output_section:
            raise ConfigError(
                "JSONLOutputSink has no `outputs:` section bound — it derives its columns "
                "from the section (RunTaskOutput + friends), so declare one; see docs/outputs.md"
            )
        by_key: dict[str, list[Any]] = {}
        order: list[str] = []
        for leaf_key, field in self._filter_consumed(
            collect_manifest_fields(self._output_section.values(), Mode.TEST)
        ):
            if field.h5_name is None:
                continue
            if leaf_key not in by_key:
                by_key[leaf_key] = []
                order.append(leaf_key)
            by_key[leaf_key].append(field)
        return tuple(
            OutputColumn(
                key=key,
                suffixes=[f.h5_name for f in by_key[key]],
                dtype=by_key[key][0].dtype,
                prefix=by_key[key][0].prefix,
            )
            for key in order
        )

    def _names_of(self, col: OutputColumn) -> list[str]:
        """A column's flat eval-H5 names under the current run name."""
        return col.column_names(self._run_name)

    def _ensure_columns(self) -> tuple[OutputColumn, ...]:
        """The columns this sink writes: the section's, narrowed by `columns`.

        Cached until the run name is known (`open_schema` re-resolves). A
        column is kept when ANY of its flat names is selected. Selection
        matches the flat name (``GN2_pb``) or the bare suffix (``pb``), so the
        filter is meaningful before the run name binds; an unmatched selection
        is only an error at `open_schema` (see `_validate_columns`).
        """
        if self._resolved is not None:
            return self._resolved
        available = self._section_columns()
        if self.columns is None:
            self._resolved = available
            return self._resolved
        wanted = set(self.columns)
        self._resolved = tuple(
            col
            for col in available
            if wanted.intersection(self._names_of(col)) or wanted.intersection(col.suffixes)
        )
        return self._resolved

    def _validate_columns(self) -> None:
        """Raise when `columns` names something the section does not mint."""
        if self.columns is None:
            return
        known: set[str] = set()
        for col in self._section_columns():
            known.update(self._names_of(col))
            known.update(str(s) for s in col.suffixes)
        if unknown := sorted(set(self.columns) - known):
            raise ConfigError(
                f"JSONLOutputSink: column(s) {unknown} are not minted by the outputs: section — "
                f"available columns are {sorted(known)}"
            )

    def _selected(self, col: OutputColumn, name: str, suffix: str) -> bool:
        """Whether one flat column name survives the `columns` filter."""
        del col
        return self.columns is None or name in self.columns or suffix in self.columns

    # -- graph node surface ------------------------------------------------

    def declare_io(self, mode: Mode) -> IO:
        """TEST requires the selected ``outputs.*`` leaves + ``meta.rows``; produces
        nothing (a terminal node). Every other mode declares nothing, so the
        planner prunes the sink outside TEST.
        """
        if not (mode & Mode.TEST):
            return IO(requires={}, produces={})
        req: dict[str, TensorSpec] = {
            col.key: TensorSpec(shape=None, dtype=None, kind="data")
            for col in self._ensure_columns()
        }
        req["meta.rows"] = TensorSpec(shape=None, dtype="int64", kind="meta")
        return IO(requires=unflatten_spec(req), produces={})

    def is_test_sink(self) -> bool:
        """Always False — auxiliary sink; `H5OutputSink` anchors the TEST demand."""
        return False

    def writer_demand(self, model_modules: Mapping[str, Any], reader: Any) -> dict[str, str]:
        """The sink's TEST ``declare_io`` requires, each mapped to a demander description."""
        del model_modules, reader
        who = "sink 'JSONLOutputSink' demanding"
        return {key: f"{who} {key}" for key in flatten_spec(self.declare_io(Mode.TEST).requires)}

    # -- lifecycle ---------------------------------------------------------

    def open_schema(self, ctx: SinkContext) -> None:
        """Resolve the output path + column schema and open the file for writing."""
        self._run_name = ctx.run_name
        self._resolved = None  # re-resolve: the run name feeds the column names
        self._validate_columns()
        self._ensure_columns()
        self.output_path = self._output_path(ctx)
        if self.output_path.exists() and not self.overwrite:
            raise ConfigError(
                f"JSONLOutputSink refuses to overwrite {self.output_path} — pass "
                "overwrite=true, or point `output` at a checkpoint-unique template"
            )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.output_path.open("w", encoding="utf-8")
        self._rows_written = 0

    def consume(self, bundle: Bundle) -> None:
        """Append one JSON object per row of this batch."""
        assert self._handle is not None, "consume before open_schema"
        rows = bundle.get("meta.rows")
        n = int(rows[1]) - int(rows[0])
        per_column: dict[str, np.ndarray] = {}
        for col in self._ensure_columns():
            values = bundle.get(col.key).detach().cpu().numpy()
            names = self._names_of(col)
            # a producer leaf is either [B, ...] with one trailing channel per
            # suffix, or a COLLAPSED single-suffix leaf ([B] global, [B, L]
            # per-token) whose last axis is not the channel axis.
            channelled = values.ndim >= 2 and values.shape[-1] == len(names)
            for idx, (name, suffix) in enumerate(zip(names, col.suffixes, strict=True)):
                if not self._selected(col, name, str(suffix)):
                    continue
                per_column[name] = values[..., idx] if channelled else values
        for row in range(n):
            record = {name: _jsonable(arr[row]) for name, arr in per_column.items()}
            self._handle.write(json.dumps(record, allow_nan=False) + "\n")
        self._rows_written += n

    def flush(self) -> None:
        """Close the file and report where it landed."""
        if self._handle is None:
            return
        self._handle.close()
        self._handle = None
        print(f"Wrote {self._rows_written:,} JSONL rows to {self.output_path}")

    def close_if_open(self) -> None:
        """Idempotently close a handle leaked by an interrupted test."""
        if self._handle is None:
            return
        self._handle.close()
        self._handle = None

    # -- helpers -----------------------------------------------------------

    def _output_path(self, ctx: SinkContext) -> Path:
        """Render the output template against the checkpoint and the test sample.

        Mirrors `H5OutputSink`'s template contract (same keys, same sample
        heuristic) so the JSONL lands beside the eval H5; raises `ConfigError`
        when ``ckpt_path`` is unset or the template names an unknown key.
        """
        ckpt_path = ctx.ckpt_path
        if ckpt_path is None:
            raise ConfigError(
                "JSONLOutputSink needs a checkpoint path — run salt test with --ckpt_path "
                "<ckpt> (the output file is named after the checkpoint)"
            )
        reader = ctx.reader
        src = getattr(reader, "filename", None) or getattr(reader, "source_path", None)
        stem = Path(src).stem if src is not None else self._run_name
        keys = {
            "ckpt_dir": str(Path(ckpt_path).parent),
            "ckpt_stem": Path(ckpt_path).stem,
            "sample": split[3] if len(split := stem.split("_")) == 4 else stem,
        }
        try:
            return Path(self.output.format(**keys))
        except KeyError as err:
            raise ConfigError(
                f"unknown JSONLOutputSink output template key {err} — available: {sorted(keys)}"
            ) from None
