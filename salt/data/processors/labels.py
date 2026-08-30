"""`Labels` — the demand-driven disk-label producer over ``labels.**``."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import ClassVar, Literal

import numpy as np

from salt.data.base import Processor, WorkerCtx
from salt.graph.errors import ConfigError
from salt.graph.planner import PlanStep
from salt.graph.spec import IO, KEY_SEP, Mode, TensorSpec, unflatten_spec


class Labels(Processor):
    """Demand-driven label producer over ``labels.**``.

    Declares the wildcard pattern ``labels.**`` (framework producer); the
    planner narrows it to the keys concretely demanded per mode and
    validates every narrowed key against the dataset schema — this keeps
    "tasks are the source of truth for which labels get loaded" without a
    CLI mutation hook. The narrowed key set is learned at bind time from the
    module's own plan step.

    Integer labels become int64 (``dtype_policy: int64-for-int``), everything
    else keeps the file dtype; sentinel values (-1 padding, -2/-3 type codes)
    pass through untouched. Malformed-value recovery (e.g.
    ``ftagTruthOriginLabel``) is the opt-in ``valid_ranges`` config with
    explicit ranges.

    In a mode where no label is demanded the module narrows to nothing and
    runs as a no-op (the kernel keeps wildcard producers with bound
    requires alive as terminal consumers); it contributes no read fields,
    so no I/O is wasted.

    On-the-fly ftag relabelling is NOT this module's job: the ftag
    ``Labeller`` derived-label producer lives in the standalone
    `FtagLabeller` processor, wired as its own ``data.modules`` key. Its
    concrete ``labels.<stream>.<label>`` produce beats this module's
    ``labels.**`` wildcard (planner concrete-beats-wildcard rule), so
    `Labels` serves every OTHER demanded label from disk and `FtagLabeller`
    owns the relabelled one. `Labels` is pure disk-label extraction.

    Parameters
    ----------
    streams : Sequence[str] | None, optional
        The streams this producer can serve (its ``raw.<stream>`` requires).
        None (the config default) defers to the framework: `SaltDataset`
        calls `bind_streams` with the reader's stream list before plan
        compilation (config-derived, static).
    dtype_policy : Literal["int64-for-int", "file"], optional
        ``int64-for-int`` (default) casts integer labels to int64; ``file``
        keeps the on-disk dtype for everything.
    valid_ranges : Mapping[str, Sequence[int]] | None, optional
        Label name -> inclusive ``[lo, hi]`` valid range (e.g.
        ``{ftagTruthOriginLabel: [-1, 7]}``). Out-of-range values raise, or
        are mapped to -1 when ``recover_malformed`` is set.
    recover_malformed : bool, optional
        Recover (warn + map to -1) instead of raising on out-of-range label
        values, by default False.

    Raises
    ------
    ConfigError
        On an unknown ``dtype_policy`` or malformed ``valid_ranges``.
    """

    allow_wildcards: ClassVar[bool] = True  # framework wildcard capability

    def __init__(
        self,
        streams: Sequence[str] | None = None,
        dtype_policy: Literal["int64-for-int", "file"] = "int64-for-int",
        valid_ranges: Mapping[str, Sequence[int]] | None = None,
        recover_malformed: bool = False,
    ) -> None:
        super().__init__()
        if dtype_policy not in {"int64-for-int", "file"}:
            raise ConfigError(
                f"unknown dtype_policy {dtype_policy!r}: expected 'int64-for-int' or 'file'"
            )
        self._streams: tuple[str, ...] | None = tuple(streams) if streams is not None else None
        self.dtype_policy = dtype_policy
        self.valid_ranges: dict[str, tuple[float, float]] = {}
        for label, bounds in (valid_ranges or {}).items():
            bounds = tuple(bounds)  # noqa: PLW2901
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ConfigError(
                    f"valid_ranges[{label!r}] must be an inclusive [lo, hi] pair, got {bounds}"
                )
            self.valid_ranges[label] = bounds
        self.recover_malformed = recover_malformed
        self._targets: tuple[tuple[str, str, str], ...] | None = None

    def bind_streams(self, streams: Sequence[str]) -> None:
        """Framework hook: adopt the reader's stream list when ``streams`` is unset.

        Called by `SaltDataset` before plan compilation (config-derived,
        static — no file I/O). A no-op when streams were configured
        explicitly.
        """
        if self._streams is None:
            self._streams = tuple(streams)

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``raw.<s>`` requires plus the ``labels.**`` wildcard produce (raises
        `ConfigError` if streams are still unresolved).
        """
        del mode
        if self._streams is None:
            raise ConfigError(
                f"Labels module {self.name!r} has no streams — set streams: explicitly or "
                "compile via SaltDataset (which forwards the reader's streams)"
            )
        requires = {f"raw.{stream}": TensorSpec(kind="data") for stream in self._streams}
        produces = {"labels.**": TensorSpec(kind="label")}
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def _parse_targets(self, step: PlanStep) -> tuple[tuple[str, str, str], ...]:
        """Parse the narrowed produces into ``(key, stream, label)`` triples."""
        targets: list[tuple[str, str, str]] = []
        for key in step.produces:
            parts = key.split(KEY_SEP)
            if len(parts) != 3 or parts[0] != "labels":
                raise ConfigError(
                    f"Labels module {self.name!r}: narrowed key {key!r} is not of the form "
                    "labels.<stream>.<label>"
                )
            _, stream, label = parts
            if self._streams is not None and stream not in self._streams:
                raise ConfigError(
                    f"Labels module {self.name!r}: key {key!r} names stream {stream!r} "
                    f"outside its served streams {list(self._streams)}"
                )
            targets.append((key, stream, label))
        return tuple(targets)

    def bind(self, ctx: WorkerCtx) -> None:
        """Learn the narrowed key set from this module's own plan step."""
        assert ctx.step is not None
        self._targets = self._parse_targets(ctx.step)

    def read_fields(self, step: PlanStep) -> dict[str, dict[str, str]]:
        """Demand exactly the narrowed label fields from the reader: each narrowed
        ``labels.<stream>.<label>`` maps to a ``raw.<stream>.<label>`` read.
        """
        out: dict[str, dict[str, str]] = {}
        for _key, stream, label in self._parse_targets(step):
            out.setdefault(stream, {})[label] = self.name
        return out

    def process(self, batch, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Extract, range-check, and cast the demanded labels; raises `ValueError` on
        out-of-range ``valid_ranges`` values unless `recover_malformed` (then mapped to -1).
        """
        del rows, mode
        assert self._targets is not None, "Labels.process called before bind()"
        out: dict[str, np.ndarray] = {}
        for key, stream, label in self._targets:
            values = batch.get(f"raw.{stream}")[label]
            if (bounds := self.valid_ranges.get(label)) is not None:
                bad = (values < bounds[0]) | (values > bounds[1])
                if bad.any():
                    if not self.recover_malformed:
                        raise ValueError(
                            f"Malformed {label!r} values outside [{bounds[0]}, {bounds[1]}] "
                            f"with values and counts: {np.unique(values, return_counts=True)} "
                            "Recover flag is off, failing."
                        )
                    warnings.warn(
                        f"Malformed {label!r} values outside [{bounds[0]}, {bounds[1]}] "
                        f"with values and counts: {np.unique(values, return_counts=True)} "
                        "Recover flag is on, converting to invalid and continuing.",
                        stacklevel=2,
                    )
                    values = np.where(bad, -1, values)
            if self.dtype_policy == "int64-for-int" and np.issubdtype(values.dtype, np.integer):
                out[key] = values.astype(np.int64)  # always copies
            else:
                out[key] = np.array(values, copy=True)  # keep file dtype (possibly f2)
        return out
