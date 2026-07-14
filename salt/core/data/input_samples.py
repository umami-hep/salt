"""`InputSamples` — the pure-source data-sourcing setup module: selects
``files[stage]`` and emits ``source.<reader>.<stage>.pattern`` plus the
``artifacts.<reader>.num`` row-cap leaf (path arithmetic only, no I/O).
"""

from __future__ import annotations

from pathlib import Path

from salt.core.data.base import DatasetModule, SetupBundle
from salt.core.graph.executor import canonical_produced
from salt.core.graph.setup_spec import (
    SETUP_STAGES,
    SetupIO,
    SetupStage,
    SourceSpec,
    unflatten_source_spec,
)
from salt.core.graph.spec import IO, Mode, check_key_component

__all__ = ["InputSamples"]


class InputSamples(DatasetModule):
    """Pure-source setup module emitting per-stage source patterns.

    Parameters
    ----------
    files : dict[str, str | Path]
        Per-stage source description: ``{train, val, test}`` -> a literal path
        or a wildcard string (resolved downstream — `InputSamples` never globs).
        Missing stages are simply not produced (e.g. a fit-only config may omit
        ``test``). Replaces the datamodule's ``train_file``/``val_file``/
        ``test_file`` kwargs.
    num : dict[str, int] | None, optional
        Per-stage row cap (``-1`` = all), by default all ``-1``. Emitted once
        as a whole-dict scalar leaf; the reader indexes
        ``ctx.get("artifacts.<reader>.num")[stage]``.

    Raises
    ------
    ValueError
        If `files` is empty, or names a stage outside ``{train, val, test}``.
    """

    def __init__(
        self,
        files: dict[str, str | Path],
        num: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        # A deep dotted CLI override (e.g. --data.modules.input_samples.init_args.files.train=...)
        # arrives as a jsonargparse Namespace, not a dict; coerce both maps so dict()
        # and stage iteration below work either way.
        if hasattr(files, "as_dict"):
            files = files.as_dict()
        if num is not None and hasattr(num, "as_dict"):
            num = num.as_dict()
        if not files:
            raise ValueError(
                "InputSamples requires a non-empty `files` map "
                "(e.g. {train: ..., val: ..., test: ...}) (plan-25 §3.3)"
            )
        bad = [stage for stage in files if stage not in SETUP_STAGES]
        if bad:
            raise ValueError(
                f"InputSamples `files` has unknown stage(s) {bad}: must be a subset of "
                f"{list(SETUP_STAGES)} (plan-25 §3.3)"
            )
        self._files: dict[str, str | Path] = dict(files)
        self._num: dict[str, int] = dict(num) if num is not None else {}
        # the active stages, in canonical order — the only stages this module
        # produces a `pattern` for.
        self._stages: tuple[SetupStage, ...] = tuple(
            stage for stage in SETUP_STAGES if stage in self._files
        )
        # the stage that carries the whole-dict `num` scalar leaf (written once
        # per write-once ctx so setup("fit")'s train+val passes don't collide on
        # it). Defaults to the first configured stage; the datamodule overrides
        # it per run via `set_num_stage` so a standalone `salt test` (only the
        # "test" pass runs) still emits the cap.
        self._first_stage: SetupStage = self._stages[0]
        # None => emit on no stage (the cap is already in a shared ctx from an
        # earlier pass); a SetupStage => emit the whole-dict on exactly that stage.
        self._num_stage: SetupStage | None = self._first_stage
        # wired by GraphDataModule.__init__ after the single-Reader guard: the
        # source.<reader>.* component the reader's handoff later resolves.
        # Unset until assembly; declare_setup_io needs it.
        self._reader: str | None = None

    def set_num_stage(self, stages: tuple[SetupStage, ...], already_emitted: bool) -> None:
        """Pin the stage carrying the whole-dict ``num`` leaf for the current pass.

        The whole-dict ``artifacts.<reader>.num`` scalar is written once per
        write-once ctx. The datamodule calls this before each setup pass with
        the pass's stage tuple and whether the leaf is already in the shared
        ctx (e.g. a ``fit`` pass already wrote the whole ``{train, val, test}``
        dict, so a later ``test`` pass on the same ctx must not re-emit it).
        When `already_emitted` is True, or no pass stage is configured here,
        ``num`` is emitted on no stage (``_num_stage = None``); otherwise on the
        first configured pass stage.

        Parameters
        ----------
        stages : tuple[SetupStage, ...]
            The stages of the current setup pass, in run order.
        already_emitted : bool
            Whether the ``num`` leaf is already present in the shared ctx.
        """
        if already_emitted:
            self._num_stage = None
            return
        self._num_stage = next((s for s in stages if s in self._stages), None)

    def declare_io(self, mode: Mode) -> IO:
        """Empty per-batch interface — `InputSamples` never produces tensors;
        setup-only, partitioned into `_setup_modules`.
        """
        del mode
        return IO()

    def _reader_name(self) -> str:
        """The wired reader name, or raise `RuntimeError` if assembly never wired `_reader`."""
        if self._reader is None:
            raise RuntimeError(
                f"InputSamples {self.name!r} has no reader name wired — it must be assembled "
                "by GraphDataModule (which sets `_reader` after the single-Reader guard, "
                "plan-25 §3.8)"
            )
        return self._reader

    def declare_setup_io(self, stage: SetupStage) -> SetupIO:
        """Declare the per-stage path produce (+ the whole-dict num scalar once).

        Source node (``requires={}``); an active stage produces
        ``source.<reader>.<stage>.pattern``, and the ``artifacts.<reader>.num``
        whole-dict scalar is produced once, on the first active stage.
        """
        if stage not in self._stages:
            return SetupIO()
        reader = self._reader_name()
        flat: dict[str, SourceSpec] = {
            f"source.{reader}.{stage}.pattern": SourceSpec(kind="path", stages=(stage,))
        }
        if stage == self._num_stage:
            # whole-dict opaque scalar leaf — written once, indexed [stage] by
            # the reader handoff.
            flat[f"artifacts.{reader}.num"] = SourceSpec(kind="scalar", stages=(self._first_stage,))
        return SetupIO(produces=unflatten_source_spec(flat))

    def setup(self, ctx: SetupBundle, stage: SetupStage) -> SetupBundle:
        """Resolve ``files[stage]`` onto the ctx (pure path arithmetic, no FS I/O).

        Writes ``source.<reader>.<stage>.pattern`` and, on the run's num-stage,
        the whole-dict ``artifacts.<reader>.num`` scalar; a wildcard passes
        through verbatim (InputSamples never globs — the reader or `VDS` resolves it).
        """
        if stage not in self._stages:
            return ctx
        reader = self._reader_name()
        out: dict[str, object] = {f"source.{reader}.{stage}.pattern": str(self._files[stage])}
        if stage == self._num_stage:
            out[f"artifacts.{reader}.num"] = self._num_dict()
        expected = set(out)
        ctx.merge(canonical_produced(out, expected, self.name), who=self.name, expected=expected)
        return ctx

    def _num_dict(self) -> dict[str, int]:
        """The whole-dict row-cap leaf (``{stage: num}``), defaulting each active
        stage to ``-1``.
        """
        return {stage: self._num.get(stage, -1) for stage in self._stages}


# the fixed deepest-key registry for the single-source path chain. Listed
# shallow->deep so `deepest_source_path` picks the deepest present key.
SOURCE_REGISTRY: tuple[str, ...] = ("pattern", "vds_path", "staged_path")


def deepest_source_path(ctx: SetupBundle, reader: str, stage: SetupStage) -> str:
    """Walk the fixed ``pattern -> vds_path -> staged_path`` registry over the resolved
    ctx and return the deepest present path for ``(reader, stage)`` (raises `KeyError`
    if none of the chain is present).
    """
    check_key_component(reader)
    found: str | None = None
    for level in SOURCE_REGISTRY:
        key = f"source.{reader}.{stage}.{level}"
        if key in ctx:
            found = ctx.get(key)
    if found is None:
        raise KeyError(
            f"no resolved source for reader {reader!r} stage {stage!r}: the setup pass produced "
            f"none of {[f'source.{reader}.{stage}.{lvl}' for lvl in SOURCE_REGISTRY]}"
        )
    return found


def source_num(ctx: SetupBundle, reader: str, stage: SetupStage) -> int:
    """The per-stage row cap from the whole-dict ``artifacts.<reader>.num`` leaf
    (``-1`` = all; defaults to ``-1`` when the leaf is absent or omits `stage`).
    """
    check_key_component(reader)
    key = f"artifacts.{reader}.num"
    if key not in ctx:
        return -1
    num = ctx.get(key)
    return int(num.get(stage, -1)) if isinstance(num, dict) else -1
