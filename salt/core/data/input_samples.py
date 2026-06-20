"""`InputSamples` — the pure-source data-sourcing setup module (plan-25 §3, W3.A).

`InputSamples` answers "which bytes back this reader, for this stage" at the
SHALLOWEST level of the superseding PATH chain (``pattern → vds_path →
staged_path``, plan-24 §3.5). It is a **setup-only** `DatasetModule`
(non-empty `declare_setup_io`, empty `declare_io`), so it lives in the
datamodule's ``_setup_modules`` namespace (plan-25 §3.6) and never reaches the
per-batch ``compile_plan``/``_check_all_modes_dead`` — it cannot trip
`AllModesDeadError`.

W3.A scope (the parity-preserving default, plan-25 §3.3 shape A): per-stage
literal files. ``InputSamples`` performs **path arithmetic only** — it selects
``files[stage]`` and emits it as ``source.<reader>.<stage>.pattern`` (PATH),
plus a whole-dict ``artifacts.<reader>.num`` (SCALAR) row-cap leaf written ONCE
across the ``setup("fit")`` train+val passes (the O-NUM-AMEND correction to
plan-24 §3.5, plan-25 §3.7). NO filesystem I/O, NO globbing — wildcards pass
through verbatim and are resolved downstream (the reader's own
``has_wildcard → create_vds`` in W3.A; the `VDS` module in W3.B).

The ``<reader>`` component is wired by `GraphDataModule` after the single-Reader
guard (plan-25 §3.8): the ctx does not exist at ``declare_setup_io`` time, so
the reader name is embedded at *declaration* time via a one-field assembly poke
(``input_samples._reader = self._reader_name``).
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
    """Pure-source setup module emitting per-stage source patterns (plan-25 §3, shape A).

    Parameters
    ----------
    files : dict[str, str | Path]
        Per-stage source description: ``{train, val, test}`` → a literal path
        or a wildcard string (resolved downstream — `InputSamples` never globs).
        Missing stages are simply not produced (e.g. a fit-only config may omit
        ``test``). Replaces the datamodule's ``train_file``/``val_file``/
        ``test_file`` kwargs (plan-25 §3.3).
    num : dict[str, int] | None, optional
        Per-stage row cap (``-1`` = all, v1 semantics), by default all ``-1``.
        Emitted ONCE as a whole-dict SCALAR leaf (plan-25 §3.7, O-NUM-AMEND);
        the reader indexes ``ctx.get("artifacts.<reader>.num")[stage]``.

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
        # the stage that carries the whole-dict `num` SCALAR leaf (written ONCE
        # per write-once ctx so setup("fit")'s train+val passes don't collide on
        # it, plan-25 §3.7 O-NUM-AMEND). Defaults to the first configured stage;
        # the datamodule overrides it per RUN via `set_num_stage` so a standalone
        # `salt test` (only the "test" pass runs) still emits the cap.
        self._first_stage: SetupStage = self._stages[0]
        # None => emit on no stage (the cap is already in a shared ctx from an
        # earlier pass); a SetupStage => emit the whole-dict on exactly that stage.
        self._num_stage: SetupStage | None = self._first_stage
        # wired by GraphDataModule.__init__ after the single-Reader guard
        # (plan-25 §3.8): the source.<reader>.* component the reader's handoff
        # later resolves. Unset until assembly; declare_setup_io needs it.
        self._reader: str | None = None

    def set_num_stage(self, stages: tuple[SetupStage, ...], already_emitted: bool) -> None:
        """Pin the stage carrying the whole-dict ``num`` leaf for the current pass.

        The whole-dict ``artifacts.<reader>.num`` SCALAR is written ONCE per
        write-once ctx (plan-25 §3.7). The datamodule calls this before each
        setup pass with the pass's stage tuple and whether the leaf is ALREADY in
        the shared ctx (e.g. a ``fit`` pass already wrote the whole ``{train, val,
        test}`` dict, so a later ``test`` pass on the same ctx must NOT re-emit
        it). When `already_emitted` is True, or no pass stage is configured here,
        ``num`` is emitted on NO stage (``_num_stage = None``); otherwise on the
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

    # -- per-batch face: EMPTY (this is a setup-only module, plan-25 §3.1) -----

    def declare_io(self, mode: Mode) -> IO:
        """Empty per-batch interface — `InputSamples` never produces tensors.

        Returns
        -------
        IO
            The empty default (no requires, no produces): this module is
            setup-only and is partitioned into ``_setup_modules`` (plan-25 §3.6).
        """
        del mode
        return IO()

    # -- setup face (plan-24 §4.1: once per stage, NOT per batch) --------------

    def _reader_name(self) -> str:
        """The wired reader name, or raise if assembly never poked it.

        Returns
        -------
        str
            The single-Reader instance name (`GraphDataModule._reader_name`).

        Raises
        ------
        RuntimeError
            If `_reader` was never wired (InputSamples used outside a
            GraphDataModule, or before assembly).
        """
        if self._reader is None:
            raise RuntimeError(
                f"InputSamples {self.name!r} has no reader name wired — it must be assembled "
                "by GraphDataModule (which sets `_reader` after the single-Reader guard, "
                "plan-25 §3.8)"
            )
        return self._reader

    def declare_setup_io(self, stage: SetupStage) -> SetupIO:
        """Declare the per-stage PATH produce (+ the whole-dict num SCALAR once).

        Source node: ``requires = {}`` always (a topo-sort root). For an active
        stage it produces ``source.<reader>.<stage>.pattern`` (PATH); the
        ``artifacts.<reader>.num`` whole-dict SCALAR leaf is produced ONCE, on
        the first active stage, so the ``setup("fit")`` train+val passes into one
        write-once ctx never collide on it (plan-25 §3.7).

        Returns
        -------
        SetupIO
            Empty for an inactive stage; otherwise the per-stage produces.
        """
        if stage not in self._stages:
            return SetupIO()
        reader = self._reader_name()
        flat: dict[str, SourceSpec] = {
            f"source.{reader}.{stage}.pattern": SourceSpec(kind="path", stages=(stage,))
        }
        if stage == self._num_stage:
            # whole-dict opaque SCALAR leaf — written once, indexed [stage] by
            # the reader handoff (plan-25 §3.7 O-NUM-AMEND).
            flat[f"artifacts.{reader}.num"] = SourceSpec(kind="scalar", stages=(self._first_stage,))
        return SetupIO(produces=unflatten_source_spec(flat))

    def setup(self, ctx: SetupBundle, stage: SetupStage) -> SetupBundle:
        """Resolve ``files[stage]`` onto the ctx (pure path arithmetic, no FS I/O).

        Writes ``source.<reader>.<stage>.pattern`` = ``str(files[stage])`` and,
        on the run's num-stage, the whole-dict ``artifacts.<reader>.num`` SCALAR
        leaf. A wildcard passes through verbatim — `InputSamples` never globs;
        the reader (W3.A) or the `VDS` module (W3.B) resolves it.

        The self-merge style (plan-25 §3.8 / kernel report note 1): this module
        owns the merge into `ctx` and returns `ctx`, so the flat dotted produces
        MUST be canonicalised to nested single-component form (`canonical_produced`
        — the same canonicalisation the per-batch executor applies) before
        `Bundle.merge`, whose component validator rejects bare dotted keys.

        Returns
        -------
        SetupBundle
            The same `ctx`, with this stage's produces merged in (write-once).
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
        """The whole-dict row-cap leaf, defaulting each active stage to ``-1``.

        Returns
        -------
        dict[str, int]
            ``{stage: num}`` for every active stage; ``-1`` (all) when unset.
        """
        return {stage: self._num.get(stage, -1) for stage in self._stages}


# the fixed deepest-key registry for the single-source PATH chain (plan-24 §3.5).
# W3.A: only `pattern` exists (VDS/ShmStage land in W3.B/W3.S). Listed
# shallow→deep so `deepest_source_path` picks the deepest PRESENT key.
SOURCE_REGISTRY: tuple[str, ...] = ("pattern", "vds_path", "staged_path")


def deepest_source_path(ctx: SetupBundle, reader: str, stage: SetupStage) -> str:
    """Walk the fixed registry over the resolved ctx and return the deepest path.

    The datamodule handoff glue (plan-25 §4.0, model (a)): after the setup pass
    completes, the reader binds the deepest PRESENT key of the
    ``pattern → vds_path → staged_path`` chain for ``(reader, stage)``. The
    reader is NOT a declared consumer of this chain — deepest-resolution is glue
    over the producer chain's already-completed write-once ctx, not a topo edge
    (so the reader's `declare_setup_io` need not vary with sibling-module
    presence).

    Parameters
    ----------
    ctx : SetupBundle
        The resolved setup bundle (post setup pass).
    reader : str
        The reader instance name.
    stage : SetupStage
        The setup stage to resolve.

    Returns
    -------
    str
        The deepest present PATH value for ``(reader, stage)``.

    Raises
    ------
    KeyError
        If no key of the chain is present (InputSamples never ran for `stage`).
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
    """Read the per-stage row cap off the whole-dict ``artifacts.<reader>.num`` leaf.

    Parameters
    ----------
    ctx : SetupBundle
        The resolved setup bundle.
    reader : str
        The reader instance name.
    stage : SetupStage
        The setup stage.

    Returns
    -------
    int
        The row cap for `stage` (``-1`` = all). Defaults to ``-1`` when the
        whole-dict leaf is absent or omits `stage`.
    """
    check_key_component(reader)
    key = f"artifacts.{reader}.num"
    if key not in ctx:
        return -1
    num = ctx.get(key)
    return int(num.get(stage, -1)) if isinstance(num, dict) else -1
