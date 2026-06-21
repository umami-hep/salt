"""`VDS` — the wildcard→virtual-dataset resolution setup module (plan-25 §5.1, W3.B).

`VDS` is the second link of the superseding PATH chain (``pattern → vds_path →
staged_path``, plan-24 §3.5): it consumes ``source.<reader>.<stage>.pattern``
(emitted by `InputSamples`) and produces ``source.<reader>.<stage>.vds_path``.
Like `InputSamples` it is a **setup-only** `DatasetModule` (non-empty
`declare_setup_io`, empty `declare_io`), so it lives in the datamodule's
``_setup_modules`` namespace (plan-25 §3.6) and — requiring ``pattern`` —
topo-sorts AFTER `InputSamples` in the setup plan.

What `VDS.setup` does (plan-25 §5.1, R-VDSROOT — HARD):

- **`vds_capable` reader + wildcard pattern** → build a real h5py virtual
  dataset (``create_vds``) and emit its path as ``vds_path``.
- **anything else** (non-wildcard pattern, OR a non-`vds_capable` ROOT reader
  whose value happens to glob) → IDENTITY edge: ``vds_path == pattern`` verbatim,
  and `create_vds` is NEVER called. A ROOT glob is a wildcard that would crash
  `create_vds`/`create_virtual_file`; the ROOT reader keeps its OWN native glob.

The build-vs-identity choice is gated on the reader's `vds_capable` flag
(O-VDS-CAP), NOT an `isinstance` check — the reader name + capability are wired
onto this module by `GraphDataModule` at assembly time (the same one-field
assembly poke `InputSamples._reader` uses, plan-25 §3.8), because the ctx does
not exist at `declare_setup_io` time.

`VDS` declares ``incompatible_with = ("ShmStage",)`` (plan-25 Rev-2): the
setup-plan compiler enforces mutual exclusion (staging a VDS copies h5py
pointers, not data — a design incompatibility). The rule LIVES ON `VDS`
because VDS owns the h5py-specific knowledge.
"""

from __future__ import annotations

from pathlib import Path

from salt.core.data.base import DatasetModule, SetupBundle
from salt.core.data.vds import create_vds, has_wildcard
from salt.core.graph.executor import canonical_produced
from salt.core.graph.setup_spec import (
    SETUP_STAGES,
    SetupIO,
    SetupStage,
    SourceSpec,
    unflatten_source_spec,
)
from salt.core.graph.spec import IO, Mode

__all__ = ["VDS"]


class VDS(DatasetModule):
    """Wildcard→VDS resolution setup module (plan-25 §5.1, shape B).

    Parameters
    ----------
    out : dict[str, str | Path] | None, optional
        Per-stage explicit VDS output paths (``{train, val, test}``,
        O-VDS-OUT). When a stage is configured here AND a real VDS is built
        (a wildcard pattern of a `vds_capable` reader), the VDS is written to
        this path; otherwise `create_vds` derives the v1 default next to the
        data. A stage omitted here (or `out=None`) uses the default. The
        legacy ``train_vds_path``/``val_vds_path``/``test_vds_path`` datamodule
        kwargs feed the auto-injected `VDS` for the migration window.

    Raises
    ------
    ValueError
        If `out` names a stage outside ``{train, val, test}``.
    """

    incompatible_with = ("ShmStage",)
    """`VDS` is structurally incompatible with `ShmStage` (plan-25 Rev-2).

    Staging a VDS to ``/dev/shm`` copies the VDS's h5py SOURCE POINTERS, not the
    underlying member-file data — the staged copy still references the original
    on-disk members, defeating the stage. Enforced by the setup-plan compiler's
    `_check_incompatibilities`.
    """

    def __init__(self, out: dict[str, str | Path] | None = None) -> None:
        super().__init__()
        # A deep dotted CLI override (e.g. --data.modules.vds.init_args.out.train=...)
        # arrives as a jsonargparse Namespace, not a dict; coerce so dict() and stage
        # iteration below work either way (mirror InputSamples.__init__).
        if out is not None and hasattr(out, "as_dict"):
            out = out.as_dict()
        bad = [stage for stage in (out or {}) if stage not in SETUP_STAGES]
        if bad:
            raise ValueError(
                f"VDS `out` has unknown stage(s) {bad}: must be a subset of "
                f"{list(SETUP_STAGES)} (plan-25 O-VDS-OUT)"
            )
        self._out: dict[str, str | Path] = dict(out) if out is not None else {}
        # wired by GraphDataModule at assembly time (plan-25 §3.8), mirroring
        # InputSamples._reader. `_reader` is the source.<reader>.* component;
        # `_vds_capable` gates build-vs-identity (O-VDS-CAP — the reader's flag,
        # not an isinstance check). Unset until assembly; declare_setup_io and
        # setup both need `_reader`.
        self._reader: str | None = None
        self._vds_capable: bool = False

    # -- per-batch face: EMPTY (this is a setup-only module, plan-25 §3.1) -----

    def declare_io(self, mode: Mode) -> IO:
        """Empty per-batch interface — `VDS` never produces tensors.

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
            If `_reader` was never wired (`VDS` used outside a GraphDataModule,
            or before assembly).
        """
        if self._reader is None:
            raise RuntimeError(
                f"VDS {self.name!r} has no reader name wired — it must be assembled "
                "by GraphDataModule (which sets `_reader`/`_vds_capable` after the "
                "single-Reader guard, plan-25 §3.8)"
            )
        return self._reader

    def declare_setup_io(self, stage: SetupStage) -> SetupIO:
        """Declare the per-stage ``pattern`` require and ``vds_path`` produce.

        Requires ``source.<reader>.<stage>.pattern`` (PATH, from `InputSamples`)
        and produces ``source.<reader>.<stage>.vds_path`` (PATH). Requiring
        ``pattern`` makes the topo-sort order `VDS` AFTER `InputSamples`. Active
        for every stage (a `VDS` resolves whichever stages `InputSamples`
        produced; an absent ``pattern`` would be caught by the compiler's
        connectivity check, not silently ignored).

        Returns
        -------
        SetupIO
            The per-stage require (`pattern`) and produce (`vds_path`).
        """
        reader = self._reader_name()
        requires = unflatten_source_spec(
            {f"source.{reader}.{stage}.pattern": SourceSpec(kind="path", stages=(stage,))}
        )
        produces = unflatten_source_spec(
            {f"source.{reader}.{stage}.vds_path": SourceSpec(kind="path", stages=(stage,))}
        )
        return SetupIO(requires=requires, produces=produces)

    def setup(self, ctx: SetupBundle, stage: SetupStage) -> SetupBundle:
        """Resolve ``pattern`` → ``vds_path`` for `stage` (build or identity).

        Reads ``source.<reader>.<stage>.pattern`` off the ctx. If the reader is
        `vds_capable` AND the pattern is a wildcard, builds a real VDS via
        `create_vds` (FileLock + ``.done`` makes it every-rank safe, plan-25
        §5.1 / R-DDP) and emits its path; otherwise the IDENTITY edge
        (``vds_path == pattern`` verbatim, `create_vds` NEVER called —
        R-VDSROOT). The whole-dict ``out[stage]`` overrides the default VDS path
        when configured.

        The self-merge style (plan-25 §3.8): this module owns the merge into
        `ctx` and returns `ctx`, so the flat dotted produces are canonicalised
        (`canonical_produced`) before `Bundle.merge`.

        Returns
        -------
        SetupBundle
            The same `ctx`, with this stage's ``vds_path`` merged in (write-once).
        """
        reader = self._reader_name()
        pattern = ctx.get(f"source.{reader}.{stage}.pattern")
        if self._vds_capable and has_wildcard(pattern):
            out = self._out.get(stage)
            vds = str(create_vds(Path(pattern), Path(out) if out is not None else None))
        else:
            # IDENTITY edge (R-VDSROOT): a non-wildcard pattern, OR a glob of a
            # non-vds_capable ROOT reader — pass through verbatim, NEVER calling
            # create_vds (a ROOT glob would crash create_virtual_file).
            vds = str(pattern)
        out_dict: dict[str, object] = {f"source.{reader}.{stage}.vds_path": vds}
        expected = set(out_dict)
        ctx.merge(
            canonical_produced(out_dict, expected, self.name), who=self.name, expected=expected
        )
        return ctx
