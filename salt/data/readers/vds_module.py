"""`VDS` — the wildcard->virtual-dataset resolution setup module.

Consumes ``source.<reader>.<stage>.pattern`` and emits ``vds_path`` (a built
h5py VDS for vds-capable readers + wildcards; an identity edge otherwise).
"""

from __future__ import annotations

from pathlib import Path

from salt.data.base import SaltDatasetModule, SetupBundle
from salt.data.readers.vds import create_vds, has_wildcard
from salt.graph.executor import canonical_produced
from salt.graph.setup_spec import (
    SETUP_STAGES,
    SetupIO,
    SetupStage,
    SourceSpec,
    unflatten_source_spec,
)
from salt.graph.spec import IO, Mode

__all__ = ["VDS"]


class VDS(SaltDatasetModule):
    """Wildcard->VDS resolution setup module.

    Parameters
    ----------
    out : dict[str, str | Path] | None, optional
        Per-stage explicit VDS output paths (``{train, val, test}``). When a
        stage is configured here and a real VDS is built (a wildcard pattern
        of a `vds_capable` reader), the VDS is written to this path;
        otherwise `create_vds` derives the default next to the data. A stage
        omitted here (or `out=None`) uses the default. The legacy
        ``train_vds_path``/``val_vds_path``/``test_vds_path`` datamodule
        kwargs feed the auto-injected `VDS` for the migration window.

    Raises
    ------
    ValueError
        If `out` names a stage outside ``{train, val, test}``.
    """

    incompatible_with = ("ShmStage",)
    """`VDS` is structurally incompatible with `ShmStage`.

    Staging a VDS to ``/dev/shm`` copies the VDS's h5py source pointers, not
    the underlying member-file data — the staged copy still references the
    original on-disk members, defeating the stage. Enforced by the
    setup-plan compiler's `_check_incompatibilities`.
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
        # wired by GraphDataModule at assembly time, mirroring
        # InputSamples._reader. `_reader` is the source.<reader>.* component;
        # `_vds_capable` gates build-vs-identity (the reader's flag, not an
        # isinstance check). Unset until assembly; declare_setup_io and setup
        # both need `_reader`.
        self._reader: str | None = None
        self._vds_capable: bool = False

    def declare_io(self, mode: Mode) -> IO:
        """Empty per-batch interface — `VDS` never produces tensors; setup-only,
        partitioned into `_setup_modules`.
        """
        del mode
        return IO()

    def _reader_name(self) -> str:
        """The wired reader name, or raise `RuntimeError` if assembly never wired `_reader`."""
        if self._reader is None:
            raise RuntimeError(
                f"VDS {self.name!r} has no reader name wired — it must be assembled "
                "by GraphDataModule (which sets `_reader`/`_vds_capable` after the "
                "single-Reader guard, plan-25 §3.8)"
            )
        return self._reader

    def declare_setup_io(self, stage: SetupStage) -> SetupIO:
        """Declare the per-stage ``pattern`` require and ``vds_path`` produce.

        Requires ``source.<reader>.<stage>.pattern`` (from `InputSamples`) and
        produces ``source.<reader>.<stage>.vds_path``; active for every stage.
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
        """Resolve ``pattern`` -> ``vds_path`` for `stage` (build or identity).

        Builds a real VDS via `create_vds` when the reader is `vds_capable`
        and the pattern is a wildcard; otherwise the identity edge
        (``vds_path == pattern`` verbatim, `create_vds` never called). The
        whole-dict ``out[stage]`` overrides the default VDS path when configured.
        """
        reader = self._reader_name()
        pattern = ctx.get(f"source.{reader}.{stage}.pattern")
        if self._vds_capable and has_wildcard(pattern):
            out = self._out.get(stage)
            vds = str(create_vds(Path(pattern), Path(out) if out is not None else None))
        else:
            # identity edge: a non-wildcard pattern, or a glob of a
            # non-vds_capable ROOT reader — pass through verbatim, never
            # calling create_vds (a ROOT glob would crash create_virtual_file).
            vds = str(pattern)
        out_dict: dict[str, object] = {f"source.{reader}.{stage}.vds_path": vds}
        expected = set(out_dict)
        ctx.merge(
            canonical_produced(out_dict, expected, self.name), who=self.name, expected=expected
        )
        return ctx
