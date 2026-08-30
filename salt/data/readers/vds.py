"""Virtual-dataset (VDS) creation for wildcard inputs (FileLock serialisation,
``.done`` marker, atomic publish, staleness check) plus the `VDS` setup module
resolving ``source.<reader>.<stage>.pattern`` to a built ``vds_path``.
"""

from __future__ import annotations

import os
import time
from contextlib import suppress
from pathlib import Path

from filelock import FileLock, Timeout
from ftag.vds import create_virtual_file

from salt.data.base import SaltDatasetModule, SetupBundle
from salt.graph.executor import canonical_produced
from salt.graph.setup_spec import (
    SETUP_STAGES,
    SetupIO,
    SetupStage,
    SourceSpec,
    unflatten_source_spec,
)
from salt.graph.spec import IO, Mode
from salt.utils import file_utils as fu

__all__ = ["VDS", "create_vds", "default_vds_path", "has_wildcard", "stage_file"]

_LOCK_TIMEOUT_S = 1800


def has_wildcard(path: Path | str) -> bool:
    """Check whether a path's filename contains glob-style wildcard characters.

    Returns
    -------
    bool
        True if the filename contains ``*``, ``?`` or ``[``.
    """
    name = Path(path).name
    return any(ch in name for ch in ("*", "?", "["))


def default_vds_path(pattern: Path) -> Path:
    """Derive the default VDS output path for a wildcard pattern.

    The wildcard marker is replaced by ``vds`` and a sibling folder
    ``<name>/vds.h5`` is created next to the matched files.

    Returns
    -------
    Path
        The default VDS file path.

    Raises
    ------
    PermissionError
        If the VDS folder cannot be created (use an explicit ``vds_path``).
    """
    vds_name = pattern.name.replace("*", "vds")
    vds_dir = pattern.parent / vds_name.removesuffix(".h5")
    try:
        vds_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as err:
        raise PermissionError(
            f"No permissions to create a VDS folder/file in {vds_dir}. "
            "Please use the custom vds_path option."
        ) from err
    return vds_dir / "vds.h5"


def _done_marker(vds_out: Path) -> Path:
    """Return the ``.done`` completion-marker path for a VDS file."""
    return vds_out.with_suffix(vds_out.suffix + ".done")


def _is_stale(out_fname: Path, members: list[Path]) -> bool:
    """Whether an existing VDS is missing or older than any member file."""
    if not out_fname.exists():
        return True
    vds_mtime = out_fname.stat().st_mtime
    return any(member.stat().st_mtime > vds_mtime for member in members)


def create_vds(pattern: Path, out_fname: Path | None = None) -> Path:
    """Create (or reuse) a VDS file for `pattern` in a multi-process-safe way.

    1. Fast path: ``.done`` marker exists and the VDS is not stale.
    2. Acquire ``<out>.lock`` (FileLock, 1800 s timeout).
    3. Re-check marker + staleness under the lock.
    4. Build into a pid-suffixed temp file, atomically rename into place.
    5. Write the ``.done`` marker (atomic rename too).

    Parameters
    ----------
    pattern : Path
        Glob-style pattern (wildcard in the filename component).
    out_fname : Path | None, optional
        Target VDS path; None derives the default next to the data.

    Returns
    -------
    Path
        Path to the final VDS file.

    Raises
    ------
    FileNotFoundError
        If the pattern matches no files.
    RuntimeError
        If acquiring the lock times out.
    """
    pattern = Path(pattern)
    members = sorted(pattern.parent.glob(pattern.name))
    if not members:
        raise FileNotFoundError(f"No files match wildcard: {pattern}")

    out_fname = default_vds_path(pattern) if out_fname is None else Path(out_fname)
    out_fname.parent.mkdir(parents=True, exist_ok=True)

    lock_path = out_fname.with_suffix(out_fname.suffix + ".lock")
    done_path = _done_marker(out_fname)

    # Fast path: already built and up to date
    if done_path.exists() and not _is_stale(out_fname, members):
        return out_fname

    lock = FileLock(str(lock_path))
    try:
        lock.acquire(timeout=_LOCK_TIMEOUT_S)
    except Timeout as exc:
        raise RuntimeError(f"Timeout waiting for VDS lock: {lock_path}") from exc

    try:
        # Re-check after acquiring the lock
        if done_path.exists():
            if not _is_stale(out_fname, members):
                return out_fname
            done_path.unlink()  # stale: invalidate and rebuild

        tmp_out = out_fname.with_name(out_fname.name + f".tmp.{os.getpid()}")
        created_path = Path(create_virtual_file(pattern=pattern, out_fname=tmp_out))

        # ensure atomic publish from tmp_out
        if (
            created_path.resolve() != tmp_out.resolve()
            and created_path.exists()
            and not tmp_out.exists()
        ):
            created_path.replace(tmp_out)
        tmp_out.replace(out_fname)

        marker_tmp = done_path.with_name(done_path.name + f".tmp.{os.getpid()}")
        marker_tmp.write_text(f"ok pid={os.getpid()} time={time.time()}\n")
        marker_tmp.replace(done_path)
        return out_fname
    finally:
        with suppress(Exception):
            lock.release()


def stage_file(src: Path, dst: Path) -> Path:
    """Copy `src` to `dst` once, multi-process-safe (the file-staging primitive).

    The on-disk twin of `create_vds`'s coordination, reusing the same FileLock +
    ``.done`` marker pattern so a DDP rank / dataloader-worker stampede copies
    the file exactly once (the rest skip via the marker), without needing a
    trainer handle: the FileLock serialises every contender and the ``.done``
    marker short-circuits the followers. The actual byte copy delegates to
    `salt.utils.file_utils.copy_file` (a no-op when ``dst`` exists).

    Parameters
    ----------
    src : Path
        Source file to stage.
    dst : Path
        Destination path (its parent is created if missing).

    Returns
    -------
    Path
        ``dst`` (the staged copy).

    Raises
    ------
    RuntimeError
        If acquiring the lock times out.
    """
    src = Path(src)
    dst = Path(dst)
    if src.resolve() == dst.resolve():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)

    done_path = dst.with_suffix(dst.suffix + ".done")
    # Fast path: already staged (marker present and the copy is in place).
    if done_path.exists() and dst.is_file():
        return dst

    lock_path = dst.with_suffix(dst.suffix + ".lock")
    lock = FileLock(str(lock_path))
    try:
        lock.acquire(timeout=_LOCK_TIMEOUT_S)
    except Timeout as exc:
        raise RuntimeError(f"Timeout waiting for staging lock: {lock_path}") from exc

    try:
        # Re-check under the lock — a contender may have just finished.
        if done_path.exists() and dst.is_file():
            return dst
        fu.copy_file(src, dst)  # no-op if dst already present (file_utils.copy_file)
        marker_tmp = done_path.with_name(done_path.name + f".tmp.{os.getpid()}")
        marker_tmp.write_text(f"ok pid={os.getpid()} time={time.time()}\n")
        marker_tmp.replace(done_path)
        return dst
    finally:
        with suppress(Exception):
            lock.release()


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
        # wired by SaltDataModule at assembly time, mirroring
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
                "by SaltDataModule (which sets `_reader`/`_vds_capable` after the "
                "single-Reader guard)"
            )
        return self._reader

    def declare_setup_io(self, stage: SetupStage) -> SetupIO:
        """Declare the per-stage ``pattern`` require and ``vds_path`` produce.

        Requires ``source.<reader>.<stage>.pattern`` (from `InputSamples`) and
        produces ``source.<reader>.<stage>.vds_path``; active for every stage.
        """
        reader = self._reader_name()
        requires = unflatten_source_spec({
            f"source.{reader}.{stage}.pattern": SourceSpec(kind="path", stages=(stage,))
        })
        produces = unflatten_source_spec({
            f"source.{reader}.{stage}.vds_path": SourceSpec(kind="path", stages=(stage,))
        })
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
