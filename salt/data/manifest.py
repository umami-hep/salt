"""`CorpusManifest` — the offline artifact a streaming run plans its shards from,
so no rank has to open a data file to learn how many rows exist or where they are.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from salt.data.base import Reader, RowBlock
from salt.graph.errors import ConfigError
from salt.logging import get_logger

__all__ = ["AUTO_MANIFEST", "MANIFEST_VERSION", "CorpusManifest", "ManifestEntry", "build_manifest"]

_LOG = get_logger(__name__)

MANIFEST_VERSION = 1
"""Artifact format version. A mismatch is a miss, never a best-effort read."""

AUTO_MANIFEST = "auto"
"""The `manifest:` config value that resolves, validates and builds on demand."""

MANIFEST_CACHE_ENV = "SALT_MANIFEST_CACHE"
"""Environment variable overriding the fallback manifest cache directory."""

_MIN_ROOT_PARTS = 3
"""Shallowest directory that may hold a manifest (``/a/b`` == 3 parts)."""


@dataclass(frozen=True)
class ManifestEntry:
    """One sequential read unit, described without reference to its contents.

    `path`, `size` and `mtime_ns` are what let a manifest be rejected when the
    corpus underneath it changed, at three `stat` calls per file rather than a
    re-read.
    """

    group: int
    start: int
    stop: int
    path: str = ""
    size: int = -1
    mtime_ns: int = -1

    @property
    def n_rows(self) -> int:
        """Rows this entry covers."""
        return self.stop - self.start

    def to_block(self) -> RowBlock:
        """The `RowBlock` the streaming partition works in."""
        return RowBlock(group=self.group, start=self.start, stop=self.stop)


@dataclass
class CorpusManifest:
    """Everything the shard planner needs about a corpus, and nothing else.

    Deliberately NOT an index: it carries per-block row counts (a few hundred
    integers for a 236-file corpus), not per-row positions. That is the whole
    point — the map-style path's row-granular index is ~160 MB of int64 per
    reader process, which is affordable once and impossible at the thousands of
    concurrent readers a O(100-1000)-GPU job creates.

    Parameters
    ----------
    entries : list[ManifestEntry]
        The blocks, ascending within each group.
    group_names : list[str]
        Sample name per group id (``["default"]`` for a single-sample reader).
    schema_hash : str
        Digest of the served schema; a manifest is only valid for the reader
        configuration that produced it.
    version : int, optional
        Artifact format version, by default `MANIFEST_VERSION`.
    meta : dict, optional
        Free-form provenance (salt/uproot versions, build time, source glob).
    """

    entries: list[ManifestEntry]
    group_names: list[str] = field(default_factory=lambda: ["default"])
    schema_hash: str = ""
    version: int = MANIFEST_VERSION
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def group_rows(self) -> list[int]:
        """Total rows per group id."""
        out = [0] * len(self.group_names)
        for entry in self.entries:
            out[entry.group] += entry.n_rows
        return out

    @property
    def n_rows(self) -> int:
        """Total rows across every group."""
        return sum(entry.n_rows for entry in self.entries)

    def blocks(self) -> list[RowBlock]:
        """The blocks, in the streaming layer's own type."""
        return [entry.to_block() for entry in self.entries]

    def save(self, path: str | Path) -> Path:
        """Write the manifest as JSON, atomically (temp file + replace).

        A half-written manifest would shard against a truncated corpus and
        silently train on part of it. The temp name carries the writer's pid, so
        racing builders each write their own and one `os.replace` wins whole:
        they can waste work, never corrupt it.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "group_names": list(self.group_names),
            "schema_hash": self.schema_hash,
            "meta": dict(self.meta),
            "entries": [asdict(entry) for entry in self.entries],
        }
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return path

    @classmethod
    def load(cls, path: str | Path) -> CorpusManifest:
        """Read a manifest.

        Raises
        ------
        ConfigError
            If the file is unreadable, malformed, or a different format version.
        """
        path = Path(path)
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"unreadable corpus manifest {path}: {exc}") from exc
        if payload.get("version") != MANIFEST_VERSION:
            raise ConfigError(
                f"corpus manifest {path} is format version {payload.get('version')!r}, "
                f"this salt reads {MANIFEST_VERSION} — rebuild it"
            )
        return cls(
            entries=[ManifestEntry(**entry) for entry in payload["entries"]],
            group_names=list(payload.get("group_names", ["default"])),
            schema_hash=str(payload.get("schema_hash", "")),
            version=int(payload["version"]),
            meta=dict(payload.get("meta", {})),
        )

    def validate(
        self,
        schema_hash: str | None = None,
        sources: Sequence[str | Path] | None = None,
    ) -> list[str]:
        """Structural + filesystem checks; returns the reasons it is stale (empty = usable).

        Opens NO data file — only what a `stat` can decide (every recorded file
        still present at the same size and mtime), plus the schema hash, so a
        manifest built for a different reader configuration cannot be used by
        accident. `sources` additionally catches the one change no per-file stat
        can see: a file added to or removed from the glob leaves every recorded
        file untouched.
        """
        problems: list[str] = []
        if schema_hash is not None and self.schema_hash and schema_hash != self.schema_hash:
            problems.append(
                f"schema hash {schema_hash} != manifest's {self.schema_hash} "
                "(reader configuration changed)"
            )
        if sources is not None:
            recorded = [str(s) for s in self.meta.get("sources", [])]
            wanted = [str(s) for s in sources]
            if recorded != wanted:
                problems.append(
                    f"corpus file list changed ({len(recorded)} -> {len(wanted)} file(s))"
                )
        for entry in self.entries:
            if not entry.path:
                continue
            p = Path(entry.path)
            try:
                st = p.stat()
            except OSError:
                problems.append(f"missing: {entry.path}")
                continue
            if entry.size >= 0 and st.st_size != entry.size:
                problems.append(f"size changed: {entry.path} ({entry.size} -> {st.st_size})")
            if entry.mtime_ns >= 0 and st.st_mtime_ns != entry.mtime_ns:
                problems.append(f"mtime changed: {entry.path}")
        return problems


def schema_digest(reader: Reader) -> str:
    """A stable digest of a reader's served schema (streams, fields, dtypes)."""
    schema = getattr(reader, "schema", None)
    if schema is None:
        return ""
    payload = {
        stream: dict(sorted(group.fields.items()))
        for stream, group in sorted(schema.groups.items())
    }
    return hashlib.blake2b(json.dumps(payload, sort_keys=True).encode(), digest_size=16).hexdigest()


def build_manifest(reader: Reader, meta: dict[str, Any] | None = None) -> CorpusManifest:
    """Build a manifest from a reader, resolving its index once.

    This is the expensive step, and the reason the artifact exists: it is paid
    deliberately, offline, once per corpus — instead of by every rank of every
    run.
    """
    blocks = reader.row_blocks()
    sources = [str(p) for p in reader.sources()]
    names = _group_names(reader)
    paths = _block_paths(reader, blocks)
    entries = []
    for block, path in zip(blocks, paths, strict=True):
        size, mtime_ns = -1, -1
        if path:
            try:
                st = Path(path).stat()
                size, mtime_ns = st.st_size, st.st_mtime_ns
            except OSError:
                pass
        entries.append(
            ManifestEntry(
                group=block.group,
                start=block.start,
                stop=block.stop,
                path=path,
                size=size,
                mtime_ns=mtime_ns,
            )
        )
    return CorpusManifest(
        entries=entries,
        group_names=names,
        schema_hash=schema_digest(reader),
        meta={"sources": sources, **(meta or {})},
    )


# --------------------------------------------------------------------------- #
# `manifest: auto` — resolve a conventional path, validate it, build on a miss
# --------------------------------------------------------------------------- #


def is_auto(manifest: Any) -> bool:
    """Whether a configured `manifest:` value asks for automatic resolution."""
    return isinstance(manifest, str) and manifest.strip().lower() == AUTO_MANIFEST


def corpus_root(sources: Sequence[str | Path]) -> Path | None:
    """The corpus's own directory, when a manifest can be written next to it.

    The sources' common ancestor, rejected unless it is a writable directory
    deep enough to BE a corpus: sources on different trees have ``/`` in common,
    which is an accident, not a corpus root.
    """
    if not sources:
        return None
    try:
        resolved = [Path(s).resolve() for s in sources]
        common = Path(os.path.commonpath([str(p) for p in resolved]))
    except ValueError:  # nothing in common (different roots)
        return None
    if common in set(resolved):  # a single file is its own commonpath
        common = common.parent
    if len(common.parts) < _MIN_ROOT_PARTS or not os.access(common, os.W_OK):
        return None
    return common if common.is_dir() else None


def resolve_manifest_path(
    reader: Reader,
    *,
    stage: str | None = None,
    num: int = -1,
    sources: Sequence[str | Path] | None = None,
) -> tuple[Path, str]:
    """Where this reader's auto manifest belongs, and whether that is ``"corpus"`` or ``"cache"``.

    Pure: every rank derives the same path without coordinating, from facts
    known before any file is opened. The filename is keyed by everything that
    decides which rows the manifest describes, so two stages of one run get two
    manifests — as they must, being two different corpora.
    """
    srcs = [str(s) for s in (reader.sources() if sources is None else sources)]
    key = json.dumps(
        [MANIFEST_VERSION, type(reader).__name__, stage, int(num), srcs], sort_keys=True
    )
    name = f"salt_manifest_{hashlib.blake2b(key.encode(), digest_size=6).hexdigest()}.json"
    root = corpus_root(srcs)
    if root is not None:
        return root / name, "corpus"
    cache = os.environ.get(MANIFEST_CACHE_ENV)
    base = Path(cache).expanduser() if cache else Path.home() / ".cache" / "salt" / "manifests"
    return base / name, "cache"


def read_manifest(
    path: str | Path, sources: Sequence[str | Path] | None = None
) -> tuple[CorpusManifest | None, list[str]]:
    """The manifest at `path` if it is present AND usable, else ``(None, why not)``.

    Absent, unreadable, wrong-version and stale are one outcome to a caller that
    can rebuild — so none of them raise, and none of them return a
    partially-trusted manifest.
    """
    path = Path(path)
    if not path.exists():
        return None, ["not built yet"]
    try:
        manifest = CorpusManifest.load(path)
    except ConfigError as exc:
        return None, [str(exc)]
    problems = manifest.validate(sources=sources)
    return (None, problems) if problems else (manifest, [])


def ensure_manifest(reader: Reader, path: Path, stage: str, where: str) -> CorpusManifest:
    """Reuse the manifest at `path`, or build it from `reader` and write it there.

    The expensive half of ``manifest: auto``, and why it belongs in Lightning's
    `prepare_data`: building resolves the reader's index, which opens every
    source file once. One rank pays that behind the framework's own barrier.
    """
    existing, problems = read_manifest(path, sources=reader.sources())
    if existing is not None:
        _LOG.info(
            f"manifest: reusing {path} ({len(existing.entries):,} blocks, {existing.n_rows:,} rows)"
        )
        return existing
    _LOG.info(
        f"manifest: building the {stage} manifest -> {path} [{where}] ({'; '.join(problems)}). "
        "This opens every source file once; for a large corpus pre-build it offline with "
        "`python -m salt.data.manifest` and point `manifest:` at the artifact."
    )
    start = time.perf_counter()
    manifest = build_manifest(reader, meta={"stage": stage})
    manifest.save(path)
    _LOG.info(
        f"manifest: built {len(manifest.entries):,} block(s), {manifest.n_rows:,} rows in "
        f"{time.perf_counter() - start:.1f} s"
    )
    return manifest


def _group_names(reader: Reader) -> list[str]:
    """Sample names for a multi-sample reader, else a single default group."""
    samples = getattr(reader, "samples", None)
    if samples:
        return [str(getattr(s, "name", i)) for i, s in enumerate(samples)]
    return ["default"]


def _paths_by_row_start(table: Any) -> dict[int, str]:
    """``{row_start: path}`` for a reader's file table; empty when it has none."""
    return {int(e.row_start): str(e.path) for e in table} if table is not None else {}


def _block_paths(reader: Reader, blocks: list[RowBlock]) -> list[str]:
    """Best-effort source path per block, for the stat-based staleness check.

    Only readers whose blocks correspond one-to-one with files can answer this.
    For anything else the path is empty and `validate` skips exactly those
    entries — a documented reduction in checking, not a silent one.
    """
    table = getattr(reader, "_table", None)
    if table is not None and len(table) >= len(blocks):
        by_start = _paths_by_row_start(table)
        return [by_start.get(b.start, "") for b in blocks]
    samples = getattr(reader, "samples", None)
    if samples:
        # one lookup table per sample, not one rebuilt per block
        by_group = {
            group: _paths_by_row_start(getattr(samples[group].reader, "_table", None))
            for group in {b.group for b in blocks}
        }
        return [by_group[b.group].get(b.start, "") for b in blocks]
    return ["" for _ in blocks]


def main() -> int:
    """CLI: build a corpus manifest from a salt config and write it to disk."""
    import argparse

    ap = argparse.ArgumentParser(description="Build a salt corpus manifest.")
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from salt.profiling import _build_datamodule

    datamodule, _model = _build_datamodule([Path(c) for c in args.config], args.overrides)
    datamodule.setup("fit")
    dataset = datamodule.train_dset
    assert dataset is not None
    manifest = build_manifest(dataset.reader)
    path = manifest.save(args.out)
    _LOG.info(
        "manifest: %d block(s), %d group(s), %s rows -> %s",
        len(manifest.entries),
        len(manifest.group_names),
        f"{manifest.n_rows:,}",
        path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
