"""`CorpusManifest` — the offline artifact a streaming run plans its shards from,
so no rank has to open a data file to learn how many rows exist or where they are.
It lives at the user-configured ``manifest:`` path and nowhere else; `ensure_manifest`
is the whole lifecycle (validate+read, build-on-missing, hard error when stale).
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

from salt import __version__
from salt.data.base import Reader, RowBlock
from salt.graph.errors import ConfigError
from salt.logging import get_logger
from salt.schema import GroupSchema, Schema

__all__ = ["CorpusManifest", "ManifestEntry", "build_manifest", "ensure_manifest"]

_LOG = get_logger(__name__)


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
    schema : dict, optional
        The served schema as ``{stream: {field: dtype}}``. Carried so plan
        compilation can validate demanded fields without asking the reader to
        probe a file for them — see `apply_schema`.
    config_digest : str, optional
        Digest of the builder's `config_fingerprint` — the one staleness signal
        no `stat` can supply, since a reconfigured reader (different ``cuts``,
        different ``groups``) moves every block boundary over byte-identical
        files.
    meta : dict, optional
        Free-form provenance (build time, source list, stage).
    """

    entries: list[ManifestEntry]
    group_names: list[str] = field(default_factory=lambda: ["default"])
    schema_hash: str = ""
    schema: dict[str, dict[str, str]] = field(default_factory=dict)
    config_digest: str = ""
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

        The payload is stamped with `salt.__version__` — the artifact's format
        identity. `load` refuses a foreign stamp, so a format change never has
        to be detected field-by-field.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "salt_version": __version__,
            "group_names": list(self.group_names),
            "schema_hash": self.schema_hash,
            "schema": {s: dict(f) for s, f in self.schema.items()},
            "config_digest": self.config_digest,
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
            If the file is unreadable, malformed, or stamped by a different
            salt version — delete it to rebuild.
        """
        path = Path(path)
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"unreadable corpus manifest {path}: {exc}") from exc
        stamp = str(payload.get("salt_version", ""))
        if stamp != __version__:
            raise ConfigError(
                f"corpus manifest {path} was written by salt {stamp or 'unknown'}, "
                f"this is salt {__version__} — delete {path} to rebuild"
            )
        return cls(
            entries=[ManifestEntry(**entry) for entry in payload["entries"]],
            group_names=list(payload.get("group_names", ["default"])),
            schema_hash=str(payload.get("schema_hash", "")),
            schema={s: dict(f) for s, f in payload.get("schema", {}).items()},
            config_digest=str(payload.get("config_digest", "")),
            meta=dict(payload.get("meta", {})),
        )

    def validate(
        self,
        schema_hash: str | None = None,
        sources: Sequence[str | Path] | None = None,
        config_digest: str | None = None,
    ) -> list[str]:
        """Structural + filesystem checks; returns the reasons it is stale (empty = usable).

        Opens NO data file — only what a `stat` can decide (every recorded file
        still present at the same size and mtime), plus the schema hash and the
        config digest, so a manifest built for a different reader configuration
        cannot be used by accident. `sources` additionally catches the one
        change no per-file stat can see: a file added to or removed from the
        glob leaves every recorded file untouched.
        """
        problems: list[str] = []
        if schema_hash is not None and self.schema_hash and schema_hash != self.schema_hash:
            problems.append(
                f"schema hash {schema_hash} != manifest's {self.schema_hash} "
                "(reader configuration changed)"
            )
        if config_digest is not None and config_digest != self.config_digest:
            problems.append(
                f"reader config digest {config_digest or '(empty)'} != manifest's "
                f"{self.config_digest or '(empty)'} (cuts/groups/reader configuration changed)"
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


def built_schema_hash(reader: Reader) -> str | None:
    """`schema_digest` when the reader ALREADY has a schema, else None.

    Never triggers a build: an unprepared reader returns None so validation skips
    the schema check rather than opening every file to perform it.
    """
    return schema_digest(reader) if getattr(reader, "schema", None) is not None else None


def config_digest(reader: Reader) -> str:
    """A digest of the reader's `config_fingerprint`; empty when it has none.

    The manifest's only OPEN-FREE staleness signal for a reconfigured reader.
    The stat checks catch a changed corpus and `schema_hash` catches a changed
    served schema, but both describe the files; neither sees a reader
    reconfigured over the SAME files — a different `cuts` (which changes row
    counts, and so every block boundary) or a different `groups`. That has to be
    caught before anything is read, which is why the digest is recorded in the
    artifact and checked by `validate`.
    """
    fingerprint = reader.config_fingerprint()
    if not fingerprint:
        return ""
    blob = json.dumps(fingerprint, sort_keys=True, default=str).encode()
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


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
    deliberately, once per corpus — instead of by every rank of every run.
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
        schema=served_schema(reader),
        config_digest=config_digest(reader),
        meta={"sources": sources, **(meta or {})},
    )


def served_schema(reader: Reader) -> dict[str, dict[str, str]]:
    """The reader's built schema as ``{stream: {field: dtype}}``; empty if it has none."""
    schema = getattr(reader, "schema", None)
    if schema is None:
        return {}
    return {stream: dict(group.fields) for stream, group in schema.groups.items()}


def apply_schema(manifest: CorpusManifest, reader: Reader) -> bool:
    """Give `reader` the manifest's schema, so plan compilation opens no data file.

    This is the difference between a warm start that still probes and one that
    does not. `schema_group` and `label_universe` both resolve `prepare()` when
    the reader has no schema, and plan compilation calls both — so a run with a
    perfectly good manifest still opened one file per stage just to re-learn
    field names the manifest already recorded.

    Reading still builds the index when a worker actually reads: this seeds the
    schema, not the row table. Returns whether anything was seeded — a reader
    that already has a schema is left alone, since its own is authoritative.
    """
    if not manifest.schema or getattr(reader, "schema", None) is not None:
        return False
    reader.schema = Schema(
        groups={
            stream: GroupSchema(fields=dict(fields)) for stream, fields in manifest.schema.items()
        }
    )
    return True


def ensure_manifest(reader: Reader, path: str | Path, stage: str) -> CorpusManifest:
    """The manifest at `path`: validated and read when present, built there when not.

    The ONE code path behind the required ``manifest:`` config value. Building
    resolves the reader's index, which opens every source file once — which is
    why the datamodule calls this first from Lightning's `prepare_data` (global
    rank 0, barriered before `setup`), so one process pays and every other rank
    finds a finished file. The atomic `save` means an uncoordinated caller can
    waste that work, never corrupt it.

    Raises
    ------
    ConfigError
        If the existing manifest is unreadable, foreign, or stale for this
        reader — it is the user's file, so it is never silently overwritten;
        the error names the path to delete to rebuild.
    """
    path = Path(path)
    if not path.exists():
        _LOG.info(
            f"manifest: building the {stage} manifest -> {path}. This opens every source "
            "file once; later runs reuse the artifact."
        )
        start = time.perf_counter()
        manifest = build_manifest(reader, meta={"stage": stage})
        manifest.save(path)
        _LOG.info(
            f"manifest: built {len(manifest.entries):,} block(s), {manifest.n_rows:,} rows in "
            f"{time.perf_counter() - start:.1f} s"
        )
        return manifest
    manifest = CorpusManifest.load(path)
    problems = manifest.validate(
        schema_hash=built_schema_hash(reader),
        sources=reader.sources(),
        config_digest=config_digest(reader),
    )
    if problems:
        hint = (
            " (one manifest cannot serve two corpora — configure per-stage paths, "
            "manifest: {train: ..., val: ...}, when stages read different file sets)"
            if any("file list changed" in p for p in problems)
            else ""
        )
        raise ConfigError(
            f"stale corpus manifest {path} for the {stage} corpus: {'; '.join(problems)} — "
            f"delete {path} to rebuild{hint}"
        )
    _LOG.info(
        f"manifest: reusing {path} ({len(manifest.entries):,} blocks, {manifest.n_rows:,} rows)"
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
