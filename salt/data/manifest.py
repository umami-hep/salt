"""`CorpusManifest` — the offline artifact a streaming run plans its shards from,
so no rank has to open a data file to learn how many rows exist or where they are.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from salt.data.base import Reader, RowBlock
from salt.graph.errors import ConfigError

__all__ = ["MANIFEST_VERSION", "CorpusManifest", "ManifestEntry", "build_manifest"]

MANIFEST_VERSION = 1
"""Artifact format version. A mismatch is a miss, never a best-effort read."""


@dataclass(frozen=True)
class ManifestEntry:
    """One sequential read unit, described without reference to its contents.

    `path` is informational for a plain reader and load-bearing for validation:
    together with `size` and `mtime_ns` it is what lets a manifest be rejected
    when the corpus underneath it changed, using three `stat` calls per file
    rather than a re-read.
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

        Atomic because a training job that reads a half-written manifest would
        shard against a truncated corpus and silently train on part of it.
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
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(path)
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

    def validate(self, schema_hash: str | None = None) -> list[str]:
        """Structural + filesystem checks; returns the reasons it is stale (empty = usable).

        Opens NO data file. Row totals come from the manifest itself; what is
        re-checked is only what a `stat` can decide — that every recorded file
        is still present at the same size and mtime — plus the schema hash, so a
        manifest built for a different reader configuration cannot be used by
        accident.
        """
        problems: list[str] = []
        if schema_hash is not None and self.schema_hash and schema_hash != self.schema_hash:
            problems.append(
                f"schema hash {schema_hash} != manifest's {self.schema_hash} "
                "(reader configuration changed)"
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
        stream: dict(sorted(group.fields.items())) for stream, group in sorted(schema.groups.items())
    }
    return hashlib.blake2b(
        json.dumps(payload, sort_keys=True).encode(), digest_size=16
    ).hexdigest()


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


def _group_names(reader: Reader) -> list[str]:
    """Sample names for a multi-sample reader, else a single default group."""
    samples = getattr(reader, "samples", None)
    if samples:
        return [str(getattr(s, "name", i)) for i, s in enumerate(samples)]
    return ["default"]


def _block_paths(reader: Reader, blocks: list[RowBlock]) -> list[str]:
    """Best-effort source path per block, for the stat-based staleness check.

    Only readers whose blocks correspond one-to-one with files can answer this;
    for anything else the path is empty and validation falls back to the schema
    hash alone. An empty path is a documented reduction in checking, not a
    silent one — `validate` skips exactly those entries.
    """
    table = getattr(reader, "_table", None)
    if table is not None and len(table) >= len(blocks):
        by_start = {int(e.row_start): str(e.path) for e in table}
        return [by_start.get(b.start, "") for b in blocks]
    samples = getattr(reader, "samples", None)
    if samples:
        out = []
        for block in blocks:
            sub = samples[block.group].reader
            sub_table = getattr(sub, "_table", None)
            if sub_table is None:
                out.append("")
                continue
            by_start = {int(e.row_start): str(e.path) for e in sub_table}
            out.append(by_start.get(block.start, ""))
        return out
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
    print(
        f"[manifest] {len(manifest.entries)} block(s), {len(manifest.group_names)} group(s), "
        f"{manifest.n_rows:,} rows -> {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
