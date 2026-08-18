"""Reader-agnostic tests for ``manifest: auto`` — stubs only, no uproot, no ROOT.

The datamodule half of the corpus manifest: where an auto manifest lands, when
it is rebuilt, and the division of labour between `prepare_data` (may build) and
`setup` (may only read). Every claim is decided by counting the stub reader's
index resolutions — its stand-in for opening data files — rather than by
trusting a file format. Nothing here imports `uproot`, `awkward`, `h5py` or
ROOT, which is what stops a missing optional dependency from silently skipping
the module.
"""

from __future__ import annotations

import glob as _glob
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from salt.data.base import RowBlock
from salt.data.datamodule import GraphDataModule
from salt.data.dataset import GraphDataset
from salt.data.iterable_dataset import IterableGraphDataset
from salt.data.manifest import (
    CorpusManifest,
    ManifestEntry,
    apply_schema,
    build_manifest,
    corpus_root,
    is_auto,
    read_manifest,
    resolve_manifest_path,
)
from salt.data.processors.features import Features
from salt.graph.spec import Mode
from salt.schema import GroupSchema
from salt.tests.unit.data.test_iterable_dataset import BlockStubReader

ROWS_PER_FILE = 25
SINKS = {mode: ["inputs.event"] for mode in (Mode.FIT, Mode.VAL, Mode.TEST)}

# Both index resolution AND schema resolution count as opening the corpus, which
# is what makes the "opens nothing" assertions mean what they say.


class FileStubReader(BlockStubReader):
    """A stub reader backed by REAL (if trivial) files: one block per file.

    Counts index resolutions on the class, because the readers under test are
    created inside the datamodule and never handed back.
    """

    opens = 0

    def __init__(self, files, offset: int = 0, seed: int = 0, fingerprint=None) -> None:  # noqa: ANN001
        self.files = [Path(f) for f in files]
        self._fingerprint = dict(fingerprint) if fingerprint else {}
        super().__init__(
            n=ROWS_PER_FILE * len(self.files),
            offset=offset,
            n_blocks=max(1, len(self.files)),
            seed=seed,
            with_jets=False,
        )
        # the (path, row_start) duck type `build_manifest` reads block paths from
        self._table = [
            SimpleNamespace(path=f, row_start=i * ROWS_PER_FILE)
            for i, f in enumerate(self.files)
        ]

    @classmethod
    def reset(cls) -> None:
        cls.opens = 0

    def _build(self) -> None:
        if not self._built:
            FileStubReader.opens += 1
        super()._build()

    def schema_group(self, stream: str) -> GroupSchema | None:
        """Resolving the schema means opening the corpus — unless it was seeded.

        This deliberately counts as an open. An earlier version answered from a
        config-derived constant, which made the "setup opens nothing" assertions
        pass without ever exercising the schema path — the very thing
        `manifest.schema` exists to close.
        """
        if self.schema is None:
            self._build()
        return self.schema.groups.get(stream) if self.schema is not None else None

    def row_blocks(self) -> list[RowBlock]:
        """One block per file (empty reader: none)."""
        if not self.files:
            return []
        self._build()
        return [
            RowBlock(group=0, start=i * ROWS_PER_FILE, stop=(i + 1) * ROWS_PER_FILE)
            for i in range(len(self.files))
        ]

    def config_fingerprint(self) -> dict:
        """Stands in for a real reader's cuts/groups config."""
        return dict(self._fingerprint)

    def with_source(self, filename, num=-1, vds_path=None, stage=None):  # noqa: ANN001
        """Re-source onto the glob's expansion — the per-stage clone."""
        del num, vds_path, stage
        return FileStubReader(
            files=sorted(_glob.glob(str(filename))), fingerprint=self._fingerprint
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _corpus(root: Path, names: list[str]) -> str:
    """Write placeholder source files under `root`; return the glob that finds them.

    Existing files are left ALONE — several tests depend on a corpus keeping the
    size and mtime a manifest recorded for it.
    """
    root.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        path = root / name
        if not path.exists():
            path.write_bytes(b"x" * (100 + i))
    return str(root / "*.root")


def _dm(train: str, val: str | None = None, **kwargs) -> GraphDataModule:
    """A streaming datamodule over the stub reader, `manifest: auto` by default."""
    modules = {
        "reader": FileStubReader(files=[]),
        "feats": Features(variables={"event": ["val"]}),
    }
    kwargs.setdefault("manifest", "auto")
    kwargs.setdefault("iterable", True)
    return GraphDataModule(
        modules=modules,
        train_file=train,
        val_file=val if val is not None else train,
        batch_size=5,
        num_workers=0,
        block_rows=None,
        sinks=SINKS,
        **kwargs,
    )


def _manifests(directory: Path) -> list[Path]:
    """Manifest artifacts in a directory, sorted."""
    return sorted(directory.glob("salt_manifest_*.json"))


def _uids(dataset: IterableGraphDataset) -> list[int]:
    """Every uid the streaming dataset emits, via the reader-level batches."""
    from salt.tests.unit.data.test_iterable_dataset import _batch_uids

    return _batch_uids(dataset)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Never touch the developer's real ``~/.cache/salt/manifests``."""
    cache = tmp_path / "cache"
    monkeypatch.setenv("SALT_MANIFEST_CACHE", str(cache))
    FileStubReader.reset()
    return cache


# --------------------------------------------------------------------------- #
# path resolution
# --------------------------------------------------------------------------- #


def test_auto_lands_next_to_a_writable_corpus(tmp_path) -> None:
    """The preferred location is the corpus's own directory."""
    root = tmp_path / "corpus"
    reader = FileStubReader(files=[]).with_source(_corpus(root, ["a.root", "b.root"]))
    path, where = resolve_manifest_path(reader, stage="train")
    assert where == "corpus"
    assert path.parent == root.resolve()


# The next two tests look like duplicates and are not: they reach the SAME cache
# branch by two different routes, because neither route alone covers every
# environment. The permissions route is the realistic one but cannot run as root
# (CI does); the geometry route can run anywhere. Deleting either leaves the
# branch untested somewhere.
@pytest.mark.skipif(os.geteuid() == 0, reason="root writes to any directory, mode bits included")
def test_auto_falls_back_to_the_cache_when_the_corpus_is_read_only(
    tmp_path, _isolated_cache
) -> None:
    """A read-only corpus directory is routine on cluster storage, not an error."""
    root = tmp_path / "corpus"
    reader = FileStubReader(files=[]).with_source(_corpus(root, ["a.root"]))
    root.chmod(0o555)
    try:
        path, where = resolve_manifest_path(reader, stage="train")
    finally:
        root.chmod(0o755)
    assert where == "cache"
    assert path.parent == _isolated_cache


def test_auto_falls_back_to_the_cache_when_there_is_no_corpus_directory(
    tmp_path, _isolated_cache
) -> None:
    """The same fallback, reached by geometry rather than permissions.

    The permissions route above cannot be tested as root; this one can, so the
    cache branch stays covered wherever the suite runs.
    """
    reader = FileStubReader(files=["/aaa/x.root", "/bbb/y.root"])
    path, where = resolve_manifest_path(reader, stage="train")
    assert where == "cache"
    assert path.parent == _isolated_cache
    assert path.name.startswith("salt_manifest_")


def test_a_broad_common_ancestor_is_not_a_corpus_root() -> None:
    """Sources with nothing but ``/`` in common have no corpus directory."""
    assert corpus_root(["/aaa/x.root", "/bbb/y.root"]) is None
    assert corpus_root([]) is None


def test_the_key_separates_stages_and_corpora(tmp_path) -> None:
    """Two stages, or two corpora, never share one manifest file."""
    proto = FileStubReader(files=[])
    train = proto.with_source(_corpus(tmp_path / "corpus", ["a.root", "b.root"]))
    other = proto.with_source(_corpus(tmp_path / "other", ["a.root"]))
    train_path, _ = resolve_manifest_path(train, stage="train")
    val_path, _ = resolve_manifest_path(train, stage="val")
    other_path, _ = resolve_manifest_path(other, stage="train")
    assert train_path != val_path
    assert {train_path, val_path}.isdisjoint({other_path})


def test_a_reconfigured_reader_keys_a_different_manifest(tmp_path) -> None:
    """A config change over the SAME files must not reuse the old manifest.

    The stat checks describe the corpus and cannot see this: change a row cut and
    every block boundary moves while every file stays byte-identical. Keying the
    artifact on the config digest is what turns that into a miss.
    """
    corpus = _corpus(tmp_path / "corpus", ["a.root", "b.root"])
    base = FileStubReader(files=[]).with_source(corpus)
    cut = FileStubReader(files=[], fingerprint={"cuts": "pt > 20"}).with_source(corpus)
    groups = FileStubReader(files=[], fingerprint={"groups": ["pt", "eta"]}).with_source(corpus)

    base_path, _ = resolve_manifest_path(base, stage="train")
    cut_path, _ = resolve_manifest_path(cut, stage="train")
    groups_path, _ = resolve_manifest_path(groups, stage="train")
    assert len({base_path, cut_path, groups_path}) == 3


def test_a_fingerprintless_reader_keys_as_before(tmp_path) -> None:
    """An empty fingerprint adds nothing to the key — readers that cannot answer
    are keyed on paths alone, exactly as they were.
    """
    corpus = _corpus(tmp_path / "corpus", ["a.root"])
    a, _ = resolve_manifest_path(FileStubReader(files=[]).with_source(corpus), stage="train")
    b, _ = resolve_manifest_path(FileStubReader(files=[]).with_source(corpus), stage="train")
    assert a == b


def test_a_changed_served_schema_is_stale(tmp_path) -> None:
    """`schema_hash` is checked when the caller can supply one for free."""
    root = tmp_path / "corpus"
    reader = FileStubReader(files=[]).with_source(_corpus(root, ["a.root"]))
    path = build_manifest(reader).save(root / "salt_manifest_probe.json")

    assert read_manifest(path, sources=reader.sources())[0] is not None
    manifest, problems = read_manifest(
        path, sources=reader.sources(), schema_hash="a-different-reader"
    )
    assert manifest is None
    assert any("schema hash" in p for p in problems)


def test_is_auto_accepts_only_the_keyword() -> None:
    """``auto`` is a mode; a path that merely contains it is a path."""
    assert is_auto("auto")
    assert is_auto(" AUTO ")
    assert not is_auto("/data/auto/manifest.json")
    assert not is_auto(None)
    assert not is_auto(Path("auto"))


# --------------------------------------------------------------------------- #
# build once, reuse after
# --------------------------------------------------------------------------- #


def test_prepare_data_builds_a_valid_manifest(tmp_path) -> None:
    """First use writes one manifest per streaming stage, describing the corpus."""
    root = tmp_path / "corpus"
    dm = _dm(_corpus(root, ["a.root", "b.root", "c.root"]))
    dm.prepare_data()
    written = _manifests(root)
    assert len(written) == 2, "one manifest per stage (train, val)"
    manifest = CorpusManifest.load(written[0])
    assert manifest.n_rows == 3 * ROWS_PER_FILE
    assert len(manifest.entries) == 3
    assert manifest.validate(sources=[root / n for n in ("a.root", "b.root", "c.root")]) == []


def test_second_run_reuses_it_and_setup_opens_nothing(tmp_path) -> None:
    """The whole point: a later run plans its shards with zero index resolutions."""
    dm = _dm(_corpus(tmp_path / "corpus", ["a.root", "b.root"]))
    dm.prepare_data()
    assert FileStubReader.opens > 0

    FileStubReader.reset()
    again = _dm(_corpus(tmp_path / "corpus", ["a.root", "b.root"]))
    again.prepare_data()
    again.setup("fit")
    assert FileStubReader.opens == 0
    assert isinstance(again.train_dset, IterableGraphDataset)
    assert isinstance(again.train_dset.manifest, CorpusManifest)
    assert len(again.train_dset) == (2 * ROWS_PER_FILE) // 5


def test_a_warm_manifest_carries_the_schema(tmp_path) -> None:
    """The manifest records the served schema, so a later run need not re-probe it."""
    root = tmp_path / "corpus"
    reader = FileStubReader(files=[]).with_source(_corpus(root, ["a.root"]))
    manifest = build_manifest(reader)
    assert manifest.schema == {"event": {"uid": "int64", "val": "float32"}}
    assert CorpusManifest.load(manifest.save(root / "m.json")).schema == manifest.schema


def test_a_seeded_reader_resolves_fields_without_opening(tmp_path) -> None:
    """`apply_schema` is what makes a warm setup zero-open: the schema comes off the
    manifest, so `schema_group` answers without resolving the index.
    """
    root = tmp_path / "corpus"
    built = build_manifest(FileStubReader(files=[]).with_source(_corpus(root, ["a.root"])))

    FileStubReader.reset()
    fresh = FileStubReader(files=[]).with_source(str(root / "*.root"))
    assert apply_schema(built, fresh) is True
    assert fresh.schema_group("event") is not None
    assert FileStubReader.opens == 0

    # a reader that already has one keeps it — its own schema is authoritative
    assert apply_schema(built, fresh) is False


def test_prepare_data_is_idempotent(tmp_path) -> None:
    """Calling it twice builds once — the second call finds its own artifact."""
    root = tmp_path / "corpus"
    dm = _dm(_corpus(root, ["a.root"]))
    dm.prepare_data()
    first = FileStubReader.opens
    dm.prepare_data()
    assert FileStubReader.opens == first
    assert len(_manifests(root)) == 2


def test_train_and_val_manifests_describe_their_own_corpus(tmp_path) -> None:
    """Different stage sources get different manifests — not one applied to both."""
    train = _corpus(tmp_path / "train", ["a.root", "b.root", "c.root"])
    val = _corpus(tmp_path / "val", ["a.root"])
    dm = _dm(train, val)
    dm.prepare_data()
    dm.setup("fit")
    assert dm.train_dset.manifest.n_rows == 3 * ROWS_PER_FILE  # type: ignore[union-attr]
    assert dm.val_dset.manifest.n_rows == ROWS_PER_FILE  # type: ignore[union-attr]


def test_streaming_reads_the_same_rows_with_and_without_a_manifest(tmp_path) -> None:
    """An auto manifest changes planning, never which rows exist."""
    corpus = _corpus(tmp_path / "corpus", ["a.root", "b.root"])
    plain = _dm(corpus, manifest=None)
    plain.setup("fit")
    auto = _dm(corpus)
    auto.prepare_data()
    auto.setup("fit")
    assert Counter(_uids(auto.train_dset)) == Counter(_uids(plain.train_dset))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# staleness
# --------------------------------------------------------------------------- #


def test_a_resized_source_file_forces_a_rebuild(tmp_path) -> None:
    """Staleness is decided by `stat`, and a stale manifest is replaced in place."""
    root = tmp_path / "corpus"
    corpus = _corpus(root, ["a.root", "b.root"])
    _dm(corpus).prepare_data()
    before = _manifests(root)
    (root / "a.root").write_bytes(b"y" * 4096)

    FileStubReader.reset()
    _dm(corpus).prepare_data()
    assert FileStubReader.opens > 0, "a changed corpus must not be planned from a stale manifest"
    assert _manifests(root) == before, "the rebuild replaces the artifact, it does not add one"
    entry = CorpusManifest.load(before[0]).entries[0]
    assert entry.size == (root / "a.root").stat().st_size


def test_a_foreign_format_version_forces_a_rebuild(tmp_path) -> None:
    """An artifact this salt cannot read is a miss, never a partial read."""
    root = tmp_path / "corpus"
    corpus = _corpus(root, ["a.root"])
    _dm(corpus).prepare_data()
    target = _manifests(root)[0]
    target.write_text('{"version": 999, "entries": [], "group_names": ["default"]}')

    FileStubReader.reset()
    _dm(corpus).prepare_data()
    assert FileStubReader.opens > 0
    assert CorpusManifest.load(target).n_rows == ROWS_PER_FILE


def test_a_new_corpus_file_keys_a_new_manifest(tmp_path) -> None:
    """A changed file LIST is a different corpus, so it gets its own artifact."""
    root = tmp_path / "corpus"
    _dm(_corpus(root, ["a.root"])).prepare_data()
    _dm(_corpus(root, ["a.root", "b.root"])).prepare_data()
    written = _manifests(root)
    assert len(written) == 4, "two stages x two corpora"
    assert {CorpusManifest.load(p).n_rows for p in written} == {
        ROWS_PER_FILE,
        2 * ROWS_PER_FILE,
    }


def test_validate_rejects_a_changed_file_list(tmp_path) -> None:
    """The check no per-file `stat` can make: a file added to (or dropped from) the glob."""
    root = tmp_path / "corpus"
    reader = FileStubReader(files=[]).with_source(_corpus(root, ["a.root", "b.root"]))
    manifest = build_manifest(reader)
    assert manifest.validate(sources=[root / "a.root", root / "b.root"]) == []
    problems = manifest.validate(sources=[root / "a.root"])
    assert problems and "file list changed" in problems[0]


def test_a_failed_write_leaves_neither_a_partial_nor_a_temp_file(tmp_path, monkeypatch) -> None:
    """Atomicity: the destination only ever appears complete."""
    root = tmp_path / "corpus"
    reader = FileStubReader(files=[]).with_source(_corpus(root, ["a.root"]))
    manifest = build_manifest(reader)
    target = root / "salt_manifest_broken.json"

    def boom(*_args, **_kwargs) -> str:
        raise OSError("disk full")

    monkeypatch.setattr(json, "dumps", boom)
    with pytest.raises(OSError, match="disk full"):
        manifest.save(target)
    assert not target.exists()
    assert list(root.glob("*.tmp")) == []
    assert list(root.glob(".*")) == []


# --------------------------------------------------------------------------- #
# the other two modes, and the prepare/setup division of labour
# --------------------------------------------------------------------------- #


def test_none_mode_is_unchanged(tmp_path) -> None:
    """No manifest configured: nothing is written and nothing is planned from one."""
    root = tmp_path / "corpus"
    dm = _dm(_corpus(root, ["a.root"]), manifest=None)
    dm.prepare_data()
    dm.setup("fit")
    assert _manifests(root) == []
    assert dm.train_dset.manifest is None  # type: ignore[union-attr]


def test_an_explicit_path_is_passed_through_untouched(tmp_path) -> None:
    """An explicit manifest is the user's assertion — never auto-built, never rewritten."""
    root = tmp_path / "corpus"
    corpus = _corpus(root, ["a.root"])
    explicit = tmp_path / "mine.json"
    build_manifest(FileStubReader(files=[]).with_source(corpus)).save(explicit)
    stamp = explicit.stat().st_mtime_ns

    FileStubReader.reset()
    dm = _dm(corpus, manifest=str(explicit))
    dm.prepare_data()
    assert FileStubReader.opens == 0, "prepare_data must not build for an explicit path"
    assert _manifests(root) == []
    # the configured value reaches the dataset verbatim (the dataset then loads
    # it lazily, so this is checked at the seam rather than after `setup`)
    dm._run_setup_pass(["train"])
    reader, num = dm._stage_reader(Mode.FIT)
    assert dm._manifest_for(Mode.FIT, reader, num) == str(explicit)
    dm.setup("fit")
    assert dm.train_dset.manifest.n_rows == ROWS_PER_FILE  # type: ignore[union-attr]
    assert explicit.stat().st_mtime_ns == stamp


def test_setup_alone_never_builds(tmp_path) -> None:
    """Without `prepare_data`, `setup` degrades to the reader index — it does not build."""
    root = tmp_path / "corpus"
    dm = _dm(_corpus(root, ["a.root", "b.root"]))
    dm.setup("fit")
    assert _manifests(root) == [], "setup() wrote a manifest — that belongs to prepare_data()"
    assert dm.train_dset.manifest is None  # type: ignore[union-attr]
    assert len(_uids(dm.train_dset)) == 2 * ROWS_PER_FILE  # type: ignore[arg-type]


def test_prepare_data_per_node_is_false() -> None:
    """The artifact goes to shared storage, so exactly one process in the JOB writes it."""
    dm = _dm("/nonexistent/*.root")
    assert dm.prepare_data_per_node is False


def test_auto_is_ignored_on_the_map_style_path(tmp_path) -> None:
    """A corpus manifest plans streaming shards; the map-style path has none to plan."""
    root = tmp_path / "corpus"
    dm = _dm(_corpus(root, ["a.root"]), iterable=False)
    dm.prepare_data()
    dm.setup("fit")
    assert _manifests(root) == []
    assert isinstance(dm.train_dset, GraphDataset)


def test_a_sourceless_reader_degrades_instead_of_failing(tmp_path) -> None:
    """A reader that cannot name its files cannot be keyed — and must still run."""

    class SourcelessReader(FileStubReader):
        def sources(self) -> list[Path]:
            return []

        def with_source(self, filename, num=-1, vds_path=None, stage=None):  # noqa: ANN001
            del num, vds_path, stage
            return SourcelessReader(files=sorted(_glob.glob(str(filename))))

    modules = {
        "reader": SourcelessReader(files=[]),
        "feats": Features(variables={"event": ["val"]}),
    }
    corpus = _corpus(tmp_path / "corpus", ["a.root"])
    dm = GraphDataModule(
        modules=modules,
        train_file=corpus,
        val_file=corpus,
        batch_size=5,
        num_workers=0,
        block_rows=None,
        iterable=True,
        manifest="auto",
        sinks=SINKS,
    )
    dm.prepare_data()
    dm.setup("fit")
    assert _manifests(tmp_path / "corpus") == []
    assert dm.train_dset.manifest is None  # type: ignore[union-attr]


def test_manifest_entry_paths_survive_the_round_trip(tmp_path) -> None:
    """The stat facts validation needs are what `build_manifest` records."""
    root = tmp_path / "corpus"
    reader = FileStubReader(files=[]).with_source(_corpus(root, ["a.root", "b.root"]))
    entries = build_manifest(reader).entries
    assert [Path(e.path).name for e in entries] == ["a.root", "b.root"]
    assert all(e.size > 0 and e.mtime_ns > 0 for e in entries)
    assert isinstance(entries[0], ManifestEntry)
