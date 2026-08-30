"""Reader-agnostic tests for the required ``manifest:`` contract — stubs only.

The datamodule half of the corpus manifest: streaming runs must name a manifest
path, an existing file is validated and read, a missing one is built at that
path (rank-0 `prepare_data`, atomic write), and a stale one is a hard error
naming the file to delete. Every "opens nothing" claim is decided by counting
the stub reader's index resolutions — its stand-in for opening data files —
rather than by trusting a file format. Nothing here imports `uproot`,
`awkward`, `h5py` or ROOT, which is what stops a missing optional dependency
from silently skipping the module.
"""

from __future__ import annotations

import glob as _glob
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from salt.data.base import RowBlock
from salt.data.datamodule import SaltDataModule
from salt.data.dataset import SaltDataset
from salt.data.iterable_dataset import IterableSaltDataset
from salt.data.manifest import (
    CorpusManifest,
    ManifestEntry,
    apply_schema,
    build_manifest,
)
from salt.data.processors.features import Features
from salt.graph.errors import ConfigError
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

    def __init__(self, files, offset: int = 0, seed: int = 0, fingerprint=None) -> None:
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
            SimpleNamespace(path=f, row_start=i * ROWS_PER_FILE) for i, f in enumerate(self.files)
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

    def with_source(self, filename, num=-1, vds_path=None, stage=None):
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


def _dm(train: str, manifest, val: str | None = None, reader=None, **kwargs) -> SaltDataModule:
    """A streaming datamodule over the stub reader, planning from `manifest`."""
    modules = {
        "reader": reader if reader is not None else FileStubReader(files=[]),
        "feats": Features(variables={"event": ["val"]}),
    }
    kwargs.setdefault("iterable", True)
    return SaltDataModule(
        modules=modules,
        train_file=train,
        val_file=val if val is not None else train,
        batch_size=5,
        num_workers=0,
        block_rows=None,
        sinks=SINKS,
        manifest=manifest,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _reset_opens():
    """Every test starts with a clean index-resolution count."""
    FileStubReader.reset()


# --------------------------------------------------------------------------- #
# the argument is required and explicit
# --------------------------------------------------------------------------- #


def test_streaming_requires_a_manifest_path(tmp_path) -> None:
    """No manifest on the streaming path is a configuration error, not a fallback."""
    with pytest.raises(ConfigError, match=r"data.manifest is required"):
        _dm(_corpus(tmp_path / "corpus", ["a.root"]), manifest=None)


def test_auto_mode_is_removed(tmp_path) -> None:
    """The old ``manifest: auto`` keyword fails loudly instead of naming a file."""
    corpus = _corpus(tmp_path / "corpus", ["a.root"])
    with pytest.raises(ConfigError, match="removed"):
        _dm(corpus, manifest="auto")
    with pytest.raises(ConfigError, match="removed"):
        _dm(corpus, manifest=" AUTO ")


def test_manifest_is_ignored_on_the_map_style_path(tmp_path) -> None:
    """A corpus manifest plans streaming shards; the map-style path has none to plan."""
    root = tmp_path / "corpus"
    target = tmp_path / "m.json"
    dm = _dm(_corpus(root, ["a.root"]), manifest=target, iterable=False)
    dm.prepare_data()
    dm.setup("fit")
    assert not target.exists()
    assert isinstance(dm.train_dset, SaltDataset)


def test_prepare_data_per_node_is_false(tmp_path) -> None:
    """The artifact goes to shared storage, so exactly one process in the JOB writes it."""
    dm = _dm("/nonexistent/*.root", manifest=tmp_path / "m.json")
    assert dm.prepare_data_per_node is False


# --------------------------------------------------------------------------- #
# missing -> built there; present -> validated and read
# --------------------------------------------------------------------------- #


def test_prepare_data_builds_at_the_named_path(tmp_path) -> None:
    """A missing manifest is built exactly where the config points, describing the corpus."""
    root = tmp_path / "corpus"
    target = tmp_path / "artifacts" / "corpus_manifest.json"
    dm = _dm(_corpus(root, ["a.root", "b.root", "c.root"]), manifest=target)
    dm.prepare_data()
    assert target.exists()
    manifest = CorpusManifest.load(target)
    assert manifest.n_rows == 3 * ROWS_PER_FILE
    assert len(manifest.entries) == 3
    assert manifest.validate(sources=[root / n for n in ("a.root", "b.root", "c.root")]) == []


def test_second_run_reuses_it_and_setup_opens_nothing(tmp_path) -> None:
    """The whole point: a later run plans its shards with zero index resolutions."""
    target = tmp_path / "m.json"
    dm = _dm(_corpus(tmp_path / "corpus", ["a.root", "b.root"]), manifest=target)
    dm.prepare_data()
    assert FileStubReader.opens > 0

    FileStubReader.reset()
    again = _dm(_corpus(tmp_path / "corpus", ["a.root", "b.root"]), manifest=target)
    again.prepare_data()
    again.setup("fit")
    assert FileStubReader.opens == 0
    assert isinstance(again.train_dset, IterableSaltDataset)
    assert isinstance(again.train_dset.manifest, CorpusManifest)
    assert len(again.train_dset) == (2 * ROWS_PER_FILE) // 5


def test_prepare_data_is_idempotent(tmp_path) -> None:
    """Calling it twice builds once — the second call reads the first one's artifact."""
    target = tmp_path / "m.json"
    dm = _dm(_corpus(tmp_path / "corpus", ["a.root"]), manifest=target)
    dm.prepare_data()
    first = FileStubReader.opens
    stamp = target.stat().st_mtime_ns
    dm.prepare_data()
    assert FileStubReader.opens == first
    assert target.stat().st_mtime_ns == stamp


def test_an_existing_manifest_is_never_rewritten(tmp_path) -> None:
    """A pre-built manifest is validated and read — never touched on disk."""
    root = tmp_path / "corpus"
    corpus = _corpus(root, ["a.root"])
    target = tmp_path / "mine.json"
    build_manifest(FileStubReader(files=[]).with_source(corpus)).save(target)
    stamp = target.stat().st_mtime_ns

    FileStubReader.reset()
    dm = _dm(corpus, manifest=target)
    dm.prepare_data()
    dm.setup("fit")
    assert FileStubReader.opens == 0
    assert dm.train_dset.manifest.n_rows == ROWS_PER_FILE  # type: ignore[union-attr]
    assert target.stat().st_mtime_ns == stamp


def test_setup_builds_when_prepare_data_never_ran(tmp_path) -> None:
    """A datamodule driven outside a `Trainer` still gets its manifest — same code path.

    Inside a `Trainer`, `prepare_data` builds behind the rank-0 barrier and
    `setup` finds a finished file; without one, `setup` builds at the same path.
    The atomic write makes an uncoordinated build wasteful, never corrupting.
    """
    target = tmp_path / "m.json"
    dm = _dm(_corpus(tmp_path / "corpus", ["a.root", "b.root"]), manifest=target)
    dm.setup("fit")
    assert target.exists()
    assert isinstance(dm.train_dset.manifest, CorpusManifest)  # type: ignore[union-attr]
    assert dm.train_dset.manifest.n_rows == 2 * ROWS_PER_FILE  # type: ignore[union-attr]


def test_a_sourceless_reader_still_streams(tmp_path) -> None:
    """A reader that cannot name its files must still run.

    Its manifest carries path-less entries (size/mtime ``-1``), which
    `validate` skips, and an empty recorded source list that a later run
    matches with its own.
    """

    class SourcelessReader(FileStubReader):
        def __init__(self, files, **kwargs) -> None:
            super().__init__(files, **kwargs)
            self._table = None  # blocks correspond to no on-disk paths

        def sources(self) -> list[Path]:
            return []

        def with_source(self, filename, num=-1, vds_path=None, stage=None):
            del num, vds_path, stage
            return SourcelessReader(files=sorted(_glob.glob(str(filename))))

    corpus = _corpus(tmp_path / "corpus", ["a.root", "b.root"])
    target = tmp_path / "m.json"
    dm = _dm(corpus, manifest=target, reader=SourcelessReader(files=[]))
    dm.prepare_data()
    dm.setup("fit")
    assert isinstance(dm.train_dset, IterableSaltDataset)
    assert dm.train_dset.manifest.n_rows == 2 * ROWS_PER_FILE  # type: ignore[union-attr]
    manifest = CorpusManifest.load(target)
    assert manifest.meta["sources"] == []
    assert all(not e.path and e.size == -1 for e in manifest.entries)

    again = _dm(corpus, manifest=target, reader=SourcelessReader(files=[]))
    again.prepare_data()  # validates ([] == [] sources, no stat loop) and reuses
    again.setup("fit")
    assert again.train_dset.manifest.n_rows == 2 * ROWS_PER_FILE  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# per-stage paths
# --------------------------------------------------------------------------- #


def test_a_mapping_gives_each_stage_its_own_manifest(tmp_path) -> None:
    """Different stage corpora need different artifacts — the mapping names them."""
    train = _corpus(tmp_path / "train", ["a.root", "b.root", "c.root"])
    val = _corpus(tmp_path / "val", ["a.root"])
    paths = {"train": tmp_path / "train.json", "val": tmp_path / "val.json"}
    dm = _dm(train, manifest=paths, val=val)
    dm.prepare_data()
    dm.setup("fit")
    assert paths["train"].exists()
    assert paths["val"].exists()
    assert dm.train_dset.manifest.n_rows == 3 * ROWS_PER_FILE  # type: ignore[union-attr]
    assert dm.val_dset.manifest.n_rows == ROWS_PER_FILE  # type: ignore[union-attr]


def test_a_mapping_missing_a_stage_is_an_error(tmp_path) -> None:
    """Every streaming stage must be named — no silent fallback path."""
    dm = _dm(_corpus(tmp_path / "corpus", ["a.root"]), manifest={"train": tmp_path / "train.json"})
    with pytest.raises(ConfigError, match="'val'"):
        dm.prepare_data()


def test_a_scalar_path_cannot_serve_two_corpora(tmp_path) -> None:
    """One shared path over different train/val file sets fails, pointing at the mapping."""
    train = _corpus(tmp_path / "train", ["a.root", "b.root"])
    val = _corpus(tmp_path / "val", ["a.root"])
    dm = _dm(train, manifest=tmp_path / "m.json", val=val)
    with pytest.raises(ConfigError, match="per-stage"):
        dm.prepare_data()


# --------------------------------------------------------------------------- #
# staleness is a hard error naming the path
# --------------------------------------------------------------------------- #


def test_a_resized_source_file_is_a_hard_error(tmp_path) -> None:
    """Staleness is decided by `stat`; a user-named file is never silently replaced."""
    root = tmp_path / "corpus"
    corpus = _corpus(root, ["a.root", "b.root"])
    target = tmp_path / "m.json"
    _dm(corpus, manifest=target).prepare_data()
    (root / "a.root").write_bytes(b"y" * 4096)

    with pytest.raises(ConfigError, match=r"delete .* to rebuild") as err:
        _dm(corpus, manifest=target).prepare_data()
    assert "size changed" in str(err.value)
    assert str(target) in str(err.value)


def test_a_changed_file_list_is_stale(tmp_path) -> None:
    """The check no per-file `stat` can make: a file added to (or dropped from) the glob."""
    root = tmp_path / "corpus"
    target = tmp_path / "m.json"
    _dm(_corpus(root, ["a.root"]), manifest=target).prepare_data()
    with pytest.raises(ConfigError, match="file list changed"):
        _dm(_corpus(root, ["a.root", "b.root"]), manifest=target).prepare_data()


def test_a_reconfigured_reader_is_stale(tmp_path) -> None:
    """A config change over the SAME files must not reuse the old manifest.

    The stat checks describe the corpus and cannot see this: change a row cut and
    every block boundary moves while every file stays byte-identical. The config
    digest recorded in the artifact is what turns that into a hard error.
    """
    corpus = _corpus(tmp_path / "corpus", ["a.root"])
    target = tmp_path / "m.json"
    _dm(corpus, manifest=target).prepare_data()

    cut = FileStubReader(files=[], fingerprint={"cuts": "pt > 20"})
    with pytest.raises(ConfigError, match="config digest"):
        _dm(corpus, manifest=target, reader=cut).prepare_data()


def test_a_matching_fingerprint_reuses(tmp_path) -> None:
    """The digest round-trips: the same configuration reads its own artifact back."""
    corpus = _corpus(tmp_path / "corpus", ["a.root"])
    target = tmp_path / "m.json"
    cut = {"cuts": "pt > 20"}
    _dm(corpus, manifest=target, reader=FileStubReader(files=[], fingerprint=cut)).prepare_data()

    FileStubReader.reset()
    _dm(corpus, manifest=target, reader=FileStubReader(files=[], fingerprint=cut)).prepare_data()
    assert FileStubReader.opens == 0


def test_a_foreign_salt_stamp_is_a_hard_error(tmp_path) -> None:
    """An artifact written by a different salt is stale by identity — never re-read."""
    corpus = _corpus(tmp_path / "corpus", ["a.root"])
    target = tmp_path / "m.json"
    target.write_text('{"salt_version": "0.0.0", "entries": [], "group_names": ["default"]}')
    with pytest.raises(ConfigError, match="delete"):
        _dm(corpus, manifest=target).prepare_data()


# --------------------------------------------------------------------------- #
# the artifact itself
# --------------------------------------------------------------------------- #


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


def test_a_failed_write_leaves_neither_a_partial_nor_a_temp_file(tmp_path, monkeypatch) -> None:
    """Atomicity: the destination only ever appears complete."""
    root = tmp_path / "corpus"
    reader = FileStubReader(files=[]).with_source(_corpus(root, ["a.root"]))
    manifest = build_manifest(reader)
    target = root / "broken.json"

    def boom(*_args, **_kwargs) -> str:
        raise OSError("disk full")

    monkeypatch.setattr(json, "dumps", boom)
    with pytest.raises(OSError, match="disk full"):
        manifest.save(target)
    assert not target.exists()
    assert list(root.glob("*.tmp")) == []
    assert list(root.glob(".*")) == []


def test_manifest_entry_paths_survive_the_round_trip(tmp_path) -> None:
    """The stat facts validation needs are what `build_manifest` records."""
    root = tmp_path / "corpus"
    reader = FileStubReader(files=[]).with_source(_corpus(root, ["a.root", "b.root"]))
    entries = build_manifest(reader).entries
    assert [Path(e.path).name for e in entries] == ["a.root", "b.root"]
    assert all(e.size > 0 and e.mtime_ns > 0 for e in entries)
    assert isinstance(entries[0], ManifestEntry)
