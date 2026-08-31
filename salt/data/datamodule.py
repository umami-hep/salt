"""`SaltDataModule` — Lightning wiring for the v2 dataset pipeline.

Per-stage readers cloned via `Reader.with_source`; batch-returning dataset
(no collate); declared setup graph for VDS resolution and optional staging.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import lightning
from torch.utils.data import DataLoader

from salt.data.base import Reader, SaltDatasetModule, SetupBundle
from salt.data.dataset import SaltDataset
from salt.data.input_samples import InputSamples, deepest_source_path, source_num
from salt.data.iterable_dataset import DEFAULT_BLOCK_ROWS, IterableSaltDataset
from salt.data.manifest import CorpusManifest, apply_schema, ensure_manifest
from salt.data.readers.vds import VDS
from salt.data.samplers import RandomBatchSampler
from salt.graph.errors import ConfigError
from salt.graph.planner import compile_setup_plan
from salt.graph.setup_executor import run_setup_plan
from salt.graph.spec import PRIMARY_MODES, Mode
from salt.utils.logging import get_logger

__all__ = ["AUTO_PREFETCH_CAP", "SaltDataModule", "auto_prefetch_factor"]

_LOG = get_logger(__name__)

AUTO_PREFETCH_CAP = 8
"""Upper bound on the prefetch depth derived for a streaming loader.

A streaming worker's output is BURSTY: it spends one long reader call filling a
`block_rows`-row block, then emits ``block_rows / batch_size`` batches almost
free. A `DataLoader` returns batches in strict round-robin worker order, so if a
worker can only buffer a couple of them it blocks on the queue, and the consumer
then waits out the NEXT worker's whole block read while every other worker sits
full and idle. Covering roughly one block of batches is what removes that stall.

The cap exists because the depth is paid in shared memory: torch's
``file_descriptor`` sharing strategy holds ``num_workers * prefetch_factor *
batch_bytes`` of ``/dev/shm`` in flight, and that scales with the worker count on
every rank. Measured on the FTAG1LITE corpus (8-file subset, 19.4 MB batches,
`block_rows` 16,384, so a full block is 17 batches):

===========  ============  ============  ============  ============
prefetch     nw4 jets/s    nw4 shm       nw8 jets/s    nw8 shm
===========  ============  ============  ============  ============
2 (old)      6,940         150 MB        8,182         311 MB
8 (cap)      8,762         505 MB        13,170        1,127 MB
17 (block)   9,983         1,030 MB      14,056        2,338 MB
24           10,148        1,419 MB      14,915        3,012 MB
===========  ============  ============  ============  ============

The cap takes 88-89% of the deepest rate for 37% of its memory, which is the
right side of that trade when the target is many ranks each running several
workers. Raise `prefetch_factor` explicitly to buy the remainder.

Note what the table does NOT show: shrinking `block_rows` does not shrink this.
Shared memory holds prefetched BATCHES, so at a fixed depth it is unchanged
(``block_rows`` 8,192 at depth 8 measured 1,108 MB against 16,384's 1,127 MB) —
a shorter block only lowers the depth NEEDED, and measured 13,214 jets/s against
13,170, i.e. no gain. Below that, 4,096-row blocks cost read rate (11,691 at
nw8), matching the read-range curve. So depth, not block size, is the knob.
"""


def auto_prefetch_factor(
    explicit: int | None,
    iterable: bool,
    block_rows: int | None,
    batch_size: int,
) -> int:
    """Batches to prefetch per worker, deriving the streaming depth when unset.

    An `explicit` value is returned untouched — a user who has sized their own
    shared memory is never overridden. ``None`` derives: 2 for the map-style
    path, whose workers produce at a steady cadence so extra depth buys nothing,
    and for `iterable` enough depth to cover one reader block
    (``ceil(block_rows / batch_size)``), clamped to ``[2, AUTO_PREFETCH_CAP]``.

    ``block_rows=None`` means the reader's own blocks are read whole; that is at
    least as bursty as any configured block, so it takes the cap.

    Parameters
    ----------
    explicit : int | None
        The configured `prefetch_factor`, or ``None`` to derive one.
    iterable : bool
        Whether the streaming dataset is in use.
    block_rows : int | None
        Rows per reader call on the streaming path.
    batch_size : int
        Rows per emitted batch.

    Returns
    -------
    int
        The prefetch depth to hand the `DataLoader` (always >= 2).
    """
    if explicit is not None:
        return int(explicit)
    if not iterable:
        return 2
    if block_rows is None:
        return AUTO_PREFETCH_CAP
    per_block = -(-int(block_rows) // int(batch_size))  # ceil
    return max(2, min(AUTO_PREFETCH_CAP, per_block))


# Map the per-stage Mode to the stage key passed through Reader.with_source(stage=...).
# Single-source readers ignore it; MultiSampleReader uses it to select each
# sub-reader's per-stage source.
_STAGE_OF_MODE: dict[Mode, str] = {Mode.FIT: "train", Mode.VAL: "val", Mode.TEST: "test"}

_SETUP_STAGES: tuple[str, ...] = ("train", "val", "test")


def _is_setup_only(module: SaltDatasetModule) -> bool:
    """Whether `module` declares setup IO for some stage but no per-batch IO.

    Such modules (InputSamples/VDS/ShmStage) must be partitioned out before
    the per-batch dataset deepcopy so the dead-module check and single-Reader
    guard don't miscount them; a dual-face reader stays in the batch set.
    """
    has_setup = any(
        not module.declare_setup_io(stage).is_empty()  # type: ignore[arg-type]
        for stage in _SETUP_STAGES
    )
    if not has_setup:
        return False
    has_batch = any(
        module.declare_io(mode).requires or module.declare_io(mode).produces
        for mode in PRIMARY_MODES
    )
    return not has_batch


class SaltDataModule(lightning.LightningDataModule):
    """LightningDataModule running the v2 dataset pipeline.

    Parameters
    ----------
    modules : dict[str, SaltDatasetModule | None]
        Dataset modules by instance name — exactly one `Reader` prototype
        (typically unbound; per-stage clones get the stage file) plus the
        processors. ``None`` entries are dropped (config null-deletion).
    train_file : str | Path | None, optional
        Training file path (wildcards trigger VDS creation).
    val_file : str | Path | None, optional
        Validation file path.
    test_file : str | Path | None, optional
        Test file path.
    batch_size : int, optional
        Rows per contiguous batch slab, by default 1000.
    num_workers : int, optional
        Dataloader worker processes, by default 0.
    num_train, num_val, num_test : int, optional
        Row counts per stage; ``-1`` = all.
    test_suff : str | None, optional
        Suffix appended to the eval-file ``{sample}`` name by the writer
        callback, by default None.
    move_files_temp : str | None, optional
        Opt-in staging root (e.g. ``/dev/shm/<user>/tmp``) for the fit reader's
        files. When set, ``setup('fit')`` arms the root and each per-stage
        `Reader` restages its own file(s) there, and ``teardown('fit')`` removes
        the root. ``None`` (default) leaves the read path byte-identical.
        Ignored under ``fast_dev_run``.
    train_vds_path, val_vds_path, test_vds_path : str | Path | None, optional
        Explicit VDS output paths for wildcard files.
    sinks : Mapping[Mode, Iterable[str]] | None, optional
        Per-mode model-boundary demand (``inputs.* / masks.* / labels.* /
        meta.rows`` keys). Set here or later via `set_sinks` — the
        `SaltModule` integration calls `set_sinks` with the model plans' source
        demands before ``setup``.
    pin_memory : bool, optional
        Pin host memory for faster GPU transfer, by default True.
    persistent_workers : bool, optional
        Keep worker processes (and their H5 handles) alive between epochs,
        by default True.
    prefetch_factor : int | None, optional
        Batches prefetched per worker. ``None`` (the default) derives it: 2 for
        the map-style path, and for `iterable` enough depth to cover one reader
        block (see `AUTO_PREFETCH_CAP`). An explicit integer always wins.
    multiprocessing_context : str | None, optional
        Worker start method (``"fork"`` / ``"spawn"`` / ``"forkserver"``).
    seed : int, optional
        Base augmentation seed for non-worker reads, by default 42.
    debug : bool, optional
        Enable the boundary non-aliasing assertion.
    iterable : bool, optional
        Build `IterableSaltDataset` (sequential, sharded streaming) instead of
        the map-style `SaltDataset`, by default False. The streaming path reads
        large contiguous blocks and shards across ranks x workers by row
        interval, which is what lets a run scale past the point where every
        reader process can hold a row-granular corpus index. Downstream is
        identical — same plan, same batch object.
    block_rows : int | None, optional
        Rows per reader call when `iterable`, by default `DEFAULT_BLOCK_ROWS`
        (16,384 — the measured knee of the read-range curve); ``None`` reads
        each of the reader's own blocks whole.
    interleave_block : int, optional
        Rows per turn of the per-shard sample round robin when `iterable`, by
        default 1.
    max_live_streams : int | None, optional
        Samples holding a resident block at once when `iterable`, by default 2.
        Peak resident rows per worker is ``max_live_streams * block_rows``.
    manifest : str | Path | Mapping[str, str | Path] | None, optional
        `CorpusManifest` path(s) — REQUIRED when `iterable`. With a manifest,
        shard planning and epoch length need no reader index and open no data
        file. A mapping gives one path per stage
        (``{train: ..., val: ..., test: ...}``); a single path is shared by
        every streaming stage, which is only valid when they read the same
        corpus. An existing file is validated and read; a missing one is built
        there by `prepare_data` (global rank 0, atomic write). A stale manifest
        is a hard error naming the file to delete — it is the user's file and
        is never silently overwritten.
    shuffle_stream : bool, optional
        Shuffle block order and within-batch row order on the fit streaming
        loader, by default True. Val/test always stream in order.

    Raises
    ------
    ConfigError
        If a `modules` entry is not a `SaltDatasetModule`, or `modules` does
        not contain exactly one `Reader`.
    """

    def __init__(
        self,
        modules: dict[str, SaltDatasetModule | None],
        train_file: str | Path | None = None,
        val_file: str | Path | None = None,
        test_file: str | Path | None = None,
        batch_size: int = 1000,
        num_workers: int = 0,
        num_train: int = -1,
        num_val: int = -1,
        num_test: int = -1,
        test_suff: str | None = None,
        move_files_temp: str | None = None,
        train_vds_path: str | Path | None = None,
        val_vds_path: str | Path | None = None,
        test_vds_path: str | Path | None = None,
        sinks: Mapping[Mode, Iterable[str]] | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int | None = None,
        multiprocessing_context: str | None = None,
        seed: int = 42,
        debug: bool = False,
        iterable: bool = False,
        block_rows: int | None = DEFAULT_BLOCK_ROWS,
        interleave_block: int = 1,
        max_live_streams: int | None = 2,
        manifest: str | Path | Mapping[str, str | Path] | None = None,
        shuffle_stream: bool = True,
    ) -> None:
        super().__init__()
        # a None module entry (config-file or CLI override) deletes the module.
        modules = {name: module for name, module in modules.items() if module is not None}
        for name, module in modules.items():
            if not isinstance(module, SaltDatasetModule):
                raise ConfigError(
                    f"module {name!r} ({type(module).__name__}) is not a SaltDatasetModule — "
                    "data-graph entries must subclass SaltDatasetModule; wrap or extend it"
                )
            module.name = name
        # single-Reader guard FIRST, over all modules, before the setup-only
        # partition below: gives us `_reader_name` so InputSamples can be wired
        # before `declare_setup_io` (which needs the reader name) is probed.
        readers = [(name, m) for name, m in modules.items() if isinstance(m, Reader)]
        if len(readers) != 1:
            raise ConfigError(
                f"SaltDataModule needs exactly one Reader in modules, got {len(readers)} "
                f"({[name for name, _ in readers]})"
            )
        self._reader_name, self._reader_proto = readers[0]
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        # If a config declares an `InputSamples` setup module it owns the
        # per-stage source patterns; the deprecated train_file/val_file/test_file
        # kwargs synthesise an implicit InputSamples so old configs keep working.
        # Must run before the partition below: wires InputSamples._reader so its
        # `declare_setup_io` (probed by `_is_setup_only`) can build keys.
        self._wire_input_samples(modules)
        # The wildcard->VDS resolution setup module. Auto-injected whenever an
        # InputSamples is present and no explicit VDS is configured; wires
        # `_reader` + `_vds_capable` on the VDS. Must also run before the
        # partition so the (setup-only) VDS lands in `_setup_modules`.
        self.train_vds_path = train_vds_path
        self.val_vds_path = val_vds_path
        self.test_vds_path = test_vds_path
        self._wire_vds(modules)
        # Partition setup-only modules OUT before the per-batch dataset deepcopy:
        # they never reach SaltDataset, so neither the dead-module check nor
        # SaltDataset's single-Reader guard miscounts them.
        self._setup_modules = {name: m for name, m in modules.items() if _is_setup_only(m)}
        self._batch_modules = {
            name: m for name, m in modules.items() if name not in self._setup_modules
        }
        self._modules = modules
        self._setup_ctx: SetupBundle | None = None
        # stages already run into `_setup_ctx`. The ctx is write-once, so a
        # stage may be planned exactly once — and `prepare_data` runs the same
        # pass `setup` would, before it.
        self._setup_done: set[str] = set()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_suff = test_suff
        # None -> staging is a no-op (default-off, byte-identical read path).
        # `move_files_temp` is the config name; `_stage_root` is the internal name
        # setup('fit')/teardown('fit') read.
        self.move_files_temp = move_files_temp
        self._stage_root: Path | None = None
        self._sinks = dict(sinks) if sinks is not None else None
        self._sink_origins: dict[Mode, dict[str, str]] | None = None
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.multiprocessing_context = multiprocessing_context
        self.seed = seed
        self.debug = debug
        self.iterable = bool(iterable)
        self.block_rows = block_rows
        self.interleave_block = int(interleave_block)
        self.max_live_streams = max_live_streams
        if isinstance(manifest, str) and manifest.strip().lower() == "auto":
            raise ConfigError(
                "manifest: 'auto' was removed — give an explicit manifest path "
                "(salt builds the file there when it is missing, and never chooses "
                "a storage location itself)"
            )
        if self.iterable and manifest is None:
            raise ConfigError(
                "data.manifest is required when data.iterable=true — give the path each "
                "streaming stage plans its shards from (a missing file is built there on "
                "rank 0); use a mapping, manifest: {train: ..., val: ..., test: ...}, when "
                "stages read different corpora"
            )
        self.manifest = manifest
        self.shuffle_stream = bool(shuffle_stream)
        # The manifest is ONE artifact on shared storage, so exactly one process
        # in the job builds it — not one per node, which is Lightning's default
        # and is right only for a per-node download. Without a shared filesystem
        # set this back to True; the atomic write makes that race wasteful,
        # never corrupting.
        self.prepare_data_per_node = False
        self.train_dset: SaltDataset | IterableSaltDataset | None = None
        self.val_dset: SaltDataset | IterableSaltDataset | None = None
        self.test_dset: SaltDataset | IterableSaltDataset | None = None

    def _wire_input_samples(self, modules: dict[str, SaltDatasetModule]) -> None:
        """Assemble the data-sourcing setup graph (mutates `modules` in place).

        1. Alias migration: with no `InputSamples` configured but the
           deprecated train_file/val_file/test_file kwargs set, synthesise an
           implicit `InputSamples` and add it to the setup-only namespace.
        2. Reader-name wiring: poke the single reader's name onto each
           `InputSamples` instance so its ``source.<reader>.*`` keys match.
        """
        existing = [(name, m) for name, m in modules.items() if isinstance(m, InputSamples)]
        if len(existing) > 1:
            raise ConfigError(
                f"SaltDataModule allows at most one InputSamples, got {len(existing)} "
                f"({[name for name, _ in existing]}); one InputSamples owns the single "
                "Reader's source chain"
            )
        if not existing:
            files = {
                stage: f
                for stage, f in (
                    ("train", self.train_file),
                    ("val", self.val_file),
                    ("test", self.test_file),
                )
                if f is not None
            }
            if files:
                num = {"train": self.num_train, "val": self.num_val, "test": self.num_test}
                implicit = InputSamples(files=files, num={s: num[s] for s in files})
                implicit.name = "input_samples"
                # partition right after picks this up into `_setup_modules`
                # (it is setup-only).
                modules["input_samples"] = implicit
                existing = [("input_samples", implicit)]
        # embed the single Reader's name so the produced
        # source.<reader>.<stage>.pattern keys match the handoff.
        for _, samples in existing:
            samples._reader = self._reader_name  # noqa: SLF001 — assembly poke
        self._input_samples: InputSamples | None = existing[0][1] if existing else None

    def _wire_vds(self, modules: dict[str, SaltDatasetModule]) -> None:
        """Assemble the wildcard->VDS setup module (mirrors `_wire_input_samples`).

        1. Auto-injection: when an `InputSamples` is present and no explicit
           `VDS` is configured, synthesise one from the deprecated
           train_vds_path/val_vds_path/test_vds_path kwargs (only when an
           `InputSamples` exists — otherwise there is no ``pattern`` to consume).
        2. Reader wiring: poke the reader's name and `vds_capable` flag onto
           the VDS, gating its build-vs-identity choice.

        An explicit `VDS` in ``data.modules`` overrides auto-injection.
        """
        existing = [(name, m) for name, m in modules.items() if isinstance(m, VDS)]
        if len(existing) > 1:
            raise ConfigError(
                f"SaltDataModule allows at most one VDS, got {len(existing)} "
                f"({[name for name, _ in existing]}); one VDS owns the single Reader's "
                "wildcard resolution"
            )
        if not existing and self._input_samples is not None:
            out = {
                stage: p
                for stage, p in (
                    ("train", self.train_vds_path),
                    ("val", self.val_vds_path),
                    ("test", self.test_vds_path),
                )
                if p is not None
            }
            implicit = VDS(out=out or None)
            implicit.name = "vds"
            # partition right after picks this up into `_setup_modules`
            # (a VDS is setup-only).
            modules["vds"] = implicit
            existing = [("vds", implicit)]
        # embed the single Reader's name (for the source.<reader>.* keys) and
        # its vds_capable flag (gates build-vs-identity) on the VDS.
        for _, vds in existing:
            vds._reader = self._reader_name  # noqa: SLF001 — assembly poke
            vds._vds_capable = self._reader_proto.vds_capable  # noqa: SLF001 — assembly poke
        self._vds: VDS | None = existing[0][1] if existing else None

    @property
    def modules(self) -> dict[str, SaltDatasetModule]:
        """All assembled dataset modules (both graphs) — use `batch_modules` for compiling."""
        return dict(self._modules)

    @property
    def batch_modules(self) -> dict[str, SaltDatasetModule]:
        """The per-batch modules (reader + processors), excluding setup-only ones."""
        return dict(self._batch_modules)

    @property
    def setup_modules(self) -> dict[str, SaltDatasetModule]:
        """The setup-only modules (InputSamples/VDS/ShmStage) — the setup graph."""
        return dict(self._setup_modules)

    @property
    def reader(self) -> Reader:
        """The configured reader prototype."""
        return self._reader_proto

    @reader.setter
    def reader(self, reader: Reader) -> None:
        """Replace the reader prototype (used by the staging trigger); keeps `modules` in sync."""
        reader.name = self._reader_name
        self._reader_proto = reader
        self._modules[self._reader_name] = reader
        self._batch_modules[self._reader_name] = reader

    def set_sinks(self, sinks: Mapping[Mode, Iterable[str]]) -> None:
        """Set the per-mode model-boundary demand.

        Must be called (or `sinks` passed at construction) before ``setup``,
        unless the attached LightningModule exposes ``sink_demand()`` (the
        `SaltModule` integration) — `setup` then adopts that automatically.
        """
        self._sinks = dict(sinks)

    def _auto_sinks(self) -> None:
        """Adopt the attached model's `sink_demand()` when no sinks were set
        (duck-typed; explicit sinks win).
        """
        if self._sinks is not None:
            return
        model = self.trainer.lightning_module if self.trainer is not None else None
        demand = getattr(model, "sink_demand", None)
        if callable(demand):
            self._sinks = dict(demand())
            origins = getattr(model, "sink_origins", None)
            if callable(origins):
                self._sink_origins = {mode: dict(who) for mode, who in origins().items()}

    def _run_setup_pass(self, stages: Iterable[str]) -> None:
        """Compile + run the setup-graph plan once per `stages` into one shared ctx.

        Accumulates disjoint stage-qualified keys across stages (e.g.
        ``setup("fit")`` covers both "train" and "val"); `_make_dataset` reads
        the resolved deepest path off this ctx. No-op with no setup modules.
        DDP-safe: every rank runs the identical pass (no filesystem touched).

        Idempotent per stage: the ctx is write-once, so an already-planned stage
        is skipped rather than re-planned — `prepare_data` runs this pass for
        the stages it must resolve sources for, and `setup` then runs whatever
        is left.
        """
        if self._setup_ctx is None:
            self._setup_ctx = SetupBundle()
        if not self._setup_modules:
            return
        stage_tuple = tuple(s for s in stages if s not in self._setup_done)
        if not stage_tuple:
            return
        # The whole-dict `num` scalar is written once per ctx: tell InputSamples
        # which pass stage carries it, and whether a prior pass already wrote it
        # into this shared ctx (so a `test` pass after `fit` doesn't re-emit the
        # {train,val,test} cap dict).
        if self._input_samples is not None:
            num_key = f"artifacts.{self._reader_name}.num"
            self._input_samples.set_num_stage(
                stage_tuple, already_emitted=num_key in self._setup_ctx
            )
        for stage in stage_tuple:
            plan = compile_setup_plan(self._setup_modules, stage)  # type: ignore[arg-type]
            run_setup_plan(plan, stage, self._setup_ctx)  # type: ignore[arg-type]
            self._setup_done.add(stage)

    def _resolve_source(self, mode: Mode) -> tuple[str | Path | None, int]:
        """Resolve the stage's source path + row cap from the setup ctx.

        Uses the `InputSamples`-resolved deepest path + per-stage ``num`` when
        available; falls back to the legacy ``train_file``/``num_train`` kwargs.
        """
        stage = _STAGE_OF_MODE[mode]
        if self._input_samples is not None and self._setup_ctx is not None:
            key = f"source.{self._reader_name}.{stage}.pattern"
            if key in self._setup_ctx:
                filename = deepest_source_path(self._setup_ctx, self._reader_name, stage)
                num = source_num(self._setup_ctx, self._reader_name, stage)
                return filename, num
        # legacy fallback (no InputSamples, no aliases): the raw kwargs.
        legacy = {
            Mode.FIT: (self.train_file, self.num_train),
            Mode.VAL: (self.val_file, self.num_val),
            Mode.TEST: (self.test_file, self.num_test),
        }
        return legacy[mode]

    def _stage_reader(self, mode: Mode) -> tuple[Reader | None, int]:
        """The stage's reader clone (config-only, unstaged, unprepared) and its row cap.

        Shared by `prepare_data` and `_make_dataset` so both key a manifest off
        exactly the same reader. `None` when the stage has no source configured.
        """
        filename, num = self._resolve_source(mode)
        if filename is None:
            return None, num
        vds = {
            Mode.FIT: self.train_vds_path,
            Mode.VAL: self.val_vds_path,
            Mode.TEST: self.test_vds_path,
        }[mode]
        reader = self._reader_proto.with_source(
            filename=filename, num=num, vds_path=vds, stage=_STAGE_OF_MODE[mode]
        )
        return reader, num

    def _make_dataset(self, mode: Mode) -> SaltDataset | IterableSaltDataset:
        """Clone the reader prototype onto a stage file and build its dataset.

        Deep-copies processors per stage — bind-time state (e.g. Labels'
        narrowed key set) must not leak across train/val/test plans sharing
        this module dict.
        """
        reader, _num = self._stage_reader(mode)
        if reader is None:
            raise ConfigError(f"no file configured for mode {mode.name}")
        if self._sinks is None:
            raise ConfigError(
                "SaltDataModule has no sinks — pass sinks= or call set_sinks() with the "
                "model boundary's demanded keys before setup"
            )
        # Resolved BEFORE staging: the manifest describes the corpus the user
        # configured, not the /dev/shm copies a staged run reads, so staging
        # never invalidates it (the blocks are the same rows either way).
        manifest = self._manifest_for(mode, reader)
        reader = self._stage(reader)
        if manifest is not None and apply_schema(manifest, reader):
            _LOG.info(
                f"manifest: seeded the {_STAGE_OF_MODE[mode]} reader's schema from the manifest "
                "— plan compilation opens no data file"
            )
        # only `_batch_modules` are copied — setup-only modules never reach a dataset
        modules = {
            name: (reader if name == self._reader_name else deepcopy(module))
            for name, module in self._batch_modules.items()
        }
        common = {
            "mode": mode,
            "sinks": self._sinks,
            "seed": self.seed,
            "debug": self.debug,
            "sink_origins": (self._sink_origins or {}).get(mode),
        }
        if not self.iterable:
            return SaltDataset(modules, **common)
        # Streaming: shuffle and drop-last are FIT-only. Val/test stream in
        # order and keep the ragged tail, which is the eval writers'
        # row-alignment contract — the same split the map-style loaders make
        # through the sampler, expressed on the dataset because an
        # IterableDataset has no sampler to make it.
        fit = mode == Mode.FIT
        return IterableSaltDataset(
            modules,
            batch_size=self.batch_size,
            shuffle=self.shuffle_stream and fit,
            drop_last=fit,
            block_rows=self.block_rows,
            interleave_block=self.interleave_block,
            max_live_streams=self.max_live_streams,
            manifest=manifest,
            **common,
        )

    def _resolve_stage_root(self, stage: str) -> Path | None:
        """The opt-in staging root for this stage, or None.

        Staging is on only for ``fit``, only when a ``move_files_temp`` root is
        configured, and only when the run is not a ``fast_dev_run``. With
        ``move_files_temp=None`` (the default) this is always None, so `_stage`
        is a no-op and the read path is byte-identical.
        """
        if stage != "fit" or not self.move_files_temp:
            return None
        if self.trainer is not None and self.trainer.fast_dev_run:
            return None
        return Path(self.move_files_temp)

    def _stage(self, reader: Reader) -> Reader:
        """Restage the reader onto ``_stage_root`` (FileLock-coordinated) when staging is
        active; returns the reader unchanged when ``_stage_root`` is None.
        """
        if self._stage_root is None:
            return reader
        return reader.restage(self._stage_root)

    # -- corpus manifest ------------------------------------------------------

    def _manifest_path(self, mode: Mode) -> Path:
        """The configured manifest path for this stage.

        A mapping is per-stage; a scalar is one path shared by every streaming
        stage (valid only when they read the same corpus — `ensure_manifest`
        errors otherwise).
        """
        if isinstance(self.manifest, Mapping):
            stage = _STAGE_OF_MODE[mode]
            if stage not in self.manifest:
                raise ConfigError(
                    f"data.manifest has no entry for stage {stage!r} — a manifest mapping "
                    f"needs a path per streaming stage, got {sorted(self.manifest)}"
                )
            return Path(self.manifest[stage])
        assert self.manifest is not None  # enforced in __init__ for iterable runs
        return Path(self.manifest)

    def prepare_data(self) -> None:
        """Build any missing corpus manifest — once, on rank 0.

        Lightning calls this hook on global rank 0 alone
        (``prepare_data_per_node = False``) and barriers every rank before
        ``setup``, which is exactly the coordination a built artifact needs: one
        writer, no lock, and everyone else finds a finished file. Idempotent — a
        second call validates and reuses the artifact the first one wrote; a
        stale one is a hard error naming the file to delete.
        """
        if not self.iterable:
            if self.manifest is not None:
                _LOG.warning(
                    "data.manifest is ignored on the map-style path — a corpus manifest "
                    "plans streaming shards, and this run has data.iterable=false"
                )
            return
        # the hook takes no stage argument; the trainer's entry point is what
        # says whether a test corpus is about to be read or is merely configured
        fn = getattr(getattr(self.trainer, "state", None), "fn", None)
        stage = "test" if str(getattr(fn, "value", "")).startswith("test") else "fit"
        modes = self._modes_for_stage(stage)
        self._run_setup_pass(_STAGE_OF_MODE[m] for m in modes)
        for mode in modes:
            reader, _num = self._stage_reader(mode)
            if reader is None:
                continue
            ensure_manifest(reader, self._manifest_path(mode), _STAGE_OF_MODE[mode])

    def _manifest_for(self, mode: Mode, reader: Reader) -> CorpusManifest | None:
        """The manifest this stage's streaming dataset plans from; None on map-style.

        The same `ensure_manifest` call `prepare_data` makes: inside a `Trainer`
        the file already exists (built behind the rank-0 barrier), so this only
        validates and reads it. A datamodule driven outside a `Trainer` builds
        here instead — uncoordinated ranks can then waste the build, never
        corrupt it, thanks to the atomic write.
        """
        if not self.iterable:
            return None
        return ensure_manifest(reader, self._manifest_path(mode), _STAGE_OF_MODE[mode])

    def setup(self, stage: str) -> None:
        """Build the per-stage datasets: ``"fit"`` -> FIT+VAL, ``"test"`` -> TEST.

        Runs the data-sourcing setup pass (source resolution + wildcard->VDS)
        into one write-once ctx first, then binds each stage's reader from it.
        """
        self._auto_sinks()
        self._stage_root = self._resolve_stage_root(stage)
        # setup("fit") resolves both train+val into one ctx; _resolve_source then
        # reads the deepest path off this ctx (pure path arithmetic, no FS I/O).
        self._run_setup_pass(_STAGE_OF_MODE[m] for m in self._modes_for_stage(stage))
        if stage == "fit":
            self.train_dset = self._make_dataset(Mode.FIT)
            self.val_dset = self._make_dataset(Mode.VAL)
            if self.trainer is None or self.trainer.is_global_zero:
                _LOG.info(f"Created training dataset with {len(self.train_dset):,} entries")
                _LOG.info(f"Created validation dataset with {len(self.val_dset):,} entries")
        if stage == "test":
            if self._resolve_source(Mode.TEST)[0] is None:
                raise ConfigError("No test file specified, see --data.test_file")
            self.test_dset = self._make_dataset(Mode.TEST)
            _LOG.info(f"Created test dataset with {len(self.test_dset):,} entries")

    @staticmethod
    def _modes_for_stage(stage: str) -> tuple[Mode, ...]:
        """The per-batch modes a Lightning stage builds (``setup("fit")`` -> FIT+VAL)."""
        if stage == "fit":
            return (Mode.FIT, Mode.VAL)
        if stage == "test":
            return (Mode.TEST,)
        return ()

    def resolved_prefetch_factor(self) -> int:
        """The effective prefetch depth for this datamodule (see `auto_prefetch_factor`)."""
        return auto_prefetch_factor(
            explicit=self.prefetch_factor,
            iterable=self.iterable,
            block_rows=self.block_rows,
            batch_size=self.batch_size,
        )

    def get_dataloader(
        self, stage: str, dataset: SaltDataset | IterableSaltDataset, shuffle: bool
    ) -> DataLoader:
        """Build a dataloader over a stage dataset.

        Map-style: ``batch_size=None`` + `RandomBatchSampler` — the dataset
        receives contiguous slices and returns complete batches; there is no
        collate step, and ``drop_last`` applies only to fit.

        Streaming: NO sampler at all. An `IterableDataset` yields whole batches
        itself and resolves its own shard from the worker/rank context, so a
        sampler would be both meaningless and (under DDP) actively wrong. The
        epoch is pushed onto the dataset here, because that is the one thing the
        map-style path gets from the sampler and streaming has nowhere else to
        get.
        """
        drop_last = stage == "fit"
        sampler: Any = None
        if isinstance(dataset, IterableSaltDataset):
            if self.trainer is not None:
                dataset.epoch = int(self.trainer.current_epoch)
        else:
            sampler = RandomBatchSampler(dataset, self.batch_size, shuffle, drop_last=drop_last)
        return DataLoader(
            dataset=dataset,
            batch_size=None,
            collate_fn=None,
            sampler=sampler,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.resolved_prefetch_factor() if self.num_workers > 0 else None,
            multiprocessing_context=self.multiprocessing_context,
        )

    def train_dataloader(self) -> DataLoader:
        """Training loader: weak shuffling, drop_last."""
        assert self.train_dset is not None, "setup('fit') has not run"
        return self.get_dataloader(dataset=self.train_dset, stage="fit", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Validation loader: sequential, no drop_last."""
        assert self.val_dset is not None, "setup('fit') has not run"
        return self.get_dataloader(dataset=self.val_dset, stage="test", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Test loader: sequential, no drop_last — the writer row-alignment contract."""
        assert self.test_dset is not None, "setup('test') has not run"
        return self.get_dataloader(dataset=self.test_dset, stage="test", shuffle=False)

    def teardown(self, stage: str | None = None) -> None:
        """Remove the staging root after fit, when staging is on (global-zero rank
        only, else no-op).
        """
        root = self._resolve_stage_root("fit") if stage == "fit" else None
        if root is None:
            return
        if self.trainer is not None and not self.trainer.is_global_zero:
            return
        import shutil

        _LOG.info("-" * 100)
        _LOG.info(f"Removing staged files under {root}")
        shutil.rmtree(root, ignore_errors=True)
        _LOG.info("-" * 100)
