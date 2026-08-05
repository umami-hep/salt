"""`GraphDataModule` — Lightning wiring for the v2 dataset pipeline.

Per-stage readers cloned via `Reader.with_source`; batch-returning dataset
(no collate); declared setup graph for VDS resolution and optional staging.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path

import lightning
from torch.utils.data import DataLoader

from salt.data.base import Reader, SaltDatasetModule, SetupBundle
from salt.data.dataset import GraphDataset
from salt.data.input_samples import InputSamples, deepest_source_path, source_num
from salt.data.readers.vds import VDS
from salt.data.samplers import RandomBatchSampler
from salt.graph.errors import ConfigError
from salt.graph.planner import compile_setup_plan
from salt.graph.setup_executor import run_setup_plan
from salt.graph.spec import PRIMARY_MODES, Mode
from salt.logging import get_logger

__all__ = ["GraphDataModule"]

_LOG = get_logger(__name__)

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


class GraphDataModule(lightning.LightningDataModule):
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
    prefetch_factor : int, optional
        Batches prefetched per worker, by default 2.
    multiprocessing_context : str | None, optional
        Worker start method (``"fork"`` / ``"spawn"`` / ``"forkserver"``).
    seed : int, optional
        Base augmentation seed for non-worker reads, by default 42.
    debug : bool, optional
        Enable the boundary non-aliasing assertion.

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
        prefetch_factor: int = 2,
        multiprocessing_context: str | None = None,
        seed: int = 42,
        debug: bool = False,
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
                f"GraphDataModule needs exactly one Reader in modules, got {len(readers)} "
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
        # they never reach GraphDataset, so neither the dead-module check nor
        # GraphDataset's single-Reader guard miscounts them.
        self._setup_modules = {name: m for name, m in modules.items() if _is_setup_only(m)}
        self._batch_modules = {
            name: m for name, m in modules.items() if name not in self._setup_modules
        }
        self._modules = modules
        self._setup_ctx: SetupBundle | None = None
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
        self.train_dset: GraphDataset | None = None
        self.val_dset: GraphDataset | None = None
        self.test_dset: GraphDataset | None = None

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
                f"GraphDataModule allows at most one InputSamples, got {len(existing)} "
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
                f"GraphDataModule allows at most one VDS, got {len(existing)} "
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
        """
        if self._setup_ctx is None:
            self._setup_ctx = SetupBundle()
        if not self._setup_modules:
            return
        stage_tuple = tuple(stages)
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

    def _make_dataset(
        self,
        mode: Mode,
        filename: str | Path | None,
        num: int,
        vds_path: str | Path | None,
    ) -> GraphDataset:
        """Clone the reader prototype onto a stage file and build its dataset.

        Deep-copies processors per stage — bind-time state (e.g. Labels'
        narrowed key set) must not leak across train/val/test plans sharing
        this module dict.
        """
        if filename is None:
            raise ConfigError(f"no file configured for mode {mode.name}")
        if self._sinks is None:
            raise ConfigError(
                "GraphDataModule has no sinks — pass sinks= or call set_sinks() with the "
                "model boundary's demanded keys before setup"
            )
        reader = self._reader_proto.with_source(
            filename=filename, num=num, vds_path=vds_path, stage=_STAGE_OF_MODE[mode]
        )
        reader = self._stage(reader)
        # deep-copy the processors per stage: bind-time state (e.g. the Labels
        # narrowed key set) is per-(dataset, mode) and must not leak between
        # the train/val/test plans sharing this module dict. Only
        # `_batch_modules` are copied — setup-only modules are never handed to
        # GraphDataset.
        modules = {
            name: (reader if name == self._reader_name else deepcopy(module))
            for name, module in self._batch_modules.items()
        }
        return GraphDataset(
            modules,
            mode=mode,
            sinks=self._sinks,
            seed=self.seed,
            debug=self.debug,
            sink_origins=(self._sink_origins or {}).get(mode),
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
            train_file, num_train = self._resolve_source(Mode.FIT)
            val_file, num_val = self._resolve_source(Mode.VAL)
            self.train_dset = self._make_dataset(
                Mode.FIT, train_file, num_train, self.train_vds_path
            )
            self.val_dset = self._make_dataset(Mode.VAL, val_file, num_val, self.val_vds_path)
            if self.trainer is None or self.trainer.is_global_zero:
                _LOG.info(f"Created training dataset with {len(self.train_dset):,} entries")
                _LOG.info(f"Created validation dataset with {len(self.val_dset):,} entries")
        if stage == "test":
            test_file, num_test = self._resolve_source(Mode.TEST)
            if test_file is None:
                raise ConfigError("No test file specified, see --data.test_file")
            self.test_dset = self._make_dataset(Mode.TEST, test_file, num_test, self.test_vds_path)
            _LOG.info(f"Created test dataset with {len(self.test_dset):,} entries")

    @staticmethod
    def _modes_for_stage(stage: str) -> tuple[Mode, ...]:
        """The per-batch modes a Lightning stage builds (``setup("fit")`` -> FIT+VAL)."""
        if stage == "fit":
            return (Mode.FIT, Mode.VAL)
        if stage == "test":
            return (Mode.TEST,)
        return ()

    def get_dataloader(self, stage: str, dataset: GraphDataset, shuffle: bool) -> DataLoader:
        """Build a batch-sampler dataloader over a stage dataset.

        ``batch_size=None`` + `RandomBatchSampler`: the dataset receives
        contiguous slices and returns complete batches; there is no collate step.
        ``drop_last`` only for fit.
        """
        drop_last = stage == "fit"
        return DataLoader(
            dataset=dataset,
            batch_size=None,
            collate_fn=None,
            sampler=RandomBatchSampler(dataset, self.batch_size, shuffle, drop_last=drop_last),
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
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
