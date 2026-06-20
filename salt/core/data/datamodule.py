"""`GraphDataModule` — Lightning wiring for the v2 dataset pipeline (design §6.1).

The v1 ``SaltDataModule`` contract kept wholesale (``datamodules.py``): one
module dict shared across the three stages, per-stage readers cloned from the
single configured reader prototype via `H5StructuredReader.with_source`,
``DataLoader(batch_size=None, collate_fn=None, sampler=RandomBatchSampler)``
(``datamodules.py:247-260``), ``drop_last`` only for fit, weak shuffling of
batch order only, rank-0 VDS pre-creation + dist barrier
(``datamodules.py:129-188``). Per-worker handle wiring needs no
``worker_init_fn``: binding is pid-guarded and lazy inside
`GraphDataset.__getitem__`, exactly like v1's ``_setup`` path.

Differences from v1, by design: the val dataset compiles the VAL plan
(v1 passed ``stage='fit'`` to the val dataset, leaking parameter
randomisation into validation — design §6.2 fixes this), and read-time
transforms are FIT-only (design §6.1).

``move_files_temp`` staging (M6 sub-wave E, gate S31; FD §6.1 1292, §10 1779;
**reader-owned as of M8 wave 3**): an OPT-IN, default-off file-staging surface.
M8 collapsed the v1 fat 3-hook subsystem (``prepare_data`` copy + ``setup``
file-repoint + ``teardown`` remove, ``datamodules.py:191-204,274-280``) into a
thin trigger: the datamodule no longer copies files itself — it arms a
``_stage_root`` in ``setup('fit')`` and each per-stage `Reader` restages its OWN
files there (`Reader.restage` -> `salt.core.data.vds.stage_file`, reusing the
FileLock + ``.done``-marker copy machinery). This auto-handles multi-file and
multi-sample readers (the v1 path staged only ``train_file``/``val_file`` and
silently dropped the rest). ``teardown('fit')`` removes the ``_stage_root`` tree.
With ``move_files_temp=None`` (the default) ``_stage`` is identity and the read
path is byte-identical. The v1 ``config_s3`` auto-download glue (file_utils
``import_data_S3`` / ``setup_S3_CLI``) stays a CLI-side concern reachable via the
``download_S3`` console entry — it is NOT wired into the datamodule (the v2 reader
takes already-local paths).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path

import lightning
import torch
from torch.utils.data import DataLoader

from salt.core.data.base import DatasetModule, Reader
from salt.core.data.dataset import GraphDataset
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import Mode
from salt.core.data.samplers import RandomBatchSampler

__all__ = ["GraphDataModule"]

# Map the per-stage Mode to the stage key passed through Reader.with_source(stage=...)
# (the plan-02 per-reader stage-sourcing hook). Single-source readers ignore it;
# MultiSampleReader uses it to select each sub-reader's per-stage source.
_STAGE_OF_MODE: dict[Mode, str] = {Mode.FIT: "train", Mode.VAL: "val", Mode.TEST: "test"}


class GraphDataModule(lightning.LightningDataModule):
    """LightningDataModule running the v2 dataset pipeline (design §6.1).

    Parameters
    ----------
    modules : dict[str, DatasetModule | None]
        Dataset modules by instance name — exactly one `Reader` prototype
        (typically unbound; per-stage clones get the stage file) plus the
        processors. Maps 1:1 onto the ``data.modules`` YAML block (§5.1).
        ``None`` entries are dropped — the assembly-time half of the design
        §5.3 null-deletion semantics (``--data.modules.X=null``).
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
        Row counts per stage; ``-1`` = all (v1 semantics).
    test_suff : str | None, optional
        Suffix appended to the eval-file ``{sample}`` name by the writer
        callback (v1 ``SaltDataModule.test_suff``, ``datamodules.py:113``;
        design §8), by default None.
    move_files_temp : str | None, optional
        OPT-IN staging root (e.g. ``/dev/shm/<user>/tmp``) for the fit reader's
        files (M6 sub-wave E, gate S31; v1 ``datamodules.py:91``). When set,
        ``setup('fit')`` arms the root and each per-stage `Reader` restages its own
        file(s) there (`Reader.restage`; M8 wave 3 — multi-file / multi-sample safe),
        and ``teardown('fit')`` removes the root. ``None`` (default) leaves the read
        path byte-identical. Ignored under ``fast_dev_run`` (v1 ``datamodules.py:191,201``).
    train_vds_path, val_vds_path, test_vds_path : str | Path | None, optional
        Explicit VDS output paths for wildcard files.
    sinks : Mapping[Mode, Iterable[str]] | None, optional
        Per-mode model-boundary demand (``inputs.* / masks.* / labels.* /
        meta.rows`` keys). Set here or later via `set_sinks` — the
        `SaltModule` integration (stage B) calls `set_sinks` with the
        model plans' source demands before ``setup``.
    pin_memory : bool, optional
        Pin host memory for faster GPU transfer, by default True.
    persistent_workers : bool, optional
        Keep worker processes (and their H5 handles) alive between epochs,
        by default True (``datamodules.py:54-61`` rationale).
    prefetch_factor : int, optional
        Batches prefetched per worker, by default 2 — safe by construction
        under the contract-#9 copy boundary (design §2.4).
    multiprocessing_context : str | None, optional
        Worker start method (``"fork"`` / ``"spawn"`` / ``"forkserver"``).
    seed : int, optional
        Base augmentation seed for non-worker reads, by default 42.
    debug : bool, optional
        Enable the boundary non-aliasing assertion (design §2.4).

    Raises
    ------
    ConfigError
        If `modules` does not contain exactly one `Reader`.
    """

    def __init__(
        self,
        modules: dict[str, DatasetModule | None],
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
        # assembly-time None filtering (design §5.3): a null entry — from a
        # config-file or CLI override — deletes the module.
        modules = {name: module for name, module in modules.items() if module is not None}
        for name, module in modules.items():
            module.name = name
        readers = [(name, m) for name, m in modules.items() if isinstance(m, Reader)]
        if len(readers) != 1:
            raise ConfigError(
                f"GraphDataModule needs exactly one Reader in modules, got {len(readers)} "
                f"({[name for name, _ in readers]}) (design §6.1)"
            )
        self._reader_name, self._reader_proto = readers[0]
        self._modules = modules
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.test_suff = test_suff
        # opt-in staging root (M6 sub-wave E, gate S31; v1 datamodules.py:91).
        # None -> staging is a no-op (default-off, byte-identical read path).
        # `move_files_temp` is the backward-compat config name; `_stage_root` is the
        # internal name the thin setup('fit')/teardown('fit') trigger reads (M8 wave 3:
        # the fat prepare_data/setup-repoint subsystem collapsed to reader.restage()).
        self.move_files_temp = move_files_temp
        self._stage_root: Path | None = None
        self.train_vds_path = train_vds_path
        self.val_vds_path = val_vds_path
        self.test_vds_path = test_vds_path
        self._sinks = dict(sinks) if sinks is not None else None
        # per-mode demand provenance (M3-review fix: a merged map mis-attributed
        # writer-demanded TEST keys to their inactive FIT demander)
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

    @property
    def modules(self) -> dict[str, DatasetModule]:
        """The configured dataset modules by instance name (read-only view).

        Returns
        -------
        dict[str, DatasetModule]
            A fresh dict of the assembled (None-filtered) modules — used by
            the static graph tooling (``salt2 graph``) to build the
            full-pipeline graph from a §5.1 config.
        """
        return dict(self._modules)

    @property
    def reader(self) -> Reader:
        """The configured reader prototype.

        Returns
        -------
        Reader
            The single reader module the per-stage clones derive from.
        """
        return self._reader_proto

    @reader.setter
    def reader(self, reader: Reader) -> None:
        """Replace the reader prototype (the M8 wave-3 ``reader.restage`` trigger).

        ``setup('fit')`` reassigns ``self.reader = self.reader.restage(root)`` when
        opt-in staging is active, so the per-stage clones derive from the staged
        reader. Keeps the instance ``name`` and the ``modules`` view consistent.
        """
        reader.name = self._reader_name
        self._reader_proto = reader
        self._modules[self._reader_name] = reader

    def set_sinks(self, sinks: Mapping[Mode, Iterable[str]]) -> None:
        """Set the per-mode model-boundary demand (the stage-B wiring hook).

        Must be called (or `sinks` passed at construction) before ``setup``,
        unless the attached LightningModule exposes ``sink_demand()`` (the
        `SaltModule` integration) — `setup` then adopts that automatically.
        """
        self._sinks = dict(sinks)

    def _auto_sinks(self) -> None:
        """Adopt the attached model's boundary demand when no sinks were set.

        Stage-B wiring (design §3.4 symmetry): a `SaltModule` attached to the
        same trainer declares its per-mode dataset-boundary demand via
        ``sink_demand()`` — duck-typed here so the data side stays free of a
        model-side import. Explicit sinks (constructor or `set_sinks`) win.
        """
        if self._sinks is not None:
            return
        model = self.trainer.lightning_module if self.trainer is not None else None
        demand = getattr(model, "sink_demand", None)
        if callable(demand):
            self._sinks = dict(demand())
            # per-mode demand provenance for §4.1 error attribution (optional
            # hook; SaltModule.sink_origins returns {Mode: {key: who}})
            origins = getattr(model, "sink_origins", None)
            if callable(origins):
                self._sink_origins = {mode: dict(who) for mode, who in origins().items()}

    # -- stage plumbing --------------------------------------------------------

    def _make_dataset(
        self,
        mode: Mode,
        filename: str | Path | None,
        num: int,
        vds_path: str | Path | None,
    ) -> GraphDataset:
        """Clone the reader prototype onto a stage file and build its dataset.

        Returns
        -------
        GraphDataset
            The stage dataset.

        Raises
        ------
        ConfigError
            If the stage file or the sinks are unset.
        """
        if filename is None:
            raise ConfigError(f"no file configured for mode {mode.name} (design §6.1)")
        if self._sinks is None:
            raise ConfigError(
                "GraphDataModule has no sinks — pass sinks= or call set_sinks() with the "
                "model boundary's demanded keys before setup (design §3.3, §6.1)"
            )
        reader = self._reader_proto.with_source(
            filename=filename, num=num, vds_path=vds_path, stage=_STAGE_OF_MODE[mode]
        )
        reader = self._stage(reader)
        # deep-copy the processors per stage: bind-time state (e.g. the Labels
        # narrowed key set) is per-(dataset, mode) and must not leak between
        # the train/val/test plans sharing this module dict (design §2.3)
        modules = {
            name: (reader if name == self._reader_name else deepcopy(module))
            for name, module in self._modules.items()
        }
        return GraphDataset(
            modules,
            mode=mode,
            sinks=self._sinks,
            seed=self.seed,
            debug=self.debug,
            sink_origins=(self._sink_origins or {}).get(mode),
        )

    @staticmethod
    def _dist_barrier() -> None:
        """Synchronise all distributed ranks (no-op outside DDP, ``datamodules.py:129-137``)."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _precreate_vds_rank0(self, stage: str) -> None:
        """Resolve VDS files on global rank 0 to avoid the startup stampede.

        Port of ``datamodules.py:139-188``: correctness is enforced by the
        FileLock inside `salt.core.data.vds.create_vds`; this is a
        performance/robustness optimisation. Other ranks wait on the barrier.
        """
        if self.trainer is None or not self.trainer.is_global_zero:
            return
        if stage == "fit":
            for filename, num, vds, stage_key in (
                (self.train_file, self.num_train, self.train_vds_path, "train"),
                (self.val_file, self.num_val, self.val_vds_path, "val"),
            ):
                if filename is not None:
                    reader = self._reader_proto.with_source(filename, num, vds, stage=stage_key)
                    self._stage(reader).prepare()
        elif stage == "test" and self.test_file is not None:
            reader = self._reader_proto.with_source(
                self.test_file, self.num_test, self.test_vds_path, stage="test"
            )
            self._stage(reader).prepare()

    def _resolve_stage_root(self, stage: str) -> Path | None:
        """The opt-in staging root for this stage, or None (the thin M8 trigger).

        Staging is on only for ``fit``, only when a ``move_files_temp`` root is
        configured, and (v1 guard, ``datamodules.py:191,201,275``) only when the run
        is not a ``fast_dev_run``. With ``move_files_temp=None`` (the default) this is
        always None, so `_stage` is a no-op and the read path is byte-identical.

        Returns
        -------
        Path | None
            The stage-root directory, or None when staging is inactive.
        """
        if stage != "fit" or not self.move_files_temp:
            return None
        if self.trainer is not None and self.trainer.fast_dev_run:
            return None
        return Path(self.move_files_temp)

    def _stage(self, reader: Reader) -> Reader:
        """Restage a per-stage reader onto ``_stage_root`` when staging is active (M8).

        The thin trigger replacing the v1 fat ``prepare_data`` copy + ``setup``
        repoint block: with ``_stage_root`` set, the reader copies its OWN sourced
        file(s) under the root (rank-0 + FileLock coordinated inside
        `Reader.restage` -> `vds.stage_file`) and returns a clone reading the copies —
        single-file H5, multi-file easyjet, and multi-sample readers all stage their
        full source set (the v1 path staged only ``train_file``/``val_file``). With
        ``_stage_root`` None this returns the reader unchanged.

        Returns
        -------
        Reader
            The staged clone, or ``reader`` unchanged when staging is inactive.
        """
        if self._stage_root is None:
            return reader
        return reader.restage(self._stage_root)

    def setup(self, stage: str) -> None:
        """Build the per-stage datasets (``datamodules.py:197-245`` contract).

        Train compiles the FIT plan, val the VAL plan (fixing v1's
        ``stage='fit'`` leak into validation), test the TEST plan. When opt-in
        temp staging is active (``move_files_temp`` set, not ``fast_dev_run``) the
        ``_stage_root`` is armed BEFORE VDS precreation / dataset building, so every
        per-stage reader restages its own files onto the root (M8 wave 3: the fat
        ``prepare_data``/``setup``-repoint subsystem collapsed to `Reader.restage`).

        Raises
        ------
        ConfigError
            If the stage's file or the sinks are unset.
        """
        self._auto_sinks()
        self._stage_root = self._resolve_stage_root(stage)
        if stage in {"fit", "test"} and self.trainer is not None:
            self._precreate_vds_rank0(stage)
            self._dist_barrier()
        if stage == "fit":
            self.train_dset = self._make_dataset(
                Mode.FIT, self.train_file, self.num_train, self.train_vds_path
            )
            self.val_dset = self._make_dataset(
                Mode.VAL, self.val_file, self.num_val, self.val_vds_path
            )
            if self.trainer is None or self.trainer.is_global_zero:
                print(f"Created training dataset with {len(self.train_dset):,} entries")
                print(f"Created validation dataset with {len(self.val_dset):,} entries")
        if stage == "test":
            if self.test_file is None:
                raise ConfigError("No test file specified, see --data.test_file")
            self.test_dset = self._make_dataset(
                Mode.TEST, self.test_file, self.num_test, self.test_vds_path
            )
            print(f"Created test dataset with {len(self.test_dset):,} entries")

    # -- dataloaders (datamodules.py:247-269 kept wholesale) -------------------

    def get_dataloader(self, stage: str, dataset: GraphDataset, shuffle: bool) -> DataLoader:
        """Build a batch-sampler dataloader over a stage dataset.

        ``batch_size=None`` + `RandomBatchSampler` (reused v1 class,
        ``samplers.py``): the dataset receives contiguous slices and returns
        complete batches; there is no collate step. ``drop_last`` only for
        fit (``datamodules.py:248``).

        Returns
        -------
        DataLoader
            The configured loader.
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
        """Training loader: weak shuffling, drop_last.

        Returns
        -------
        DataLoader
            The train loader.
        """
        assert self.train_dset is not None, "setup('fit') has not run"
        return self.get_dataloader(dataset=self.train_dset, stage="fit", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Validation loader: sequential, no drop_last (``datamodules.py:265-266``).

        Returns
        -------
        DataLoader
            The val loader.
        """
        assert self.val_dset is not None, "setup('fit') has not run"
        return self.get_dataloader(dataset=self.val_dset, stage="test", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Test loader: sequential, no drop_last — the writer row-alignment contract.

        Returns
        -------
        DataLoader
            The test loader.
        """
        assert self.test_dset is not None, "setup('test') has not run"
        return self.get_dataloader(dataset=self.test_dset, stage="test", shuffle=False)

    def teardown(self, stage: str | None = None) -> None:
        """Remove the staging root after fit when staging is on (M8 wave-3 thin trigger).

        The cleanup half of the collapsed staging subsystem (v1
        ``datamodules.py:271-281``): a no-op unless ``move_files_temp`` is set (and not
        ``fast_dev_run``); only the global-zero rank cleans up, and only after ``fit``.
        Removes the ENTIRE ``_stage_root`` tree (every staged copy + its FileLock /
        ``.done`` markers — single-file, multi-file, and multi-sample staging all land
        under the one root), so it generalises the v1 two-file ``remove_files_temp``.
        With ``move_files_temp=None`` this never touches the filesystem.

        Parameters
        ----------
        stage : str | None, optional
            The Lightning stage being torn down, by default None.
        """
        root = self._resolve_stage_root("fit") if stage == "fit" else None
        if root is None:
            return
        if self.trainer is not None and not self.trainer.is_global_zero:
            return
        import shutil  # noqa: PLC0415 - opt-in staging path only

        print("-" * 100)
        print(f"Removing staged files under {root}")
        shutil.rmtree(root, ignore_errors=True)
        print("-" * 100)
