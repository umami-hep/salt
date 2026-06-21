"""`GraphDataModule` — Lightning wiring for the v2 dataset pipeline (design §6.1).

The v1 ``SaltDataModule`` contract kept wholesale (``datamodules.py``): one
module dict shared across the three stages, per-stage readers cloned from the
single configured reader prototype via `H5StructuredReader.with_source`,
``DataLoader(batch_size=None, collate_fn=None, sampler=RandomBatchSampler)``
(``datamodules.py:247-260``), ``drop_last`` only for fit, weak shuffling of
batch order only. Per-worker handle wiring needs no
``worker_init_fn``: binding is pid-guarded and lazy inside
`GraphDataset.__getitem__`, exactly like v1's ``_setup`` path.

Wildcard→VDS resolution (v1's rank-0 VDS pre-creation + DDP barrier,
``datamodules.py:129-188``) is now a declared setup-graph module (`VDS`,
plan-25 W3.B): auto-injected alongside `InputSamples`, it produces
``source.<reader>.<stage>.vds_path`` and `create_vds`'s FileLock + ``.done``
marker make every-rank execution safe with no rank-0 gating or barrier.

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
from torch.utils.data import DataLoader

from salt.core.data.base import DatasetModule, Reader, SetupBundle
from salt.core.data.dataset import GraphDataset
from salt.core.data.input_samples import InputSamples, deepest_source_path, source_num
from salt.core.data.samplers import RandomBatchSampler
from salt.core.data.vds_module import VDS
from salt.core.graph.errors import ConfigError
from salt.core.graph.planner import compile_setup_plan
from salt.core.graph.setup_executor import run_setup_plan
from salt.core.graph.spec import PRIMARY_MODES, Mode

__all__ = ["GraphDataModule"]

# Map the per-stage Mode to the stage key passed through Reader.with_source(stage=...)
# (the plan-02 per-reader stage-sourcing hook). Single-source readers ignore it;
# MultiSampleReader uses it to select each sub-reader's per-stage source.
_STAGE_OF_MODE: dict[Mode, str] = {Mode.FIT: "train", Mode.VAL: "val", Mode.TEST: "test"}

# The setup stages a setup-only module is probed over (plan-24 §3.3).
_SETUP_STAGES: tuple[str, ...] = ("train", "val", "test")


def _is_setup_only(module: DatasetModule) -> bool:
    """Whether `module` participates ONLY in the setup graph (plan-24 §4.4).

    A setup-only module declares a non-empty `declare_setup_io` for SOME stage
    AND an empty `declare_io` for EVERY per-batch mode. Such a module (e.g.
    `InputSamples`/`VDS`/`ShmStage`) MUST be partitioned out before the
    per-batch dataset deepcopy, or `compile_plan`'s `_check_all_modes_dead`
    (planner.py — a module inactive in every mode raises `AllModesDeadError`)
    and `GraphDataset`'s single-Reader guard would both miscount it (plan-25
    §3.6). A dual-face reader has a non-empty `declare_io`, so it stays in the
    per-batch set and appears in BOTH graphs.

    Returns
    -------
    bool
        True when the module is setup-only.
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
        # Single-Reader guard FIRST (over all modules): a Reader always has a
        # non-empty per-batch face, so it is never setup-only and the count is
        # the same before/after the partition. Doing it up front gives us
        # `_reader_name` so InputSamples can be wired (plan-25 §3.8) BEFORE the
        # partition probes `declare_setup_io` (which needs the reader name).
        readers = [(name, m) for name, m in modules.items() if isinstance(m, Reader)]
        if len(readers) != 1:
            raise ConfigError(
                f"GraphDataModule needs exactly one Reader in modules, got {len(readers)} "
                f"({[name for name, _ in readers]}) (design §6.1)"
            )
        self._reader_name, self._reader_proto = readers[0]
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        # plan-25 W3.A: the data-sourcing setup graph. If a config declares an
        # `InputSamples` setup module it OWNS the per-stage source patterns; the
        # deprecated train_file/val_file/test_file kwargs synthesise an IMPLICIT
        # InputSamples for one migration window (plan-25 O-ALIAS-WINDOW / §3.3)
        # so old configs + exp-12's `--data.train_file` keep working unchanged.
        # MUST run before the partition: wires InputSamples._reader so its
        # `declare_setup_io` (probed by `_is_setup_only`) can build keys.
        self._wire_input_samples(modules)
        # plan-25 W3.B: the wildcard→VDS resolution setup module. Auto-injected
        # whenever an InputSamples is present (explicit or alias-synthesised) and
        # no explicit VDS is configured; wires `_reader` + `_vds_capable` on the
        # VDS (auto-injected OR explicit). MUST also run before the partition so
        # the (setup-only) VDS lands in `_setup_modules`. The *_vds_path kwargs
        # feed the auto-injected VDS's per-stage `out`, so they are assigned here
        # (before `_wire_vds`), not with the rest of the plumbing below.
        self.train_vds_path = train_vds_path
        self.val_vds_path = val_vds_path
        self.test_vds_path = test_vds_path
        self._wire_vds(modules)
        # plan-24 §4.4 / plan-25 §3.6 namespace split: partition setup-only
        # modules OUT before the per-batch dataset deepcopy. setup-only modules
        # (InputSamples/VDS/ShmStage) never reach GraphDataset, so neither
        # `_check_all_modes_dead` nor GraphDataset's single-Reader guard
        # miscounts them. The setup pass iterates `_setup_modules` + the
        # dual-face reader (which stays in `_batch_modules`).
        self._setup_modules = {name: m for name, m in modules.items() if _is_setup_only(m)}
        self._batch_modules = {
            name: m for name, m in modules.items() if name not in self._setup_modules
        }
        self._modules = modules
        self._setup_ctx: SetupBundle | None = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_suff = test_suff
        # opt-in staging root (M6 sub-wave E, gate S31; v1 datamodules.py:91).
        # None -> staging is a no-op (default-off, byte-identical read path).
        # `move_files_temp` is the backward-compat config name; `_stage_root` is the
        # internal name the thin setup('fit')/teardown('fit') trigger reads (M8 wave 3:
        # the fat prepare_data/setup-repoint subsystem collapsed to reader.restage()).
        self.move_files_temp = move_files_temp
        self._stage_root: Path | None = None
        # (train/val/test)_vds_path are assigned earlier, before `_wire_vds`.
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

    def _wire_input_samples(self, modules: dict[str, DatasetModule]) -> None:
        """Assemble the data-sourcing setup graph (plan-25 §3.8 / W3.A).

        Two responsibilities:

        1. **Alias migration window** (plan-25 O-ALIAS-WINDOW): if no
           `InputSamples` is configured but the deprecated
           ``train_file``/``val_file``/``test_file`` kwargs are set, synthesise
           an IMPLICIT `InputSamples` from them (with the matching per-stage
           ``num``) and add it to the setup-only namespace. This keeps old
           configs and exp-12's ``--data.train_file`` working unchanged.
        2. **Reader-name wiring** (plan-25 §3.8): the ``source.<reader>.*`` keys
           are embedded at *declaration* time, so each `InputSamples` instance is
           poked with the single reader's name AFTER the single-Reader guard.

        Parameters
        ----------
        modules : dict[str, DatasetModule]
            The full assembled module dict (mutated in place when an implicit
            `InputSamples` is synthesised).

        Raises
        ------
        ConfigError
            If more than one `InputSamples` is configured (the single Reader can
            have exactly one source owner, plan-25 §3.1).
        """
        existing = [(name, m) for name, m in modules.items() if isinstance(m, InputSamples)]
        if len(existing) > 1:
            raise ConfigError(
                f"GraphDataModule allows at most one InputSamples, got {len(existing)} "
                f"({[name for name, _ in existing]}); one InputSamples owns the single "
                "Reader's source chain (plan-25 §3.1)"
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
                # add to the module dict; the partition right after picks it up
                # into `_setup_modules` via `_is_setup_only` (it is setup-only).
                modules["input_samples"] = implicit
                existing = [("input_samples", implicit)]
        # plan-25 §3.8 reader-name wiring: embed the single Reader's name so the
        # produced source.<reader>.<stage>.pattern keys match the handoff.
        for _, samples in existing:
            samples._reader = self._reader_name  # noqa: SLF001 — assembly poke (plan-25 §3.8)
        self._input_samples: InputSamples | None = existing[0][1] if existing else None

    def _wire_vds(self, modules: dict[str, DatasetModule]) -> None:
        """Assemble the wildcard→VDS setup module (plan-25 §5.1 / W3.B).

        Two responsibilities, mirroring `_wire_input_samples`:

        1. **Auto-injection** (plan-25 §3.3 shape B + §5.1 "VDS, PATH, always"):
           shape-B YAML shows only ``input_samples`` with glob values and NO
           explicit ``vds:`` block — so a `VDS` must be present transparently.
           Whenever an `InputSamples` is present (explicit or alias-synthesised)
           and no explicit `VDS` is configured, synthesise one (with per-stage
           ``out`` paths from the deprecated ``train_vds_path``/``val_vds_path``/
           ``test_vds_path`` kwargs) and add it to the setup-only namespace.
        2. **Reader wiring** (plan-25 §3.8): poke the single Reader's name and
           its `vds_capable` flag onto the `VDS` (auto-injected OR explicit), so
           its ``declare_setup_io``/``setup`` keys match the handoff and the
           build-vs-identity choice is gated on the reader's capability.

        An auto-injected `VDS` is only synthesised when an `InputSamples` exists:
        with no source owner there is no ``pattern`` for the `VDS` to consume.
        An explicit `VDS` in ``data.modules`` overrides the auto-injection and
        carries its own O-VDS-OUT paths.

        Parameters
        ----------
        modules : dict[str, DatasetModule]
            The full assembled module dict (mutated in place when an implicit
            `VDS` is synthesised).

        Raises
        ------
        ConfigError
            If more than one `VDS` is configured (one VDS owns the single
            Reader's wildcard resolution, plan-25 §5.1).
        """
        existing = [(name, m) for name, m in modules.items() if isinstance(m, VDS)]
        if len(existing) > 1:
            raise ConfigError(
                f"GraphDataModule allows at most one VDS, got {len(existing)} "
                f"({[name for name, _ in existing]}); one VDS owns the single Reader's "
                "wildcard resolution (plan-25 §5.1)"
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
            # add to the module dict; the partition right after picks it up into
            # `_setup_modules` via `_is_setup_only` (a VDS is setup-only).
            modules["vds"] = implicit
            existing = [("vds", implicit)]
        # plan-25 §3.8 reader wiring: embed the single Reader's name (for the
        # source.<reader>.* keys) and its vds_capable flag (gates build-vs-identity,
        # O-VDS-CAP) on the VDS (auto-injected OR explicit).
        for _, vds in existing:
            vds._reader = self._reader_name  # noqa: SLF001 — assembly poke (plan-25 §3.8)
            vds._vds_capable = self._reader_proto.vds_capable  # noqa: SLF001 — assembly poke
        self._vds: VDS | None = existing[0][1] if existing else None

    @property
    def modules(self) -> dict[str, DatasetModule]:
        """All assembled dataset modules by instance name — the UNION (plan-25 §3.6).

        Includes BOTH the per-batch modules and the setup-only modules
        (InputSamples/VDS/ShmStage). ``salt2 graph`` renders the setup graph and
        the per-batch graph as two distinct topologies, so the per-batch compile
        must use `batch_modules`, NOT this union (a setup-only module in the
        per-batch compile trips `AllModesDeadError`, plan-25 §3.6).

        Returns
        -------
        dict[str, DatasetModule]
            A fresh dict of the assembled (None-filtered) modules.
        """
        return dict(self._modules)

    @property
    def batch_modules(self) -> dict[str, DatasetModule]:
        """The PER-BATCH modules (reader + processors), excluding setup-only ones.

        The namespace handed to `GraphDataset` / the per-batch `compile_plan`
        (plan-25 §3.6): setup-only modules are partitioned out so the per-batch
        compile and its single-Reader guard see exactly the tensor pipeline.

        Returns
        -------
        dict[str, DatasetModule]
            A fresh dict of the per-batch modules.
        """
        return dict(self._batch_modules)

    @property
    def setup_modules(self) -> dict[str, DatasetModule]:
        """The SETUP-only modules (InputSamples/VDS/ShmStage) — the setup graph.

        Returns
        -------
        dict[str, DatasetModule]
            A fresh dict of the setup-only modules.
        """
        return dict(self._setup_modules)

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
        self._batch_modules[self._reader_name] = reader

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

    # -- the data-sourcing setup pass (plan-24 §4.3 / plan-25 W3.A) ------------

    def _run_setup_pass(self, stages: Iterable[str]) -> None:
        """Compile + run the setup-graph plan once per `stages` into one ctx.

        The setup analogue of building the per-batch plans (plan-24 §4.3): for
        each stage it compiles the setup plan over ``_setup_modules`` and walks
        it (`run_setup_plan`) into ONE shared write-once ctx, so e.g.
        ``setup("fit")`` accumulates both ``"train"`` and ``"val"`` into disjoint
        stage-qualified keys (plan-24 §3.5). ``_make_dataset`` then reads the
        resolved deepest PATH off this ctx. A no-op when no setup modules are
        configured (the ctx stays empty and the kwargs path is used).

        DDP-safe by construction (plan-24 §4.6): every rank runs the identical
        pass; W3.A modules touch no filesystem, so there is no copy to serialise.

        Parameters
        ----------
        stages : Iterable[str]
            The setup stages to resolve (``"train"``/``"val"``/``"test"``).
        """
        if self._setup_ctx is None:
            self._setup_ctx = SetupBundle()
        if not self._setup_modules:
            return
        stage_tuple = tuple(stages)
        # The whole-dict `num` SCALAR is written ONCE per ctx (plan-25 §3.7): tell
        # the InputSamples which pass stage carries it, and whether a prior pass
        # already wrote it into this shared ctx (so a `test` pass after `fit`
        # doesn't re-emit the {train,val,test} cap dict).
        if self._input_samples is not None:
            num_key = f"artifacts.{self._reader_name}.num"
            self._input_samples.set_num_stage(
                stage_tuple, already_emitted=num_key in self._setup_ctx
            )
        for stage in stage_tuple:
            plan = compile_setup_plan(self._setup_modules, stage)  # type: ignore[arg-type]
            run_setup_plan(plan, stage, self._setup_ctx)  # type: ignore[arg-type]

    def _resolve_source(self, mode: Mode) -> tuple[str | Path | None, int]:
        """Resolve the stage's source PATH + row cap from the setup ctx (plan-25 §4.0).

        The datamodule handoff glue: when an `InputSamples` resolved this stage,
        return its deepest present PATH (W3.A: ``pattern``) and the per-stage
        ``num`` off the whole-dict SCALAR leaf. With no setup ctx populated (no
        `InputSamples` and no aliases) it falls back to the legacy
        ``train_file``/``num_train`` kwargs — the migration window's last resort.

        Returns
        -------
        tuple[str | Path | None, int]
            ``(filename, num)`` for `mode`'s stage.
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
        # the train/val/test plans sharing this module dict (design §2.3).
        # Only `_batch_modules` are copied — setup-only modules (plan-24 §4.4)
        # are NEVER handed to GraphDataset, so the per-batch compile and the
        # single-Reader guard see exactly the per-batch namespace.
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
        ``stage='fit'`` leak into validation), test the TEST plan. The
        data-sourcing setup graph runs FIRST (plan-25 W3.A): the setup pass
        resolves each stage's source PATH onto one write-once ctx, then
        ``_make_dataset`` binds the reader from the ctx-resolved deepest key via
        the existing handoff (``with_source(filename=ctx_path, num=..., stage=)``).
        Wildcard→VDS resolution is now part of that setup pass (the `VDS` module,
        plan-25 W3.B) — the rank-0 VDS precreation + DDP barrier are gone, since
        `create_vds`'s FileLock + ``.done`` marker make every-rank execution safe
        (plan-25 §5.1 / R-DDP). When opt-in temp staging is active
        (``move_files_temp`` set, not ``fast_dev_run``) the ``_stage_root`` is
        armed BEFORE dataset building, so every per-stage reader restages its own
        files onto the root (M8 wave 3).

        Raises
        ------
        ConfigError
            If the stage's file or the sinks are unset.
        """
        self._auto_sinks()
        self._stage_root = self._resolve_stage_root(stage)
        # plan-25 W3.A: run the data-sourcing setup pass once per stage into one
        # ctx (setup("fit") resolves both train+val). _resolve_source then reads
        # the deepest PATH off this ctx; pure path arithmetic, no FS I/O.
        self._run_setup_pass(_STAGE_OF_MODE[m] for m in self._modes_for_stage(stage))
        if stage == "fit":
            train_file, num_train = self._resolve_source(Mode.FIT)
            val_file, num_val = self._resolve_source(Mode.VAL)
            self.train_dset = self._make_dataset(
                Mode.FIT, train_file, num_train, self.train_vds_path
            )
            self.val_dset = self._make_dataset(Mode.VAL, val_file, num_val, self.val_vds_path)
            if self.trainer is None or self.trainer.is_global_zero:
                print(f"Created training dataset with {len(self.train_dset):,} entries")
                print(f"Created validation dataset with {len(self.val_dset):,} entries")
        if stage == "test":
            test_file, num_test = self._resolve_source(Mode.TEST)
            if test_file is None:
                raise ConfigError("No test file specified, see --data.test_file")
            self.test_dset = self._make_dataset(
                Mode.TEST, test_file, num_test, self.test_vds_path
            )
            print(f"Created test dataset with {len(self.test_dset):,} entries")

    @staticmethod
    def _modes_for_stage(stage: str) -> tuple[Mode, ...]:
        """The per-batch modes a Lightning stage builds (``setup("fit")`` → FIT+VAL).

        Returns
        -------
        tuple[Mode, ...]
            The modes whose source PATH the setup pass must resolve for `stage`.
        """
        if stage == "fit":
            return (Mode.FIT, Mode.VAL)
        if stage == "test":
            return (Mode.TEST,)
        return ()

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
