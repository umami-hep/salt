"""Picklable harness for the real 2-rank gloo + forkserver streaming test: everything a
spawned rank or a forkserver DataLoader worker unpickles lives here, importable by name.
"""

from __future__ import annotations

import json
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch

from salt.data.base import Reader, RowBlock, WorkerCtx
from salt.data.datamodule import SaltDataModule
from salt.data.processors.features import Features
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.schema import GroupSchema, Schema

N_ROWS = 238
"""Rows in the single-group corpus; uids ``0..237``."""

BATCH = 8
N_BLOCKS = 3

_ROW = np.dtype([("uid", "int64"), ("val", "float32")])


class StreamStubReader(Reader):
    """In-memory single-group reader over the ``event`` stream; ``uid`` is the row index.

    No `files`/`_table`, so its manifest is sourceless and path-less.
    """

    def __init__(self, n_rows: int = N_ROWS, n_blocks: int = N_BLOCKS, seed: int = 0):
        super().__init__()
        self.n_blocks = int(n_blocks)
        rng = np.random.default_rng(seed)
        self._rows = np.empty((int(n_rows),), dtype=_ROW)
        self._rows["uid"] = np.arange(int(n_rows), dtype=np.int64)
        self._rows["val"] = rng.uniform(0, 1, size=int(n_rows)).astype(np.float32)
        self.schema = Schema(
            groups={"event": GroupSchema(fields={"uid": "int64", "val": "float32"})}
        )

    @property
    def streams(self) -> tuple[str, ...]:
        return ("event",)

    def declare_io(self, mode: Mode) -> IO:
        del mode
        flat = {"raw.event": TensorSpec(shape=("B",), kind="data", fields=("uid", "val"))}
        return IO(produces=unflatten_spec(flat))

    def __len__(self) -> int:
        return len(self._rows)

    def row_blocks(self) -> list[RowBlock]:
        """`n_blocks` near-equal blocks of group 0."""
        n = len(self._rows)
        edges = [(i * n) // self.n_blocks for i in range(self.n_blocks + 1)]
        return [
            RowBlock(group=0, start=lo, stop=hi)
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)
            if hi > lo
        ]

    def read_block(self, block: RowBlock, mode: Mode) -> dict[str, np.ndarray]:
        del mode
        return {"raw.event": self._rows[block.start : block.stop].copy()}

    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        del mode
        return {"raw.event": self._rows[rows].copy()}

    def with_source(self, filename, num=-1, stage=None):  # noqa: ANN001
        del filename, num, stage
        return self

    def sources(self) -> list[Path]:
        return []

    def prepare(self) -> None:
        pass

    def bind(self, ctx: WorkerCtx) -> None:
        del ctx

    def schema_group(self, stream: str) -> GroupSchema | None:
        return self.schema.groups.get(stream) if self.schema is not None else None


def make_datamodule(out_dir: Path) -> SaltDataModule:
    """A streaming datamodule over `StreamStubReader`, 2 persistent forkserver workers."""
    return SaltDataModule(
        modules={
            "reader": StreamStubReader(),
            "feats": Features(variables={"event": ["uid", "val"]}),
        },
        train_file=str(out_dir / "unused"),
        val_file=str(out_dir / "unused"),
        batch_size=BATCH,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context="forkserver",
        pin_memory=False,
        iterable=True,
        block_rows=None,
        manifest={
            "train": str(out_dir / "train_manifest.json"),
            "val": str(out_dir / "val_manifest.json"),
        },
        sinks={Mode.FIT: ["inputs.event"], Mode.VAL: ["inputs.event"]},
        seed=42,
        shuffle_stream=False,
    )


def _batches(loader) -> list[list[int]]:  # noqa: ANN001
    """Iterate `loader` fully; per-batch uids (column 0 of ``inputs.event``, exact in float32)."""
    return [[int(u) for u in batch["inputs"]["event"][:, 0].tolist()] for batch in loader]


def child_main(rank: int, world_size: int, out_dir: str) -> None:
    """One rank: join the file-store gloo group, stream train/val, write ``rank{rank}.json``.

    Any failure writes its traceback to ``rank{rank}.error.txt`` and re-raises, so
    the process exits non-zero.
    """
    try:
        torch.set_num_threads(1)
        import torch.distributed as dist

        dist.init_process_group(
            "gloo",
            init_method=f"file://{out_dir}/pg",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        sub = Path(out_dir) / f"rank{rank}"
        sub.mkdir(parents=True, exist_ok=True)
        dm = make_datamodule(sub)
        # no prepare_data: each rank builds its own manifests under its own dir
        dm.setup("fit")
        train = dm.train_dataloader()
        val = dm.val_dataloader()
        result = {
            "train": {"len": len(train), "batches": _batches(train)},
            "val": {"len": len(val), "batches": _batches(val)},
        }
        del train, val  # shuts the persistent workers down before the barrier
        (Path(out_dir) / f"rank{rank}.json").write_text(json.dumps(result))
        dist.barrier()
        dist.destroy_process_group()
    except BaseException:
        (Path(out_dir) / f"rank{rank}.error.txt").write_text(traceback.format_exc())
        raise
