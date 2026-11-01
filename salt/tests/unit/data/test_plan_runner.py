"""Tests for `salt.data.plan_runner` — the per-worker first-batch contract guard."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from salt.data.base import Processor, Reader
from salt.data.dataset import SaltDataset
from salt.graph.errors import MutationError, UndeclaredAccessError
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.schema import GroupSchema, Schema

N, T = 8, 3


class ArrayReader(Reader):
    """In-memory reader: a ``jets`` sequence stream and a scalar ``event`` stream."""

    def __init__(self) -> None:
        super().__init__()
        rng = np.random.default_rng(0)
        self._jets = np.zeros((N, T), dtype=[("pt", "f4")])
        self._jets["pt"] = rng.uniform(1.0, 2.0, size=(N, T))
        self._event = np.zeros((N,), dtype=[("val", "f4")])
        self._event["val"] = rng.uniform(size=N)
        self.schema = Schema(
            groups={
                "jets": GroupSchema(fields={"pt": "float32"}),
                "event": GroupSchema(fields={"val": "float32"}),
            }
        )

    @property
    def streams(self) -> tuple[str, ...]:
        return ("jets", "event")

    def declare_io(self, mode: Mode) -> IO:
        del mode
        return IO(
            produces=unflatten_spec({
                "raw.jets": TensorSpec(kind="data", fields=("pt",)),
                "raw.event": TensorSpec(kind="data", fields=("val",)),
            })
        )

    def __len__(self) -> int:
        return N

    def schema_group(self, stream: str) -> GroupSchema | None:
        return self.schema.groups.get(stream)

    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        del mode
        # views of the reader's own arrays: the "reusable buffer" of a real reader
        return {"raw.jets": self._jets[rows], "raw.event": self._event[rows]}

    def aliases(self, array: np.ndarray) -> bool:
        return bool(np.shares_memory(array, self._jets))


class ToyFeatures(Processor):
    """Declares ``raw.jets`` -> ``inputs.jets``; can snoop ``raw.event`` or alias the buffer."""

    def __init__(self, snoop: bool = False, alias: bool = False) -> None:
        super().__init__()
        self.snoop = snoop
        self.alias = alias

    def declare_io(self, mode: Mode) -> IO:
        del mode
        return IO(
            requires=unflatten_spec({"raw.jets": TensorSpec(kind="data", fields=("pt",))}),
            produces=unflatten_spec({"inputs.jets": TensorSpec(dtype="float32", kind="data")}),
        )

    def process(self, batch, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        del rows, mode
        pt = batch.get("raw.jets")["pt"]
        if self.snoop:
            batch.get("raw.event")  # undeclared read
        return {"inputs.jets": pt if self.alias else pt.copy()}


def dataset(debug: bool = False, **toy) -> SaltDataset:
    modules = {"reader": ArrayReader(), "toy": ToyFeatures(**toy)}
    return SaltDataset(modules, Mode.FIT, sinks=["inputs.jets"], debug=debug)


SNOOP = r"\[mode=FIT\] module 'toy' read bundle key 'raw.event'"


class TestWorkerFirstBatchGuard:
    def test_first_batch_guarded_second_not(self):
        ds = dataset(snoop=True)
        with pytest.raises(UndeclaredAccessError, match=SNOOP):
            ds[0:4]
        out = ds[0:4]  # same process: already bound, the guard was consumed
        assert out["inputs"]["jets"].shape == (4, T)

    def test_a_new_process_rebinds_and_rearms(self):
        ds = dataset(snoop=True)
        with pytest.raises(UndeclaredAccessError, match=SNOOP):
            ds[0:4]
        ds[0:4]
        ds._bound_pid = -1  # noqa: SLF001 - what a forked dataloader worker sees: another pid
        with pytest.raises(UndeclaredAccessError, match=SNOOP):
            ds[4:8]
        ds[4:8]

    def test_an_unpickled_worker_copy_is_guarded(self):
        ds = dataset(snoop=True)
        with pytest.raises(UndeclaredAccessError, match=SNOOP):
            ds[0:4]
        worker_copy = pickle.loads(pickle.dumps(ds))  # the spawn/forkserver worker path
        with pytest.raises(UndeclaredAccessError, match=SNOOP):
            worker_copy[0:4]

    def test_debug_guards_every_batch(self):
        ds = dataset(debug=True, snoop=True)
        for _ in range(3):
            with pytest.raises(UndeclaredAccessError, match=SNOOP):
                ds[0:4]

    def test_declared_reads_pass_guarded_and_unguarded(self):
        for debug in (False, True):
            ds = dataset(debug=debug)
            first, second = ds[0:4], ds[0:4]
            np.testing.assert_array_equal(
                first["inputs"]["jets"].numpy(), second["inputs"]["jets"].numpy()
            )

    def test_aliasing_assertion_fires_under_debug_only(self):
        with pytest.raises(MutationError, match="'inputs.jets' aliases a reusable reader buffer"):
            dataset(debug=True, alias=True)[0:4]
        plain = dataset(alias=True)  # the first-batch guard is read tracking only
        plain[0:4]
        plain[0:4]
