"""`SaltDataset` — the map-style dataset that runs the compiled dataset plan;
one ``__getitem__`` = one full batch, ending at the numpy->torch boundary.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from salt.data.plan_runner import MODEL_VISIBLE_NAMESPACES, _PlanRunner

__all__ = ["MODEL_VISIBLE_NAMESPACES", "SaltDataset"]


class SaltDataset(_PlanRunner, Dataset):
    """Map-style dataset executing a compiled dataset plan per batch slice.

    Constructor parameters and their errors are documented on `_PlanRunner`;
    this class adds only the map-style addressing (`__len__`, `__getitem__`).
    """

    def __len__(self) -> int:
        """Return the number of rows served (delegates to the reader)."""
        return len(self._reader)

    def __getitem__(self, rows: slice) -> dict[str, Any]:
        """Run the dataset plan for one contiguous batch slice.

        `rows` must be a slice with start/stop set (the `RandomBatchSampler`
        contract); returns the model-visible nested dict with torch tensors
        at the leaves — ``raw.*`` never crosses the boundary.
        """
        if not isinstance(rows, slice) or rows.start is None or rows.stop is None:
            raise TypeError(
                f"SaltDataset is indexed by contiguous slices with start/stop, got {rows!r} "
                "(the RandomBatchSampler contract)"
            )
        self._maybe_bind()
        return self._run_plan(rows)
