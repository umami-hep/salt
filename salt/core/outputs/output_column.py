"""`OutputColumn` — one ``outputs.*`` leaf's declarative H5 column schema."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import KEY_SEP
from salt.core.outputs.collect_outputs import _OUTPUTS_NAMESPACE


@dataclass(frozen=True)
class OutputColumn:
    """One ``outputs.*`` leaf's declarative H5 column schema.

    The producer emits a bare ``outputs.<stream>.<name>`` tensor; the sink
    owns the serialisation schema the trace cannot recover — the per-column
    suffixes, the H5 dtype, and the run-name prefix — reproducing the v1
    ``task.output_names`` column-naming contract without the sink reaching
    into the task.

    Parameters
    ----------
    key : str
        The ``outputs.<stream>.<name>`` leaf this column declares. Concrete
        and under the ``outputs`` namespace.
    suffixes : Sequence[str]
        Per-channel logical suffixes the leaf's last dim expands into, in
        last-dim order (e.g. ``["pb", "pc", "pu"]``). The H5 column is
        ``{run_name}_{suffix}`` when `prefix` (default), else the bare
        `suffix`.
    dtype : str, optional
        The H5 column dtype (numpy descriptor), by default ``"f4"``.
    prefix : bool, optional
        Whether to prefix each suffix with ``{run_name}_``, by default True
        (False reproduces a bare-column family like the v1 ``VertexIndex``).

    Raises
    ------
    ConfigError
        For a non-``outputs`` key, a wildcard key, or an empty suffix list.
    """

    key: str
    suffixes: Sequence[str]
    dtype: str = "f4"
    prefix: bool = True

    def __post_init__(self) -> None:
        parts = self.key.split(KEY_SEP)
        if any(part in {"*", "**"} for part in parts):
            raise ConfigError(
                f"OutputColumn key {self.key!r} contains a wildcard — sink demand keys "
                "are concrete (design §2.2)"
            )
        if parts[0] != _OUTPUTS_NAMESPACE:
            raise ConfigError(
                f"OutputColumn key {self.key!r} is not under the {_OUTPUTS_NAMESPACE!r} "
                "namespace — sinks consume producer outputs.* leaves, not raw predictions "
                "(design §2)"
            )
        if not list(self.suffixes):
            raise ConfigError(
                f"OutputColumn {self.key!r} needs a non-empty suffix list — name the "
                "per-channel columns the leaf expands into (design §1 declarative table)"
            )

    @property
    def stream(self) -> str:
        """The leaf's stream (``outputs.<stream>.<name>`` middle segment)."""
        return self.key.split(KEY_SEP)[1]

    def column_names(self, run_name: str) -> list[str]:
        """The H5 column names for this leaf (run-name prefixed unless bare)."""
        return [f"{run_name}_{s}" if self.prefix else str(s) for s in self.suffixes]

    def np_dtype(self, run_name: str) -> np.dtype:
        """The structured numpy dtype this leaf contributes to its group."""
        return np.dtype([(col, self.dtype) for col in self.column_names(run_name)])
