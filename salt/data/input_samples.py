"""`InputSamples` — the per-stage source-file and row-cap config holder the
datamodule reads each stage's reader source from.
"""

from __future__ import annotations

from pathlib import Path

from salt.data.base import SaltDatasetModule
from salt.graph.spec import IO, Mode

__all__ = ["InputSamples"]

SETUP_STAGES: tuple[str, ...] = ("train", "val", "test")


class InputSamples(SaltDatasetModule):
    """Per-stage source files and row caps for the datamodule's single reader.

    A config holder, not a graph node: the datamodule excludes it from the
    per-batch modules and reads each stage's ``(file, num)`` via `source`.

    Parameters
    ----------
    files : dict[str, str | Path]
        Per-stage source: ``{train, val, test}`` -> a literal path or a
        wildcard string (passed through verbatim — the reader resolves it).
        A stage left out is unconfigured (e.g. a fit-only config may omit
        ``test``). Replaces the datamodule's ``train_file``/``val_file``/
        ``test_file`` kwargs.
    num : dict[str, int] | None, optional
        Per-stage row cap (``-1`` = all), by default ``-1`` for every stage.

    Raises
    ------
    ValueError
        If `files` is empty, or names a stage outside ``{train, val, test}``.
    """

    def __init__(
        self,
        files: dict[str, str | Path],
        num: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        if not files:
            raise ValueError(
                "InputSamples requires a non-empty `files` map "
                "(e.g. {train: ..., val: ..., test: ...})"
            )
        bad = [stage for stage in files if stage not in SETUP_STAGES]
        if bad:
            raise ValueError(
                f"InputSamples `files` has unknown stage(s) {bad}: must be a subset of "
                f"{list(SETUP_STAGES)}"
            )
        self.files: dict[str, str | Path] = dict(files)
        self.num: dict[str, int] = dict(num) if num is not None else {}

    def declare_io(self, mode: Mode) -> IO:
        """Empty — a config holder produces no per-batch tensors."""
        del mode
        return IO()

    def source(self, stage: str) -> tuple[str | None, int]:
        """The ``(file, num)`` for `stage`: ``files[stage]`` as a string (None when
        the stage is unconfigured) and its row cap (``-1`` when unset).
        """
        f = self.files.get(stage)
        return (None if f is None else str(f)), int(self.num.get(stage, -1))
