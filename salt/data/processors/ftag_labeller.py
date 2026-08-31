"""`FtagLabeller` — on-the-fly ftag flavour relabelling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from ftag import Labeller

from salt.data.base import Processor
from salt.graph.errors import ConfigError
from salt.graph.planner import PlanStep
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec


class FtagLabeller(Processor):
    """On-the-fly ftag flavour relabelling: ``raw.<stream> -> labels.<stream>.<label>``.

    An explicit, pluggable `Processor` wired as its own ``data.modules`` key
    — a behaviour is added by adding a module to YAML. The GN3X boosted-Higgs
    classes are not a precomputed ``flavour_label`` column; they are derived
    from R10TruthLabel_R22v1 + the ghost-hadron counts at read time.

    The derived label is produced CONCRETELY (``labels.<stream>.<label>``,
    never a wildcard); a concrete produce beats the `Labels` ``labels.**``
    wildcard via the planner's concrete-beats-wildcard rule — the SAME
    mechanism ``MultiTarget.target`` relies on. Write-once is preserved:
    this module becomes the SOLE producer of that key, and `Labels` no
    longer serves it.

    The labeller's cut variables (``Labeller.variables``, deduped) are
    declared as the read fields for the produced key so they enter the
    per-mode demand-narrowed read set; the derived label field itself is
    NOT read from disk (it has no on-disk producer). Derivation runs
    post-`Reader.read`, pre-torch: int -> int64 under ``dtype_policy``.

    With ``require_labels: true`` the labeller RAISES on any object that
    matches no class; with ``require_labels: false`` unmatched objects are
    dropped, so the derived label array may be SHORTER than the batch. The
    full-batch length-coherence with inputs/other labels under
    ``require_labels=False`` is not resolved by this module.

    Parameters
    ----------
    stream : str, optional
        The stream the labeller relabels, by default ``"jets"``. The
        labeller's cut variables are demanded from ``raw.<stream>``.
    label : str, optional
        The label key this module produces, by default ``"flavour_label"``.
        Only ``labels.<stream>.<label>`` is derived; every other label is
        read from the file by `Labels`.
    class_names : Sequence[str] | None, optional
        Target ftag flavour class names, in label-index order. REQUIRED
        (the empty-class guard fires when empty).
    require_labels : bool, optional
        Whether every object must be labelled: True raises on an unlabelled
        object, False drops it. By default True.
    dtype_policy : Literal["int64-for-int", "file"], optional
        ``int64-for-int`` (default) casts integer labels to int64; ``file``
        keeps the on-disk dtype.

    Raises
    ------
    ConfigError
        On an unknown ``dtype_policy``, or ``class_names`` empty (the
        empty-class guard).
    """

    def __init__(
        self,
        stream: str = "jets",
        label: str = "flavour_label",
        class_names: Sequence[str] | None = None,
        require_labels: bool = True,
        dtype_policy: Literal["int64-for-int", "file"] = "int64-for-int",
    ) -> None:
        super().__init__()
        if dtype_policy not in {"int64-for-int", "file"}:
            raise ConfigError(
                f"unknown dtype_policy {dtype_policy!r}: expected 'int64-for-int' or 'file'"
            )
        self.stream = str(stream)
        self.label = str(label)
        self.dtype_policy = dtype_policy
        self.require_labels = bool(require_labels)
        if not class_names:
            raise ConfigError(
                f"FtagLabeller module {self.name!r}: class_names is empty — specify the "
                "target classes for relabelling"
            )
        self.class_names = tuple(class_names)
        self.labeller = Labeller(list(self.class_names), self.require_labels)

    def _labeller_variables(self) -> tuple[str, ...]:
        """The labeller's cut variables, de-duplicated, in first-seen order
        (``Labeller.variables`` is a flat sum over per-class cuts, so it carries
        duplicates).
        """
        return tuple(dict.fromkeys(self.labeller.variables))

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``raw.<stream>`` require -> CONCRETE ``labels.<stream>.<label>`` produce.

        The produce is concrete (never a wildcard, never schema-narrowed)
        so it beats the `Labels` ``labels.**`` wildcard via the planner's
        concrete-beats-wildcard rule.
        """
        del mode
        requires = {f"raw.{self.stream}": TensorSpec(kind="data")}
        produces = {f"labels.{self.stream}.{self.label}": TensorSpec(kind="label")}
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def read_fields(self, step: PlanStep) -> dict[str, dict[str, str]]:
        """Demand the labeller's deduped cut variables from ``raw.<stream>`` — the derived
        label itself is not on disk, so it has no direct read demand.
        """
        del step
        out: dict[str, dict[str, str]] = {}
        for var in self._labeller_variables():
            out.setdefault(self.stream, {}).setdefault(var, self.name)
        return out

    def process(self, batch, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Derive labels from the raw structured array via `Labeller.get_labels` (after
        checking cut variables are present), cast to int64 under ``dtype_policy``; with
        ``require_labels=False`` unmatched objects are dropped, so the output may be
        shorter than the batch.
        """
        del rows, mode
        raw = batch.get(f"raw.{self.stream}")
        present = set(raw.dtype.names or ())
        missing = [var for var in self._labeller_variables() if var not in present]
        if missing:
            raise ValueError(
                f"FtagLabeller module {self.name!r}: not enough fields to apply labelling cuts on "
                f"stream {self.stream!r} — missing labeller variables {missing}"
            )
        # get_labels raises under require_labels on an unlabelled object,
        # otherwise drops it
        derived = self.labeller.get_labels(raw)
        if self.dtype_policy == "int64-for-int" and np.issubdtype(derived.dtype, np.integer):
            out = derived.astype(np.int64)
        else:
            out = np.array(derived, copy=True)
        return {f"labels.{self.stream}.{self.label}": out}
