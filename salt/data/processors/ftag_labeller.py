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

    With row selection (``select_field`` + ``select_value``, e.g. a ``jet_type``
    column in a mixed small-R/large-R batch) the output ALWAYS has the batch
    length: selected rows are labelled (an unlabelled SELECTED row still raises)
    and unselected rows carry ``-1``, the classification loss's ignore label.
    Selection therefore requires ``require_labels=True``: dropping rows would break
    batch alignment with the other type's rows.

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
    select_field : str | None, optional
        A field of ``raw.<stream>`` restricting which objects this module
        labels, by default None (every object). Only objects whose
        `select_field` equals `select_value` are labelled, and
        ``require_labels`` applies to those alone; every other object gets
        ``-1``, the label a classification loss ignores. This is how a stream
        mixing object types (e.g. small-R and large-R jets told apart by
        ``jet_type``) carries one label per type.
    select_value : int | None, optional
        The `select_field` value of the objects to label. Set together with
        `select_field`.

    Raises
    ------
    ConfigError
        On an unknown ``dtype_policy``, ``class_names`` empty (the empty-class
        guard), only one of ``select_field``/``select_value`` given, or row
        selection with ``require_labels=False``.
    """

    def __init__(
        self,
        stream: str = "jets",
        label: str = "flavour_label",
        class_names: Sequence[str] | None = None,
        require_labels: bool = True,
        dtype_policy: Literal["int64-for-int", "file"] = "int64-for-int",
        select_field: str | None = None,
        select_value: int | None = None,
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
        if (select_field is None) != (select_value is None):
            raise ConfigError(
                f"FtagLabeller module {self.name!r}: select_field and select_value are set "
                f"together (got select_field={select_field!r}, "
                f"select_value={select_value!r})"
            )
        if select_field is not None and not self.require_labels:
            raise ConfigError(
                f"FtagLabeller module {self.name!r}: row selection "
                f"({select_field!r} == {select_value!r}) requires require_labels=True — "
                "dropping unlabelled rows would break batch alignment with the other type's rows"
            )
        self.select_field: str | None = None if select_field is None else str(select_field)
        self.select_value: int | None = None if select_value is None else int(select_value)

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
        """Demand the labeller's deduped cut variables (plus any ``select_field``) from
        ``raw.<stream>`` — the derived label itself is not on disk.
        """
        del step
        out: dict[str, dict[str, str]] = {}
        for var in self._labeller_variables():
            out.setdefault(self.stream, {}).setdefault(var, self.name)
        if self.select_field is not None:
            out.setdefault(self.stream, {}).setdefault(self.select_field, self.name)
        return out

    def _out_dtype(self, derived: np.ndarray) -> np.dtype:
        """int64 for integer labels under ``int64-for-int``, else the derived dtype."""
        if self.dtype_policy == "int64-for-int" and np.issubdtype(derived.dtype, np.integer):
            return np.dtype(np.int64)
        return derived.dtype

    def process(self, batch, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Derive labels from the raw structured array via `Labeller.get_labels` (after
        checking cut variables are present), cast to int64 under ``dtype_policy``; with
        ``require_labels=False`` unmatched objects are dropped, so the output may be
        shorter than the batch. With row selection the output has the batch length.
        """
        del rows, mode
        label_key = f"labels.{self.stream}.{self.label}"
        raw = batch.get(f"raw.{self.stream}")
        present = set(raw.dtype.names or ())
        missing = [var for var in self._labeller_variables() if var not in present]
        if missing:
            raise ValueError(
                f"FtagLabeller module {self.name!r}: not enough fields to apply labelling cuts on "
                f"stream {self.stream!r} — missing labeller variables {missing}"
            )
        if self.select_field is None:
            # get_labels raises under require_labels on an unlabelled object,
            # otherwise drops it
            derived = self.labeller.get_labels(raw)
            return {label_key: derived.astype(self._out_dtype(derived))}
        if self.select_field not in present:
            raise ValueError(
                f"FtagLabeller module {self.name!r}: not enough fields to apply row selection on "
                f"stream {self.stream!r} — missing select_field {[self.select_field]}"
            )
        sel = np.asarray(raw[self.select_field] == self.select_value, dtype=bool)
        if not sel.any():
            # skip the labeller on an empty selection; every row is ignored
            dtype = np.dtype(np.int64) if self.dtype_policy == "int64-for-int" else np.dtype(int)
            return {label_key: np.full(len(raw), -1, dtype=dtype)}
        # require_labels is True here, so an unlabelled SELECTED row raises
        derived = self.labeller.get_labels(raw[sel])
        out = np.full(len(raw), -1, dtype=self._out_dtype(derived))
        out[sel] = derived
        return {label_key: out}
