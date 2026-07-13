"""`FtagLabeller` — on-the-fly ftag flavour relabelling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from ftag import Labeller

from salt.core.data.base import Processor
from salt.core.graph.errors import ConfigError
from salt.core.graph.planner import PlanStep
from salt.core.graph.spec import IO, Mode, TensorSpec, unflatten_spec


class FtagLabeller(Processor):
    """On-the-fly ftag flavour relabelling: ``raw.<stream> -> labels.<stream>.<label>``.

    The standalone counterpart of v1's implicitly-triggered labeller. v1
    gated the ftag ``Labeller`` (NOT v1-salt — ``ftag/labeller.py``)
    implicitly on ``input_name == global_object and label ==
    'flavour_label'``; v2 makes it an explicit, pluggable `Processor` wired
    as its own ``data.modules`` key — a behaviour is added by adding a
    module to YAML, not by flipping ``use_labeller=True`` on the `Labels`
    monolith. The GN3X boosted-Higgs classes are NOT a precomputed
    ``flavour_label`` column — they are derived from R10TruthLabel_R22v1 +
    the ghost-hadron counts at read time.

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

    With ``require_labels: true`` (v1 GN3X) the labeller RAISES on any
    object that matches no class; with ``require_labels: false`` (v1
    GN2X_qcdsplit) unmatched objects are dropped, so the derived label
    array may be SHORTER than the batch — exactly v1's
    ``self.labeller.get_labels(batch)`` behaviour. This is faithful v1
    parity AT THE LABEL-DERIVATION LEVEL: v1 likewise does NOT row-filter
    the inputs/other-label columns to the dropped subset, so the v1 bundle
    is itself length-mismatched (inputs vs flavour_label) under
    ``require_labels=False``. The full-batch length-coherence
    reconciliation is NOT settled by this split — a data-bearing
    GN2X_qcdsplit forward/integration test must decide whether to
    row-filter the whole bundle or to confirm v1's length-mismatch is
    handled identically.

    Parameters
    ----------
    stream : str, optional
        The stream the labeller relabels (v1's implicit ``global_object``;
        default ``"jets"``). The labeller's cut variables are demanded from
        ``raw.<stream>``.
    label : str, optional
        The label key this module produces (v1's implicit
        ``"flavour_label"``; default ``"flavour_label"``). Only
        ``labels.<stream>.<label>`` is derived; every other label is read
        from the file by `Labels`.
    class_names : Sequence[str] | None, optional
        Target ftag flavour class names, in label-index order (v1
        ``LabellerConfig.class_names``; GN3X 9-class, GN2X_qcdsplit
        7-class). REQUIRED (the empty-class guard fires when empty).
    require_labels : bool, optional
        Whether every object must be labelled (v1
        ``LabellerConfig.require_labels``): True raises on an unlabelled
        object, False drops it. By default True.
    dtype_policy : Literal["int64-for-int", "file"], optional
        ``int64-for-int`` (v1, default) casts integer labels to int64;
        ``file`` keeps the on-disk dtype.

    Raises
    ------
    ConfigError
        On an unknown ``dtype_policy``, or ``class_names`` empty (the
        empty-class guard). NOTE the deliberate exception-TYPE promotion:
        v1's empty-class guard raises a bare ``ValueError``; v2
        standardises structural config-validation failures on
        ``ConfigError`` (the framework's named config-error type, NOT a
        ``ValueError`` subclass), as every processor does. The other two
        labeller guards (missing-field, require_labels-on-unlabelled) keep
        v1's ``ValueError`` (raised at ``process`` time, not construction).
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
        # mirror v1 LabellerConfig: the empty-class guard MUST fire when
        # class_names is empty.
        if not class_names:
            raise ConfigError(
                f"FtagLabeller module {self.name!r}: class_names is empty — specify the "
                "target classes for relabelling (v1 LabellerConfig empty-class guard, "
                "configs.py:189)"
            )
        self.class_names = tuple(class_names)
        # the ftag Labeller IS the parity reference (ftag/labeller.py); v1
        # builds it identically (Labeller(class_names, require_labels))
        self.labeller = Labeller(list(self.class_names), self.require_labels)

    def _labeller_variables(self) -> tuple[str, ...]:
        """The labeller's cut variables, de-duplicated, in first-seen order.

        ``Labeller.variables`` is a flat sum over the per-class cut
        variables, so it carries duplicates — dedupe before declaring them
        as read fields.
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
        """Demand the labeller's cut variables from ``raw.<stream>``.

        The derived label is NOT on disk (it has no on-disk producer), so
        instead the labeller's cut variables (``Labeller.variables``,
        deduped) are demanded from ``raw.<stream>`` so they enter the
        per-mode demand-narrowed read set.
        """
        del step
        out: dict[str, dict[str, str]] = {}
        for var in self._labeller_variables():
            out.setdefault(self.stream, {}).setdefault(var, self.name)
        return out

    def process(self, batch, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Derive the on-the-fly labeller labels from the raw structured array.

        Mirrors v1 ``process_labels``: the missing-field guard (every
        ``Labeller.variables`` cut variable must be present) then
        ``Labeller.get_labels`` on the WHOLE structured array, cast to
        int64 under ``dtype_policy``. The ``get_labels`` output is a fresh
        array, so it never aliases the reader buffer.

        Returns
        -------
        dict[str, np.ndarray]
            ``{labels.<stream>.<label>: derived array}`` — int64 (or file
            dtype under ``dtype_policy='file'``). With
            ``require_labels=False`` unmatched objects are dropped, so the
            array may be shorter than the batch (v1 parity).

        Raises
        ------
        ValueError
            If a labeller cut variable is absent from the raw stream, or —
            under ``require_labels`` — if any object matches no class.
        """
        del rows, mode
        raw = batch.get(f"raw.{self.stream}")
        present = set(raw.dtype.names or ())
        missing = [var for var in self._labeller_variables() if var not in present]
        if missing:
            raise ValueError(
                f"FtagLabeller module {self.name!r}: not enough fields to apply labelling cuts on "
                f"stream {self.stream!r} — missing labeller variables {missing} (v1 field-subset "
                "check, datasets.py:622-624)"
            )
        # get_labels raises under require_labels on an unlabelled object and
        # otherwise drops it — v1 parity.
        derived = self.labeller.get_labels(raw)
        if self.dtype_policy == "int64-for-int" and np.issubdtype(derived.dtype, np.integer):
            out = derived.astype(np.int64)
        else:
            out = np.array(derived, copy=True)
        return {f"labels.{self.stream}.{self.label}": out}
