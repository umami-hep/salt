"""Shipped dataset processors: `Features`, `Labels`, `MultiTarget`,
`MaskFormerTargets` (design §6.2).

Each replaces an if-branch of the v1 ``SaltDataset.__getitem__`` god-loop
(``datasets.py:417-559``). `MultiTarget` (M5 sub-wave A3) ports v1's
conditional target replacement (``datasets.py:237-248,648,695-739``).
`MaskFormerTargets` (M5 sub-wave C) ports v1's object-target construction
(``datasets.py:549-553,636-644``, FD 1090-1110). Further v1 branches
(``Parameters``) land with their workloads (TODO(M6) per design §9.5) — the
structure here is the template.
"""

from __future__ import annotations

import operator
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from ftag import Labeller
from numpy.lib.recfunctions import structured_to_unstructured as s2u

from salt.core.data.base import Processor, WorkerCtx
from salt.core.graph.errors import ConfigError, SchemaError
from salt.core.graph.planner import PlanStep
from salt.core.graph.spec import IO, KEY_SEP, Mode, TensorSpec, sym_dim, unflatten_spec

__all__ = ["Features", "FtagLabeller", "Labels", "MaskFormerTargets", "MultiTarget"]

# v1 OPERATORS (datasets.py:29-36) — the conditional-replacement comparators,
# applied to the selection label against the configured value.
_OPERATORS: dict[str, Callable[[Any, Any], Any]] = {
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}


class Features(Processor):
    """``raw.* -> inputs.*`` float32 materialisation (design §6.2).

    THE documented one-copy-per-batch aliasing boundary (design §2.4): the
    ``structured_to_unstructured`` conversion is the mandatory copy that
    separates reusable reader buffers from anything handed to the trainer —
    enforced here with an explicit ``may_share_memory`` guard instead of
    v1's implicit-and-undocumented reliance on ``s2u``+``maybe_copy``
    (``datasets.py:511,515``, pain point in the data-pipeline map).

    Column order = the configured list order — the ONE place column order is
    defined (design §5.1); the produced specs carry ``fields`` metadata so
    downstream column lookups resolve by name (design §2.2). v1 semantics
    kept in order: ``s2u`` -> ``nan_to_num`` (optional) -> zero padded rows
    via the pad mask -> finite check (``datasets.py:504-537``).

    Parameters
    ----------
    variables : Mapping[str, Sequence[str]]
        Stream name -> ordered input variable list.
    non_finite_to_num : bool, optional
        Convert NaN/inf to zero before masking (``datasets.py:513-514``).
    ignore_finite_checks : bool, optional
        Warn instead of raising on non-finite inputs (``datasets.py:527-537``).

    Raises
    ------
    ConfigError
        On an empty or duplicate-containing variable list.
    """

    def __init__(
        self,
        variables: Mapping[str, Sequence[str]],
        non_finite_to_num: bool = False,
        ignore_finite_checks: bool = False,
    ) -> None:
        super().__init__()
        if not variables:
            raise ConfigError("Features needs at least one stream in 'variables' (design §6.2)")
        self.variables: dict[str, list[str]] = {}
        for stream, names in variables.items():
            names = list(names)  # noqa: PLW2901
            if not names:
                raise ConfigError(f"Features stream {stream!r} has an empty variable list")
            if len(set(names)) != len(names):
                raise ConfigError(f"Features stream {stream!r} has duplicate variables: {names}")
            self.variables[stream] = names
        self.non_finite_to_num = non_finite_to_num
        self.ignore_finite_checks = ignore_finite_checks

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``raw.<s> (+ optional masks.<s>) -> inputs.<s>`` per stream.

        The mask require is optional: ``global_object`` streams have no mask
        producer and the planner drops the port (design §2.2).

        Returns
        -------
        IO
            The declared interface.
        """
        del mode
        requires: dict[str, TensorSpec] = {}
        produces: dict[str, TensorSpec] = {}
        for stream, names in self.variables.items():
            fields = tuple(names)
            requires[f"raw.{stream}"] = TensorSpec(kind="data", fields=fields)
            requires[f"masks.{stream}"] = TensorSpec(dtype="bool", kind="pad_mask", optional=True)
            produces[f"inputs.{stream}"] = TensorSpec(dtype="float32", kind="data", fields=fields)
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def process(self, batch, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Materialise float32 input arrays from the structured raws.

        Returns
        -------
        dict[str, np.ndarray]
            ``{"inputs.<stream>": [B, F] / [B, T, F] float32}`` — fresh
            arrays, guaranteed not to alias the reader buffers.

        Raises
        ------
        ValueError
            On non-finite inputs (unless ``ignore_finite_checks``), exactly
            as v1 (``datasets.py:535-537``).
        """
        del rows, mode
        out: dict[str, np.ndarray] = {}
        for stream, names in self.variables.items():
            raw = batch.get(f"raw.{stream}")
            # column order = config list order (structured multi-field
            # indexing reorders, datasets.py:506-511)
            flat = s2u(raw[names], dtype=np.float32)
            if np.may_share_memory(flat, raw):
                # the mandatory copy (contract #9): s2u may return a view for
                # uniform layouts — never hand a buffer alias downstream
                flat = flat.copy()
            if self.non_finite_to_num:
                flat = np.nan_to_num(flat, posinf=0, neginf=0)
            mask_key = f"masks.{stream}"
            if mask_key in batch:
                flat[batch.get(mask_key)] = 0.0  # zero padded rows (datasets.py:524)
            if not np.isfinite(flat).all():
                if self.ignore_finite_checks:
                    warnings.warn(
                        f"Non-finite inputs for {stream!r}. But ignore finite flag is on, "
                        "make sure this is intentional.",
                        stacklevel=2,
                    )
                else:
                    raise ValueError(f"Non-finite inputs for {stream!r}.")
            out[f"inputs.{stream}"] = flat
        return out


class Labels(Processor):
    """Demand-driven label producer over ``labels.**`` (design §3.3, §6.2).

    Declares the wildcard pattern ``labels.**`` (framework producer); the
    planner narrows it to the keys concretely demanded per mode and validates
    every narrowed key against the dataset schema (design §2.2 rules (a)-(d))
    — this keeps "tasks are the source of truth for which labels get loaded"
    (``cli.py:337-392``) without the CLI mutation hook (design §6.5). The
    narrowed key set is learned at bind time from the module's own plan step.

    v1 dtype semantics kept (``datasets.py:629-634``): integer labels become
    int64 (``dtype_policy: int64-for-int``), everything else keeps the file
    dtype; sentinel values (-1 padding, -2/-3 type codes) pass through
    untouched. The v1 ``ftagTruthOriginLabel`` malformed-recovery becomes the
    opt-in ``valid_ranges`` config with explicit ranges — no longer
    heuristically triggered (``datasets.py:631-632, 779-836``; design §6.2).

    In a mode where no label is demanded the module narrows to nothing and
    runs as a no-op (the kernel keeps wildcard producers with bound requires
    alive as terminal consumers); it contributes no read fields, so no I/O
    is wasted.

    On-the-fly ftag relabelling is NOT this module's job (M8 sub-wave 1): the
    ftag ``Labeller`` derived-label producer lives in the standalone
    `FtagLabeller` processor, wired as its own ``data.modules`` key. Its
    concrete ``labels.<stream>.<label>`` produce beats this module's
    ``labels.**`` wildcard (planner concrete-beats-wildcard rule), so `Labels`
    serves every OTHER demanded label from disk and `FtagLabeller` owns the
    relabelled one. `Labels` is pure disk-label extraction.

    Parameters
    ----------
    streams : Sequence[str] | None, optional
        The streams this producer can serve (its ``raw.<stream>`` requires).
        None (the config default) defers to the framework: `GraphDataset`
        calls `bind_streams` with the reader's stream list before plan
        compilation (config-derived, static).
    dtype_policy : Literal["int64-for-int", "file"], optional
        ``int64-for-int`` (v1, default) casts integer labels to int64;
        ``file`` keeps the on-disk dtype for everything.
    valid_ranges : Mapping[str, Sequence[int]] | None, optional
        Label name -> inclusive ``[lo, hi]`` valid range (e.g.
        ``{ftagTruthOriginLabel: [-1, 7]}``). Out-of-range values raise, or
        are mapped to -1 when ``recover_malformed`` is set.
    recover_malformed : bool, optional
        Recover (warn + map to -1) instead of raising on out-of-range label
        values, by default False.

    Raises
    ------
    ConfigError
        On an unknown ``dtype_policy`` or malformed ``valid_ranges``.
    """

    allow_wildcards: ClassVar[bool] = True  # framework wildcard capability (design §2.2)

    def __init__(
        self,
        streams: Sequence[str] | None = None,
        dtype_policy: Literal["int64-for-int", "file"] = "int64-for-int",
        valid_ranges: Mapping[str, Sequence[int]] | None = None,
        recover_malformed: bool = False,
    ) -> None:
        super().__init__()
        if dtype_policy not in {"int64-for-int", "file"}:
            raise ConfigError(
                f"unknown dtype_policy {dtype_policy!r}: expected 'int64-for-int' or 'file'"
            )
        self._streams: tuple[str, ...] | None = tuple(streams) if streams is not None else None
        self.dtype_policy = dtype_policy
        self.valid_ranges: dict[str, tuple[float, float]] = {}
        for label, bounds in (valid_ranges or {}).items():
            bounds = tuple(bounds)  # noqa: PLW2901
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ConfigError(
                    f"valid_ranges[{label!r}] must be an inclusive [lo, hi] pair, got {bounds}"
                )
            self.valid_ranges[label] = bounds
        self.recover_malformed = recover_malformed
        self._targets: tuple[tuple[str, str, str], ...] | None = None

    def bind_streams(self, streams: Sequence[str]) -> None:
        """Framework hook: adopt the reader's stream list when ``streams`` is unset.

        Called by `GraphDataset` before plan compilation (config-derived,
        static — no file I/O). A no-op when streams were configured
        explicitly.
        """
        if self._streams is None:
            self._streams = tuple(streams)

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``raw.<s>`` requires plus the ``labels.**`` wildcard produce.

        Returns
        -------
        IO
            The declared interface.

        Raises
        ------
        ConfigError
            If the served streams are still unresolved (set ``streams`` or
            compile through `GraphDataset`).
        """
        del mode
        if self._streams is None:
            raise ConfigError(
                f"Labels module {self.name!r} has no streams — set streams: explicitly or "
                "compile via GraphDataset (which forwards the reader's streams)"
            )
        requires = {f"raw.{stream}": TensorSpec(kind="data") for stream in self._streams}
        produces = {"labels.**": TensorSpec(kind="label")}
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def _parse_targets(self, step: PlanStep) -> tuple[tuple[str, str, str], ...]:
        """Parse the narrowed produces into ``(key, stream, label)`` triples.

        Returns
        -------
        tuple[tuple[str, str, str], ...]
            One triple per narrowed ``labels.<stream>.<label>`` key.

        Raises
        ------
        ConfigError
            On a narrowed key not of the ``labels.<stream>.<label>`` form,
            or a stream outside the served set.
        """
        targets: list[tuple[str, str, str]] = []
        for key in step.produces:
            parts = key.split(KEY_SEP)
            if len(parts) != 3 or parts[0] != "labels":
                raise ConfigError(
                    f"Labels module {self.name!r}: narrowed key {key!r} is not of the form "
                    "labels.<stream>.<label> (design §3.3)"
                )
            _, stream, label = parts
            if self._streams is not None and stream not in self._streams:
                raise ConfigError(
                    f"Labels module {self.name!r}: key {key!r} names stream {stream!r} "
                    f"outside its served streams {list(self._streams)}"
                )
            targets.append((key, stream, label))
        return tuple(targets)

    def bind(self, ctx: WorkerCtx) -> None:
        """Learn the narrowed key set from this module's own plan step."""
        assert ctx.step is not None
        self._targets = self._parse_targets(ctx.step)

    def read_fields(self, step: PlanStep) -> dict[str, dict[str, str]]:
        """Demand exactly the narrowed label fields from the reader (design §6.1).

        The demanded field IS the label name — each narrowed
        ``labels.<stream>.<label>`` maps to a ``raw.<stream>.<label>`` read.

        Returns
        -------
        dict[str, dict[str, str]]
            ``{stream: {field: this module}}`` from the narrowed produces —
            the wildcard-producer override of the default requires-fields rule.
        """
        out: dict[str, dict[str, str]] = {}
        for _key, stream, label in self._parse_targets(step):
            out.setdefault(stream, {})[label] = self.name
        return out

    def process(self, batch, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Extract, range-check and cast the demanded labels.

        Returns
        -------
        dict[str, np.ndarray]
            ``{narrowed_key: fresh array}`` — exactly the narrowed key set
            (`Bundle.merge` enforces it); copies never alias the reader
            buffers (field extraction + ``astype``/``copy``).

        Raises
        ------
        ValueError
            On out-of-range values for a ``valid_ranges`` label when
            ``recover_malformed`` is off (v1 ``datasets.py:832-836``).
        """
        del rows, mode
        assert self._targets is not None, "Labels.process called before bind()"
        out: dict[str, np.ndarray] = {}
        for key, stream, label in self._targets:
            values = batch.get(f"raw.{stream}")[label]
            if (bounds := self.valid_ranges.get(label)) is not None:
                bad = (values < bounds[0]) | (values > bounds[1])
                if bad.any():
                    if not self.recover_malformed:
                        raise ValueError(
                            f"Malformed {label!r} values outside [{bounds[0]}, {bounds[1]}] "
                            f"with values and counts: {np.unique(values, return_counts=True)} "
                            "Recover flag is off, failing."
                        )
                    warnings.warn(
                        f"Malformed {label!r} values outside [{bounds[0]}, {bounds[1]}] "
                        f"with values and counts: {np.unique(values, return_counts=True)} "
                        "Recover flag is on, converting to invalid and continuing.",
                        stacklevel=2,
                    )
                    values = np.where(bad, -1, values)
            if self.dtype_policy == "int64-for-int" and np.issubdtype(values.dtype, np.integer):
                out[key] = values.astype(np.int64)  # always copies (datasets.py:633-634)
            else:
                out[key] = np.array(values, copy=True)  # keep file dtype (possibly f2)
        return out


class FtagLabeller(Processor):
    """On-the-fly ftag flavour relabelling: ``raw.<stream> -> labels.<stream>.<label>``.

    The standalone counterpart of v1's implicitly-triggered labeller (M8 sub-wave 1;
    design §6.2, FD 1303-1304). v1 gated the ftag ``Labeller`` (NOT v1-salt —
    ``ftag/labeller.py``) implicitly on ``input_name == global_object and
    label == 'flavour_label'`` (``datasets.py:609-626``); v2 makes it an explicit,
    pluggable `Processor` wired as its own ``data.modules`` key — a behaviour is
    added by adding a module to YAML, not by flipping ``use_labeller=True`` on the
    `Labels` monolith. The GN3X boosted-Higgs classes are NOT a precomputed
    ``flavour_label`` column — they are derived from R10TruthLabel_R22v1 + the
    ghost-hadron counts at read time.

    The derived label is produced CONCRETELY (``labels.<stream>.<label>``, never a
    wildcard); a concrete produce beats the `Labels` ``labels.**`` wildcard via the
    planner's concrete-beats-wildcard rule — the SAME mechanism ``MultiTarget.target``
    relies on (planner rule (a)). Write-once is preserved: this module becomes the
    SOLE producer of that key, and `Labels` no longer serves it.

    The labeller's cut variables (``Labeller.variables``, deduped) are declared as
    the read fields for the produced key (FD §6.1 1280-1282) so they enter the
    per-mode demand-narrowed read set; the derived label field itself is NOT read
    from disk (it has no on-disk producer). Derivation runs post-`Reader.read`,
    pre-torch (FD §2.4): int -> int64 under ``dtype_policy``.

    With ``require_labels: true`` (v1 GN3X) the labeller RAISES (ftag
    ``labeller.py:70``) on any object that matches no class; with
    ``require_labels: false`` (v1 GN2X_qcdsplit) unmatched objects are dropped
    (``labeller.py:73``), so the derived label array may be SHORTER than the batch —
    exactly v1's ``self.labeller.get_labels(batch)`` behaviour (``datasets.py:626``).
    This is faithful v1 parity AT THE LABEL-DERIVATION LEVEL (LB1's scope): v1
    likewise does NOT row-filter the inputs/other-label columns to the dropped
    subset (``datasets.py`` ``process_labels`` has no such filter), so the v1 bundle
    is itself length-mismatched (inputs vs flavour_label) under
    ``require_labels=False``. The full-batch length-coherence reconciliation is NOT
    settled by this split (it is a deferred M7 data-path question, independent of
    this extraction) — a data-bearing GN2X_qcdsplit forward/integration test must
    decide whether to row-filter the whole bundle or to confirm v1's length-mismatch
    is handled identically, and record it as a Key Decision.

    Parameters
    ----------
    stream : str, optional
        The stream the labeller relabels (v1's implicit ``global_object``; default
        ``"jets"``). The labeller's cut variables are demanded from ``raw.<stream>``.
    label : str, optional
        The label key this module produces (v1's implicit ``"flavour_label"``;
        default ``"flavour_label"``). Only ``labels.<stream>.<label>`` is derived;
        every other label is read from the file by `Labels`.
    class_names : Sequence[str] | None, optional
        Target ftag flavour class names, in label-index order (v1
        ``LabellerConfig.class_names``; GN3X 9-class, GN2X_qcdsplit 7-class).
        REQUIRED (the empty-class guard fires when empty).
    require_labels : bool, optional
        Whether every object must be labelled (v1
        ``LabellerConfig.require_labels``): True raises on an unlabelled object,
        False drops it. By default True.
    dtype_policy : Literal["int64-for-int", "file"], optional
        ``int64-for-int`` (v1, default) casts integer labels to int64; ``file``
        keeps the on-disk dtype.

    Raises
    ------
    ConfigError
        On an unknown ``dtype_policy``, or ``class_names`` empty (the empty-class
        guard, v1 ``configs.py:189``). NOTE the deliberate exception-TYPE promotion:
        v1's empty-class guard raises a bare ``ValueError`` (``configs.py:189``); v2
        standardises structural config-validation failures on ``ConfigError`` (the
        framework's named config-error type, NOT a ``ValueError`` subclass — see
        ``graph/errors.py``), as every M1-M5 processor does. The other two labeller
        guards (missing-field, require_labels-on-unlabelled) keep v1's ``ValueError``
        (raised at ``process`` time, not construction).
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
        # mirror v1 LabellerConfig (configs.py:154-197): the empty-class guard
        # (v1 configs.py:189) MUST fire when class_names is empty.
        if not class_names:
            raise ConfigError(
                f"FtagLabeller module {self.name!r}: class_names is empty — specify the "
                "target classes for relabelling (v1 LabellerConfig empty-class guard, "
                "configs.py:189)"
            )
        self.class_names = tuple(class_names)
        # the ftag Labeller IS the parity reference (ftag/labeller.py); v1 builds it
        # identically (Labeller(class_names, require_labels), datasets.py:205)
        self.labeller = Labeller(list(self.class_names), self.require_labels)

    def _labeller_variables(self) -> tuple[str, ...]:
        """The labeller's cut variables, de-duplicated, in first-seen order.

        ``Labeller.variables`` (``labeller.py:45``) is a flat sum over the
        per-class cut variables, so it carries duplicates — dedupe before
        declaring them as read fields.

        Returns
        -------
        tuple[str, ...]
            The unique cut variables the labeller reads from the raw stream.
        """
        return tuple(dict.fromkeys(self.labeller.variables))

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``raw.<stream>`` require -> CONCRETE ``labels.<stream>.<label>`` produce.

        The produce is concrete (never a wildcard, never schema-narrowed) so it
        beats the `Labels` ``labels.**`` wildcard via the planner's
        concrete-beats-wildcard rule.

        Returns
        -------
        IO
            The declared interface.
        """
        del mode
        requires = {f"raw.{self.stream}": TensorSpec(kind="data")}
        produces = {f"labels.{self.stream}.{self.label}": TensorSpec(kind="label")}
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def read_fields(self, step: PlanStep) -> dict[str, dict[str, str]]:
        """Demand the labeller's cut variables from ``raw.<stream>`` (design §6.1).

        The derived label is NOT on disk (it has no on-disk producer), so instead
        the labeller's cut variables (``Labeller.variables``, deduped) are demanded
        from ``raw.<stream>`` — the explicit "declare the extra read fields" rule
        (FD §6.1 1280-1282) that lets them enter the per-mode demand-narrowed read
        set.

        Returns
        -------
        dict[str, dict[str, str]]
            ``{stream: {var: this module}}`` for every deduped labeller cut variable.
        """
        del step
        out: dict[str, dict[str, str]] = {}
        for var in self._labeller_variables():
            out.setdefault(self.stream, {}).setdefault(var, self.name)
        return out

    def process(self, batch, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Derive the on-the-fly labeller labels from the raw structured array.

        Mirrors v1 ``process_labels`` (``datasets.py:619-626``): the missing-field
        guard (every ``Labeller.variables`` cut variable must be present,
        ``datasets.py:622-624``) then ``Labeller.get_labels`` on the WHOLE structured
        array, cast to int64 under ``dtype_policy``. The ``get_labels`` output is a
        fresh array (``labeller.py:66-73``), so it never aliases the reader buffer.

        Returns
        -------
        dict[str, np.ndarray]
            ``{labels.<stream>.<label>: derived array}`` — int64 (or file dtype
            under ``dtype_policy='file'``). With ``require_labels=False`` unmatched
            objects are dropped, so the array may be shorter than the batch (ftag
            ``labeller.py:73``; v1 parity).

        Raises
        ------
        ValueError
            If a labeller cut variable is absent from the raw stream (the v1
            field-subset check, ``datasets.py:622-624``), or — under
            ``require_labels`` — if any object matches no class (ftag
            ``labeller.py:70``).
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
        # get_labels raises under require_labels on an unlabelled object
        # (labeller.py:70) and otherwise drops it (labeller.py:73) — v1 parity.
        derived = self.labeller.get_labels(raw)
        if self.dtype_policy == "int64-for-int" and np.issubdtype(derived.dtype, np.integer):
            out = derived.astype(np.int64)
        else:
            out = np.array(derived, copy=True)
        return {f"labels.{self.stream}.{self.label}": out}


class MultiTarget(Processor):
    """Conditional row-wise target replacement (design §6.2; M5 sub-wave A3).

    Ports v1's ``multi_target`` feature (``datasets.py:237-248,648,695-739``):
    for each configured rule, the per-row value of an output label is replaced
    by a ``source`` label wherever a ``sel_label`` satisfies an operator
    comparison against a literal ``value`` — ``np.where(op(sel, value), source,
    running)`` (v1's ``torch.where``, ``datasets.py:733-739``). Two output modes,
    exactly as v1 (``datasets.py:237-248``):

    - ``target:`` — REPLACE an existing label. The base (pre-replacement) values
      are read from the raw stream (the column the task would otherwise consume
      directly), so this processor becomes the SOLE producer of
      ``labels.<stream>.<target>`` (a concrete produce beats the `Labels`
      wildcard, planner rule (a)) — write-once is preserved with no two-producer
      conflict.
    - ``custom_target:`` — CREATE a NEW label initialised to a NaN placeholder
      (v1 ``inject_custom_target_placeholders``, ``datasets.py:648-693``), then
      fill it where the condition holds.

    Multiple rules MAY name the same output — they apply SEQUENTIALLY over a
    running array, exactly as v1 mutates the labels dict in place
    (``datasets.py:709-739``). The shipped ``regression_multi_target.yaml`` uses
    this: two rules both write ``pt_label_handle`` (one ``ID==15``, one
    ``ID!=15``). All rules for one output must agree on the mode (all
    ``custom_target`` or all ``target``) — the base is established once (NaN
    placeholder or the raw column) and each rule layers a ``np.where`` on top.

    Each rule declares its own ``labels.<stream>.<sel_label>`` and
    ``labels.<stream>.<source>`` dependencies (produced by `Labels`, so the
    sel/source casting policy stays in ONE place). A ``sel_label``/``source``
    may not be a MultiTarget output (no producer→producer chaining — v1 reads
    sel/source from the file-loaded labels, never from a replaced target).

    Parameters
    ----------
    replacements : Sequence[Mapping[str, Any]]
        Ordered replacement rules. Each rule is a mapping with keys:

        - ``stream`` (v1 ``input_name``) — the labelled stream;
        - ``sel_label`` — the selection label compared against ``value``;
        - ``op`` — one of ``== != >= <= > <`` (v1 ``opp``, ``datasets.py:29-36``);
        - ``value`` — the literal compared against ``sel_label``;
        - ``source`` — the label whose value is written where the condition holds;
        - exactly one of ``target`` (replace existing) or ``custom_target``
          (create new) — the output label name.

    Raises
    ------
    ConfigError
        On an empty/malformed rule, an unknown operator, both/neither of
        ``target``/``custom_target``, mixed modes for one output, or a
        ``sel_label``/``source`` that is itself an output.
    """

    def __init__(self, replacements: Sequence[Mapping[str, Any]]) -> None:
        super().__init__()
        if not replacements:
            raise ConfigError(
                "MultiTarget needs at least one entry in 'replacements' (design §6.2)"
            )
        self.rules: list[dict[str, Any]] = [self._checked_rule(dict(rule)) for rule in replacements]
        # group by output, preserving first-seen order (the per-output base is
        # established once, then each rule layers in declaration order)
        self._outputs: dict[tuple[str, str], bool] = {}  # (stream, output) -> is_custom
        for rule in self.rules:
            key = (rule["stream"], rule["output"])
            if key in self._outputs and self._outputs[key] != rule["is_custom"]:
                raise ConfigError(
                    f"MultiTarget: output {rule['output']!r} on stream {rule['stream']!r} mixes "
                    "'target' and 'custom_target' rules — all rules for one output must agree "
                    "on the mode (v1 datasets.py:237-248)"
                )
            self._outputs.setdefault(key, rule["is_custom"])
        # a sel/source may not be an output (no chaining; v1 reads them from the
        # file-loaded labels, never a replaced value)
        for rule in self.rules:
            for ref in ("sel_label", "source"):
                if (rule["stream"], rule[ref]) in self._outputs:
                    raise ConfigError(
                        f"MultiTarget: rule {ref} {rule[ref]!r} on stream {rule['stream']!r} is "
                        "itself a MultiTarget output — chaining replacements is not supported "
                        "(v1 parity, datasets.py:695-739)"
                    )

    @staticmethod
    def _checked_rule(rule: dict[str, Any]) -> dict[str, Any]:
        """Validate one replacement rule and normalise it to a flat dict.

        Returns
        -------
        dict[str, Any]
            ``{stream, sel_label, op, value, source, output, is_custom}``.

        Raises
        ------
        ConfigError
            On a missing field, unknown operator, or a both/neither
            target/custom_target mistake (v1 datasets.py:237-248).
        """
        has_target = "target" in rule and rule["target"] is not None
        has_custom = "custom_target" in rule and rule["custom_target"] is not None
        if has_target and has_custom:
            raise ConfigError(
                f"MultiTarget: a rule cannot set both 'target' and 'custom_target' — use 'target' "
                f"to replace an existing label or 'custom_target' to create one (got {rule})"
            )
        if not has_target and not has_custom:
            raise ConfigError(
                f"MultiTarget: a rule must set either 'target' or 'custom_target' (got {rule})"
            )
        op = rule.get("op")
        if op not in _OPERATORS:
            raise ConfigError(
                f"MultiTarget: unknown operator {op!r} — allowed operators are "
                f"{sorted(_OPERATORS)} (v1 datasets.py:29-36)"
            )
        missing = [k for k in ("stream", "sel_label", "value", "source") if rule.get(k) is None]
        if missing:
            raise ConfigError(
                f"MultiTarget: rule is missing required fields {missing} "
                f"(stream, sel_label, op, value, source) (got {rule})"
            )
        return {
            "stream": str(rule["stream"]),
            "sel_label": str(rule["sel_label"]),
            "op": str(op),
            "value": rule["value"],
            "source": str(rule["source"]),
            "output": str(rule["target"]) if has_target else str(rule["custom_target"]),
            "is_custom": has_custom,
        }

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``labels.<s>.{sel,source}`` (+ raw target) -> ``labels.<s>.<out>``.

        ``target:`` outputs also require the raw target column (the
        pre-replacement base values), declared via ``raw.<stream>`` fields so
        the reader narrows to it; ``custom_target:`` outputs need no base column
        (the placeholder is NaN). Every produced output is a TRAINING-gated
        label leaf — the conditional targets feed only the loss.

        Returns
        -------
        IO
            The declared interface.
        """
        del mode
        requires: dict[str, TensorSpec] = {}
        raw_fields: dict[str, list[str]] = {}
        produces: dict[str, TensorSpec] = {}
        label = {"dtype": "float32", "kind": "label", "modes": Mode.TRAINING}
        for rule in self.rules:
            stream = rule["stream"]
            requires[f"labels.{stream}.{rule['sel_label']}"] = TensorSpec(
                kind="label", modes=Mode.TRAINING
            )
            requires[f"labels.{stream}.{rule['source']}"] = TensorSpec(
                kind="label", modes=Mode.TRAINING
            )
        for (stream, output), is_custom in self._outputs.items():
            produces[f"labels.{stream}.{output}"] = TensorSpec(**label)
            if not is_custom:
                # the base (pre-replacement) values come from the raw stream
                raw_fields.setdefault(stream, []).append(output)
        for stream, fields in raw_fields.items():
            requires[f"raw.{stream}"] = TensorSpec(
                kind="data", fields=tuple(dict.fromkeys(fields)), modes=Mode.TRAINING
            )
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def process(self, batch, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Apply each conditional replacement over a per-output running array.

        ``np.where`` is the v1 ``torch.where`` (datasets.py:735-739): keep
        ``source`` where ``op(sel, value)`` holds, else the running value. The
        base is established once per output (raw target column, or a NaN
        placeholder for ``custom_target``); rules layer in declaration order
        (v1 in-place mutation, datasets.py:709-739).

        Returns
        -------
        dict[str, np.ndarray]
            ``{labels.<stream>.<output>: fresh array}`` — one per output.
        """
        del rows, mode
        running: dict[tuple[str, str], np.ndarray] = {}
        for (stream, output), is_custom in self._outputs.items():
            if is_custom:
                # v1 inject_custom_target_placeholders (datasets.py:686-690):
                # a NaN-filled column shaped/typed like the first rule's source.
                # DEVIATION from v1: v1 uses dtype=batch[source].dtype verbatim;
                # v2 promotes any sub-float32 source (e.g. f2) to >=float32 via
                # np.result_type so a NaN-filled regression placeholder always has
                # the range to hold log/ratio targets. For f4/f8 sources the two
                # agree byte-for-byte (the only dtypes any shipped
                # regression_multi_target.yaml source uses — HadronConeExclTruthLabelPt
                # and pt are both f4); the divergence is reachable only with an f2
                # source, which no shipped config has. R4 gates the f4 path bitwise.
                src0 = next(
                    r["source"]
                    for r in self.rules
                    if (r["stream"], r["output"]) == (stream, output)
                )
                template = batch.get(f"labels.{stream}.{src0}")
                running[stream, output] = np.full(
                    template.shape, np.nan, dtype=np.result_type(template.dtype, np.float32)
                )
            else:
                running[stream, output] = np.array(batch.get(f"raw.{stream}")[output], copy=True)
        for rule in self.rules:
            stream, output = rule["stream"], rule["output"]
            sel = batch.get(f"labels.{stream}.{rule['sel_label']}")
            source = batch.get(f"labels.{stream}.{rule['source']}")
            mask = _OPERATORS[rule["op"]](sel, rule["value"])
            running[stream, output] = np.where(mask, source, running[stream, output])
        return {f"labels.{stream}.{output}": arr for (stream, output), arr in running.items()}


# the v2 bundle stream name for the reconstructed objects (FD uses labels.objects.*,
# independent of the file group name the raw object features live in, e.g. truth_hadrons)
_OBJECT_STREAM = "objects"


@dataclass(frozen=True)
class _ObjectCut:
    """A single field-bound cut on MaskFormer truth objects (MFU-2 config surface).

    Lightweight v2 port of v1 ``salt.utils.configs.ObjectCut`` (configs.py) — a
    keep-rule ``min <= batch[field] <= max``. It is CARRIED on the processor for
    MFU-3 (object selection lands inside ``process()`` there); MFU-2 only parses
    and validates it. Defined locally rather than importing the orphaned v1
    dataclass (plan 39, Decision 5). NaN handling / PV exemption / drop semantics
    are MFU-3 concerns and not encoded here.

    Parameters
    ----------
    field : str
        Object-group field the cut bounds.
    min : float | None, optional
        Inclusive lower bound; ``None`` disables it, by default None.
    max : float | None, optional
        Inclusive upper bound; ``None`` disables it, by default None.

    Raises
    ------
    ConfigError
        When both ``min`` and ``max`` are ``None`` (v1 ObjectCut.__post_init__).
    """

    field: str
    min: float | None = None
    max: float | None = None

    def __post_init__(self) -> None:
        if self.min is None and self.max is None:
            raise ConfigError(
                f"MaskFormerTargets: ObjectCut on field {self.field!r} must set at least one of "
                "min/max (v1 ObjectCut.__post_init__, configs.py)"
            )


class MaskFormerTargets(Processor):
    """MaskFormer object targets: ``object_class`` + per-constituent ``masks`` (design §6.2).

    Ports the v1 object-target construction (``datasets.py:549-553,636-644``;
    FD 1090-1110) into a single demand-gated processor. From the truth-object
    group it produces, under the ``objects`` bundle stream:

    - ``labels.objects.object_class`` ``[B, M]`` int64 — the raw object class
      label remapped through ``class_map`` (the v1 ``x[x == k] = v`` loop,
      ``datasets.py:641-643``), with the **null class validated LAST** (FD 1108;
      v1 ``MaskformerObjectConfig`` ``__post_init__``, configs.py:39-42).
    - ``labels.objects.masks`` ``[B, M, T]`` bool — the per-object-by-constituent
      truth mask: ``constituent_id == object_id`` (the v1 `build_target_masks`
      equality, mask_utils.py:36-37) over the truncated constituent stream.
    - ``labels.objects.<target>`` ``[B, M]`` float32 per ``regression_targets``
      entry — the raw per-object regression labels the object-regression task
      stacks + scales (v1 reads ``labels["objects"][target]``, task.py:458-460).

    Two v1 IN-PLACE id mutations are eliminated (FD 1095, "NO in-place id
    mutation"):

    - v1's class remap mutates the loaded tensor sequentially (``x[x == k] = v``,
      ``datasets.py:642``) — order-dependent and silently corrupting whenever a
      ``mapped`` value collides with a not-yet-visited ``raw`` value. v2 builds
      the mapped column from the ORIGINAL raw values via a single vectorised
      ``np.select`` against a copy, so the mapping is atomic and collision-proof.
    - v1's `build_target_masks` mutates the object-id tensor in place
      (``object_ids[object_ids == -1] = -999``, mask_utils.py:36) before the
      equality, leaking a sentinel back into the caller's labels dict. v2
      computes the equality on a private sentinel-substituted COPY, so the
      published ``object_class`` / id columns are never touched.

    DEMAND-gated, not mode-gated (FD 1090-1095): all three product families are
    declared in ALL modes; ordinary sink pruning removes the module from a plan
    only when nothing demands its outputs. The `MaskFormerObjectWriter` demands
    ``labels.objects.{object_class,masks}`` in TEST (truth columns), so this
    module IS in the test plan — matching v1, where the object labels are built
    stage-independently (``datasets.py:549``, ``process_labels`` has no stage
    gate). It is pruned from ONNX (nothing demands truth there).

    Parameters
    ----------
    object_class : str
        Object class label field in the object group (v1 ``object.class_label``;
        e.g. ``flavour``). Remapped through ``class_map`` to ``object_class``.
    object_id : str
        Object identity field used to build masks (v1 ``object.id_label``; e.g.
        ``barcode``).
    constituent_id : str
        Constituent identity field tested against ``object_id`` to build masks
        (v1 ``constituent.id_label``; e.g. ``ftagTruthParentBarcode``).
    class_map : Mapping[str, Mapping[str, Any]]
        ``{name: {raw: int | list[int], mapped: int, weight?: float}}`` (v1
        ``object.object_classes``). MUST contain a ``null`` entry mapped LAST
        (``mapped == len(class_map) - 1``), and the ``mapped`` values MUST be
        exactly ``range(len(class_map))`` (v1 configs.py:39-42). jsonargparse may
        parse the YAML ``null:`` key as a Python ``None`` — both spellings are
        accepted, normalised to ``"null"``. ``raw`` may be a single int (the
        common case) or a list/tuple of ints that all map to the same ``mapped``
        index — a class *merge* (v1 ``class_map`` tuple keys). An optional scalar
        ``weight`` per class feeds :attr:`object_weights` (v1, default ``1.0``).
    object_stream : str
        File group holding the object features (v1 ``object.name``; e.g.
        ``truth_hadrons``). The reader serves it as ``raw.<object_stream>``.
    constituent_stream : str
        File group holding the constituent features (v1 ``constituent.name``;
        e.g. ``tracks``). The mask's last dim aligns with this stream's token
        count.
    regression_targets : Sequence[str] | None, optional
        Per-object regression label fields published under
        ``labels.objects.<target>`` (v1 reads them via the object-regression
        task's ``get_targets``), by default None (no regression labels).
    num_objects : int | None, optional
        The number of object queries ``M`` (v1 ``num_objects``,
        MaskFormer.yaml:36). When set, the produced shapes carry it as a
        concrete dim (a static check that the file's object count matches the
        decoder's query bank); None leaves ``M`` symbolic. Doubles as the legacy
        alias bridged to ``max_objects`` (see below).
    cuts : Sequence[_ObjectCut | Mapping[str, Any]] | None, optional
        Per-jet field cuts for MFU-3 object selection (v1 ``object.cuts``).
        STORED, not consumed in MFU-2 — ``process()`` data behaviour is unchanged.
        dicts are coerced to :class:`_ObjectCut`. Default None.
    sort_by : str | None, optional
        Object-group field to sort survivors by before truncation (v1
        ``object.sort_by``). STORED for MFU-3. Default None (file slot order).
    sort_descending : bool, optional
        Sort direction for ``sort_by`` (v1 ``object.sort_descending``). STORED for
        MFU-3. Default True.
    pv_class : int | None, optional
        Mapped class index identifying the primary vertex pinned at slot 0 (v1
        ``object.pv_class``). STORED for MFU-3. Validated to a non-null mapped
        index. ``None`` disables PV pinning. Default 0.
    max_objects : int | None, optional
        Max object slots retained per jet after MFU-3 selection (v1
        ``object.max_objects``). STORED for MFU-3. ``None`` auto-links to
        ``num_objects`` (the decoder query bank) via the legacy bridge.
    max_lxy_mm : float | None, optional
        |Lxy| threshold (mm) above which a vertex is re-labelled to null in MFU-3
        (v1 ``object.max_lxy_mm``). STORED for MFU-3. Default None (disabled).
    lxy_field : str, optional
        Name of the Lxy field used by ``max_lxy_mm`` (v1 ``object.lxy_field``).
        STORED for MFU-3. Default ``"Lxy"``.

    Attributes
    ----------
    object_weights : list[float]
        Per-class loss weights ordered by mapped index, derived from each class's
        optional ``weight`` (v1 ``object_weights``). Carried for the MFU-5 loss.

    Raises
    ------
    ConfigError
        On a missing ``null`` class, a null not mapped last, ``mapped`` values
        that are not ``range(len(class_map))``, a raw value shared across mapped
        indices, a non-scalar class weight, a duplicate regression target, a
        non-positive ``num_objects`` / ``max_objects``, an out-of-range
        ``pv_class``, or an ``_ObjectCut`` with neither min nor max.
    """

    def __init__(
        self,
        object_class: str,
        object_id: str,
        constituent_id: str,
        class_map: Mapping[str, Mapping[str, int]],
        object_stream: str,
        constituent_stream: str,
        regression_targets: Sequence[str] | None = None,
        num_objects: int | None = None,
        cuts: Sequence[_ObjectCut | Mapping[str, Any]] | None = None,
        sort_by: str | None = None,
        sort_descending: bool = True,
        pv_class: int | None = 0,
        max_objects: int | None = None,
        max_lxy_mm: float | None = None,
        lxy_field: str = "Lxy",
    ) -> None:
        super().__init__()
        self.object_class = str(object_class)
        self.object_id = str(object_id)
        self.constituent_id = str(constituent_id)
        self.object_stream = str(object_stream)
        self.constituent_stream = str(constituent_stream)
        self._raw_to_mapped = self._checked_class_map(class_map)
        # per-class loss weights, ordered by mapped index (v1 derived property
        # MaskformerObjectConfig.object_weights, configs.py). Carried for the MFU-5
        # loss; unused in MFU-2/MFU-3. Defaults to 1.0 per class.
        self.object_weights: list[float] = self._class_weights(class_map)
        self.regression_targets: tuple[str, ...] = tuple(regression_targets or ())
        if len(set(self.regression_targets)) != len(self.regression_targets):
            raise ConfigError(
                f"MaskFormerTargets: duplicate regression targets in {self.regression_targets}"
            )

        # --- object-selection config surface (MFU-2: stored, NOT consumed) -------
        # Generic per-jet field cuts. jsonargparse may hand dicts → coerce to
        # _ObjectCut (v1 MaskformerObjectConfig.__post_init__ dict→ObjectCut).
        self.cuts: tuple[_ObjectCut, ...] = tuple(
            c if isinstance(c, _ObjectCut) else _ObjectCut(**dict(c)) for c in (cuts or ())
        )
        self.sort_by = sort_by
        self.sort_descending = bool(sort_descending)
        self.max_lxy_mm = max_lxy_mm
        self.lxy_field = str(lxy_field)

        # MFU-3 IDENTITY GATE (plan 39, risk #1). Object selection — and crucially
        # the pv_class PV-pin REORDER — runs ONLY when the user EXPLICITLY configured
        # cuts / sort_by / max_objects. It MUST key on the PRE-bridge `max_objects`
        # argument: the num_objects -> max_objects bridge below sets self.max_objects
        # from the decoder query bank for EVERY MaskFormer config, so gating on
        # self.max_objects would fire selection unconditionally and break byte-
        # identity with MFU-2 (pv_class defaults to 0, so the PV-pin would silently
        # reorder slot 0). Upstream's _needs_object_selection (datasets.py:562) gates
        # on max_objects too, but upstream's CLI populates max_objects from the
        # decoder so it always selects; v2 deliberately keeps the unconfigured path a
        # no-op. max_lxy relabel is a SEPARATE gate (self.max_lxy_mm is not None).
        self._should_select: bool = (
            bool(self.cuts) or sort_by is not None or max_objects is not None
        )

        # In v2 num_objects is the decoder query bank (the declared label M dim,
        # set from mask_decoder.num_objects in convert.py) and max_objects is the
        # MFU-3 selection truncation count. They MUST agree: declare_io produces
        # object_class with M == num_objects while _select_objects truncates to
        # max_objects, so a config that sets BOTH to different values would emit a
        # [B, max_objects] array against a declared [B, num_objects] shape. The
        # bridge below only equalises them when one is None, so guard the both-set
        # case explicitly (v1 had a single alias, so this is newly reachable in v2).
        if (
            num_objects is not None
            and max_objects is not None
            and num_objects != max_objects
        ):
            raise ConfigError(
                f"MaskFormerTargets: num_objects ({num_objects}) and max_objects "
                f"({max_objects}) are both set but differ — num_objects is the decoder "
                "query bank (declared label M) and max_objects is the selection "
                "truncation count; they must be equal (set only one, or set both equal)"
            )

        # legacy num_objects <-> max_objects bridge (v1 MaskformerObjectConfig
        # __post_init__). In v2 num_objects is ALSO the decoder query bank (M, set
        # from mask_decoder.num_objects in convert.py); this bridge therefore
        # auto-links the MFU-3 truncation count (max_objects) to that query bank
        # when the config leaves it unset — exactly v1's "max_objects auto-linked
        # from model.mask_decoder.num_objects". max_objects wins when both are set.
        if max_objects is None and num_objects is not None:
            max_objects = num_objects
        if num_objects is None and max_objects is not None:
            num_objects = max_objects
        if num_objects is not None and num_objects < 1:
            raise ConfigError(f"MaskFormerTargets: num_objects must be >= 1, got {num_objects}")
        if max_objects is not None and max_objects < 1:
            raise ConfigError(f"MaskFormerTargets: max_objects must be >= 1, got {max_objects}")
        self.num_objects = num_objects
        self.max_objects = max_objects

        # PV class must be a valid non-null mapped index (v1 MaskformerObjectConfig
        # __post_init__ pv_class validation, configs.py). null_index == n_non_null.
        self.pv_class = pv_class
        if pv_class is not None:
            n_non_null = self.null_index
            if not (0 <= pv_class < n_non_null):
                raise ConfigError(
                    f"MaskFormerTargets: pv_class={pv_class} must be in [0, {n_non_null - 1}] "
                    "(non-null mapped indices; v1 configs.py)"
                )

        # MFU-3 derived raw-id sets (mapped-index world -> raw-id world). PV raws:
        # every raw whose mapped == pv_class (upstream pv_raw_values, configs.py:275)
        # — these classes get pinned at slot 0, exempt from cuts/sorts. null raw: the
        # raw whose mapped == null_index (upstream null_raw_value, configs.py:297) —
        # the sentinel written to pad slots' class field so the np.select class-map
        # maps them cleanly back to null_index.
        self._pv_raw_values: tuple[int, ...] = (
            tuple(r for r, m in self._raw_to_mapped.items() if m == self.pv_class)
            if self.pv_class is not None
            else ()
        )
        self._null_raw_value: int = next(
            r for r, m in self._raw_to_mapped.items() if m == self.null_index
        )

    @staticmethod
    def _checked_class_map(class_map: Mapping[str, Mapping[str, Any]]) -> dict[int, int]:
        """Validate the class map (null LAST, mapped == range) and return raw->mapped.

        Reproduces the v1 ``MaskformerObjectConfig`` invariants (configs.py:36-42)
        WITHOUT mutating the loaded ids: ``null`` present (``None`` or the string
        ``"null"`` accepted, jsonargparse may cast the YAML ``null:`` key to
        ``None``), null mapped LAST, and the ``mapped`` set exactly
        ``range(len(class_map))``.

        A class entry's ``raw`` may be a single int OR a list/tuple of ints that
        all map to the SAME ``mapped`` index — a class *merge* (v1
        ``MaskformerObjectConfig.class_map`` property, configs.py, which keys the
        map by a tuple of raws). The returned flat ``{raw: mapped}`` dict expands
        each merged raw to its shared mapped index, so ``process()``'s
        ``np.select`` over the dict keys handles merges with NO logic change. A
        single-int ``raw`` produces the IDENTICAL dict as before. Merged raw
        values MUST be disjoint across classes (no raw maps to two mapped indices).

        Returns
        -------
        dict[int, int]
            ``{raw: mapped}`` for every class, one key per (expanded) raw value.

        Raises
        ------
        ConfigError
            On a missing null, a null not mapped last, a malformed mapped set, an
            empty raw list, or a raw value shared across two mapped indices.
        """
        if not class_map:
            raise ConfigError("MaskFormerTargets: class_map must not be empty (FD 1108)")
        # jsonargparse may parse a YAML ``null:`` key as Python None — accept both.
        names = {
            ("null" if name is None else str(name)): dict(spec) for name, spec in class_map.items()
        }
        if "null" not in names:
            raise ConfigError(
                "MaskFormerTargets: class_map must contain a 'null' (no-object) class "
                "(v1 MaskformerObjectConfig, configs.py:36)"
            )
        n = len(names)
        raw_to_mapped: dict[int, int] = {}
        for name, spec in names.items():
            if "mapped" not in spec:
                raise ConfigError(
                    f"MaskFormerTargets: class_map[{name!r}] is missing a 'mapped' index"
                )
            if name != "null" and "raw" not in spec:
                raise ConfigError(
                    f"MaskFormerTargets: class_map[{name!r}] is missing a 'raw' index "
                    "(only the 'null' class may omit it; v1 object_classes)"
                )
            mapped = int(spec["mapped"])
            # the null class's raw id defaults to -1 (v1 MaskFormer.yaml:177: null raw -1).
            # raw may be a scalar (the common case) or a list/tuple → class merge.
            raw_spec = spec.get("raw", -1)
            if isinstance(raw_spec, (list, tuple)):
                if not raw_spec:
                    raise ConfigError(
                        f"MaskFormerTargets: class_map[{name!r}] 'raw' list must not be empty"
                    )
                raws = [int(r) for r in raw_spec]
            else:
                raws = [int(raw_spec)]
            for r in raws:
                if r in raw_to_mapped and raw_to_mapped[r] != mapped:
                    raise ConfigError(
                        f"MaskFormerTargets: raw class id {r} is mapped to multiple classes "
                        f"({raw_to_mapped[r]} and {mapped}) — merged raws must be disjoint "
                        "across mapped indices (v1 object_classes)"
                    )
                raw_to_mapped[r] = mapped
        if names["null"]["mapped"] != n - 1:
            raise ConfigError(
                f"MaskFormerTargets: the 'null' class must be mapped LAST (to {n - 1}), got "
                f"{names['null']['mapped']} (v1 configs.py:39 'Null class must be last')"
            )
        if set(raw_to_mapped.values()) != set(range(n)):
            raise ConfigError(
                f"MaskFormerTargets: mapped class indices {sorted(set(raw_to_mapped.values()))} "
                f"must be exactly range({n}) (v1 configs.py:42)"
            )
        return raw_to_mapped

    @staticmethod
    def _class_weights(class_map: Mapping[str, Mapping[str, Any]]) -> list[float]:
        """Per-class loss weights ordered by mapped index (v1 object_weights).

        Ports v1 ``MaskformerObjectConfig.object_weights`` (configs.py): each class
        carries an optional scalar ``weight`` (default ``1.0``). v1 returns them in
        dict-iteration order; v2 orders explicitly by ``mapped`` index so the list
        index aligns with the class index the MFU-5 loss expects. Stored, not
        consumed, in MFU-2.

        Returns
        -------
        list[float]
            One weight per class, index ``i`` == the weight of mapped class ``i``.

        Raises
        ------
        ConfigError
            When a class ``weight`` is a list (must be a scalar; v1 assert).
        """
        names = {
            ("null" if name is None else str(name)): dict(spec) for name, spec in class_map.items()
        }
        by_mapped: dict[int, float] = {}
        for name, spec in names.items():
            w = spec.get("weight", 1.0)
            if isinstance(w, (list, tuple)):
                raise ConfigError(
                    f"MaskFormerTargets: class_map[{name!r}] 'weight' must be a scalar, got {w!r} "
                    "(v1 object_weights, configs.py)"
                )
            by_mapped[int(spec["mapped"])] = float(w)
        return [by_mapped[i] for i in range(len(by_mapped))]

    @property
    def null_index(self) -> int:
        """The mapped index of the null/no-object class (== num_classes).

        Returns
        -------
        int
            ``max(mapped values)`` == ``num_classes - 1`` — the matcher's
            ``num_classes`` sentinel. Uses the max mapped value (not the count of
            raw keys) so it stays correct when classes merge multiple raws.
        """
        return max(self._raw_to_mapped.values())

    def declare_io(self, mode: Mode) -> IO:
        """Declare the raw object/constituent fields -> ``labels.objects.*`` (ALL modes).

        Requires the object class + id + regression fields from
        ``raw.<object_stream>`` and the constituent id from
        ``raw.<constituent_stream>``; produces ``labels.objects.object_class``
        ``[B, M]``, ``labels.objects.masks`` ``[B, M, T]`` and one
        ``labels.objects.<target>`` ``[B, M]`` per regression target. Every
        product is declared in ALL modes — the gate is DEMAND, not mode (FD
        1090-1095); the planner prunes the module from a plan that demands none.

        Returns
        -------
        IO
            The declared interface.
        """
        del mode
        # base fields always read (MFU-2). When selection is active, the cut/sort
        # fields must also be read so _select_objects can evaluate them; when the
        # Lxy relabel is active, lxy_field too. Both are added ONLY under their gate,
        # so the unconfigured path declares the IDENTICAL requires as MFU-2 (byte-
        # identity). dict.fromkeys dedupes while preserving first-seen order.
        obj_field_list = [self.object_class, self.object_id, *self.regression_targets]
        if self._should_select:
            obj_field_list += [c.field for c in self.cuts]
            if self.sort_by is not None:
                obj_field_list.append(self.sort_by)
        if self.max_lxy_mm is not None:
            obj_field_list.append(self.lxy_field)
        obj_fields = tuple(dict.fromkeys(obj_field_list))
        requires = {
            f"raw.{self.object_stream}": TensorSpec(kind="data", fields=obj_fields),
            f"raw.{self.constituent_stream}": TensorSpec(
                kind="data", fields=(self.constituent_id,)
            ),
        }
        m: int | str = self.num_objects if self.num_objects is not None else sym_dim("M", self.name)
        tok = sym_dim("T", self.constituent_stream)
        produces: dict[str, TensorSpec] = {
            f"labels.{_OBJECT_STREAM}.object_class": TensorSpec(
                shape=("B", m), dtype="int64", kind="label"
            ),
            f"labels.{_OBJECT_STREAM}.masks": TensorSpec(
                shape=("B", m, tok), dtype="bool", kind="label"
            ),
        }
        for target in self.regression_targets:
            produces[f"labels.{_OBJECT_STREAM}.{target}"] = TensorSpec(
                shape=("B", m), dtype="float32", kind="label"
            )
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def _select_objects(self, obj: np.ndarray) -> np.ndarray:
        """Per-jet cuts -> PV-pin -> sort -> truncate (faithful port of v1 _select_objects).

        Ports ``salt/data/datasets.py::_select_objects`` (upstream 39-165) into the
        mapped-index world. Four per-jet phases:

        1. **CUTS** — drop vertices failing any :class:`_ObjectCut`
           (``min <= field <= max``); NaN (float) FAILS the cut (strict); AND across
           cuts. PV is exempt.
        2. **PV-PIN** — vertices whose raw class is in :attr:`_pv_raw_values` (raws
           mapping to ``pv_class``) go to slot 0 in input order, EXEMPT from
           cuts/sorts.
        3. **SORT** — surviving non-PV vertices sorted by :attr:`sort_by`
           (``np.argsort`` ``kind="stable"``; reversed if :attr:`sort_descending`).
        4. **TRUNCATE** — keep ``concat([pv, non_pv])[:n_out]``,
           ``n_out = max_objects``.

        **Pad convention (risk #2).** The production v2 reader appends an
        authoritative ``valid`` bool field to every assembled stream
        (``stream.py:251-255``), keyed exactly as upstream's ``valid``; it is used
        directly when present (upstream-faithful). The hand-built unit fixtures omit
        ``valid``, so the code falls back to the v2 pad sentinel: signed-int label
        fields — INCLUDING ``object_id`` — are padded with ``INT_PAD_SENTINEL = -1``
        (``stream.py``), so ``object_id == -1`` <=> ``~valid``. Either source feeds
        cut-candidate masking and the PV partition uniformly. A NEW ``[B, n_out]``
        structured array is built (pad slots default to 0/False), then pad slots are
        sentinel-filled: ``id -> -1`` (so ``build_target_masks`` treats them as
        invalid) and the class field ``-> _null_raw_value`` (so the np.select
        class-map maps them to ``null_index``).

        Parameters
        ----------
        obj : np.ndarray
            Structured object array ``[B, M_in]`` from ``raw.<object_stream>``.

        Returns
        -------
        np.ndarray
            A NEW structured ``[B, n_out]`` array of the same dtype, with all fields
            permuted/truncated in lockstep and pad slots sentinel-filled.

        Raises
        ------
        SchemaError
            If a configured cut/sort field (or the class field needed for PV) is
            absent from the object dtype.
        """
        n_jets, n_in = obj.shape
        n_out = self.max_objects if self.max_objects is not None else n_in

        # required fields must be present in the structured dtype (declare_io adds
        # them to the read set; this guards a schema-less / mis-wired read).
        needed: set[str] = {c.field for c in self.cuts}
        if self.sort_by is not None:
            needed.add(self.sort_by)
        if self._pv_raw_values:
            needed.add(self.object_class)
        missing = needed - set(obj.dtype.names or ())
        if missing:
            raise SchemaError(
                f"MaskFormerTargets: object selection needs fields {sorted(missing)} but "
                f"raw.{self.object_stream} has {obj.dtype.names}. Add them to the object "
                "stream variables/read set."
            )

        # candidate (non-pad) slots. The production v2 reader appends an
        # authoritative `valid` bool field to every assembled stream
        # (stream.py:251-255), keyed exactly as upstream's `valid`; prefer it so a
        # legitimate object with object_id == -1 is NOT silently dropped. The
        # hand-built unit fixtures omit `valid`, so fall back to the v2 pad sentinel
        # object_id == -1 (INT_PAD_SENTINEL, stream.py) == ~valid.
        if "valid" in (obj.dtype.names or ()):
            valid = np.asarray(obj["valid"]).astype(bool)
        else:
            valid = np.asarray(obj[self.object_id]) != -1

        # per-cut keep mask; NaN (float) FAILS the cut (strict); AND across cuts.
        keep = valid.copy()
        for cut in self.cuts:
            vals = np.asarray(obj[cut.field])
            ok = np.ones(vals.shape, dtype=bool)
            if np.issubdtype(vals.dtype, np.floating):
                ok &= ~np.isnan(vals)
            if cut.min is not None:
                ok &= vals >= cut.min
            if cut.max is not None:
                ok &= vals <= cut.max
            keep &= ok

        # PV identification (vectorised). PV is pinned at slot 0 and exempt from
        # cuts/sorts. Restricted to non-pad slots.
        if self._pv_raw_values:
            pv_mask = np.isin(np.asarray(obj[self.object_class]), self._pv_raw_values) & valid
        else:
            pv_mask = np.zeros((n_jets, n_in), dtype=bool)

        out = np.zeros((n_jets, n_out), dtype=obj.dtype)  # pad slots default 0/False
        filled = np.zeros((n_jets, n_out), dtype=bool)
        sort_vals = np.asarray(obj[self.sort_by]) if self.sort_by is not None else None
        for j in range(n_jets):
            is_pv_j = pv_mask[j]
            pv_idx = np.where(is_pv_j)[0]
            non_pv_idx = np.where(keep[j] & ~is_pv_j)[0]
            if sort_vals is not None and non_pv_idx.size:
                order = np.argsort(sort_vals[j][non_pv_idx], kind="stable")
                if self.sort_descending:
                    order = order[::-1]
                non_pv_idx = non_pv_idx[order]
            chosen = np.concatenate([pv_idx, non_pv_idx])[:n_out]
            if chosen.size:
                out[j, : chosen.size] = obj[j][chosen]
                filled[j, : chosen.size] = True

        # sentinel-fill pad slots so they don't collide with real ids/classes
        # (v1 datasets.py:148-164). 'filled' (not a valid field) marks the pads.
        pad = ~filled
        if pad.any():
            out[self.object_id][pad] = -1
            out[self.object_class][pad] = self._null_raw_value
        return out

    def process(self, batch, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Build the object class, the truth masks, and the raw regression labels.

        ``object_class`` is mapped from the ORIGINAL raw values via a single
        vectorised ``np.select`` (no sequential in-place remap); ``masks`` is the
        ``constituent_id == object_id`` broadcast over a private sentinel-
        substituted COPY of the ids (no in-place id mutation). Regression labels
        are copied out verbatim as float32.

        Returns
        -------
        dict[str, np.ndarray]
            ``labels.objects.object_class`` ``[B, M]`` int64,
            ``labels.objects.masks`` ``[B, M, T]`` bool, and one
            ``labels.objects.<target>`` ``[B, M]`` float32 per regression target.
        """
        del rows, mode
        obj = batch.get(f"raw.{self.object_stream}")
        con = batch.get(f"raw.{self.constituent_stream}")
        out: dict[str, np.ndarray] = {}

        # MFU-3 object selection (cuts -> PV-pin -> sort -> truncate). GATED: only
        # when the user explicitly configured cuts/sort_by/max_objects. When OFF,
        # `obj` is the raw stream untouched -> the blocks below are byte-identical to
        # MFU-2. The selection rebuilds a NEW [B, n_out] structured array so EVERY
        # field (class, id, regression, lxy) is permuted/truncated IN LOCKSTEP.
        if self._should_select:
            obj = self._select_objects(obj)

        # object_class: map raw -> mapped from the ORIGINAL values (atomic, no
        # sequential x[x==k]=v mutation; v1 datasets.py:641-643). Unmapped raw
        # values fall through to the null index (v1 reads only configured classes,
        # and any object whose raw class is not in the map is a no-object slot).
        raw_class = np.asarray(obj[self.object_class])
        conds = [raw_class == raw for raw in self._raw_to_mapped]
        choices = [self._raw_to_mapped[raw] for raw in self._raw_to_mapped]
        object_class = np.select(conds, choices, default=self.null_index).astype(np.int64)

        # MFU-3 Lxy relabel (SEPARATE gate, self.max_lxy_mm is not None; v1
        # datasets.py:814-820). AFTER the class-map: vertices with |Lxy| > max_lxy_mm
        # cannot be reconstructed by the tracker -> re-label to null. NaN-safe:
        # np.abs(nan) > thr is always False, so null/pad slots (NaN or 0 Lxy) are
        # never accidentally relabelled.
        if self.max_lxy_mm is not None:
            lxy = np.asarray(obj[self.lxy_field])
            object_class[np.abs(lxy) > self.max_lxy_mm] = self.null_index
        out[f"labels.{_OBJECT_STREAM}.object_class"] = object_class

        # masks: constituent_id == object_id, [B, M] x [B, T] -> [B, M, T]. v1
        # build_target_masks substitutes -1 ids with -999 IN PLACE before the
        # equality (mask_utils.py:36); we do it on a private COPY so the published
        # object_class / ids are untouched. The substitution makes invalid (-1)
        # objects never match a constituent (constituent ids are non-negative).
        object_ids = np.array(obj[self.object_id], copy=True)
        object_ids[object_ids == -1] = -999
        constituent_ids = np.asarray(con[self.constituent_id])
        # [B, M, 1] == [B, 1, T] -> [B, M, T]
        out[f"labels.{_OBJECT_STREAM}.masks"] = (
            object_ids[:, :, None] == constituent_ids[:, None, :]
        )

        # raw per-object regression labels (float32; the task stacks + scales them)
        for target in self.regression_targets:
            out[f"labels.{_OBJECT_STREAM}.{target}"] = np.asarray(obj[target], dtype=np.float32)
        return out
