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
from typing import Any, ClassVar, Literal

import numpy as np
from ftag import Labeller
from numpy.lib.recfunctions import structured_to_unstructured as s2u

from salt.core.data.base import Processor, WorkerCtx
from salt.core.graph.errors import ConfigError
from salt.core.graph.planner import PlanStep
from salt.core.graph.spec import IO, KEY_SEP, Mode, TensorSpec, sym_dim, unflatten_spec

__all__ = ["Features", "Labels", "MaskFormerTargets", "MultiTarget"]

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

    On-the-fly Labeller (M6 sub-wave A; design §6.2, FD 1303-1304)
    -------------------------------------------------------------
    The ftag ``Labeller`` (NOT v1-salt — ``ftag/labeller.py``) is an OPT-IN
    init-arg, "no longer heuristically triggered" (v1 gated implicitly on
    ``input_name == global_object and label == 'flavour_label'``,
    ``datasets.py:609-626``). Set ``use_labeller: true`` plus ``class_names``
    (the v1 ``LabellerConfig{use_labeller, class_names, require_labels}``,
    ``configs.py:154-197``) and the one ``labeller_stream.labeller_label`` key
    the labeller serves. When that exact narrowed key is demanded, this module
    derives it on the fly from the raw structured array via
    ``Labeller.get_labels`` (post-`Reader.read`, pre-torch boundary, FD
    §2.4) — int -> int64 under ``dtype_policy``, no longer read from the file.
    The labeller's cut variables (``Labeller.variables``) are declared as extra
    read fields so they enter the per-mode demand-narrowed read set (FD §6.1
    1280-1282); the derived label field itself is NOT read from disk.

    With ``require_labels: true`` (v1 GN3X) the labeller RAISES (ftag
    ``labeller.py:70``) on any object that matches no class; with
    ``require_labels: false`` (v1 GN2X_qcdsplit) unmatched objects are dropped
    (``labeller.py:73``), so the derived label array may be shorter than the
    batch — exactly v1's ``self.labeller.get_labels(batch)`` behaviour
    (``datasets.py:626``).

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
    use_labeller : bool, optional
        Enable the on-the-fly ftag `Labeller` (v1 ``LabellerConfig.use_labeller``).
        Requires ``class_names`` (the empty-class guard, v1 ``configs.py:189``).
        By default False (the labeller is OFF; ``labeller_*`` args are ignored).
    class_names : Sequence[str] | None, optional
        Target ftag flavour class names, in label-index order (v1
        ``LabellerConfig.class_names``; GN3X 9-class, GN2X_qcdsplit 7-class).
        Required when ``use_labeller`` is set.
    require_labels : bool, optional
        Whether every object must be labelled (v1
        ``LabellerConfig.require_labels``): True raises on an unlabelled
        object, False drops it. By default True. Ignored when
        ``use_labeller`` is False.
    labeller_stream : str, optional
        The stream the labeller relabels (v1's implicit ``global_object``;
        default ``"jets"``). The labeller's cut variables are demanded from
        ``raw.<labeller_stream>``.
    labeller_label : str, optional
        The label key the labeller produces (v1's implicit ``"flavour_label"``;
        default ``"flavour_label"``). Only ``labels.<labeller_stream>.
        <labeller_label>`` is derived on the fly; every other demanded label is
        read from the file as before.

    Raises
    ------
    ConfigError
        On an unknown ``dtype_policy``, malformed ``valid_ranges``, or
        ``use_labeller`` set without ``class_names`` (the empty-class guard,
        v1 ``configs.py:189``). NOTE the deliberate exception-TYPE promotion: v1's
        empty-class guard raises a bare ``ValueError`` (``configs.py:189``); v2
        standardises structural config-validation failures on ``ConfigError`` (the
        framework's named config-error type, NOT a ``ValueError`` subclass — see
        ``graph/errors.py``), as every M1-M5 processor does. The other two
        labeller guards (missing-field, require_labels-on-unlabelled) keep v1's
        ``ValueError`` (raised at ``process`` time, not construction). An M7
        v1->v2 converter wiring v1 exception expectations should map the v1
        ``LabellerConfig`` ``ValueError`` contract to ``ConfigError``.
    """

    allow_wildcards: ClassVar[bool] = True  # framework wildcard capability (design §2.2)

    def __init__(
        self,
        streams: Sequence[str] | None = None,
        dtype_policy: Literal["int64-for-int", "file"] = "int64-for-int",
        valid_ranges: Mapping[str, Sequence[int]] | None = None,
        recover_malformed: bool = False,
        use_labeller: bool = False,
        class_names: Sequence[str] | None = None,
        require_labels: bool = True,
        labeller_stream: str = "jets",
        labeller_label: str = "flavour_label",
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
        # -- on-the-fly Labeller (M6 sub-wave A) -------------------------------
        # mirror v1 LabellerConfig (configs.py:154-197): use_labeller off -> the
        # labeller args are inert (v1 __post_init__ zeroes them, configs.py:178-185);
        # use_labeller on -> the empty-class guard (v1 configs.py:189) MUST fire.
        self.use_labeller = bool(use_labeller)
        self.labeller_stream = str(labeller_stream)
        self.labeller_label = str(labeller_label)
        self.labeller: Labeller | None = None
        self.class_names: tuple[str, ...] = ()
        self.require_labels = bool(require_labels)
        if self.use_labeller:
            if not class_names:
                raise ConfigError(
                    f"Labels module {self.name!r}: use_labeller is True but class_names is empty — "
                    "specify the target classes for relabelling (v1 LabellerConfig empty-class "
                    "guard, configs.py:189)"
                )
            self.class_names = tuple(class_names)
            # the ftag Labeller IS the parity reference (ftag/labeller.py); v1
            # builds it identically (Labeller(class_names, require_labels),
            # datasets.py:205)
            self.labeller = Labeller(list(self.class_names), self.require_labels)
        elif class_names:
            warnings.warn(
                f"Labels module {self.name!r}: class_names is set but use_labeller is False — "
                "the labeller config is ignored (v1 configs.py:178-185)",
                stacklevel=2,
            )
            self.require_labels = False
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

    def _is_labeller_target(self, stream: str, label: str) -> bool:
        """Whether ``(stream, label)`` is the on-the-fly labeller's output key.

        Returns
        -------
        bool
            True when the labeller is active and this key is the configured
            ``labeller_stream.labeller_label`` (the v1 gate
            ``input_name == global_object and label == 'flavour_label'``,
            ``datasets.py:616-617``).
        """
        return (
            self.labeller is not None
            and stream == self.labeller_stream
            and label == self.labeller_label
        )

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
        assert self.labeller is not None
        return tuple(dict.fromkeys(self.labeller.variables))

    def read_fields(self, step: PlanStep) -> dict[str, dict[str, str]]:
        """Demand exactly the narrowed label fields from the reader (design §6.1).

        For an ordinary label the demanded field IS the label name. For the
        on-the-fly labeller target the label field is NOT on disk (it is
        derived), so instead the labeller's cut variables (``Labeller.
        variables``, deduped) are demanded from ``raw.<labeller_stream>`` — the
        explicit "declare the extra read fields" rule (FD §6.1 1280-1282) that
        lets them enter the per-mode demand-narrowed read set.

        Returns
        -------
        dict[str, dict[str, str]]
            ``{stream: {field: this module}}`` from the narrowed produces —
            the wildcard-producer override of the default requires-fields rule.
        """
        out: dict[str, dict[str, str]] = {}
        for _key, stream, label in self._parse_targets(step):
            if self._is_labeller_target(stream, label):
                for var in self._labeller_variables():
                    out.setdefault(stream, {}).setdefault(var, self.name)
            else:
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
            ``recover_malformed`` is off (v1 ``datasets.py:832-836``), or — for
            the on-the-fly labeller target — when a labeller cut variable is
            missing from the raw stream (v1 missing-field guard,
            ``datasets.py:622-624``) or an object is unlabelled under
            ``require_labels`` (ftag ``labeller.py:70``).
        """
        del rows, mode
        assert self._targets is not None, "Labels.process called before bind()"
        out: dict[str, np.ndarray] = {}
        for key, stream, label in self._targets:
            if self._is_labeller_target(stream, label):
                out[key] = self._derive_labeller(batch.get(f"raw.{stream}"), stream)
                continue
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

    def _derive_labeller(self, raw: np.ndarray, stream: str) -> np.ndarray:
        """Derive the on-the-fly labeller labels from the raw structured array.

        Mirrors v1 ``process_labels`` (``datasets.py:619-626``): the
        missing-field guard (every ``Labeller.variables`` cut variable must be
        present, ``datasets.py:622-624``) then ``Labeller.get_labels`` on the
        WHOLE structured array, cast to int64 under ``dtype_policy``. The
        ``get_labels`` output is a fresh array (``labeller.py:66-73``), so it
        never aliases the reader buffer.

        Returns
        -------
        np.ndarray
            The derived int64 labels (or file dtype under
            ``dtype_policy='file'``). With ``require_labels=False`` unmatched
            objects are dropped, so the array may be shorter than the batch
            (ftag ``labeller.py:73``; v1 parity). This is faithful v1 parity AT
            THE LABEL-DERIVATION LEVEL (LB1's scope): v1 likewise does NOT
            row-filter the inputs/other-label columns to the dropped subset
            (``datasets.py`` ``process_labels`` has no such filter), so the v1
            bundle is itself length-mismatched (inputs vs flavour_label) under
            ``require_labels=False``. The full-batch length-coherence reconciliation
            is NOT exercised here (sub-wave A is data-free) and belongs to the M7
            data-path wave — a data-bearing GN2X_qcdsplit forward/integration test
            must decide whether to row-filter the whole bundle or to confirm v1's
            length-mismatch is handled identically, and record it as a Key Decision.

        Raises
        ------
        ValueError
            If a labeller cut variable is absent from the raw stream (the v1
            field-subset check, ``datasets.py:622-624``), or — under
            ``require_labels`` — if any object matches no class (ftag
            ``labeller.py:70``).
        """
        assert self.labeller is not None
        present = set(raw.dtype.names or ())
        missing = [var for var in self._labeller_variables() if var not in present]
        if missing:
            raise ValueError(
                f"Labels module {self.name!r}: not enough fields to apply labelling cuts on "
                f"stream {stream!r} — missing labeller variables {missing} (v1 field-subset "
                "check, datasets.py:622-624)"
            )
        # get_labels raises under require_labels on an unlabelled object
        # (labeller.py:70) and otherwise drops it (labeller.py:73) — v1 parity.
        derived = self.labeller.get_labels(raw)
        if self.dtype_policy == "int64-for-int" and np.issubdtype(derived.dtype, np.integer):
            return derived.astype(np.int64)
        return np.array(derived, copy=True)


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
    class_map : Mapping[str, Mapping[str, int]]
        ``{name: {raw: int, mapped: int}}`` (v1 ``object.object_classes``). MUST
        contain a ``null`` entry mapped LAST (``mapped == len(class_map) - 1``),
        and the ``mapped`` values MUST be exactly ``range(len(class_map))`` (v1
        configs.py:39-42). jsonargparse may parse the YAML ``null:`` key as a
        Python ``None`` — both spellings are accepted, normalised to ``"null"``.
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
        decoder's query bank); None leaves ``M`` symbolic.

    Raises
    ------
    ConfigError
        On a missing ``null`` class, a null not mapped last, ``mapped`` values
        that are not ``range(len(class_map))``, a duplicate regression target,
        or a non-positive ``num_objects``.
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
    ) -> None:
        super().__init__()
        self.object_class = str(object_class)
        self.object_id = str(object_id)
        self.constituent_id = str(constituent_id)
        self.object_stream = str(object_stream)
        self.constituent_stream = str(constituent_stream)
        self._raw_to_mapped = self._checked_class_map(class_map)
        self.regression_targets: tuple[str, ...] = tuple(regression_targets or ())
        if len(set(self.regression_targets)) != len(self.regression_targets):
            raise ConfigError(
                f"MaskFormerTargets: duplicate regression targets in {self.regression_targets}"
            )
        if num_objects is not None and num_objects < 1:
            raise ConfigError(f"MaskFormerTargets: num_objects must be >= 1, got {num_objects}")
        self.num_objects = num_objects

    @staticmethod
    def _checked_class_map(class_map: Mapping[str, Mapping[str, int]]) -> dict[int, int]:
        """Validate the class map (null LAST, mapped == range) and return raw->mapped.

        Reproduces the v1 ``MaskformerObjectConfig`` invariants (configs.py:36-42)
        WITHOUT mutating the loaded ids: ``null`` present (``None`` or the string
        ``"null"`` accepted, jsonargparse may cast the YAML ``null:`` key to
        ``None``), null mapped LAST, and the ``mapped`` set exactly
        ``range(len(class_map))``.

        Returns
        -------
        dict[int, int]
            ``{raw: mapped}`` for every class.

        Raises
        ------
        ConfigError
            On a missing null, a null not mapped last, or a malformed mapped set.
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
            # the null class's raw id defaults to -1 (v1 MaskFormer.yaml:177: null raw -1)
            raw_to_mapped[int(spec.get("raw", -1))] = int(spec["mapped"])
        if names["null"]["mapped"] != n - 1:
            raise ConfigError(
                f"MaskFormerTargets: the 'null' class must be mapped LAST (to {n - 1}), got "
                f"{names['null']['mapped']} (v1 configs.py:39 'Null class must be last')"
            )
        if set(raw_to_mapped.values()) != set(range(n)):
            raise ConfigError(
                f"MaskFormerTargets: mapped class indices {sorted(raw_to_mapped.values())} must be "
                f"exactly range({n}) (v1 configs.py:42)"
            )
        return raw_to_mapped

    @property
    def null_index(self) -> int:
        """The mapped index of the null/no-object class (== num_classes).

        Returns
        -------
        int
            ``len(class_map) - 1`` — the matcher's ``num_classes`` sentinel.
        """
        return len(self._raw_to_mapped) - 1

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
        obj_fields = (self.object_class, self.object_id, *self.regression_targets)
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

        # object_class: map raw -> mapped from the ORIGINAL values (atomic, no
        # sequential x[x==k]=v mutation; v1 datasets.py:641-643). Unmapped raw
        # values fall through to the null index (v1 reads only configured classes,
        # and any object whose raw class is not in the map is a no-object slot).
        raw_class = np.asarray(obj[self.object_class])
        conds = [raw_class == raw for raw in self._raw_to_mapped]
        choices = [self._raw_to_mapped[raw] for raw in self._raw_to_mapped]
        out[f"labels.{_OBJECT_STREAM}.object_class"] = np.select(
            conds, choices, default=self.null_index
        ).astype(np.int64)

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
