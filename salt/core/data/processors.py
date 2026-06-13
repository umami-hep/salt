"""Shipped dataset processors: `Features`, `Labels`, `MultiTarget` (design §6.2).

Each replaces an if-branch of the v1 ``SaltDataset.__getitem__`` god-loop
(``datasets.py:417-559``). `MultiTarget` (M5 sub-wave A3) ports v1's
conditional target replacement (``datasets.py:237-248,648,695-739``). Further
v1 branches (``Parameters``, ``MaskFormerTargets``) land with their workloads
(TODO(M5/M6) per design §9.5) — the structure here is the template.
"""

from __future__ import annotations

import operator
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as s2u

from salt.core.data.base import Processor, WorkerCtx
from salt.core.graph.errors import ConfigError
from salt.core.graph.planner import PlanStep
from salt.core.graph.spec import IO, KEY_SEP, Mode, TensorSpec, unflatten_spec

__all__ = ["Features", "Labels", "MultiTarget"]

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

        The mask require is optional: ``vector`` streams have no mask
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

    TODO(M6): the on-the-fly ftag ``Labeller`` opt-in (design §6.2).

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

        Returns
        -------
        dict[str, dict[str, str]]
            ``{stream: {label_field: this module}}`` from the narrowed
            produces — the wildcard-producer override of the default
            requires-fields rule.
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
