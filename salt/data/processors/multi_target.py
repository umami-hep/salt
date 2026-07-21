"""`MultiTarget` — conditional row-wise target replacement."""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from salt.core.data.base import Processor
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import IO, Mode, TensorSpec, unflatten_spec

# v1 operators — the conditional-replacement comparators, applied to the
# selection label against the configured value.
_OPERATORS: dict[str, Callable[[Any, Any], Any]] = {
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}


class MultiTarget(Processor):
    """Conditional row-wise target replacement.

    For each configured rule, the per-row value of an output label is
    replaced by a ``source`` label wherever a ``sel_label`` satisfies an
    operator comparison against a literal ``value`` —
    ``np.where(op(sel, value), source, running)``. Two output modes:

    - ``target:`` — REPLACE an existing label. The base (pre-replacement)
      values are read from the raw stream (the column the task would
      otherwise consume directly), so this processor becomes the SOLE
      producer of ``labels.<stream>.<target>`` (a concrete produce beats
      the `Labels` wildcard) — write-once is preserved with no
      two-producer conflict.
    - ``custom_target:`` — CREATE a NEW label initialised to a NaN
      placeholder, then fill it where the condition holds.

    Multiple rules MAY name the same output — they apply SEQUENTIALLY over
    a running array. The shipped ``regression_multi_target.yaml`` uses
    this: two rules both write ``pt_label_handle`` (one ``ID==15``, one
    ``ID!=15``). All rules for one output must agree on the mode (all
    ``custom_target`` or all ``target``) — the base is established once
    (NaN placeholder or the raw column) and each rule layers a
    ``np.where`` on top.

    Each rule declares its own ``labels.<stream>.<sel_label>`` and
    ``labels.<stream>.<source>`` dependencies (produced by `Labels`, so the
    sel/source casting policy stays in ONE place). A ``sel_label``/``source``
    may not be a MultiTarget output (no producer->producer chaining).

    Parameters
    ----------
    replacements : Sequence[Mapping[str, Any]]
        Ordered replacement rules. Each rule is a mapping with keys:

        - ``stream`` — the labelled stream;
        - ``sel_label`` — the selection label compared against ``value``;
        - ``op`` — one of ``== != >= <= > <``;
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
        """Validate one replacement rule and normalise it to a flat dict."""
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
        """Declare ``labels.<s>.{sel,source}`` (+ raw target for ``target:`` outputs) ->
        ``labels.<s>.<out>``, TRAINING-only; ``custom_target:`` outputs need no base column.
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
        """Apply each conditional replacement: keep ``source`` where ``op(sel, value)``
        holds, else the running value, established once per output then layered per rule.
        """
        del rows, mode
        running: dict[tuple[str, str], np.ndarray] = {}
        for (stream, output), is_custom in self._outputs.items():
            if is_custom:
                # v1 inject_custom_target_placeholders: a NaN-filled column
                # shaped/typed like the first rule's source.
                # DEVIATION from v1: v1 uses dtype=batch[source].dtype verbatim;
                # v2 promotes any sub-float32 source (e.g. f2) to >=float32 via
                # np.result_type so a NaN-filled regression placeholder always has
                # the range to hold log/ratio targets. For f4/f8 sources the two
                # agree byte-for-byte (the only dtypes any shipped
                # regression_multi_target.yaml source uses — HadronConeExclTruthLabelPt
                # and pt are both f4); the divergence is reachable only with an f2
                # source, which no shipped config has.
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
