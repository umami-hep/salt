"""`Features` — ``raw.* -> inputs.*`` float32 materialisation."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as s2u

from salt.data.base import Processor
from salt.graph.errors import ConfigError
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec


class Features(Processor):
    """``raw.* -> inputs.*`` float32 materialisation.

    THE documented one-copy-per-batch aliasing boundary: the
    ``structured_to_unstructured`` conversion is the mandatory copy that
    separates reusable reader buffers from anything handed to the trainer,
    enforced here with an explicit ``may_share_memory`` guard.

    Column order = the configured list order — the ONE place column order is
    defined; the produced specs carry ``fields`` metadata so downstream
    column lookups resolve by name. Applied in order: ``s2u`` ->
    ``nan_to_num`` (optional) -> zero padded rows via the pad mask -> finite
    check.

    Parameters
    ----------
    variables : Mapping[str, Sequence[str]]
        Stream name -> ordered input variable list.
    non_finite_to_num : bool, optional
        Convert NaN/inf to zero before masking.
    ignore_finite_checks : bool, optional
        Warn instead of raising on non-finite inputs.

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
            raise ConfigError("Features needs at least one stream in 'variables'")
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
        """Declare ``raw.<s>`` (+ optional ``masks.<s>``) -> ``inputs.<s>`` per stream; the
        mask require is optional (``global_object`` streams have no mask producer).
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
        """Materialise float32 input arrays: ``s2u`` -> optional ``nan_to_num`` -> zero
        padded rows -> finite check (raises `ValueError` unless `ignore_finite_checks`).
        """
        del rows, mode
        out: dict[str, np.ndarray] = {}
        for stream, names in self.variables.items():
            raw = batch.get(f"raw.{stream}")
            # column order = config list order (structured multi-field indexing reorders)
            flat = s2u(raw[names], dtype=np.float32)
            if np.may_share_memory(flat, raw):
                # the mandatory copy: s2u may return a view for uniform
                # layouts — never hand a buffer alias downstream
                flat = flat.copy()
            if self.non_finite_to_num:
                flat = np.nan_to_num(flat, posinf=0, neginf=0)
            mask_key = f"masks.{stream}"
            if mask_key in batch:
                flat[batch.get(mask_key)] = 0.0  # zero padded rows
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
