"""Normaliser and MaskedInputNormaliser GraphModules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import (
    IO,
    Mode,
    TensorSpec,
    sym_dim,
    unflatten_spec,
)
from salt.model.base import SaltModelModule
from salt.model.bind import ResolvedSchema
from salt.model.modules.stream_embed import _stream_len


def _checked_streams(streams: Sequence[str], who: str, what: str = "streams") -> tuple[str, ...]:
    """``tuple(streams)``; `ConfigError` ``"<who>: ..."`` when empty or duplicated."""
    if not streams:
        raise ConfigError(f"{who}: {what} must be a non-empty sequence")
    if len(set(streams)) != len(tuple(streams)):
        raise ConfigError(f"{who}: duplicate {what} in {tuple(streams)}")
    return tuple(streams)


def _norm_specs(
    prefix: str, name: str, streams: Sequence[str], global_object: str | None
) -> dict[str, TensorSpec]:
    """``{<prefix>.<stream>: spec}``: global ``(B, F)`` or sequence ``(B, T, F)``.

    The last dim is the instance-scoped symbol ``F:<name>.<stream>`` on both
    ``inputs`` and ``normed``, so the dataset-declared width unifies onto ``normed``.
    """
    specs: dict[str, TensorSpec] = {}
    for s in streams:
        width = sym_dim("F", f"{name}.{s}")
        specs[f"{prefix}.{s}"] = TensorSpec(
            shape=("B", width) if s == global_object else ("B", _stream_len(s), width),
            dtype="float32",
        )
    return specs


def _normed_fields(
    streams: Sequence[str], fields: Mapping[str, tuple[str, ...]]
) -> dict[str, tuple[str, ...]]:
    """Propagate ``inputs.<stream>`` column names to ``normed.<stream>`` (column-preserving)."""
    return {f"normed.{s}": fields[f"inputs.{s}"] for s in streams if f"inputs.{s}" in fields}


def _via(stream: str, key: str) -> str:
    """``" (stream '<stream>')"`` when the dataset key differs from the stream, else ``""``."""
    return f" (stream {stream!r})" if key != stream else ""


def _norm_entries(
    table: Mapping[str, Any], path: Path, stream: str, key: str, variables: Sequence[str], who: str
) -> tuple[Tensor, Tensor]:
    """Checked float32 ``(means, stds)`` of `variables` from ``norm_dict[key]`` (`ConfigError`).

    Finiteness and zero-std are checked on the float32 values the buffers
    hold, so a std that underflows float32 is a zero std.
    """
    if missing := set(variables) - set(table):
        raise ConfigError(
            f"{who}: missing variables {sorted(missing)} for {key!r}{_via(stream, key)} in "
            f"{path}. Choose from {sorted(table)}.\n"
            f"  fix: add mean/std entries for {sorted(missing)} to {path}, or remove them from "
            f"the features variable list "
            f"(config: data.modules.features.init_args.variables.{stream})"
        )
    pairs: list[tuple[float, float]] = []
    for variable in variables:
        entry = table[variable]
        try:
            pairs.append((float(entry["mean"]), float(entry["std"])))
        except (KeyError, TypeError, ValueError):
            raise ConfigError(
                f"{who}: entry for {stream}.{variable} in {path} must be a "
                f"{{mean, std}} mapping, got {entry!r}"
            ) from None
    mean_t, std_t = (torch.as_tensor([p[i] for p in pairs], dtype=torch.float32) for i in (0, 1))
    for bad, what in (
        (~(torch.isfinite(mean_t) & torch.isfinite(std_t)), "non-finite normalisation parameters"),
        (std_t == 0, "zero standard deviation"),
    ):
        if bad.any():
            names = [v for v, b in zip(variables, bad.tolist(), strict=True) if b]
            raise ConfigError(f"{who}: {what} for {stream} variables {names} in {path}.")
    return mean_t, std_t


class Normaliser(SaltModelModule):
    """Config-constructed input normalisation.

    The DEFAULT input normaliser: loads a precomputed ``norm_dict.yaml``
    (fixed per-stream, per-variable ``{mean, std}``). For a self-normalising
    variant that learns statistics online (no norm dict), use
    ``class_path: salt.model.modules.MaskedInputNormaliser`` instead.

    ``materialise()`` is the only file-touching hook (loads the norm dict and
    fills the buffers) — skipped on checkpoint load, where values arrive via
    the state_dict. Produces NEW ``normed.<stream>`` keys; never mutates
    ``inputs.*``. Norm-dict entries are looked up by the reader's dataset name
    for each stream (`ResolvedSchema.dataset_of`), falling back to the stream
    name.
    """

    def __init__(
        self,
        norm_dict: str | Path,
        streams: Sequence[str],
        global_object: str | None = None,
    ) -> None:
        """Capture config only (no file I/O here).

        Parameters
        ----------
        norm_dict : str | Path
            Path to the normalisation dictionary YAML; read at `materialise`, never here.
        streams : Sequence[str]
            Streams to normalise.
        global_object : str | None, optional
            The stream that is a per-object vector (``[B, F]``) rather than
            a padded sequence (``[B, T, F]``), by default None.

        Raises
        ------
        ConfigError
            If `streams` is empty, contains duplicates, or `global_object`
            is not one of them.
        """
        super().__init__()
        self.streams = _checked_streams(streams, "Normaliser")
        if global_object is not None and global_object not in streams:
            raise ConfigError(
                f"Normaliser: global_object {global_object!r} is not in streams {self.streams}"
            )
        self.norm_dict_path = Path(norm_dict)
        self.global_object = global_object
        self._fields: dict[str, tuple[str, ...]] = {}
        self._datasets: dict[str, str] = {}
        self._bound = False
        # python mirror of the `materialised` buffer, read by forward. Reading
        # the buffer there costs a `.item()`, which is a graph break under
        # torch.compile on a module that runs once per step; a plain bool is a
        # dynamo guard instead. Kept in step in `bind`, `materialise` and
        # `_load_from_state_dict` — the only three places the buffer changes.
        self._materialised_flag = False

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``inputs.<stream>`` -> ``normed.<stream>`` for every stream."""
        del mode
        args = (self.name, self.streams, self.global_object)
        return IO(
            requires=unflatten_spec(_norm_specs("inputs", *args)),
            produces=unflatten_spec(_norm_specs("normed", *args)),
        )

    def derived_fields(self, fields: Mapping[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
        return _normed_fields(self.streams, fields)

    def bind(self, schema: ResolvedSchema) -> None:
        """Allocate normalisation buffers (means/stds per stream); raises if bound twice."""
        if self._bound:
            raise RuntimeError(f"Normaliser {self.name!r}: bind() called twice")
        for stream in self.streams:
            key = f"inputs.{stream}"
            width = schema.width(key)
            self._fields[stream] = schema.fields_of(key)
            self._datasets[stream] = schema.dataset_of(stream)
            self.register_buffer(f"means_{stream}", torch.zeros(width))
            self.register_buffer(f"stds_{stream}", torch.ones(width))
        self.register_buffer("materialised", torch.tensor(False))
        self._materialised_flag = False
        self._bound = True

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Re-sync the python `materialised` mirror after the buffers are loaded."""
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        flag = getattr(self, "materialised", None)
        if flag is not None:
            self._materialised_flag = bool(flag)

    def preflight(self, datasets: Mapping[str, str] | None = None) -> None:
        """Fail-fast, data-free norm-dict validation.

        Reads ONLY the norm-dict YAML: the file must exist, parse, and carry
        every configured stream; when already bound, per-variable mean/std
        entries are checked too, in the float32 `materialise` stores. Called
        by `SaltModule.setup` on fresh fits (hard error) and by ``salt graph
        validate`` (warning).

        Parameters
        ----------
        datasets : Mapping[str, str] | None, optional
            Stream->dataset map for the unbound validate path; bound modules
            use the map captured at `bind`.

        Raises
        ------
        ConfigError
            On a missing/unparsable norm dict, a missing stream, or (when
            bound) missing/malformed/non-finite/zero-std variable entries.
        """
        path, who = self.norm_dict_path, f"Normaliser {self.name!r} preflight"
        fix = (
            f"  fix: point model.modules.{self.name}.init_args.norm_dict at the "
            "preprocessing norm_dict.yaml for this sample"
        )
        if not path.is_file():
            raise ConfigError(f"{who}: norm dict not found: {path}\n{fix}")
        try:
            with open(path) as fh:
                norm_dict = yaml.safe_load(fh)
        except yaml.YAMLError as err:
            raise ConfigError(f"{who}: norm dict {path} is not valid YAML: {err}\n{fix}") from err
        if not isinstance(norm_dict, dict):
            raise ConfigError(f"{who}: norm dict {path} must be a mapping\n{fix}")
        lookup = self._datasets if datasets is None else datasets
        for stream in self.streams:
            key = lookup.get(stream, stream)
            if key not in norm_dict:
                raise ConfigError(
                    f"{who}: missing input type {key!r}{_via(stream, key)} in {path}. "
                    f"Choose from {sorted(norm_dict)}."
                )
            if self._fields:  # bound; the unbound `salt graph validate` path checks streams only
                _norm_entries(norm_dict[key], path, stream, key, self._fields[stream], who)

    def materialise(self) -> None:
        """Fill the buffers from the norm dict (the only file I/O this module does).

        Per-variable entries are parsed and checked as in `preflight`
        (`ConfigError`), stream by stream.

        Raises
        ------
        RuntimeError
            If called before `bind`.
        ValueError
            If the norm dict is missing one of this module's streams.
        """
        if not self._bound:
            raise RuntimeError(f"Normaliser {self.name!r}: materialise() before bind()")
        path, who = self.norm_dict_path, f"Normaliser {self.name!r} materialise"
        with open(path) as fh:
            norm_dict = yaml.safe_load(fh)
        for stream in self.streams:
            key = self._datasets.get(stream, stream)
            if key not in norm_dict:
                raise ValueError(
                    f"Missing input type {key!r}{_via(stream, key)} in {path}. "
                    f"Choose from {sorted(norm_dict)}."
                )
            means, stds = _norm_entries(
                norm_dict[key], path, stream, key, self._fields[stream], who
            )
            with torch.no_grad():
                getattr(self, f"means_{stream}").copy_(means)
                getattr(self, f"stds_{stream}").copy_(stds)
        self.materialised.fill_(True)
        self._materialised_flag = True

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """``normed.<stream> = (inputs.<stream> - means) / stds``; raises if never materialised."""
        del mode
        # skip under tracing: the tensor->bool read would emit a spurious
        # TracerWarning on every export. Tracing is still guarded —
        # OnnxAdapter rejects unmaterialised modules at construction, before
        # any trace (the eager path keeps this check). The python mirror rather
        # than the buffer, so torch.compile does not break here on `.item()`.
        if not torch.jit.is_tracing() and not self._materialised_flag:
            raise RuntimeError(
                f"Normaliser {self.name!r}: forward before materialise() — on a fresh fit "
                "call materialise(); on checkpoint load the state_dict provides the values"
            )
        return {
            f"normed.{s}": (b.get(f"inputs.{s}") - getattr(self, f"means_{s}"))
            / getattr(self, f"stds_{s}")
            for s in self.streams
        }


class MaskedInputNormaliser(SaltModelModule):
    """Self-normalising input layer with online masked running statistics.

    OPT-IN alternative to the fixed-norm-dict `Normaliser`: learns mean/var
    online from valid (non-padded) objects, no file I/O. ALWAYS applies the
    frozen running buffers (even in train mode); the buffers are updated from
    masked batch moments separately, only when training and off tracing —
    so eval/inference/ONNX is a pure affine transform with no mask
    dependency.
    """

    def __init__(
        self,
        streams: Sequence[str],
        global_object: str | None = None,
        momentum: float | None = 0.1,
        eps: float = 1e-5,
    ) -> None:
        """Capture config only (no file I/O here).

        ``momentum=None`` gives a cumulative moving average instead of an EMA.
        """
        super().__init__()
        self.streams = _checked_streams(streams, "MaskedInputNormaliser")
        if global_object is not None and global_object not in streams:
            raise ConfigError(
                f"MaskedInputNormaliser: global_object {global_object!r} is not in "
                f"streams {self.streams}"
            )
        if momentum is not None and not 0.0 <= momentum <= 1.0:
            raise ConfigError(
                f"MaskedInputNormaliser: momentum must be None or in [0, 1], got {momentum}"
            )
        if eps <= 0.0:
            raise ConfigError(f"MaskedInputNormaliser: eps must be positive, got {eps}")
        self.global_object = global_object
        self.momentum = None if momentum is None else float(momentum)
        self.eps = float(eps)
        self._bound = False

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``inputs.<stream>`` (+ optional TRAINING-only ``masks``) -> ``normed``."""
        del mode
        args = (self.name, self.streams, self.global_object)
        requires = _norm_specs("inputs", *args)
        for stream in self.streams:
            if stream != self.global_object:
                requires[f"masks.{stream}"] = TensorSpec(
                    shape=("B", _stream_len(stream)),
                    dtype="bool",
                    kind="pad_mask",
                    modes=Mode.TRAINING,
                    optional=True,
                )
        return IO(
            requires=unflatten_spec(requires),
            produces=unflatten_spec(_norm_specs("normed", *args)),
        )

    def derived_fields(self, fields: Mapping[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
        return _normed_fields(self.streams, fields)

    def bind(self, schema: ResolvedSchema) -> None:
        """Allocate the running-stat buffers (identity init); raises if bound twice."""
        if self._bound:
            raise RuntimeError(f"MaskedInputNormaliser {self.name!r}: bind() called twice")
        for stream in self.streams:
            width = schema.width(f"inputs.{stream}")
            self.register_buffer(f"running_mean_{stream}", torch.zeros(width))
            self.register_buffer(f"running_var_{stream}", torch.ones(width))
            self.register_buffer(f"num_batches_tracked_{stream}", torch.zeros((), dtype=torch.long))
            # total VALID-object count seen (cumulative-averaging path only)
            self.register_buffer(f"num_objects_seen_{stream}", torch.zeros((), dtype=torch.long))
        self._bound = True

    @torch.no_grad()
    def _update_running_stats(self, stream: str, x: Tensor, valid: Tensor | None) -> None:
        """Update one stream's running stats from masked batch moments.

        ``valid=None`` treats every row as valid. All-reduces sum/sumsq/count
        across DDP ranks before deriving batch mean/var (never averages
        per-rank mean/var directly). No-op when nothing is valid (count 0).
        """
        flat = x.reshape(-1, x.shape[-1]) if valid is None else x[valid]
        count = torch.tensor(flat.shape[0], dtype=x.dtype, device=x.device)
        s_sum, s_sumsq = flat.sum(0), (flat * flat).sum(0)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            packed = torch.cat([s_sum, s_sumsq, count.reshape(1)])
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
            f = s_sum.shape[0]
            s_sum, s_sumsq, count = packed[:f], packed[f : 2 * f], packed[2 * f]
        if count.item() == 0:
            return  # all-padded batch (global) — nothing valid to learn from
        batch_mean = s_sum / count
        # population (biased) variance over valid objects: E[x^2] - E[x]^2,
        # clamped to >= 0 against tiny float negatives.
        batch_var = (s_sumsq / count - batch_mean * batch_mean).clamp_min(0.0)
        getattr(self, f"num_batches_tracked_{stream}").add_(1)
        running_mean = getattr(self, f"running_mean_{stream}")
        running_var = getattr(self, f"running_var_{stream}")
        if self.momentum is None:
            # cumulative moving average: object-count-weighted parallel moment
            # combination (Chan et al.) — converges to the exact pooled mean/var
            seen = getattr(self, f"num_objects_seen_{stream}")
            n_old = seen.to(batch_mean.dtype)
            n_new = n_old + count
            delta = batch_mean - running_mean
            running_mean.copy_(running_mean + delta * (count / n_new))
            m_old, m_new = running_var * n_old, batch_var * count
            running_var.copy_((m_old + m_new + delta * delta * (n_old * count / n_new)) / n_new)
            seen.add_(count.long())
        else:
            running_mean.mul_(1 - self.momentum).add_(batch_mean, alpha=self.momentum)
            running_var.mul_(1 - self.momentum).add_(batch_var, alpha=self.momentum)

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Apply frozen running-stat normalisation; updates stats only when training (untraced)."""
        if self.training and bool(mode & Mode.TRAINING) and not torch.jit.is_tracing():
            for stream in self.streams:
                # the global object and a mask-less stream (optional port with
                # no producer) count every row valid; otherwise ~pad_mask
                # (mask True == padded) selects the valid rows
                mask_key = f"masks.{stream}"
                no_mask = stream == self.global_object or mask_key not in b
                valid = None if no_mask else ~b.get(mask_key)
                self._update_running_stats(stream, b.get(f"inputs.{stream}"), valid)
        return {
            f"normed.{s}": (b.get(f"inputs.{s}") - getattr(self, f"running_mean_{s}"))
            / torch.sqrt(getattr(self, f"running_var_{s}") + self.eps)
            for s in self.streams
        }
