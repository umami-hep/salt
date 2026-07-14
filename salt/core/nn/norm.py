"""Normaliser and MaskedInputNormaliser GraphModules."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import yaml
from torch import Tensor

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import (
    IO,
    Mode,
    TensorSpec,
    sym_dim,
    unflatten_spec,
)
from salt.core.nn.base import SaltModelModule
from salt.core.nn.bind import ResolvedSchema
from salt.core.nn.stream_embed import _stream_len


class Normaliser(SaltModelModule):
    """Config-constructed input normalisation.

    The DEFAULT input normaliser: loads a precomputed ``norm_dict.yaml``
    (fixed per-stream, per-variable ``{mean, std}``). For a self-normalising
    variant that learns statistics online (no norm dict), use
    ``class_path: salt.core.nn.MaskedInputNormaliser`` instead.

    ``materialise()`` is the only file-touching hook (loads the norm dict and
    fills the buffers) — skipped on checkpoint load, where values arrive via
    the state_dict. Produces NEW ``normed.<stream>`` keys; never mutates
    ``inputs.*``.
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
        if not streams:
            raise ConfigError("Normaliser: streams must be a non-empty sequence")
        if len(set(streams)) != len(tuple(streams)):
            raise ConfigError(f"Normaliser: duplicate streams in {tuple(streams)}")
        if global_object is not None and global_object not in streams:
            raise ConfigError(
                f"Normaliser: global_object {global_object!r} is not in streams {tuple(streams)}"
            )
        self.norm_dict_path = Path(norm_dict)
        self.streams = tuple(streams)
        self.global_object = global_object
        self._fields: dict[str, tuple[str, ...]] = {}
        self._bound = False

    def _spec(self, stream: str) -> TensorSpec:
        """Shared spec for ``inputs``/``normed``: global ``(B, F)`` or sequence ``(B, T, F)``."""
        width = sym_dim("F", f"{self.name}.{stream}")
        shape: tuple[int | str, ...] = (
            ("B", width) if stream == self.global_object else ("B", _stream_len(stream), width)
        )
        return TensorSpec(shape=shape, dtype="float32")

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``inputs.<stream>`` -> ``normed.<stream>`` for every stream."""
        del mode
        return IO(
            requires=unflatten_spec({f"inputs.{s}": self._spec(s) for s in self.streams}),
            produces=unflatten_spec({f"normed.{s}": self._spec(s) for s in self.streams}),
        )

    def bind(self, schema: ResolvedSchema) -> None:
        """Allocate normalisation buffers (means/stds per stream); raises if bound twice."""
        if self._bound:
            raise RuntimeError(f"Normaliser {self.name!r}: bind() called twice (design §2.3)")
        for stream in self.streams:
            key = f"inputs.{stream}"
            width = schema.width(key)
            self._fields[stream] = schema.fields_of(key)
            self.register_buffer(f"means_{stream}", torch.zeros(width))
            self.register_buffer(f"stds_{stream}", torch.ones(width))
        self.register_buffer("materialised", torch.tensor(False))
        self._bound = True

    def preflight(self) -> None:
        """Fail-fast, data-free norm-dict validation.

        Reads ONLY the norm-dict YAML: the file must exist, parse, and carry
        every configured stream; when already bound, per-variable mean/std
        entries are checked too. Called by `SaltModule.setup` on fresh fits
        (hard error) and by ``salt2 graph validate`` (warning).

        Raises
        ------
        ConfigError
            On a missing/unparsable norm dict, a missing stream, or (when
            bound) missing/non-finite/zero-std variable entries.
        """
        path = self.norm_dict_path
        prefix = f"Normaliser {self.name!r} preflight"
        fix = (
            f"  fix: point model.modules.{self.name}.init_args.norm_dict at the "
            "preprocessing norm_dict.yaml for this sample"
        )
        if not path.is_file():
            raise ConfigError(f"{prefix}: norm dict not found: {path}\n{fix}")
        try:
            with open(path) as fh:
                norm_dict = yaml.safe_load(fh)
        except yaml.YAMLError as err:
            raise ConfigError(
                f"{prefix}: norm dict {path} is not valid YAML: {err}\n{fix}"
            ) from err
        if not isinstance(norm_dict, dict):
            raise ConfigError(f"{prefix}: norm dict {path} must be a mapping\n{fix}")
        for stream in self.streams:
            if stream not in norm_dict:
                raise ConfigError(
                    f"{prefix}: missing input type {stream!r} in {path}. "
                    f"Choose from {sorted(norm_dict)}."
                )
            if not self._fields:
                continue  # unbound (the data-free `salt2 graph validate` path)
            variables = self._fields[stream]
            if missing := set(variables) - set(norm_dict[stream]):
                raise ConfigError(
                    f"{prefix}: missing variables {sorted(missing)} for {stream!r} in {path}. "
                    f"Choose from {sorted(norm_dict[stream])}.\n"
                    f"  fix: add mean/std entries for {sorted(missing)} to {path}, or remove "
                    f"them from the features variable list "
                    f"(config: data.modules.features.init_args.variables.{stream})"
                )
            for variable in variables:
                entry = norm_dict[stream][variable]
                try:
                    mean, std = float(entry["mean"]), float(entry["std"])
                except (KeyError, TypeError, ValueError):
                    raise ConfigError(
                        f"{prefix}: entry for {stream}.{variable} in {path} must be a "
                        f"{{mean, std}} mapping, got {entry!r}"
                    ) from None
                if not (torch.isfinite(torch.tensor(mean)) and torch.isfinite(torch.tensor(std))):
                    raise ConfigError(
                        f"{prefix}: non-finite normalisation parameters for "
                        f"{stream}.{variable} in {path}."
                    )
                if std == 0:
                    raise ConfigError(
                        f"{prefix}: zero standard deviation for {stream}.{variable} in {path}."
                    )

    def materialise(self) -> None:
        """Fill the buffers from the norm dict (the only file I/O this module does).

        Missing streams/variables, non-finite values, and zero stds are errors.

        Raises
        ------
        RuntimeError
            If called before `bind`.
        ValueError
            If the norm dict is missing this module's streams or variables,
            or contains non-finite means/stds or zero stds.
        """
        if not self._bound:
            raise RuntimeError(f"Normaliser {self.name!r}: materialise() before bind()")
        with open(self.norm_dict_path) as fh:
            norm_dict = yaml.safe_load(fh)
        for stream in self.streams:
            if stream not in norm_dict:
                raise ValueError(
                    f"Missing input type {stream!r} in {self.norm_dict_path}. "
                    f"Choose from {sorted(norm_dict)}."
                )
            variables = self._fields[stream]
            if missing := set(variables) - set(norm_dict[stream]):
                raise ValueError(
                    f"Missing variables {sorted(missing)} for {stream!r} in "
                    f"{self.norm_dict_path}. Choose from {sorted(norm_dict[stream])}.\n"
                    f"  fix: add mean/std entries for {sorted(missing)} to "
                    f"{self.norm_dict_path}, or remove them from the features variable "
                    f"list (config: data.modules.features.init_args.variables.{stream})"
                )
            means = torch.as_tensor(
                [float(norm_dict[stream][v]["mean"]) for v in variables], dtype=torch.float32
            )
            stds = torch.as_tensor(
                [float(norm_dict[stream][v]["std"]) for v in variables], dtype=torch.float32
            )
            if not torch.isfinite(means).all() or not torch.isfinite(stds).all():
                raise ValueError(
                    f"Non-finite normalisation parameters for {stream!r} in {self.norm_dict_path}."
                )
            if (stds == 0).any():
                raise ValueError(
                    f"Zero standard deviation for {stream!r} in {self.norm_dict_path}."
                )
            with torch.no_grad():
                getattr(self, f"means_{stream}").copy_(means)
                getattr(self, f"stds_{stream}").copy_(stds)
        self.materialised.fill_(True)

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """``normed.<stream> = (inputs.<stream> - means) / stds``; raises if never materialised."""
        del mode
        # skip under tracing: the tensor->bool read would emit a spurious
        # TracerWarning on every export. Tracing is still guarded —
        # OnnxAdapter rejects unmaterialised modules at construction, before
        # any trace (the eager path keeps this check).
        if not torch.jit.is_tracing() and not bool(self.materialised):
            raise RuntimeError(
                f"Normaliser {self.name!r}: forward before materialise() — on a fresh fit "
                "call materialise(); on checkpoint load the state_dict provides the values "
                "(design §2.3)"
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
    dependency. The legacy ``norm_dict`` constructor arg is accepted for
    config compatibility but ignored.
    """

    def __init__(
        self,
        streams: Sequence[str],
        global_object: str | None = None,
        norm_dict: str | Path | None = None,
        momentum: float | None = 0.1,
        eps: float = 1e-5,
    ) -> None:
        """Capture config only (no file I/O here).

        ``norm_dict`` is deprecated/ignored. ``momentum=None`` gives a
        cumulative moving average instead of an EMA.
        """
        super().__init__()
        if not streams:
            raise ConfigError("MaskedInputNormaliser: streams must be a non-empty sequence")
        if len(set(streams)) != len(tuple(streams)):
            raise ConfigError(f"MaskedInputNormaliser: duplicate streams in {tuple(streams)}")
        if global_object is not None and global_object not in streams:
            raise ConfigError(
                f"MaskedInputNormaliser: global_object {global_object!r} is not in "
                f"streams {tuple(streams)}"
            )
        if momentum is not None and not 0.0 <= momentum <= 1.0:
            raise ConfigError(
                f"MaskedInputNormaliser: momentum must be None or in [0, 1], got {momentum}"
            )
        if eps <= 0.0:
            raise ConfigError(f"MaskedInputNormaliser: eps must be positive, got {eps}")
        # norm_dict is intentionally ignored (stats are learned online); kept in
        # the signature only so existing configs / CLI overrides still parse.
        del norm_dict
        self.streams = tuple(streams)
        self.global_object = global_object
        self.momentum = None if momentum is None else float(momentum)
        self.eps = float(eps)
        self._bound = False

    def _spec(self, stream: str) -> TensorSpec:
        """Shared spec for ``inputs``/``normed``: global ``(B, F)`` or sequence ``(B, T, F)``.

        The last dim is the instance-scoped symbol ``F:<name>.<stream>`` on
        both sides, so the dataset-declared width unifies onto ``normed``.
        """
        width = sym_dim("F", f"{self.name}.{stream}")
        shape: tuple[int | str, ...] = (
            ("B", width) if stream == self.global_object else ("B", _stream_len(stream), width)
        )
        return TensorSpec(shape=shape, dtype="float32")

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``inputs.<stream>`` (+ optional TRAINING-only ``masks``) -> ``normed``."""
        del mode
        requires: dict[str, TensorSpec] = {f"inputs.{s}": self._spec(s) for s in self.streams}
        for stream in self.streams:
            if stream == self.global_object:
                continue
            requires[f"masks.{stream}"] = TensorSpec(
                shape=("B", _stream_len(stream)),
                dtype="bool",
                kind="pad_mask",
                modes=Mode.TRAINING,
                optional=True,
            )
        return IO(
            requires=unflatten_spec(requires),
            produces=unflatten_spec({f"normed.{s}": self._spec(s) for s in self.streams}),
        )

    def bind(self, schema: ResolvedSchema) -> None:
        """Allocate the running-stat buffers (identity init); raises if bound twice."""
        if self._bound:
            raise RuntimeError(
                f"MaskedInputNormaliser {self.name!r}: bind() called twice (design §2.3)"
            )
        for stream in self.streams:
            width = schema.width(f"inputs.{stream}")
            self.register_buffer(f"running_mean_{stream}", torch.zeros(width))
            self.register_buffer(f"running_var_{stream}", torch.ones(width))
            self.register_buffer(
                f"num_batches_tracked_{stream}", torch.zeros((), dtype=torch.long)
            )
            # total VALID-object count seen (cumulative-averaging path only)
            self.register_buffer(f"num_objects_seen_{stream}", torch.zeros((), dtype=torch.long))
        self._bound = True

    @staticmethod
    def _masked_moments(x: Tensor, valid: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        """Per-feature ``(sum, sumsq, count)`` over valid objects (``valid=None`` -> all rows)."""
        # valid is None for the global object (all rows valid); otherwise gather
        # the valid (non-padded) rows of the sequence stream.
        flat = x.reshape(-1, x.shape[-1]) if valid is None else x[valid]
        count = torch.tensor(flat.shape[0], dtype=x.dtype, device=x.device)
        return flat.sum(0), (flat * flat).sum(0), count

    @torch.no_grad()
    def _update_running_stats(self, stream: str, x: Tensor, valid: Tensor | None) -> None:
        """Update the running stats for one stream from masked batch moments.

        All-reduces sum/sumsq/count across DDP ranks before deriving batch
        mean/var (never averages per-rank mean/var directly). No-op if the
        stream is all-padded (count 0) for this batch.
        """
        s_sum, s_sumsq, count = self._masked_moments(x, valid)
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
            new_mean = running_mean + delta * (count / n_new)
            m_old = running_var * n_old
            m_new = batch_var * count
            new_var = (m_old + m_new + delta * delta * (n_old * count / n_new)) / n_new
            running_mean.copy_(new_mean)
            running_var.copy_(new_var)
            seen.add_(count.long())
        else:
            mom = self.momentum
            running_mean.mul_(1 - mom).add_(batch_mean, alpha=mom)
            running_var.mul_(1 - mom).add_(batch_var, alpha=mom)

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Apply frozen running-stat normalisation; updates stats only when training (untraced)."""
        updating = self.training and bool(mode & Mode.TRAINING) and not torch.jit.is_tracing()
        if updating:
            for stream in self.streams:
                x = b.get(f"inputs.{stream}")
                # global object has no pad mask; a sequence stream with a pad
                # mask gathers valid rows via ~pad_mask (mask True == padded);
                # a mask-less stream (optional port with no producer) treats
                # every object as valid (valid=None).
                mask_key = f"masks.{stream}"
                if stream == self.global_object or mask_key not in b:
                    valid = None
                else:
                    valid = ~b.get(mask_key)
                self._update_running_stats(stream, x, valid)
        out: dict[str, Tensor] = {}
        for stream in self.streams:
            mean = getattr(self, f"running_mean_{stream}")
            var = getattr(self, f"running_var_{stream}")
            out[f"normed.{stream}"] = (b.get(f"inputs.{stream}") - mean) / torch.sqrt(
                var + self.eps
            )
        return out
