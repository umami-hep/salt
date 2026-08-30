"""`_PlanRunner` — the dataset-plan machinery shared by the map-style and streaming
datasets: compile, validate, bind per worker, execute steps, cross the torch boundary.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from dataclasses import replace
from difflib import get_close_matches
from typing import Any

import numpy as np
import torch
from torch.utils.data import get_worker_info

from salt.data.base import RAW_NAMESPACE, Reader, SaltDatasetModule, WorkerCtx
from salt.data.processors.labels import Labels
from salt.graph.bundle import Bundle
from salt.graph.errors import _SUGGESTION_CUTOFF, ConfigError, MutationError, SchemaError
from salt.graph.executor import canonical_produced
from salt.graph.planner import Plan, Sinks, compile_plan
from salt.graph.spec import KEY_SEP, Mode, TensorSpec
from salt.utils.array_utils import maybe_copy

__all__ = ["MODEL_VISIBLE_NAMESPACES", "_PlanRunner"]

MODEL_VISIBLE_NAMESPACES = ("inputs", "masks", "labels", "meta")
"""Bundle namespaces converted at the numpy->torch boundary."""

_DERIVED_ATTRS = ("_plan", "_read_fields", "_bound_pid", "_exec_steps", "_boundary_keys")
"""Attributes rebuilt from the config by `_recompute_derived`, dropped when pickling."""


def _is_model_visible(key: str) -> bool:
    """Whether a produced key crosses the numpy->torch boundary to the model."""
    return key.split(KEY_SEP, 1)[0] in MODEL_VISIBLE_NAMESPACES


class _PlanRunner:
    """Compiled-plan host: one `Reader`, N processors, one torch-boundary conversion.

    Carries everything that does not depend on HOW rows are addressed — plan
    compilation and static schema validation, the demand-narrowed read set,
    per-worker binding, step execution, the numpy->torch boundary, and the
    pickling contract. `SaltDataset` adds map-style ``__getitem__``;
    `IterableSaltDataset` adds streaming ``__iter__``. Both run the SAME plan
    through `_run_plan`, which is what makes a streaming batch and a map-style
    batch the same object downstream.

    Parameters
    ----------
    modules : dict[str, SaltDatasetModule]
        The dataset modules by instance name (exactly one `Reader`). Names are
        assigned from the dict keys.
    mode : Mode
        The primary mode to compile for (FIT/VAL/TEST/ONNX).
    sinks : Sinks
        The model-boundary demand: a flat iterable of dotted keys for `mode`,
        or a ``{Mode: keys}`` mapping. Required — label narrowing is
        demand-driven.
    seed : int, optional
        Base seed for read-time augmentations when not running in a dataloader
        worker (workers use torch's per-worker seed), by default 42.
    debug : bool, optional
        Assert at the torch boundary that no leaf aliases a reusable reader
        buffer, by default False.
    sink_origins : Mapping[str, str] | None, optional
        Demanded-key -> demander description, used only to improve error
        messages, by default None.

    Raises
    ------
    ConfigError
        If `sinks` is missing, or not exactly one module is a `Reader`.
    SchemaError
        When a module's declared raw fields are absent from the schema artifact.
    GraphError
        Any plan-compilation error (connectivity, kinds, shapes, cycles).
    """

    def __init__(
        self,
        modules: dict[str, SaltDatasetModule],
        mode: Mode,
        sinks: Sinks,
        seed: int = 42,
        debug: bool = False,
        sink_origins: Mapping[str, str] | None = None,
    ) -> None:
        if sinks is None:
            raise ConfigError(
                "the dataset needs explicit sinks (the model boundary's demanded keys) — "
                "label narrowing is demand-driven"
            )
        for name, module in modules.items():
            module.name = name  # instance names come from the config dict key
        readers = [m for m in modules.values() if isinstance(m, Reader)]
        if len(readers) != 1:
            raise ConfigError(
                f"the dataset needs exactly one Reader module, got {len(readers)} "
                f"({[r.name for r in readers]})"
            )
        self._modules = modules
        self._reader = readers[0]
        self._mode = mode
        self._sinks: Sinks = dict(sinks) if isinstance(sinks, Mapping) else list(sinks)
        self._sink_origins = dict(sink_origins) if sink_origins is not None else None
        self._seed = seed
        self._debug = debug
        # framework wiring: demand-driven producers adopt the reader's streams
        for module in modules.values():
            if isinstance(module, Labels):
                module.bind_streams(self._reader.streams)
        if self._reader.schema is None:
            warnings.warn(
                f"reader {self._reader.name!r} has no schema artifact: field spellings "
                "cannot be checked statically; a misspelled key will fail at worker bind",
                stacklevel=3,
            )
        self._recompute_derived()

    def _recompute_derived(self) -> None:
        """Build everything implied by the modules + config: the compiled plan, the
        demand-narrowed read set, and the per-batch loop caches.

        Called from `__init__` and again after unpickling (`__getstate__` drops
        exactly `_DERIVED_ATTRS`), so this is the single place that decides what
        the two paths have to agree on.

        The loop caches turn the frozen plan into a step sequence plus the
        model-visible boundary key list once, instead of per batch.
        """
        self._plan: Plan = self._compile()
        self._read_fields: dict[str, dict[str, str]] = self._collect_read_fields()
        self._bound_pid: int | None = None
        self._exec_steps = tuple(
            (step.name, step.module, isinstance(step.module, Reader), set(step.produces))
            for step in self._plan.steps
        )
        self._boundary_keys = tuple(
            (key, tuple(key.split(KEY_SEP)))
            for step in self._plan.steps
            for key in step.produces
            if _is_model_visible(key)
        )

    def _compile(self) -> Plan:
        """Compile the dataset plan, then statically validate its raw-field demands."""
        plan = compile_plan(
            self._modules,  # type: ignore[arg-type]
            self._mode,
            sources={},
            schema=self._reader.label_universe(),
            sinks=self._sinks,
            sink_origins=self._sink_origins,
        )
        self._check_raw_fields(plan)
        return plan

    def _check_raw_fields(self, plan: Plan) -> None:
        """Raise `SchemaError` (with nearest-name suggestions) for a demanded raw field the
        reader's schema artifact does not have. Readers without one are checked at bind.
        """
        for step in plan.steps:
            for key, spec in step.requires.items():
                parts = key.split(KEY_SEP)
                if parts[0] != RAW_NAMESPACE or len(parts) != 2 or spec.fields is None:
                    continue
                gschema = self._reader.schema_group(parts[1])
                if gschema is None:  # no schema artifact: bind-time checks remain
                    continue
                # declaration order, not set order: with two fields missing the
                # error must always name the same one
                for field in spec.fields:
                    if field in gschema.fields:
                        continue
                    near = get_close_matches(
                        field, sorted(gschema.fields), n=3, cutoff=_SUGGESTION_CUTOFF
                    )
                    hint = f"; nearest: {', '.join(near)}" if near else ""
                    raise SchemaError(
                        f"field {field!r} demanded by module {step.name!r} not present in "
                        f"the schema for stream {parts[1]!r}{hint}"
                    )

    def _collect_read_fields(self) -> dict[str, dict[str, str]]:
        """Compute the demand-narrowed per-stream read set from the plan.

        Label fields carry their model-side demand provenance when known
        (`sink_origins`), so a reader bind error names the task module the
        user configured, not just the `Labels` relay.
        """
        origins = self._sink_origins or {}
        out: dict[str, dict[str, str]] = {}
        for step in self._plan.steps:
            collector = getattr(step.module, "read_fields", None)
            if collector is None:
                continue
            for stream, fields in collector(step).items():
                bucket = out.setdefault(stream, {})
                for field, who in fields.items():
                    origin = origins.get(f"labels.{stream}.{field}")
                    bucket.setdefault(field, f"{who} (for {origin})" if origin else who)
        return out

    @property
    def plan(self) -> Plan:
        """The compiled dataset plan for this dataset's mode."""
        return self._plan

    @property
    def modules(self) -> dict[str, SaltDatasetModule]:
        """The configured dataset modules by instance name (read-only view)."""
        return dict(self._modules)

    @property
    def read_fields(self) -> dict[str, dict[str, str]]:
        """The demand-narrowed per-stream read set with demand provenance (fresh copies)."""
        return {stream: dict(fields) for stream, fields in self._read_fields.items()}

    @property
    def reader(self) -> Reader:
        """The dataset's single reader module."""
        return self._reader

    @property
    def mode(self) -> Mode:
        """The mode this dataset's plan was compiled for."""
        return self._mode

    def boundary_specs(self) -> dict[str, TensorSpec]:
        """The model-visible produced leaves, in plan order — the model plan's ``sources``."""
        return {
            key: spec
            for step in self._plan.steps
            for key, spec in step.produces.items()
            if _is_model_visible(key)
        }

    def _maybe_bind(self) -> None:
        """Bind all plan modules once per worker process (pid-guarded for fork inheritance)."""
        pid = os.getpid()
        if self._bound_pid == pid:
            return
        info = get_worker_info()
        ctx = WorkerCtx(
            mode=self._mode,
            read_fields=self._read_fields,
            seed=int(info.seed) if info is not None else self._seed,
            worker_id=info.id if info is not None else 0,
            num_workers=info.num_workers if info is not None else 0,
        )
        for step in self._plan.steps:
            module = step.module
            if isinstance(module, SaltDatasetModule):
                module.bind(replace(ctx, step=step))
        self._bound_pid = pid

    def _run_plan(self, rows: slice, raw: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
        """Run every plan step for one batch and cross the torch boundary.

        `raw` is the reader step's produced dict when the caller has already
        read it (the streaming path, which reads in blocks and slices batches
        out of them); ``None`` means call the reader for `rows` (the map-style
        path). Processors receive `rows` unchanged — they ignore it (`Features`
        and `Labels` both ``del rows``), so the two paths produce the same
        object.
        """
        bundle = Bundle()
        for name, module, is_reader, expected in self._exec_steps:
            if is_reader:
                produced = (
                    raw if raw is not None else module.read(rows, self._mode)  # type: ignore[attr-defined]
                )
            else:
                produced = module.process(bundle, rows, self._mode)  # type: ignore[attr-defined]
            bundle.merge(
                canonical_produced(produced, expected, name),
                who=name,
                expected=expected,
            )
        return self._to_torch(bundle)

    def _to_torch(self, bundle: Bundle) -> dict[str, Any]:
        """Convert model-visible bundle leaves to torch (the framework boundary); under
        ``debug=True`` raises `MutationError` when a leaf aliases a reader buffer.
        """
        out: dict[str, Any] = {}
        # `_boundary_keys` is the plan-exact model-visible key list, precomputed
        # in `_build_caches`
        for key, parts in self._boundary_keys:
            value = bundle.get(key)
            if isinstance(value, np.ndarray):
                if self._debug and self._reader.aliases(value):
                    raise MutationError(
                        f"[mode={self._mode.name}] boundary leaf {key!r} aliases a reusable "
                        "reader buffer — it would be overwritten by the next batch in this "
                        "worker; copy at the producing module"
                    )
                value = torch.from_numpy(maybe_copy(value))
            node = out
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return out

    def __getstate__(self) -> dict[str, Any]:
        """Drop the derived state — the compiled plan holds a `MappingProxyType` and does
        not pickle. `__setstate__` rebuilds it (deterministic: same config -> same plan).
        """
        return {k: v for k, v in self.__dict__.items() if k not in _DERIVED_ATTRS}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the config and recompile everything derived from it."""
        self.__dict__.update(state)
        self._recompute_derived()
