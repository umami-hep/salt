"""`GraphDataset` — the map-style dataset that runs the compiled dataset plan;
one ``__getitem__`` = one full batch, ending at the numpy->torch boundary.
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
from torch.utils.data import Dataset, get_worker_info

from salt.core.data.base import RAW_NAMESPACE, DatasetModule, Reader, WorkerCtx
from salt.core.data.labels import Labels
from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import _SUGGESTION_CUTOFF, ConfigError, MutationError, SchemaError
from salt.core.graph.executor import canonical_produced
from salt.core.graph.planner import Plan, Sinks, compile_plan
from salt.core.graph.spec import KEY_SEP, Mode, TensorSpec
from salt.core.utils.array_utils import maybe_copy

__all__ = ["MODEL_VISIBLE_NAMESPACES", "GraphDataset"]

MODEL_VISIBLE_NAMESPACES = ("inputs", "masks", "labels", "meta")
"""Bundle namespaces converted at the numpy->torch boundary."""


class GraphDataset(Dataset):
    """Map-style dataset executing a compiled dataset plan per batch slice.

    Parameters
    ----------
    modules : dict[str, DatasetModule]
        The dataset modules by instance name (exactly one `Reader`). Names
        are assigned from the dict keys.
    mode : Mode
        The primary mode to compile for (FIT/VAL/TEST/ONNX).
    sinks : Sinks
        The model-boundary demand: a flat iterable of dotted keys for `mode`,
        or a ``{Mode: keys}`` mapping (enables the all-modes-dead check
        across modes). Required — label narrowing is demand-driven.
    seed : int, optional
        Base seed for read-time augmentations when not running in a
        dataloader worker (workers use torch's per-worker seed), default 42.
    debug : bool, optional
        Assert at the torch boundary that no leaf aliases a reusable reader
        buffer, by default False.
    sink_origins : Mapping[str, str] | None, optional
        Demanded-key -> demander description (the model-side module behind
        each sink, `SaltModule.sink_origins`). Used only to upgrade plan and
        reader error messages to a clearer attribution, by default None.

    Raises
    ------
    ConfigError
        If `sinks` is missing, or not exactly one module is a `Reader`.
    SchemaError
        When a module's declared raw fields are absent from the schema
        artifact (static validation, design §2.6).
    GraphError
        Any plan-compilation error (connectivity, kinds, shapes, cycles).
    """

    def __init__(
        self,
        modules: dict[str, DatasetModule],
        mode: Mode,
        sinks: Sinks,
        seed: int = 42,
        debug: bool = False,
        sink_origins: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        if sinks is None:
            raise ConfigError(
                "GraphDataset needs explicit sinks (the model boundary's demanded keys) — "
                "label narrowing is demand-driven (design §3.3, §6.1)"
            )
        for name, module in modules.items():
            module.name = name  # instance names come from the config dict key
        readers = [m for m in modules.values() if isinstance(m, Reader)]
        if len(readers) != 1:
            raise ConfigError(
                f"GraphDataset needs exactly one Reader module, got {len(readers)} "
                f"({[r.name for r in readers]}) (design §6.1)"
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
                "cannot be checked statically; a misspelled key will fail at worker bind "
                "(design §2.6)",
                stacklevel=2,
            )
        self._plan: Plan = self._compile()
        self._read_fields: dict[str, dict[str, str]] = self._collect_read_fields()
        self._bound_pid: int | None = None
        self._build_caches()

    def _compile(self) -> Plan:
        """Compile the dataset plan and statically validate raw-field demands.

        Raises
        ------
        SchemaError
            When a step demands a raw field absent from the schema artifact
            (nearest-name suggestions included).
        """
        universe = self._reader.label_universe()
        plan = compile_plan(
            self._modules,  # type: ignore[arg-type]
            self._mode,
            sources={},
            schema=universe,
            sinks=self._sinks,
            sink_origins=self._sink_origins,
        )
        for step in plan.steps:
            for key, spec in step.requires.items():
                parts = key.split(KEY_SEP)
                if parts[0] != RAW_NAMESPACE or len(parts) != 2 or spec.fields is None:
                    continue
                gschema = self._reader.schema_group(parts[1])
                if gschema is None:  # no schema artifact: bind-time checks remain
                    continue
                for field in spec.fields:
                    if field in gschema.fields:
                        continue
                    near = get_close_matches(
                        field, sorted(gschema.fields), n=3, cutoff=_SUGGESTION_CUTOFF
                    )
                    hint = f"; nearest: {', '.join(near)}" if near else ""
                    raise SchemaError(
                        f"field {field!r} demanded by module {step.name!r} not present in "
                        f"the schema for stream {parts[1]!r}{hint} (design §2.6)"
                    )
        return plan

    def _collect_read_fields(self) -> dict[str, dict[str, str]]:
        """Compute the demand-narrowed per-stream read set from the plan.

        Label fields carry their model-side demand provenance when known
        (`sink_origins`), so a reader bind error names the task module the
        user configured, not just the `Labels` relay.

        Returns
        -------
        dict[str, dict[str, str]]
            ``{stream: {field: demanding module}}``, in plan/demand order.
        """
        out: dict[str, dict[str, str]] = {}
        for step in self._plan.steps:
            collector = getattr(step.module, "read_fields", None)
            if collector is None:
                continue
            for stream, fields in collector(step).items():
                bucket = out.setdefault(stream, {})
                for field, who in fields.items():
                    origin = (
                        self._sink_origins.get(f"labels.{stream}.{field}")
                        if self._sink_origins is not None
                        else None
                    )
                    bucket.setdefault(field, f"{who} (for {origin})" if origin else who)
        return out

    def _build_caches(self) -> None:
        """Precompute the per-batch loop state from the frozen plan (hot path).

        The plan is static, so the step sequence (module, reader flag,
        declared key set) and the model-visible boundary keys are computed
        once here instead of being rebuilt every ``__getitem__`` (`Bundle.merge`
        enforces produced == declared, so the boundary key list is exact).
        Rebuilt after unpickling (`__setstate__` recompiles the plan).
        """
        self._exec_steps = tuple(
            (step.name, step.module, isinstance(step.module, Reader), frozenset(step.produces))
            for step in self._plan.steps
        )
        self._boundary_keys = tuple(
            (key, tuple(parts))
            for step in self._plan.steps
            for key in step.produces
            if (parts := key.split(KEY_SEP))[0] in MODEL_VISIBLE_NAMESPACES
        )

    @property
    def plan(self) -> Plan:
        """The compiled dataset plan for this dataset's mode."""
        return self._plan

    @property
    def modules(self) -> dict[str, DatasetModule]:
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

    def boundary_specs(self) -> dict[str, TensorSpec]:
        """The model-visible produced leaves — the model plan's ``sources``.

        Returns
        -------
        dict[str, TensorSpec]
            ``{dotted_key: spec}`` for every produced key under the
            model-visible namespaces, in plan order.
        """
        out: dict[str, TensorSpec] = {}
        for step in self._plan.steps:
            out.update({
                key: spec
                for key, spec in step.produces.items()
                if key.split(KEY_SEP, 1)[0] in MODEL_VISIBLE_NAMESPACES
            })
        return out

    def _maybe_bind(self) -> None:
        """Bind all plan modules once per worker process (pid-guarded).

        The pid guard covers fork inheritance: a dataset forked into a
        dataloader worker re-binds there, giving each worker its own lazy H5
        handles and reusable buffers.
        """
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
            if isinstance(module, DatasetModule):
                module.bind(replace(ctx, step=step))
        self._bound_pid = pid

    def __len__(self) -> int:
        """Return the number of rows served (delegates to the reader)."""
        return len(self._reader)

    def __getitem__(self, rows: slice) -> dict[str, Any]:
        """Run the dataset plan for one contiguous batch slice.

        Parameters
        ----------
        rows : slice
            Contiguous row range with ``start`` and ``stop`` set (the
            `RandomBatchSampler` contract).

        Returns
        -------
        dict[str, Any]
            The model-visible nested dict (``inputs`` / ``masks`` /
            ``labels`` / ``meta``) with torch tensors at the leaves —
            ``raw.*`` never crosses the boundary.

        Raises
        ------
        TypeError
            If `rows` is not a slice with start and stop. (Under
            ``debug=True``, `_to_torch` additionally raises `MutationError`
            on a boundary leaf aliasing a reusable reader buffer.)
        """
        if not isinstance(rows, slice) or rows.start is None or rows.stop is None:
            raise TypeError(
                f"GraphDataset is indexed by contiguous slices with start/stop, got {rows!r} "
                "(samplers.py:41-55 contract)"
            )
        self._maybe_bind()
        bundle = Bundle()
        for name, module, is_reader, expected in self._exec_steps:
            if is_reader:
                produced = module.read(rows, self._mode)  # type: ignore[attr-defined]
            else:
                produced = module.process(bundle, rows, self._mode)  # type: ignore[attr-defined]
            bundle.merge(
                canonical_produced(produced, expected, name),
                who=name,
                expected=expected,
            )
        return self._to_torch(bundle)

    def _to_torch(self, bundle: Bundle) -> dict[str, Any]:
        """Convert model-visible bundle leaves to torch (the framework boundary).

        Raises
        ------
        MutationError
            Under ``debug=True``, when a leaf aliases a reader buffer.
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
                        "worker; copy at the producing module (design §2.4 contract #9)"
                    )
                value = torch.from_numpy(maybe_copy(value))
            node = out
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return out

    def __getstate__(self) -> dict[str, Any]:
        """Drop the compiled plan (holds MappingProxyType — not picklable).

        The plan is recompiled in `__setstate__` (deterministic: same config
        -> same plan and plan_hash).
        """
        state = self.__dict__.copy()
        state["_plan"] = None
        state["_read_fields"] = None
        state["_bound_pid"] = None
        state["_exec_steps"] = None
        state["_boundary_keys"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore and recompile the plan (static, config-only)."""
        self.__dict__.update(state)
        self._plan = self._compile()
        self._read_fields = self._collect_read_fields()
        self._bound_pid = None
        self._build_caches()
