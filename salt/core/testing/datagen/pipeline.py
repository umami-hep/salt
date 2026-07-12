"""``Pipeline`` -- an ordered ``list[GenModule]`` + a linear contract validator.

Wires modules using the same ``class_path`` / ``init_args`` block grammar salt
uses for a model's module collection, but as an **ordered list** rather than
salt's name-keyed ``dict[str, GraphModule]`` -- because order is the authoring
intent ("first Jets, then Tracks, then the writer").

The validator mirrors salt's ``planner.py`` connectivity / duplicate-producer /
cycle / terminal-consumer checks, linearised (order is given, no topo-sort).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .modules.base import GenModule


class RecipeError(ValueError):
    """Raised when a recipe's module ordering / contract is invalid."""


def _group_of(key: str) -> str:
    return key.split(".", 1)[0]


class Pipeline:
    def __init__(
        self,
        modules: list[GenModule],
        seed: int = 42,
        n_samples: int = 1000,
        flags: dict[str, bool] | None = None,
    ):
        self.modules = list(modules)
        self.seed = seed
        self.n_samples = n_samples
        self.flags = flags or {}
        self.validate()

    # -- validation ------------------------------------------------------- #
    def validate(self) -> None:
        available: set[str] = set()
        producer_of: dict[str, str] = {}
        has_terminal = False

        for i, m in enumerate(self.modules):
            cls = type(m).__name__
            requires = list(getattr(m, "requires", []) or [])
            produces = list(getattr(m, "produces", []) or [])
            mutates = list(getattr(m, "mutates", []) or [])

            # Rule 4: self-referential require (residual "cycle").
            if set(requires) & set(produces):
                bad = set(requires) & set(produces)
                raise RecipeError(
                    f"module {i} {cls!r} requires and produces the same key(s) {sorted(bad)} "
                    f"(a module cannot depend on its own output)"
                )

            # Rule 1: ordering / missing producer.
            # A GROUP key ("tracks") resolves if the group was produced. A FIELD
            # key ("tracks.ftagTruthParentBarcode") must be produced EXACTLY --
            # producing the group key does NOT auto-satisfy an arbitrary field
            # requirement. Producers that want a field requirable declare the
            # field key in `produces` (Constituents -> "<name>.valid"; the
            # inserter -> "<track>.<link_field>"). This lets a downstream test/
            # module require a specific field and have it fail if absent.
            for key in requires:
                is_field_key = "." in key
                satisfied = key in available if is_field_key else (
                    key in available or _group_of(key) in available
                )
                if not satisfied:
                    raise RecipeError(
                        f"module {i} {cls!r} requires {key!r} which no earlier module produces"
                    )

            # Rule 3: undeclared mutation -- you can only mutate what already exists.
            for key in mutates:
                if key not in available and _group_of(key) not in available:
                    raise RecipeError(
                        f"module {i} {cls!r} mutates {key!r} which no earlier module produces"
                    )

            # Rule 2: duplicate producers.
            for key in produces:
                if key in producer_of:
                    raise RecipeError(
                        f"{key!r} produced by both {producer_of[key]!r} and {cls!r}"
                    )
                producer_of[key] = cls
                available.add(key)

            # Rule 5: terminal writer = a consumer that produces nothing. Writers
            # (H5Writer / NormWriter / ClassDictWriter) produce nothing; they
            # either declare the groups they write (non-empty `requires`) or
            # default to writing ALL groups (empty `requires`, detected via the
            # `path` attribute). Both forms count as a terminal writer.
            if not produces and (
                any("." not in k for k in requires) or getattr(m, "path", None) is not None
            ):
                has_terminal = True

        if not has_terminal:
            raise RecipeError(
                "recipe has no terminal writer (a module that requires a group "
                "and produces nothing)"
            )

    # -- run -------------------------------------------------------------- #
    def run(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        data: dict[str, np.ndarray] = {}

        # Producers come before writers in the list, so we collect each
        # producer's GroupSpec as we go and hand the running list to any writer
        # right before it runs (so the writer reconstructs its thin Schema).
        producer_specs: list = []
        for m in self.modules:
            if getattr(m, "n_samples", None) is None:
                m.n_samples = self.n_samples
            if getattr(m, "flags", None) is None:
                m.flags = self.flags

            is_writer = not m.produces and hasattr(m, "_group_specs")
            if is_writer:
                m._group_specs = list(producer_specs)

            data = m(data, rng)

            if not is_writer:
                spec = m.group_spec()
                if spec is not None:
                    producer_specs.append(spec)
        return data

    def set_output_dir(self, output_dir) -> None:
        """Rewrite every writer module's ``path`` to land under ``output_dir``."""
        output_dir = Path(output_dir)
        for m in self.modules:
            if getattr(m, "path", None) is not None:
                m.path = str(output_dir / os.path.basename(m.path))


def load_pipeline(path: str) -> Pipeline:
    """Load a recipe YAML via jsonargparse (``class_path`` / ``init_args``)."""
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--flags", type=dict, default={})
    parser.add_argument("--modules", type=list[GenModule])
    cfg = parser.parse_path(path)
    init = parser.instantiate_classes(cfg)
    return Pipeline(
        modules=init.modules,
        seed=init.seed,
        n_samples=init.n_samples,
        flags=init.flags,
    )
