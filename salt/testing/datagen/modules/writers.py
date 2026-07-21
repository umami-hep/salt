"""Terminal writer modules: ``H5Writer`` / ``NormWriter`` / ``ClassDictWriter``,
each reusing an ``io.py`` function with a reconstructed thin ``Schema``.
"""

from __future__ import annotations

import yaml

from ..io import compute_class_dict, compute_norm_dict, write_h5
from ._util import reconstruct_schema
from .base import GenModule


class _Writer(GenModule):
    """Common base: holds a ``path``, requires its groups, produces nothing."""

    def __init__(self, path: str, groups: list[str] | None = None):
        self.path = path
        self.groups = list(groups) if groups else None
        self.requires = list(groups) if groups else []
        self.produces = []
        self.mutates = []
        # injected by the pipeline before __call__
        self._group_specs: list = []

    def _schema(self, data):
        n = next(iter(data.values())).shape[0] if data else 1
        # keep only specs for groups that are actually in data
        specs = [g for g in self._group_specs if g.name in data]
        return reconstruct_schema(specs, n_samples=n)


class H5Writer(_Writer):
    def __init__(
        self,
        path: str,
        groups: list[str] | None = None,
        attrs: dict | None = None,
        flags: dict | None = None,
    ):
        super().__init__(path, groups)
        self.attrs = attrs
        self.flags = flags

    def __call__(self, data, rng):
        out = data if self.groups is None else {g: data[g] for g in self.groups}
        attrs = dict(self.attrs or {})
        if self.flags:
            attrs["_flags"] = self.flags
        write_h5(out, self.path, attrs=attrs, schema=self._schema(out))
        return data


class NormWriter(_Writer):
    def __call__(self, data, rng):
        out = data if self.groups is None else {g: data[g] for g in self.groups}
        nd = compute_norm_dict(out, schema=self._schema(out))
        with open(self.path, "w") as fh:
            yaml.safe_dump(nd, fh)
        return data


class ClassDictWriter(_Writer):
    def __init__(
        self,
        path: str,
        groups: list[str] | None = None,
        flags: dict | None = None,
    ):
        super().__init__(path, groups)
        self.flags = flags

    def __call__(self, data, rng):
        out = data if self.groups is None else {g: data[g] for g in self.groups}
        cd = compute_class_dict(out, schema=self._schema(out), flags=self.flags or {})
        with open(self.path, "w") as fh:
            yaml.safe_dump(cd, fh)
        return data
