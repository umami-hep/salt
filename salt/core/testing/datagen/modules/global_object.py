"""``GlobalObject`` / ``Jets`` -- the [N]-shape global group module.

Reuses ``engine._build_group_array`` (global path). Holds a dict-form group
spec and parses it via ``schema.parse_schema`` at call time so ``_parse_field``
runs (i4 int defaults) and validation fires -- NEVER hand-builds GroupSpec /
FieldSpec.
"""

from __future__ import annotations

import numpy as np

from ..engine import _build_group_array
from ..schema import parse_schema
from ._fields import resolve_fields
from ._util import infer_n
from .base import GenModule


class GlobalObject(GenModule):
    def __init__(
        self,
        name: str,
        fields: list[dict] | str,
        fill_float: float = float("nan"),
        fill_int: int = -1,
        attrs: dict | None = None,
        n_samples: int | None = None,
        flags: dict[str, bool] | None = None,
    ):
        self.name = name
        self._fields = resolve_fields(fields)
        self._group_dict = {
            "name": name,
            "kind": "global",
            "fields": self._fields,
            "attrs": attrs or {},
        }
        self._fill = {"fill": {"float": fill_float, "int": fill_int}}
        self.n_samples = n_samples
        self.flags = flags
        self.produces = [name]
        self.requires = []
        self.mutates = []
        self._group_spec = None

    def _schema(self, data):
        sch = parse_schema(
            {
                "n_samples": infer_n(data, self.n_samples),
                "groups": [self._group_dict],
                **self._fill,
            }
        )
        self._group_spec = sch.group(self.name)
        return sch

    def __call__(self, data, rng):
        sch = self._schema(data)
        group = sch.group(self.name)
        arr, _ = _build_group_array(rng, sch, group, self.flags or {})
        data[self.name] = arr
        return data

    def group_spec(self):
        if self._group_spec is None:
            # build (no data needed: n_samples irrelevant for spec)
            self._schema({})
        return self._group_spec


class Jets(GlobalObject):
    """``GlobalObject`` defaulting ``name='jets'``."""

    def __init__(
        self,
        fields: list[dict] | str,
        name: str = "jets",
        fill_float: float = float("nan"),
        fill_int: int = -1,
        attrs: dict | None = None,
        n_samples: int | None = None,
        flags: dict[str, bool] | None = None,
    ):
        super().__init__(
            name=name,
            fields=fields,
            fill_float=fill_float,
            fill_int=fill_int,
            attrs=attrs,
            n_samples=n_samples,
            flags=flags,
        )
