"""``GlobalObject`` / ``Jets`` — the [N]-shape global group module (parses its
dict-form spec via ``schema.parse_schema``; never hand-builds specs).
"""

from __future__ import annotations

from ._fields import resolve_fields
from .base import _GroupModule


class GlobalObject(_GroupModule):
    """Generate a one-per-sample group (jets, events) with no item axis."""

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


class Jets(GlobalObject):
    """``GlobalObject`` defaulting ``name='jets'``."""

    def __init__(self, fields: list[dict] | str, name: str = "jets", **kw):
        super().__init__(name=name, fields=fields, **kw)
