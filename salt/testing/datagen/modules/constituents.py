"""``Constituents`` + thin named subclasses — the [N, M] collection modules
(parse dict-form specs via ``schema.parse_schema``; never hand-build specs).
"""

from __future__ import annotations

from ..engine import _build_group_array
from ..schema import parse_schema
from ._fields import resolve_fields
from ._util import infer_n
from .base import GenModule


class Constituents(GenModule):
    """Generate a variable-length constituent group with a per-item ``valid`` mask."""

    def __init__(
        self,
        name: str,
        max_items: int,
        fields: list[dict] | str,
        valid_fraction: float = 0.5,
        min_valid: int = 0,
        mask_invalid: bool = True,
        fill_float: float = float("nan"),
        fill_int: int = -1,
        n_samples: int | None = None,
        flags: dict[str, bool] | None = None,
    ):
        self.name = name
        self._fields = resolve_fields(fields)
        self._group_dict = {
            "name": name,
            "kind": "constituent",
            "max_items": max_items,
            "valid_fraction": valid_fraction,
            "min_valid": min_valid,
            "mask_invalid": mask_invalid,
            "fields": self._fields,
        }
        self._fill = {"fill": {"float": fill_float, "int": fill_int}}
        self.n_samples = n_samples
        self.flags = flags
        self.produces = [name, f"{name}.valid"]
        self.requires = []
        self.mutates = []
        self._group_spec = None

    def _schema(self, data):
        sch = parse_schema({
            "n_samples": infer_n(data, self.n_samples),
            "groups": [self._group_dict],
            **self._fill,
        })
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
            self._schema({})
        return self._group_spec


# -- thin named subclasses ---------------------------------------------------- #
# Each fixes a default `name` (and sometimes `max_items` / `mask_invalid`). All
# constructor params are fully type-annotated so jsonargparse can resolve them
# from a recipe's class_path/init_args block (no untyped **kwargs).
class Tracks(Constituents):
    """Constituent group named ``tracks`` (40 items by default)."""

    def __init__(
        self,
        fields: list[dict] | str,
        name: str = "tracks",
        max_items: int = 40,
        valid_fraction: float = 0.5,
        min_valid: int = 0,
        mask_invalid: bool = True,
        n_samples: int | None = None,
        flags: dict[str, bool] | None = None,
    ):
        super().__init__(
            name=name,
            max_items=max_items,
            fields=fields,
            valid_fraction=valid_fraction,
            min_valid=min_valid,
            mask_invalid=mask_invalid,
            n_samples=n_samples,
            flags=flags,
        )


class TracksLoose(Tracks):
    """Constituent group named ``tracks_loose``."""

    def __init__(self, fields: list[dict] | str, name: str = "tracks_loose", **kw):
        super().__init__(fields=fields, name=name, **kw)


class Flows(Tracks):
    """Constituent group named ``flows``."""

    def __init__(self, fields: list[dict] | str, name: str = "flows", **kw):
        super().__init__(fields=fields, name=name, **kw)


class Charged(Tracks):
    """Constituent group named ``charged``."""

    def __init__(self, fields: list[dict] | str, name: str = "charged", **kw):
        super().__init__(fields=fields, name=name, **kw)


class Neutral(Tracks):
    """Constituent group named ``neutral``."""

    def __init__(self, fields: list[dict] | str, name: str = "neutral", **kw):
        super().__init__(fields=fields, name=name, **kw)


class Objects(Tracks):
    """Constituent group named ``objects`` (20 items by default)."""

    def __init__(self, fields: list[dict] | str, name: str = "objects", max_items: int = 20, **kw):
        super().__init__(fields=fields, name=name, max_items=max_items, **kw)


class Electrons(Tracks):
    """Constituent group named ``electrons``."""

    def __init__(
        self,
        fields: list[dict] | str,
        name: str = "electrons",
        max_items: int = 10,
        mask_invalid: bool = False,
        **kw,
    ):
        super().__init__(
            fields=fields, name=name, max_items=max_items, mask_invalid=mask_invalid, **kw
        )


class Constituent(Tracks):
    """Single constituent stream (regression heads): name supplied by the recipe."""

    def __init__(self, fields: list[dict] | str, name: str, **kw):
        super().__init__(fields=fields, name=name, **kw)


class TruthHadrons(Tracks):
    """Link-free truth-hadron object stream (e.g. truth_hadron_regression)."""

    def __init__(
        self,
        fields: list[dict] | str,
        name: str = "truth_hadrons",
        max_items: int = 5,
        min_valid: int = 1,
        **kw,
    ):
        super().__init__(fields=fields, name=name, max_items=max_items, min_valid=min_valid, **kw)
