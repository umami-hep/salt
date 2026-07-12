"""``TruthHadronInserter`` -- owns BOTH sides of the maskformer link.

It produces the ``truth_hadrons`` object group (real ``barcode`` ids + a
``flavour`` object-class label) AND injects the matching
``ftagTruthParentBarcode`` link field into the pre-existing ``tracks`` array,
reusing ``engine._resolve_link``.

Critical wiring: ``_resolve_link(rng, schema, data, g, f, flags)`` where
``g`` is the SOURCE-group GroupSpec (the resolver does ``src = data[g.name]``
internally) and ``f`` is the ``LinkField`` that lives INSIDE that source group's
``.fields`` -- NOT a free-standing/hand-built link field, and NOT the structured
array.

Critical typing: every field -- including the barcode ``id`` and the
``ftagTruthParentBarcode`` ``link`` -- is routed through ``parse_schema`` (hence
``_parse_field``), so they default to ``'i4'`` (32-bit int). We NEVER hand-build
``LinkField`` / ``IdField`` / ``GroupSpec`` (the bare dataclasses default ``f4``).
"""

from __future__ import annotations

import numpy as np

from .. import engine
from ..engine import _build_group_array
from ..schema import LinkField, parse_schema
from ._fields import resolve_fields
from ._util import add_field, field_dicts_from_array, infer_n
from .base import GenModule


class TruthHadronInserter(GenModule):
    def __init__(
        self,
        hadron_fields: list[dict] | str,
        hadron_name: str = "truth_hadrons",
        track_name: str = "tracks",
        max_items: int = 5,
        valid_fraction: float = 0.5,
        min_valid: int = 1,
        link_field: str = "ftagTruthParentBarcode",
        references: str | None = None,
        unmatched_fraction: float = 0.0,
        unmatched_value: int = -1,
        required_match: bool = False,
        select_over: str = "valid_ids",
        fill_float: float = float("nan"),
        fill_int: int = -1,
        n_samples: int | None = None,
        flags: dict[str, bool] | None = None,
    ):
        self.hadron_name = hadron_name
        self.track_name = track_name
        self.link_field = link_field
        self._hadron_fields = resolve_fields(hadron_fields)
        if references is None:
            references = f"{hadron_name}.barcode"
        # Hold DICT-FORM specs only. Parsed via parse_schema in __call__.
        self._hadron_dict = {
            "name": hadron_name,
            "kind": "constituent",
            "max_items": max_items,
            "valid_fraction": valid_fraction,
            "min_valid": min_valid,
            "fields": self._hadron_fields,
        }
        self._link_dict = {
            "name": link_field,
            "type": "link",
            "references": references,
            "unmatched_fraction": unmatched_fraction,
            "unmatched_value": unmatched_value,
            "required_match": required_match,
            "select_over": select_over,
        }
        self._fill = {"fill": {"float": fill_float, "int": fill_int}}
        self.n_samples = n_samples
        self.flags = flags
        self.requires = [track_name]
        self.produces = [hadron_name, f"{track_name}.{link_field}"]
        self.mutates = [track_name]
        self._group_specs = None

    def _build_schema(self, data):
        # Reconstruct the existing tracks group's dict-form field specs, append
        # the link field, and parse the 2-group schema. parse_schema routes the
        # barcode id + the link through _parse_field (i4 default) and runs
        # _propagate_required_match_min_valid (min_valid>=1 semantics).
        track_fields = field_dicts_from_array(data[self.track_name])
        track_fields.append(self._link_dict)
        tracks_prime = {
            "name": self.track_name,
            "kind": "constituent",
            "max_items": int(data[self.track_name].shape[1]),
            "valid_fraction": 0.5,
            "min_valid": 0,
            "fields": track_fields,
        }
        schema = parse_schema(
            {
                "n_samples": infer_n(data, self.n_samples),
                "groups": [self._hadron_dict, tracks_prime],
                **self._fill,
            }
        )
        self._group_specs = [
            schema.group(self.hadron_name),
            schema.group(self.track_name),
        ]
        return schema

    def __call__(self, data, rng):
        schema = self._build_schema(data)

        # 1. build truth_hadrons (real barcode ids + flavour label, valid mask, -1 fill)
        hads, _ = _build_group_array(
            rng, schema, schema.group(self.hadron_name), self.flags or {}
        )

        # 2. widen the existing tracks array with the link column (i4 from parsed field)
        track_group = schema.group(self.track_name)
        link: LinkField = next(
            fld for fld in track_group.fields if fld.name == self.link_field
        )
        tracks_widened = add_field(
            data[self.track_name], self.link_field, np.dtype(link.dtype)
        )

        # 3. resolve the link: g = source GroupSpec (resolver does src = data[g.name]);
        #    f = its LinkField.
        data_slice = {self.hadron_name: hads, self.track_name: tracks_widened}
        engine._resolve_link(rng, schema, data_slice, track_group, link, self.flags or {})

        # 4. write both back
        data[self.hadron_name] = hads
        data[self.track_name] = tracks_widened
        return data

    def group_spec(self):
        # The inserter contributes the hadron group spec to the writer-side
        # schema (the tracks spec is contributed by the Tracks module). Return
        # the hadron GroupSpec; the pipeline collects all module group_specs.
        if self._group_specs is None:
            schema = parse_schema(
                {
                    "n_samples": 1,
                    "groups": [self._hadron_dict],
                    **self._fill,
                }
            )
            return schema.group(self.hadron_name)
        return self._group_specs[0]
