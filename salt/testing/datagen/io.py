"""Public IO + dict-derivation functions for the schema-driven generator:
``write_h5``, ``compute_norm_dict``, ``compute_class_dict`` (stats from
produced data, field roles and lengths from the schema).
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from .schema import (
    DistributionField,
    LabelField,
    Schema,
    as_schema,
)

# Deliberately standalone (not salt.utils.logging.get_logger): this package is kept
# extractable as its own standalone datagen package, so it takes no salt.*
# imports. Its __name__ is already under "salt.", so once salt.utils.logging
# configures the "salt" root logger, records from here reach the same handler
# and level anyway — behaviour is identical either way.
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# write_h5
# --------------------------------------------------------------------------- #
def write_h5(
    data: dict[str, np.ndarray],
    path: str | Path,
    attrs: dict | None = None,
    schema: dict | Schema | None = None,
) -> None:
    """Write produced structured arrays to an HDF5 file.

    File attrs:
      * ``f.attrs['unique_jets']`` = len of the primary global group.
      * ``f.attrs['config']``      = ``schema.file_attrs.config`` (default '{}').
    Group attrs:
      * ``class_names`` from ``label`` fields -> ``f[group].attrs[field] = names``.
        These live in the SCHEMA (structured arrays cannot carry them), so the
        schema MUST be passed for class-name attrs to be written.
      * caller-supplied ``attrs`` is merged last and can override anything.
    """
    sch = as_schema(schema) if schema is not None else None
    flags = (attrs or {}).pop("_flags", {}) if isinstance(attrs, dict) else {}

    with h5py.File(path, "w") as f:
        # write datasets in dict order
        for name, arr in data.items():
            f.create_dataset(name, data=arr)

        # file attrs
        if sch is not None:
            primary = sch.unique_jets_from()
            if primary in data:
                f.attrs["unique_jets"] = len(data[primary])
            f.attrs["config"] = sch.config_attr
        else:
            # fall back to first dataset for unique_jets
            first = next(iter(data))
            f.attrs["unique_jets"] = len(data[first])
            f.attrs["config"] = "{}"

        # class-name group attrs from the schema
        if sch is not None:
            for g in sch.groups:
                if g.name not in data:
                    continue
                for fld in g.fields:
                    if isinstance(fld, LabelField):
                        names = fld.resolved_class_names(flags)
                        if names is not None:
                            f[g.name].attrs[fld.name] = list(names)

        # caller-supplied attrs (merged last)
        if attrs:
            for key, val in attrs.items():
                f.attrs[key] = val


# --------------------------------------------------------------------------- #
# compute_norm_dict
# --------------------------------------------------------------------------- #
def _feature_fields_for_group(sch: Schema | None, group_name: str, arr: np.ndarray):
    """Yield (field_name) that should be normalised (distribution features only).

    With a schema: distribution fields only. Without: float dtype, non-`valid`.
    """
    if sch is not None:
        try:
            g = sch.group(group_name)
        except KeyError:
            g = None
        if g is not None:
            # alias groups carry no fields of their own -> use the source group's
            if g.alias_of is not None:
                g = sch.group(g.alias_of)
            for fld in g.fields:
                if isinstance(fld, DistributionField):
                    yield fld.name
            return
    # schema-less / unknown group: infer from dtype
    for name in arr.dtype.names:
        if name == "valid":
            continue
        if arr.dtype[name].kind == "f":
            yield name


def compute_norm_dict(
    data: dict[str, np.ndarray],
    schema: dict | Schema | None = None,
) -> dict:
    """Real per-feature mean/std from produced arrays.

    Computed over distribution (float) features only; skips id/link/label/valid.
    Invalid entries (NaN, and -1 int fills) are excluded via nan-aware
    reductions. Zero-valid / zero-variance columns fall back to the loadable
    no-op ``{'mean': 0.0, 'std': 1.0}``.
    """
    sch = as_schema(schema) if schema is not None else None
    out: dict[str, dict[str, dict[str, float]]] = {}

    for group_name, arr in data.items():
        group_out: dict[str, dict[str, float]] = {}
        for fname in _feature_fields_for_group(sch, group_name, arr):
            col = np.asarray(arr[fname], dtype=np.float64).reshape(-1)
            # exclude invalid: NaNs are already NaN; also drop -1 fills defensively
            valid_vals = col[~np.isnan(col)]
            if valid_vals.size == 0:
                group_out[fname] = {"mean": 0.0, "std": 1.0}
                continue
            mean = float(np.mean(valid_vals))
            std = float(np.std(valid_vals))
            if not np.isfinite(mean) or not np.isfinite(std) or std == 0.0:
                group_out[fname] = {"mean": 0.0, "std": 1.0}
            else:
                group_out[fname] = {"mean": mean, "std": std}
        if group_out:
            out[group_name] = group_out
    return out


# --------------------------------------------------------------------------- #
# compute_class_dict
# --------------------------------------------------------------------------- #
def compute_class_dict(
    data: dict[str, np.ndarray],
    schema: dict | Schema | None = None,
    flags: dict[str, bool] | None = None,
) -> dict:
    """Schema-length, data-counted class weights.

    For each ``label`` field, emit exactly one weight per SCHEMA-DECLARED class
    (length+order from ``schema.classes``, resolved per active flags), so the
    list length equals the consuming head's ``output_size``. Counts come from
    the produced VALID data (inverse-frequency, excluding -1 fill and
    ``valid==False`` slots). Any schema class with zero observed count -> 0.0.
    """
    if schema is None:
        raise ValueError("compute_class_dict requires a schema for length/order guarantees")
    sch = as_schema(schema)
    flags = flags or {}
    out: dict[str, dict[str, list[float]]] = {}

    for g in sch.groups:
        if g.alias_of is not None:
            # aliases inherit the same labels; only emit if present in data
            if g.name not in data:
                continue
            src = sch.group(g.alias_of)
            label_fields = [f for f in src.fields if isinstance(f, LabelField)]
            group_name_for_arr = g.name
        else:
            if g.name not in data:
                continue
            label_fields = [f for f in g.fields if isinstance(f, LabelField)]
            group_name_for_arr = g.name

        if not label_fields:
            continue
        arr = data[group_name_for_arr]
        is_constituent = "valid" in arr.dtype.names
        valid_mask = arr["valid"] if is_constituent else None

        group_out: dict[str, list[float]] = {}
        for fld in label_fields:
            classes = fld.resolved_classes(flags)
            col = np.asarray(arr[fld.name])
            if is_constituent:
                col = col[valid_mask]
            col = col.reshape(-1)
            # exclude invalid fill
            fill = fld.invalid_fill if fld.invalid_fill is not None else sch.fill_int
            col = col[col != fill]

            counts = {c: int(np.sum(col == c)) for c in classes}
            total = int(sum(counts.values()))
            n_classes = len(classes)
            weights: list[float] = []
            for c in classes:
                cnt = counts[c]
                if cnt == 0:
                    log.debug("class %s of %s.%s has zero count -> weight 0.0", c, g.name, fld.name)
                    weights.append(0.0)
                else:
                    weights.append(total / (n_classes * cnt))
            group_out[fld.name] = weights
        if group_out:
            out[g.name] = group_out
    return out
