"""The two-phase, dependency-sorted generation engine: independent fields per
group, then topological ``link`` resolution, then ``alias_of`` materialisation.
"""

from __future__ import annotations

import numpy as np

from .schema import (
    DIST_DEFAULTS,
    DistributionField,
    GroupSpec,
    IdField,
    LabelField,
    LinkField,
    Schema,
    as_schema,
    parse_nan_where,
    topo_sort_link_groups,
)


def _np_dtype(dtype_str: str) -> np.dtype:
    return np.dtype(dtype_str)


def _is_float_dtype(dtype_str: str) -> bool:
    return np.dtype(dtype_str).kind == "f"


def _build_valid_mask(
    rng: np.random.Generator, n: int, m: int, valid_fraction: float, min_valid: int
) -> np.ndarray:
    """Boolean [n, m] mask, valid entries sorted to the front of each sample."""
    valid = rng.choice([True, False], size=(n, m), p=[valid_fraction, 1.0 - valid_fraction])
    # np.sort puts False(0) before True(1); reverse -> valid (True) packed to front
    valid = np.sort(valid, axis=-1)[:, ::-1]
    if min_valid >= 1:
        valid[:, : min(min_valid, m)] = True
    return np.ascontiguousarray(valid)


def _draw_distribution(
    rng: np.random.Generator, fspec: DistributionField, shape: tuple[int, ...]
) -> np.ndarray:
    params = {**DIST_DEFAULTS[fspec.dist], **fspec.params}
    if fspec.dist == "normal":
        out = rng.normal(params["mean"], params["std"], shape)
    elif fspec.dist == "uniform":
        out = rng.uniform(params["low"], params["high"], shape)
    elif fspec.dist == "lognormal":
        out = rng.lognormal(params["mean"], params["sigma"], shape)
    elif fspec.dist == "exponential":
        out = rng.exponential(params["scale"], shape)
    else:  # constant
        out = np.full(shape, params["value"])
    return out.astype(_np_dtype(fspec.dtype))


def _draw_label(
    rng: np.random.Generator,
    fspec: LabelField,
    shape: tuple[int, ...],
    flags: dict[str, bool],
) -> np.ndarray:
    sample_classes = fspec.resolved_sample_classes(flags)
    p = None
    if fspec.weights is not None and len(fspec.weights) == len(sample_classes):
        w = np.asarray(fspec.weights, dtype=float)
        p = w / w.sum()
    out = rng.choice(np.asarray(sample_classes), size=shape, p=p)
    if fspec.sorted and out.ndim == 2:
        out = np.sort(out, axis=-1)[:, ::-1]
    return out.astype(_np_dtype(fspec.dtype))


def _draw_id(
    rng: np.random.Generator, fspec: IdField, shape: tuple[int, ...]
) -> np.ndarray:
    lo, hi = fspec.range
    if fspec.scope == "global":
        n = int(np.prod(shape))
        if fspec.unique:
            pool = np.arange(lo, hi)
            if n > pool.size:
                raise ValueError(
                    f"id field cannot draw {n} unique global ids from range [{lo},{hi})"
                )
            vals = rng.choice(pool, size=n, replace=False)
        else:
            vals = rng.integers(lo, hi, size=n)
        return vals.reshape(shape).astype(_np_dtype(fspec.dtype))

    # scope == jet: unique per row
    n, m = shape
    if fspec.unique:
        span = hi - lo
        if m > span:
            raise ValueError(
                f"id field cannot draw {m} unique per-sample ids from range [{lo},{hi})"
            )
        out = np.empty((n, m), dtype=np.int64)
        base = np.arange(lo, hi)
        for i in range(n):
            out[i] = rng.choice(base, size=m, replace=False)
    else:
        out = rng.integers(lo, hi, size=(n, m))
    return out.astype(_np_dtype(fspec.dtype))


def _fill_value(fspec, schema: Schema):
    """Resolve the invalid-fill sentinel for a field."""
    if fspec.invalid_fill is not None:
        return fspec.invalid_fill
    if _is_float_dtype(fspec.dtype):
        return schema.fill_float
    return schema.fill_int


def _build_group_array(
    rng: np.random.Generator,
    schema: Schema,
    g: GroupSpec,
    flags: dict[str, bool],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Phase-1 build of one (non-alias) group's structured array.

    Returns ``(structured_array, valid_mask_or_None)``. Link fields are
    allocated (zeros) here and resolved later in phase 2.
    """
    n = schema.n_samples
    if g.is_constituent:
        m = int(g.max_items)
        shape: tuple[int, ...] = (n, m)
        valid = _build_valid_mask(rng, n, m, g.valid_fraction, g.min_valid)
    else:
        shape = (n,)
        valid = None

    # build structured dtype (field order preserved; valid appended for constituents)
    dtype_fields = [(f.name, _np_dtype(f.dtype)) for f in g.fields]
    if g.is_constituent:
        dtype_fields.append(("valid", np.bool_))
    arr = np.zeros(shape, dtype=np.dtype(dtype_fields))

    # draw fields
    for f in g.fields:
        if isinstance(f, DistributionField):
            arr[f.name] = _draw_distribution(rng, f, shape)
        elif isinstance(f, LabelField):
            arr[f.name] = _draw_label(rng, f, shape, flags)
        elif isinstance(f, IdField):
            arr[f.name] = _draw_id(rng, f, shape)
        elif isinstance(f, LinkField):
            pass  # resolved in phase 2

    # nan_where (applied after labels drawn, on valid slots too -- label-driven)
    for f in g.fields:
        if isinstance(f, DistributionField) and f.nan_where is not None:
            parsed = parse_nan_where(f.nan_where)
            ref_field, literal = parsed
            col = arr[f.name]
            col[arr[ref_field] == literal] = np.nan
            arr[f.name] = col

    if g.is_constituent:
        arr["valid"] = valid
        if g.mask_invalid:
            _apply_invalid_fill(arr, valid, schema, g)

    return arr, valid


def _apply_invalid_fill(
    arr: np.ndarray, valid: np.ndarray, schema: Schema, g: GroupSpec
) -> None:
    """Apply per-field invalid-fill sentinels to invalid constituent slots.

    Link fields are skipped here -- they are filled by the resolver in phase 2.
    """
    mask = ~valid
    for f in g.fields:
        if isinstance(f, LinkField):
            continue
        fill = _fill_value(f, schema)
        col = arr[f.name]
        col[mask] = fill
        arr[f.name] = col


def _resolve_link(
    rng: np.random.Generator,
    schema: Schema,
    data: dict[str, np.ndarray],
    g: GroupSpec,
    f: LinkField,
    flags: dict[str, bool],
) -> None:
    """Resolve one ``link`` field against its referenced ``id`` field."""
    n = schema.n_samples
    src = data[g.name]
    src_valid = src["valid"] if "valid" in src.dtype.names else np.ones(src.shape, bool)

    ref_group = schema.group(f.ref_group)
    ref_arr = data[f.ref_group]
    ref_ids = ref_arr[f.ref_field]
    if ref_group.kind == "constituent":
        ref_valid = ref_arr["valid"]
    else:
        ref_valid = np.ones(ref_arr.shape, bool)

    unmatched = f.unmatched_value if f.unmatched_value is not None else schema.fill_int
    invalid_fill = f.invalid_fill if f.invalid_fill is not None else schema.fill_int

    out = src[f.name].copy()
    m = out.shape[1]

    for i in range(n):
        # valid referenced ids in sample i
        if ref_group.kind == "constituent":
            if f.select_over == "all_slots":
                pool = np.atleast_1d(ref_ids[i])  # all slots (incl invalid -1)
            else:
                pool = np.atleast_1d(ref_ids[i][ref_valid[i]])
        else:
            # global reference: single id of sample i (valid unless it is the fill)
            single = ref_ids[i]
            pool = np.array([single]) if single != schema.fill_int else np.array([])

        # required_match invariant: phase 1 guarantees >=1 valid referent per
        # sample for a constituent reference (see _propagate_required_match_min_valid
        # in schema.py). If the pool is still empty here something upstream is
        # inconsistent -- fail loudly rather than promoting an invalid-filled slot
        # (whose id is the fill sentinel) to "valid".
        if f.required_match and pool.size == 0 and ref_group.kind == "constituent":
            raise RuntimeError(
                f"required_match link {g.name}.{f.name} -> {f.ref_group}.{f.ref_field}: "
                f"no valid referent in sample {i} despite min_valid>=1 guarantee. "
                f"This indicates a schema-resolution bug (the referenced group's "
                f"min_valid was not propagated)."
            )

        valid_slots = np.where(src_valid[i])[0]
        for t in valid_slots:
            if f.unmatched_fraction > 0 and rng.random() < f.unmatched_fraction:
                out[i, t] = unmatched
            elif pool.size > 0:
                out[i, t] = pool[rng.integers(0, pool.size)]
            else:
                out[i, t] = unmatched
        # invalid source slots -> invalid fill
        out[i, ~src_valid[i]] = invalid_fill

    src[f.name] = out


def generate_data(
    schema: dict | Schema,
    flags: dict[str, bool] | None = None,
) -> dict[str, np.ndarray]:
    """Schema -> {group_name: structured ndarray}. Pure, no IO.

    ``flags`` resolves build-time switches (make_xbb / is_gn3 / inc_taus /
    inc_params). Deterministic given ``schema.seed``.
    """
    schema = as_schema(schema)
    flags = flags or {}
    rng = np.random.default_rng(schema.seed)

    data: dict[str, np.ndarray] = {}

    # Phase 1: build all non-alias, emitted groups in schema order
    for g in schema.groups:
        if g.alias_of is not None:
            continue
        if g.emit_if is not None and not flags.get(g.emit_if):
            continue
        arr, _ = _build_group_array(rng, schema, g, flags)
        data[g.name] = arr

    # Phase 2: resolve link fields in topological order
    for gname in topo_sort_link_groups(schema):
        if gname not in data:
            continue
        g = schema.group(gname)
        for f in g.fields:
            if isinstance(f, LinkField):
                _resolve_link(rng, schema, data, g, f, flags)

    # Finally: materialise alias groups (post-link-resolution copy)
    for g in schema.groups:
        if g.alias_of is None:
            continue
        if g.alias_of not in data:
            continue
        data[g.name] = data[g.alias_of].copy()

    return data
