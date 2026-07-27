"""Shared helpers for the generation modules: ``add_field``, ``infer_n``,
``field_dicts_from_array``, ``reconstruct_schema``.
"""

from __future__ import annotations

import numpy as np

from ..schema import GroupSpec, Schema


def add_field(arr: np.ndarray, name: str, dtype) -> np.ndarray:
    """Return a copy of ``arr`` widened with one new (zero-initialised) field.

    All existing columns are copied across unchanged (same dtype, same values);
    the new column is appended at the end. ``arr`` may be 1-D (global) or 2-D
    (constituent). ``dtype`` is the new column's element dtype.
    """
    new_dtype = np.dtype(arr.dtype.descr + [(name, np.dtype(dtype).str)])
    out = np.zeros(arr.shape, dtype=new_dtype)
    for fname in arr.dtype.names:
        out[fname] = arr[fname]
    return out


def infer_n(data: dict[str, np.ndarray], explicit: int | None) -> int:
    """Resolve the row count: explicit override, else infer from existing data.

    Modules receive their pipeline-level ``n_samples`` injected onto ``self``
    before ``__call__``. This helper prefers that explicit value; if it is
    ``None`` (heterogeneous edge case) it falls back to the row count of the
    first existing group in ``data``.
    """
    if explicit is not None:
        return int(explicit)
    if data:
        first = next(iter(data.values()))
        return int(first.shape[0])
    raise ValueError("infer_n: no explicit n_samples and no existing data to infer from")


def field_dicts_from_array(arr: np.ndarray) -> list[dict]:
    """Reconstruct dict-form *distribution* field specs from a structured dtype.

    Used by ``TruthHadronInserter`` to rebuild the existing ``tracks`` group's
    field list so it can append the link field and re-parse the whole group via
    ``parse_schema``. Preserves each column's name + dtype; ``valid`` is excluded
    (the engine re-appends it for constituent groups).

    Every existing column (including labels/ids already drawn) is emitted as a
    ``distribution`` spec purely to carry the right dtype -- the reconstructed
    schema's ``tracks`` group is used ONLY for ``_resolve_link`` (which reads
    ``g.name`` and the LinkField), never to redraw values: the inserter passes
    the already-built widened array into ``_resolve_link`` directly.
    """
    specs: list[dict] = []
    for fname in arr.dtype.names:
        if fname == "valid":
            continue
        specs.append({
            "name": fname,
            "type": "distribution",
            "dtype": np.dtype(arr.dtype[fname]).str,
        })
    return specs


def reconstruct_schema(
    group_specs: list[GroupSpec],
    n_samples: int,
    fill_float: float = float("nan"),
    fill_int: int = -1,
    file_attrs: dict | None = None,
) -> Schema:
    """Build a thin ``Schema`` from a list of already-parsed ``GroupSpec``s.

    Hands the writers exactly what ``write_h5`` / ``compute_norm_dict`` /
    ``compute_class_dict`` need (class_names attrs + the
    class_dict-length==output_size guarantee) without re-running schema
    validation (the specs were already parsed via ``parse_schema``).
    """
    return Schema(
        n_samples=n_samples,
        groups=list(group_specs),
        fill_float=fill_float,
        fill_int=fill_int,
        file_attrs=dict(file_attrs or {"config": "{}"}),
    )
