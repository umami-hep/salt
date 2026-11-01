"""The synthetic salt-v1 training file for tests: ``write_dataset`` over ``SALT_V1_SCHEMA``."""

from __future__ import annotations

from pathlib import Path

from salt.testing.datagen import SALT_V1_SCHEMA, write_dataset

__all__ = ["FLAGS", "dummy_h5"]

FLAGS = frozenset({"make_xbb", "inc_taus", "inc_params", "is_gn3"})


def dummy_h5(
    dir: Path, *, name: str = "pp_output_train.h5", seed: int | None = None, **flags: bool
) -> tuple[Path, Path, Path]:
    """Write ``dir/name`` + its computed ``dummy_norm_dict.yaml``/``dummy_class_dict.yaml``.

    The dict names are distinct from ``write_parity_norm_dict``'s ``norm_dict.yaml``/
    ``class_dict.yaml`` so both can share one directory. ``flags`` are the schema's
    build-time switches (``make_xbb``/``inc_taus``/``inc_params``/``is_gn3``).
    """
    unknown = set(flags) - FLAGS
    if unknown:
        raise TypeError(f"dummy_h5() got unknown flag(s) {sorted(unknown)}; known: {sorted(FLAGS)}")
    h5, nd, cd = dir / name, dir / "dummy_norm_dict.yaml", dir / "dummy_class_dict.yaml"
    write_dataset(
        h5, SALT_V1_SCHEMA, flags=flags or None, norm_dict_path=nd, class_dict_path=cd, seed=seed
    )
    return h5, nd, cd
