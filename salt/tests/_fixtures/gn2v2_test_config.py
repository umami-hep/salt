"""The GN2v2 test config, derived from the shipped one rather than duplicated.

Shipped configs are real configs. Where a test needs something smaller, or a
different ``outputs:`` shape, it derives it here from
``salt/configs/gn2v2-opendata.yaml`` instead of a parallel toy config living in
the shipped tree.

The shrink is modest on purpose: with ``fast_dev_run``/``limit_*_batches`` and a
small ``batch_size`` the cost of these tests is dominated by import and config
parsing, not by the model. Measured on the shipped configs, a 256-dim 4-layer
model fits in ~6.7 s against ~5.1 s for a 16-dim 2-layer one — so the shrink
buys around a second and a half, and exists only to keep a full parametrised
sweep brisk.

``derive()`` materialises into a per-process temp directory so the result is
usable from module scope (several suites need the path in a ``parametrize``
decorator, which runs at import time).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import yaml

from salt.main import CONFIG_DIR

SHIPPED = CONFIG_DIR / "gn2v2-opendata.yaml"

# Encoder/embedding widths for tests. The graph shape under test is invariant to
# these; only the wall-clock changes.
SMALL: dict[str, Any] = {
    "model": {
        "init_args": {
            "modules": {
                "track_embed": {"init_args": {"out_dim": 16}},
                "encoder": {
                    "init_args": {
                        "dim": 16,
                        "out_dim": 16,
                        "num_layers": 2,
                        "attention": {"num_heads": 2, "attn_type": "torch-math"},
                    }
                },
            }
        }
    },
    "data": {"batch_size": 100},
}

# The head shape the committed output goldens are anchored to: a 3-class
# flavour head and a track_vertexing deferred out of TEST. The shipped config
# carries 4 classes (it has taujets) and exposes vertexing everywhere, so
# without this the H5 column table and ONNX tuple would both move and every
# golden would have to be recaptured. The goldens are an oracle for the outputs
# MACHINERY, not for a shipped model, so their subject is pinned here and the
# shipped config stays free to change.
GOLDEN_HEADS: dict[str, Any] = {
    "model": {
        "init_args": {
            "modules": {
                "jets_classification": {
                    "init_args": {"class_names": ["bjets", "cjets", "ujets"]}
                },
                "track_vertexing": {"init_args": {"expose": ["fit", "val"]}},
            }
        }
    }
}

# The mode-split outputs shape: the flavour head serialised to H5 + ONNX, the
# track heads H5-only. Distinct from the shipped single-RunTaskOutput form, and
# exercised by the sink/inference suites.
SPLIT_OUTPUTS: dict[str, Any] = {
    "outputs": {
        "run_tasks": None,
        # no `modes:` — the default is test + export, i.e. eval H5 AND ONNX
        "jets_out": {
            "class_path": "salt.outputs.RunTaskOutput",
            "init_args": {"tasks": ["jets_classification"]},
        },
        # eval H5 only ('export' is the ONNX mode's name, not 'onnx')
        "origin_out": {
            "class_path": "salt.outputs.RunTaskOutput",
            "init_args": {"tasks": ["track_origin"], "modes": ["test"]},
        },
    }
}

_CACHE: dict[str, Path] = {}


def _deep_merge(base: Any, patch: Any) -> Any:
    """jsonargparse cross-config semantics: dicts merge, null deletes, lists replace."""
    if not isinstance(base, dict) or not isinstance(patch, dict):
        return patch
    out = dict(base)
    for key, value in patch.items():
        if value is None and key in out:
            del out[key]
        elif key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def derive(name: str, *patches: dict[str, Any]) -> Path:
    """Write ``gn2v2-opendata.yaml`` with ``patches`` deep-merged over it.

    Parameters
    ----------
    name : str
        Filename stem; also the cache key, so repeated calls in one process
        return the same path.
    *patches : dict[str, Any]
        Config fragments merged left to right over the shipped config.

    Returns
    -------
    Path
        The written config.
    """
    if name in _CACHE:
        return _CACHE[name]
    config = yaml.safe_load(SHIPPED.read_text())
    for patch in patches:
        config = _deep_merge(config, patch)
    out_dir = Path(tempfile.mkdtemp(prefix="salt_test_configs_"))
    path = out_dir / f"{name}.yaml"
    path.write_text(yaml.dump(config, sort_keys=False))
    _CACHE[name] = path
    return path


def small_config() -> Path:
    """The shipped GN2v2, shrunk, on the golden head shape and split outputs.

    The drop-in replacement for the former ``gn2v2-dummy.yaml``.
    """
    return derive("gn2v2-small", SMALL, GOLDEN_HEADS, SPLIT_OUTPUTS)


def full_family_config() -> Path:
    """As ``small_config``, but one RunTaskOutput over all three heads.

    The drop-in replacement for the former ``gn2v2-dummy-cutover34.yaml``.
    """
    return derive(
        "gn2v2-small-full-family",
        SMALL,
        GOLDEN_HEADS,
        {
            "model": {
                "init_args": {
                    "modules": {"track_vertexing": {"init_args": {"expose": None}}}
                }
            },
            "outputs": {
                "jets_out": None,
                "origin_out": None,
                "run_tasks": {
                    "class_path": "salt.outputs.RunTaskOutput",
                    "init_args": {
                        "tasks": [
                            "jets_classification",
                            "track_origin",
                            "track_vertexing",
                        ]
                    },
                },
            },
        },
    )


def h5_only_config() -> Path:
    """As ``small_config``, but test-only so no ONNX tuple is assembled.

    The drop-in replacement for the former ``gn2v2-dummy-cutover.yaml``.
    """
    return derive(
        "gn2v2-small-h5-only",
        SMALL,
        GOLDEN_HEADS,
        SPLIT_OUTPUTS,
        {
            "outputs": {
                "jets_out": {
                    "class_path": "salt.outputs.RunTaskOutput",
                    "init_args": {"modes": ["test"]},
                }
            }
        },
    )


def data_overrides(h5: Path, schema: Path, norm_dict: Path) -> list[str]:
    """CLI overrides pointing the shipped InputSamples sourcing at a fixture file."""
    return [
        f"--data.modules.input_samples.init_args.files.train={h5}",
        f"--data.modules.input_samples.init_args.files.val={h5}",
        f"--data.modules.input_samples.init_args.files.test={h5}",
        f"--data.modules.reader.init_args.schema={schema}",
        f"--model.modules.norm.init_args.norm_dict={norm_dict}",
        # shipped configs assume large training machines
        "--data.num_workers=0",
    ]
