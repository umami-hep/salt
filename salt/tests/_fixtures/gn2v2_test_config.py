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
from salt.tests._fixtures.gn2v2_fixture import JET_VARIABLES, TRACK_VARIABLES

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
    "data": {
        "batch_size": 100,
        # Shipped configs assume large training machines; a fixture-sized run
        # must not fork eight workers per DataLoader.
        "num_workers": 0,
        # The shipped config sources its files through an InputSamples module
        # pointing at ${DATA_*_PATH}. Tests hand over a single fixture file via
        # --data.train_file, so the module is deleted and the plain file path is
        # used instead. Without this the derived config ignores the fixture.
        "modules": {"input_samples": None},
    },
}

# The input surface the salt fixture generators actually write.
#
# The shipped config names the two track significances the way the ATLAS
# open-data sample does (`lifetimeSigned*`); `write_dummy_file` derives its track
# block from the parity norm dict, i.e. from `TRACK_VARIABLES`, which uses the
# v1 GN2 names (`IP3D_signed_*`). A config demanding a field the fixture file
# does not carry fails at schema resolution, so the derived config declares the
# variables the fixture provides — this list, not the shipped one, is what the
# suites' width assertions (19 track features) are anchored to.
FIXTURE_VARIABLES: dict[str, Any] = {
    "data": {
        "modules": {
            "features": {
                "init_args": {
                    "variables": {
                        "jets": list(JET_VARIABLES),
                        "tracks": list(TRACK_VARIABLES),
                    }
                }
            }
        }
    }
}

# The head shape the committed output goldens are anchored to: a 3-class
# flavour head and a track_vertexing deferred out of TEST. The shipped config
# carries 4 classes (it has taujets) and exposes vertexing everywhere, so
# without this the H5 column table and ONNX tuple would both move and every
# golden would have to be recaptured. The goldens are an oracle for the outputs
# MACHINERY, not for a shipped model, so their subject is pinned here and the
# shipped config stays free to change.
GOLDEN_HEADS: dict[str, Any] = {
    # H5 column prefixes and ONNX output names are both derived from the run
    # name, so reproducing the goldens means pinning it too.
    "name": "GN2v2_dummy",
    "export": {"model_name": "GN2v2dummy"},
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

class _ExplicitNull:
    """A key that must be WRITTEN as a YAML null, not removed.

    `_deep_merge` follows the jsonargparse cross-config rule where a null patch
    value DELETES the key. Some keys need the opposite: `expose: null` is a
    value `VertexingTaskModule` reads to mean "every mode", so dropping the key
    leaves the module on its default and the head silently disappears from
    TEST/ONNX. Patches that mean the value carry this sentinel instead of None.
    """

    def __repr__(self) -> str:
        return "NULL"


NULL = _ExplicitNull()

_CACHE: dict[str, Path] = {}


def _deep_merge(base: Any, patch: Any) -> Any:
    """jsonargparse cross-config semantics: dicts merge, null deletes, lists replace."""
    if not isinstance(base, dict) or not isinstance(patch, dict):
        return patch
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, _ExplicitNull):
            out[key] = None
        elif value is None and key in out:
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
    return derive("gn2v2-small", SMALL, FIXTURE_VARIABLES, GOLDEN_HEADS, SPLIT_OUTPUTS)


def full_family_config() -> Path:
    """As ``small_config``, but one RunTaskOutput over all three heads.

    The drop-in replacement for the former ``gn2v2-dummy-cutover34.yaml``.
    """
    return derive(
        "gn2v2-small-full-family",
        SMALL,
        FIXTURE_VARIABLES,
        GOLDEN_HEADS,
        {
            # NULL, not None: `expose: null` means "every mode" and must reach
            # the written config. Deleting the key instead leaves the deferral
            # from GOLDEN_HEADS in place, and run_tasks then demands a
            # preds.tracks.track_vertexing that no module produces in TEST/ONNX.
            "model": {
                "init_args": {
                    "modules": {"track_vertexing": {"init_args": {"expose": NULL}}}
                }
            },
            "outputs": {
                "jets_out": NULL,
                "origin_out": NULL,
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
        FIXTURE_VARIABLES,
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
    """CLI overrides pointing a derived config at a fixture file."""
    return [
        f"--data.train_file={h5}",
        f"--data.val_file={h5}",
        f"--data.modules.reader.init_args.schema={schema}",
        f"--model.modules.norm.init_args.norm_dict={norm_dict}",
        # shipped configs assume large training machines
        "--data.num_workers=0",
    ]
