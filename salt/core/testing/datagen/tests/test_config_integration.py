"""CONFIG->RECIPE integration test (CPU-only): validates against the REAL config.

It does NOT trust a recipe-internal variable list. For every (salt config,
recipe) pair in ``config_recipes.yaml`` it generates the mapped recipe via the
modular ``Pipeline`` -> ``H5Writer``, parses the SALT CONFIG ITSELF (the real
``data:`` block: ``variables``, ``input_map``, ``parameters``,
``global_object``, resolving base.yaml inheritance) and the real per-task
``label`` declarations, and constructs the REAL ``salt.data.SaltDataset`` with
those exact variables/labels/global_object/input_map and reads a slice.

``SaltDataset.check_file`` KeyErrors if any declared variable/group is absent
from the generated H5; ``process_labels`` KeyErrors if a declared label column is
absent. So a green parametrisation means each recipe is a genuine
VARIABLE-SUPERSET of every config mapped to it, validated against the config's
ACTUAL declared variables (not a generic subset).

No GPU, no training, no model, no datamodule.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from salt.core.testing.datagen.pipeline import load_pipeline

SaltDataset = pytest.importorskip("salt.data").SaltDataset

# Resolve the datagen package dir, the recipes dir, the salt config roots, and
# the mapping file -- all relative to this test (no hardcoded absolute paths).
_DATAGEN_DIR = Path(__file__).resolve().parent.parent
_RECIPES_DIR = _DATAGEN_DIR / "recipes"
_MAPPING_FILE = _DATAGEN_DIR / "config_recipes.yaml"
# salt package dir = .../salt/  (datagen lives at salt/core/testing/datagen)
_SALT_PKG_DIR = _DATAGEN_DIR.parent.parent.parent
_CONFIG_ROOT = _SALT_PKG_DIR / "configs"
_TEST_CONFIG_ROOT = _SALT_PKG_DIR / "tests" / "configs"
_BASE_CONFIG = _CONFIG_ROOT / "base.yaml"


# --------------------------------------------------------------------------- #
# config parsing -- read the REAL salt config (base.yaml inheritance + tasks)
# --------------------------------------------------------------------------- #
def _resolve_config_path(rel: str, is_test_config: bool) -> Path:
    root = _TEST_CONFIG_ROOT if is_test_config else _CONFIG_ROOT
    p = root / rel
    if not p.exists():  # fallback: try the other root (run_combined-style)
        alt = (_CONFIG_ROOT if is_test_config else _TEST_CONFIG_ROOT) / rel
        if alt.exists():
            return alt
    return p


def _data_block(cfg_path: Path) -> dict:
    """Return the merged ``data:`` block (base.yaml <- config), config wins."""
    base = yaml.safe_load(_BASE_CONFIG.read_text()) or {}
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    base_data = dict(base.get("data", {}) or {})
    cfg_data = dict(cfg.get("data", {}) or {})
    base_data.update(cfg_data)
    return base_data


def _extract_task_labels(cfg_path: Path) -> dict[str, list[str]]:
    """Walk the model tree and collect ``{input_name: [label, ...]}`` for every
    Classification/Vertexing-style task that declares a real ``label`` field.

    Regression ``targets`` are NOT included: they are read model-side by the
    RegressionTask, not by ``SaltDataset.__getitem__``. Derived/placeholder
    labels (``multi_target`` custom targets that are not H5 fields) are skipped.
    """
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    labels: dict[str, list[str]] = {}

    # custom_target names injected by multi_target are NOT real h5 fields
    derived: set[str] = set()
    data = cfg.get("data", {}) or {}
    for entry in data.get("multi_target", []) or []:
        if "custom_target" in entry:
            derived.add(entry["custom_target"])

    def walk(o):
        if isinstance(o, dict):
            cp = str(o.get("class_path", ""))
            if "Task" in cp:
                ia = o.get("init_args", {}) or {}
                inp = ia.get("input_name")
                lab = ia.get("label")
                if inp and lab and lab not in derived:
                    labels.setdefault(inp, [])
                    if lab not in labels[inp]:
                        labels[inp].append(lab)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(cfg)
    return labels


def _load_mapping():
    doc = yaml.safe_load(_MAPPING_FILE.read_text())
    out = []
    for m in doc["mapping"]:
        out.append(
            (
                m["config"],
                m["recipe"],
                dict(m.get("flags", {}) or {}),
                bool(m.get("test_config", False)),
            )
        )
    return out


_MAPPING = _load_mapping()


# --------------------------------------------------------------------------- #
# recipe generation
# --------------------------------------------------------------------------- #
def _generate(recipe: str, flags: dict, out_dir: Path) -> Path:
    pipe = load_pipeline(str(_RECIPES_DIR / f"{recipe}.yaml"))
    pipe.flags = dict(flags)
    # propagate flags to every module so flavour_label / class dicts resolve
    for m in pipe.modules:
        m.flags = dict(flags)
    pipe.set_output_dir(out_dir)
    pipe.run()
    return out_dir / f"{recipe}.h5"


# --------------------------------------------------------------------------- #
# the parametrized integration test
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "config_rel, recipe, flags, is_test_config",
    _MAPPING,
    ids=[m[0] for m in _MAPPING],
)
def test_config_reads_back_against_real_recipe(
    tmp_path, config_rel, recipe, flags, is_test_config
):
    cfg_path = _resolve_config_path(config_rel, is_test_config)
    assert cfg_path.exists(), f"config not found: {config_rel}"

    h5 = _generate(recipe, flags, tmp_path)

    data = _data_block(cfg_path)
    variables = data.get("variables") or {}
    assert variables, f"{config_rel}: empty variables block"
    input_map = data.get("input_map")
    parameters = data.get("parameters")
    global_object = data.get("global_object") or "jets"

    # the config's REAL per-task labels (input_name -> [label,...]); only keep
    # those whose input stream is actually declared in `variables` (so the reader
    # has a group to read them from) -- this still exercises every classification
    # label the config declares for a present stream.
    all_labels = _extract_task_labels(cfg_path)
    labels = {k: v for k, v in all_labels.items() if k in variables or k == "global"}

    ds = SaltDataset(
        filename=str(h5),
        norm_dict={},
        variables=variables,
        stage="train",
        labels=labels or None,
        input_map=input_map,
        parameters=parameters,
        global_object=global_object,
    )

    assert len(ds) > 0, f"{config_rel}: empty dataset"

    # read a real slice -- check_file already ran in __init__; this exercises
    # _setup + read_direct over every declared variable AND label column.
    n = min(64, len(ds))
    inputs, pad_masks, lbls = ds[0:n]

    # --- inputs: every declared variable stream read back with right shape ---
    for stream, feats in variables.items():
        if stream == "parameters":
            # parameters resolve to the global_object group; flat (n, n_params)
            assert inputs["parameters"].shape == (n, len(feats))
            assert inputs["parameters"].dtype == torch.float32
            continue
        resolved = "global" if stream == "global" else stream
        if not feats:
            continue
        assert resolved in inputs, f"{config_rel}: stream {resolved} missing from inputs"
        t = inputs[resolved]
        assert t.dtype == torch.float32
        if t.dim() == 2:  # global group: (n, F)
            assert t.shape == (n, len(feats))
        else:  # constituent: (n, M, F) + bool pad mask
            assert t.shape[0] == n
            assert t.shape[2] == len(feats)
            assert pad_masks[resolved].shape == (n, t.shape[1])
            assert pad_masks[resolved].dtype == torch.bool
            # padded slots zeroed (mask_invalid streams)
            assert (t[pad_masks[resolved]] == 0).all()

    # --- labels: every declared classification label read back as torch.long ---
    for stream, names in labels.items():
        resolved = global_object if stream == "global" else stream
        assert resolved in lbls, f"{config_rel}: labels for {resolved} missing"
        for name in names:
            assert lbls[resolved][name].dtype == torch.long, (
                f"{config_rel}: label {resolved}.{name} not torch.long"
            )
            assert lbls[resolved][name].shape[0] == n


def test_mapping_covers_all_test_suite_recipes():
    """Sanity: the mapping references only recipes that exist on disk and the
    three topologies the suite exercises (flavour_tagger, maskformer, jets_only)."""
    used = {recipe for _, recipe, _, _ in _MAPPING}
    assert used == {"flavour_tagger", "maskformer_truth_hadron", "jets_only"}
    for recipe in used:
        assert (_RECIPES_DIR / f"{recipe}.yaml").exists()
