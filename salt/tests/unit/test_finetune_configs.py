"""Parse-level validation of the tutorial fine-tuning base + overlay configs.

`gn3large_base.yaml` is the in-repo mirror of the PUBLIC GN3Large bundle; the
four `finetune_gn3large_add_*.yaml` / `finetune_gn3large_xbb_transfer.yaml`
overlays stack on it — worked examples 1-4 in `docs/tutorials/finetuning.md`
(add the `calo` stream, add a b/c-charge jet head, add `mass` to the `jets`
stream, then a full backbone transfer to the boosted-Xbb sample). These are
tutorial configs shipped under `docs/tutorials/configs/finetuning/`, not
shipped model configs, so they carry no pipeline fixture. Their norm dicts
live under `docs/tutorials/configs/finetuning/norm_dicts/`. No model
instantiated, no checkpoint loaded; the H5-schema and DOT-reproducibility
checks skip when their inputs are absent.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml

from salt.main import CONFIG_DIR
from salt.merge_config import main as merge_config_main
from salt.schedule import TrainingSchedule

# CONFIG_DIR is <repo>/salt/configs, so .parents[1] is the repo root. The
# tutorial's fine-tuning configs live under docs/, not salt/configs/.
TUTORIAL_CONFIG_DIR = CONFIG_DIR.parents[1] / "docs" / "tutorials" / "configs" / "finetuning"

# Its module names are the freeze-spec universe for every overlay below.
BASE = TUTORIAL_CONFIG_DIR / "gn3large_base.yaml"

# Worked examples 1-4: add calo, add a charge head, add a jets variable, and
# transfer the backbone to the boosted-Xbb sample.
OVERLAY_CALO = "finetune_gn3large_add_calo.yaml"
OVERLAY_CHARGE = "finetune_gn3large_add_charge_head.yaml"
OVERLAY_JETVARS = "finetune_gn3large_add_jet_vars.yaml"
OVERLAY_XBB = "finetune_gn3large_xbb_transfer.yaml"

# The three FTAG overlays (calo/charge/jetvars) stack on the same p7085 sample
# as the base; the Xbb overlay changes both sample and task.
FTAG_OVERLAYS = (OVERLAY_CALO, OVERLAY_CHARGE, OVERLAY_JETVARS)
ALL_OVERLAYS = (OVERLAY_CALO, OVERLAY_CHARGE, OVERLAY_JETVARS, OVERLAY_XBB)

NORM_DICTS_DIR = TUTORIAL_CONFIG_DIR / "norm_dicts"
NORM_DICT_CALO = NORM_DICTS_DIR / "norm_dict_p7085_calo.yaml"
NORM_DICT_JETS_EXTRA = NORM_DICTS_DIR / "norm_dict_p7085_jets_extra.yaml"

# The base's 11 parameter-bearing modules: backbone (shared representation) +
# the 5 task heads. PARAM_FREE modules carry no weights of their own (a plain
# list restatement / a loss wrapper), so they are neither "loaded" nor "new"
# in an `--init_from` accounting.
HEADS: frozenset[str] = frozenset({
    "jets_classification",
    "track_origin",
    "track_vertexing",
    "track_type",
    "jet_pt_regression",
})
PARAM_FREE: frozenset[str] = frozenset({"concat", "split", "loss"})
BACKBONE: frozenset[str] = frozenset({
    "norm",
    "track_embed",
    "flow_embed",
    "electron_embed",
    "encoder",
    "pool",
})

# docs/assets/finetuning/ — committed freeze graphs, produced by
# docs/tutorials/finetuning_render_graphs.sh. CONFIG_DIR is <repo>/salt/configs,
# so .parents[1] is the repo root.
ASSETS_DIR = CONFIG_DIR.parents[1] / "docs" / "assets" / "finetuning"


pytestmark = pytest.mark.usefixtures("_no_comet")


@pytest.fixture
def _no_comet(monkeypatch):
    # keyless CI: _wire_experiment_logger already forces the logger offline
    # with no COMET_API_KEY, but don't let a developer's cached key change
    # what these parse-only tests exercise.
    monkeypatch.delenv("COMET_API_KEY", raising=False)


def _load(name: str) -> dict:
    path = TUTORIAL_CONFIG_DIR / name
    assert path.is_file(), f"shipped tutorial config missing: {path}"
    return yaml.safe_load(path.read_text())


def _load_shipped(name: str) -> dict:
    """Load a PRODUCTION config from `salt/configs`.

    The tutorial overlays moved to `docs/tutorials/configs/finetuning/`, but the
    shipped models they compare themselves against did not. Anything under
    `salt/configs` is read through here, not through `_load`.
    """
    path = CONFIG_DIR / name
    assert path.is_file(), f"shipped production config missing: {path}"
    return yaml.safe_load(path.read_text())


def _base_module_names() -> list[str]:
    cfg = yaml.safe_load(BASE.read_text())
    return list(cfg["model"]["init_args"]["modules"])


def _resolve(class_path: str) -> type:
    module_path, _, cls_name = class_path.rpartition(".")
    return getattr(importlib.import_module(module_path), cls_name)


def _iter_class_paths(node: Any):
    """Yield every ``class_path`` string found anywhere under `node`."""
    if isinstance(node, dict):
        class_path = node.get("class_path")
        if isinstance(class_path, str):
            yield class_path
        for value in node.values():
            yield from _iter_class_paths(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_class_paths(item)


def _header_and_body(text: str) -> tuple[str, str]:
    """Split `text` into its leading run of ``#``-prefixed lines and the rest."""
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    return "".join(lines[:i]), "".join(lines[i:])


def _merged(tmp_path: Path, *overlays: str) -> tuple[dict, Path]:
    """Stack `BASE` + `overlays` through the real ``salt merge-config`` parser.

    Runs with `tmp_path` as cwd so the base's relative `norm_dict_v2.yaml`
    resolves harmlessly (merge-config is parse-only — it never opens the file).
    Returns the yaml-loaded merged dict and the merged-YAML output path; the
    per-stage ``.dot`` files land next to it, named
    ``<output.stem>_stage{NN}_{name}.dot``.
    """
    output = tmp_path / "merged.yaml"
    args = ["--config", str(BASE)]
    for overlay in overlays:
        args += ["--config", str(TUTORIAL_CONFIG_DIR / overlay)]
    args += ["--merged.output", str(output), "--merged.plots", "false"]
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path)
        merge_config_main(args)
    return yaml.safe_load(output.read_text()), output


class TestBase:
    """`gn3large_base.yaml` — the regenerated public GN3Large bundle mirror."""

    def test_header_and_body_sha256_self_consistent(self):
        text = BASE.read_text()
        assert text.startswith("#"), "gn3large_base.yaml must open with a header comment"
        header, body = _header_and_body(text)
        digest = hashlib.sha256(body.encode()).hexdigest()
        sha_lines = [
            line for line in header.splitlines() if line.strip().startswith("# body-sha256:")
        ]
        assert len(sha_lines) == 1, f"expected one '# body-sha256:' header line, got {sha_lines}"
        recorded = sha_lines[0].split("# body-sha256:", 1)[1].strip()
        assert recorded == digest, "header body-sha256 does not match the actual body hash"
        assert body.splitlines()[0] == "name: GN4_big"

    def test_no_salt_core_strings(self):
        assert "salt.core." not in BASE.read_text()

    def test_no_export_callbacks_or_include(self):
        cfg = yaml.safe_load(BASE.read_text())
        assert "export" not in cfg
        assert "callbacks" not in cfg
        assert "include" not in cfg

    def test_norm_dict_is_the_relative_sibling(self):
        cfg = yaml.safe_load(BASE.read_text())
        norm = cfg["model"]["init_args"]["modules"]["norm"]["init_args"]
        assert norm["norm_dict"] == "norm_dict_v2.yaml"

    def test_base_alone_merges_and_desugars_to_fit(self, tmp_path):
        merged, output = _merged(tmp_path)
        assert merged["training_schedule"]["stages"] == {"fit": {}}
        encoder = merged["model"]["init_args"]["modules"]["encoder"]["init_args"]
        assert encoder["dim"] == 1024
        assert encoder["num_layers"] == 8
        assert merged["model"]["init_args"]["optimizer"] == "AdamW"
        stem = output.with_suffix("")
        assert Path(f"{stem}_stage00_fit.dot").is_file()

    def test_bundle_round_trip(self):
        default_bundle_dir = (
            "/data/ccra-data/projects/salt-improvements/studies/2026_06_11_modularise-salt/"
            "studies/2026_07_19_fine-tuning-approaches/experiments/"
            "13_code_docs-finetuning-public-baseline/bundle"
        )
        bundle_dir = Path(os.environ.get("SALT_GN3LARGE_BUNDLE_DIR", default_bundle_dir))
        bundle = bundle_dir / "config_v2.yaml"
        if not bundle.is_file():
            pytest.skip(f"public bundle copy not found at {bundle}")
        base_text = BASE.read_text()
        _, base_body = _header_and_body(base_text)
        assert yaml.safe_load(bundle.read_text()) == yaml.safe_load(base_text)
        assert bundle.read_text() == base_body


class TestRepoGuard:
    def test_no_salt_core_class_paths_anywhere(self):
        offenders = [
            str(path)
            for root in (CONFIG_DIR, TUTORIAL_CONFIG_DIR)
            for path in sorted(root.rglob("*.yaml"))
            for class_path in _iter_class_paths(yaml.safe_load(path.read_text()) or {})
            if class_path.startswith("salt.core.")
        ]
        assert not offenders, f"salt.core. class_path(s) found in: {offenders}"


class TestOverlaysShape:
    @pytest.mark.parametrize("name", ALL_OVERLAYS)
    def test_no_top_level_include(self, name):
        cfg = _load(name)
        assert "include" not in cfg

    @pytest.mark.parametrize("name", ALL_OVERLAYS)
    def test_precision_is_16_mixed(self, name):
        cfg = _load(name)
        assert cfg["trainer"]["precision"] == "16-mixed"

    @pytest.mark.parametrize("name", FTAG_OVERLAYS)
    def test_no_name_override(self, name):
        # calo/charge/jetvars keep the base's `name: GN4_big` so pre- and
        # post-eval H5 columns line up with the base's own eval columns;
        # only the Xbb overlay (a real domain/task change) sets its own name.
        cfg = _load(name)
        assert "name" not in cfg


class TestNormDicts:
    """`docs/tutorials/configs/finetuning/norm_dicts/` — shipped alongside the overlays."""

    def test_calo_dict_shape_and_stats(self):
        cfg = yaml.safe_load(NORM_DICT_CALO.read_text())
        assert set(cfg) == {"calo"}
        calo = cfg["calo"]
        assert len(calo) == 47
        for name, stats in calo.items():
            mean, std = stats["mean"], stats["std"]
            assert math.isfinite(mean), f"{name}: mean is not finite"
            assert math.isfinite(std), f"{name}: std is not finite"
            assert std != 0, f"{name}: std is zero"

        overlay_calo_vars = _load(OVERLAY_CALO)["data"]["modules"]["features"]["init_args"][
            "variables"
        ]["calo"]
        assert list(calo) == overlay_calo_vars

    def test_jets_extra_dict_groups_are_supersets_of_the_base(self):
        cfg = yaml.safe_load(NORM_DICT_JETS_EXTRA.read_text())
        assert set(cfg) == {"jets", "tracks", "flows", "electrons"}
        base_vars = yaml.safe_load(BASE.read_text())["data"]["modules"]["features"]["init_args"][
            "variables"
        ]
        for stream in ("jets", "tracks", "flows", "electrons"):
            assert set(base_vars[stream]) <= set(cfg[stream]), (
                f"norm_dict_p7085_jets_extra.yaml's {stream!r} group must cover every "
                f"base variable for that stream"
            )
        assert {"pt_btagJes", "eta_btagJes"} <= set(cfg["jets"])
        # Deliberately NOT asserted: coverage of `mass` (the jets-overlay's new
        # variable). At this commit the shipped file is
        # the bundle's norm_dict_v2.yaml body verbatim, with the new jets
        # stats appended by the parent experiment's `data` stage in the
        # closing commit — see test_jets_extra_header_is_marked_placeholder.

    def test_every_norm_dict_starts_with_a_header(self):
        files = sorted(NORM_DICTS_DIR.glob("*.yaml"))
        assert files, f"no norm dicts found under {NORM_DICTS_DIR}"
        for path in files:
            assert path.read_text().startswith("#"), f"{path} must open with a header comment"

    def test_jets_extra_dict_covers_every_overlay_variable(self):
        # Replaces the earlier header-says-PLACEHOLDER check, which guarded the
        # file while `jets.mass` was still missing. The dict is complete now, so
        # the real requirement is the one `Normaliser.preflight` enforces: every
        # variable the overlay configures has a finite, non-degenerate stat.
        cfg = yaml.safe_load(NORM_DICT_JETS_EXTRA.read_text())
        overlay_vars = _load(OVERLAY_JETVARS)["data"]["modules"]["features"]["init_args"][
            "variables"
        ]
        for stream, names in overlay_vars.items():
            assert stream in cfg, f"norm dict has no {stream!r} group"
            for name in names:
                assert name in cfg[stream], (
                    f"{stream}.{name} is configured by {OVERLAY_JETVARS} but has no "
                    f"{{mean, std}} entry -- Normaliser.preflight would fail on it"
                )
                mean, std = cfg[stream][name]["mean"], cfg[stream][name]["std"]
                assert math.isfinite(mean), f"{stream}.{name}: mean is not finite"
                assert math.isfinite(std), f"{stream}.{name}: std is not finite"
                assert std != 0, f"{stream}.{name}: std is zero"

        assert "mass" in cfg["jets"], "jets.mass is the variable this overlay adds"

    def test_jets_extra_header_does_not_claim_to_be_incomplete(self):
        header, _ = _header_and_body(NORM_DICT_JETS_EXTRA.read_text())
        for stale in ("PLACEHOLDER", "INCOMPLETE", "CANNOT be used"):
            assert stale not in header, (
                f"header still says {stale!r}, but the dict is complete "
                f"(see test_jets_extra_dict_covers_every_overlay_variable)"
            )


class TestCaloOverlay:
    """`finetune_gn3large_add_calo.yaml` — worked example 1 (add the calo stream)."""

    NEW: ClassVar[frozenset[str]] = frozenset({"norm_calo", "calo_embed"})

    def test_raw_reader_and_features(self):
        cfg = _load(OVERLAY_CALO)
        groups = cfg["data"]["modules"]["reader"]["init_args"]["groups"]
        assert groups == {"calo": {"global_object": False}}
        calo_vars = cfg["data"]["modules"]["features"]["init_args"]["variables"]["calo"]
        assert isinstance(calo_vars, list)
        assert len(calo_vars) == 47
        assert len(set(calo_vars)) == 47, "duplicate calo variable(s)"

    def test_raw_new_modules(self):
        cfg = _load(OVERLAY_CALO)
        modules = cfg["model"]["init_args"]["modules"]

        norm_calo = modules["norm_calo"]
        assert norm_calo["init_args"] == {
            "norm_dict": "norm_dicts/norm_dict_p7085_calo.yaml",
            "streams": ["calo"],
        }

        calo_embed = modules["calo_embed"]
        assert _resolve(calo_embed["class_path"]).__name__ == "StreamEmbed"
        embed_args = calo_embed["init_args"]
        assert embed_args["stream"] == "calo"
        assert embed_args["context"] == ["normed.jets"]
        assert embed_args["out_dim"] == 1024

        assert modules["concat"]["init_args"]["streams"] == ["tracks", "flows", "electrons", "calo"]

    def test_raw_onnx_export_inputs(self):
        base_inputs = yaml.safe_load(BASE.read_text())["outputs"]["onnx_export"]["init_args"][
            "inputs"
        ]
        overlay_inputs = _load(OVERLAY_CALO)["outputs"]["onnx_export"]["init_args"]["inputs"]
        assert len(overlay_inputs) == 5
        assert overlay_inputs[:4] == base_inputs
        assert overlay_inputs[4] == {
            "port": "inputs.calo",
            "name": "calo_features",
            "sequence": True,
            "dyn_axis": "n_calo",
        }

    def test_raw_static_partition_11_2_0(self):
        cfg = _load(OVERLAY_CALO)
        base_modules = set(_base_module_names())
        overlay_modules = cfg["model"]["init_args"]["modules"]
        nulled = {name for name, spec in overlay_modules.items() if spec is None}
        new = {
            name
            for name, spec in overlay_modules.items()
            if spec is not None and name not in base_modules
        }
        assert nulled == set()
        assert new == self.NEW
        retained = base_modules - nulled - PARAM_FREE
        assert retained == BACKBONE | HEADS
        assert len(retained) == 11

    def test_merged_modules_and_reader_groups(self, tmp_path):
        merged, _ = _merged(tmp_path, OVERLAY_CALO)
        base_modules = set(_base_module_names())
        non_null_modules = {
            name
            for name, spec in merged["model"]["init_args"]["modules"].items()
            if spec is not None
        }
        assert non_null_modules == base_modules | self.NEW

        groups = merged["data"]["modules"]["reader"]["init_args"]["groups"]
        assert set(groups) == {"jets", "tracks", "flows", "electrons", "calo"}

    def test_merged_schedule(self, tmp_path):
        merged, output = _merged(tmp_path, OVERLAY_CALO)
        stages = merged["training_schedule"]["stages"]
        assert list(stages) == ["calo_warmup", "full_finetune"]
        assert stages["calo_warmup"]["epochs"] == 3
        assert stages["calo_warmup"]["early_stop"]["patience"] == 1
        assert stages["full_finetune"]["early_stop"]["patience"] == 2

        non_null_modules = {
            name
            for name, spec in merged["model"]["init_args"]["modules"].items()
            if spec is not None
        }
        sched = TrainingSchedule.from_config(merged["training_schedule"], list(non_null_modules))
        assert [s.name for s in sched.stages] == ["calo_warmup", "full_finetune"]
        calo_warmup, full_finetune = sched.stages
        assert sched.frozen_names(calo_warmup) == BACKBONE | PARAM_FREE
        assert sched.frozen_names(full_finetune) == set()
        sched.validate_epochs(8)

        stem = output.with_suffix("")
        assert Path(f"{stem}_stage00_calo_warmup.dot").is_file()
        assert Path(f"{stem}_stage01_full_finetune.dot").is_file()


class TestChargeHeadOverlay:
    """`finetune_gn3large_add_charge_head.yaml` — worked example 2 (add a jet head)."""

    def test_raw_jets_bccharge_module(self):
        cfg = _load(OVERLAY_CHARGE)
        jets_bccharge = cfg["model"]["init_args"]["modules"]["jets_bccharge"]
        assert _resolve(jets_bccharge["class_path"]).__name__ == "ClassificationTaskModule"
        args = jets_bccharge["init_args"]
        assert args["input"] == "pooled.global"
        assert args["stream"] == "jets"
        assert args["label"] == "HadronGhostInitialTruthLabelPdgId"
        assert args["class_names"] == ["bquark", "antibquark", "cquark", "anticquark", "other"]
        assert args["weight_source"] is None
        assert "weight" not in args

    def test_raw_label_map_matches_gn3epclv01(self):
        cfg = _load(OVERLAY_CHARGE)
        label_map = cfg["model"]["init_args"]["modules"]["jets_bccharge"]["init_args"]["label_map"]
        assert all(isinstance(key, int) for key in label_map)
        assert set(label_map.values()) == {0, 1, 2, 3, 4}

        reference = _load_shipped("GN3EPCLV01.yaml")
        reference_map = reference["model"]["init_args"]["modules"]["jets_bccharge"]["init_args"][
            "label_map"
        ]
        assert label_map == reference_map

    def test_raw_run_tasks_restated(self):
        cfg = _load(OVERLAY_CHARGE)
        base_tasks = yaml.safe_load(BASE.read_text())["outputs"]["run_tasks"]["init_args"]["tasks"]
        tasks = cfg["outputs"]["run_tasks"]["init_args"]["tasks"]
        assert tasks == [*base_tasks, "jets_bccharge"]
        assert len(tasks) == 6

    def test_raw_static_partition_11_1_0(self):
        cfg = _load(OVERLAY_CHARGE)
        base_modules = set(_base_module_names())
        overlay_modules = cfg["model"]["init_args"]["modules"]
        nulled = {name for name, spec in overlay_modules.items() if spec is None}
        new = {
            name
            for name, spec in overlay_modules.items()
            if spec is not None and name not in base_modules
        }
        assert nulled == set()
        assert new == {"jets_bccharge"}
        retained = base_modules - nulled - PARAM_FREE
        assert retained == BACKBONE | HEADS
        assert len(retained) == 11

    def test_merged_run_tasks_and_schedule(self, tmp_path):
        merged, output = _merged(tmp_path, OVERLAY_CHARGE)
        non_null_modules = {
            name
            for name, spec in merged["model"]["init_args"]["modules"].items()
            if spec is not None
        }
        run_tasks = merged["outputs"]["run_tasks"]["init_args"]["tasks"]
        assert set(run_tasks) <= non_null_modules

        stages = merged["training_schedule"]["stages"]
        assert list(stages) == ["head_warmup", "full_finetune"]

        sched = TrainingSchedule.from_config(merged["training_schedule"], list(non_null_modules))
        assert [s.name for s in sched.stages] == ["head_warmup", "full_finetune"]
        head_warmup = sched.stages[0]
        assert sched.frozen_names(head_warmup) == BACKBONE | PARAM_FREE
        sched.validate_epochs(8)

        stem = output.with_suffix("")
        assert Path(f"{stem}_stage00_head_warmup.dot").is_file()
        assert Path(f"{stem}_stage01_full_finetune.dot").is_file()


class TestJetVarsOverlay:
    """`finetune_gn3large_add_jet_vars.yaml` — worked example 3 (add a jets variable)."""

    NULLED: ClassVar[frozenset[str]] = frozenset({
        "norm",
        "track_embed",
        "flow_embed",
        "electron_embed",
    })
    NEW: ClassVar[frozenset[str]] = frozenset({
        "norm_p7085",
        "track_embed_p7085",
        "flow_embed_p7085",
        "electron_embed_p7085",
    })

    def test_raw_variables_only_restate_jets(self):
        cfg = _load(OVERLAY_JETVARS)
        variables = cfg["data"]["modules"]["features"]["init_args"]["variables"]
        assert variables == {"jets": ["pt_btagJes", "eta_btagJes", "mass"]}

    def test_raw_static_partition_7_4_4(self):
        cfg = _load(OVERLAY_JETVARS)
        base_modules = set(_base_module_names())
        overlay_modules = cfg["model"]["init_args"]["modules"]
        nulled = {name for name, spec in overlay_modules.items() if spec is None}
        new = {
            name
            for name, spec in overlay_modules.items()
            if spec is not None and name not in base_modules
        }
        assert nulled == self.NULLED
        assert new == self.NEW
        retained = base_modules - nulled - PARAM_FREE
        assert retained == {"encoder", "pool"} | HEADS
        assert len(retained) == 7

    def test_raw_new_modules_shape(self):
        cfg = _load(OVERLAY_JETVARS)
        modules = cfg["model"]["init_args"]["modules"]

        norm_p7085 = modules["norm_p7085"]["init_args"]
        assert norm_p7085 == {
            "norm_dict": "norm_dicts/norm_dict_p7085_jets_extra.yaml",
            "streams": ["jets", "tracks", "flows", "electrons"],
            "global_object": "jets",
        }

        for embed_name, stream in (
            ("track_embed_p7085", "tracks"),
            ("flow_embed_p7085", "flows"),
            ("electron_embed_p7085", "electrons"),
        ):
            embed_args = modules[embed_name]["init_args"]
            assert embed_args["stream"] == stream
            assert embed_args["context"] == ["normed.jets"]
            assert embed_args["out_dim"] == 1024

    def test_merged_variables_and_concat(self, tmp_path):
        merged, _ = _merged(tmp_path, OVERLAY_JETVARS)
        base_modules = set(_base_module_names())
        non_null_modules = {
            name
            for name, spec in merged["model"]["init_args"]["modules"].items()
            if spec is not None
        }
        assert non_null_modules == (base_modules - self.NULLED) | self.NEW

        variables = merged["data"]["modules"]["features"]["init_args"]["variables"]
        assert len(variables["tracks"]) == 24
        assert len(variables["flows"]) == 5
        assert len(variables["electrons"]) == 28
        assert len(variables["jets"]) == 3

        assert merged["model"]["init_args"]["modules"]["concat"]["init_args"]["streams"] == [
            "tracks",
            "flows",
            "electrons",
        ]

    def test_merged_schedule(self, tmp_path):
        merged, output = _merged(tmp_path, OVERLAY_JETVARS)
        stages = merged["training_schedule"]["stages"]
        assert list(stages) == ["embed_warmup", "full_finetune"]

        non_null_modules = {
            name
            for name, spec in merged["model"]["init_args"]["modules"].items()
            if spec is not None
        }
        sched = TrainingSchedule.from_config(merged["training_schedule"], list(non_null_modules))
        assert [s.name for s in sched.stages] == ["embed_warmup", "full_finetune"]
        embed_warmup = sched.stages[0]
        assert sched.frozen_names(embed_warmup) == {"encoder", "pool"} | PARAM_FREE
        sched.validate_epochs(8)

        stem = output.with_suffix("")
        assert Path(f"{stem}_stage00_embed_warmup.dot").is_file()
        assert Path(f"{stem}_stage01_full_finetune.dot").is_file()


class TestXbbTransferOverlay:
    """`finetune_gn3large_xbb_transfer.yaml` — worked example 4 (backbone transfer)."""

    NULLED: ClassVar[set[str]] = {
        "norm",
        "track_embed",
        "flow_embed",
        "electron_embed",
        "jets_classification",
        "track_type",
        "jet_pt_regression",
    }
    NEW: ClassVar[set[str]] = {
        "norm_xbb",
        "track_embed_xbb",
        "flow_embed_xbb",
        "xbb_classification",
    }
    LOADED: ClassVar[set[str]] = {"encoder", "pool", "track_origin", "track_vertexing"}
    PARAM_FREE: ClassVar[set[str]] = {"concat", "split", "loss"}

    def test_static_partition_matches_init_from_accounting(self):
        cfg = _load(OVERLAY_XBB)
        base_modules = set(_base_module_names())
        overlay_modules = cfg["model"]["init_args"]["modules"]
        nulled = {name for name, spec in overlay_modules.items() if spec is None}
        new = {
            name
            for name, spec in overlay_modules.items()
            if spec is not None and name not in base_modules
        }
        assert nulled == self.NULLED
        assert new == self.NEW
        retained = base_modules - self.NULLED - self.PARAM_FREE
        assert retained == self.LOADED

    def test_class_paths_resolve(self):
        cfg = _load(OVERLAY_XBB)
        class_paths = list(_iter_class_paths(cfg))
        assert class_paths, "expected at least one class_path in the overlay"
        for class_path in class_paths:
            _resolve(class_path)
        norm_xbb = cfg["model"]["init_args"]["modules"]["norm_xbb"]
        assert _resolve(norm_xbb["class_path"]).__name__ == "MaskedInputNormaliser"

    def test_pad_max_is_100(self):
        cfg = _load(OVERLAY_XBB)
        groups = cfg["data"]["modules"]["reader_xbb"]["init_args"]["groups"]
        assert groups["tracks"]["pad_max"] == 100
        assert groups["flows"]["pad_max"] == 100

    def test_stacked_merge_shape(self, tmp_path):
        merged, output = _merged(tmp_path, OVERLAY_XBB)

        modules = merged["model"]["init_args"]["modules"]
        non_null_modules = {name for name, spec in modules.items() if spec is not None}
        expected_modules = {
            "norm_xbb",
            "track_embed_xbb",
            "flow_embed_xbb",
            "concat",
            "encoder",
            "split",
            "pool",
            "track_origin",
            "track_vertexing",
            "xbb_classification",
            "loss",
        }
        assert non_null_modules == expected_modules

        # Data-side rename idiom: the inherited reader/features are null-deleted and
        # the Xbb ones declared under new names (dict deep-merge cannot drop a
        # nested `groups:`/`variables:` key, so restating them would keep electrons).
        data_modules = merged["data"]["modules"]
        assert data_modules["reader"] is None
        assert data_modules["features"] is None
        groups = data_modules["reader_xbb"]["init_args"]["groups"]
        assert set(groups) == {"jets", "tracks", "flows"}
        assert groups["flows"]["dataset"] == "flow"

        variables = data_modules["features_xbb"]["init_args"]["variables"]
        assert set(variables) == {"jets", "tracks", "flows"}

        assert modules["concat"]["init_args"]["streams"] == ["tracks", "flows"]

        run_tasks = merged["outputs"]["run_tasks"]["init_args"]["tasks"]
        assert set(run_tasks) <= non_null_modules

        stages = merged["training_schedule"]["stages"]
        assert list(stages) == ["backbone_frozen", "full_finetune"]
        assert stages["full_finetune"]["early_stop"]["patience"] == 3

        sched = TrainingSchedule.from_config(merged["training_schedule"], list(non_null_modules))
        backbone_frozen = sched.stages[0]
        assert sched.frozen_names(backbone_frozen) == {
            "encoder",
            "pool",
            "track_origin",
            "track_vertexing",
            "concat",
            "split",
            "loss",
        }

        stem = output.with_suffix("")
        assert Path(f"{stem}_stage00_backbone_frozen.dot").is_file()
        assert Path(f"{stem}_stage01_full_finetune.dot").is_file()

    def test_xbb_sample_schema(self):
        # Local disk path; unrelated to the EOS rename (the EOS mirror of this
        # sample is `/eos/user/n/npond/salt-data/finetuning/xbb-finetune/`).
        default_sample_dir = "/data/ccra-data/ftag_studies/finetune_cernbox"
        sample_dir = Path(os.environ.get("SALT_XBB_SAMPLE_DIR", default_sample_dir))
        xbb = sample_dir / "pp_output_train_small.h5"
        if not xbb.is_file():
            pytest.skip(f"local Xbb sample not found at {xbb}")
        import h5py

        cfg = _load(OVERLAY_XBB)
        groups = cfg["data"]["modules"]["reader_xbb"]["init_args"]["groups"]
        variables = cfg["data"]["modules"]["features_xbb"]["init_args"]["variables"]

        with h5py.File(xbb, "r") as f:
            for stream, group_cfg in groups.items():
                if group_cfg is None:
                    continue
                dataset_name = group_cfg.get("dataset", stream)
                dtype_names = f[dataset_name].dtype.names
                for var in variables.get(stream) or []:
                    assert var in dtype_names, f"{var!r} missing from {dataset_name!r} dtype"
            jets_names = f["jets"].dtype.names
            tracks_names = f["tracks"].dtype.names
            assert "flavour_label" in jets_names
            assert "ftagTruthOriginLabel" in tracks_names
            assert "ftagTruthVertexIndex" in tracks_names

            inputs_copy_vars = cfg["outputs"]["inputs_copy"]["init_args"]["variables"]["jets"]
            for var in inputs_copy_vars:
                assert var in jets_names, f"{var!r} missing from jets dtype (inputs_copy)"


class TestP7085Schema:
    """The real p7085 sample carries every variable examples 1-3 read."""

    def test_calo_and_track_variables_present(self):
        default_sample_dir = (
            "/data/ccra-data/projects/salt-improvements/fullstats-datasets/p7085/"
            "user.alfroch.FTAG-Tau-Training-Samples-260331_p7085"
        )
        sample_dir = Path(os.environ.get("SALT_P7085_SAMPLE_DIR", default_sample_dir))
        sample = sample_dir / "pp_output_val_260331_split_003.h5"
        if not sample.is_file():
            pytest.skip(f"local p7085 sample not found at {sample}")
        import h5py

        base_vars = yaml.safe_load(BASE.read_text())["data"]["modules"]["features"]["init_args"][
            "variables"
        ]
        calo_vars = _load(OVERLAY_CALO)["data"]["modules"]["features"]["init_args"]["variables"][
            "calo"
        ]

        with h5py.File(sample, "r") as f:
            calo_names = f["calo"].dtype.names
            for var in calo_vars:
                assert var in calo_names, f"{var!r} missing from calo dtype"

            tracks_names = f["tracks_ghost"].dtype.names
            for var in base_vars["tracks"]:
                assert var in tracks_names, f"{var!r} missing from tracks_ghost dtype"
            for label in (
                "ftagTruthOriginLabel",
                "ftagTruthVertexIndex",
                "ftagTruthTypeLabel",
            ):
                assert label in tracks_names, f"{label!r} missing from tracks_ghost dtype"

            flows_names = f["flows"].dtype.names
            for var in base_vars["flows"]:
                assert var in flows_names, f"{var!r} missing from flows dtype"

            electrons_names = f["electrons"].dtype.names
            for var in base_vars["electrons"]:
                assert var in electrons_names, f"{var!r} missing from electrons dtype"

            jets_names = f["jets"].dtype.names
            for var in base_vars["jets"]:
                assert var in jets_names, f"{var!r} missing from jets dtype"
            for label in (
                "mass",
                "HadronGhostInitialTruthLabelPdgId",
                "flavour_label",
            ):
                assert label in jets_names, f"{label!r} missing from jets dtype"


class TestFreezeGraphs:
    """DOT reproducibility: the committed `docs/assets/finetuning/` graphs."""

    def test_dot_files_reproduce_committed_ones(self, tmp_path):
        if not ASSETS_DIR.is_dir():
            pytest.skip(
                f"docs/assets/finetuning not found at {ASSETS_DIR} — produced by "
                "docs/tutorials/finetuning_render_graphs.sh. This is EXPECTED at "
                "the code-phase commit: the parent experiment's `gate` stage "
                "renders these assets, and the orchestrator commits them."
            )

        dir_base = tmp_path / "base"
        dir_calo = tmp_path / "calo"
        dir_charge = tmp_path / "charge"
        dir_jetvar = tmp_path / "jetvar"
        dir_xbb = tmp_path / "xbb"
        for d in (dir_base, dir_calo, dir_charge, dir_jetvar, dir_xbb):
            d.mkdir()

        _, output_base = _merged(dir_base)
        _, output_calo = _merged(dir_calo, OVERLAY_CALO)
        _, output_charge = _merged(dir_charge, OVERLAY_CHARGE)
        _, output_jetvar = _merged(dir_jetvar, OVERLAY_JETVARS)
        _, output_xbb = _merged(dir_xbb, OVERLAY_XBB)

        stem_base = output_base.with_suffix("")
        stem_calo = output_calo.with_suffix("")
        stem_charge = output_charge.with_suffix("")
        stem_jetvar = output_jetvar.with_suffix("")
        stem_xbb = output_xbb.with_suffix("")

        pairs = [
            (Path(f"{stem_base}_stage00_fit.dot"), ASSETS_DIR / "gn3large_stage00_fit.dot"),
            (
                Path(f"{stem_calo}_stage00_calo_warmup.dot"),
                ASSETS_DIR / "merged_calo_stage00_calo_warmup.dot",
            ),
            (
                Path(f"{stem_calo}_stage01_full_finetune.dot"),
                ASSETS_DIR / "merged_calo_stage01_full_finetune.dot",
            ),
            (
                Path(f"{stem_charge}_stage00_head_warmup.dot"),
                ASSETS_DIR / "merged_charge_stage00_head_warmup.dot",
            ),
            (
                Path(f"{stem_charge}_stage01_full_finetune.dot"),
                ASSETS_DIR / "merged_charge_stage01_full_finetune.dot",
            ),
            (
                Path(f"{stem_jetvar}_stage00_embed_warmup.dot"),
                ASSETS_DIR / "merged_jetvar_stage00_embed_warmup.dot",
            ),
            (
                Path(f"{stem_jetvar}_stage01_full_finetune.dot"),
                ASSETS_DIR / "merged_jetvar_stage01_full_finetune.dot",
            ),
            (
                Path(f"{stem_xbb}_stage00_backbone_frozen.dot"),
                ASSETS_DIR / "merged_xbb_stage00_backbone_frozen.dot",
            ),
            (
                Path(f"{stem_xbb}_stage01_full_finetune.dot"),
                ASSETS_DIR / "merged_xbb_stage01_full_finetune.dot",
            ),
        ]
        assert len(pairs) == 9
        for regenerated, committed in pairs:
            assert regenerated.is_file(), f"expected regenerated {regenerated}"
            assert committed.is_file(), f"missing committed {committed}"
            assert regenerated.read_text() == committed.read_text(), (
                f"{committed.name} is not reproducible from BASE + overlay at this commit"
            )
            png = committed.with_suffix(".png")
            assert png.is_file(), f"missing committed png sibling {png}"
