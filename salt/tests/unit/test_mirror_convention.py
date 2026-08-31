"""Enforce the unit-test mirror convention: every source file ``salt/<pkg>/<mod>.py``
has a dedicated unit test at ``salt/tests/unit/<pkg>/test_<mod>.py``, or an explicit
entry below saying why not. Extra test files (multi-file families, cross-cutting
suites) are always allowed — only sources are checked.
"""

from __future__ import annotations

from collections.abc import Container, Iterable, Mapping
from pathlib import Path, PurePosixPath

_REPO = Path(__file__).resolve().parents[3]

# Sources that need no dedicated test file anywhere. `__init__.py` is exempt by rule.
EXEMPT: dict[str, str] = {
    "salt/graph/errors.py": "exception hierarchy with no logic; raised and asserted suite-wide",
}

# Whole subtrees exempt with a single justification.
EXEMPT_TREES: dict[str, str] = {
    "salt/testing": "test-support code (canned inputs + datagen); exercised by its consumers",
}

# Sources whose dedicated coverage lives at a non-mirror position. Every listed
# test file must exist; if a real mirror test appears the entry must be removed.
COVERED_ELSEWHERE: dict[str, tuple[str, ...]] = {
    # CLI parser + config stacking are driven end-to-end by the main/merge suites
    "salt/parser.py": (
        "salt/tests/unit/test_main.py",
        "salt/tests/unit/test_merge_config.py",
    ),
    "salt/config_utils.py": ("salt/tests/unit/test_include.py",),
    # needs a Trainer, so its contract tests are integration-level
    "salt/callbacks/schedule.py": (
        "salt/tests/integration/multistage_training/test_training_schedule.py",
        "salt/tests/integration/multistage_training/test_schedule_resume.py",
    ),
    # reader/dataset ABCs asserted throughout unit/data
    "salt/data/base.py": (
        "salt/tests/unit/data/readers/test_multisample_reader.py",
        "salt/tests/unit/data/test_iterable_dataset.py",
    ),
    "salt/data/datamodule.py": (
        "salt/tests/unit/data/test_manifest.py",
        "salt/tests/unit/data/test_iterable_dataset.py",
        "salt/tests/unit/graph/test_setup_executor.py",
    ),
    "salt/data/dataset.py": (
        "salt/tests/unit/data/test_row_cuts_h5.py",
        "salt/tests/unit/data/test_manifest.py",
    ),
    "salt/data/dtypes.py": ("salt/tests/unit/data/test_row_cuts_h5.py",),
    # _PlanRunner is the shared base of both dataset classes
    "salt/data/plan_runner.py": (
        "salt/tests/unit/data/test_iterable_dataset.py",
        "salt/tests/unit/data/test_manifest.py",
    ),
    # map-style batch sampler engaged by every fit/test lifecycle run
    "salt/data/samplers.py": ("salt/tests/unit/graph/test_graph_cli.py",),
    "salt/data/sharding.py": ("salt/tests/unit/data/test_iterable_dataset.py",),
    "salt/data/processors/features.py": (
        "salt/tests/unit/data/test_iterable_dataset.py",
        "salt/tests/unit/data/test_label_stripped_read.py",
    ),
    "salt/data/processors/labels.py": ("salt/tests/unit/data/test_label_stripped_read.py",),
    # config-driven processors exercised through the shipped-config lifecycles
    "salt/data/processors/ftag_labeller.py": (
        "salt/tests/integration/pipeline/test_pipeline.py",
    ),
    "salt/data/processors/maskformer_targets.py": (
        "salt/tests/unit/model/modules/test_maskdecoder.py",
        "salt/tests/integration/pipeline/test_pipeline.py",
    ),
    "salt/data/processors/multi_target.py": ("salt/tests/integration/pipeline/test_pipeline.py",),
    "salt/data/readers/reader.py": (
        "salt/tests/unit/data/readers/test_uproot_reader.py",
        "salt/tests/unit/test_profiling.py",
    ),
    "salt/model/modules/edge_embed.py": ("salt/tests/unit/model/test_compile_regression.py",),
    "salt/model/modules/maskformer_matched_loss.py": (
        "salt/tests/unit/model/test_maskformer_test_bind.py",
    ),
    # raw nn blocks exercised through the modules that assemble them
    "salt/model/nn/attention.py": ("salt/tests/unit/model/modules/test_transformer_encoder.py",),
    "salt/model/nn/dense.py": ("salt/tests/unit/model/modules/test_stream_embed.py",),
    "salt/model/nn/layernorm.py": ("salt/tests/unit/model/modules/test_transformer_encoder.py",),
    "salt/model/nn/maskformer_loss.py": (
        "salt/tests/unit/model/modules/test_maskdecoder.py",
        "salt/tests/unit/model/test_maskformer_test_bind.py",
    ),
    "salt/model/nn/matcher.py": ("salt/tests/unit/model/modules/test_maskdecoder.py",),
    "salt/model/nn/posenc.py": ("salt/tests/unit/model/nn/test_featurewise.py",),
    "salt/model/nn/transformer.py": (
        "salt/tests/unit/model/modules/test_transformer_encoder.py",
    ),
    "salt/outputs/input_copy_writer.py": ("salt/tests/unit/outputs/test_dumb_sink_guards.py",),
    "salt/outputs/maskformer.py": (
        "salt/tests/unit/outputs/test_maskformer_eval_h5.py",
        "salt/tests/unit/outputs/test_producer_manifests.py",
    ),
    "salt/outputs/output_schema.py": (
        "salt/tests/unit/outputs/test_object_groups.py",
        "salt/tests/unit/outputs/test_section_sinks.py",
    ),
    "salt/outputs/pad_mask_writer.py": ("salt/tests/unit/outputs/test_section_demand.py",),
    "salt/outputs/run_task_output.py": (
        "salt/tests/unit/outputs/test_section_demand.py",
        "salt/tests/unit/outputs/test_dumb_sink_guards.py",
    ),
    "salt/outputs/sinks/registry.py": (
        "salt/tests/unit/callbacks/test_sink_adapter.py",
        "salt/tests/unit/outputs/test_runtime_sink_selection.py",
    ),
    "salt/outputs/sinks/sink.py": (
        "salt/tests/unit/outputs/test_sink_consumes.py",
        "salt/tests/unit/outputs/test_sink_node.py",
    ),
    "salt/outputs/sinks/onnx/check.py": ("salt/tests/integration/pipeline/test_pipeline.py",),
    "salt/outputs/sinks/onnx/config.py": (
        "salt/tests/unit/outputs/test_export_fold.py",
        "salt/tests/unit/outputs/sinks/onnx/test_adapter.py",
    ),
    "salt/outputs/sinks/onnx/metadata.py": ("salt/tests/unit/outputs/sinks/onnx/test_export.py",),
    "salt/outputs/sinks/onnx/reduces.py": (
        "salt/tests/unit/outputs/test_hard_reduce_nodes.py",
        "salt/tests/unit/outputs/sinks/onnx/test_adapter.py",
    ),
    "salt/utils/array_utils.py": ("salt/tests/integration/pipeline/test_pipeline.py",),
    "salt/utils/file_utils.py": ("salt/tests/unit/data/readers/test_vds.py",),
    "salt/utils/mask_utils.py": ("salt/tests/unit/outputs/test_maskformer_eval_h5.py",),
    "salt/utils/scalers.py": ("salt/tests/unit/model/modules/tasks/test_regression.py",),
    "salt/utils/union_find.py": ("salt/tests/unit/model/test_get_output.py",),
}


def iter_sources(repo: Path) -> list[str]:
    """All product .py files as repo-relative POSIX paths (tests excluded)."""
    tests = repo / "salt" / "tests"
    return sorted(
        p.relative_to(repo).as_posix()
        for p in (repo / "salt").rglob("*.py")
        if not p.is_relative_to(tests)
    )


def mirror_path(source: str) -> str:
    """``salt/a/b.py`` -> ``salt/tests/unit/a/test_b.py``."""
    rel = PurePosixPath(source).relative_to("salt")
    return (PurePosixPath("salt/tests/unit") / rel.parent / f"test_{rel.name}").as_posix()


def check_mirrors(
    sources: Iterable[str],
    existing: Container[str],
    exempt: Mapping[str, str],
    exempt_trees: Mapping[str, str],
    covered: Mapping[str, tuple[str, ...]],
) -> dict[str, list[str]]:
    """Pure classification of every source against the convention; returns problems."""
    problems: dict[str, list[str]] = {
        "missing mirror test (add one, or an entry in this file)": [],
        "allowlisted but a mirror test now exists (delete the entry)": [],
        "listed in both EXEMPT and COVERED_ELSEWHERE": [],
        "COVERED_ELSEWHERE names a test file that does not exist": [],
        "stale allowlist entry (source file is gone)": [],
    }
    source_set = set(sources)
    for src in sorted(source_set):
        if PurePosixPath(src).name == "__init__.py":
            continue
        mirror = mirror_path(src)
        allowlisted = src in exempt or src in covered
        if mirror in existing:
            if allowlisted:
                problems["allowlisted but a mirror test now exists (delete the entry)"].append(src)
            continue
        if src in exempt and src in covered:
            problems["listed in both EXEMPT and COVERED_ELSEWHERE"].append(src)
            continue
        if src in exempt or any(src.startswith(f"{tree}/") for tree in exempt_trees):
            continue
        if src in covered:
            problems["COVERED_ELSEWHERE names a test file that does not exist"].extend(
                f"{src} -> {t}" for t in covered[src] if t not in existing
            )
            continue
        problems["missing mirror test (add one, or an entry in this file)"].append(
            f"{src} (expected {mirror})"
        )
    for entry in [*exempt, *covered]:
        if entry not in source_set:
            problems["stale allowlist entry (source file is gone)"].append(entry)
    return {label: hits for label, hits in problems.items() if hits}


def _existing_test_files(repo: Path) -> frozenset[str]:
    return frozenset(
        p.relative_to(repo).as_posix() for p in (repo / "salt" / "tests").rglob("*.py")
    )


def test_every_source_has_a_mirror_unit_test():
    """One unit-test file per source .py, with every exception justified above."""
    problems = check_mirrors(
        iter_sources(_REPO), _existing_test_files(_REPO), EXEMPT, EXEMPT_TREES, COVERED_ELSEWHERE
    )
    report = "\n".join(
        f"{label}:\n" + "\n".join(f"  {hit}" for hit in hits) for label, hits in problems.items()
    )
    assert not problems, f"unit-test mirror convention violated:\n{report}"


def _main() -> int:
    """Stdlib-only CI entry point: run the same check without pytest, print, exit non-zero."""
    problems = check_mirrors(
        iter_sources(_REPO), _existing_test_files(_REPO), EXEMPT, EXEMPT_TREES, COVERED_ELSEWHERE
    )
    if not problems:
        print("mirror convention OK")
        return 0
    for label, hits in problems.items():
        print(f"{label}:")
        for hit in hits:
            print(f"  {hit}")
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(_main())
