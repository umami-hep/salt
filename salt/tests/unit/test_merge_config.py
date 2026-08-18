"""Tests for ``salt merge-config`` (`salt.merge_config`) and its render support:
fit-parity merged dump, per-stage freeze-graph naming/annotation, legacy
desugar, ``--merged.plots`` gating, and the `dot_source` regression gate.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest
import yaml

from salt.config_utils import disable_logger_in_config
from salt.graph.planner import compile_plan
from salt.graph.render import dot_source
from salt.graph.spec import Mode, TensorSpec, unflatten_spec
from salt.main import SaltCLI
from salt.main import main as salt_main
from salt.merge_config import main as merge_config_main
from salt.schema import dump_schema, save_schema
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict
from salt.tests._fixtures.toys import ToyEmbed, ToyHead, ToySource, ToyWildcardLabels
from salt.testing.inputs import write_dummy_file
from salt.tests._fixtures.gn2v2_test_config import small_config

DUMMY_CFG = small_config()

# a two-stage schedule twin of the fine-tuning example: head_warmup freezes
# everything but jets_classification (via trainable:), full_finetune frees all.
TWO_STAGE_SCHEDULE_YAML = """
training_schedule:
  stages:
    head_warmup:
      epochs: 5
      trainable: [jets_classification]
      lrs:
        max: 1.0e-4
    full_finetune:
      frozen: []
      lrs:
        initial: 1.0e-7
        max: 1.0e-5
"""


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("merge_config")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    h5_path = base / "pp_output_train.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"h5": h5_path, "nd": nd_path, "schema": schema_path}


def fit_args(data: dict[str, Path], *extra: str) -> list[str]:
    """The gn2v2-opendata config (logger disabled) + its documented overrides."""
    cfg = disable_logger_in_config(str(DUMMY_CFG))
    return [
        "--config",
        cfg,
        f"--data.train_file={data['h5']}",
        f"--data.val_file={data['h5']}",
        f"--data.modules.reader.init_args.schema={data['schema']}",
        f"--model.modules.norm.init_args.norm_dict={data['nd']}",
        *extra,
    ]


def write_yaml(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def child_key_order(text: str, key: str) -> list[str]:
    """Direct child keys of the ``key:`` mapping in `text`, in dump order."""
    lines = text.split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.strip() == f"{key}:")
    base = len(lines[start]) - len(lines[start].lstrip())
    child_indent: int | None = None
    order: list[str] = []
    for ln in lines[start + 1 :]:
        if not ln.strip():
            continue
        ind = len(ln) - len(ln.lstrip())
        if ind <= base:
            break
        if child_indent is None:
            child_indent = ind
        if ind == child_indent and not ln.lstrip().startswith("- "):
            order.append(ln.strip().split(":")[0])
    return order


def print_config_dump(args: list[str]) -> str:
    """The run-free ``--print_config`` dump for `args` (the fit config surface)."""
    buffer = io.StringIO()
    with pytest.raises(SystemExit) as excinfo, contextlib.redirect_stdout(buffer):
        SaltCLI(args=[*args, "--print_config"], run=False)
    assert excinfo.value.code == 0
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# render support: frozen styling + title (and the byte-identity regression gate)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_plan():
    modules = {
        "source": ToySource(),
        "embed": ToyEmbed(),
        "labels": ToyWildcardLabels(),
        "head": ToyHead(),
    }
    for name, module in modules.items():
        module.name = name
    return compile_plan(
        modules,
        Mode.FIT,
        sources=unflatten_spec({"raw.x": TensorSpec(shape=("B", 8), dtype="float32")}),
        schema=("labels.x",),
        sinks=["losses.total"],
    )


class TestDotSourceAnnotation:
    def test_default_output_byte_identical(self, toy_plan):
        # frozen=None/title=None must be byte-identical to an empty frozen set /
        # no title — the regression gate protecting existing `salt graph plot`.
        base = dot_source(toy_plan)
        assert base == dot_source(toy_plan, frozen=None, title=None)
        assert base == dot_source(toy_plan, frozen=frozenset())
        # no annotation artefacts leak into the unannotated render
        assert "#bdbdbd" not in base
        assert "labelloc" not in base
        assert "frozen" not in base

    def test_frozen_module_styled_and_badged(self, toy_plan):
        dot = dot_source(toy_plan, frozen=frozenset({"embed"}))
        assert "#bdbdbd" in dot  # frozen fill
        assert ">frozen<" in dot  # badge
        # a non-frozen module keeps its namespace fill (not the frozen grey)
        head_card = dot.split('"head" [label=')[1].split("];")[0]
        assert "#bdbdbd" not in head_card

    def test_title_becomes_graph_label(self, toy_plan):
        dot = dot_source(toy_plan, title="stage 1/2: head_warmup — frozen: embed")
        assert "labelloc=" in dot
        assert 'label="stage 1/2: head_warmup' in dot


# ---------------------------------------------------------------------------
# merge-config: fit-parity dump
# ---------------------------------------------------------------------------


class TestMergedDumpParity:
    def test_merged_yaml_matches_print_config(self, data, tmp_path):
        # gn2v2-opendata declares no training_schedule, so F1 injects the desugared
        # single-fit stage — strip it before the fit-parity comparison.
        args = fit_args(data)
        out = tmp_path / "merged.yaml"
        rc = merge_config_main([*args, "--merged.output", str(out), "--merged.plots", "false"])
        assert rc == 0
        merged_obj = yaml.safe_load(out.read_text())
        injected = merged_obj.pop("training_schedule", None)
        assert injected == {"stages": {"fit": {}}}  # the materialized desugar block
        printed_obj = yaml.safe_load(print_config_dump(args))
        printed_obj.pop("training_schedule", None)  # print_config emits none for a legacy config
        assert merged_obj == printed_obj

    def test_scheduled_config_dump_unchanged(self, data, tmp_path):
        # a config that DOES declare a schedule is dumped byte-identically to
        # print_config — no materialization, no marker (F1 only touches legacy).
        sched = write_yaml(tmp_path, "sched.yaml", TWO_STAGE_SCHEDULE_YAML)
        args = fit_args(data, "--config", sched)
        out = tmp_path / "merged.yaml"
        merge_config_main([*args, "--merged.output", str(out), "--merged.plots", "false"])
        text = out.read_text()
        assert "materialized by merge-config" not in text
        assert yaml.safe_load(text) == yaml.safe_load(print_config_dump(args))

    def test_legacy_materializes_desugared_schedule(self, data, tmp_path):
        # F1: a legacy config's merged YAML gains an explicit training_schedule
        # matching TrainingSchedule.desugar_legacy semantics exactly.
        from salt.schedule import TrainingSchedule

        out = tmp_path / "merged.yaml"
        merge_config_main([*fit_args(data), "--merged.output", str(out), "--merged.plots", "false"])
        text = out.read_text()
        assert "materialized by merge-config" in text  # marker comment present
        merged = yaml.safe_load(text)
        assert merged["training_schedule"] == {"stages": {"fit": {}}}
        module_names = list(merged["model"]["init_args"]["modules"])
        got = TrainingSchedule.from_config(merged["training_schedule"], module_names)
        legacy = TrainingSchedule.desugar_legacy(module_names)
        assert [s.name for s in got.stages] == [s.name for s in legacy.stages] == ["fit"]
        gs, ls = got.stages[0], legacy.stages[0]
        assert (gs.frozen, gs.trainable, gs.optimizer, gs.lrs, gs.epochs, gs.order) == (
            ls.frozen,
            ls.trainable,
            ls.optimizer,
            ls.lrs,
            ls.epochs,
            ls.order,
        )
        assert got.frozen_names(gs) == set()  # everything trainable

    def test_merge_only_options_absent_from_dump(self, data, tmp_path):
        out = tmp_path / "merged.yaml"
        merge_config_main([*fit_args(data), "--merged.output", str(out), "--merged.plots=false"])
        merged_obj = yaml.safe_load(out.read_text())
        assert "merged" not in merged_obj

    def test_missing_output_is_config_error(self, data):
        from salt.graph.errors import ConfigError

        with pytest.raises(ConfigError, match="merged.output"):
            merge_config_main(fit_args(data))


# ---------------------------------------------------------------------------
# merge-config: per-stage freeze graphs
# ---------------------------------------------------------------------------


class TestStagePlots:
    def test_two_stage_files_named_and_annotated(self, data, tmp_path):
        sched = write_yaml(tmp_path, "sched.yaml", TWO_STAGE_SCHEDULE_YAML)
        out = tmp_path / "merged.yaml"
        merge_config_main([
            *fit_args(data, "--config", sched),
            "--merged.output",
            str(out),
            "--merged.plots",
            "false",
        ])
        stage0 = tmp_path / "merged_stage00_head_warmup.dot"
        stage1 = tmp_path / "merged_stage01_full_finetune.dot"
        assert stage0.is_file()
        assert stage1.is_file()

        dot0 = stage0.read_text()
        # head_warmup: trainable=[jets_classification] => everything else frozen
        assert "#bdbdbd" in dot0
        assert ">frozen<" in dot0
        label0 = next(ln for ln in dot0.splitlines() if ln.strip().startswith("label="))
        assert "stage 1/2: head_warmup" in label0
        assert "encoder" in label0  # a frozen module named in the caption
        assert "jets_classification" not in label0  # the one trainable module

        dot1 = stage1.read_text()
        # full_finetune: frozen=[] => nothing frozen, no frozen styling
        assert "#bdbdbd" not in dot1
        label1 = next(ln for ln in dot1.splitlines() if ln.strip().startswith("label="))
        assert "stage 2/2: full_finetune" in label1
        assert "(none)" in label1

    def test_legacy_config_single_fit_stage(self, data, tmp_path):
        # no training_schedule => one desugared `fit` stage, nothing frozen.
        out = tmp_path / "merged.yaml"
        merge_config_main([*fit_args(data), "--merged.output", str(out), "--merged.plots", "false"])
        only = list(tmp_path.glob("merged_stage*.dot"))
        assert [p.name for p in only] == ["merged_stage00_fit.dot"]
        dot = only[0].read_text()
        assert "#bdbdbd" not in dot
        assert "stage 1/1: fit" in dot

    def test_plots_false_writes_dot_without_images(self, data, tmp_path):
        out = tmp_path / "merged.yaml"
        merge_config_main([*fit_args(data), "--merged.output", str(out), "--merged.plots", "false"])
        assert (tmp_path / "merged_stage00_fit.dot").is_file()
        # no rasterisation requested => no PNG/PDF siblings
        assert not list(tmp_path.glob("*.png"))
        assert not list(tmp_path.glob("*.pdf"))


# ---------------------------------------------------------------------------
# dispatch + help
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_help_exits_zero(self, capsys):
        assert merge_config_main(["--help"]) == 0
        assert "merge-config" in capsys.readouterr().out

    def test_help_via_salt_main(self, capsys):
        assert salt_main(["merge-config", "--help"]) == 0
        assert "merge-config" in capsys.readouterr().out

    def test_salt_main_dispatch_writes_merged(self, data, tmp_path):
        out = tmp_path / "merged.yaml"
        rc = salt_main([
            "merge-config",
            *fit_args(data),
            "--merged.output",
            str(out),
            "--merged.plots",
            "false",
        ])
        assert rc == 0
        assert out.is_file()


# ---------------------------------------------------------------------------
# class_path-before-init_args serialization order (DeepMergeParser.dump)
# ---------------------------------------------------------------------------

# an overlay that touches only a module's init_args — the exact pattern that made
# jsonargparse emit that module's `init_args` before its `class_path` (F2 repro).
OVERRIDE_INIT_ARGS_YAML = """
model:
  init_args:
    modules:
      track_embed:
        init_args:
          out_dim: 16
"""

_MISORDERED_YAML = """model:
  init_args:
    modules:
      norm:
        init_args:
          norm_dict: /x/nd.yaml
          streams:
          - jets
          - tracks
        class_path: salt.model.modules.Normaliser
      track_embed:
        class_path: salt.model.modules.StreamEmbed
        init_args:
          out_dim: 16
"""


class TestClassPathOrdering:
    def test_transform_reorders_and_preserves_semantics(self):
        from salt.parser import _class_path_before_init_args

        out = _class_path_before_init_args(_MISORDERED_YAML)
        # the misordered `norm` now reads class_path-first...
        assert child_key_order(out, "norm")[:2] == ["class_path", "init_args"]
        # ...the already-correct `track_embed` is untouched...
        assert child_key_order(out, "track_embed")[:2] == ["class_path", "init_args"]
        # ...and it is a pure serialization change (reparsed object identical).
        assert yaml.safe_load(out) == yaml.safe_load(_MISORDERED_YAML)

    def test_transform_idempotent_and_noop_when_ordered(self):
        from salt.parser import _class_path_before_init_args

        once = _class_path_before_init_args(_MISORDERED_YAML)
        assert _class_path_before_init_args(once) == once  # idempotent
        assert _class_path_before_init_args(once) == once  # already-ordered no-op

    def test_overlay_override_dumps_class_path_first(self, data, tmp_path):
        # regression on the exact repro: an overlay that overrides only a module's
        # init_args must still dump class_path before init_args for that module.
        overlay = write_yaml(tmp_path, "over.yaml", OVERRIDE_INIT_ARGS_YAML)
        out = tmp_path / "merged.yaml"
        merge_config_main([
            *fit_args(data, "--config", overlay),
            "--merged.output",
            str(out),
            "--merged.plots",
            "false",
        ])
        text = out.read_text()
        assert child_key_order(text, "track_embed")[:2] == ["class_path", "init_args"]
        # every class_path/init_args module reads class_path-first
        for module in ("norm", "track_embed", "encoder", "jets_classification"):
            order = child_key_order(text, module)
            assert order.index("class_path") < order.index("init_args")


class TestStageCaption:
    """The per-stage graph caption reflects early_stop + callbacks."""

    def test_plain_stage_caption(self):
        from salt.merge_config import _stage_title
        from salt.schedule import StageConfig

        title = _stage_title(0, 2, "warmup", frozenset({"encoder"}), StageConfig(name="warmup"))
        assert title == "stage 1/2: warmup — frozen: encoder"

    def test_caption_includes_early_stop(self):
        from salt.merge_config import _stage_title
        from salt.schedule import EarlyStopConfig, StageConfig

        stage = StageConfig(
            name="full",
            early_stop=EarlyStopConfig(monitor="val/loss", mode="min", patience=5),
        )
        title = _stage_title(1, 2, "full", frozenset(), stage)
        assert "early_stop: val/loss (min, patience 5)" in title
        assert "frozen: (none)" in title

    def test_caption_includes_stage_callback_count(self):
        from salt.merge_config import _stage_title
        from salt.schedule import StageConfig

        stage = StageConfig(name="full", callbacks=({"class_path": "pkg.A"},))
        title = _stage_title(1, 2, "full", frozenset(), stage)
        assert "+1 stage callback(s)" in title

    def test_caption_includes_lr_scheduler_class(self):
        from salt.merge_config import _stage_title
        from salt.schedule import LRSchedulerConfig, StageConfig

        stage = StageConfig(
            name="full",
            lr_scheduler=LRSchedulerConfig(class_path="torch.optim.lr_scheduler.ReduceLROnPlateau"),
        )
        title = _stage_title(1, 2, "full", frozenset(), stage)
        assert "lr_scheduler: ReduceLROnPlateau" in title
