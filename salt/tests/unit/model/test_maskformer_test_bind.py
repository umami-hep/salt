"""MaskFormer TEST-mode bind: the decorator chain now genuinely consumes
`preds.objects.regression` in TEST (MaskFormerObjects -> MFLeadVertexDecorator),
so the TEST plan carries the width and binds it — unlike the old 5-slot hadron
example, where TEST had no consumer for it at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from salt.cli import load_config
from salt.graph.errors import ConfigError
from salt.graph.planner import compile_plan
from salt.graph.spec import Mode
from salt.model.modules import MaskFormerMatchedLoss
from salt.model.bind import BindError, ResolvedSchema, bind_all, resolve_bind_schema

# this file is at salt/tests/unit/model/ — the configs live at salt/configs/
_MASKFORMER = str(Path(__file__).parents[3] / "configs" / "MaskFormer.yaml")
_OVERRIDES = ["model.modules.norm.init_args.norm_dict=unused.yaml"]

_REG_PRED_KEY = "preds.objects.regression"
_REG_TGT_KEY = "targets.objects.regression"


def _compile(mode: Mode):
    """Load MaskFormer.yaml and compile one mode's plan (static, no data)."""
    cfg = load_config(_MASKFORMER, _OVERRIDES)
    plan = compile_plan(
        cfg.modules,
        mode,
        cfg.sources,
        schema=cfg.schema,
        sinks=cfg.sinks,
        sink_origins=cfg.sink_origins.get(mode),
    )
    return cfg, plan


class TestMaskFormerTestModeBind:
    """TEST-mode bind resolves the regression width (the decorator chain consumes it);
    FIT+VAL bind the same width via the matched-loss/matcher path.
    """

    def test_test_mode_bind_resolves_regression_width(self):
        """``SaltModule.setup('test')`` shape: TEST plan alone -> bind_all must not raise.

        Unlike the old 5-slot hadron example (no TEST consumer), the vertexing
        config's TEST plan pulls `preds.objects.regression` in through
        MaskFormerObjects -> MFLeadVertexDecorator, so the width IS present.
        """
        cfg, test_plan = _compile(Mode.TEST)
        schema = resolve_bind_schema([test_plan])
        assert schema.widths.get(_REG_PRED_KEY) == 5
        bind_all(cfg.model_modules, schema)

    def test_fit_mode_binds_regression_width(self):
        """``SaltModule.setup('fit')`` shape: FIT+VAL -> the width resolves and binds."""
        cfg_fit, fit_plan = _compile(Mode.FIT)
        _, val_plan = _compile(Mode.VAL)
        schema = resolve_bind_schema([fit_plan, val_plan])
        assert schema.widths.get(_REG_PRED_KEY) == 5
        assert schema.widths.get(_REG_TGT_KEY) == 5
        bind_all(cfg_fit.model_modules, schema)

    def test_test_plan_contains_decorator_chain(self):
        """TEST wires the full object -> decorator chain; FIT/VAL wire neither node."""
        _, test_plan = _compile(Mode.TEST)
        assert {"maskformer_objects", "mf_lead_vertex", "regression"} <= set(
            test_plan.module_names
        )
        _, fit_plan = _compile(Mode.FIT)
        _, val_plan = _compile(Mode.VAL)
        for plan in (fit_plan, val_plan):
            names = set(plan.module_names)
            assert "maskformer_objects" not in names
            assert "mf_lead_vertex" not in names


class TestMaskFormerMatchedLossBindGuard:
    """Unit-level checks of the width-agreement guard in isolation."""

    @staticmethod
    def _loss() -> MaskFormerMatchedLoss:
        loss = MaskFormerMatchedLoss(
            num_classes=2,
            num_queries=5,
            loss_weights={"object_class_ce": 2.0, "mask_dice": 2.0, "regression": 2.0},
        )
        loss.name = "mf_matched_loss"
        return loss

    def test_bind_skips_when_keys_absent(self):
        """A TEST/ONNX-only schema (regression keys pruned) -> bind is a no-op."""
        loss = self._loss()
        assert "regression" in loss.components
        loss.bind(ResolvedSchema(widths={"preds.jets.jets_classification": 3}))

    def test_bind_passes_on_matching_widths(self):
        """Present + equal pred/target widths -> bind validates and passes."""
        loss = self._loss()
        loss.bind(ResolvedSchema(widths={_REG_PRED_KEY: 5, _REG_TGT_KEY: 5}))

    def test_bind_still_rejects_mismatched_widths(self):
        """The guard must NOT disable the real check: present but unequal -> ConfigError."""
        loss = self._loss()
        with pytest.raises(ConfigError, match="regression prediction width"):
            loss.bind(ResolvedSchema(widths={_REG_PRED_KEY: 5, _REG_TGT_KEY: 4}))

    def test_missing_width_lookup_still_raises_binderror(self):
        """Sanity: ``schema.width`` on an absent key still raises ``BindError``."""
        schema = ResolvedSchema(widths={"preds.jets.jets_classification": 3})
        with pytest.raises(BindError, match="no statically resolved width"):
            schema.width(_REG_PRED_KEY)
