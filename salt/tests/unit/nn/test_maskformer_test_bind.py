"""Regression guard: the MaskFormer TEST-mode two-phase bind must not require the"""

from __future__ import annotations

from pathlib import Path

import pytest

from salt.cli import load_config
from salt.graph.errors import ConfigError
from salt.graph.planner import compile_plan
from salt.graph.spec import Mode
from salt.model.modules import MaskFormerMatchedLoss
from salt.model.bind import BindError, ResolvedSchema, bind_all, resolve_bind_schema

# this file is at salt/tests/unit/nn/ — the configs live at salt/configs/
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
    """The exp-21 TEST-mode bind defect (job 2642) and its FIT-mode counterpart."""

    def test_test_mode_bind_succeeds_without_regression_width(self):
        """``SaltModule.setup('test')`` shape: TEST plan alone -> bind_all must not raise."""
        cfg, test_plan = _compile(Mode.TEST)
        schema = resolve_bind_schema([test_plan])
        # the key is genuinely absent in TEST (the regression head opts out of TEST)
        assert _REG_PRED_KEY not in schema.widths
        # the fix: binding the model modules against the TEST-only schema must succeed
        bind_all(cfg.model_modules, schema)

    def test_fit_mode_binds_regression_width(self):
        """``SaltModule.setup('fit')`` shape: FIT+VAL -> the width resolves and binds."""
        cfg_fit, fit_plan = _compile(Mode.FIT)
        _, val_plan = _compile(Mode.VAL)
        schema = resolve_bind_schema([fit_plan, val_plan])
        assert schema.widths.get(_REG_PRED_KEY) == 5
        assert schema.widths.get(_REG_TGT_KEY) == 5
        bind_all(cfg_fit.model_modules, schema)


class TestMaskFormerMatchedLossBindGuard:
    """Unit-level checks of the width-agreement guard in isolation."""

    @staticmethod
    def _loss() -> MaskFormerMatchedLoss:
        loss = MaskFormerMatchedLoss(
            num_classes=2,
            num_objects=5,
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
        """Sanity: ``schema.width`` on an absent key raises the exp-21 BindError shape."""
        schema = ResolvedSchema(widths={"preds.jets.jets_classification": 3})
        with pytest.raises(BindError, match="no statically resolved width"):
            schema.width(_REG_PRED_KEY)
