"""Regression guard: the MaskFormer TEST-mode two-phase bind must not require the
FIT-only object-regression prediction width.

Bug (exp-21 smoke #3, job 2642): ``salt2 test`` on the shipped ``MaskFormer.yaml``
crashed at the TEST-stage two-phase bind with::

    salt.core.graph.BindError: no statically resolved width for bundle key
    'preds.objects.regression'

Root cause: ``MaskFormerMatchedLoss.bind`` (`salt/core/nn/maskformer_loss.py`)
unconditionally looked up ``schema.width('preds.objects.regression')``. The matched
loss is FIT|VAL-only (its ``declare_io`` is empty in TEST), and the object-regression
head opts out of TEST via ``expose: [fit, val, onnx]`` — so that key is pruned from a
TEST-only bind schema. But ``bind_all`` runs ``bind`` on EVERY configured module
regardless of the compiled mode, and ``SaltModule.setup('test')`` builds the schema
from the TEST plan ALONE (`saltmodule.py` — FIT stage compiles FIT+VAL, TEST stage
compiles TEST only), so the width never resolves and the lookup raised.

Why the pre-existing suites missed it: FIT-stage binds compile FIT+VAL together (the
regression head is exposed there), and the render/plot ``_resolve_widths`` unifies
ALL modes, so both always supply the width. The TEST-alone bind was the untested
combination — first exercised by a real ``salt2 test`` on MaskFormer.

Fix: `MaskFormerMatchedLoss.bind` guards the lookup on schema presence — it validates
the pred/target width agreement only when both keys are in the schema (the FIT-stage
bind), and skips otherwise. A genuinely missing regression producer in a training
plan still fails earlier as a planner ``ConnectivityError``, so the guard cannot mask
a real training-mode wiring bug.

These tests are data-free static plan compiles (CPU, no data files, no GPU) — the
``norm_dict=unused.yaml`` override matches the shipped config's static-validation
recipe (MaskFormer.yaml header).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from salt.core.cli import load_config
from salt.core.graph.errors import ConfigError
from salt.core.graph.planner import compile_plan
from salt.core.graph.spec import Mode
from salt.core.nn import MaskFormerMatchedLoss
from salt.core.nn.bind import BindError, ResolvedSchema, bind_all, resolve_bind_schema

# this file is at salt/tests/unit/nn/ — the configs live at salt/core/configs/
_MASKFORMER = str(Path(__file__).parents[3] / "core" / "configs" / "MaskFormer.yaml")
_OVERRIDES = ["model.modules.norm.init_args.norm_dict=unused.yaml"]

_REG_PRED_KEY = "preds.objects.regression"
_REG_TGT_KEY = "targets.objects.regression"


def _compile(mode: Mode):
    """Load MaskFormer.yaml and compile one mode's plan (static, no data).

    Returns
    -------
    tuple[GraphConfig, Plan]
        The freshly loaded config (live modules) and the compiled plan.
    """
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
        """``SaltModule.setup('test')`` shape: TEST plan alone -> bind_all must not raise.

        This is the exact failing path — the regression pred key is pruned from the
        TEST schema, and the matched loss's ``bind`` must tolerate its absence.
        """
        cfg, test_plan = _compile(Mode.TEST)
        schema = resolve_bind_schema([test_plan])
        # the key is genuinely absent in TEST (the regression head opts out of TEST)
        assert _REG_PRED_KEY not in schema.widths
        # the fix: binding the model modules against the TEST-only schema must succeed
        bind_all(cfg.model_modules, schema)

    def test_fit_mode_binds_regression_width(self):
        """``SaltModule.setup('fit')`` shape: FIT+VAL -> the width resolves and binds.

        Guards against the fix accidentally suppressing the real training-mode
        validation — the width must still resolve (5 targets: pt/Lxy/deta/dphi/mass)
        and ``bind_all`` must run clean.
        """
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
