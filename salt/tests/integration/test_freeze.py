"""W2 gates for the module-freeze machinery (plan D1 freeze semantics).

Gates:

- **G2a** — a frozen module has ``requires_grad=False`` on all its params AND is
  in ``eval()`` mode during training, and STAYS in eval across an epoch boundary
  (Lightning re-calls ``model.train()`` every epoch; `SaltModule.train` must
  re-assert eval on frozen modules).
- **G2b** — the optimizer is built over TRAINABLE params only: its param set
  equals ``{p for p in model.parameters() if p.requires_grad}`` and excludes the
  frozen module's params — checked for AdamW AND HybridMuonAdamW.
- **G2d** — no ``training_schedule`` → zero behaviour change: the optimizer holds
  every parameter and nothing is frozen.
- **G2e** — composition smoke: ``--init_from`` + a single-stage schedule freezing
  every loaded module → only the new module's params reach the optimizer, and
  training runs with finite loss.

All DataLoaders use ``num_workers=0`` (agent memcg gotcha).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning import Callback, Trainer

from salt.data import Features, GraphDataModule, H5StructuredReader, Labels
from salt.model.modules.losses import LossSum
from salt.model.modules.tasks import ClassificationTaskModule
from salt.model.saltmodule import SaltModule
from salt.schema import dump_schema, save_schema
from salt.testing.inputs import write_dummy_file
from salt.tests._fixtures.gn2v2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    build_gn2v2_modules,
    write_parity_norm_dict,
)

pytestmark = pytest.mark.integration

LRS = {"initial": 1e-3, "max": 5e-3, "end": 1e-4, "pct_start": 0.1}


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("freeze")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    h5_path = base / "pp_output_train.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


def build_datamodule(data) -> GraphDataModule:
    modules = {
        "reader": H5StructuredReader(groups={"jets": {}, "tracks": {}}, schema=data["schema"]),
        "features": Features(
            variables={"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}
        ),
        "labels": Labels(),
    }
    return GraphDataModule(
        modules, batch_size=100, num_workers=0,
        train_file=data["h5"], val_file=data["h5"], test_file=data["h5"],
    )


def build_model(modules, **kwargs) -> SaltModule:
    return SaltModule(modules, lrs=LRS, **kwargs)


def make_trainer(**kwargs) -> Trainer:
    kwargs.setdefault("max_epochs", 1)
    kwargs.setdefault("limit_train_batches", 1)
    kwargs.setdefault("limit_val_batches", 1)
    return Trainer(
        accelerator="cpu", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, log_every_n_steps=1, num_sanity_val_steps=0,
        **kwargs,
    )


def offline_bind(model: SaltModule, dm: GraphDataModule, max_epochs: int = 10) -> None:
    """Bind a model against a datamodule WITHOUT a training loop, with a stub
    trainer exposing the fields `configure_optimizers`/schedule-setup read.
    """
    dm.set_sinks(model.sink_demand())
    dm.setup("fit")
    model._trainer = SimpleNamespace(  # noqa: SLF001
        datamodule=dm, callbacks=[], max_epochs=max_epochs, estimated_stepping_batches=100
    )
    model.setup("fit")


def _param_ids(params) -> set[int]:
    return {id(p) for p in params}


def _optimizer_param_ids(opt) -> set[int]:
    return {id(p) for group in opt.param_groups for p in group["params"]}


def _module_param_ids(model: SaltModule, name: str) -> set[int]:
    return {id(p) for p in model.net[name].parameters()}


class TestG2aFreezeAppliedAndPersists:
    def test_frozen_params_requires_grad_false_and_eval(self, data):
        model = build_model(
            build_gn2v2_modules(data["nd"]),
            training_schedule={"stages": {"fit": {"frozen": ["encoder", "track_embed"]}}},
        )
        offline_bind(model, build_datamodule(data))

        for name in ("encoder", "track_embed"):
            assert not any(p.requires_grad for p in model.net[name].parameters()), name
            assert not model.net[name].training, name  # in eval mode
        # a non-frozen module keeps grads + train mode
        assert all(p.requires_grad for p in model.net["pool"].parameters())

    def test_frozen_module_stays_eval_across_epoch_boundary(self, data):
        # Lightning calls model.train() at every epoch start; the override must
        # keep the frozen module in eval each epoch.
        seen: list[bool] = []

        class Rec(Callback):
            def on_train_epoch_start(self, _t, module):
                seen.append(module.net["encoder"].training)

        model = build_model(
            build_gn2v2_modules(data["nd"]),
            training_schedule={"stages": {"fit": {"frozen": ["encoder"]}}},
        )
        make_trainer(max_epochs=2, callbacks=[Rec()]).fit(model, build_datamodule(data))

        assert len(seen) == 2  # two epochs observed
        assert seen == [False, False]  # frozen module NEVER in train mode
        assert not model.net["encoder"].training  # still eval after fit


class TestG2bOptimizerExcludesFrozen:
    @pytest.mark.parametrize("optimizer", ["AdamW", "HybridMuonAdamW"])
    def test_optimizer_param_set_is_trainable_only(self, data, optimizer):
        model = build_model(
            build_gn2v2_modules(data["nd"]),
            optimizer=optimizer,
            training_schedule={"stages": {"fit": {"frozen": ["encoder"]}}},
        )
        offline_bind(model, build_datamodule(data))
        [opt], _ = model.configure_optimizers()

        trainable = _param_ids(p for p in model.parameters() if p.requires_grad)
        opt_ids = _optimizer_param_ids(opt)
        assert opt_ids == trainable
        # the frozen encoder's params are excluded entirely
        assert not (_module_param_ids(model, "encoder") & opt_ids)
        # sanity: encoder actually HAS params (so the exclusion is meaningful)
        assert _module_param_ids(model, "encoder")


class TestG2dNoScheduleNoChange:
    def test_optimizer_holds_all_params(self, data):
        model = build_model(build_gn2v2_modules(data["nd"]))  # no training_schedule
        offline_bind(model, build_datamodule(data))
        [opt], _ = model.configure_optimizers()

        all_ids = _param_ids(model.parameters())
        assert _optimizer_param_ids(opt) == all_ids
        assert model._frozen_module_names == set()  # noqa: SLF001
        assert all(p.requires_grad for p in model.parameters())


class TestMultiStageBindsAtFit:
    """A >1-stage schedule is now EXECUTED (W3 removed W2's fit-time rejection):
    it binds and applies stage 0's freeze mask at setup.
    """

    def test_multi_stage_binds_and_applies_stage0_freeze(self, data):
        model = build_model(
            build_gn2v2_modules(data["nd"]),
            training_schedule={
                "stages": {"warmup": {"epochs": 1, "frozen": ["encoder"]}, "full": {}}
            },
        )
        assert model._schedule.is_multi_stage  # noqa: SLF001
        offline_bind(model, build_datamodule(data))  # no rejection
        assert model._current_stage_index == 0  # noqa: SLF001
        assert model._frozen_module_names == {"encoder"}  # noqa: SLF001 - stage-0 mask
        assert not any(p.requires_grad for p in model.net["encoder"].parameters())


class TestG2eInitFromComposesWithFreeze:
    @staticmethod
    def _surgery_modules(nd) -> dict:
        # drop track_vertexing, add a new head — mirrors the W1 G1b surgery
        modules = build_gn2v2_modules(nd)
        del modules["track_vertexing"]
        extra = ClassificationTaskModule(
            stream="jets", label="flavour_label", class_names=["bjets", "cjets", "ujets"],
            input="pooled.global", dense={"hidden_layers": [16], "activation": "ReLU"},
        )
        extra.name = "extra_jets"
        modules["extra_jets"] = extra
        modules["loss"] = LossSum()
        return modules

    @pytest.fixture(scope="class")
    def base_ckpt(self, data) -> Path:
        model = build_model(build_gn2v2_modules(data["nd"]))
        make_trainer().fit(model, build_datamodule(data))
        ckpt = data["dir"] / "freeze_base.ckpt"
        model._trainer.save_checkpoint(ckpt)  # noqa: SLF001
        return ckpt

    def test_only_new_module_trains(self, data, base_ckpt):
        # freeze EVERY loaded (retained) module; only extra_jets (new) trains
        retained = [
            "norm", "track_embed", "encoder", "pool", "jets_classification", "track_origin",
        ]
        losses: list[torch.Tensor] = []
        captured: dict[str, object] = {}

        class Rec(Callback):
            def on_train_start(self, trainer, _m):
                captured["opt"] = trainer.optimizers[0]

            def on_train_batch_end(self, _t, _m, outputs, _b, _i):
                losses.append(outputs["loss"].detach())

        model = build_model(
            self._surgery_modules(data["nd"]),
            training_schedule={"stages": {"fit": {"frozen": retained}}},
        )
        model._init_from = str(base_ckpt)  # noqa: SLF001
        make_trainer(callbacks=[Rec()]).fit(model, build_datamodule(data))

        assert model._init_warm_started  # noqa: SLF001 - warm start ran
        assert losses  # training ran
        assert all(torch.isfinite(x) for x in losses)  # finite loss throughout
        opt_ids = _optimizer_param_ids(captured["opt"])
        assert opt_ids == _module_param_ids(model, "extra_jets")  # ONLY the new head
        # every retained module is frozen out of the optimizer
        for name in retained:
            assert not (_module_param_ids(model, name) & opt_ids), name
