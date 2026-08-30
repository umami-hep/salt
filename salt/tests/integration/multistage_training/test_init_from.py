"""The ``--init_from`` weights-only warm-start path.

Covered:

- an identical-architecture warm start yields a state_dict
  bitwise-equal to a plain strict load of the same checkpoint.
- a swap-one-head config: retained modules load fully, the new head
  is fresh-inited + materialised, the dropped module's keys are skipped.
- partial coverage of a retained module (changed inner width) hard
  fails with a clear `ConfigError`.
- a newly-added `Normaliser` (absent from the checkpoint) materialises
  from its norm_dict and forward-passes;
  a retained `Normaliser` is NOT re-materialised (keeps loaded stats).

The offline bind-then-load harness mirrors the v1->v2 converter's
``_strict_load_check`` (scripts/convert_v1_model.py). All DataLoaders use
``num_workers=0``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning import Callback, Trainer

from salt.data import Features, SaltDataModule, H5StructuredReader, Labels
from salt.graph import ConfigError
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
RETAINED = {  # the params-bearing gn2v2 modules present in every ckpt
    "norm", "track_embed", "encoder", "pool",
    "jets_classification", "track_origin", "track_vertexing",
}


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("init_from")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    h5_path = base / "pp_output_train.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


def build_datamodule(data) -> SaltDataModule:
    modules = {
        "reader": H5StructuredReader(groups={"jets": {}, "tracks": {}}, schema=data["schema"]),
        "features": Features(
            variables={"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}
        ),
        "labels": Labels(),
    }
    return SaltDataModule(
        modules, batch_size=100, num_workers=0,
        train_file=data["h5"], val_file=data["h5"], test_file=data["h5"],
    )


def build_model(modules) -> SaltModule:
    return SaltModule(modules, lrs=LRS)


def make_trainer(**kwargs) -> Trainer:
    kwargs.setdefault("max_epochs", 1)
    kwargs.setdefault("limit_train_batches", 2)
    kwargs.setdefault("limit_val_batches", 1)
    return Trainer(
        accelerator="cpu", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, log_every_n_steps=1, num_sanity_val_steps=0,
        **kwargs,
    )


def offline_bind(model: SaltModule, dm: SaltDataModule, stage: str = "fit") -> None:
    """Bind a model against a datamodule WITHOUT a training loop (converter
    pattern) — triggers `setup(stage)` and any `--init_from` warm start.
    """
    dm.set_sinks(model.sink_demand())
    dm.setup(stage)
    model._trainer = SimpleNamespace(datamodule=dm, callbacks=[])  # noqa: SLF001
    model.setup(stage)


@pytest.fixture(scope="module")
def base_ckpt(data) -> Path:
    """A trained GN2v2 checkpoint (weights + salt_core payload)."""  # noqa: DOC201
    model = build_model(build_gn2v2_modules(data["nd"]))
    dm = build_datamodule(data)
    make_trainer().fit(model, dm)
    ckpt = data["dir"] / "base.ckpt"
    model._trainer.save_checkpoint(ckpt)  # noqa: SLF001
    return ckpt


class TestIdenticalArch:
    def test_warm_start_bitwise_equals_strict_load(self, data, base_ckpt):
        # reference: the existing data-less STRICT load
        strict = SaltModule.load_from_checkpoint(
            base_ckpt, modules=build_gn2v2_modules(data["nd"]), map_location="cpu"
        )
        # candidate: an identical-arch --init_from warm start
        warm = build_model(build_gn2v2_modules(data["nd"]))
        warm._init_from = str(base_ckpt)  # noqa: SLF001
        offline_bind(warm, build_datamodule(data))

        assert warm._init_warm_started  # noqa: SLF001
        assert warm._init_loaded_modules == RETAINED  # noqa: SLF001
        strict_sd, warm_sd = strict.state_dict(), warm.state_dict()
        assert set(strict_sd) == set(warm_sd)
        for key, value in strict_sd.items():
            assert torch.equal(value, warm_sd[key]), key


class TestSwapHead:
    @staticmethod
    def _surgery_modules(nd) -> dict:
        # drop track_vertexing (-> a dropped ckpt module) and add a distinctly
        # named jets head (-> a new module, fresh init + materialised)
        modules = build_gn2v2_modules(nd)
        del modules["track_vertexing"]
        extra = ClassificationTaskModule(
            stream="jets", label="flavour_label", class_names=["bjets", "cjets", "ujets"],
            input="pooled.global", dense={"hidden_layers": [16], "activation": "ReLU"},
        )
        extra.name = "extra_jets"
        modules["extra_jets"] = extra
        modules["loss"] = LossSum()  # fresh -> SaltModule re-narrows over the new head set
        return modules

    def test_retained_load_new_fresh_dropped_skipped(self, data, base_ckpt):
        model = build_model(self._surgery_modules(data["nd"]))
        model._init_from = str(base_ckpt)  # noqa: SLF001
        offline_bind(model, build_datamodule(data))

        loaded = model._init_loaded_modules  # noqa: SLF001
        # every retained gn2v2 module loaded; the new head did NOT
        assert {"norm", "track_embed", "encoder", "pool",
                "jets_classification", "track_origin"} <= loaded
        assert "extra_jets" not in loaded
        assert "track_vertexing" not in loaded  # dropped (not in this config)
        # retained weights are bitwise from the checkpoint
        ckpt_sd = torch.load(base_ckpt, map_location="cpu", weights_only=False)["state_dict"]
        model_sd = model.state_dict()
        assert torch.equal(model_sd["net.pool.gate_nn.weight"], ckpt_sd["net.pool.gate_nn.weight"])

    def test_swap_head_trains_end_to_end(self, data, base_ckpt):
        # the full path: warm start inside trainer.fit -> selective materialise
        # -> the fresh head trains alongside the frozen-nothing backbone
        losses: list[torch.Tensor] = []

        class Rec(Callback):
            def on_train_batch_end(self, _t, _m, outputs, _b, _i):
                losses.append(outputs["loss"].detach())

        model = build_model(self._surgery_modules(data["nd"]))
        model._init_from = str(base_ckpt)  # noqa: SLF001
        make_trainer(callbacks=[Rec()]).fit(model, build_datamodule(data))
        assert losses and all(torch.isfinite(x) for x in losses)
        assert model._init_warm_started  # noqa: SLF001


class TestPartialCoverage:
    def test_changed_inner_width_hard_fails(self, data, base_ckpt):
        # same module names, but a wider encoder -> retained modules only
        # PARTIALLY covered -> hard ConfigError before any load
        modules = build_gn2v2_modules(data["nd"], embed_dim=24, out_dim=24)
        model = build_model(modules)
        model._init_from = str(base_ckpt)  # noqa: SLF001
        with pytest.raises(ConfigError, match=r"PARTIALLY covered.*swap"):
            offline_bind(model, build_datamodule(data))


class TestSelectiveMaterialise:
    def test_new_normaliser_materialises_and_forward_passes(self, data, base_ckpt, tmp_path):
        # a checkpoint that never carried a normaliser (strip net.norm.*): the
        # config's `norm` is a NEW module -> must materialise
        ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
        ckpt["state_dict"] = {
            k: v for k, v in ckpt["state_dict"].items() if not k.startswith("net.norm.")
        }
        stripped = tmp_path / "no_norm.ckpt"
        torch.save(ckpt, stripped)

        model = build_model(build_gn2v2_modules(data["nd"]))
        model._init_from = str(stripped)  # noqa: SLF001
        make_trainer(limit_train_batches=1).fit(model, build_datamodule(data))

        norm = model.net["norm"]
        assert "norm" not in model._init_loaded_modules  # noqa: SLF001 - norm is NEW
        assert bool(norm.materialised)  # selective materialise ran on it
        expect_means = torch.tensor([0.1 * (i + 1) for i in range(len(JET_VARIABLES))])
        assert torch.allclose(norm.means_jets, expect_means)

    def test_retained_normaliser_not_rematerialised(self, data, base_ckpt):
        # a FULL warm start with a BOGUS norm_dict path: `norm` is retained
        # (loaded), so on_fit_start must SKIP it -> no file access, stats survive
        modules = build_gn2v2_modules(data["dir"] / "does_not_exist.yaml")
        model = build_model(modules)
        model._init_from = str(base_ckpt)  # noqa: SLF001
        make_trainer(limit_train_batches=1).fit(model, build_datamodule(data))

        assert "norm" in model._init_loaded_modules  # noqa: SLF001
        # buffers came from the checkpoint (the bogus norm_dict was never read)
        ckpt_sd = torch.load(base_ckpt, map_location="cpu", weights_only=False)["state_dict"]
        assert torch.equal(model.net["norm"].means_jets, ckpt_sd["net.norm.means_jets"])
        assert bool(model.net["norm"].materialised)


class TestMutualExclusivityUnit:
    """--init_from vs --ckpt_path guard, exercised without a full CLI parse."""

    def test_setup_test_never_warm_starts(self, data, base_ckpt):
        # init_from is fit-only: a `test` setup must ignore it entirely
        model = build_model(build_gn2v2_modules(data["nd"]))
        model._init_from = str(base_ckpt)  # noqa: SLF001
        offline_bind(model, build_datamodule(data), stage="test")
        assert not model._init_warm_started  # noqa: SLF001
