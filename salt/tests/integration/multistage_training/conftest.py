"""Shared data, datamodule and Trainer builders for the multistage training tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from lightning import Trainer

from salt.data import Features, H5StructuredReader, Labels, SaltDataModule
from salt.model.saltmodule import SaltModule
from salt.schema import dump_schema, save_schema
from salt.tests._fixtures.dummy_h5 import dummy_h5
from salt.tests._fixtures.gn2v2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    write_parity_norm_dict,
)


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("multistage")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    h5_path, _, _ = dummy_h5(base)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


def _data_modules(data) -> dict:
    return {
        "reader": H5StructuredReader(groups={"jets": {}, "tracks": {}}, schema=data["schema"]),
        "features": Features(
            variables={"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}
        ),
        "labels": Labels(),
    }


def build_datamodule(data) -> SaltDataModule:
    return SaltDataModule(
        _data_modules(data), batch_size=100, num_workers=0,
        train_file=data["h5"], val_file=data["h5"], test_file=data["h5"],
    )


class _DeterministicDataModule(SaltDataModule):
    """A SaltDataModule whose train loader does NOT shuffle — so epoch e yields
    the same batches every run (resume-equivalence needs a deterministic loader).
    """

    def train_dataloader(self):  # noqa: D102 - see class docstring
        return self.get_dataloader(dataset=self.train_dset, stage="fit", shuffle=False)


def build_dm(data) -> _DeterministicDataModule:
    return _DeterministicDataModule(
        _data_modules(data), batch_size=100, num_workers=0,
        train_file=data["h5"], val_file=data["h5"], test_file=data["h5"],
    )


def make_trainer(*, max_epochs: int, callbacks: list, enable_checkpointing: bool = False,
                 **kwargs) -> Trainer:
    # 5 train steps/epoch (1000 jets / batch 100 = 10, limited to 5) so a 4-epoch
    # fit has estimated_stepping_batches == 20 (a 2-stage split of [10, 10]); a
    # per-stage OneCycle needs >~8 steps or its warmup phase degenerates.
    # enable_checkpointing must be True on the leg that carries a ModelCheckpoint.
    return Trainer(
        accelerator="cpu", devices=1, logger=False, max_epochs=max_epochs,
        limit_train_batches=5, limit_val_batches=1, num_sanity_val_steps=0,
        enable_checkpointing=enable_checkpointing, enable_progress_bar=False,
        enable_model_summary=False, log_every_n_steps=1, callbacks=callbacks, **kwargs,
    )


def _module_param_ids(model: SaltModule, name: str) -> set[int]:
    return {id(p) for p in model.net[name].parameters()}
