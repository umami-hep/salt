"""Gate G7e: per-stage scoped callbacks.

Top-level (global) callbacks ALWAYS propagate for the whole fit and are never
re-instantiated (Lightning fixes ``trainer.callbacks`` at fit start). A stage may
declare additional scoped `callbacks`; the auto-injected `StageScopedCallbacks`
coordinator instantiates the active stage's scoped callbacks FRESH at stage entry,
forwards Lightning's per-stage hooks to them ONLY while their stage is active, and
tears them down at exit. The user's effective per-stage set is therefore the
persistent globals plus the freshly-instantiated stage-scoped delegates.

G7e: a stage-scoped callback receives hooks only within its stage; fresh instance
at stage entry; the combined-with-global set is correct; teardown is clean; bad
callback specs fail at fit start, not at the boundary. All DataLoaders use
``num_workers=0`` (agent memcg gotcha).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from lightning import Callback, Trainer, seed_everything

from salt.callbacks.schedule import StageScopedCallbacks, TrainingScheduleCallback
from salt.data import Features, GraphDataModule, H5StructuredReader, Labels
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

SEED = 1234
LRS = {"initial": 1e-3, "max": 5e-3, "end": 1e-4, "pct_start": 0.25}

# module-level sink so freshly-instantiated (per-stage) delegates can record into a
# shared place; cleared at the start of each test.
HOOK_LOG: list[tuple[str, str, int]] = []  # (tag, hook, stage_index)


class HookProbe(Callback):
    """Records ``(tag, hook, active_stage_index)`` into `HOOK_LOG` for the hooks the
    coordinator forwards, plus setup/teardown. Instantiated from a stage
    ``callbacks`` spec (or passed directly as a persistent global).
    """

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.epoch_starts = 0

    def setup(self, trainer, pl_module, stage) -> None:  # noqa: D102
        HOOK_LOG.append((self.tag, "setup", pl_module._current_stage_index))  # noqa: SLF001

    def teardown(self, trainer, pl_module, stage) -> None:  # noqa: D102
        HOOK_LOG.append((self.tag, "teardown", pl_module._current_stage_index))  # noqa: SLF001

    def on_train_epoch_start(self, trainer, pl_module) -> None:  # noqa: D102
        self.epoch_starts += 1
        HOOK_LOG.append((self.tag, "epoch_start", pl_module._current_stage_index))  # noqa: SLF001

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # noqa: D102
        HOOK_LOG.append((self.tag, "batch_end", pl_module._current_stage_index))  # noqa: SLF001


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("stagecb")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    h5_path = base / "pp_output_train.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"h5": h5_path, "nd": nd_path, "schema": schema_path}


def build_dm(data) -> GraphDataModule:
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


def make_trainer(*, max_epochs: int, callbacks: list) -> Trainer:
    return Trainer(
        accelerator="cpu", devices=1, logger=False, max_epochs=max_epochs,
        limit_train_batches=5, limit_val_batches=1, num_sanity_val_steps=0,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, log_every_n_steps=1, callbacks=callbacks,
    )


_PROBE = "salt.tests.integration.test_stage_callbacks.HookProbe"


class TestG7eStageScopedCallbacks:
    SCHEDULE = {
        "stages": {
            "a": {"epochs": 2, "callbacks": [{"class_path": _PROBE, "init_args": {"tag": "A"}}]},
            "b": {"callbacks": [{"class_path": _PROBE, "init_args": {"tag": "B"}}]},
        }
    }

    def _run(self, data) -> HookProbe:
        HOOK_LOG.clear()
        seed_everything(SEED, workers=True)
        model = SaltModule(build_gn2v2_modules(data["nd"]), lrs=LRS, training_schedule=self.SCHEDULE)
        global_probe = HookProbe(tag="GLOBAL")
        # coordinator AFTER the transition driver so it sees the advanced stage;
        # global_probe is a persistent top-level callback (never re-instantiated).
        make_trainer(
            max_epochs=4,
            callbacks=[TrainingScheduleCallback(), StageScopedCallbacks(), global_probe],
        ).fit(model, build_dm(data))
        return global_probe

    def test_stage_scoped_callbacks_only_fire_within_their_stage(self, data):
        self._run(data)
        a_hook_stages = {s for (t, h, s) in HOOK_LOG if t == "A" and h in {"epoch_start", "batch_end"}}
        b_hook_stages = {s for (t, h, s) in HOOK_LOG if t == "B" and h in {"epoch_start", "batch_end"}}
        assert a_hook_stages == {0}  # stage-A delegate only active in stage 0
        assert b_hook_stages == {1}  # stage-B delegate only active in stage 1
        # A saw exactly stage 0's two epochs; B saw stage 1's two epochs.
        assert sum(1 for (t, h, _) in HOOK_LOG if t == "A" and h == "epoch_start") == 2
        assert sum(1 for (t, h, _) in HOOK_LOG if t == "B" and h == "epoch_start") == 2

    def test_scoped_callbacks_are_set_up_and_torn_down(self, data):
        self._run(data)
        assert ("A", "setup", 0) in HOOK_LOG  # A set up on entering stage 0
        assert any(t == "A" and h == "teardown" for (t, h, _) in HOOK_LOG)  # A torn down at exit
        assert ("B", "setup", 1) in HOOK_LOG  # B set up on entering stage 1
        assert any(t == "B" and h == "teardown" for (t, h, _) in HOOK_LOG)  # B torn down at fit end

    def test_global_callback_persists_across_all_stages(self, data):
        global_probe = self._run(data)
        global_stages = {s for (t, h, s) in HOOK_LOG if t == "GLOBAL" and h == "epoch_start"}
        assert global_stages == {0, 1}  # global fires in BOTH stages (persistent)
        assert global_probe.epoch_starts == 4  # one instance, accumulated all epochs

    def test_scoped_delegate_is_a_fresh_instance_per_stage(self, data):
        self._run(data)
        # A and B are distinct freshly-instantiated delegates (their setups carry
        # different tags at different stages — no shared/re-used instance).
        setups = [(t, s) for (t, h, s) in HOOK_LOG if h == "setup" and t in {"A", "B"}]
        assert ("A", 0) in setups
        assert ("B", 1) in setups


class TestG7eValidation:
    def test_bad_class_path_fails_at_fit_start(self, data):
        schedule = {
            "stages": {"fit": {"callbacks": [{"class_path": "salt.nonexistent.NoSuchCallback"}]}}
        }
        seed_everything(SEED, workers=True)
        model = SaltModule(build_gn2v2_modules(data["nd"]), lrs=LRS, training_schedule=schedule)
        with pytest.raises((ImportError, ModuleNotFoundError, AttributeError)):
            make_trainer(
                max_epochs=2, callbacks=[StageScopedCallbacks()]
            ).fit(model, build_dm(data))
