"""The shipped fine-tuning templates are trained through, not merely parsed.

``finetune_gn3large.yaml`` and ``finetune_gn3large_new_head.yaml`` are the
worked examples ``docs/tutorials/finetuning.md`` tells users to copy, so they
have to keep working as configs, not just as YAML. ``test_finetune_configs.py``
loads and inspects them; nothing trained through them, which is exactly how a
documented template rots silently.

Each template is exercised the way its own header says to use it: stack it on a
saved run's ``config.yaml``, warm-start from that run's checkpoint, and run the
full multi-stage schedule. Both stages must execute — a warm-up that freezes
everything but one head, then a full fine-tune — because that is what breaks if
a ``trainable:`` entry stops naming a real module or the schedule schema moves.

``fast_dev_run`` cannot drive these: it pins ``max_epochs`` to 1 and the shipped
schedules allocate 5 epochs to the warm-up alone. Runs here are kept cheap with
``limit_{train,val}_batches`` instead, so every epoch boundary is still crossed.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from salt.main import CONFIG_DIR
from salt.main import main as salt_main
from salt.schema import dump_schema, save_schema
from salt.testing.inputs import write_dummy_file, write_dummy_norm_dict
from salt.utils.array_utils import join_structured_arrays

pytestmark = pytest.mark.cpu_always

# The in-repo, production-faithful GN3Large stand-in the templates target: the
# same base test_finetune_configs.py validates their freeze specs against.
BASE_CFG = CONFIG_DIR / "GN3" / "GN3V00.yaml"
SAME_HEADS = CONFIG_DIR / "finetune" / "finetune_gn3large.yaml"
NEW_HEAD = CONFIG_DIR / "finetune" / "finetune_gn3large_new_head.yaml"

# The new-head template's schedule: 5 warm-up epochs, then >=1 full fine-tune.
# 6 is the cheapest total that exercises both stages.
TOTAL_EPOCHS = 6


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    """GN3-flavour synthetic inputs, plus the new head's extra jet label."""
    base = tmp_path_factory.mktemp("finetune_templates")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_dummy_norm_dict(nd_path, cd_path, is_gn3=True)
    h5_path = base / "pp_output_train.h5"
    write_dummy_file(h5_path, nd_path, is_gn3=True)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)

    # The new-head template bolts on a head labelled `large_r_flavour_label`,
    # which no shipped fixture carries — that label IS the domain shift the
    # template demonstrates, so the test supplies it rather than dodging it.
    new_h5 = base / "pp_output_train_large_r.h5"
    with h5py.File(h5_path) as src, h5py.File(new_h5, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        for name, dataset in src.items():
            if name != "jets":
                dst.create_dataset(name, data=dataset[:])
                for key, value in dataset.attrs.items():
                    dst[name].attrs[key] = value
                continue
            jets = dataset[:]
            rng = np.random.default_rng(42)
            extra = rng.integers(0, 4, size=len(jets)).astype("i4")
            extra = extra.view(np.dtype([("large_r_flavour_label", "i4")]))
            dst.create_dataset("jets", data=join_structured_arrays([jets, extra]))
            for key, value in dataset.attrs.items():
                dst["jets"].attrs[key] = value
            dst["jets"].attrs["large_r_flavour_label"] = ["hbb", "hcc", "top", "qcd"]
    new_schema = base / "schema_large_r.yaml"
    save_schema(dump_schema(new_h5), new_schema)

    return {
        "dir": base,
        "nd": nd_path,
        "h5": h5_path,
        "schema": schema_path,
        "new_h5": new_h5,
        "new_schema": new_schema,
    }


def _data_args(h5: Path, schema: Path) -> list[str]:
    return [
        f"--data.train_file={h5}",
        f"--data.val_file={h5}",
        f"--data.modules.reader.init_args.schema={schema}",
        # shipped configs assume large training machines
        "--data.num_workers=0",
        # the fixture holds 1000 jets; GN3V00 ships batch_size 1000
        "--data.batch_size=50",
    ]


def _trainer_args(root: Path, epochs: int) -> list[str]:
    return [
        f"--trainer.default_root_dir={root}",
        # auto, not cpu: on a GPU runner these must exercise the GPU path
        "--trainer.accelerator=auto",
        f"--trainer.max_epochs={epochs}",
        # every epoch boundary is still crossed, which is what the multi-stage
        # schedule keys off; only the work inside each epoch shrinks
        "--trainer.limit_train_batches=1",
        "--trainer.limit_val_batches=1",
        "--trainer.num_sanity_val_steps=0",
        "--trainer.log_every_n_steps=1",
        # base ships a default-ON CometLogger; off so no offline archive lands
        "--trainer.logger=false",
        # null-delete the base ProgressBar: the stock enable_progress_bar=false
        # cannot coexist with a configured bar
        "--callbacks.progress=null",
    ]


@pytest.fixture(scope="module")
def pretrained(data, tmp_path_factory) -> dict[str, Path]:
    """A real 1-epoch GN3V00 run: the checkpoint + saved config a user warm-starts from."""
    root = tmp_path_factory.mktemp("finetune_base")
    rc = salt_main([
        "fit",
        "--config",
        str(BASE_CFG),
        *_data_args(data["h5"], data["schema"]),
        *_trainer_args(root, 1),
        f"--model.modules.norm.init_args.norm_dict={data['nd']}",
    ])
    assert rc == 0, "the GN3V00 base run the templates warm-start from must fit"

    ckpts = sorted(root.rglob("*.ckpt"))
    assert ckpts, f"the base run wrote no checkpoint under {root}"
    configs = sorted(root.rglob("config.yaml"))
    assert configs, f"the base run saved no config.yaml under {root}"
    return {"ckpt": ckpts[0], "config": configs[0]}


def _run_template(template: Path, pretrained, data, root: Path, h5: Path, schema: Path) -> int:
    return salt_main([
        "fit",
        "--config",
        str(pretrained["config"]),
        "--config",
        str(template),
        f"--init_from={pretrained['ckpt']}",
        *_data_args(h5, schema),
        *_trainer_args(root, TOTAL_EPOCHS),
        f"--model.modules.norm.init_args.norm_dict={data['nd']}",
    ])


def test_base_run_carries_the_modules_the_templates_freeze(pretrained):
    """The saved base config names the head both templates warm up."""
    cfg = yaml.safe_load(pretrained["config"].read_text())
    modules = cfg["model"]["init_args"]["modules"]
    assert "jets_classification" in modules, (
        "finetune_gn3large.yaml warms up `jets_classification`; the base config "
        "no longer defines it"
    )


class TestSameHeadsTemplate:
    """`finetune_gn3large.yaml` — worked example A, trained end to end."""

    def test_trains_through_both_schedule_stages(self, pretrained, data, tmp_path):
        rc = _run_template(
            SAME_HEADS, pretrained, data, tmp_path, data["h5"], data["schema"]
        )
        assert rc == 0, (
            "the shipped finetune_gn3large.yaml template failed to train. It is a "
            "documented worked example (docs/tutorials/finetuning.md) — fix the "
            "template, not this test."
        )

    def test_writes_a_finetuned_checkpoint(self, pretrained, data, tmp_path):
        rc = _run_template(
            SAME_HEADS, pretrained, data, tmp_path, data["h5"], data["schema"]
        )
        assert rc == 0
        ckpts = sorted(tmp_path.rglob("*.ckpt"))
        assert ckpts, "the fine-tune run produced no checkpoint"


class TestNewHeadTemplate:
    """`finetune_gn3large_new_head.yaml` — worked example B, trained end to end."""

    def test_trains_through_with_the_added_head(self, pretrained, data, tmp_path):
        rc = _run_template(
            NEW_HEAD, pretrained, data, tmp_path, data["new_h5"], data["new_schema"]
        )
        assert rc == 0, (
            "the shipped finetune_gn3large_new_head.yaml template failed to train. "
            "It is a documented worked example (docs/tutorials/finetuning.md) — fix "
            "the template, not this test."
        )

    def test_added_head_is_absent_from_the_pretrained_checkpoint(self, pretrained):
        """The new head really is new — the warm start cannot be a no-op."""
        import torch  # noqa: PLC0415

        state = torch.load(pretrained["ckpt"], map_location="cpu", weights_only=False)
        keys = state.get("state_dict", state)
        assert not any("large_r_jet_classification" in k for k in keys), (
            "the base checkpoint already carries the head the template adds, so "
            "this test would no longer prove the new-head path works"
        )
