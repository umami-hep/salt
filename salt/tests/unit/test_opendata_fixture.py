"""The shipped open-data config's runtime ``setup('test')`` on the opendata fixture
(the C3 leg of the study's sink-prep parity probe, kept green in-tree).
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
import yaml

from salt.graph.spec import Mode
from salt.main import CONFIG_DIR, SaltCLI
from salt.outputs.sinks.registry import iter_sinks
from salt.tests._fixtures.gn2v2_fixture import JET_VARIABLES
from salt.tests._fixtures.opendata_fixture import (
    OPENDATA_CLASS_NAMES,
    OPENDATA_TRACK_VARIABLES,
    build_opendata_data,
    opendata_overrides,
)
from salt.utils.config_utils import disable_logger_in_config

_OPENDATA_CFG = CONFIG_DIR / "gn2v2-opendata.yaml"


def test_fixture_matches_shipped_opendata_config():
    """Drift guard: the fixture's variable/class lists match the shipped config."""
    cfg = yaml.safe_load(_OPENDATA_CFG.read_text())
    features = cfg["data"]["modules"]["features"]["init_args"]["variables"]
    assert features["tracks"] == OPENDATA_TRACK_VARIABLES
    assert features["jets"] == list(JET_VARIABLES)
    class_names = cfg["model"]["init_args"]["modules"]["jets_classification"]["init_args"][
        "class_names"
    ]
    assert class_names == OPENDATA_CLASS_NAMES


@pytest.fixture(scope="module")
def opendata(tmp_path_factory) -> dict:
    """Module-scoped fixture data for the shipped opendata config."""
    return build_opendata_data(tmp_path_factory.mktemp("opendata_fixture"))


def test_shipped_opendata_compiles_test_plan_on_fixture(opendata):
    """``salt fit``'s SHIPPED gn2v2-opendata config compiles a TEST plan (run-free)
    against the fixture, mirroring the runtime setup('test') recipe.
    """
    args = [
        "--config",
        disable_logger_in_config(str(_OPENDATA_CFG)),
        *(f"--{k}={v}" for k, v in opendata_overrides(opendata).items()),
        "--trainer.accelerator=cpu",
        "--trainer.devices=1",
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*args parameter is intended to run from within Python.*"
        )
        cli = SaltCLI(args=args, run=False)
    model, dm = cli.model, cli.datamodule
    model._trainer = SimpleNamespace(  # noqa: SLF001
        callbacks=[*iter_sinks(cli.trainer), *cli.trainer.callbacks], datamodule=dm
    )
    dm.set_sinks(model.sink_demand())
    dm.setup("test")
    model.setup("test")

    assert Mode.TEST in model.plans
    # the run-free parse injects the implicit H5 sink under this name (salt/main.py:949),
    # and the runtime prep folds it as a node
    assert "h5_output" in model.plans[Mode.TEST].module_names
    # the runtime TEST boundary demand appends meta.rows last
    assert model.sink_demand()[Mode.TEST][-1] == "meta.rows"
