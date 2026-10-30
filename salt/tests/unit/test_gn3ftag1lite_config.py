"""The shipped small-R two-stage example `GN3/GN3FTAG1LITE.yaml`: config-only jsonargparse
instantiation (module groups expanded), and its plans compiled on a synthetic corpus."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from jsonargparse import ArgumentParser
from lightning import Callback

from salt.callbacks import InputNormalisationPlotter, KinematicReweightPlotter, PriorClosureMetrics
from salt.data import MODEL_VISIBLE_NAMESPACES, Processor
from salt.data.readers.ftag1lite_categories import SHOWERS, SIM_TYPES, SUBCAMPAIGNS, onehot_fields
from salt.graph.bundle import Bundle
from salt.graph.errors import ConnectivityError
from salt.graph.planner import compile_plan
from salt.graph.spec import Mode, flatten_spec, unflatten_spec
from salt.main import CONFIG_DIR
from salt.model.base import SaltModelModule
from salt.model.bind import resolve_bind_schema
from salt.model.modules import (ClassWeight, IterativeReweighting, KinematicWeight, LossSum,
                                MaskedInputNormaliser, SelectFields, VectorConcat)
from salt.model.saltmodule import _accumulator_sinks, _expand_module_groups
from salt.model.sink_prep import assert_no_dead_preds, boundary_demand, prepare_sinks
from salt.schedule import TrainingSchedule
from salt.tests._fixtures import joint_ftag1lite_corpus as fx
from salt.utils.config_utils import expand_includes

CONFIG = CONFIG_DIR / "GN3" / "GN3FTAG1LITE.yaml"
GHOST_FLAVOURS = [f"ghost{f}jets" for f in ("b", "c", "s", "ud", "g", "tau")]
FIT_VAL = (Mode.FIT, Mode.VAL)
VARIABLES = ["pt", "eta", "subcampaign", "shower", "sim_type"]  # spec + plotter demand order
PRIOR_ONLY = {"prior_input", "kinematic_weight"}  # the only consumers of the container one-hot
PRIOR_PRED = "preds.jets.kinematic_prior"


def _instantiate(kind, block):
    # deepcopy: parse_object would otherwise mutate the shared `cfg` into Namespaces
    parser = ArgumentParser(exit_on_error=False)
    parser.add_argument("--x", type=kind)
    built = parser.instantiate_classes(parser.parse_object({"x": copy.deepcopy(block)})).x
    if isinstance(built, dict):  # as SaltModule.__init__: expand groups, name, narrow losses.**
        built = _expand_module_groups(built)
        for key, module in built.items():
            module.name = key
        for loss in [m for m in built.values() if isinstance(m, LossSum) and not m.narrowed]:
            loss.narrow(LossSum.collect_loss_keys(built, Mode.FIT))
    return built


def _expanded(block) -> dict[str, SaltModelModule]:
    return _instantiate(dict[str, SaltModelModule], block)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(Path(expand_includes(str(CONFIG))).read_text())


@pytest.fixture(scope="module")
def modules(cfg) -> dict[str, SaltModelModule]:
    return _expanded(cfg["model"]["init_args"]["modules"])


@pytest.fixture(scope="module")
def plotter(cfg) -> KinematicReweightPlotter:
    return _instantiate(Callback, cfg["callbacks"]["reweight_plots"])


@pytest.fixture(scope="module")
def norm_plotter(cfg) -> InputNormalisationPlotter:
    return _instantiate(Callback, cfg["callbacks"]["input_norm_plots"])


def test_two_stage_schedule(cfg, modules):
    schedule = TrainingSchedule.from_config(cfg["training_schedule"], tuple(modules))
    assert [stage.name for stage in schedule.stages] == ["priors", "main"]
    priors, main = schedule.stages
    assert (priors.frac_samples, priors.epochs, priors.num_samples) == (0.1, None, None)
    # main is the final stage (the remaining max_epochs) and carries no sample budget
    assert priors.is_sample_stage and not main.is_sample_stage and schedule.stop_after is None
    assert priors.skip_losses == ("jets_classification",)
    assert main.skip_losses == ("kinematic_prior",)
    schedule.validate_epochs(cfg["trainer"]["max_epochs"])


def test_kinematic_weight_feeds_the_main_head(modules):
    head, kw, cw = (modules[k] for k in ("jets_classification", "kinematic_weight", "class_weight"))
    assert isinstance(kw, KinematicWeight) and isinstance(cw, ClassWeight)
    sample_weights = (kw.weight_field, cw.weight_key.split(".")[-1])
    assert head.sample_weights == sample_weights == ("nn_kinematic_weight", "flavour_class_weight")
    assert head.weight_label_keys == (kw.weight_key, cw.weight_key)
    assert head.loss_cfg["init_args"]["reduction"] == "none" and kw.balanced_prior is True
    prior = modules["kinematic_prior"]  # class-balanced CE
    assert prior.sample_weights == ("flavour_class_weight",)
    assert prior.weight_label_keys == (cw.weight_key,)
    assert prior.loss_cfg["init_args"]["reduction"] == "none"
    assert cw.n_classes == 6 and cw.label_key == kw.label_key and kw.target == "inclusive"
    assert list(kw.variables) == VARIABLES and set(kw.published) == set(VARIABLES)  # nn: published


def test_class_weight_is_label_only_and_training_only(modules):
    cw = modules["class_weight"]
    io = cw.declare_io(Mode.FIT)
    requires, produces = flatten_spec(io.requires), flatten_spec(io.produces)
    assert set(requires) == {"labels.jets.flavour_label"}
    assert set(produces) == {"labels.jets.flavour_class_weight"}
    assert "preds.jets.kinematic_prior" not in requires  # no graph cycle with the prior
    for io in (cw.declare_io(Mode.TEST), cw.declare_io(Mode.ONNX)):
        assert flatten_spec(io.requires) == {} and flatten_spec(io.produces) == {}


def test_prior_input_and_data_modules(cfg, modules):
    assert isinstance(modules["norm"], MaskedInputNormaliser)
    assert isinstance(modules["prior_kin"], SelectFields)
    assert isinstance(modules["prior_input"], VectorConcat)
    prior = modules["kinematic_prior"]  # exposed in every mode
    assert flatten_spec(prior.declare_io(Mode.TEST).produces)[prior.pred_key].active_in(Mode.TEST)
    data_modules = cfg["data"]["modules"]
    assert data_modules["container_onehot"]["class_path"] == "salt.data.ContainerOneHot"
    assert data_modules["container_onehot"]["init_args"]["include_sim_type"] is True
    assert data_modules["flavour_labeller"]["class_path"] == "salt.data.FtagLabeller"
    assert data_modules["flavour_labeller"]["init_args"]["class_names"] == GHOST_FLAVOURS
    io = {n: flatten_spec(m.declare_io(Mode.FIT).requires) for n, m in modules.items()}
    assert {n for n, req in io.items() if "inputs.container_onehot" in req} == PRIOR_ONLY


def test_reweight_plotter_parses_config_only(cfg, modules, plotter):
    block = cfg["callbacks"]["reweight_plots"]
    assert "variables" not in block["init_args"]  # the one spec lives on kinematic_weight
    assert isinstance(plotter, KinematicReweightPlotter)
    assert plotter.weight_module == "kinematic_weight"
    plots = block["init_args"]["plots"]
    splits = {(x, s) for x in ("pt", "eta") for s in ("subcampaign", "shower", "sim_type")}
    assert {(p["x"], p["split_by"]) for p in plots if "split_by" in p} == splits
    assert {p["x"] for p in plots if "split_by" not in p} == set(VARIABLES)
    # |eta| edges reach 5.0 so forward jets are plotted
    assert all(p["edges"][0] == 0.0 and p["edges"][-1] == 5.0 and len(p["edges"]) == 21
               for p in plots if p["x"] == "eta")
    kw = modules["kinematic_weight"]
    used = tuple(kw.published[v] for v in VARIABLES)
    assert tuple(plotter.fit_val_demand(modules)) == (kw.label_key, kw.weight_key, *used)
    for var, vocab in (("subcampaign", SUBCAMPAIGNS), ("shower", SHOWERS), ("sim_type", SIM_TYPES)):
        assert list(kw.variables[var].categories) == list(vocab)
    group = cfg["model"]["init_args"]["modules"]["kinematic_reweighting"]
    assert not any(key.startswith("bin_") for key in group["init_args"]["weight"])


def test_input_norm_plotter_parses_config_only(cfg, modules, norm_plotter):
    assert isinstance(norm_plotter, InputNormalisationPlotter)
    assert norm_plotter.normaliser == "norm" and norm_plotter.gaussian is False
    assert norm_plotter.class_task == "jets_classification"
    assert "output_dir" not in cfg["callbacks"]["input_norm_plots"]["init_args"]
    assert isinstance(modules["norm"], MaskedInputNormaliser)
    demand = ("inputs.jets", "inputs.tracks", "inputs.flows", "masks.tracks", "masks.flows")
    assert tuple(norm_plotter.fit_val_demand(modules)) == (*demand, "labels.jets.flavour_label")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory, cfg):
    """A seam `FTAG1LITEReader` on a synthetic corpus + the shipped data modules (bar reader);
    the `flows` alias only gives the shipped Features a raw.flows (raw names go unchecked)."""
    pytest.importorskip("uproot")
    pytest.importorskip("awkward")
    tmp = tmp_path_factory.mktemp("cfg_corpus")
    root = fx.build_joint_corpus(tmp, n_files_per_container=1, n_events=10).root
    groups = {**fx.SMALL_GROUPS, "flows": fx.SMALL_GROUPS["towers"]}
    reader = fx.make_reader(root, "small_r", small={"groups": groups, "cuts": fx.SMALL_CUTS})
    reader.prepare()
    reader.name = "reader"
    blocks = {k: v for k, v in cfg["data"]["modules"].items() if k != "reader"}
    data_modules = _instantiate(dict[str, Processor], blocks)
    data_modules["labels"].bind_streams(reader.streams)  # the Labels module
    return reader, data_modules


def _compile(modules, reader, mode, sinks, origins=None):
    return compile_plan({"reader": reader, **modules}, mode, {}, schema=reader.label_universe(),
                        sinks=sinks, sink_origins=origins)


def _plans(mods, pipeline, callbacks, modes=FIT_VAL, keep_accumulators=False):
    reader, data_modules = pipeline
    prep = prepare_sinks(mods, (), output_section={}, reader=reader, callbacks=callbacks,
                         modes=modes, writer_demand=False)
    full, by = {**data_modules, **prep.modules}, prep.by_mode
    sinks = {m: [*by[m].anchors, *(_accumulator_sinks(full, m) if keep_accumulators else ())]
             for m in modes}
    return prep, full, {m: _compile(full, reader, m, sinks[m], by[m].anchor_origins)
                        for m in modes}


def test_fit_val_plans_resolve_with_plotter_demand(pipeline, modules, plotter, norm_plotter):
    """Plotter demand = KinematicWeight's published keys, never a reader field: the full
    pipeline (`salt.cli._load_fit_config`, no trainer) and the runtime split both resolve."""
    (reader, data_modules), kw = pipeline, modules["kinematic_weight"]
    prep, _, plans = _plans(modules, pipeline, [plotter, norm_plotter])
    demand, model_plans = boundary_demand(modules, prep), []
    for mode in FIT_VAL:
        steps = plans[mode].steps
        names = [s.name for s in steps]
        assert (names.index("class_weight") < names.index("kinematic_prior")
                < names.index("kinematic_weight") < names.index("jets_classification"))
        produced = {k for s in steps for k in s.produces}
        assert set(plotter.fit_val_demand(modules)) <= produced
        assert "labels.jets.flavour_label" in set(norm_plotter.fit_val_demand(modules)) <= produced
        assert all("labels.jets.subcampaign" not in {*s.requires, *s.produces} for s in steps)
        ds_steps = _compile(data_modules, reader, mode, *demand[mode]).steps  # the dataset plan
        boundary = {k: s for step in ds_steps for k, s in step.produces.items()
                    if k.split(".")[0] in MODEL_VISIBLE_NAMESPACES}
        model_plans.append(compile_plan(prep.modules, mode, unflatten_spec(boundary),
                                        sinks=prep.by_mode[mode].anchors))
    kw.bind(schema := resolve_bind_schema(model_plans))
    assert schema.fields_of("inputs.container_onehot") == onehot_fields(True)  # as declared
    assert onehot_fields(True)[11:] == ("sim_type_FullSim", "sim_type_AF3")  # 13 columns
    onehot = torch.zeros(4, len(onehot_fields(True)))
    onehot[:, 3] = onehot[:, len(SUBCAMPAIGNS) + 1] = 1.0  # subcampaign code 3, shower code 1
    onehot[:, 11] = 1.0  # sim_type code 0 (FullSim)
    pt, eta = torch.tensor([25e3, 50e3, 120e3, 9e5]), torch.tensor([-2.1, -0.3, 0.4, 1.7])
    labels = {"flavour_label": torch.tensor([0, 1, 2, 3]), "pt": pt, "eta": eta}
    out = kw(Bundle({"labels": {"jets": labels}, "inputs": {"container_onehot": onehot},
                     "preds": {"jets": {"kinematic_prior": torch.zeros(4, 6)}}}), Mode.VAL)
    assert out[kw.published["subcampaign"]].tolist() == [3] * 4
    assert out[kw.published["shower"]].tolist() == [1] * 4
    assert out[kw.published["sim_type"]].tolist() == [0] * 4
    assert torch.allclose(out[kw.published["pt"]], pt * 1e-3)
    assert torch.allclose(out[kw.published["eta"]], eta.abs())
    # negative control: run 01's plotter demand of the reader-side code key
    stub = SimpleNamespace(fit_val_demand=lambda _m: ("labels.jets.subcampaign",))
    with pytest.raises(ConnectivityError, match="not in the declared schema"):
        _plans(modules, pipeline, [stub], (Mode.FIT,))


def test_group_n1_is_todays_pair(cfg, modules, pipeline, plotter, norm_plotter):
    block = cfg["model"]["init_args"]["modules"]
    grp, ia = block["kinematic_reweighting"], block["kinematic_reweighting"]["init_args"]
    assert (grp["class_path"], ia["n_iterations"]) == ("salt.model.modules.IterativeReweighting", 1)
    assert isinstance(_instantiate(SaltModelModule, grp), IterativeReweighting)
    # the explicit 16dd804 pair: what N=1 expands to, under today's names
    pair = {name: {"class_path": f"salt.model.modules.{path}", "init_args": ia[part]}
            for name, path, part in (("kinematic_prior", "tasks.ClassificationTaskModule", "prior"),
                                     ("kinematic_weight", "KinematicWeight", "weight"))}
    assert {"kinematic_reweighting", *pair} & set(modules) == set(pair)
    kw, demand = modules["kinematic_weight"], PriorClosureMetrics().fit_val_demand(modules)
    assert kw.eta == 1.0 and kw.chain == () and tuple(demand) == (kw.label_key, PRIOR_PRED)
    # C3: the pair, at the group's position, compiles the SAME FIT/VAL plans (plan hashes)
    explicit = dict(kv for key, spec in block.items()
                    for kv in (pair.items() if key == "kinematic_reweighting" else [(key, spec)]))
    cbs = [plotter, norm_plotter]
    got, want = ({m: p.plan_hash for m, p in _plans(_expanded(b), pipeline, cbs)[2].items()}
                 for b in (block, explicit))
    assert got == want


def test_n3_plans_compile(cfg, pipeline, plotter, norm_plotter):
    block = copy.deepcopy(cfg["model"]["init_args"]["modules"])
    block["kinematic_reweighting"]["init_args"]["n_iterations"] = 3
    mods = _expanded(block)
    heads = [f"kinematic_{kind}{sfx}" for sfx in ("_1", "_2", "") for kind in ("prior", "weight")]
    it1, it2 = "nn_kinematic_weight_iter_1", "nn_kinematic_weight_iter_2"
    assert set(heads) <= set(mods) and mods["kinematic_weight_2"].chain == (it1,)
    assert mods["kinematic_weight"].chain == (it2,)
    assert mods["kinematic_prior"].sample_weights == ("flavour_class_weight", it2)
    assert tuple(PriorClosureMetrics().fit_val_demand(mods)) == (
        "labels.jets.flavour_label", *(f"{PRIOR_PRED}{sfx}" for sfx in ("_1", "_2", "")),
        f"labels.jets.{it1}", f"labels.jets.{it2}")
    _, full, plans = _plans(mods, pipeline, [plotter, norm_plotter, PriorClosureMetrics()])
    for plan in plans.values():  # FIT and VAL: the heads run in iteration order
        order = [plan.module_names.index(n) for n in [*heads, "jets_classification"]]
        assert order == sorted(order)
    plan = _compile(full, pipeline[0], Mode.TEST, [PRIOR_PRED, "preds.jets.jets_classification"])
    assert "kinematic_prior" in plan.module_names  # intermediate heads + every weight dropped
    assert not {*heads[:4], "kinematic_weight"} & set(plan.module_names)
    assert_no_dead_preds(mods, plan)


def test_priors_stage_plans_skip_encoder_chain(cfg, modules, pipeline, norm_plotter):
    """Priors' plans drop the encoder chain + head but keep the accumulators; `None` restores."""
    schedule = TrainingSchedule.from_config(cfg["training_schedule"], tuple(modules))
    cbs, loss = [norm_plotter, PriorClosureMetrics()], modules["loss"]
    full = {m: p.plan_hash for m, p in _plans(modules, pipeline, cbs)[2].items()}
    loss.set_active(loss.resolve_active(schedule.stages[0].skip_losses))
    try:
        plans = _plans(modules, pipeline, cbs, keep_accumulators=True)[2]
        assert "kinematic_weight" not in _plans(modules, pipeline, cbs)[2][Mode.FIT].module_names
    finally:
        loss.set_active(None)
    chain = {"track_embed", "flow_embed", "concat", "encoder", "pool", "jets_classification"}
    for plan in plans.values():
        names = set(plan.module_names)
        assert not names & chain
        assert {"norm", "class_weight", "kinematic_prior", "kinematic_weight", "loss"} <= names
        assert all("losses.jets_classification" not in {*s.requires, *s.produces}
                   for s in plan.steps)
    assert {m: p.plan_hash for m, p in _plans(modules, pipeline, cbs)[2].items()} == full


def test_main_stage_plans_keep_prior_net_drop_its_loss(cfg, modules, pipeline, norm_plotter):
    """Main's plans still run the frozen prior net for the weights, without its CE loss."""
    schedule = TrainingSchedule.from_config(cfg["training_schedule"], tuple(modules))
    cbs, loss = [norm_plotter, PriorClosureMetrics()], modules["loss"]
    loss.set_active(loss.resolve_active(schedule.stages[1].skip_losses))
    try:
        plans = _plans(modules, pipeline, cbs, keep_accumulators=True)[2]
    finally:
        loss.set_active(None)
    for plan in plans.values():
        assert {"kinematic_prior", "kinematic_weight", "jets_classification", "loss"} <= set(
            plan.module_names)
        assert all("losses.kinematic_prior" not in s.requires for s in plan.steps)  # not summed
