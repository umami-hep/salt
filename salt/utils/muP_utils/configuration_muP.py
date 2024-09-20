"""Helper functions for muP."""

from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml
from torch import load

from salt.utils.muP_utils.functions_check_muP import (
    get_coord_data,
    get_training_data,
    plot_coord_data,
    plot_training_data,
)

# Paths are hard-coded as the lifetime of these files should last 1 execution
folder_path = "./temp_muP/"
MODEL_MAIN_PATH = "main_model.pt"
MODEL_BASE_PATH = "base_model.pt"
MODEL_DELTA_PATH = "delta_model.pt"
MODEL_BASE_CHECK_PATH = "base_check_model.pt"
MODEL_DELTA_CHECK_PATH = "delta_check_model.pt"
SHAPE_PATH = "storeshapes"
SHAPE_CHECK_PATH = "storeshapes_check"


def get_paths(store_path=None):
    if store_path is None:
        store_path = folder_path
    folder = Path(store_path)
    folder.mkdir(exist_ok=True)
    shape_path = Path(store_path) / SHAPE_PATH
    shape_check_path = Path(store_path) / SHAPE_CHECK_PATH
    model_main_path = Path(store_path) / MODEL_MAIN_PATH
    model_base_path = Path(store_path) / MODEL_BASE_PATH
    model_delta_path = Path(store_path) / MODEL_DELTA_PATH
    model_base_check_path = Path(store_path) / MODEL_BASE_CHECK_PATH
    model_delta_check_path = Path(store_path) / MODEL_DELTA_CHECK_PATH
    return {
        "folder": folder,
        "store_path": store_path,
        "shape_path": shape_path,
        "shape_check_path": shape_check_path,
        "model_main_path": model_main_path,
        "model_base_path": model_base_path,
        "model_delta_path": model_delta_path,
        "model_base_check_path": model_base_check_path,
        "model_delta_check_path": model_delta_check_path,
    }


def recursive_set_val(dictionary, keyTarget, valueSet):
    for key, value in dictionary.items():
        if key == keyTarget:
            return True, key, valueSet
        if isinstance(value, dict):
            found, subkey, subvalue = recursive_set_val(value, keyTarget, valueSet)
            if found:
                value[subkey] = subvalue
                return True, key, value
    return False, None, None


def set_val_nestedKey(dictionary, keyTarget, valueSet):
    """Returns the updated nested dictionary with the (sub-)key keyTarget updated to value valueSet
    Recursively iterates over the keys of nested dictionaries until the keyTarget is found.

    Note: dictionary can be a list of dictionary (useful for init_nets)
    """
    adapt_to_list, found = False, False
    if isinstance(dictionary, list):
        # Some logic to detect if it's a list of dictionary
        adapt_to_list = True
        dictionary = {f"entry_{i}": dictionary[i] for i in range(len(dictionary))}

    # Core step: the recursive search
    for key, value in dictionary.items():
        if key == keyTarget:
            # Key is on the top level of dictionary, no need for recursiveness
            changeKey, changeVal = key, valueSet
            break
        if isinstance(value, dict):
            found, subkey, subvalue = recursive_set_val(value, keyTarget, valueSet)
            if found:
                value[subkey] = subvalue
                changeKey, changeVal = key, value

    if found:
        dictionary[changeKey] = changeVal
        # Further processing if started with list of dictionaries
        if adapt_to_list:
            dictionary = list(dictionary.values())
        return dictionary
    raise KeyError(f"Key {keyTarget} not found in the (nested) dictionary {dictionary}")


def update_config(muP_config, cfg_out, parameter_val):
    for param_config in muP_config.values():
        assert len(param_config["apply_to"]) == len(
            param_config["parameter_name"]
        ), "Define one parameter name per module to apply to."
        param_dic = zip(
            param_config["apply_to"],
            param_config["parameter_name"],
            [param_config[parameter_val]] * len(param_config["apply_to"]),
            strict=True,
        )
        for model_adapt, paramter_adapt, val in param_dic:
            assert (
                model_adapt in cfg_out["model"]["model"]["init_args"]
            ), f"Model key {model_adapt} not found in init_args"
            cfg_out["model"]["model"]["init_args"][model_adapt] = set_val_nestedKey(
                cfg_out["model"]["model"]["init_args"][model_adapt], paramter_adapt, val
            )
    return cfg_out


def generate_base_delta_config(main_config, mod_type):
    """mod_type is either base or delta."""
    with open(main_config) as file:
        cfg = yaml.safe_load(file)
    cfg_out = deepcopy(cfg)

    assert "muP_config" in cfg["model"], "To use the muP, you need to define it in the model config"
    muP_config = cfg["model"]["muP_config"]

    path_dict = get_paths()
    cfg_out_path = (
        Path(path_dict["store_path"]) / "modbase.yaml"
        if mod_type == "base"
        else Path(path_dict["store_path"]) / "./moddelta.yaml"
    )
    parameter_val = "parameter_base" if mod_type == "base" else "parameter_delta"

    shape_path = None
    if "shape_path" in muP_config:
        shape_path = muP_config.pop("shape_path")

    cfg_out = update_config(muP_config, cfg_out, parameter_val)

    # clean out the muP config
    cfg_out["model"]["muP_config"] = {}
    with open(cfg_out_path, "w") as file:
        yaml.dump(cfg_out, file, sort_keys=False)
    return cfg_out_path, shape_path


def generate_config_muPtest(main_config, variations, muP=True):
    """Creates configs with varying dimensions in variations."""
    with open(main_config) as file:
        cfg = yaml.safe_load(file)

    configs = []
    path_dict = get_paths()
    for i, var in enumerate(variations):
        cfg_out = deepcopy(cfg)
        muP_config = {
            "embed_dim": {
                "apply_to": ["init_nets", "encoder"],
                "parameter_name": ["output_size", "embed_dim"],
                "parameter": var,
            }
        }
        cfg_out = update_config(muP_config, cfg_out, "parameter")
        if not muP:
            cfg_out["model"].pop("muP_config")
            remove_muP = {
                "muP": {
                    "apply_to": ["encoder"],
                    "parameter_name": ["muP"],
                    "parameter": False,
                }
            }
            cfg_out = update_config(remove_muP, cfg_out, "parameter")

        cfg_out_path = (
            Path(path_dict["store_path"]) / f"config_muP_{i}.yaml"
            if muP
            else Path(path_dict["store_path"]) / f"config_sP_{i}.yaml"
        )
        configs.append(cfg_out_path)
        with open(cfg_out_path, "w") as file:
            yaml.dump(cfg_out, file, sort_keys=False)
    return configs


def get_model_path(mod_type, path_dict=None):
    if path_dict is None:
        path_dict = get_paths()
    if "temp_" in mod_type:
        return (
            path_dict["model_base_check_path"]
            if "base" in mod_type
            else path_dict["model_delta_check_path"]
            if "delta" in mod_type
            else Path(path_dict["store_path"]) / f"./{mod_type}.pt"
        )
    return (
        path_dict["model_base_path"]
        if "base" in mod_type
        else path_dict["model_delta_path"]
        if "delta" in mod_type
        else path_dict["model_main_path"]
    )


def get_models_muPtest(variations, modInd=None, modType=None):
    models_muP, models_sP = {}, {}
    path_dict = get_paths()
    if modInd is not None:
        if modType is not None:
            if modType:
                models_muP[variations[modInd]] = (
                    Path(path_dict["store_path"]) / f"temp_muP_{modInd}.pt"
                )
            else:
                models_sP[variations[modInd]] = (
                    Path(path_dict["store_path"]) / f"temp_sP_{modInd}.pt"
                )
        else:
            models_muP[variations[modInd]] = Path(path_dict["store_path"]) / f"temp_muP_{modInd}.pt"
            models_sP[variations[modInd]] = Path(path_dict["store_path"]) / f"temp_sP_{modInd}.pt"
        return models_muP, models_sP

    for i, var in enumerate(variations):
        models_muP[var] = Path(path_dict["store_path"]) / f"temp_muP_{i}.pt"
        models_sP[var] = Path(path_dict["store_path"]) / f"temp_sP_{i}.pt"
    return models_muP, models_sP


def load_models(check=False):
    path_dict = get_paths()
    if check:
        return load(path_dict["model_base_check_path"]), load(path_dict["model_delta_check_path"])
    return load(path_dict["model_base_path"]), load(path_dict["model_delta_path"])


def store_shapes_mup(path=None, check=False):
    from mup import make_base_shapes

    path_dict = get_paths(store_path=path)
    store_at = path_dict["shape_path"] if not check else path_dict["shape_check_path"]
    base, delta = load_models(check)
    make_base_shapes(base, delta, store_at)


def instantiate_mup(model, load_from=None, check=False):
    from mup import set_base_shapes

    rescale = not (check)
    path_dict = get_paths(store_path=load_from)
    load_from = str(path_dict["shape_path"]) if not check else str(path_dict["shape_check_path"])
    set_base_shapes(model, load_from, rescale_params=rescale)


def clean_environment():
    path_dict = get_paths()
    path_dict["folder"].unlink()


def coord_check(
    muP,
    models,
    dataloaders,
    optimiser,
    save_path,
    lr=1e-3,
    nsteps=20,
    nseeds=2,
    name_contains=None,
    legend=False,
    mod="GN2",
    save_df=True,
    doPlots=True,
):
    """Coord check for muP. Advised to use a large LR.
    Based on the muP GitHub repo.
    """
    df = get_coord_data(
        models,
        dataloaders,
        mup=muP,
        optimizer=optimiser,
        fix_data=True,
        lr=lr,
        nseeds=nseeds,
        nsteps=nsteps,
        cuda=False,
    )
    if save_df:
        df.to_csv(save_path / "mup.csv" if muP else save_path / "sp.csv")

    if doPlots:
        prm = "μP" if muP else "SP"
        return plot_coord_data(
            df,
            legend=legend,
            name_contains=name_contains,
            save_to=save_path / "mup.png" if muP else save_path / "sp.png",
            suptitle=f"{prm} {mod} with {optimiser} lr={lr} nseeds={nseeds}",
            face_color="xkcd:light grey" if not muP else None,
        )
    return None


def training_check(
    muP,
    models,
    dataloaders,
    optimiser,
    lr,
    nsteps,
    nseeds,
    save_path,
    legend="full",
    mod="GN2",
    save_df=True,
    save_model=None,
    doPlots=True,
    noTQDM=True,
    load_data=None,
):
    """Coord check for muP. Advised to use a large LR.
    Based on the muP GitHub repo.
    """
    if load_data is None:
        df = get_training_data(
            models,
            dataloaders,
            mup=muP,
            optimizer=optimiser,
            lr=lr,
            nseeds=nseeds,
            nsteps=nsteps,
            save_model=save_model,
            cuda=False,
            noTQDM=noTQDM,
        )
        if save_df:
            if save_model is not None:
                width = list(models.keys())[0]
                df.to_csv(save_model / f"width_{width}.csv")
            else:
                df.to_csv(save_path / "mup.csv" if muP else save_path / "sp.csv")
    else:
        df = pd.read_csv(load_data)
    if not doPlots:
        return None

    prm = "μP" if muP else "SP"
    return plot_training_data(
        df,
        legend=legend,
        save_to=save_path / "mup_train.png" if muP else save_path / "sp_train.png",
        suptitle=f"{prm} {mod} with {optimiser} lr={lr} nseeds={nseeds}",
        face_color="xkcd:light grey" if not muP else None,
    )


def check_muP(
    models_muP: dict,
    models_sP: dict,
    lr: float = 1e-2,
    nsteps_training: int = 1500,
    nsteps_coocheck: int = 3,
    nseeds: int = 5,
    batch_size: int = 2000,
    num_workers: int = 10,
    num_train: int = -1,
    train_file: str = "./pp_output_train.h5",
    norm_dict: str = "./norm_dict.yaml",
    save_path: str = "./plots_mu",
    doCoord: bool = True,
    doCoordPlots: bool = True,
    doTraining: bool = True,
    doTrainingPlots: bool = True,
):
    """Script to check muP is well set.

    Parameters
    ----------
    models_muP : dict,
        Dictionary of muP models
        Format: {width: path to model}
    models_sP: dict,
        Dictionary of sP models
        Format: {width: path to model}
    lr : float, default: 1e-2,
        learning rate to use for the tests
    nsteps_training : int, default: 1500,
        nsteps of training check
    nsteps_coocheck : int, default: 3,
        nsteps of coordinate check
    nseeds : int, default: 5,
        number of seeds to use
    batch_size : int, default: 2000,
        mini batch size to use
    num_workers : int, default: 10,
        number of parallel dataloaders to use
    num_train : int, default: -1,
        number of data points to use (-1: all)
    train_file : str, default: "./pp_output_train.h
        path to the training file
    norm_dict : str, default: "./norm_dict.yaml",
        path to the normalisation dictionary
    save_path : str, default: "./plots_mu",
        path to save the results
    doCoord : bool, default: True,
        whether to do the coordinate check
    doCoordPlots : bool, default: True,
        whether to plot the coordinate check results
    doTraining : bool, default: True,
        whether to do the training check
    doTrainingPlots : bool, default: True,
        whether to plot the training check results
    """
    from salt.data.datamodules import SaltDataModule, SaltDataset

    datamodule = SaltDataModule(
        train_file=train_file,
        val_file="",
        batch_size=batch_size,
        num_workers=num_workers,
        num_train=num_train,
        num_val=0,
        num_test=0,
        stage="fit",
    )
    variables = {
        "jets": ["pt_btagJes", "eta_btagJes"],
        "tracks": [
            "d0",
            "z0SinTheta",
            "dphi",
            "deta",
            "qOverP",
            "IP3D_signed_d0_significance",
            "IP3D_signed_z0_significance",
            "phiUncertainty",
            "thetaUncertainty",
            "qOverPUncertainty",
            "numberOfPixelHits",
            "numberOfSCTHits",
            "numberOfInnermostPixelLayerHits",
            "numberOfNextToInnermostPixelLayerHits",
            "numberOfInnermostPixelLayerSharedHits",
            "numberOfInnermostPixelLayerSplitHits",
            "numberOfPixelSharedHits",
            "numberOfPixelSplitHits",
            "numberOfSCTSharedHits",
        ],
    }
    datamodule.train_dset = SaltDataset(
        filename=datamodule.train_file,
        num=datamodule.num_train,
        stage="fit",
        norm_dict=norm_dict,
        variables=variables,
        labels={"jets": ["flavour_label"]},
    )
    dataloader = datamodule.get_dataloader(dataset=datamodule.train_dset, stage="fit", shuffle=True)
    plot_dir = Path(save_path)
    plot_dir.mkdir(exist_ok=True)
    if doCoord:
        coord_check(
            True,
            models_muP,
            dataloader,
            "adamw",
            plot_dir,
            lr=lr,
            nsteps=nsteps_coocheck,
            nseeds=nseeds,
            name_contains=["init_nets", "encoder"],
            doPlots=doCoordPlots,
        )
        coord_check(
            False,
            models_sP,
            dataloader,
            "adamw",
            plot_dir,
            lr=lr,
            nsteps=nsteps_coocheck,
            nseeds=nseeds,
            name_contains=["init_nets", "encoder"],
            doPlots=doCoordPlots,
        )
    if doTraining and models_muP:
        save_muP_models = plot_dir / "trained_muP_models"
        training_check(
            True,
            models_muP,
            dataloader,
            "adamw",
            lr=lr,
            nsteps=nsteps_training,
            nseeds=nseeds,
            save_path=plot_dir,
            save_model=save_muP_models,
            doPlots=doTrainingPlots,
        )
    if doTraining and models_sP:
        save_sP_models = plot_dir / "trained_sP_models"
        training_check(
            False,
            models_sP,
            dataloader,
            "adamw",
            lr=lr,
            nsteps=nsteps_training,
            nseeds=nseeds,
            save_path=plot_dir,
            save_model=save_sP_models,
            doPlots=doTrainingPlots,
        )
