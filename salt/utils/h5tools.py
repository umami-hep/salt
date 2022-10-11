import h5py
import numpy as np
import pandas as pd


def get_fields(h5_path, group, variables, num=None):
    """Return fields from h5 file."""
    with h5py.File(h5_path) as f:
        num = int(num) if num is not None else None
        return f[group].fields(variables)[:num]


def get_jet_df(h5_path, variables, group="jets", num_jets=None):
    """Load jet variables from input h5 to DataFrame."""

    data = get_fields(h5_path, group, variables, num_jets)
    return pd.DataFrame(data)


def get_track_df(
    h5_path, track_variables, group="tracks", jet_variables=None, num_jets=None
):
    """Load flattened, valid tracks from input h5 to DataFrame.

    Optionally add repeated jet information for each track.
    """

    # get track df
    valid = get_fields(h5_path, group, "valid", num_jets)
    track_data = get_fields(h5_path, group, track_variables, num_jets)
    df = pd.DataFrame(track_data[valid])

    # add jet info
    if jet_variables:
        num_tracks_per_jet = valid.sum(axis=-1)
        for jet_var in jet_variables:
            jet_data = get_fields(h5_path, "jets", jet_var, num_jets)
            df[f"jet_{jet_var}"] = np.repeat(jet_data, num_tracks_per_jet)

    return df


def get_num_tracks_per_jet(h5_path, group="tracks", num_jets=None):
    """Returns the number tracks per jet.

    This may also be already provided by a jet variable.
    """

    valid = get_fields(h5_path, group, "valid", num_jets)
    return valid.sum(axis=-1)


def standardise_column_names(df):
    """Standardise column names for downstream functions.

    Ensure each dataframe only contains predictions from a single model.
    """

    n_models = [x for x in df.columns if "_pb" in x]
    assert len(n_models) <= 1, "You should load only one model per DataFrame."

    def rename_func(x):
        if x.endswith("_pb"):
            return "p_b"
        elif x.endswith("_pc"):
            return "p_c"
        elif x.endswith("_pu"):
            return "p_u"
        elif x == "HadronConeExclTruthLabelID":
            return "flavour"
        elif x == "n_tracks_loose":
            return "n_tracks"
        elif x == "nPromptLeptons":
            return "n_truth_promptLepton"
        else:
            return x

    return df.rename(rename_func, axis="columns")


def load_jets(
    h5_path,
    variables,
    group="jets",
    num_jets=None,
    standardise_columns=True,
    rescale_pt=False,
    remove_prompt_leptons=True,
):
    """Wrapper function to load jet DataFrames."""

    # load jets
    df = get_jet_df(h5_path, variables, group, num_jets)

    # rename columns
    if standardise_columns:
        df = standardise_column_names(df)

    # convert pt columns from MeV -> GeV
    if rescale_pt:
        for col in ["pt", "jet_pt", "pt_btagJes"]:
            if col in df.columns:
                df[col] *= 0.001

    # remove electron jets (for Z' Ext sample)
    if remove_prompt_leptons:
        with h5py.File(h5_path) as f:
            if "nPromptLeptons" in f["jets"].dtype.names:
                n_prompt_lepton = f["jets"]["nPromptLeptons"][:num_jets]
            elif "n_truth_promptLepton" in f["jets"].dtype.names:
                n_prompt_lepton = f["jets"]["n_truth_promptLepton"][:num_jets]
        df = df[n_prompt_lepton == 0]

    return df
