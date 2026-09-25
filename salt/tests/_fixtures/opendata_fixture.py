"""Opendata-shaped dummy H5 fixture for the SHIPPED ``gn2v2-opendata.yaml``.

Unlike ``gn2v2_fixture`` (v1 ``IP3D_signed_*`` track names, used by 14 shipped
configs), the open-data config uses ``lifetimeSigned*`` names, 4 jet classes (taujets)
and an explicit ``InputSamples``. Uses only APIs present at ``f1b7664``: the study's
parity probe loads this file by path under that pinned tree.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import yaml

from salt.schema import dump_schema, save_schema
from salt.tests._fixtures.gn2v2_fixture import ELECTRON_VARIABLES, JET_VARIABLES
from salt.testing.inputs import write_dummy_file

__all__ = [
    "OPENDATA_CLASS_NAMES",
    "OPENDATA_TRACK_VARIABLES",
    "build_opendata_data",
    "opendata_overrides",
    "write_opendata_norm_dict",
]

# gn2v2-opendata.yaml data.modules.features tracks list
OPENDATA_TRACK_VARIABLES = [
    "d0",
    "z0SinTheta",
    "dphi",
    "deta",
    "qOverP",
    "lifetimeSignedD0Significance",
    "lifetimeSignedZ0SinThetaSignificance",
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
]

# gn2v2-opendata.yaml model.init_args.modules.jets_classification.init_args.class_names
OPENDATA_CLASS_NAMES = ["bjets", "cjets", "ujets", "taujets"]


def write_opendata_norm_dict(nd_path: Path, cd_path: Path) -> None:
    """Write the opendata norm/class dicts with DISTINCT per-variable constants."""
    sd = {
        stream: {
            v: {"mean": round(0.1 * (i + 1), 6), "std": round(1.0 + 0.05 * (i + 1), 6)}
            for i, v in enumerate(variables)
        }
        for stream, variables in (
            ("jets", JET_VARIABLES),
            ("tracks", OPENDATA_TRACK_VARIABLES),
            ("electrons", ELECTRON_VARIABLES),
        )
    }
    with open(nd_path, "w") as file:
        yaml.dump(sd, file, sort_keys=False)

    # the shipped config has weight_source: null, so this is written only for
    # parity with the sibling fixture (gn2v2_fixture.write_parity_norm_dict)
    cd = {
        "jets": {
            "HadronConeExclTruthLabelID": [1.0, 2.0, 2.0, 2.0],
            "flavour_label": [1.0, 2.0, 2.0, 2.0],
        },
        "tracks": {"ftagTruthOriginLabel": [4.2, 73.7, 1.0, 17.5, 12.3, 12.5, 141.7, 22.3]},
    }
    with open(cd_path, "w") as file:
        yaml.dump(cd, file, sort_keys=False)


def build_opendata_data(base: Path) -> dict[str, Path]:
    """Write the norm/class dicts, dummy H5 and schema for the shipped opendata config."""
    base.mkdir(parents=True, exist_ok=True)
    nd, cd = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_opendata_norm_dict(nd, cd)
    h5 = base / "pp_output_opendata.h5"
    # inc_taus=True is LOAD-BEARING: it writes the 4-entry flavour_label attr that
    # check_class_names needs; jets/tracks columns come from the (opendata) norm dict.
    write_dummy_file(h5, nd, inc_taus=True)
    schema = base / "schema.yaml"
    save_schema(dump_schema(h5), schema)
    return {"dir": base, "h5": h5, "nd": nd, "cd": cd, "schema": schema}


def opendata_overrides(data: Mapping[str, Path]) -> dict[str, str]:
    """Dotted-key -> value overrides (no leading ``--``) pointing the shipped
    config at the fixture; callers render them.
    """
    return {
        # the shipped InputSamples (${DATA_*_PATH} literals) makes _wire_input_samples
        # ignore train/val/test_file, so override its files; whole-dict JSON form per
        # salt/tests/integration/pipeline/test_pipeline.py:1817
        "data.modules.input_samples.init_args.files": json.dumps({
            "train": str(data["h5"]),
            "val": str(data["h5"]),
            "test": str(data["h5"]),
        }),
        # the reader needs an explicit schema (no machine paths in the shipped config)
        "data.modules.reader.init_args.schema": str(data["schema"]),
        # the norm module needs the fixture's norm dict, not a real one
        "model.modules.norm.init_args.norm_dict": str(data["nd"]),
        # shipped configs assume large training machines
        "data.num_workers": "0",
    }
