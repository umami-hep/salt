"""Shared fixtures/helpers/toy-writers for the salt/core/writers unit + integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from salt.core.graph.bundle import Bundle
from salt.core.graph.spec import Mode, TensorSpec
from salt.core.main import CONFIG_DIR
from salt.core.nn import bind_all, materialise_all, resolve_bind_schema
from salt.core.schema import dump_schema, save_schema
from salt.core.outputs.writer_base import WriteCtx, Writer, WriterDeclareCtx
from salt.core.testing.inputs import write_dummy_file
from salt.tests._fixtures.gn2v2_fixture import (
    ORIGIN_CLASSES,
    build_gn2v2_modules,
    compile_gn2v2,
    write_parity_norm_dict,
)

DUMMY_CFG = CONFIG_DIR / "gn2v2-dummy.yaml"
RUN_NAME = "GN2v2_dummy"  # the dummy config's `name:`
N_JETS, L_FILE = 1000, 40  # write_dummy_file geometry
PRED_KEYS = [
    "preds.jets.jets_classification",
    "preds.tracks.track_origin",
    "preds.tracks.track_vertexing",
]
JET_PROB_COLS = ["salt_pb", "salt_pc", "salt_pu"]
ORIGIN_PROB_COLS = [f"salt_p{c}" for c in ORIGIN_CLASSES]


class JetCountWriter(Writer):
    """The design §8 custom-writer journey: one new column, four YAML lines."""

    def requires(self, ctx):
        del ctx
        return {"masks.tracks": TensorSpec(shape=None, dtype="bool", kind="pad_mask")}

    def columns(self, ctx):
        del ctx
        return {"jets": np.dtype([("n_valid_tracks", "i4")])}

    def write(self, bundle, rows):
        del rows
        mask = bundle.get("masks.tracks").cpu().numpy()
        counts = (~mask).sum(-1, keepdims=True).astype("i4")
        return {"jets": u2s(counts, np.dtype([("n_valid_tracks", "i4")]))}


class NamedFeatureWriter(Writer):
    """Resolves a feature column BY NAME via ``ctx.feature_fields`` (M3-review fix)."""

    COLUMN = "eta_by_name"
    DTYPE = np.dtype([("eta_by_name", "f4")])

    def requires(self, ctx):
        del ctx
        return {"inputs.jets": TensorSpec(dtype="float32", kind="data")}

    def columns(self, ctx):
        assert "inputs.jets" in ctx.feature_fields  # the bind-time field names
        return {"jets": self.DTYPE}

    def write(self, bundle, rows):
        del rows
        # no index arithmetic off the config — resolve the column by name
        idx = self.ctx.feature_fields["inputs.jets"].index("eta_btagJes")
        vals = bundle.get("inputs.jets").cpu().numpy()[:, idx : idx + 1].astype("f4")
        return {"jets": u2s(vals, self.DTYPE)}


class WrongDtypeMaskWriter(Writer):
    """A writer that DECLARES the wrong dtype for a consumed key (negative control)."""

    DTYPE = np.dtype([("wrong_dtype_probe", "i4")])

    def requires(self, ctx):
        del ctx
        return {"masks.tracks": TensorSpec(shape=None, dtype="float32", kind="pad_mask")}

    def columns(self, ctx):
        del ctx
        return {"jets": self.DTYPE}

    def write(self, bundle, rows):
        del rows
        mask = bundle.get("masks.tracks").cpu().numpy()
        return {"jets": u2s((~mask).sum(-1, keepdims=True).astype("i4"), self.DTYPE)}


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("m3_writers")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    # exactly four underscore parts -> sample heuristic yields 'ttbar' (PW:169)
    h5_path = base / "pp_output_test_ttbar.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


@pytest.fixture(scope="module")
def modules(data):
    # config-only module dict: requires/columns need attrs, not bound layers
    return build_gn2v2_modules(data["nd"])


@pytest.fixture(scope="module")
def bound_modules(data):
    # bound + materialised module dict: needed wherever get_h5 runs the eval
    # conversion (since the P1.5 flip ClassificationTaskModule.get_h5 calls
    # self.task.run_inference, which exists only after bind). The TEST plan is
    # compiled, the schema resolved and the heads built so the softmax op-chain
    # runs on the synthetic raw-logit bundle exactly as on the live oracle path.
    modules = build_gn2v2_modules(data["nd"])
    bind_all(modules, resolve_bind_schema(compile_gn2v2(modules, Mode.TEST)))
    materialise_all(modules)
    return modules


def declare_ctx(modules) -> WriterDeclareCtx:
    return WriterDeclareCtx(
        model_modules=modules, streams=("jets", "tracks"), sequence_streams=("tracks",)
    )


def write_ctx(data, modules, out: Path | None = None, run_name: str = "salt") -> WriteCtx:
    return WriteCtx(
        output_path=out or (data["dir"] / "unit_out.h5"),
        total=N_JETS,
        run_name=run_name,
        source_path=data["h5"],
        streams=("jets", "tracks"),
        sequence_streams=("tracks",),
        group_datasets={"jets": "jets", "tracks": "tracks"},
        seq_lengths={"tracks": L_FILE},
        model_modules=modules,
        batch_size=100,
    )


def make_preds_bundle(b: int = 5, length: int = L_FILE, seed: int = 3) -> Bundle:
    # Since the P1.5 flip the classification task forward publishes RAW logits in
    # TEST and the eval softmax moved INTO get_h5 (it run_inference-s the raw
    # leaf before packing), so this synthetic TEST bundle feeds RAW logits (NOT
    # pre-softmaxed) for the classification heads — feeding softmaxed values here
    # would double-convert. The masked-softmax track_origin get_h5 reads
    # masks.tracks, so the bundle carries it (True = padded; the last two
    # positions are padded).
    #
    # Plan 34 W34.3 flips vertexing too: its TEST forward now publishes the RAW
    # [E, 1] edge scores (the union-find moved INTO get_h5 / get_output). So this
    # bundle feeds RAW edge scores for the COMPRESSED valid-node graph — one
    # directed edge score per ordered pair of valid (non-padded) tracks per row
    # (E = b * n_valid * (n_valid - 1)); get_h5 union-finds them into the [B, L]
    # per-node assignment column (padded rows -> -inf -> int32 -2147483648).
    gen = torch.Generator().manual_seed(seed)
    jets = torch.randn(b, 3, generator=gen)
    origin = torch.randn(b, length, 8, generator=gen)
    mask = torch.zeros(b, length, dtype=torch.bool)
    mask[:, -2:] = True  # padded track positions (True = padded)
    n_valid = length - 2
    n_edges = b * n_valid * (n_valid - 1)
    vertex = torch.randn(n_edges, 1, generator=gen)  # RAW [E, 1] edge scores
    return Bundle({
        "preds": {
            "jets": {"jets_classification": jets},
            "tracks": {"track_origin": origin, "track_vertexing": vertex},
        },
        "masks": {"tracks": mask},
    })
