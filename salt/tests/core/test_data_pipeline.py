"""Tests for salt.core.data (M2 stage A1) — including the seed of gate G1.

G1 (plan 05): same dummy file, same slice -> the v2 pipeline's output tensors
are IDENTICAL to v1 ``SaltDataset.__getitem__``. Key-convention mapping,
asserted explicitly throughout:

    v1 inputs[s]        == v2 batch["inputs"][s]
    v1 pad_masks[s]     == v2 batch["masks"][s]      (True = padded)
    v1 labels[s][l]     == v2 batch["labels"][s][l]

Also covered: selections flowing into features+masks+labels exactly as v1,
truncation, demand-narrowed read columns, meta.rows, FIT-only transforms,
per-worker handle isolation (num_workers=2 smoke), VDS wildcard + staleness,
and the static schema-validation error paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import MappingProxyType

import h5py
import numpy as np
import pytest
import torch

from salt.core.data import (
    Features,
    GraphDataModule,
    GraphDataset,
    H5StructuredReader,
    Labels,
    MultiTarget,
    create_vds,
)
from salt.core.data.base import WorkerCtx
from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError, ConnectivityError, SchemaError
from salt.core.graph.planner import PlanStep
from salt.core.graph.spec import Mode, TensorSpec, flatten_spec
from salt.core.schema import dump_schema, save_schema
from salt.data.datasets import SaltDataset
from salt.data.transforms import GaussianNoise
from salt.utils.inputs import write_dummy_file, write_dummy_norm_dict

JET_VARS = ["pt_btagJes", "eta_btagJes"]
TRACK_VARS = ["d0", "z0SinTheta", "dphi", "deta"]
JET_LABELS = ["flavour_label"]
# ftagTruthTypeLabel carries -2/-3 sentinel codes in the dummy file — the
# int64 cast must preserve them exactly
TRACK_LABELS = ["ftagTruthOriginLabel", "ftagTruthVertexIndex", "ftagTruthTypeLabel"]

FIT_SINKS = [
    "inputs.jets",
    "inputs.tracks",
    "masks.tracks",
    *[f"labels.jets.{label}" for label in JET_LABELS],
    *[f"labels.tracks.{label}" for label in TRACK_LABELS],
]
TEST_SINKS = ["inputs.jets", "inputs.tracks", "masks.tracks", "meta.rows"]
SINKS = {Mode.FIT: FIT_SINKS, Mode.VAL: FIT_SINKS, Mode.TEST: TEST_SINKS}
N_JETS = 1000  # write_dummy_file size


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("data_pipeline")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_dummy_norm_dict(nd_path, cd_path)
    h5_path = base / "pp_output_train.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


def build_v1(data, *, selections=None, num_inputs=None, stage="fit") -> SaltDataset:
    return SaltDataset(
        filename=data["h5"],
        norm_dict=data["nd"],
        variables={"jets": list(JET_VARS), "tracks": list(TRACK_VARS)},
        stage=stage,
        labels={"jets": list(JET_LABELS), "tracks": list(TRACK_LABELS)},
        selections=selections,
        num_inputs=num_inputs,
    )


def build_modules(data, *, selections=None, truncate=None, transforms=None, with_schema=True):
    reader = H5StructuredReader(
        groups={"jets": {}, "tracks": {"truncate": truncate}},
        schema=data["schema"] if with_schema else None,
        filename=data["h5"],
        selections=selections,
        transforms=transforms,
    )
    features = Features(variables={"jets": list(JET_VARS), "tracks": list(TRACK_VARS)})
    return {"reader": reader, "features": features, "labels": Labels()}


def build_v2(data, mode=Mode.FIT, sinks=None, **kwargs) -> GraphDataset:
    return GraphDataset(build_modules(data, **kwargs), mode=mode, sinks=sinks or SINKS)


def assert_batches_match(v1_batch, v2_batch, with_labels=True):
    """The explicit v1<->v2 key-convention map of gate G1."""
    inputs, pad_masks, labels = v1_batch
    assert torch.equal(v2_batch["inputs"]["jets"], inputs["jets"])
    assert torch.equal(v2_batch["inputs"]["tracks"], inputs["tracks"])
    assert v2_batch["inputs"]["jets"].dtype == torch.float32
    assert v2_batch["inputs"]["tracks"].dtype == torch.float32
    assert torch.equal(v2_batch["masks"]["tracks"], pad_masks["tracks"])
    assert v2_batch["masks"]["tracks"].dtype == torch.bool
    assert "jets" not in v2_batch["masks"]  # vector stream: no mask, as v1
    if with_labels:
        for label in JET_LABELS:
            assert torch.equal(v2_batch["labels"]["jets"][label], labels["jets"][label])
            assert v2_batch["labels"]["jets"][label].dtype == torch.int64
        for label in TRACK_LABELS:
            assert torch.equal(v2_batch["labels"]["tracks"][label], labels["tracks"][label])
            assert v2_batch["labels"]["tracks"][label].dtype == torch.int64


class TestG1DatasetParity:
    """The G1 seed: v2 pipeline output identical to v1 SaltDataset.__getitem__."""

    @pytest.mark.parametrize("rows", [np.s_[0:100], np.s_[137:425], np.s_[900:1000]])
    def test_identity_fit(self, data, rows):
        v1 = build_v1(data)
        v2 = build_v2(data)
        assert len(v1) == len(v2) == N_JETS
        assert_batches_match(v1[rows], v2[rows])

    def test_identity_val_mode(self, data):
        # v1 builds the val dataset with stage='fit' (datamodules.py:220-226);
        # with no transforms/parameters the v2 VAL plan must read identically
        v1 = build_v1(data, stage="fit")
        v2 = build_v2(data, mode=Mode.VAL)
        assert_batches_match(v1[np.s_[0:250]], v2[np.s_[0:250]])

    def test_identity_with_selections(self, data):
        # selections flow into features, masks AND labels exactly as v1
        # (datasets.py:464-466 before :519-546; design §2.4)
        selections = {"tracks": ["d0 > 0.2", "deta < 0.8"]}
        rows = np.s_[0:300]
        v1 = build_v1(data, selections=selections)
        v2 = build_v2(data, selections=selections)
        assert_batches_match(v1[rows], v2[rows])
        # the cut must actually bite: more padded tracks than without it
        clean = build_v2(data)[rows]
        cut = v2[rows]
        assert cut["masks"]["tracks"].sum() > clean["masks"]["tracks"].sum()
        # and failing tracks' int labels were sentinel'd (-1) pre-derivation
        assert not torch.equal(
            cut["labels"]["tracks"]["ftagTruthTypeLabel"],
            clean["labels"]["tracks"]["ftagTruthTypeLabel"],
        )

    def test_truncate_matches_v1(self, data):
        rows = np.s_[50:200]
        v1 = build_v1(data, num_inputs={"tracks": 7})
        v2 = build_v2(data, truncate=7)
        v2_batch = v2[rows]
        assert v2_batch["inputs"]["tracks"].shape[1] == 7
        assert_batches_match(v1[rows], v2_batch)

    def test_mask_polarity_and_zeroing(self, data):
        rows = np.s_[0:200]
        batch = build_v2(data)[rows]
        with h5py.File(data["h5"]) as f:
            valid = f["tracks"]["valid"][rows]
        mask = batch["masks"]["tracks"]
        assert torch.equal(mask, torch.from_numpy(~valid))  # True = padded
        assert (batch["inputs"]["tracks"][mask] == 0).all()  # padded rows zeroed


class TestReader:
    def test_demand_narrowed_read_columns(self, data):
        # selection on a field NOT among the features: it must enter the read
        # set; everything undemanded must stay out (design §6.1)
        ds = build_v2(data, selections={"tracks": ["pt > 0.0"]})
        ds[np.s_[0:10]]  # trigger bind
        track_fields = set(ds.reader._buffers["tracks"].dtype.names)
        assert track_fields == set(TRACK_VARS) | set(TRACK_LABELS) | {"pt", "valid"}
        jet_fields = set(ds.reader._buffers["jets"].dtype.names)
        assert jet_fields == set(JET_VARS) | set(JET_LABELS)
        assert "mass" not in jet_fields  # no v1 read-all-jets amplification

    def test_buffer_dtype_matches_v1(self, data):
        # same read set => same get_dtype buffer (file order, as_half) as v1
        v1 = build_v1(data)
        v1[np.s_[0:1]]
        v2 = build_v2(data)
        v2[np.s_[0:1]]
        assert v2.reader._buffers["tracks"].dtype == v1.arrays["tracks"].dtype

    def test_meta_rows_in_test_mode_only(self, data):
        test_ds = build_v2(data, mode=Mode.TEST)
        batch = test_ds[np.s_[40:90]]
        assert torch.equal(batch["meta"]["rows"], torch.tensor([40, 90], dtype=torch.int64))
        assert "labels" not in batch  # no demand, no labels (design §6)
        fit_batch = build_v2(data)[np.s_[40:90]]
        assert "meta" not in fit_batch

    def test_labels_in_test_mode_when_demanded(self, data):
        sinks = dict(SINKS)
        sinks[Mode.TEST] = [*TEST_SINKS, "labels.jets.flavour_label"]
        batch = build_v2(data, mode=Mode.TEST, sinks=sinks)[np.s_[0:50]]
        with h5py.File(data["h5"]) as f:
            expected = f["jets"]["flavour_label"][0:50].astype(np.int64)
        assert torch.equal(batch["labels"]["jets"]["flavour_label"], torch.from_numpy(expected))

    def test_pid_guard_rebinds_handles(self, data, monkeypatch):
        ds = build_v2(data)
        ds[np.s_[0:10]]
        first_handle = ds.reader._h5
        real_pid = os.getpid()
        monkeypatch.setattr(os, "getpid", lambda: real_pid + 4242)
        batch = ds[np.s_[10:20]]  # simulated fork: must re-open per "process"
        assert ds.reader._h5 is not first_handle
        assert ds.reader._pid == real_pid + 4242
        monkeypatch.undo()
        with h5py.File(data["h5"]) as f:
            expected = ~f["tracks"]["valid"][10:20]
        assert torch.equal(batch["masks"]["tracks"], torch.from_numpy(expected))

    def test_num_rows_semantics(self, data):
        modules = build_modules(data)
        modules["reader"] = modules["reader"].with_source(data["h5"], num=128)
        assert len(GraphDataset(modules, mode=Mode.FIT, sinks=SINKS)) == 128
        modules2 = build_modules(data)
        modules2["reader"] = modules2["reader"].with_source(data["h5"], num=N_JETS + 1)
        with pytest.raises(ValueError, match="available"):
            len(GraphDataset(modules2, mode=Mode.FIT, sinks=SINKS))

    def test_transforms_fit_only(self, data):
        noise = GaussianNoise(noise_params={"tracks": [{"variable": "d0", "mean": 5.0, "std": 0.01}]})
        rows = np.s_[0:100]
        clean = build_v2(data, mode=Mode.VAL)[rows]
        noisy_val = build_v2(data, mode=Mode.VAL, transforms=[noise])[rows]
        # VAL: transforms are FIT-only by mode flag (design §6.1) — untouched
        assert torch.equal(noisy_val["inputs"]["tracks"], clean["inputs"]["tracks"])
        noisy_fit = build_v2(data, mode=Mode.FIT, transforms=[noise])[rows]
        mask = clean["masks"]["tracks"]
        assert not torch.equal(noisy_fit["inputs"]["tracks"][~mask], clean["inputs"]["tracks"][~mask])
        # labels and masks untouched by the input-variable noise
        assert torch.equal(noisy_fit["masks"]["tracks"], mask)


class TestStaticValidation:
    def test_feature_typo_is_schema_error(self, data):
        modules = build_modules(data)
        modules["features"] = Features(
            variables={"jets": ["pt_btagJe"], "tracks": list(TRACK_VARS)}
        )
        with pytest.raises(SchemaError, match="pt_btagJes"):
            GraphDataset(modules, mode=Mode.FIT, sinks=SINKS)

    def test_label_typo_is_connectivity_error(self, data):
        sinks = {Mode.FIT: [*TEST_SINKS[:3], "labels.jets.flavour_labelll"]}
        with pytest.raises(ConnectivityError, match="flavour_label"):
            build_v2(data, sinks=sinks)

    def test_no_schema_requires_vector_flags(self, data):
        with pytest.raises(ConfigError, match="vector"):
            H5StructuredReader(groups={"jets": {}}, schema=None, filename=data["h5"])

    def test_no_schema_explicit_flags_work_with_warning(self, data):
        reader = H5StructuredReader(
            groups={"jets": {"vector": True}, "tracks": {"vector": False}},
            schema=None,
            filename=data["h5"],
        )
        modules = {
            "reader": reader,
            "features": Features(variables={"jets": list(JET_VARS), "tracks": list(TRACK_VARS)}),
            "labels": Labels(),
        }
        with pytest.warns(UserWarning, match="checked statically"):
            ds = GraphDataset(modules, mode=Mode.FIT, sinks=SINKS)
        assert_batches_match(build_v1(data)[np.s_[0:64]], ds[np.s_[0:64]])

    def test_bind_time_field_error_without_schema(self, data):
        reader = H5StructuredReader(
            groups={"jets": {"vector": True}},
            schema=None,
            filename=data["h5"],
        )
        modules = {
            "reader": reader,
            "features": Features(variables={"jets": ["pt_btagJe"]}),
        }
        with pytest.warns(UserWarning, match="checked statically"):
            ds = GraphDataset(modules, mode=Mode.FIT, sinks={Mode.FIT: ["inputs.jets"]})
        with pytest.raises(SchemaError, match="nearest: pt_btagJes"):
            ds[np.s_[0:10]]

    def test_sinks_required(self, data):
        with pytest.raises(ConfigError, match="sinks"):
            GraphDataset(build_modules(data), mode=Mode.FIT, sinks=None)

    def test_boundary_specs(self, data):
        specs = build_v2(data).boundary_specs()
        assert set(FIT_SINKS) <= set(specs)
        assert specs["inputs.jets"].fields == tuple(JET_VARS)
        assert specs["inputs.tracks"].fields == tuple(TRACK_VARS)
        assert specs["inputs.tracks"].dtype == "float32"
        assert specs["masks.tracks"].kind == "pad_mask"
        assert specs["labels.tracks.ftagTruthOriginLabel"].kind == "label"


class TestLabelsUnit:
    @staticmethod
    def _bound_labels(**kwargs) -> Labels:
        labels = Labels(streams=["tracks"], **kwargs)
        labels.name = "labels"
        step = PlanStep(
            name="labels",
            module=labels,
            requires=MappingProxyType({}),
            produces=MappingProxyType({
                "labels.tracks.ftagTruthOriginLabel": TensorSpec(kind="label")
            }),
        )
        labels.bind(WorkerCtx(mode=Mode.FIT, read_fields={}, seed=0, step=step))
        return labels

    @staticmethod
    def _bundle(values: np.ndarray) -> Bundle:
        arr = np.zeros(len(values), dtype=[("ftagTruthOriginLabel", "i4")])
        arr["ftagTruthOriginLabel"] = values
        bundle = Bundle()
        bundle.set("raw.tracks", arr)
        return bundle

    def test_valid_range_recover(self):
        labels = self._bound_labels(
            valid_ranges={"ftagTruthOriginLabel": (-1, 7)}, recover_malformed=True
        )
        with pytest.warns(UserWarning, match="Malformed"):
            out = labels.process(self._bundle(np.array([0, 1, 99, -5])), np.s_[0:4], Mode.FIT)
        produced = out["labels.tracks.ftagTruthOriginLabel"]
        assert produced.dtype == np.int64
        np.testing.assert_array_equal(produced, [0, 1, -1, -1])

    def test_valid_range_raises_without_recover(self):
        labels = self._bound_labels(valid_ranges={"ftagTruthOriginLabel": (-1, 7)})
        with pytest.raises(ValueError, match="Malformed"):
            labels.process(self._bundle(np.array([0, 99])), np.s_[0:2], Mode.FIT)

    def test_no_range_no_check(self):
        labels = self._bound_labels()
        out = labels.process(self._bundle(np.array([0, 99, -2])), np.s_[0:3], Mode.FIT)
        np.testing.assert_array_equal(out["labels.tracks.ftagTruthOriginLabel"], [0, 99, -2])


class TestMultiTarget:
    """M5 sub-wave A3: conditional row-wise target replacement (v1 parity).

    Mirrors v1 ``apply_multi_target_replacements`` (datasets.py:695-739) +
    ``inject_custom_target_placeholders`` (datasets.py:648-693) exactly:
    ``np.where(op(sel, value), source, base)`` where ``base`` is the raw
    target column (``target:`` mode) or a NaN placeholder (``custom_target:``).
    """

    @staticmethod
    def _bundle(sel, source, raw_target=None):
        b = Bundle()
        b.set("labels.jets.flav", np.asarray(sel))
        b.set("labels.jets.src", np.asarray(source, dtype=np.float32))
        if raw_target is not None:
            arr = np.zeros(len(raw_target), dtype=[("tgt", "f4")])
            arr["tgt"] = raw_target
            b.set("raw.jets", arr)
        return b

    def test_construct_validations(self):
        with pytest.raises(ConfigError, match="at least one"):
            MultiTarget(replacements=[])
        with pytest.raises(ConfigError, match="unknown operator"):
            MultiTarget(replacements=[
                {"stream": "jets", "sel_label": "f", "op": "~=", "value": 1, "source": "s",
                 "target": "t"}
            ])
        with pytest.raises(ConfigError, match="both 'target' and 'custom_target'"):
            MultiTarget(replacements=[
                {"stream": "jets", "sel_label": "f", "op": "==", "value": 1, "source": "s",
                 "target": "t", "custom_target": "c"}
            ])
        with pytest.raises(ConfigError, match="either 'target' or 'custom_target'"):
            MultiTarget(replacements=[
                {"stream": "jets", "sel_label": "f", "op": "==", "value": 1, "source": "s"}
            ])
        with pytest.raises(ConfigError, match="mixes"):
            # same output, but one target: + one custom_target: — illegal mix
            MultiTarget(replacements=[
                {"stream": "jets", "sel_label": "f", "op": "==", "value": 1, "source": "s",
                 "target": "t"},
                {"stream": "jets", "sel_label": "g", "op": ">", "value": 0, "source": "u",
                 "custom_target": "t"},
            ])
        with pytest.raises(ConfigError, match="itself a MultiTarget output"):
            MultiTarget(replacements=[
                {"stream": "jets", "sel_label": "f", "op": "==", "value": 1, "source": "s",
                 "target": "a"},
                {"stream": "jets", "sel_label": "a", "op": ">", "value": 0, "source": "u",
                 "custom_target": "b"},
            ])

    def test_target_mode_replaces_existing(self):
        """`target:` keeps source where the condition holds, raw target otherwise."""
        mt = MultiTarget(replacements=[
            {"stream": "jets", "sel_label": "flav", "op": "==", "value": 5, "source": "src",
             "target": "tgt"}
        ])
        mt.name = "multi_target"
        sel = [5, 0, 5, 1]
        source = [10.0, 20.0, 30.0, 40.0]
        raw_target = [-1.0, -2.0, -3.0, -4.0]
        out = mt.process(self._bundle(sel, source, raw_target), np.s_[0:4], Mode.FIT)
        produced = out["labels.jets.tgt"]
        # v1 torch.where(flav==5, src, raw_tgt)
        np.testing.assert_array_equal(produced, [10.0, -2.0, 30.0, -4.0])

    def test_custom_target_mode_nan_placeholder(self):
        """`custom_target:` is NaN where the condition is false (v1 placeholder)."""
        mt = MultiTarget(replacements=[
            {"stream": "jets", "sel_label": "flav", "op": ">=", "value": 4, "source": "src",
             "custom_target": "newt"}
        ])
        mt.name = "multi_target"
        sel = [5, 0, 4, 1]
        source = [10.0, 20.0, 30.0, 40.0]
        out = mt.process(self._bundle(sel, source), np.s_[0:4], Mode.FIT)
        produced = out["labels.jets.newt"]
        # filled with source where flav>=4, NaN elsewhere (no raw column read)
        assert np.array_equal(produced[[0, 2]], [10.0, 30.0])
        assert np.isnan(produced[[1, 3]]).all()

    def test_declares_concrete_target_produce(self):
        """The output is a CONCRETE produce that beats the Labels wildcard."""
        mt = MultiTarget(replacements=[
            {"stream": "jets", "sel_label": "flav", "op": "==", "value": 5, "source": "src",
             "target": "tgt"},
            {"stream": "jets", "sel_label": "flav", "op": "<", "value": 0, "source": "src2",
             "custom_target": "newt"},
        ])
        mt.name = "multi_target"
        io = mt.declare_io(Mode.FIT)
        produces = set(flatten_spec(io.produces))
        assert produces == {"labels.jets.tgt", "labels.jets.newt"}
        requires = set(flatten_spec(io.requires))
        # sel/source come from Labels; the target raw column is read for the
        # `target:` rule only (the custom_target needs no base column)
        assert "labels.jets.flav" in requires
        assert "labels.jets.src" in requires
        assert "labels.jets.src2" in requires
        assert "raw.jets" in requires

    def test_v1_parity_against_apply_multi_target_replacements(self):
        """Bitwise parity with v1's torch.where over a labels dict."""
        import torch as _torch

        from salt.data.datasets import OPERATORS as V1_OPERATORS

        rng = np.random.default_rng(7)
        sel = rng.integers(0, 6, size=64)
        source = rng.standard_normal(64).astype(np.float32)
        raw_target = rng.standard_normal(64).astype(np.float32)
        mt = MultiTarget(replacements=[
            {"stream": "jets", "sel_label": "flav", "op": ">=", "value": 4, "source": "src",
             "target": "tgt"}
        ])
        mt.name = "multi_target"
        v2 = mt.process(self._bundle(sel, source, raw_target), np.s_[0:64], Mode.FIT)[
            "labels.jets.tgt"
        ]
        # v1 reference: torch.where(OPERATORS[op](sel, value), source, target)
        mask = V1_OPERATORS[">="](_torch.as_tensor(sel), 4)
        v1 = _torch.where(
            mask, _torch.as_tensor(source), _torch.as_tensor(raw_target)
        ).numpy()
        np.testing.assert_array_equal(v2, v1)

    def test_two_rules_one_custom_output_chain_like_v1(self):
        """Two rules writing one custom_target chain sequentially (the shipped config).

        Mirrors ``regression_multi_target.yaml``: ``ID==15 -> Pt`` then
        ``ID!=15 -> pt`` both fill ``pt_label_handle`` over a single NaN-base
        running array (v1 in-place mutation, datasets.py:709-739) — no NaN
        survives because the two conditions partition the rows.
        """
        mt = MultiTarget(replacements=[
            {"stream": "jets", "sel_label": "flav_id", "op": "==", "value": 15,
             "source": "Pt", "custom_target": "pt_label_handle"},
            {"stream": "jets", "sel_label": "flav_id", "op": "!=", "value": 15,
             "source": "pt", "custom_target": "pt_label_handle"},
        ])
        mt.name = "multi_target"
        # one rule's output, but the source differs per rule -> need both labels
        b = Bundle()
        flav = np.array([15, 5, 15, 0])
        b.set("labels.jets.flav_id", flav)
        b.set("labels.jets.Pt", np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32))
        b.set("labels.jets.pt", np.array([11.0, 22.0, 33.0, 44.0], dtype=np.float32))
        out = mt.process(b, np.s_[0:4], Mode.FIT)
        produced = out["labels.jets.pt_label_handle"]
        assert set(out) == {"labels.jets.pt_label_handle"}  # one output, two rules
        # rows 0,2 (ID==15) take Pt; rows 1,3 (ID!=15) take pt; no NaN remains
        np.testing.assert_array_equal(produced, [100.0, 22.0, 300.0, 44.0])
        assert not np.isnan(produced).any()


class TestDataModule:
    def test_loader_num_workers_matches_direct(self, data):
        dm = GraphDataModule(
            modules=build_modules(data),
            train_file=data["h5"],
            val_file=data["h5"],
            batch_size=256,
            num_workers=2,
            sinks=SINKS,
            pin_memory=False,
            persistent_workers=False,
        )
        dm.setup("fit")
        loader = dm.val_dataloader()
        assert len(loader.sampler) == 4  # 3 full + 1 partial (drop_last only for fit)
        batches = list(loader)
        assert len(batches) == 4
        seen = 0
        for i, batch in enumerate(batches):
            rows = np.s_[i * 256 : min((i + 1) * 256, N_JETS)]
            expected = dm.val_dset[rows]
            assert torch.equal(batch["inputs"]["tracks"], expected["inputs"]["tracks"])
            assert torch.equal(batch["masks"]["tracks"], expected["masks"]["tracks"])
            assert torch.equal(
                batch["labels"]["jets"]["flavour_label"],
                expected["labels"]["jets"]["flavour_label"],
            )
            seen += batch["inputs"]["jets"].shape[0]
        assert seen == N_JETS

    def test_train_loader_drops_last(self, data):
        dm = GraphDataModule(
            modules=build_modules(data),
            train_file=data["h5"],
            val_file=data["h5"],
            batch_size=256,
            sinks=SINKS,
            pin_memory=False,
        )
        dm.setup("fit")
        assert len(dm.train_dataloader().sampler) == 3  # drop_last for fit
        assert dm.train_dset.plan.mode == Mode.FIT
        assert dm.val_dset.plan.mode == Mode.VAL

    def test_sinks_required_at_setup(self, data):
        dm = GraphDataModule(
            modules=build_modules(data), train_file=data["h5"], val_file=data["h5"]
        )
        with pytest.raises(ConfigError, match="sinks"):
            dm.setup("fit")


class TestVDS:
    @pytest.fixture(scope="class")
    def vds_data(self, tmp_path_factory):
        base = tmp_path_factory.mktemp("vds")
        nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
        write_dummy_norm_dict(nd_path, cd_path)
        members = [base / "pp_output_train_split1.h5", base / "pp_output_train_split2.h5"]
        for member in members:
            write_dummy_file(member, nd_path)
        schema_path = base / "schema.yaml"
        save_schema(dump_schema(members[0]), schema_path)
        return {"dir": base, "members": members, "schema": schema_path, "nd": nd_path}

    def test_wildcard_reader_via_vds(self, vds_data):
        reader = H5StructuredReader(
            groups={"jets": {}, "tracks": {}},
            schema=vds_data["schema"],
            filename=vds_data["dir"] / "pp_output_train_split*.h5",
            vds_path=vds_data["dir"] / "vds_out" / "vds.h5",
        )
        modules = {
            "reader": reader,
            "features": Features(variables={"jets": list(JET_VARS), "tracks": list(TRACK_VARS)}),
            "labels": Labels(),
        }
        ds = GraphDataset(modules, mode=Mode.FIT, sinks=SINKS)
        assert len(ds) == 2 * N_JETS
        batch = ds[np.s_[N_JETS - 5 : N_JETS + 5]]  # crosses the member boundary
        with h5py.File(vds_data["members"][0]) as f1, h5py.File(vds_data["members"][1]) as f2:
            expected = np.concatenate([
                f1["jets"]["pt_btagJes"][-5:],
                f2["jets"]["pt_btagJes"][:5],
            ]).astype(np.float32)
        np.testing.assert_array_equal(batch["inputs"]["jets"][:, 0].numpy(), expected)

    def test_vds_staleness_rebuild(self, vds_data):
        pattern = vds_data["dir"] / "pp_output_train_split*.h5"
        out = vds_data["dir"] / "vds_stale" / "vds.h5"
        first = create_vds(pattern, out)
        first_mtime = first.stat().st_mtime
        # reuse while fresh
        assert create_vds(pattern, out).stat().st_mtime == first_mtime
        # a member newer than the VDS forces a rebuild (design §6.1 staleness fix)
        future = first_mtime + 100
        os.utime(vds_data["members"][1], (future, future))
        rebuilt = create_vds(pattern, out)
        assert rebuilt.stat().st_mtime > first_mtime


class TestSinkOriginAttribution:
    """Stage-E fix: dataset-plan errors name the model-side demander (§4.1)."""

    ORIGIN = "'jets_classification' (config: model.modules.jets_classification)"

    def make_modules(self, data):
        return {
            "reader": H5StructuredReader(
                groups={"jets": {}, "tracks": {}}, schema=dump_schema(data["h5"])
            ),
            "features": Features(variables={"jets": JET_VARS, "tracks": TRACK_VARS}),
            "labels": Labels(),
        }

    def test_label_typo_error_names_the_task_module(self, data):
        bad_key = "labels.jets.flavor_label"  # typo'd label, schema-validated
        with pytest.raises(ConnectivityError) as excinfo:
            GraphDataset(
                self.make_modules(data),
                mode=Mode.FIT,
                sinks=[*FIT_SINKS, bad_key],
                sink_origins={bad_key: self.ORIGIN},
            )
        message = str(excinfo.value)
        assert self.ORIGIN in message  # the demanding module + config address
        assert "<sinks>" not in message  # planner placeholder never leaks
        assert "flavour_label" in message  # nearest-key suggestion intact

    def test_reader_bind_error_carries_origin(self, data, tmp_path):
        # no schema artifact -> the typo survives to worker bind, where the
        # reader must still attribute the demand to the task module
        modules = self.make_modules(data)
        modules["reader"] = H5StructuredReader(
            groups={"jets": {"vector": True}, "tracks": {"vector": False}},
            filename=data["h5"],
        )
        bad_key = "labels.jets.flavor_label"
        dset = GraphDataset(
            modules,
            mode=Mode.FIT,
            sinks=[*FIT_SINKS, bad_key],
            sink_origins={bad_key: self.ORIGIN},
        )
        with pytest.raises(SchemaError, match="jets_classification") as excinfo:
            dset[np.s_[0:10]]
        assert "nearest: flavour_label" in str(excinfo.value)
