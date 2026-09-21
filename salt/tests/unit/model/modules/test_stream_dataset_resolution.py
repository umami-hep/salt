"""Stream→dataset resolution of file-keyed dict lookups (norm dict, class dict)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from salt.data import Features, H5StructuredReader, Labels, SaltDataModule
from salt.graph.errors import ConfigError
from salt.graph.spec import Mode
from salt.model.bind import ResolvedSchema, reader_stream_datasets, resolve_bind_schema
from salt.model.modules.losses import LossSum
from salt.model.modules.norm import Normaliser
from salt.model.modules.tasks import ClassificationTaskModule
from salt.model.saltmodule import SaltModule
from salt.schema import dump_schema, save_schema
from salt.testing.inputs import write_dummy_file
from salt.tests._fixtures.gn2v2_fixture import (
    JET_VARIABLES,
    ORIGIN_CLASSES,
    TRACK_VARIABLES,
    build_gn2v2_modules,
    compile_gn2v2,
    write_parity_norm_dict,
)
from salt.tests.unit.model.test_saltmodule import make_trainer

LRS = {"initial": 1e-3, "max": 5e-3, "end": 1e-4, "pct_start": 0.1}


# -- fixtures / helpers -------------------------------------------------------


@pytest.fixture
def parity_dicts(tmp_path: Path) -> tuple[Path, Path]:
    """Fresh parity norm_dict.yaml (jets/tracks/electrons) + class_dict.yaml (jets/tracks)."""
    nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
    write_parity_norm_dict(nd, cd)
    return nd, cd


def rekey(path: Path, old: str, new: str) -> None:
    """Rename a top-level YAML key in place (builds a dataset-keyed dict fixture)."""
    payload = yaml.safe_load(path.read_text())
    payload[new] = payload.pop(old)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def make_norm(nd: Path, streams: tuple[str, ...], global_object: str) -> Normaliser:
    """A named `Normaliser` over `streams` reading `nd`."""
    norm = Normaliser(norm_dict=nd, streams=list(streams), global_object=global_object)
    norm.name = "norm"
    return norm


def _vars_of(stream: str) -> list[str]:
    return JET_VARIABLES if stream in ("jets", "global") else TRACK_VARIABLES


def schema_for(streams: tuple[str, ...], datasets: dict[str, str]) -> ResolvedSchema:
    """A `ResolvedSchema` over `inputs.<stream>` for `streams`, carrying `datasets`."""
    widths = {f"inputs.{s}": len(_vars_of(s)) for s in streams}
    fields = {f"inputs.{s}": tuple(_vars_of(s)) for s in streams}
    return ResolvedSchema(widths=widths, fields=fields, datasets=datasets)


def expected_means(nd: Path, key: str, variables: list[str]) -> torch.Tensor:
    """The `mean` values for `key` in the norm dict at `nd`, in `variables` order."""
    d = yaml.safe_load(nd.read_text())
    return torch.tensor([float(d[key][v]["mean"]) for v in variables])


def expected_stds(nd: Path, key: str, variables: list[str]) -> torch.Tensor:
    """The `std` values for `key` in the norm dict at `nd`, in `variables` order."""
    d = yaml.safe_load(nd.read_text())
    return torch.tensor([float(d[key][v]["std"]) for v in variables])


# -- Normaliser ----------------------------------------------------------------


class TestNormaliserStreamDatasetResolution:
    def test_n1_tracks_remapped_to_tracks_ghost_resolves(self, parity_dicts):
        """N1: stream 'tracks' remapped to dataset 'tracks_ghost' resolves the norm dict."""
        nd, _cd = parity_dicts
        rekey(nd, "tracks", "tracks_ghost")
        norm = make_norm(nd, ("jets", "tracks"), "jets")
        norm.bind(schema_for(("jets", "tracks"), {"tracks": "tracks_ghost"}))
        norm.preflight()
        norm.materialise()
        assert torch.allclose(
            norm.means_tracks, expected_means(nd, "tracks_ghost", TRACK_VARIABLES)
        )
        assert torch.allclose(norm.stds_tracks, expected_stds(nd, "tracks_ghost", TRACK_VARIABLES))
        assert torch.allclose(norm.means_jets, expected_means(nd, "jets", JET_VARIABLES))

    def test_n1_without_map_raises_naming_stream(self, parity_dicts):
        """N1 without a stream->dataset map raises naming the raw stream 'tracks'."""
        nd, _cd = parity_dicts
        rekey(nd, "tracks", "tracks_ghost")
        norm = make_norm(nd, ("jets", "tracks"), "jets")
        norm.bind(schema_for(("jets", "tracks"), {}))
        with pytest.raises(ConfigError, match="missing input type 'tracks'"):
            norm.preflight()
        with pytest.raises(ValueError, match="Missing input type 'tracks'"):
            norm.materialise()

    def test_n2_global_stream_resolves_to_jets_dataset(self, parity_dicts):
        """N2: stream 'global' remapped to dataset 'jets' resolves the norm dict."""
        nd, _cd = parity_dicts
        norm = make_norm(nd, ("global",), "global")
        norm.bind(schema_for(("global",), {"global": "jets"}))
        norm.preflight()
        norm.materialise()
        assert torch.allclose(norm.means_global, expected_means(nd, "jets", JET_VARIABLES))

    def test_n2_without_map_raises_naming_global(self, parity_dicts):
        """N2 without a stream->dataset map raises naming the raw stream 'global'."""
        nd, _cd = parity_dicts
        norm = make_norm(nd, ("global",), "global")
        norm.bind(schema_for(("global",), {}))
        with pytest.raises(ConfigError, match="missing input type 'global'"):
            norm.preflight()
        with pytest.raises(ValueError, match="Missing input type 'global'"):
            norm.materialise()

    def test_unbound_preflight_takes_explicit_map(self, parity_dicts):
        """The unbound `salt graph validate` path takes an explicit datasets= map."""
        nd, _cd = parity_dicts
        rekey(nd, "tracks", "tracks_ghost")
        norm = make_norm(nd, ("jets", "tracks"), "jets")
        norm.preflight(datasets={"tracks": "tracks_ghost"})  # unbound, passes
        with pytest.raises(ConfigError, match="missing input type 'tracks'"):
            norm.preflight()  # unbound, no map

    def test_stream_equals_dataset_unchanged(self, parity_dicts):
        """stream==dataset (the unmapped default) is byte-identical to an explicit identity map."""
        nd, _cd = parity_dicts
        norm_a = make_norm(nd, ("jets", "tracks"), "jets")
        norm_a.bind(schema_for(("jets", "tracks"), {}))
        norm_a.preflight()
        norm_a.materialise()

        norm_b = make_norm(nd, ("jets", "tracks"), "jets")
        norm_b.bind(schema_for(("jets", "tracks"), {"jets": "jets", "tracks": "tracks"}))
        norm_b.preflight()
        norm_b.materialise()

        assert torch.equal(norm_a.means_tracks, norm_b.means_tracks)
        assert torch.allclose(norm_a.means_tracks, expected_means(nd, "tracks", TRACK_VARIABLES))

        # regression guard for test_preflight.py:53
        with pytest.raises(ConfigError, match="missing input type 'muons'"):
            make_norm(nd, ("jets", "muons"), "jets").preflight()

    def test_missing_in_both_raises_once_naming_both(self, parity_dicts):
        """A stream missing under both its dataset key AND its raw stream name names both."""
        nd, _cd = parity_dicts
        payload = yaml.safe_load(nd.read_text())
        del payload["tracks"]
        nd.write_text(yaml.safe_dump(payload, sort_keys=False))
        norm = make_norm(nd, ("jets", "tracks"), "jets")
        norm.bind(schema_for(("jets", "tracks"), {"tracks": "tracks_ghost"}))
        with pytest.raises(ConfigError) as exc:
            norm.preflight()
        assert "'tracks_ghost'" in str(exc.value)
        assert "'tracks'" in str(exc.value)
        with pytest.raises(ValueError) as exc:
            norm.materialise()
        assert "'tracks_ghost'" in str(exc.value)
        assert "'tracks'" in str(exc.value)


# -- ClassificationTaskModule ---------------------------------------------------


class TestClassificationStreamDatasetResolution:
    def test_n3_class_dict_keyed_by_dataset_resolves(self, parity_dicts):
        """N3: stream 'tracks' remapped to dataset 'tracks_ghost' resolves the class dict."""
        _nd, cd = parity_dicts
        rekey(cd, "tracks", "tracks_ghost")
        task = ClassificationTaskModule(
            stream="tracks",
            label="ftagTruthOriginLabel",
            class_names=ORIGIN_CLASSES,
            weight_source={"from_class_dict": str(cd)},
        )
        task.name = "track_origin"
        task.bind(
            ResolvedSchema(widths={"encoded.tracks": 16}, datasets={"tracks": "tracks_ghost"})
        )
        assert torch.equal(task.loss.weight, torch.ones(8))
        task.materialise()
        expected = torch.tensor([4.2, 73.7, 1.0, 17.5, 12.3, 12.5, 141.7, 22.3])
        assert torch.allclose(task.loss.weight, expected)

    def test_n3_without_map_raises_naming_stream(self, parity_dicts):
        """N3 without a stream->dataset map raises naming the raw stream 'tracks'."""
        _nd, cd = parity_dicts
        rekey(cd, "tracks", "tracks_ghost")
        task = ClassificationTaskModule(
            stream="tracks",
            label="ftagTruthOriginLabel",
            class_names=ORIGIN_CLASSES,
            weight_source={"from_class_dict": str(cd)},
        )
        task.name = "track_origin"
        task.bind(ResolvedSchema(widths={"encoded.tracks": 16}))
        with pytest.raises(ValueError, match="not found in class dict") as exc:
            task.materialise()
        assert "for stream 'tracks'" in str(exc.value)

    def test_n3_remapped_key_absent_names_both(self, parity_dicts):
        """A class dict missing both the dataset key and the raw stream key names both."""
        _nd, cd = parity_dicts
        payload = yaml.safe_load(cd.read_text())
        del payload["tracks"]
        cd.write_text(yaml.safe_dump(payload, sort_keys=False))
        task = ClassificationTaskModule(
            stream="tracks",
            label="ftagTruthOriginLabel",
            class_names=ORIGIN_CLASSES,
            weight_source={"from_class_dict": str(cd)},
        )
        task.name = "track_origin"
        task.bind(
            ResolvedSchema(widths={"encoded.tracks": 16}, datasets={"tracks": "tracks_ghost"})
        )
        with pytest.raises(ValueError) as exc:
            task.materialise()
        assert "'tracks_ghost'" in str(exc.value)
        assert "'tracks'" in str(exc.value)


# -- ResolvedSchema / reader_stream_datasets / resolve_bind_schema --------------


class TestResolvedSchemaDatasets:
    def test_dataset_of_defaults_to_stream(self):
        """`dataset_of` falls back to the stream name when it is not in `datasets`."""
        schema = ResolvedSchema(widths={})
        assert schema.dataset_of("x") == "x"
        assert schema.datasets == {}

        schema = ResolvedSchema(widths={}, datasets={"a": "b"})
        assert schema.dataset_of("a") == "b"
        assert schema.dataset_of("c") == "c"

    def test_reader_stream_datasets_duck_typing(self):
        """`reader_stream_datasets` duck-types on `reader.groups[stream].dataset`."""
        assert reader_stream_datasets(None) == {}
        assert reader_stream_datasets(object()) == {}
        reader = H5StructuredReader(
            groups={
                "jets": {"global_object": True},
                "tracks": {"global_object": False, "dataset": "tracks_ghost"},
            },
            schema=None,
        )
        assert reader_stream_datasets(reader) == {"jets": "jets", "tracks": "tracks_ghost"}

    def test_resolve_bind_schema_carries_map(self, parity_dicts):
        """`resolve_bind_schema(plan, datasets=...)` carries the map without changing widths/fields."""
        nd, _cd = parity_dicts
        plan = compile_gn2v2(build_gn2v2_modules(nd), Mode.FIT)
        schema_unmapped = resolve_bind_schema(plan)
        schema_mapped = resolve_bind_schema(plan, datasets={"tracks": "tracks_ghost"})
        assert schema_unmapped.datasets == {}
        assert schema_mapped.dataset_of("tracks") == "tracks_ghost"
        assert schema_unmapped.widths == schema_mapped.widths
        assert schema_unmapped.fields == schema_mapped.fields


# -- end-to-end fit through the real setup/on_fit_start wiring ------------------


class TestEndToEndRemappedFit:
    def test_fit_resolves_norm_dict_through_reader_map(self, tmp_path):
        """A real fit resolves 'tracks'->'tracks_ghost' through SaltModule.setup/on_fit_start."""
        nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
        write_parity_norm_dict(nd, cd)
        h5 = tmp_path / "pp_output_train.h5"
        write_dummy_file(h5, nd)
        rekey(nd, "tracks", "tracks_ghost")
        schema_path = tmp_path / "schema.yaml"
        save_schema(dump_schema(h5), schema_path)

        dm = SaltDataModule(
            {
                "reader": H5StructuredReader(
                    groups={"jets": {}, "tracks": {"dataset": "tracks_ghost"}},
                    schema=schema_path,
                ),
                "features": Features(
                    variables={"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}
                ),
                "labels": Labels(),
            },
            batch_size=100,
            num_workers=0,
            train_file=h5,
            val_file=h5,
            test_file=h5,
        )
        model = SaltModule(build_gn2v2_modules(nd) | {"loss": LossSum()}, lrs=LRS)
        make_trainer(max_epochs=1, limit_train_batches=1, limit_val_batches=1).fit(model, dm)

        assert model.schema.dataset_of("tracks") == "tracks_ghost"
        assert model.schema.dataset_of("jets") == "jets"
        assert torch.allclose(
            model._graph_modules["norm"].means_tracks,  # noqa: SLF001
            expected_means(nd, "tracks_ghost", TRACK_VARIABLES),
        )
