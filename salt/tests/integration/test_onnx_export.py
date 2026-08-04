"""End-to-end export tests: trace, ORT agreement, metadata, CLI."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from salt.graph import Bundle, Executor, Mode
from salt.graph.spec import GraphModule
from salt.model.modules import (
    Concat,
    GlobalAttentionPooling,
    Normaliser,
    Split,
    StreamEmbed,
    TransformerEncoder,
    bind_all,
    resolve_bind_schema,
)
from salt.model.modules.tasks import ClassificationTaskModule
from salt.outputs.sinks.onnx import (
    ExportConfig,
    ExportInput,
    check_onnx,
    compile_onnx_plan,
    export_graph,
    make_session,
    resolve_export_config,
)
from salt.outputs import OnnxExportSink, SeqClassIndex
from salt.tests._fixtures.gn2v2_fixture import (
    ELECTRON_VARIABLES,
    JET_VARIABLES,
    TRACK_VARIABLES,
    build_gn2v2_modules,
    compile_gn2v2,
    write_parity_norm_dict,
)
from salt.tests.unit.onnx.test_adapter import (
    VARIABLES,
    bind_producers,
    gn2_export_cfg,
    gn2_folded_modules,
    gn2_resolved,
)

SWEEP = [{"tracks": length} for length in (0, 1, 2, 7, 21, 39)]
"""Unit-scale sweep: the zero-token edge case + a spread; full sweep is gate O1."""


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Deterministically-weighted GN2 fixture exported through the FOLDED
    conversion-node path."""
    tmp = tmp_path_factory.mktemp("onnx_export")
    write_parity_norm_dict(tmp / "norm_dict.yaml", tmp / "class_dict.yaml")
    torch.manual_seed(42)  # deterministic non-trivial weights (retired v1 transfer stand-in)
    modules = gn2_folded_modules(tmp)
    resolved = gn2_resolved()
    plan = compile_onnx_plan(modules, resolved, VARIABLES)
    bind_all(modules, resolve_bind_schema([plan]))
    modules["norm"].materialise()
    result = export_graph(
        modules,
        gn2_export_cfg(),
        VARIABLES,
        tmp / "network.onnx",
        outputs=[],
        run_name="GN2_v2",
    )
    return SimpleNamespace(modules=modules, result=result, tmp=tmp)


class TestExportedModel:
    def test_subset_sweep_at_1e6(self, exported):
        # gate O1 at unit scale: v2-torch vs v2-ONNX, incl. L=0
        result = check_onnx(
            exported.result.adapter,
            exported.result.onnx_path,
            trials=2,
            float_rtol=1e-6,
            float_atol=1e-6,
            lengths_grid=SWEEP,
        )
        assert result.passed, result.failures
        assert result.n_cases == 2 * len(SWEEP)
        assert all(diff <= 1e-6 for diff in result.worst_abs_diff.values())

    @pytest.mark.parametrize("length", [0, 5, 39])
    def test_eager_test_plan_spot_check(self, exported, length):
        # gate O2 at unit scale: the eager TEST-plan softmax (independent
        # Executor path on the SAME weights) vs v2 ONNX — replaces the retired
        # v1 forward oracle
        session = make_session(exported.result.onnx_path)
        test_plan = compile_gn2v2(exported.modules, Mode.TEST)
        gen = torch.Generator().manual_seed(7)
        jets = torch.rand(1, len(JET_VARIABLES), generator=gen)
        tracks = torch.rand(length, len(TRACK_VARIABLES), generator=gen)
        b = Bundle()
        b.set("inputs.jets", jets.clone())
        b.set("inputs.tracks", tracks.unsqueeze(0).clone())
        b.set("masks.tracks", torch.zeros((1, length), dtype=torch.bool))
        with torch.no_grad():
            res = Executor(test_plan).run(b)
        p_ref = torch.softmax(res.get("preds.jets.jets_classification"), dim=-1).numpy().ravel()
        outputs = session.run(
            None, {"jet_features": jets.numpy(), "track_features": tracks.numpy()}
        )
        p_onnx = np.array(outputs[:3]).ravel()
        assert np.max(np.abs(p_ref - p_onnx)) <= 1e-6

    def test_perturbed_weights_fail_the_checker(self, exported):
        # negative control (gate O5 pattern): the checker must FAIL when the
        # eager reference and the exported graph genuinely differ. The
        # perturbation is ASYMMETRIC (one output-layer bias element): a
        # uniform first-layer shift can cancel through the softmax to ~1e-8
        # (observed) and would not exercise the comparison.
        head = exported.modules["jets_classification"]
        weight = list(head.parameters())[-1]  # the output-layer bias [n_classes]
        original = weight.detach().clone()
        try:
            with torch.no_grad():
                weight[0] += 1.0
            result = check_onnx(
                exported.result.adapter,
                exported.result.onnx_path,
                trials=1,
                float_rtol=1e-6,
                float_atol=1e-6,
                lengths_grid=[{"tracks": 11}],
            )
            assert not result.passed
            assert result.failures
        finally:
            with torch.no_grad():
                weight.copy_(original)

    def test_gnn_config_metadata(self, exported):
        # v1 envelope: key set AND order, with plan_hash appended last
        session = make_session(exported.result.onnx_path)
        meta = session.get_modelmeta()
        assert meta.description == "GN2v2"  # doc_string = model_name (to_onnx.py:853)
        info = json.loads(meta.custom_metadata_map["gnn_config"])
        assert list(info) == [
            "ckpt_path",
            "layers",
            "nodes",
            "config.yaml",
            "metadata.yaml",
            "salt_export_hash",
            "onnx_model_version",
            "output_names",
            "model_name",
            "inputs",
            "input_sequences",
            "combine_outputs",
            "rename_outputs",
            "plan_hash",  # the single ADDITIVE v2 key, after the full v1 set
        ]
        assert info["onnx_model_version"] == "v1"
        assert info["model_name"] == "GN2v2"
        assert info["output_names"] == exported.result.adapter.output_names
        # global variables: '_btagJes' stripped (to_onnx.py:828); placeholders
        assert info["inputs"] == [
            {
                "name": "jet_var",
                "variables": [
                    {"name": "pt", "offset": 0.0, "scale": 1.0},
                    {"name": "eta", "offset": 0.0, "scale": 1.0},
                ],
            }
        ]
        # sequence variables: unstripped, v1 athena_name
        assert info["input_sequences"][0]["name"] == "tracks_r22default_sd0sort"
        assert [v["name"] for v in info["input_sequences"][0]["variables"]] == list(TRACK_VARIABLES)
        assert info["combine_outputs"] == []
        assert info["rename_outputs"] == {}
        assert info["plan_hash"] == exported.result.plan.plan_hash

    def test_overwrite_in_place_export(self, exported):
        # exporting again over the same path is the caller's choice — the
        # programmatic core overwrites; the CLI guards with --overwrite
        again = export_graph(
            exported.modules,
            gn2_export_cfg(),
            VARIABLES,
            exported.result.onnx_path,
            outputs=[],
            run_name="GN2_v2",
        )
        assert again.plan.plan_hash == exported.result.plan.plan_hash


# the two-dynamic-axes Split grid (the in-tree de-risk;
# the release-blocker version on bigger widths is gate O3)


def build_two_stream_modules(norm_dict) -> dict[str, GraphModule]:
    torch.manual_seed(42)
    dense = {"hidden_layers": [16], "activation": "ReLU"}
    modules: dict[str, GraphModule] = {
        "norm": Normaliser(
            norm_dict=norm_dict, streams=["jets", "tracks", "electrons"], global_object="jets"
        ),
        "track_embed": StreamEmbed(
            stream="tracks", out_dim=16, dense=dense, context=["normed.jets"]
        ),
        "electron_embed": StreamEmbed(
            stream="electrons", out_dim=16, dense=dense, context=["normed.jets"]
        ),
        "concat": Concat(streams=["tracks", "electrons"]),
        "encoder": TransformerEncoder(
            dim=16,
            num_layers=2,
            out_dim=16,
            attention={"num_heads": 2, "attn_type": "torch-math"},
            dense={"activation": "ReLU", "gated": False},
        ),
        "split": Split(streams=["tracks", "electrons"]),
        "pool": GlobalAttentionPooling(input="encoded.seq", out="pooled.global"),
        "jets_classification": ClassificationTaskModule(
            stream="jets",
            label="flavour_label",
            class_names=["bjets", "cjets", "ujets"],
            input="pooled.global",
            dense=dense,
        ),
        "track_origin": ClassificationTaskModule(
            stream="tracks",
            label="ftagTruthOriginLabel",
            class_names=[f"o{i}" for i in range(8)],
            context="pooled.global",
            dense=dense,
        ),
        "electron_origin": ClassificationTaskModule(
            stream="electrons",
            label="ftagTruthOriginLabel",
            class_names=[f"e{i}" for i in range(4)],
            context="pooled.global",
            dense=dense,
        ),
    }
    for name, module in modules.items():
        module.name = name
    return modules


@pytest.fixture(scope="module")
def two_stream(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("onnx_two_stream")
    write_parity_norm_dict(tmp / "norm_dict.yaml", tmp / "class_dict.yaml")
    modules = build_two_stream_modules(tmp / "norm_dict.yaml")
    variables = {
        "jets": list(JET_VARIABLES),
        "tracks": list(TRACK_VARIABLES),
        "electrons": list(ELECTRON_VARIABLES),
    }
    export_cfg = ExportConfig(
        model_name="TwoStream",
        inputs=[
            ExportInput(port="inputs.jets", name="jet_features"),
            ExportInput(
                port="inputs.tracks", name="track_features", sequence=True, dyn_axis="n_tracks"
            ),
            ExportInput(
                port="inputs.electrons",
                name="electron_features",
                sequence=True,
                dyn_axis="n_electrons",
            ),
        ],
    )
    # folded path: conversion nodes + OnnxExportSink (the off-graph manifest
    # is retired). The jets head's softmax + the two argmax aux leaves fold into
    # ClassProbs/SeqClassIndex nodes named by the sink.
    from salt.outputs import ClassProbs

    def _n(node, name):
        node.name = name
        return node

    modules.update({
        "jet_probs": _n(ClassProbs(task="jets_classification", stream="jets"), "jet_probs"),
        "track_origin_index": _n(
            SeqClassIndex(task="track_origin", stream="tracks"), "track_origin_index"
        ),
        "electron_origin_index": _n(
            SeqClassIndex(task="electron_origin", stream="electrons"), "electron_origin_index"
        ),
        "onnx_export": _n(OnnxExportSink(), "onnx_export"),
    })
    # the tuple is collected from the producers above: the jet probs inherit
    # pb/pc/pu from their task, and each SeqClassIndex inherits its pascal-cased
    # per-token name (TrackOrigin / ElectronOrigin) and its n_<stream> axis.
    bind_producers(modules)
    resolved = resolve_export_config(export_cfg, "two_stream")
    plan = compile_onnx_plan(modules, resolved, variables)
    bind_all(modules, resolve_bind_schema([plan]))
    modules["norm"].materialise()
    result = export_graph(
        modules, export_cfg, variables, tmp / "two.onnx", outputs=[], run_name="two_stream"
    )
    return SimpleNamespace(result=result, variables=variables)


class TestTwoDynamicAxes:
    def test_split_grid_including_zeros(self, two_stream):
        # known risk: the eager Split slicing is provably WRONG under
        # traced export with two dynamic axes (recipe spike: 15/15 grid
        # points mismatched); the index_select export branch must agree
        # with eager v2 on the full grid INCLUDING zero-length streams
        grid = [{"tracks": lt, "electrons": le} for lt in (0, 1, 3, 11) for le in (0, 2, 5)]
        result = check_onnx(
            two_stream.result.adapter,
            two_stream.result.onnx_path,
            trials=2,
            float_rtol=1e-6,
            float_atol=1e-6,
            lengths_grid=grid,
        )
        assert result.passed, result.failures
        assert result.n_cases == 2 * len(grid)

    def test_independent_axes_actually_vary(self, two_stream):
        # the grid is only meaningful if the two axes feed DIFFERENT
        # outputs: TrackOrigin tracks n_tracks, ElectronOrigin n_electrons
        session = make_session(two_stream.result.onnx_path)
        gen = torch.Generator().manual_seed(9)
        jets = torch.rand(1, len(JET_VARIABLES), generator=gen)
        tracks = torch.rand(4, len(TRACK_VARIABLES), generator=gen)
        electrons = torch.rand(2, len(ELECTRON_VARIABLES), generator=gen)
        outputs = session.run(
            None,
            {
                "jet_features": jets.numpy(),
                "track_features": tracks.numpy(),
                "electron_features": electrons.numpy(),
            },
        )
        names = two_stream.result.adapter.output_names
        assert outputs[names.index("TwoStream_TrackOrigin")].shape == (4,)
        assert outputs[names.index("TwoStream_ElectronOrigin")].shape == (2,)


# the salt surface: export-block parsing + the CLI from a real checkpoint


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory):
    """A real checkpoint + saved run config for the ``salt export`` CLI tests."""
    from lightning import Trainer

    from salt.data import Features, GraphDataModule, H5StructuredReader, Labels
    from salt.main import CONFIG_DIR
    from salt.model.saltmodule import SaltModule
    from salt.testing.inputs import write_dummy_file

    tmp_path = tmp_path_factory.mktemp("onnx_cli_run")
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    write_parity_norm_dict(tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml")
    write_dummy_file(str(tmp_path / "dummy_test_file_ttbar.h5"), str(tmp_path / "norm_dict.yaml"))
    torch.manual_seed(42)  # deterministic checkpoint weights (retired v1 transfer stand-in)
    modules = build_gn2v2_modules(tmp_path / "norm_dict.yaml")
    model = SaltModule(
        modules,
        lrs={"initial": 1e-7, "max": 1e-3, "end": 1e-5, "pct_start": 0.01},
        name="GN2v2_dummy",
    )
    dm = GraphDataModule(
        modules={
            "reader": H5StructuredReader(
                groups={"jets": {"global_object": True}, "tracks": {"global_object": False}}
            ),
            "features": Features(variables=VARIABLES),
            "labels": Labels(),
        },
        test_file=tmp_path / "dummy_test_file_ttbar.h5",
        batch_size=100,
        num_test=100,
        num_workers=0,
        pin_memory=False,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_test_batches=0,
    )
    trainer.test(model, datamodule=dm)
    ckpt = run_dir / "checkpoints" / "epoch=000-loss=0.10000.ckpt"
    trainer.save_checkpoint(ckpt)
    config = yaml.safe_load((CONFIG_DIR / "gn2v2-opendata.yaml").read_text())
    config["model"]["init_args"]["modules"]["norm"]["init_args"]["norm_dict"] = str(
        tmp_path / "norm_dict.yaml"
    )
    (run_dir / "config.yaml").write_text(yaml.dump(config, sort_keys=False))
    return SimpleNamespace(ckpt=ckpt, run_dir=run_dir, config=config)


class TestSaltSurface:
    def test_export_block_round_trips_through_the_parser(self):
        from salt.main import CONFIG_DIR, SaltCLI

        cli = SaltCLI(
            args=[
                "--config",
                str(CONFIG_DIR / "gn2v2-opendata.yaml"),
                "--model.modules.norm.init_args.norm_dict=unused.yaml",
            ],
            run=False,
        )
        export_cfg = cli._get(cli.config_init, "export")
        assert isinstance(export_cfg, ExportConfig)
        assert export_cfg.model_name == "GN2v2dummy"
        assert [entry.port for entry in export_cfg.inputs] == ["inputs.jets", "inputs.tracks"]
        assert export_cfg.inputs[1].sequence is True
        assert export_cfg.inputs[1].dyn_axis == "n_tracks"
        # the shipped configs carry NO export.outputs — the manifest
        # derives from the writers (rename/combine empty by default)
        assert export_cfg.outputs == []
        assert export_cfg.rename == {}
        assert export_cfg.combine == []

    def test_export_contract_round_trips_onto_the_sink(self):
        # the contract's NEW config home is the ONNX sink: jsonargparse must
        # resolve its `inputs:` entries into ExportInput dataclasses
        from salt.main import CONFIG_DIR, SaltCLI

        cli = SaltCLI(
            args=[
                "--config",
                str(CONFIG_DIR / "MaskFormer.yaml"),
                "--model.modules.norm.init_args.norm_dict=unused.yaml",
            ],
            run=False,
        )
        assert cli._get(cli.config_init, "export") is None  # migrated: no block left
        sink = cli._get(cli.config_init, "outputs")["onnx_export"]
        assert isinstance(sink, OnnxExportSink)
        assert sink.model_name == "MFv2"
        assert [entry.port for entry in sink.inputs] == ["inputs.jets", "inputs.tracks"]
        assert sink.inputs[1].sequence is True
        assert sink.inputs[1].dyn_axis == "n_tracks"
        assert sink.rename == {}
        assert sink.combine == []

    def test_dispatch_wired_into_salt(self):
        from salt.main import main as salt_main

        with pytest.raises(SystemExit) as excinfo:
            salt_main(["export", "--help"])
        assert excinfo.value.code == 0

    def test_cli_export_from_checkpoint(self, cli_run, capsys):
        from salt.outputs.sinks.onnx.export import main as export_main

        rc = export_main([
            "--ckpt_path",
            str(cli_run.ckpt),
            "--trials",
            "1",
            "--max-length",
            "8",
            "--float-atol",
            "1e-6",
        ])
        assert rc == 0
        onnx_path = cli_run.run_dir / "network.onnx"
        assert onnx_path.is_file()
        out = capsys.readouterr().out
        # int8 outputs get a POSITIVE verdict row (review fix: the
        # checker table used to confirm float outputs only)
        assert "GN2v2dummy_VertexIndex" in out
        assert "int8 exact over" in out
        # the traced plan table is written next to the .onnx
        plan_txt = cli_run.run_dir / "plan_onnx.txt"
        assert plan_txt.is_file()
        assert "[mode=ONNX]" in plan_txt.read_text()
        session = make_session(onnx_path)
        meta = session.get_modelmeta()
        assert meta.description == "GN2v2dummy"
        info = json.loads(meta.custom_metadata_map["gnn_config"])
        assert info["model_name"] == "GN2v2dummy"
        assert info["ckpt_path"] == str(Path(cli_run.ckpt).resolve())
        assert info["config.yaml"]["name"] == "GN2v2_dummy"
        # without --overwrite a second export must refuse (to_onnx.py:710-711)
        # with ONE actionable line, not a traceback (review fix)
        rc2 = export_main(["--ckpt_path", str(cli_run.ckpt), "--no-check"])
        assert rc2 == 1
        err = capsys.readouterr().err
        assert "Found existing file" in err
        assert "-o/--overwrite" in err

    def test_export_less_config_error_is_actionable(self, cli_run, tmp_path, capsys):
        # a run config trained without ANY export contract (no top-level block,
        # nothing on the sink) must fail naming the sink's inputs: as the home
        from salt.outputs.sinks.onnx.export import main as export_main

        config = dict(cli_run.config)
        config.pop("export")
        no_export_cfg = tmp_path / "no_export.yaml"
        no_export_cfg.write_text(yaml.dump(config, sort_keys=False))
        rc = export_main(["--ckpt_path", str(cli_run.ckpt), "-c", str(no_export_cfg)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "declares no input" in err
        assert "OnnxExportSink" in err
        assert "init_args.inputs" in err  # the config address to fix

    def test_export_block_stacked_as_second_config(self, cli_run, tmp_path):
        # the documented escape hatch: -c is repeatable, later files
        # deep-merge on top (the fit semantics) — an export-block-only
        # override file completes a run config trained without one
        from salt.outputs.sinks.onnx.export import main as export_main

        config = dict(cli_run.config)
        export_block = {"export": config.pop("export")}
        no_export_cfg = tmp_path / "no_export.yaml"
        no_export_cfg.write_text(yaml.dump(config, sort_keys=False))
        override_cfg = tmp_path / "export_block.yaml"
        override_cfg.write_text(yaml.dump(export_block, sort_keys=False))
        out_path = tmp_path / "stacked.onnx"
        rc = export_main([
            "--ckpt_path",
            str(cli_run.ckpt),
            "-c",
            str(no_export_cfg),
            "-c",
            str(override_cfg),
            "--output",
            str(out_path),
            "--trials",
            "1",
            "--max-length",
            "4",
            "--float-atol",
            "1e-6",
        ])
        assert rc == 0
        assert out_path.is_file()
        meta = make_session(out_path).get_modelmeta()
        assert meta.description == "GN2v2dummy"

    def test_manifest_flag_prints_without_checkpoint(self, cli_run, capsys):
        # salt export --manifest: the OnnxExportSink-derived manifest, no ckpt
        # needed (the off-graph writer manifest is retired — the sink names the
        # folded conversion outputs.* leaves)
        from salt.outputs.sinks.onnx.export import main as export_main

        rc = export_main(["--manifest", "-c", str(cli_run.run_dir / "config.yaml")])
        assert rc == 0
        out = capsys.readouterr().out
        assert "ONNX output manifest (folded conversion nodes, model_name=GN2v2dummy)" in out
        for name in ("GN2v2dummy_pb", "GN2v2dummy_TrackOrigin", "GN2v2dummy_VertexIndex"):
            assert name in out
        assert "folded conversion node (outputs.* leaf)" in out

    def test_config_declared_outputs_hard_error_through_the_cli(self, cli_run, tmp_path, capsys):
        # the retired export.outputs carrier must fire on the CLI path, through
        # the deprecated top-level block (its only remaining spelling)
        from salt.outputs.sinks.onnx.export import main as export_main

        config = dict(cli_run.config)
        config["export"] = dict(config["export"])
        config["export"]["outputs"] = [
            {"port": "preds.jets.jets_classification", "names": ["pb", "pc", "pu"]}
        ]
        legacy_cfg = tmp_path / "legacy_outputs.yaml"
        legacy_cfg.write_text(yaml.dump(config, sort_keys=False))
        rc = export_main(["--ckpt_path", str(cli_run.ckpt), "-c", str(legacy_cfg)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "export.outputs was REMOVED" in err
        assert "OnnxExportSink" in err
