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
    compile_gn2v2,
    write_parity_norm_dict,
)
from salt.tests._fixtures.gn2v2_test_config import full_family_config, small_config
from salt.tests.unit.outputs.sinks.onnx.test_adapter import (
    VARIABLES,
    bind_producers,
    gn2_export_cfg,
    gn2_folded_modules,
    gn2_resolved,
)

SWEEP = [{"tracks": length} for length in (0, 1, 2, 7, 21, 39)]
"""Unit-scale sweep: the zero-token edge case + a spread (the full L-sweep runs in check_onnx)."""


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
        # v2-torch vs v2-ONNX, incl. L=0
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
        # the eager TEST-plan softmax (independent Executor path on the
        # SAME weights) vs v2 ONNX
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
        # negative control: the checker must FAIL when the
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


# the two-dynamic-axes Split grid


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
        # eager Split slicing is provably WRONG under traced export with
        # two dynamic axes (15/15 grid points mismatched when tried); the
        # index_select export branch must agree with eager v2 on the full
        # grid INCLUDING zero-length streams
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


def _real_cli_run(tmp_path_factory, config_path: Path, label: str) -> SimpleNamespace:
    """A REAL 1-epoch ``salt fit`` of ``config_path`` — checkpoint + the
    trainer's OWN saved config.yaml (pattern: ``test_h5_sink.py``'s
    ``data``/``ckpt`` fixtures), relocated into the
    ``run_dir/checkpoints/{ckpt}`` + ``run_dir/config.yaml`` layout the tests
    in this module (which invoke ``salt export`` with no ``-c``) rely on for
    ckpt-based config inference (``export.py``:
    ``ckpt.parents[1] / "config.yaml"``).

    Kills the hand-built-fixture/config drift class (pipeline #15650554
    diagnosis item 2): the OLD ``cli_run`` fixture wrote a checkpoint from
    hand-built ``build_gn2v2_modules()`` (hidden layer width 16) but saved
    ``small_config()`` VERBATIM (shipped arch [256]/[128,64,32]) — loading
    that mismatched state_dict into the shipped architecture then failed with
    a state_dict shape error. A real fit trains AND saves the SAME config, so
    the two can never drift.
    """
    import shutil

    from salt.main import main as salt_main
    from salt.schema import dump_schema, save_schema
    from salt.testing.inputs import write_dummy_file
    from salt.utils.config_utils import disable_logger_in_config

    tmp_path = tmp_path_factory.mktemp(label)
    nd_path, cd_path = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    h5_path = tmp_path / "dummy_test_file_ttbar.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = tmp_path / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)

    fit_dir = tmp_path / "fit"
    rc = salt_main([
        "fit",
        "--config",
        disable_logger_in_config(str(config_path)),
        f"--data.train_file={h5_path}",
        f"--data.val_file={h5_path}",
        f"--data.modules.reader.init_args.schema={schema_path}",
        f"--model.modules.norm.init_args.norm_dict={nd_path}",
        "--data.num_workers=0",
        f"--trainer.default_root_dir={fit_dir}",
        "--trainer.accelerator=cpu",
        "--trainer.max_epochs=1",
        "--trainer.limit_train_batches=2",
        "--trainer.limit_val_batches=2",
        "--trainer.num_sanity_val_steps=0",
        "--trainer.log_every_n_steps=1",
        "--trainer.logger=false",
        "--callbacks.progress=null",
    ])
    assert rc == 0, f"salt fit on {config_path.name} must run end-to-end"
    ckpts = sorted(fit_dir.rglob("*.ckpt"))
    assert ckpts, f"no checkpoint under {fit_dir}"
    configs = sorted(fit_dir.rglob("config.yaml"))
    assert configs, f"no saved config.yaml under {fit_dir}"

    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    ckpt = run_dir / "checkpoints" / ckpts[0].name
    shutil.copy(ckpts[0], ckpt)
    shutil.copy(configs[0], run_dir / "config.yaml")
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    return SimpleNamespace(ckpt=ckpt, run_dir=run_dir, config=config)


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory):
    """``small_config()`` (SPLIT_OUTPUTS: flavour head H5+ONNX, track heads
    H5-only) — see ``_real_cli_run``.
    """
    return _real_cli_run(tmp_path_factory, small_config(), "onnx_cli_run")


@pytest.fixture(scope="module")
def cli_run_full_family(tmp_path_factory):
    """``full_family_config()`` (one ``RunTaskOutput`` over all three heads,
    ``track_vertexing`` un-deferred) — see ``_real_cli_run``. Needed
    separately from ``cli_run``: ``small_config()``'s SPLIT_OUTPUTS shape
    makes ``track_origin`` H5-only and defers ``track_vertexing`` entirely,
    so ITS ONNX manifest carries only the flavour head — a manifest test
    asserting TrackOrigin/VertexIndex leaves needs THIS config instead
    (pipeline #15650554 diagnosis item 3).
    """
    return _real_cli_run(tmp_path_factory, full_family_config(), "onnx_cli_run_full_family")


class TestSaltSurface:
    def test_export_contract_round_trips_onto_the_dummy_sink(self):
        from salt.main import SaltCLI
        from salt.utils.config_utils import disable_logger_in_config

        cli = SaltCLI(
            args=[
                "--config",
                disable_logger_in_config(str(small_config())),
                "--model.modules.norm.init_args.norm_dict=unused.yaml",
            ],
            run=False,
        )
        sink = cli._get(cli.config_init, "outputs")["onnx_export"]  # noqa: SLF001 - main.py precedent
        assert isinstance(sink, OnnxExportSink)
        export_cfg = sink.export_config("GN2v2_dummy")
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
        from salt.utils.config_utils import disable_logger_in_config

        cli = SaltCLI(
            args=[
                "--config",
                disable_logger_in_config(str(CONFIG_DIR / "MaskFormer.yaml")),
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

    def test_cli_export_from_checkpoint(self, cli_run_full_family, capsys):
        # needs the full-family config, not cli_run/small_config(): small_config()'s
        # SPLIT_OUTPUTS makes track_origin H5-only (modes: [test]) and defers
        # track_vertexing (expose: [fit, val]) out of its ONNX export set entirely,
        # so its export tuple is pb/pc/pu only — zero int8 outputs, and no
        # VertexIndex leaf for the assertions below to find (Opus review finding,
        # same class of bug as diagnosis item 3 / the manifest test fix).
        from salt.outputs.sinks.onnx.export import main as export_main

        rc = export_main([
            "--ckpt_path",
            str(cli_run_full_family.ckpt),
            "--trials",
            "1",
            "--max-length",
            "8",
            "--float-atol",
            "1e-6",
        ])
        assert rc == 0
        onnx_path = cli_run_full_family.run_dir / "network.onnx"
        assert onnx_path.is_file()
        out = capsys.readouterr().out
        # int8 outputs get a POSITIVE verdict row, not only the floats
        assert "GN2v2dummy_VertexIndex" in out
        assert "int8 exact over" in out
        # the traced plan table is written next to the .onnx
        plan_txt = cli_run_full_family.run_dir / "plan_onnx.txt"
        assert plan_txt.is_file()
        assert "[mode=ONNX]" in plan_txt.read_text()
        session = make_session(onnx_path)
        meta = session.get_modelmeta()
        assert meta.description == "GN2v2dummy"
        info = json.loads(meta.custom_metadata_map["gnn_config"])
        assert info["model_name"] == "GN2v2dummy"
        assert info["ckpt_path"] == str(Path(cli_run_full_family.ckpt).resolve())
        assert info["config.yaml"]["name"] == "GN2v2_dummy"
        # without --overwrite a second export must refuse (to_onnx.py:710-711)
        # with ONE actionable line, not a traceback
        rc2 = export_main(["--ckpt_path", str(cli_run_full_family.ckpt), "--no-check"])
        assert rc2 == 1
        err = capsys.readouterr().err
        assert "Found existing file" in err
        assert "-o/--overwrite" in err

    def test_export_less_config_error_is_actionable(self, cli_run, tmp_path, capsys):
        # a run config trained without ANY export contract (no sink declared)
        # must fail naming the sink's inputs: as the home
        from salt.outputs.sinks.onnx.export import main as export_main

        config = dict(cli_run.config)
        outputs = dict(config["outputs"])
        del outputs["onnx_export"]
        config["outputs"] = outputs
        no_export_cfg = tmp_path / "no_export.yaml"
        no_export_cfg.write_text(yaml.dump(config, sort_keys=False))
        rc = export_main(["--ckpt_path", str(cli_run.ckpt), "-c", str(no_export_cfg)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "declares no input" in err
        assert "OnnxExportSink" in err
        assert "init_args.inputs" in err  # the config address to fix

    def test_export_sink_stacked_as_second_config(self, cli_run, tmp_path):
        # the documented escape hatch: -c is repeatable, later files
        # deep-merge on top (the fit semantics) — a sink-only override file
        # completes a run config trained without one
        from salt.outputs.sinks.onnx.export import main as export_main

        config = dict(cli_run.config)
        outputs = dict(config["outputs"])
        sink_only = {"outputs": {"onnx_export": outputs.pop("onnx_export")}}
        config["outputs"] = outputs
        no_export_cfg = tmp_path / "no_export.yaml"
        no_export_cfg.write_text(yaml.dump(config, sort_keys=False))
        override_cfg = tmp_path / "export_sink.yaml"
        override_cfg.write_text(yaml.dump(sink_only, sort_keys=False))
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

    def test_manifest_flag_prints_without_checkpoint(self, cli_run_full_family, capsys):
        # salt export --manifest: the OnnxExportSink-derived manifest, no ckpt
        # needed (the off-graph writer manifest is retired — the sink names the
        # folded conversion outputs.* leaves). Needs the full-family config
        # (see cli_run_full_family): small_config()'s SPLIT_OUTPUTS defers
        # track_vertexing and keeps track_origin H5-only, so its ONNX manifest
        # never carries TrackOrigin/VertexIndex at all (pipeline #15650554
        # diagnosis item 3).
        from salt.outputs.sinks.onnx.export import main as export_main

        rc = export_main(["--manifest", "-c", str(cli_run_full_family.run_dir / "config.yaml")])
        assert rc == 0
        out = capsys.readouterr().out
        assert "ONNX output manifest (folded conversion nodes, model_name=GN2v2dummy)" in out
        for name in ("GN2v2dummy_pb", "GN2v2dummy_TrackOrigin", "GN2v2dummy_VertexIndex"):
            assert name in out
        assert "folded conversion node (outputs.* leaf)" in out
