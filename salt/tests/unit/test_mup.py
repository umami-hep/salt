"""Tests for the muP routing surface + tooling (`salt.model.mup`)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from salt.graph import ConfigError, Mode
from salt.graph.planner import compile_plan
from salt.main import CONFIG_DIR
from salt.model.mup import (
    _combined_graph,
    _parse_cli,
    build_model_at_widths,
    coord_check,
    generate_shapes,
    plot_coord_data,
    setup_mup,
)
from salt.model.bind import resolve_bind_schema
from salt.model.saltmodule import (
    SaltModule,
    module_mup_enabled,
    module_supports_mup,
    validate_mup_routing,
)
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict

DUMMY_CFG = str(CONFIG_DIR / "GN2/gn2v2-dummy.yaml")
LRS = {"initial": 1e-4, "max": 5e-4, "end": 1e-5, "pct_start": 0.1}


@pytest.fixture
def norm_dict(tmp_path) -> Path:
    nd = tmp_path / "nd.yaml"
    cd = tmp_path / "cd.yaml"
    write_parity_norm_dict(nd, cd)
    return nd


@pytest.fixture
def mup_override(tmp_path) -> Path:
    """A GN2_muP-routing override stacked on gn2v2-dummy.yaml (mup flags + routing)."""
    override = {
        "name": "GN2muP_test",
        "model": {
            "init_args": {
                "optimizer": "AdamW",
                "mup": {
                    "shape_path": str(tmp_path / "shapes.bsh"),
                    "apply_to": ["track_embed", "encoder"],
                },
                "modules": {
                    "track_embed": {"init_args": {"mup": True}},
                    "encoder": {"init_args": {"mup": True}},
                },
            }
        },
    }
    path = tmp_path / "mup_override.yaml"
    path.write_text(yaml.safe_dump(override, sort_keys=False))
    return path


# GN2_muP.yaml scaled-down dims
_MUP_TRACK_VARIABLES: tuple[str, ...] = ("d0", "z0SinTheta", "dphi", "deta", "qOverP")
_MUP_JET_VARIABLES: tuple[str, ...] = ("pt_btagJes", "eta_btagJes")
_MUP_CLASS_NAMES: tuple[str, ...] = ("bjets", "cjets", "ujets")
_MUP_EMBED_DIM = 16
_MUP_OUT_DIM = 8
_MUP_NUM_HEADS = 2
_MUP_NUM_LAYERS = 2


def _muP_modules(norm_dict: Path) -> dict:
    """A GN2_muP-shaped module dict with mup embed + mup encoder."""
    from salt.model.modules import (  # noqa: PLC0415 - test-local fixture
        Concat,
        GlobalAttentionPooling,
        LossSum,
        Normaliser,
        StreamEmbed,
        TransformerEncoder,
    )
    from salt.model.modules.tasks import ClassificationTaskModule  # noqa: PLC0415

    modules: dict = {
        "norm": Normaliser(norm_dict=norm_dict, streams=["jets", "tracks"], global_object="jets"),
        "track_embed": StreamEmbed(
            stream="tracks",
            out_dim=_MUP_EMBED_DIM,
            dense={"hidden_layers": [_MUP_EMBED_DIM], "activation": "ReLU"},
            context=["normed.jets"],
            mup=True,
        ),
        "concat": Concat(streams=["tracks"]),
        "encoder": TransformerEncoder(
            dim=_MUP_EMBED_DIM,
            num_layers=_MUP_NUM_LAYERS,
            out_dim=_MUP_OUT_DIM,
            attention={"num_heads": _MUP_NUM_HEADS, "attn_type": "torch-math"},
            dense={"activation": "ReLU"},
            mup=True,
        ),
        "pool": GlobalAttentionPooling(input="encoded.seq", out="pooled.global"),
        "jets_classification": ClassificationTaskModule(
            stream="jets",
            label="flavour_label",
            class_names=list(_MUP_CLASS_NAMES),
            input="pooled.global",
            dense={"hidden_layers": [_MUP_OUT_DIM], "activation": "ReLU"},
        ),
        "loss": LossSum(),
    }
    for name, module in modules.items():
        module.name = name
    modules["loss"].narrow(LossSum.collect_loss_keys(modules))
    return modules


class TestRoutingValidator:
    def test_module_supports_and_enabled(self, norm_dict):
        mods = _muP_modules(norm_dict)
        assert module_supports_mup(mods["track_embed"])
        assert module_supports_mup(mods["encoder"])
        assert module_mup_enabled(mods["track_embed"])  # built with mup=True
        assert not module_supports_mup(mods["pool"])  # no mup init_arg

    def test_valid_routing_normalises(self, norm_dict):
        cfg = validate_mup_routing(
            {"apply_to": ["track_embed", "encoder"]}, _muP_modules(norm_dict)
        )
        assert cfg == {"apply_to": ["track_embed", "encoder"], "shape_path": None}

    def test_none_is_passthrough(self, norm_dict):
        assert validate_mup_routing(None, _muP_modules(norm_dict)) is None

    def test_apply_to_non_mup_module_errors(self, norm_dict):
        with pytest.raises(ConfigError, match="no 'mup' init_arg"):
            validate_mup_routing({"apply_to": ["pool"]}, _muP_modules(norm_dict))

    def test_apply_to_unknown_module_errors(self, norm_dict):
        with pytest.raises(ConfigError, match="not a configured module"):
            validate_mup_routing({"apply_to": ["nope"]}, _muP_modules(norm_dict))

    def test_mup_on_module_outside_apply_to_warns(self, norm_dict):
        with pytest.warns(UserWarning, match="NOT in"):
            validate_mup_routing({"apply_to": ["track_embed"]}, _muP_modules(norm_dict))

    def test_unknown_key_and_empty_apply_to_error(self, norm_dict):
        with pytest.raises(ConfigError, match="unknown key"):
            validate_mup_routing(
                {"apply_to": ["encoder"], "bogus": 1}, _muP_modules(norm_dict)
            )
        with pytest.raises(ConfigError, match="non-empty 'apply_to'"):
            validate_mup_routing({"apply_to": []}, _muP_modules(norm_dict))

    def test_saltmodule_stores_validated_cfg(self, norm_dict):
        m = SaltModule(
            modules=_muP_modules(norm_dict), lrs=LRS, mup={"apply_to": ["track_embed", "encoder"]}
        )
        assert m.mup_cfg == {"apply_to": ["track_embed", "encoder"], "shape_path": None}


class TestMuAdamWSwap:
    def test_muadamw_when_mup_configured(self, norm_dict):
        from mup.optim import MuAdamW

        m = SaltModule(
            modules=_muP_modules(norm_dict), lrs=LRS, mup={"apply_to": ["track_embed", "encoder"]}
        )
        assert m._get_optimizer_class() is MuAdamW  # noqa: SLF001

    def test_adamw_when_no_mup(self, norm_dict):
        from torch.optim import AdamW

        m = SaltModule(modules=_muP_modules(norm_dict), lrs=LRS)
        assert m._get_optimizer_class() is AdamW  # noqa: SLF001
        assert m.mup_cfg is None

    def test_mup_overrides_named_optimizer(self, norm_dict):
        from mup.optim import MuAdamW

        # even with optimizer="lion" the mup block forces MuAdamW
        m = SaltModule(
            modules=_muP_modules(norm_dict),
            lrs=LRS,
            optimizer="lion",
            mup={"apply_to": ["track_embed", "encoder"]},
        )
        assert m._get_optimizer_class() is MuAdamW  # noqa: SLF001


class TestShapeGeneration:
    def test_generate_shapes_writes_file(self, norm_dict, mup_override, tmp_path):
        out, base_shapes = generate_shapes(
            [DUMMY_CFG, str(mup_override)],
            save_path=str(tmp_path / "gen.bsh"),
            base_width=8,
            delta_width=16,
            set_overrides=[f"model.modules.norm.init_args.norm_dict={norm_dict}"],
        )
        assert out.is_file()
        assert base_shapes  # non-empty infshape dict

    def test_equal_widths_error(self, norm_dict, mup_override, tmp_path):
        with pytest.raises(ConfigError, match="must differ"):
            generate_shapes(
                [DUMMY_CFG, str(mup_override)],
                save_path=str(tmp_path / "x.bsh"),
                base_width=16,
                delta_width=16,
                set_overrides=[f"model.modules.norm.init_args.norm_dict={norm_dict}"],
            )

    def test_shape_applied_at_bind_resolves_width_mult(self, norm_dict, mup_override):
        from mup import MuReadout

        # generate shapes (base 8, delta 16) into the path the override points at
        generate_shapes(
            [DUMMY_CFG, str(mup_override)],
            base_width=8,
            delta_width=16,
            set_overrides=[f"model.modules.norm.init_args.norm_dict={norm_dict}"],
        )
        cli = _parse_cli(
            [DUMMY_CFG, str(mup_override)],
            [f"model.modules.norm.init_args.norm_dict={norm_dict}"],
        )
        model = cli.model
        combined = _combined_graph(cli)
        plan = compile_plan(combined, Mode.FIT, sources={}, sinks=["loss.total"])
        model._bind(resolve_bind_schema([plan]))  # noqa: SLF001 - applies the shape file
        out_proj = model.net["encoder"].encoder.out_proj
        assert isinstance(out_proj, MuReadout)
        # 16-width model on an 8-width base -> width_mult == 2.0
        assert math.isclose(out_proj.width_mult(), 2.0)

    def test_missing_shape_file_errors_at_bind(self, norm_dict, tmp_path):
        # mup block points at a non-existent shape file -> bind raises
        override = {
            "model": {
                "init_args": {
                    "mup": {
                        "shape_path": str(tmp_path / "absent.bsh"),
                        "apply_to": ["track_embed", "encoder"],
                    },
                    "modules": {
                        "track_embed": {"init_args": {"mup": True}},
                        "encoder": {"init_args": {"mup": True}},
                    },
                }
            }
        }
        ov = tmp_path / "absent_override.yaml"
        ov.write_text(yaml.safe_dump(override))
        cli = _parse_cli(
            [DUMMY_CFG, str(ov)], [f"model.modules.norm.init_args.norm_dict={norm_dict}"]
        )
        model = cli.model
        combined = _combined_graph(cli)
        plan = compile_plan(combined, Mode.FIT, sources={}, sinks=["loss.total"])
        with pytest.raises(ConfigError, match="does not exist"):
            model._bind(resolve_bind_schema([plan]))  # noqa: SLF001


class TestCoordCheck:
    def test_coord_check_data_and_plot(self, norm_dict, mup_override, tmp_path):
        df = coord_check(
            [DUMMY_CFG, str(mup_override)],
            widths=[8, 16],
            nsteps=2,
            set_overrides=[f"model.modules.norm.init_args.norm_dict={norm_dict}"],
        )
        assert list(df.columns) == ["width", "module", "t", "l1"]
        assert not df.empty
        # both widths recorded, and the apply_to module subtrees appear
        assert set(df["width"].unique()) == {8, 16}
        assert any(m.startswith("track_embed") for m in df["module"])
        assert any(m.startswith("encoder") for m in df["module"])
        out = tmp_path / "coord.png"
        plot_coord_data(df, out, title="test")
        assert out.is_file()


class TestSetupMupEntry:
    def test_setup_mup_forwards_to_mup_shapes(self, norm_dict, mup_override, tmp_path):
        rc = setup_mup([
            "-c", DUMMY_CFG, "-c", str(mup_override),
            "--set", f"model.modules.norm.init_args.norm_dict={norm_dict}",
            "--save-path", str(tmp_path / "via_setup.bsh"),
            "--base-width", "8", "--delta-width", "16",
        ])
        assert rc == 0
        assert (tmp_path / "via_setup.bsh").is_file()


class TestBuildModelAtWidths:
    def test_width_override_changes_apply_to_widths(self, norm_dict, mup_override):
        overrides = [f"model.modules.norm.init_args.norm_dict={norm_dict}"]
        m32 = build_model_at_widths([DUMMY_CFG, str(mup_override)], 32, overrides, bind=False)
        # the apply_to modules took the swept width
        assert m32.net["track_embed"].out_dim == 32
        assert m32.net["encoder"].dim == 32
