"""Unit tests for the standalone config-constructed nn modules (plan 05, stage A2).

Covers, per module: construction from plain kwargs (no YAML anchors, no
``input_size`` arithmetic — design §2.3), `declare_io` correctness (keys,
kinds, mode gating), bind-time shape inference via `resolve_bind_schema`,
the Normaliser materialise lifecycle, the declarative class-weight source,
and full FIT/TEST executions through the M1 Executor in debug mode
(read-tracking + write-once + mutation detection on).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from salt.core.graph import Bundle, ConfigError, Executor, Mode, compile_plan, flatten_spec
from salt.core.nn import (
    BindError,
    Concat,
    GlobalAttentionPooling,
    LossSum,
    Normaliser,
    ResolvedSchema,
    Split,
    StreamEmbed,
    TransformerEncoder,
    bind_all,
    materialise_all,
    resolve_bind_schema,
)
from salt.core.nn.tasks import ClassificationTaskModule, VertexingTaskModule
from salt.tests.core.gn2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    build_test_gn2,
    make_gn2_batch,
    write_parity_norm_dict,
)
from salt.tests.core.gn2v2_fixture import (
    ORIGIN_CLASSES,
    build_gn2v2_modules,
    compile_gn2v2,
    make_gn2_labels,
)

B, T = 6, 10


@pytest.fixture
def norm_paths(tmp_path):
    """Write the parity norm/class dicts; return (norm_dict, class_dict) paths."""
    nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
    write_parity_norm_dict(nd, cd)
    return nd, cd


@pytest.fixture
def gn2v2(norm_paths):
    """Build, compile (FIT), bind, and materialise the small GN2v2."""
    modules = build_gn2v2_modules(norm_paths[0])
    plan = compile_gn2v2(modules, Mode.FIT)
    schema = resolve_bind_schema(plan)
    bind_all(modules, schema)
    materialise_all(modules)
    return modules, plan, schema


def fit_bundle(n_electrons: int = 0) -> Bundle:
    """Build a FIT-mode bundle from the deterministic v1 batch + labels."""
    inputs, masks = make_gn2_batch(B, T, n_electrons=n_electrons)
    labels = make_gn2_labels(B, T)
    b = Bundle()
    for stream, x in inputs.items():
        b.set(f"inputs.{stream}", x)
    for stream, m in masks.items():
        b.set(f"masks.{stream}", m)
    for stream, fields in labels.items():
        for name, val in fields.items():
            b.set(f"labels.{stream}.{name}", val)
    return b


# ---------------------------------------------------------------------------
# bind schema resolution
# ---------------------------------------------------------------------------


class TestResolvedSchema:
    def test_widths_resolve_through_the_graph(self, gn2v2):
        """Symbolic widths bind from source/config declarations (design §2.3)."""
        _, _, schema = gn2v2
        assert schema.width("inputs.jets") == len(JET_VARIABLES)
        assert schema.width("inputs.tracks") == len(TRACK_VARIABLES)
        # through the Normaliser's shared symbol
        assert schema.width("normed.tracks") == len(TRACK_VARIABLES)
        # config-concrete produces
        assert schema.width("embed.tracks") == 16
        assert schema.width("encoded.seq") == 16
        # through Split / Pooling instance symbols
        assert schema.width("encoded.tracks") == 16
        assert schema.width("pooled.global") == 16

    def test_fields_come_from_the_source_declaration(self, gn2v2):
        _, _, schema = gn2v2
        assert schema.fields_of("inputs.tracks") == tuple(TRACK_VARIABLES)
        with pytest.raises(BindError, match="no declared fields"):
            schema.fields_of("encoded.seq")

    def test_unknown_width_raises_with_suggestions(self, gn2v2):
        _, _, schema = gn2v2
        with pytest.raises(BindError, match="encoded.tracks"):
            schema.width("encoded.trakcs")

    def test_meta_and_scalar_keys_have_no_width(self, gn2v2):
        _, _, schema = gn2v2
        for key in ("seq.layout", "loss.total"):
            with pytest.raises(BindError):
                schema.width(key)

    def test_conflicting_widths_raise(self):
        from salt.core.nn.bind import _DimBindings

        dims = _DimBindings()
        dims.bind("F:x", 3, "here")
        with pytest.raises(BindError, match="conflicting widths"):
            dims.bind("F:x", 4, "there")


# ---------------------------------------------------------------------------
# construction + declare_io per module
# ---------------------------------------------------------------------------


class TestNormaliser:
    def test_init_does_no_file_io(self, tmp_path):
        """__init__ records the path only — the file need not exist (design §2.3)."""
        norm = Normaliser(norm_dict=tmp_path / "absent.yaml", streams=["tracks"])
        io = norm.declare_io(Mode.FIT)
        assert set(flatten_spec(io.requires)) == {"inputs.tracks"}
        assert set(flatten_spec(io.produces)) == {"normed.tracks"}

    def test_global_object_is_rank_two(self):
        norm = Normaliser(norm_dict="x.yaml", streams=["jets", "tracks"], global_object="jets")
        norm.name = "norm"
        flat = flatten_spec(norm.declare_io(Mode.FIT).requires)
        assert len(flat["inputs.jets"].shape) == 2
        assert len(flat["inputs.tracks"].shape) == 3

    def test_config_errors(self):
        with pytest.raises(ConfigError, match="non-empty"):
            Normaliser(norm_dict="x.yaml", streams=[])
        with pytest.raises(ConfigError, match="duplicate"):
            Normaliser(norm_dict="x.yaml", streams=["a", "a"])
        with pytest.raises(ConfigError, match="global_object"):
            Normaliser(norm_dict="x.yaml", streams=["a"], global_object="b")

    def test_materialise_fills_buffers_to_v1_values(self, norm_paths, gn2v2):
        """Buffer values equal v1 InputNorm's from the same norm dict."""
        modules, _, _ = gn2v2
        wrapper = build_test_gn2(norm_paths[0].parent)
        norm = modules["norm"]
        assert torch.equal(norm.means_tracks, wrapper.norm.tracks_means)
        assert torch.equal(norm.stds_tracks, wrapper.norm.tracks_stds)
        assert torch.equal(norm.means_jets, wrapper.norm.jets_means)
        assert bool(norm.materialised)

    def test_forward_before_materialise_raises(self, norm_paths):
        modules = build_gn2v2_modules(norm_paths[0])
        plan = compile_gn2v2(modules, Mode.FIT)
        bind_all(modules, resolve_bind_schema(plan))
        with pytest.raises(RuntimeError, match="materialise"):
            modules["norm"](fit_bundle(), Mode.FIT)

    def test_forward_produces_new_keys_and_never_mutates(self, gn2v2):
        modules, _, _ = gn2v2
        b = fit_bundle()
        before = b.get("inputs.tracks").clone()
        out = modules["norm"](b, Mode.FIT)
        assert set(out) == {"normed.jets", "normed.tracks"}
        assert torch.equal(b.get("inputs.tracks"), before)
        expected = (before - modules["norm"].means_tracks) / modules["norm"].stds_tracks
        assert torch.equal(out["normed.tracks"], expected)

    def test_materialise_missing_variable_raises(self, tmp_path, norm_paths):
        import yaml

        with open(norm_paths[0]) as fh:
            nd = yaml.safe_load(fh)
        del nd["tracks"]["d0"]
        bad = tmp_path / "bad_norm.yaml"
        with open(bad, "w") as fh:
            yaml.dump(nd, fh)
        modules = build_gn2v2_modules(bad)
        bind_all(modules, resolve_bind_schema(compile_gn2v2(modules, Mode.FIT)))
        with pytest.raises(ValueError, match="d0"):
            modules["norm"].materialise()


class TestStreamEmbed:
    def test_width_keys_rejected(self):
        with pytest.raises(ConfigError, match="input_size"):
            StreamEmbed(stream="tracks", out_dim=8, dense={"input_size": 21})

    def test_bind_infers_input_plus_context_width(self, gn2v2):
        modules, _, _ = gn2v2
        embed = modules["track_embed"]
        # 19 track vars + 2 jet context vars — initnet.py:54-59 inference
        assert embed.net.input_size == len(TRACK_VARIABLES) + len(JET_VARIABLES)
        assert embed.net.output_size == 16

    def test_declare_io(self):
        embed = StreamEmbed(stream="tracks", out_dim=8, context=["normed.jets"])
        embed.name = "track_embed"
        io = embed.declare_io(Mode.FIT)
        assert set(flatten_spec(io.requires)) == {"normed.tracks", "normed.jets"}
        assert set(flatten_spec(io.produces)) == {"embed.tracks"}
        assert flatten_spec(io.produces)["embed.tracks"].shape[-1] == 8

    def test_context_prepended_in_v1_order(self, gn2v2):
        """Single GN2 context entry reproduces v1's [global, stream] layout."""
        modules, _, _ = gn2v2
        b = fit_bundle()
        normed = modules["norm"](b, Mode.FIT)
        for key, val in normed.items():
            b.set(key, val)
        out = modules["track_embed"](b, Mode.FIT)
        x = torch.cat(
            [
                b.get("normed.jets").unsqueeze(1).expand(B, T, -1),
                b.get("normed.tracks"),
            ],
            dim=-1,
        )
        assert torch.equal(out["embed.tracks"], modules["track_embed"].net(x))


class TestConcat:
    def test_registers_rejected_in_m2(self):
        with pytest.raises(ConfigError, match="registers are internal to TransformerEncoder"):
            Concat(streams=["tracks"], registers=8)

    def test_layout_and_order(self):
        concat = Concat(streams=["tracks", "electrons"])
        concat.name = "concat"
        b = Bundle()
        b.set("embed.tracks", torch.ones(2, 3, 4))
        b.set("embed.electrons", 2 * torch.ones(2, 2, 4))
        b.set("masks.tracks", torch.zeros(2, 3, dtype=torch.bool))
        b.set("masks.electrons", torch.ones(2, 2, dtype=torch.bool))
        out = concat(b, Mode.FIT)
        assert out["seq.x"].shape == (2, 5, 4)
        assert out["seq.layout"] == {"tracks": (0, 3), "electrons": (3, 5)}
        assert torch.equal(out["seq.x"][:, :3], b.get("embed.tracks"))
        assert out["seq.mask"].tolist() == [[False] * 3 + [True] * 2] * 2


class TestTransformerEncoder:
    def test_attention_config_required(self):
        with pytest.raises(ConfigError, match="num_heads"):
            TransformerEncoder(dim=16, num_layers=1, attention={})

    def test_declare_io_concrete_widths(self):
        enc = TransformerEncoder(dim=16, num_layers=1, out_dim=8, attention={"num_heads": 2})
        enc.name = "encoder"
        io = enc.declare_io(Mode.FIT)
        assert flatten_spec(io.requires)["seq.x"].shape[-1] == 16
        produces = flatten_spec(io.produces)
        assert produces["encoded.seq"].shape[-1] == 8
        assert produces["masks.registers"].shape == ("B", 1)
        assert produces["masks.registers"].kind == "pad_mask"

    def test_registers_internal_and_mask_published(self, gn2v2):
        modules, plan, _ = gn2v2
        b = Executor(plan).run(fit_bundle())
        # register row appended after the T track positions
        assert b.get("encoded.seq").shape == (B, T + 1, 16)
        assert b.get("masks.registers").shape == (B, 1)
        assert not b.get("masks.registers").any()


class TestSplitAndPooling:
    def test_split_slices_by_layout(self, gn2v2):
        modules, plan, _ = gn2v2
        b = Executor(plan).run(fit_bundle())
        assert b.get("encoded.tracks").shape == (B, T, 16)
        assert torch.equal(b.get("encoded.tracks"), b.get("encoded.seq")[:, :T])

    def test_pooling_binds_gate_width(self, gn2v2):
        modules, _, _ = gn2v2
        assert modules["pool"].pool_net.gate_nn.in_features == 16

    def test_pooled_shape(self, gn2v2):
        _, plan, _ = gn2v2
        b = Executor(plan).run(fit_bundle())
        assert b.get("pooled.global").shape == (B, 16)


class TestLossSum:
    def test_unnarrowed_declare_raises(self):
        loss = LossSum()
        loss.name = "loss"
        with pytest.raises(ConfigError, match="framework"):
            loss.declare_io(Mode.FIT)

    def test_explicit_losses_config(self):
        loss = LossSum(losses=["jets_classification", "losses.track_origin"])
        loss.name = "loss"
        io = loss.declare_io(Mode.FIT)
        assert set(flatten_spec(io.requires)) == {
            "losses.jets_classification",
            "losses.track_origin",
        }
        assert set(flatten_spec(io.produces)) == {"loss.total"}

    def test_inactive_outside_training(self):
        loss = LossSum(losses=["a"])
        loss.name = "loss"
        io = loss.declare_io(Mode.TEST)
        assert not flatten_spec(io.requires) and not flatten_spec(io.produces)

    def test_unknown_weight_key_rejected(self):
        with pytest.raises(ConfigError, match="unknown loss keys"):
            LossSum(losses=["a"], weights={"b": 2.0})

    def test_weighted_sum(self):
        loss = LossSum(losses=["a", "b"], weights={"b": 2.0})
        loss.name = "loss"
        b = Bundle()
        b.set("losses.a", torch.tensor(1.0))
        b.set("losses.b", torch.tensor(3.0))
        out = loss(b, Mode.FIT)
        assert out["loss.total"].item() == pytest.approx(7.0)

    def test_collect_loss_keys(self, gn2v2):
        modules, _, _ = gn2v2
        assert LossSum.collect_loss_keys(modules) == (
            "losses.jets_classification",
            "losses.track_origin",
            "losses.track_vertexing",
        )

    def test_double_narrow_rejected(self, gn2v2):
        modules, _, _ = gn2v2
        with pytest.raises(ConfigError, match="already fixed"):
            modules["loss"].narrow(["losses.x"])


# ---------------------------------------------------------------------------
# task modules
# ---------------------------------------------------------------------------


class TestClassificationTaskModule:
    def test_class_names_required(self):
        with pytest.raises(ConfigError, match="class_names"):
            ClassificationTaskModule(stream="jets", label="flavour_label", class_names=[])

    def test_sequence_inference(self):
        seq_task = ClassificationTaskModule(stream="tracks", label="l", class_names=["a", "b"])
        glob_task = ClassificationTaskModule(
            stream="jets", label="l", class_names=["a", "b"], input="pooled.global"
        )
        assert seq_task.sequence and not glob_task.sequence

    def test_declare_io_mode_gating(self):
        task = ClassificationTaskModule(
            stream="tracks", label="lbl", class_names=["a", "b"], context="pooled.global"
        )
        task.name = "track_origin"
        io = task.declare_io(Mode.FIT)
        req = flatten_spec(io.requires)
        assert set(req) == {
            "encoded.tracks",
            "pooled.global",
            "masks.tracks",
            "labels.tracks.lbl",
        }
        assert req["labels.tracks.lbl"].kind == "label"
        assert req["labels.tracks.lbl"].modes == Mode.TRAINING
        assert req["masks.tracks"].kind == "pad_mask"
        produces = flatten_spec(io.produces)
        assert produces["preds.tracks.track_origin"].modes == Mode.ALL
        assert produces["losses.track_origin"].modes == Mode.TRAINING
        assert produces["losses.track_origin"].kind == "loss"

    def test_bind_builds_v1_head_with_inferred_widths(self, gn2v2):
        modules, _, _ = gn2v2
        head = modules["track_origin"].task
        assert head.net.input_size == 16
        assert head.net.context_size == 16
        assert head.net.output_size == len(ORIGIN_CLASSES)
        assert head.loss.ignore_index == -1  # v1 contract (task.py:123-124)

    def test_weight_source_literal_conflict(self):
        with pytest.raises(ConfigError, match="already specified"):
            ClassificationTaskModule(
                stream="tracks",
                label="l",
                class_names=["a", "b"],
                loss={"class_path": "torch.nn.CrossEntropyLoss", "init_args": {"weight": [1, 2]}},
                weight_source={"from_class_dict": "cd.yaml"},
            )

    def test_weight_source_materialise(self, norm_paths):
        """Class weights fill the CE buffer at materialise (design §3.3)."""
        modules = build_gn2v2_modules(norm_paths[0], class_dict=norm_paths[1])
        bind_all(modules, resolve_bind_schema(compile_gn2v2(modules, Mode.FIT)))
        head = modules["track_origin"].task
        assert torch.equal(head.loss.weight, torch.ones(8))  # bind: allocated, identity
        materialise_all(modules)
        expected = torch.tensor([4.2, 73.7, 1.0, 17.5, 12.3, 12.5, 141.7, 22.3])
        assert torch.allclose(head.loss.weight, expected)
        # the buffer is in the state dict (resume inherits it, cli.py:253-267)
        assert "task.loss.weight" in modules["track_origin"].state_dict()

    def test_weight_source_length_mismatch_raises(self, norm_paths):
        """The fixture class dict has 4 flavour weights but 3 class names."""
        task = ClassificationTaskModule(
            stream="jets",
            label="flavour_label",
            class_names=["bjets", "cjets", "ujets"],
            input="pooled.global",
            weight_source={"from_class_dict": str(norm_paths[1])},
        )
        task.name = "jets_classification"
        task.bind(ResolvedSchema(widths={"pooled.global": 16}))
        with pytest.raises(ValueError, match="3 class_names"):
            task.materialise()


class TestVertexingTaskModule:
    def test_label_must_contain_vertex_index(self):
        with pytest.raises(ConfigError, match="VertexIndex"):
            VertexingTaskModule(stream="tracks", label="vtx", origin_label="o")

    def test_unknown_origin_weighting_key(self):
        with pytest.raises(ConfigError, match="origin_weighting"):
            VertexingTaskModule(
                stream="tracks",
                label="ftagTruthVertexIndex",
                origin_label="o",
                origin_weighting={"heavy": [3], "bogus": [1]},
            )

    def test_declares_origin_label_dependency(self):
        task = VertexingTaskModule(
            stream="tracks", label="ftagTruthVertexIndex", origin_label="ftagTruthOriginLabel"
        )
        task.name = "track_vertexing"
        req = flatten_spec(task.declare_io(Mode.FIT).requires)
        assert "labels.tracks.ftagTruthOriginLabel" in req
        assert "labels.tracks.ftagTruthVertexIndex" in req

    def test_bind_pair_widths_and_reduction_check(self, gn2v2):
        modules, _, _ = gn2v2
        head = modules["track_vertexing"].task
        assert head.net.input_size == 2 * 16  # pair concat (task.py:894-896)
        assert head.net.context_size == 16
        task = VertexingTaskModule(
            stream="tracks",
            label="ftagTruthVertexIndex",
            origin_label="o",
            loss={"class_path": "torch.nn.BCEWithLogitsLoss", "init_args": {"reduction": "mean"}},
        )
        task.name = "v"
        with pytest.raises(ConfigError, match="reduction"):
            task.bind(ResolvedSchema(widths={"encoded.tracks": 16}))

    def test_default_origin_weighting_matches_v1(self, gn2v2):
        """Configured default ids reproduce v1's hardcoded weights bitwise."""
        from salt.models.task import VertexingTask as V1

        modules, _, _ = gn2v2
        head = modules["track_vertexing"].task
        labels = torch.tensor([[0, 1, 3, 4, 5, 2]])
        n = labels.shape[1]
        adjmat = ~torch.eye(n, dtype=torch.bool).unsqueeze(0)
        v1_weights = V1.get_weights(head, labels, adjmat)
        v2_weights = head.get_weights(labels, adjmat)
        assert torch.equal(v1_weights, v2_weights)


# ---------------------------------------------------------------------------
# full plan execution (FIT + TEST), debug mode on
# ---------------------------------------------------------------------------


class TestGn2V2Execution:
    def test_fit_plan_runs_in_debug_mode(self, gn2v2):
        """Debug execution: read-tracking, write-once, and mutation checks all pass."""
        modules, plan, _ = gn2v2
        b = Executor(plan).run(fit_bundle(), debug=True)
        total = b.get("loss.total")
        assert total.shape == ()
        assert torch.isfinite(total)
        expected = (
            b.get("losses.jets_classification")
            + b.get("losses.track_origin")
            + b.get("losses.track_vertexing")
        )
        assert torch.equal(total, expected)

    def test_fit_preds_are_raw_logits(self, gn2v2):
        modules, plan, _ = gn2v2
        b = Executor(plan).run(fit_bundle())
        logits = b.get("preds.jets.jets_classification")
        assert logits.shape == (B, 3)
        assert not torch.allclose(logits.sum(-1), torch.ones(B))  # not softmaxed

    def test_test_plan_converts_predictions(self, gn2v2):
        """TEST preds are converted physical values (design §3.3)."""
        modules, _, _ = gn2v2
        plan = compile_gn2v2(modules, Mode.TEST)
        assert "loss" not in plan.module_names  # LossSum inactive outside TRAINING
        b = Bundle()
        inputs, masks = make_gn2_batch(B, T)
        for stream, x in inputs.items():
            b.set(f"inputs.{stream}", x)
        b.set("masks.tracks", masks["tracks"])
        with torch.no_grad():
            b = Executor(plan).run(b, debug=True)
        probs = b.get("preds.jets.jets_classification")
        assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-6)
        track_probs = b.get("preds.tracks.track_origin")
        valid = ~masks["tracks"]
        assert torch.allclose(track_probs[valid].sum(-1), torch.ones(int(valid.sum())), atol=1e-6)
        # vertexing TEST output: per-node assignments unflattened to the
        # batch shape, padded positions -inf (task.py:985-986,1003-1035)
        assert b.get("preds.tracks.track_vertexing").shape == (B, T, 1)

    def test_onnx_plan_keeps_raw_vertexing_scores(self, gn2v2):
        """ONNX vertexing publishes raw edge scores for the export reduce (§3.3)."""
        modules, _, _ = gn2v2
        plan = compile_gn2v2(modules, Mode.ONNX)
        b = Bundle()
        inputs, masks = make_gn2_batch(B, T)
        for stream, x in inputs.items():
            b.set(f"inputs.{stream}", x)
        b.set("masks.tracks", masks["tracks"])
        with torch.no_grad():
            b = Executor(plan).run(b)
        scores = b.get("preds.tracks.track_vertexing")
        assert scores.ndim == 2
        assert scores.shape[1] == 1  # [E, 1] raw edge scores

    def test_all_modules_are_nn_modules(self, gn2v2):
        """The module dict must be nn.ModuleDict-compatible (SaltModule, design §3.4)."""
        modules, _, _ = gn2v2
        assert all(isinstance(m, nn.Module) for m in modules.values())
        nn.ModuleDict(modules)  # must not raise


class TestOriginWeightingConfig:
    def test_name_based_origin_weighting_rejected_with_pointer(self):
        # design §5.1 uses class NAMES; M2 accepts integer ids only — the
        # error must say so instead of a bare int() ValueError (stage-E fix)
        with pytest.raises(ConfigError, match="INTEGER origin ids") as excinfo:
            VertexingTaskModule(
                stream="tracks",
                label="ftagTruthVertexIndex",
                origin_label="ftagTruthOriginLabel",
                origin_weighting={"heavy": ["FromB", "FromBC", "FromC"], "fake": ["Fake"]},
            )
        assert "M3" in str(excinfo.value)


class TestNoIOGuard:
    """Design §2.3 CI guard: declare_io and bind run under a no-I/O trap.

    The rule itself ("declare_io/bind touch no files") was only ever verified
    manually; this pins it in CI over the shipped GN2v2 module set (stage-E
    design-compliance fix). materialise() stays the ONLY file-touching hook —
    proven by releasing the trap and pointing it at a nonexistent norm dict.
    """

    def test_declare_and_bind_are_file_free(self, monkeypatch):
        # construction is config capture only — safe to build under the trap
        def _forbid(*args, **kwargs):
            raise AssertionError(
                f"file I/O during declare_io/bind (design §2.3): open({args!r})"
            )

        monkeypatch.setattr("builtins.open", _forbid)
        monkeypatch.setattr("h5py.File", _forbid)
        monkeypatch.setattr("pathlib.Path.open", _forbid)
        modules = build_gn2v2_modules("/nonexistent/norm_dict.yaml")
        for mode in (Mode.FIT, Mode.VAL, Mode.TEST, Mode.ONNX):
            for module in modules.values():
                module.declare_io(mode)
        plans = [compile_gn2v2(modules, mode) for mode in (Mode.FIT, Mode.TEST)]
        bind_all(modules, resolve_bind_schema(plans))

        # release the trap: materialise IS the sanctioned file hook and must
        # be the first thing that touches the (nonexistent) norm dict
        monkeypatch.undo()
        with pytest.raises(FileNotFoundError):
            materialise_all(modules)
