"""Unit tests for Split + GlobalAttentionPooling paths (mirror of pooling.py/split.py)."""

from __future__ import annotations

import torch

from salt.core.graph import (
    Executor,
    Mode,
    compile_plan,
    flatten_spec,
)
from salt.core.nn import (
    Concat,
    GlobalAttentionPooling,
    LossSum,
    Normaliser,
    StreamEmbed,
    bind_all,
    materialise_all,
    resolve_bind_schema,
)
from salt.core.nn.tasks import (
    ClassificationTaskModule,
)
from salt.tests._fixtures.gn2v2_fixture import (
    gn2v2_sources,
)
from salt.tests.unit.nn.conftest import B, T, fit_bundle


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


# encoder-less pooling (M5; DiPS/DeepSets — init_nets + pool_net, no encoder)


def _encoderless_modules(norm_dict):
    """A DiPS-shaped module dict: norm -> embed -> concat -> pool -> head -> loss."""
    dense = {"hidden_layers": [16], "activation": "ReLU"}
    modules = {
        "norm": Normaliser(norm_dict=norm_dict, streams=["jets", "tracks"], global_object="jets"),
        "track_embed": StreamEmbed(
            stream="tracks", out_dim=16, dense=dense, context=["normed.jets"]
        ),
        "concat": Concat(streams=["tracks"]),
        "pool": GlobalAttentionPooling(input="seq.x", out="pooled.global"),
        "jets_classification": ClassificationTaskModule(
            stream="jets",
            label="flavour_label",
            class_names=["bjets", "cjets", "ujets"],
            input="pooled.global",
            dense=dense,
        ),
        "loss": LossSum(),
    }
    for name, module in modules.items():
        module.name = name
    modules["loss"].narrow(LossSum.collect_loss_keys(modules))
    return modules


class TestEncoderlessPooling:
    def test_registers_require_is_optional(self):
        """`masks.registers` is optional so an encoder-less config plan-compiles."""
        pool = GlobalAttentionPooling(input="seq.x")
        pool.name = "pool"
        req = flatten_spec(pool.declare_io(Mode.FIT).requires)
        assert req["masks.registers"].optional
        assert not req["seq.mask"].optional

    def test_encoderless_plan_compiles_fit_and_test(self, norm_paths):
        """No encoder -> no `masks.registers` producer; FIT and TEST still compile."""
        modules = _encoderless_modules(norm_paths[0])
        fit = compile_plan(modules, Mode.FIT, sources=gn2v2_sources(), sinks=["loss.total"])
        test = compile_plan(
            modules,
            Mode.TEST,
            sources=gn2v2_sources(),
            sinks=["preds.jets.jets_classification"],
        )
        # the encoder is genuinely absent, but pooling still resolves
        assert "encoder" not in fit.module_names
        assert "pool" in fit.module_names and "pool" in test.module_names
        for plan in (fit, test):
            assert all(edge.key != "masks.registers" for edge in plan.edges), (
                "no register mask edge on the encoder-less path"
            )

    def test_encoderless_forward_runs(self, norm_paths):
        """Full FIT execution (debug: read-tracking + write-once) with no encoder."""
        modules = _encoderless_modules(norm_paths[0])
        plan = compile_plan(modules, Mode.FIT, sources=gn2v2_sources(), sinks=["loss.total"])
        bind_all(modules, resolve_bind_schema(plan))
        materialise_all(modules)
        b = Executor(plan).run(fit_bundle(), debug=True)
        assert b.get("pooled.global").shape == (B, 16)
        assert torch.isfinite(b.get("loss.total"))

    def test_encoderless_pool_equals_v1_no_registers(self, norm_paths):
        """The pooled vector == a direct v1 GAP call with the {"seq": seq.mask}"""
        modules = _encoderless_modules(norm_paths[0])
        plan = compile_plan(modules, Mode.FIT, sources=gn2v2_sources(), sinks=["loss.total"])
        bind_all(modules, resolve_bind_schema(plan))
        materialise_all(modules)
        # capture the seq.x / seq.mask the pool actually sees
        captured: dict[str, torch.Tensor] = {}
        executor = Executor(plan)
        b = executor.run(fit_bundle())
        captured["seq.x"] = b.get("seq.x")
        captured["seq.mask"] = b.get("seq.mask")
        v2_pooled = b.get("pooled.global")
        v1_pool = modules["pool"].pool_net  # the SAME composed v1 instance
        v1_pooled = v1_pool({"seq": captured["seq.x"]}, pad_mask={"seq": captured["seq.mask"]})
        assert torch.equal(v2_pooled, v1_pooled)
