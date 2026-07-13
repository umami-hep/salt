"""v1 -> v2 state-dict transfer tests, driven by a synthetic v1 GN2 state dict
built from the frozen key/shape schema (`make_v1_gn2_state_dict`).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from salt.core.graph import Bundle, Executor, Mode
from salt.core.nn import bind_all, map_v1_state_dict, resolve_bind_schema
from salt.tests._fixtures.gn2v2_fixture import (
    build_gn2v2_modules,
    compile_gn2v2,
    make_gn2_batch,
    make_gn2_labels,
    make_v1_gn2_state_dict,
)

B, T = 6, 10
TASKS = [
    ("jets", "jets_classification"),
    ("tracks", "track_origin"),
    ("tracks", "track_vertexing"),
]


@pytest.fixture
def norm_dict(tmp_path):
    from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict

    nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
    write_parity_norm_dict(nd, cd)
    return nd


@pytest.fixture
def transferred(norm_dict):
    """Synthetic v1 state dict + v2 modules with transferred weights (strict load) + FIT plan."""
    v1_sd = make_v1_gn2_state_dict()
    modules = build_gn2v2_modules(norm_dict)
    plan = compile_gn2v2(modules, Mode.FIT)
    bind_all(modules, resolve_bind_schema(plan))
    # NOTE: no materialise() — transfer is the checkpoint-load path
    # (design §2.3: values arrive via the state_dict, incl. the
    # `materialised` flag).
    mapped = map_v1_state_dict(v1_sd, modules)
    holder = nn.ModuleDict(modules)
    holder.load_state_dict(mapped, strict=True)
    holder.eval()
    return v1_sd, modules, plan


def v2_fit_bundle() -> Bundle:
    """Bundle with the deterministic batch + labels."""
    inputs, masks = make_gn2_batch(B, T)
    labels = make_gn2_labels(B, T)
    b = Bundle()
    for stream, x in inputs.items():
        b.set(f"inputs.{stream}", x.clone())
    for stream, m in masks.items():
        b.set(f"masks.{stream}", m.clone())
    for stream, fields in labels.items():
        for name, val in fields.items():
            b.set(f"labels.{stream}.{name}", val.clone())
    return b


class TestStateDictMapping:
    def test_strict_load_covers_every_key(self, norm_dict):
        """The mapping is total: strict=True load succeeds in both directions."""
        v1_sd = make_v1_gn2_state_dict()
        modules = build_gn2v2_modules(norm_dict)
        bind_all(modules, resolve_bind_schema(compile_gn2v2(modules, Mode.FIT)))
        mapped = map_v1_state_dict(v1_sd, modules)
        # flattened target layout: task heads carry net/loss directly
        # (no `.task.` segment) and pooling carries gate_nn directly
        # (no `.pool_net.` segment)
        assert "jets_classification.net.net.0.weight" in mapped
        assert "track_vertexing.net.net.0.weight" in mapped
        assert "pool.gate_nn.weight" in mapped
        assert "pool.gate_nn.bias" in mapped
        assert not any(".task." in key or ".pool_net." in key for key in mapped)
        holder = nn.ModuleDict(modules)
        missing = set(holder.state_dict()) - set(mapped)
        extra = set(mapped) - set(holder.state_dict())
        assert not missing and not extra
        holder.load_state_dict(mapped, strict=True)

    def test_norm_buffers_and_flag(self, transferred):
        v1_sd, modules, _ = transferred
        norm = modules["norm"]
        assert torch.equal(norm.means_tracks, v1_sd["norm.tracks_means"])
        assert torch.equal(norm.stds_jets, v1_sd["norm.jets_stds"])
        assert bool(norm.materialised)

    def test_embed_and_head_weights_transferred(self, transferred):
        v1_sd, modules, _ = transferred
        v2_dense = modules["track_embed"].net
        assert torch.equal(v1_sd["model.init_nets.0.net.net.0.weight"], v2_dense.net[0].weight)
        v2_task = modules["track_vertexing"]  # v1 model.tasks index 2
        assert torch.equal(v1_sd["model.tasks.2.net.net.0.weight"], v2_task.net.net[0].weight)
        # v1 model.pool_net.* lands on the pooling module's own gate layer
        assert torch.equal(v1_sd["model.pool_net.gate_nn.weight"], modules["pool"].gate_nn.weight)
        assert torch.equal(v1_sd["model.pool_net.gate_nn.bias"], modules["pool"].gate_nn.bias)

    def test_unknown_v1_key_is_an_error(self, transferred):
        """Nothing is dropped silently — unmapped v1 keys raise."""
        v1_sd, modules, _ = transferred
        sd = dict(v1_sd)
        sd["model.mystery.weight"] = torch.zeros(1)
        with pytest.raises(ValueError, match="model.mystery.weight"):
            map_v1_state_dict(sd, modules)

    def test_missing_stream_embed_is_an_error(self, transferred):
        v1_sd, modules, _ = transferred
        incomplete = {k: m for k, m in modules.items() if k != "track_embed"}
        with pytest.raises(ValueError, match="StreamEmbed"):
            map_v1_state_dict(v1_sd, incomplete)

    def test_losses_require_grad_in_train_mode(self, norm_dict):
        """The FIT path is trainable after transfer: loss.total carries grad everywhere."""
        v1_sd = make_v1_gn2_state_dict()
        modules = build_gn2v2_modules(norm_dict)
        plan = compile_gn2v2(modules, Mode.FIT)
        bind_all(modules, resolve_bind_schema(plan))
        holder = nn.ModuleDict(modules)
        holder.load_state_dict(map_v1_state_dict(v1_sd, modules), strict=True)
        b = Executor(plan).run(v2_fit_bundle())
        b.get("loss.total").backward()
        for name in ("track_embed", "encoder", "pool", "jets_classification"):
            grads = [p.grad for p in modules[name].parameters() if p.grad is not None]
            assert grads, f"no gradients reached module {name!r}"
