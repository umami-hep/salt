"""v1 -> v2 state-dict transfer + forward comparison (plan 05, stage A2; feeds gate G3).

Strategy: build the small v1 GN2 (`gn2_fixture`), build the EQUIVALENT v2
module dict from plain config (`gn2v2_fixture`), transfer weights via
`map_v1_state_dict` (strict load — every parameter/buffer accounted for),
then run both sides on the same deterministic batch WITH labels and compare
predictions and per-task losses.

PASS criterion is ``<= 1e-6`` absolute, NOT bitwise: the v2 production task
path slices per-stream tensors via `Split` before the heads (design §3.3),
where v1 hands the heads the full register-augmented sequence and lets
``input_name_mask`` slice internally (task.py:201-204). The selected values
are identical, but the slice is materialised by a different op
(narrow/slice vs boolean advanced indexing), and downstream GEMMs run on
separately-materialised buffers — mathematically equal, with no bitwise
guarantee across BLAS paths. The M2 gates therefore use 1e-6/curve criteria
on this path (plan 05 risk 3); the BITWISE guarantee remains the job of the
plan-04 parity gate over the v1-wrapping modules.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from salt.core.graph import Bundle, Executor, Mode
from salt.core.nn import bind_all, map_v1_state_dict, resolve_bind_schema
from salt.tests.core.gn2_fixture import build_test_gn2, make_gn2_batch
from salt.tests.core.gn2v2_fixture import (
    build_gn2v2_modules,
    compile_gn2v2,
    make_gn2_labels,
)

B, T = 6, 10
ATOL = 1e-6
TASKS = [
    ("jets", "jets_classification"),
    ("tracks", "track_origin"),
    ("tracks", "track_vertexing"),
]


@pytest.fixture
def transferred(tmp_path):
    """v1 wrapper + v2 modules with transferred weights (strict load) + FIT plan."""
    wrapper = build_test_gn2(tmp_path)
    modules = build_gn2v2_modules(tmp_path / "norm_dict.yaml")
    plan = compile_gn2v2(modules, Mode.FIT)
    bind_all(modules, resolve_bind_schema(plan))
    # NOTE: no materialise() — transfer is the checkpoint-load path
    # (design §2.3: values arrive via the state_dict, incl. the
    # `materialised` flag).
    mapped = map_v1_state_dict(wrapper.state_dict(), modules)
    holder = nn.ModuleDict(modules)
    holder.load_state_dict(mapped, strict=True)
    holder.eval()
    return wrapper, modules, plan


def v2_fit_bundle() -> Bundle:
    """Bundle with the deterministic batch + labels (clones — v1 mutates its dicts)."""
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


def v1_fit_forward(wrapper):
    """Run the v1 training-path forward (with labels) on cloned dicts.

    Returns
    -------
    tuple[dict, dict]
        ``(preds, per-task loss dict)`` as `ModelWrapper.forward` returns
        them (loss keyed by task name, each already task-weighted).
    """
    inputs, masks = make_gn2_batch(B, T)
    labels = make_gn2_labels(B, T)
    with torch.no_grad():
        preds, loss = wrapper(
            {k: v.clone() for k, v in inputs.items()},
            {k: v.clone() for k, v in masks.items()},
            {s: {k: v.clone() for k, v in d.items()} for s, d in labels.items()},
        )
    return preds, loss


class TestStateDictMapping:
    def test_strict_load_covers_every_key(self, tmp_path):
        """The mapping is total: strict=True load succeeds in both directions."""
        wrapper = build_test_gn2(tmp_path)
        modules = build_gn2v2_modules(tmp_path / "norm_dict.yaml")
        bind_all(modules, resolve_bind_schema(compile_gn2v2(modules, Mode.FIT)))
        mapped = map_v1_state_dict(wrapper.state_dict(), modules)
        holder = nn.ModuleDict(modules)
        missing = set(holder.state_dict()) - set(mapped)
        extra = set(mapped) - set(holder.state_dict())
        assert not missing and not extra
        holder.load_state_dict(mapped, strict=True)

    def test_norm_buffers_and_flag(self, transferred):
        wrapper, modules, _ = transferred
        norm = modules["norm"]
        assert torch.equal(norm.means_tracks, wrapper.norm.tracks_means)
        assert torch.equal(norm.stds_jets, wrapper.norm.jets_stds)
        assert bool(norm.materialised)

    def test_embed_and_head_weights_transferred(self, transferred):
        wrapper, modules, _ = transferred
        v1_dense = wrapper.model.init_nets[0].net
        v2_dense = modules["track_embed"].net
        assert torch.equal(v1_dense.net[0].weight, v2_dense.net[0].weight)
        v1_task = wrapper.model.tasks[2]  # track_vertexing
        v2_task = modules["track_vertexing"].task
        assert torch.equal(v1_task.net.net[0].weight, v2_task.net.net[0].weight)

    def test_unknown_v1_key_is_an_error(self, transferred):
        """Nothing is dropped silently — unmapped v1 keys raise."""
        wrapper, modules, _ = transferred
        sd = dict(wrapper.state_dict())
        sd["model.mystery.weight"] = torch.zeros(1)
        with pytest.raises(ValueError, match="model.mystery.weight"):
            map_v1_state_dict(sd, modules)

    def test_missing_stream_embed_is_an_error(self, transferred):
        wrapper, modules, _ = transferred
        incomplete = {k: m for k, m in modules.items() if k != "track_embed"}
        with pytest.raises(ValueError, match="StreamEmbed"):
            map_v1_state_dict(wrapper.state_dict(), incomplete)


class TestForwardEquivalence:
    """v2 (transferred weights, Split path) vs v1, <= 1e-6 — see module docstring."""

    def test_fit_preds_match(self, transferred):
        wrapper, _, plan = transferred
        v1_preds, _ = v1_fit_forward(wrapper)
        with torch.no_grad():
            b = Executor(plan).run(v2_fit_bundle(), debug=True)
        for stream, task in TASKS:
            v1_out = v1_preds[stream][task]
            v2_out = b.get(f"preds.{stream}.{task}")
            assert v2_out.shape == v1_out.shape, (stream, task)
            diff = (v2_out - v1_out).abs().max().item()
            assert diff <= ATOL, f"{stream}.{task}: max |diff| {diff} > {ATOL}"

    def test_fit_losses_match(self, transferred):
        wrapper, _, plan = transferred
        _, v1_loss = v1_fit_forward(wrapper)
        with torch.no_grad():
            b = Executor(plan).run(v2_fit_bundle())
        for _, task in TASKS:
            diff = (b.get(f"losses.{task}") - v1_loss[task]).abs().item()
            assert diff <= ATOL, f"{task}: |loss diff| {diff} > {ATOL}"
        total = sum(v1_loss.values())
        assert (b.get("loss.total") - total).abs().item() <= ATOL

    def test_losses_require_grad_in_train_mode(self, tmp_path):
        """The FIT path is trainable: loss.total carries grad back to all modules."""
        wrapper = build_test_gn2(tmp_path)
        modules = build_gn2v2_modules(tmp_path / "norm_dict.yaml")
        plan = compile_gn2v2(modules, Mode.FIT)
        bind_all(modules, resolve_bind_schema(plan))
        holder = nn.ModuleDict(modules)
        holder.load_state_dict(map_v1_state_dict(wrapper.state_dict(), modules), strict=True)
        b = Executor(plan).run(v2_fit_bundle())
        b.get("loss.total").backward()
        for name in ("track_embed", "encoder", "pool", "jets_classification"):
            grads = [p.grad for p in modules[name].parameters() if p.grad is not None]
            assert grads, f"no gradients reached module {name!r}"

    def test_without_transfer_outputs_differ(self, tmp_path):
        """Negative control: fresh v2 init does NOT match v1 (the comparison has teeth)."""
        wrapper = build_test_gn2(tmp_path)
        modules = build_gn2v2_modules(tmp_path / "norm_dict.yaml")
        plan = compile_gn2v2(modules, Mode.FIT)
        bind_all(modules, resolve_bind_schema(plan))
        from salt.core.nn import materialise_all

        materialise_all(modules)  # norm values, but fresh random weights
        v1_preds, _ = v1_fit_forward(wrapper)
        with torch.no_grad():
            b = Executor(plan).run(v2_fit_bundle())
        diff = (
            (b.get("preds.jets.jets_classification") - v1_preds["jets"]["jets_classification"])
            .abs()
            .max()
            .item()
        )
        assert diff > ATOL
