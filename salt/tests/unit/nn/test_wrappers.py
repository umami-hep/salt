"""Per-wrapper unit tests for the GN2 forward-parity wrappers (plan 04, stage 2).

Strategy: build the small v1 GN2 from the stage-1 recipe, wrap its live
submodules via `from_v1`, compile a TEST plan over the dataset boundary
(sources = ``inputs.*`` / ``masks.*``, sinks = the three ``preds.*`` leaves),
execute it ONCE through the M1 Executor in debug mode (read-tracking +
mutation detection on), and then assert — per wrapper — that the produced
bundle leaf is BITWISE equal to the corresponding v1 intermediate computed
manually. Per-stage checks localise wiring bugs that an end-to-end diff
would only flag globally.

PASS criterion is bitwise (``torch.equal``): both sides share the same
nn.Module instances and the same op order (stage-1 probe confirmed CPU
bitwise repeatability), so any tolerance would only mask wiring bugs.
"""

from __future__ import annotations

import pytest
import torch

from salt.core.graph import Bundle, Executor, Mode, compile_plan
from salt.core.nn import from_v1, v1_sinks, v1_sources
from salt.core.nn.wrappers import Split
from salt.tests._fixtures.gn2_fixture import build_test_gn2, make_gn2_batch, v1_forward

OUT_DIM = 16
N_TRACKS = 10
EXPECTED_SINKS = [
    "preds.jets.jets_classification",
    "preds.tracks.track_origin",
    "preds.tracks.track_vertexing",
]


# ---------------------------------------------------------------------------
# fixtures (module-scoped: one v1 model, one batch, one executed v2 bundle)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def v1(tmp_path_factory):
    return build_test_gn2(tmp_path_factory.mktemp("gn2_norm"), out_dim=OUT_DIM)


@pytest.fixture(scope="module")
def batch():
    return make_gn2_batch(n_tracks=N_TRACKS)


@pytest.fixture(scope="module")
def v1_preds(v1, batch):
    inputs, pad_masks = batch
    return v1_forward(v1, inputs, pad_masks)


@pytest.fixture(scope="module")
def v1_stages(v1, batch):
    """Manual v1 intermediates, stage by stage, on fresh clones.

    Reproduces the exact v1 forward chain (modelwrapper.py:221-224,
    saltmodel.py:128-170) so each wrapper's output has a per-stage
    reference, not just the end-to-end preds.
    """
    inputs, pad_masks = batch
    with torch.no_grad():
        # norm rebinds the dict's keys (inputnorm.py:103-106) -> fresh dict
        normed = v1.norm({k: v.clone() for k, v in inputs.items()})
        # init dispatch (saltmodel.py:128-129)
        embed = v1.model.init_nets[0](dict(normed))
        # encoder + tuple unpack (saltmodel.py:150-153); _add_registers
        # mutates these literals only (transformer.py:777,785)
        enc_masks = {"tracks": pad_masks["tracks"].clone()}
        encoded, enc_masks = v1.model.encoder({"tracks": embed}, pad_mask=enc_masks)
        # pooling over {"embed_xs"} with the post-register mask dict
        # (saltmodel.py:169-170)
        pooled = v1.model.pool_net({"embed_xs": encoded}, pad_mask=enc_masks)
    return {
        "normed": normed,
        "embed_tracks": embed,
        "encoded": encoded,
        "enc_masks": enc_masks,
        "pooled": pooled,
    }


@pytest.fixture(scope="module")
def v2(v1):
    return from_v1(v1)


@pytest.fixture(scope="module")
def plan(v1, v2):
    return compile_plan(v2, Mode.TEST, sources=v1_sources(v1), sinks=v1_sinks(v1))


@pytest.fixture(scope="module")
def executed(plan, batch):
    """The v2 bundle after one debug-mode Executor run (the parity subject)."""
    inputs, pad_masks = batch
    bundle = Bundle({
        "inputs": {k: v.clone() for k, v in inputs.items()},
        "masks": {k: v.clone() for k, v in pad_masks.items()},
    })
    with torch.no_grad():
        # debug=True: undeclared reads raise UndeclaredAccessError, in-place
        # mutation of bundle tensors raises MutationError — mechanical proof
        # of the wiring discipline (executor.py:108-177).
        Executor(plan).run(bundle, debug=True)
    return bundle


# ---------------------------------------------------------------------------
# batch sanity: the data must be able to excite the bugs this gate hunts
# ---------------------------------------------------------------------------


def test_batch_has_real_padding(batch):
    _, pad_masks = batch
    mask = pad_masks["tracks"]
    assert mask.any(), "no padded positions: mask-polarity bugs would be unexcitable"
    n_valid = (~mask).sum(dim=1)
    assert (n_valid == 1).any(), "need one jet with exactly one valid track (edge case)"
    assert (n_valid == 0).any(), "need one zero-track jet (production edge case)"
    assert (n_valid > 0).any(), "need jets with valid tracks too"
    assert (n_valid < mask.shape[1]).any()


# ---------------------------------------------------------------------------
# plan structure: the v2 path demonstrably runs through the M1 kernel
# ---------------------------------------------------------------------------


def test_plan_modules_and_order(plan):
    names = list(plan.module_names)
    assert set(names) == {
        "norm",
        "tracks_embed",
        "concat",
        "encoder",
        "pool",
        "jets_classification",
        "track_origin",
        "track_vertexing",
    }
    order = {name: i for i, name in enumerate(names)}
    assert order["norm"] < order["tracks_embed"] < order["concat"] < order["encoder"]
    assert order["encoder"] < order["pool"]
    assert all(order["pool"] < order[t] for t in ("track_origin", "track_vertexing"))
    assert order["pool"] < order["jets_classification"]
    assert len(plan.plan_hash) == 64  # sha256 hex


def test_executed_bundle_has_all_keys(executed):
    keys = set(executed.keys())
    expected = {
        "inputs.jets",
        "inputs.tracks",
        "masks.tracks",
        "normed.jets",
        "normed.tracks",
        "embed.tracks",
        "seq.x",
        "seq.mask",
        "seq.layout",
        "encoded.seq",
        "masks.registers",
        "pooled.global",
        *EXPECTED_SINKS,
    }
    assert keys == expected


# ---------------------------------------------------------------------------
# instance sharing: weights identical by construction, no whole-model shortcut
# ---------------------------------------------------------------------------


def test_from_v1_shares_instances(v1, v2):
    assert v2["norm"].norm is v1.norm
    assert v2["tracks_embed"].init_net is v1.model.init_nets[0]
    assert v2["encoder"].encoder is v1.model.encoder
    assert v2["pool"].pool_net is v1.model.pool_net
    assert v2["jets_classification"].task is v1.model.tasks[0]
    assert v2["track_origin"].task is v1.model.tasks[1]
    assert v2["track_vertexing"].task is v1.model.tasks[2]
    # anti-false-parity: no wrapper may hold the whole v1 model/wrapper and
    # secretly call its forward instead of wiring through the graph
    for module in v2.values():
        held = vars(module).get("_modules", {}).values() if hasattr(module, "_modules") else []
        for sub in held:
            assert sub is not v1.model
            assert sub is not v1
    assert len(v2) == 8


def test_instance_names_match_keys(v2):
    for key, module in v2.items():
        assert module.name == key


# ---------------------------------------------------------------------------
# per-wrapper parity vs manually computed v1 intermediates (all bitwise)
# ---------------------------------------------------------------------------


def test_normaliser_matches_v1(executed, v1_stages):
    assert torch.equal(executed.get("normed.jets"), v1_stages["normed"]["jets"])
    assert torch.equal(executed.get("normed.tracks"), v1_stages["normed"]["tracks"])


def test_normaliser_actually_normalises(executed):
    # parity norm dict has DISTINCT nonzero means / non-unit stds per
    # variable (write_parity_norm_dict): a missed norm stage cannot silently
    # pass, and neither can constant-order or scale mis-wiring
    assert not torch.equal(executed.get("normed.tracks"), executed.get("inputs.tracks"))
    assert not torch.equal(executed.get("normed.jets"), executed.get("inputs.jets"))


def test_inputs_not_mutated(executed, batch):
    inputs, pad_masks = batch
    assert torch.equal(executed.get("inputs.jets"), inputs["jets"])
    assert torch.equal(executed.get("inputs.tracks"), inputs["tracks"])
    assert torch.equal(executed.get("masks.tracks"), pad_masks["tracks"])


def test_stream_embed_matches_v1(executed, v1_stages):
    assert torch.equal(executed.get("embed.tracks"), v1_stages["embed_tracks"])


def test_concat_single_stream(executed, v1_stages):
    # single-stream concat: same values, fresh tensor (no aliasing)
    assert torch.equal(executed.get("seq.x"), v1_stages["embed_tracks"])
    assert executed.get("seq.x").data_ptr() != executed.get("embed.tracks").data_ptr()
    assert torch.equal(executed.get("seq.mask"), executed.get("masks.tracks"))
    assert executed.get("seq.layout") == {"tracks": (0, N_TRACKS)}


def test_encoder_matches_v1(executed, v1_stages, v1_preds):
    encoded = executed.get("encoded.seq")
    assert encoded.shape == (6, N_TRACKS + 1, OUT_DIM)  # register row kept
    assert torch.equal(encoded, v1_stages["encoded"])
    # and against the full v1 forward's embed_xs (saltmodel.py:154)
    assert torch.equal(encoded, v1_preds["embed_xs"])


def test_encoder_registers_mask(executed, v1_stages):
    reg = executed.get("masks.registers")
    assert reg.shape == (6, 1)
    assert reg.dtype == torch.bool
    assert not reg.any()  # registers are never padded (transformer.py:635)
    assert torch.equal(reg, v1_stages["enc_masks"]["REGISTERS"])


def test_pooling_matches_v1(executed, v1_stages, v1_preds):
    pooled = executed.get("pooled.global")
    assert torch.equal(pooled, v1_stages["pooled"])
    assert torch.equal(pooled, v1_preds["global_rep"])


def test_task_jets_classification_matches_v1(executed, v1_preds):
    out = executed.get("preds.jets.jets_classification")
    assert out.shape == (6, 3)
    assert torch.equal(out, v1_preds["jets"]["jets_classification"])


def test_task_track_origin_matches_v1(executed, v1_preds):
    out = executed.get("preds.tracks.track_origin")
    assert out.shape == (6, N_TRACKS, 8)  # raw logits incl. rows at padded positions
    assert torch.equal(out, v1_preds["tracks"]["track_origin"])


def test_task_track_vertexing_matches_v1(executed, v1_preds, batch):
    _, pad_masks = batch
    n_valid = (~pad_masks["tracks"]).sum(dim=1)
    n_edges = int((n_valid * (n_valid - 1)).sum())  # E = sum n_v * (n_v - 1)
    out = executed.get("preds.tracks.track_vertexing")
    assert out.shape == (n_edges, 1), "E mismatch = mask-wiring bug"
    assert torch.equal(out, v1_preds["tracks"]["track_vertexing"])


# ---------------------------------------------------------------------------
# Split (off the parity task path — standalone check against seq slices)
# ---------------------------------------------------------------------------


def test_split_matches_seq_slices(executed):
    split = Split(("tracks",))
    split.name = "split"
    with torch.no_grad():
        out = split(executed, Mode.TEST)
    assert set(out) == {"encoded.tracks"}
    assert torch.equal(out["encoded.tracks"], executed.get("encoded.seq")[:, :N_TRACKS])


# ---------------------------------------------------------------------------
# sensitivity: the comparison must be able to FAIL on the hunted bug class
# ---------------------------------------------------------------------------


def test_mask_dict_order_is_load_bearing(v1, executed):
    """A reordered mask dict silently selects the wrong slice (task.py:73-78).

    Proves this batch can excite the dict-order bug the wrappers comment on:
    if it could not, bitwise parity would be a mask-order coincidence.
    """
    task = v1.model.tasks[1]  # track_origin
    encoded = executed.get("encoded.seq")
    pooled = executed.get("pooled.global")
    tracks_mask = executed.get("masks.tracks")
    reg_mask = executed.get("masks.registers")
    with torch.no_grad():
        good, _ = task(
            encoded, None, {"tracks": tracks_mask, "REGISTERS": reg_mask}, context=pooled
        )
        bad, _ = task(encoded, None, {"REGISTERS": reg_mask, "tracks": tracks_mask}, context=pooled)
    assert torch.equal(good, executed.get("preds.tracks.track_origin"))
    assert not torch.equal(good, bad)


def test_v2_double_run_is_bitwise_repeatable(plan, batch):
    inputs, pad_masks = batch
    results = []
    for _ in range(2):
        bundle = Bundle({
            "inputs": {k: v.clone() for k, v in inputs.items()},
            "masks": {k: v.clone() for k, v in pad_masks.items()},
        })
        with torch.no_grad():
            Executor(plan).run(bundle)
        results.append({key: bundle.get(key) for key in EXPECTED_SINKS})
    for key in EXPECTED_SINKS:
        assert torch.equal(results[0][key], results[1][key])
