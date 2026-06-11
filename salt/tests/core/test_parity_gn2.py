"""End-to-end tests for the GN2 forward-parity gate (plan 04, stages 3-4).

Positive: the gate passes bitwise on the small GN2 (via the real CLI entry
point) — including the two-stream GN2e-style variant — and writes a complete
report proving the v2 side ran through the M1 Executor. Negative controls:
six deliberate mis-wirings — task pad-mask dict order, skipping the
normaliser, pad-mask polarity flip, pooling pad-mask dict order, rolled
per-variable norm constants, reversed multi-stream concat — must each make
the gate FAIL, proving the comparison has teeth on exactly the bug class the
gate hunts (silent wrong slice, missing norm, wrong edge count, misaligned
pooling mask, norm-constant misordering, wrong stream layout).
"""

from __future__ import annotations

import json

import torch

from salt.core.graph import IO, TensorSpec, sym_dim, unflatten_spec
from salt.core.nn import Concat, ConstituentTask, Normaliser, Pooling, StreamEmbed
from salt.core.parity_gn2 import main, run_parity

EXPECTED_SINKS = {
    "preds.jets.jets_classification",
    "preds.tracks.track_origin",
    "preds.tracks.track_vertexing",
}
EXPECTED_MODULES = {
    "norm",
    "tracks_embed",
    "concat",
    "encoder",
    "pool",
    "jets_classification",
    "track_origin",
    "track_vertexing",
}


# ---------------------------------------------------------------------------
# positive: the gate passes end to end via the CLI entry point
# ---------------------------------------------------------------------------


def test_gate_passes_end_to_end(tmp_path, capsys):
    code = main(["--outdir", str(tmp_path)])
    assert code == 0

    report = json.loads((tmp_path / "parity_report.json").read_text())
    assert report["passed"] is True

    # every preds leaf compared, all bitwise (max abs diff exactly 0.0)
    leaves = {r["key"]: r for r in report["leaves"]}
    assert set(leaves) == EXPECTED_SINKS
    for record in leaves.values():
        assert record["passed"] is True
        assert record["bitwise"] is True
        # exact zero is intentional: compare_leaf sets 0.0 verbatim when bitwise
        assert record["max_abs_diff"] == 0.0  # noqa: RUF069
        assert record["ref_shape"] == record["got_shape"]
        assert record["ref_dtype"] == record["got_dtype"] == "torch.float32"

    # bonus intermediates: encoded.seq <-> embed_xs, pooled.global <-> global_rep
    intermediates = {r["key"]: r for r in report["intermediates"]}
    assert set(intermediates) == {"encoded.seq", "pooled.global"}
    assert all(r["bitwise"] and r["passed"] for r in intermediates.values())

    # proof the v2 side ran through the M1 Executor: full plan + executed keys
    assert set(report["plan"]["module_names"]) == EXPECTED_MODULES
    assert len(report["plan"]["plan_hash"]) == 64  # sha256 hex
    assert report["plan"]["executor_debug"] is True
    executed = set(report["plan"]["executed_bundle_keys"])
    assert {"seq.x", "seq.mask", "encoded.seq", "masks.registers", "pooled.global"} <= executed
    assert executed >= EXPECTED_SINKS

    # no silent criterion downgrade: nothing registered as justified non-bitwise
    assert report["justified_nonbitwise"] == {}

    # the batch exercises the production edge cases: a zero-track jet
    # (pooling's all-padded ONNX branch, zero-edge vertexing contribution)
    # and a one-track jet — run_parity itself raises if the zero-track jet
    # is missing, this pins the report contract
    assert 0 in report["config"]["valid_counts"]["tracks"]
    assert 1 in report["config"]["valid_counts"]["tracks"]

    # human table on stdout includes the verdict, the plan hash, and the steps
    out = capsys.readouterr().out
    assert "PARITY GATE: PASS" in out
    assert report["plan"]["plan_hash"] in out
    for name in EXPECTED_MODULES:
        assert name in out


def test_gate_passes_other_batch_shape(tmp_path):
    # determinism settings hold away from the default batch geometry
    code = main(["--outdir", str(tmp_path), "--batch-size", "3", "--n-tracks", "7", "--seed", "7"])
    assert code == 0


def test_gate_passes_two_streams(tmp_path):
    """Two sequence streams (GN2e-style electrons): concat order is genuinely exercised.

    With a single sequence stream, multi-stream concat order / per-stream
    mask dict order are vacuous (reversing a 1-tuple is the identity —
    stage-4 critic finding). This variant makes them load-bearing; the
    reversed-concat negative control below proves it.
    """
    code, report = run_parity(tmp_path, with_electrons=True)
    assert code == 0
    assert report["passed"] is True
    assert report["config"]["with_electrons"] is True

    # the second stream is a real plan participant, not a bystander
    assert "electrons_embed" in report["plan"]["module_names"]
    executed = set(report["plan"]["executed_bundle_keys"])
    assert {"inputs.electrons", "masks.electrons", "normed.electrons", "embed.electrons"} <= executed

    # heterogeneous electron padding incl. a zero-electron jet (typical jets)
    assert 0 in report["config"]["valid_counts"]["electrons"]
    assert any(n > 0 for n in report["config"]["valid_counts"]["electrons"])

    # same three task leaves, all bitwise
    leaves = {r["key"]: r for r in report["leaves"]}
    assert set(leaves) == EXPECTED_SINKS
    assert all(r["bitwise"] and r["passed"] for r in leaves.values())
    assert all(r["bitwise"] and r["passed"] for r in report["intermediates"])


# ---------------------------------------------------------------------------
# negative control 1: reordered task pad-mask dict (silent wrong slice)
# ---------------------------------------------------------------------------


class _RegistersFirstTask(ConstituentTask):
    """Deliberate mis-wiring: REGISTERS first in the task pad-mask dict.

    input_name_mask concatenates per-stream widths in DICT ORDER
    (task.py:73-78) — this selects the wrong sequence slice with an
    UNCHANGED output shape, the silent failure the gate must catch.
    """

    def forward(self, b, mode):
        del mode
        masks = {"REGISTERS": b.get("masks.registers")}
        for stream in self.streams:
            masks[stream] = b.get(f"masks.{stream}")
        preds, _loss = self.task(b.get("encoded.seq"), None, masks, context=b.get("pooled.global"))
        return {f"preds.{self.task.input_name}.{self.task.name}": preds}


def test_gate_fails_on_task_mask_dict_order(tmp_path):
    def hook(modules):
        old = modules["track_origin"]
        modules["track_origin"] = _RegistersFirstTask(old.task, old.streams)
        return modules

    code, report = run_parity(tmp_path, modules_hook=hook)
    assert code == 1
    assert report["passed"] is False
    leaves = {r["key"]: r for r in report["leaves"]}
    origin = leaves["preds.tracks.track_origin"]
    assert origin["passed"] is False
    assert origin["bitwise"] is False
    assert origin["ref_shape"] == origin["got_shape"]  # shape unchanged: silent wrong slice
    assert origin["max_abs_diff"] > 0
    # the failure is localised: untouched leaves still pass
    assert leaves["preds.jets.jets_classification"]["passed"] is True
    assert leaves["preds.tracks.track_vertexing"]["passed"] is True


# ---------------------------------------------------------------------------
# negative control 2: feed UN-normalised inputs to the embed (norm skipped)
# ---------------------------------------------------------------------------


class _RawInputEmbed(StreamEmbed):
    """Deliberate mis-wiring: reads ``inputs.*`` instead of ``normed.*``.

    The parity norm dict has DISTINCT nonzero means and non-unit stds per
    variable (write_parity_norm_dict) — a nontrivial affine map, so skipping
    the normaliser must change every downstream leaf.
    """

    def declare_io(self, mode):
        del mode
        variables = self.init_net.variables
        requires = {
            f"inputs.{self.stream}": TensorSpec(
                shape=("B", sym_dim("T", self.stream), len(variables[self.stream])),
                dtype="float32",
            )
        }
        if self.context_stream is not None:
            requires[f"inputs.{self.context_stream}"] = TensorSpec(
                shape=("B", len(variables[self.context_stream])), dtype="float32"
            )
        return IO(
            requires=unflatten_spec(requires),
            produces=unflatten_spec({
                f"embed.{self.stream}": TensorSpec(
                    shape=("B", sym_dim("T", self.stream), self.out_dim), dtype="float32"
                )
            }),
        )

    def forward(self, b, mode):
        del mode
        fresh = {self.stream: b.get(f"inputs.{self.stream}")}
        if self.context_stream is not None:
            fresh[self.context_stream] = b.get(f"inputs.{self.context_stream}")
        return {f"embed.{self.stream}": self.init_net(fresh)}


def test_gate_fails_when_norm_is_skipped(tmp_path):
    def hook(modules):
        modules["tracks_embed"] = _RawInputEmbed(modules["tracks_embed"].init_net)
        return modules

    code, report = run_parity(tmp_path, modules_hook=hook)
    assert code == 1
    assert report["passed"] is False
    # nothing consumes normed.* any more -> norm is demand-pruned from the plan
    assert "norm" not in report["plan"]["module_names"]
    # the (x - 1) shift propagates through everything downstream
    assert all(r["passed"] is False for r in report["leaves"])
    assert all(r["passed"] is False for r in report["intermediates"])


# ---------------------------------------------------------------------------
# negative control 3: pad-mask polarity flip (wrong edge count -> shape gate)
# ---------------------------------------------------------------------------


class _FlippedPolarityVertexing(ConstituentTask):
    """Deliberate mis-wiring: inverts the stream pad masks (True = valid).

    The vertexing adjacency is built from the mask VALUES (task.py:874-881),
    so a polarity flip changes the edge count E — caught by the shape check
    BEFORE any value comparison.
    """

    def forward(self, b, mode):
        del mode
        masks = {stream: ~b.get(f"masks.{stream}") for stream in self.streams}
        masks["REGISTERS"] = b.get("masks.registers")
        preds, _loss = self.task(b.get("encoded.seq"), None, masks, context=b.get("pooled.global"))
        return {f"preds.{self.task.input_name}.{self.task.name}": preds}


def test_gate_fails_on_mask_polarity_flip(tmp_path):
    def hook(modules):
        old = modules["track_vertexing"]
        modules["track_vertexing"] = _FlippedPolarityVertexing(old.task, old.streams)
        return modules

    # Default geometry: the zero-track jet (0 -> 10 valid under the flip)
    # breaks the mirror symmetry that used to make the flipped edge count
    # coincide at n_tracks=10. With n_valid={1,0,5,9,4,6} the counts are
    # E_orig=134 vs E_flip=224 — the shape gate fires. (Even where E
    # coincides, the flip fails through VALUES: stage-4 critic verified
    # max|diff|=1.28e-1 on the old symmetric batch; the pooling control
    # below keeps a value-level mask-path failure in the suite.)
    code, report = run_parity(tmp_path, modules_hook=hook)
    assert code == 1
    assert report["passed"] is False
    vert = {r["key"]: r for r in report["leaves"]}["preds.tracks.track_vertexing"]
    assert vert["passed"] is False
    assert vert["ref_shape"] != vert["got_shape"]  # E mismatch
    assert vert["max_abs_diff"] is None  # shape checked first, no value diff computed
    assert "shape" in vert["note"].lower()


# ---------------------------------------------------------------------------
# negative control 4: pooling pad-mask dict order (misaligned mask, values)
# ---------------------------------------------------------------------------


class _RegistersFirstPooling(Pooling):
    """Deliberate mis-wiring: REGISTERS first in the pooling pad-mask dict.

    GlobalAttentionPooling concatenates mask values in dict order
    (pooling.py:56) — REGISTERS-first shifts every mask value by one
    position relative to the encoded sequence, so the attention weights
    mask the WRONG tokens with an UNCHANGED output shape.
    """

    def forward(self, b, mode):
        del mode
        x = {"seq": b.get("encoded.seq")}
        pad = {"REGISTERS": b.get("masks.registers"), "seq": b.get("seq.mask")}
        return {"pooled.global": self.pool_net(x, pad_mask=pad)}


def test_gate_fails_on_pooling_mask_dict_order(tmp_path):
    def hook(modules):
        modules["pool"] = _RegistersFirstPooling(modules["pool"].pool_net)
        return modules

    code, report = run_parity(tmp_path, modules_hook=hook)
    assert code == 1
    assert report["passed"] is False
    # pooled.global fails through VALUES (shape unchanged) ...
    intermediates = {r["key"]: r for r in report["intermediates"]}
    pooled = intermediates["pooled.global"]
    assert pooled["passed"] is False
    assert pooled["ref_shape"] == pooled["got_shape"]
    assert pooled["max_abs_diff"] > 0
    # ... and propagates to ALL task leaves (directly or via context), again
    # with unchanged shapes — the edge count depends only on the track masks
    leaves = {r["key"]: r for r in report["leaves"]}
    for record in leaves.values():
        assert record["passed"] is False
        assert record["ref_shape"] == record["got_shape"]
        assert record["max_abs_diff"] > 0
    # localisation: the encoder (upstream of pooling) is untouched
    assert intermediates["encoded.seq"]["passed"] is True


# ---------------------------------------------------------------------------
# negative control 5: rolled per-variable norm constants (field-order bug)
# ---------------------------------------------------------------------------


class _RolledNormaliser(Normaliser):
    """Deliberate mis-wiring: per-variable norm constants rolled by one.

    Simulates a field-order mismatch between the norm dict and the input
    columns — exactly the class expected in M2 phase 1, where the v2
    Normaliser is built from config instead of sharing the v1 instance.
    Catchable ONLY because the fixture's constants are distinct per variable:
    under salt's uniform mean=1/std=1 dummy dict this roll is the identity
    and the gate false-PASSed (stage-4 critic finding).
    """

    def forward(self, b, mode):
        del mode
        out = {}
        for stream in self.streams:
            x = b.get(f"inputs.{stream}")
            means = torch.roll(getattr(self.norm, f"{stream}_means"), 1)
            stds = torch.roll(getattr(self.norm, f"{stream}_stds"), 1)
            out[f"normed.{stream}"] = (x - means) / stds
        return out


def test_gate_fails_on_rolled_norm_constants(tmp_path):
    def hook(modules):
        modules["norm"] = _RolledNormaliser(modules["norm"].norm)
        return modules

    code, report = run_parity(tmp_path, modules_hook=hook)
    assert code == 1
    assert report["passed"] is False
    # the misordered constants corrupt every stream at the source
    assert all(r["passed"] is False for r in report["leaves"])
    assert all(r["passed"] is False for r in report["intermediates"])


# ---------------------------------------------------------------------------
# negative control 6: reversed multi-stream concat (wrong sequence layout)
# ---------------------------------------------------------------------------


def test_gate_fails_on_reversed_concat_two_streams(tmp_path):
    """Reversed Concat must FAIL — but only two streams make it excitable.

    The tasks rebuild their pad-mask dicts in the ORIGINAL stream order, so
    a reversed sequence layout makes ``input_name_mask`` select the wrong
    slice (silently — output shapes are unchanged because per-stream widths
    are fixed). With the single-stream default this control would pass
    trivially (stage-4 critic finding), hence ``with_electrons=True``.
    """

    def hook(modules):
        modules["concat"] = Concat(tuple(reversed(modules["concat"].streams)))
        return modules

    code, report = run_parity(tmp_path, with_electrons=True, modules_hook=hook)
    assert code == 1
    assert report["passed"] is False
    # the encoded sequence is a row permutation of v1's: same shape, new values
    intermediates = {r["key"]: r for r in report["intermediates"]}
    enc = intermediates["encoded.seq"]
    assert enc["passed"] is False
    assert enc["ref_shape"] == enc["got_shape"]
    # constituent tasks slice the wrong tokens: silent value-level failure
    leaves = {r["key"]: r for r in report["leaves"]}
    origin = leaves["preds.tracks.track_origin"]
    assert origin["passed"] is False
    assert origin["ref_shape"] == origin["got_shape"]
    assert origin["max_abs_diff"] > 0
