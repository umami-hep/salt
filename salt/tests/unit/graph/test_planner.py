"""Tests for salt.graph.planner."""

import pytest

from salt.graph.errors import (
    AllModesDeadError,
    ConfigError,
    ConnectivityError,
    CycleError,
    KindError,
    ShapeError,
)
from salt.graph.planner import (
    SINKS,
    SOURCES,
    DeadOutput,
    Edge,
    compile_plan,
    deadcode,
)
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec

# toy fixtures (no physics)


class Toy:
    """Minimal GraphModule: static IO from flat dotted-key spec dicts."""

    def __init__(self, name, requires=None, produces=None):
        self.name = name
        self._io = IO(
            requires=unflatten_spec(requires or {}),
            produces=unflatten_spec(produces or {}),
        )

    def declare_io(self, mode):
        return self._io


class WildToy(Toy):
    """Framework-shipped wildcard producer."""

    allow_wildcards = True


def ts(**kwargs):
    return TensorSpec(**kwargs)


def mods(*modules):
    return {m.name: m for m in modules}


SRC_X = unflatten_spec({"inputs.x": TensorSpec()})
SRC_XY = unflatten_spec({"inputs.x": TensorSpec(), "inputs.y": TensorSpec()})


def chain_ab():
    a = Toy("a", requires={"inputs.x": ts()}, produces={"embed.x": ts()})
    b = Toy("b", requires={"embed.x": ts()}, produces={"preds.x": ts()})
    return a, b


# basic topologies


class TestLinearChain:
    def test_order_edges_and_mode(self):
        a, b = chain_ab()
        plan = compile_plan({"b": b, "a": a}, Mode.FIT, SRC_X)
        assert plan.module_names == ("a", "b")
        assert plan.mode == Mode.FIT
        assert Edge(SOURCES, "inputs.x", "a") in plan.edges
        assert Edge("a", "embed.x", "b") in plan.edges
        assert len(plan.plan_hash) == 64
        assert set(plan.sources) == {"inputs.x"}

    def test_steps_carry_module_and_resolved_io(self):
        a, b = chain_ab()
        plan = compile_plan(mods(a, b), Mode.FIT, SRC_X)
        step_a = plan.step("a")
        assert step_a.module is a
        assert set(step_a.requires) == {"inputs.x"}
        assert set(step_a.produces) == {"embed.x"}
        with pytest.raises(KeyError, match="no step"):
            plan.step("nope")

    def test_executor_facing_produces_key_set(self):
        # the executor passes set(step.produces) to Bundle.merge(expected=...)
        a, b = chain_ab()
        plan = compile_plan(mods(a, b), Mode.FIT, SRC_X)
        assert set(plan.step("b").produces) == {"preds.x"}


class TestDiamond:
    def test_diamond_order_ties_broken_by_config_order(self):
        # tie-break = config declaration order, NOT module name
        a = Toy("a", requires={"inputs.x": ts()}, produces={"embed.x": ts()})
        b = Toy("b", requires={"embed.x": ts()}, produces={"left.x": ts()})
        c = Toy("c", requires={"embed.x": ts()}, produces={"right.x": ts()})
        d = Toy("d", requires={"left.x": ts(), "right.x": ts()}, produces={"preds.x": ts()})
        plan = compile_plan(mods(d, c, b, a), Mode.FIT, SRC_X)
        assert plan.module_names == ("a", "c", "b", "d")  # c declared before b
        assert Edge("b", "left.x", "d") in plan.edges
        assert Edge("c", "right.x", "d") in plan.edges


class TestDisconnectedSubgraphs:
    def test_both_subgraphs_planned_in_config_order(self):
        # ties (independent subgraphs) follow config declaration order
        a = Toy("a", requires={"inputs.x": ts()}, produces={"embed.x": ts()})
        b = Toy("b", requires={"embed.x": ts()}, produces={"preds.x": ts()})
        c = Toy("c", requires={"inputs.y": ts()}, produces={"embed.y": ts()})
        d = Toy("d", requires={"embed.y": ts()}, produces={"preds.y": ts()})
        plan = compile_plan(mods(d, b, c, a), Mode.FIT, SRC_XY)
        assert plan.module_names == ("c", "d", "a", "b")


# connectivity errors (quality bar)


class TestMissingProducer:
    def test_names_consumer_and_suggests_nearest_key(self):
        a = Toy("a", requires={"inputs.x": ts()}, produces={"embed.tracks": ts()})
        b = Toy("b", requires={"embed.trcks": ts()}, produces={"preds.x": ts()})
        with pytest.raises(ConnectivityError) as exc:
            compile_plan(mods(a, b), Mode.FIT, SRC_X)
        msg = str(exc.value)
        assert "'b'" in msg
        assert "embed.trcks" in msg
        assert "embed.tracks" in msg  # did-you-mean suggestion
        # quality bar: availability AND a concrete fix, alongside the suggestion
        assert "available keys:" in msg
        assert "fix: correct the require in module 'b'" in msg

    def test_no_close_match_still_lists_availability_and_fix(self):
        a = Toy("a", requires={"inputs.x": ts()}, produces={"embed.tracks": ts()})
        b = Toy("b", requires={"zzz.qqq": ts()}, produces={"preds.x": ts()})
        with pytest.raises(ConnectivityError) as exc:
            compile_plan(mods(a, b), Mode.FIT, SRC_X)
        msg = str(exc.value)
        assert "did you mean" not in msg
        assert "available keys: embed.tracks, inputs.x, preds.x" in msg
        assert "fix: correct the require in module 'b'" in msg

    def test_names_producer_available_in_another_mode(self):
        lab = Toy("lab", produces={"labels.x": ts(kind="label", modes=Mode.TRAINING)})
        writer = Toy("writer", requires={"labels.x": ts(kind="label", modes=Mode.TEST)})
        with pytest.raises(ConnectivityError) as exc:
            compile_plan(mods(lab, writer), Mode.TEST, {})
        msg = str(exc.value)
        assert "'writer'" in msg
        assert "'lab'" in msg
        assert "FIT" in msg

    def test_source_active_in_other_mode_is_mentioned(self):
        src = unflatten_spec({"inputs.x": TensorSpec(modes=Mode.TRAINING)})
        a = Toy("a", requires={"inputs.x": ts()}, produces={"embed.x": ts()})
        with pytest.raises(ConnectivityError) as exc:
            compile_plan(mods(a), Mode.TEST, src)
        assert "sources provide 'inputs.x'" in str(exc.value)

    def test_missing_sink_key(self):
        a, b = chain_ab()
        with pytest.raises(ConnectivityError, match="preds.nope"):
            compile_plan(mods(a, b), Mode.FIT, SRC_X, sinks=["preds.nope"])


class TestDuplicateProducers:
    def test_two_concrete_producers(self):
        a1 = Toy("a1", requires={"inputs.x": ts()}, produces={"embed.x": ts()})
        a2 = Toy("a2", requires={"inputs.x": ts()}, produces={"embed.x": ts()})
        with pytest.raises(ConnectivityError) as exc:
            compile_plan(mods(a1, a2), Mode.FIT, SRC_X)
        msg = str(exc.value)
        assert "'a1'" in msg
        assert "'a2'" in msg
        assert "embed.x" in msg

    def test_module_clashing_with_source(self):
        a = Toy("a", produces={"inputs.x": ts()})
        with pytest.raises(ConnectivityError, match="two producers"):
            compile_plan(mods(a), Mode.FIT, SRC_X)

    def test_two_wildcard_producers_for_one_key(self):
        w1 = WildToy("w1", produces={"labels.**": ts(kind="label")})
        w2 = WildToy("w2", produces={"labels.**": ts(kind="label")})
        t = Toy("t", requires={"labels.jets.flav": ts(kind="label")})
        with pytest.raises(ConnectivityError) as exc:
            compile_plan(mods(w1, w2, t), Mode.FIT, {})
        msg = str(exc.value)
        assert "'w1'" in msg
        assert "'w2'" in msg


# kind typing


class TestKindChecking:
    def test_kind_mismatch_is_static_error(self):
        lab = Toy("lab", produces={"labels.t": ts(kind="label")})
        pool = Toy("pool", requires={"labels.t": ts(kind="pad_mask")}, produces={"pooled.t": ts()})
        with pytest.raises(KindError) as exc:
            compile_plan(mods(lab, pool), Mode.FIT, {})
        msg = str(exc.value)
        assert "'pool'" in msg
        assert "'lab'" in msg
        assert "pad_mask" in msg
        assert "label" in msg

    def test_matching_kinds_pass(self):
        lab = Toy("lab", produces={"masks.t": ts(kind="pad_mask")})
        pool = Toy("pool", requires={"masks.t": ts(kind="pad_mask")}, produces={"pooled.t": ts()})
        plan = compile_plan(mods(lab, pool), Mode.FIT, {})
        assert plan.module_names == ("lab", "pool")


# symbolic-dim unification


class TestShapeUnification:
    def test_symbolic_dims_unify_across_chain(self):
        src = unflatten_spec({"inputs.t": TensorSpec(shape=("B", "T:t", 4))})
        emb = Toy(
            "emb",
            requires={"inputs.t": ts(shape=("B", "T:t", 4))},
            produces={"embed.t": ts(shape=("B", "T:t", "D"))},
        )
        pool = Toy(
            "pool",
            requires={"embed.t": ts(shape=("B", "T:t", 128))},
            produces={"pooled.t": ts(shape=("B", 128))},
        )
        plan = compile_plan(mods(emb, pool), Mode.FIT, src)
        assert plan.module_names == ("emb", "pool")

    def test_conflicting_symbolic_binding_names_both_endpoints(self):
        emb = Toy("emb", produces={"embed.t": ts(shape=("B", "D"))})
        emb2 = Toy("emb2", produces={"embed.u": ts(shape=("B", "D"))})
        c1 = Toy("c1", requires={"embed.t": ts(shape=("B", 128))}, produces={"out.a": ts()})
        c2 = Toy("c2", requires={"embed.u": ts(shape=("B", 64))}, produces={"out.b": ts()})
        with pytest.raises(ShapeError) as exc:
            compile_plan(mods(emb, emb2, c1, c2), Mode.FIT, {})
        msg = str(exc.value)
        assert "128" in msg
        assert "64" in msg
        assert "embed.t" in msg
        assert "embed.u" in msg

    def test_concrete_size_mismatch(self):
        a = Toy("a", produces={"embed.x": ts(shape=("B", 64))})
        b = Toy("b", requires={"embed.x": ts(shape=("B", 128))}, produces={"out.x": ts()})
        with pytest.raises(ShapeError, match="size mismatch"):
            compile_plan(mods(a, b), Mode.FIT, {})

    def test_rank_mismatch(self):
        a = Toy("a", produces={"embed.x": ts(shape=("B", "T:t", 64))})
        b = Toy("b", requires={"embed.x": ts(shape=("B", 64))}, produces={"out.x": ts()})
        with pytest.raises(ShapeError, match="rank mismatch"):
            compile_plan(mods(a, b), Mode.FIT, {})

    def test_dtype_mismatch(self):
        a = Toy("a", produces={"embed.x": ts(dtype="float32")})
        b = Toy("b", requires={"embed.x": ts(dtype="float16")}, produces={"out.x": ts()})
        with pytest.raises(ShapeError, match="dtype mismatch"):
            compile_plan(mods(a, b), Mode.FIT, {})

    def test_symbolic_alias_chain_conflict(self):
        # D unified with E via an edge; binding D=128 then E=64 must conflict
        a = Toy("a", produces={"embed.x": ts(shape=("B", "D"))})
        b = Toy("b", requires={"embed.x": ts(shape=("B", "E"))}, produces={"out.x": ts(shape=("B", "E"))})
        c1 = Toy("c1", requires={"out.x": ts(shape=("B", 64))}, produces={"fin.a": ts()})
        c2 = Toy("c2", requires={"embed.x": ts(shape=("B", 128))}, produces={"fin.b": ts()})
        with pytest.raises(ShapeError):
            compile_plan(mods(a, b, c1, c2), Mode.FIT, {})


# cycles


class TestCycles:
    def test_two_module_cycle_names_path(self):
        a = Toy("a", requires={"k.b": ts()}, produces={"k.a": ts()})
        b = Toy("b", requires={"k.a": ts()}, produces={"k.b": ts()})
        with pytest.raises(CycleError) as exc:
            compile_plan(mods(a, b), Mode.FIT, {})
        msg = str(exc.value)
        assert "a" in msg
        assert "b" in msg
        assert "k.a" in msg
        assert "k.b" in msg

    def test_self_loop(self):
        a = Toy("a", requires={"k.a": ts()}, produces={"k.a": ts()})
        with pytest.raises((CycleError, ConnectivityError)):
            # a both produces and requires k.a: a -(k.a)-> a
            compile_plan(mods(a), Mode.FIT, {})

    def test_three_module_cycle(self):
        a = Toy("a", requires={"k.c": ts()}, produces={"k.a": ts()})
        b = Toy("b", requires={"k.a": ts()}, produces={"k.b": ts()})
        c = Toy("c", requires={"k.b": ts()}, produces={"k.c": ts()})
        with pytest.raises(CycleError, match="cycle"):
            compile_plan(mods(a, b, c), Mode.FIT, {})


# optional ports (consumed-if-present)


class TestOptionalPorts:
    def test_absent_optional_is_dropped_silently(self):
        a = Toy("a", requires={"inputs.x": ts()}, produces={"embed.x": ts()})
        c = Toy(
            "c",
            requires={"embed.x": ts(), "aux.x": ts(optional=True)},
            produces={"preds.x": ts()},
        )
        plan = compile_plan(mods(a, c), Mode.FIT, SRC_X)
        assert set(plan.step("c").requires) == {"embed.x"}
        assert not any(e.key == "aux.x" for e in plan.edges)

    def test_present_optional_is_consumed(self):
        a = Toy("a", requires={"inputs.x": ts()}, produces={"embed.x": ts()})
        x = Toy("x", requires={"inputs.x": ts()}, produces={"aux.x": ts()})
        c = Toy(
            "c",
            requires={"embed.x": ts(), "aux.x": ts(optional=True)},
            produces={"preds.x": ts()},
        )
        plan = compile_plan(mods(a, x, c), Mode.FIT, SRC_X)
        assert set(plan.step("c").requires) == {"embed.x", "aux.x"}
        assert Edge("x", "aux.x", "c") in plan.edges

    def test_optional_demand_does_not_narrow_wildcards(self):
        w = WildToy("w", produces={"labels.**": ts(kind="label")})
        c = Toy(
            "c",
            requires={"labels.jets.flav": ts(kind="label", optional=True)},
            produces={"preds.x": ts()},
        )
        plan = compile_plan(mods(w, c), Mode.FIT, {})
        assert set(plan.step("w").produces) == set()
        assert set(plan.step("c").requires) == set()


# per-mode and demand pruning


class TestModePruning:
    def test_fit_only_loss_absent_in_test_plan(self):
        a, b = chain_ab()
        loss = Toy(
            "loss",
            requires={"preds.x": ts(modes=Mode.TRAINING)},
            produces={"losses.x": ts(kind="loss", modes=Mode.TRAINING)},
        )
        fit_plan = compile_plan(mods(a, b, loss), Mode.FIT, SRC_X)
        test_plan = compile_plan(mods(a, b, loss), Mode.TEST, SRC_X)
        assert "loss" in fit_plan.module_names
        assert "loss" not in test_plan.module_names


class TestDemandPruning:
    def test_unconsumed_branch_dropped_when_sinks_given(self):
        a, b = chain_ab()
        aux = Toy("aux", requires={"inputs.x": ts()}, produces={"aux.y": ts()})
        plan = compile_plan(mods(a, b, aux), Mode.FIT, SRC_X, sinks=["preds.x"])
        assert plan.module_names == ("a", "b")
        assert Edge("b", "preds.x", SINKS) in plan.edges
        assert not any(e.producer == "aux" or e.consumer == "aux" for e in plan.edges)

    def test_no_pruning_without_sinks(self):
        a, b = chain_ab()
        aux = Toy("aux", requires={"inputs.x": ts()}, produces={"aux.y": ts()})
        plan = compile_plan(mods(a, b, aux), Mode.FIT, SRC_X)
        assert plan.module_names == ("a", "b", "aux")  # b before aux: config order

    def test_terminal_consumer_anchors_demand(self):
        # a writer-like module (requires, no produces) keeps its chain alive
        a, b = chain_ab()
        writer = Toy("writer", requires={"preds.x": ts()})
        aux = Toy("aux", requires={"inputs.x": ts()}, produces={"aux.y": ts()})
        plan = compile_plan(mods(a, b, writer, aux), Mode.FIT, SRC_X, sinks=[])
        assert plan.module_names == ("a", "b", "writer")


class TestAllModesDead:
    def test_dead_in_every_mode_raises(self):
        a, b = chain_ab()
        aux = Toy("aux", requires={"inputs.x": ts()}, produces={"aux.y": ts()})
        sinks = {Mode.ALL: ["preds.x"]}
        with pytest.raises(AllModesDeadError) as exc:
            compile_plan(mods(a, b, aux), Mode.FIT, SRC_X, sinks=sinks)
        msg = str(exc.value)
        assert "'aux'" in msg
        assert "Toy" in msg

    def test_alive_in_another_mode_is_only_pruned(self):
        # flat sinks apply to the compiled mode only -> aux provably alive in VAL
        a, b = chain_ab()
        aux = Toy("aux", requires={"inputs.x": ts()}, produces={"aux.y": ts()})
        plan = compile_plan(mods(a, b, aux), Mode.FIT, SRC_X, sinks=["preds.x"])
        assert "aux" not in plan.module_names

    def test_per_mode_sinks_keep_test_demanded_module(self):
        a, b = chain_ab()
        lab = Toy("lab", produces={"labels.x": ts(kind="label")})
        sinks = {Mode.TEST: ["preds.x", "labels.x"], Mode.TRAINING | Mode.ONNX: ["preds.x"]}
        fit_plan = compile_plan(mods(a, b, lab), Mode.FIT, SRC_X, sinks=sinks)
        test_plan = compile_plan(mods(a, b, lab), Mode.TEST, SRC_X, sinks=sinks)
        assert "lab" not in fit_plan.module_names  # pruned in fit, alive in test
        assert "lab" in test_plan.module_names

    def test_module_with_no_active_ports_anywhere_raises(self):
        a, b = chain_ab()
        noop = Toy("noop")
        with pytest.raises(AllModesDeadError, match="'noop'"):
            compile_plan(mods(a, b, noop), Mode.FIT, SRC_X)


# wildcard narrowing (rules (a)-(d))


class TestWildcardNarrowing:
    def test_narrowed_to_concrete_demand_and_frozen_into_plan(self):
        lab = WildToy("lab", produces={"labels.**": ts(kind="label", modes=Mode.TRAINING)})
        t1 = Toy(
            "t1",
            requires={"labels.jets.flav": ts(kind="label", modes=Mode.TRAINING)},
            produces={"losses.t1": ts(kind="loss", modes=Mode.TRAINING)},
        )
        t2 = Toy(
            "t2",
            requires={"labels.tracks.origin": ts(kind="label", modes=Mode.TRAINING)},
            produces={"losses.t2": ts(kind="loss", modes=Mode.TRAINING)},
        )
        plan = compile_plan(mods(lab, t1, t2), Mode.FIT, {})
        assert set(plan.step("lab").produces) == {"labels.jets.flav", "labels.tracks.origin"}
        assert Edge("lab", "labels.jets.flav", "t1") in plan.edges
        assert Edge("lab", "labels.tracks.origin", "t2") in plan.edges
        assert plan.module_names[0] == "lab"

    def test_single_star_matches_one_component_only(self):
        norm = WildToy("norm", produces={"normed.*": ts()})
        c1 = Toy("c1", requires={"normed.tracks": ts()}, produces={"out.a": ts()})
        plan = compile_plan(mods(norm, c1), Mode.FIT, {})
        assert set(plan.step("norm").produces) == {"normed.tracks"}

        c2 = Toy("c2", requires={"normed.a.b": ts()}, produces={"out.b": ts()})
        with pytest.raises(ConnectivityError, match="normed.a.b"):
            compile_plan(mods(norm, c1, c2), Mode.FIT, {})

    def test_concrete_producer_beats_wildcard(self):
        lab = WildToy("lab", produces={"labels.**": ts(kind="label")})
        conc = Toy("conc", produces={"labels.jets.flav": ts(kind="label")})
        t1 = Toy("t1", requires={"labels.jets.flav": ts(kind="label")}, produces={"out.a": ts()})
        plan = compile_plan(mods(lab, conc, t1), Mode.FIT, {})
        assert Edge("conc", "labels.jets.flav", "t1") in plan.edges
        assert set(plan.step("lab").produces) == set()

    def test_user_module_may_not_declare_wildcards(self):
        user = Toy("user", produces={"labels.**": ts(kind="label")})
        with pytest.raises(ConfigError) as exc:
            compile_plan(mods(user), Mode.FIT, {})
        msg = str(exc.value)
        assert "wildcard" in msg
        assert "'user'" in msg

    def test_wildcard_requires_rejected(self):
        w = WildToy("w", requires={"labels.**": ts(kind="label")}, produces={"out.x": ts()})
        with pytest.raises(ConfigError, match="require"):
            compile_plan(mods(w), Mode.FIT, {})

    def test_self_feed_rejected_transitively(self):
        w = WildToy("w", requires={"feats.z": ts()}, produces={"labels.**": ts(kind="label")})
        x = Toy("x", requires={"labels.q": ts(kind="label")}, produces={"feats.z": ts()})
        with pytest.raises(CycleError) as exc:
            compile_plan(mods(w, x), Mode.FIT, {})
        msg = str(exc.value)
        assert "wildcard producer 'w'" in msg
        assert "labels.q" in msg
        assert "feats.z" in msg

    def test_self_feed_rejected_directly(self):
        w = WildToy(
            "w",
            requires={"labels.a": ts(kind="label")},
            produces={"labels.**": ts(kind="label")},
        )
        with pytest.raises(CycleError, match="wildcard producer 'w'"):
            compile_plan(mods(w), Mode.FIT, {})

    def test_narrowed_keys_validated_against_schema(self):
        schema = {"labels.jets.flav", "labels.tracks.origin"}
        lab = WildToy("lab", produces={"labels.**": ts(kind="label")})
        typo = Toy("typo", requires={"labels.jets.flavz": ts(kind="label")}, produces={"o.x": ts()})
        with pytest.raises(ConnectivityError) as exc:
            compile_plan(mods(lab, typo), Mode.FIT, {}, schema=schema)
        msg = str(exc.value)
        assert "'typo'" in msg
        assert "labels.jets.flavz" in msg
        assert "labels.jets.flav" in msg  # nearest schema key suggestion

    def test_schema_valid_narrowing_passes(self):
        schema = {"labels.jets.flav"}
        lab = WildToy("lab", produces={"labels.**": ts(kind="label")})
        t1 = Toy("t1", requires={"labels.jets.flav": ts(kind="label")}, produces={"o.x": ts()})
        plan = compile_plan(mods(lab, t1), Mode.FIT, {}, schema=schema)
        assert set(plan.step("lab").produces) == {"labels.jets.flav"}

    def test_narrowing_drops_keys_demanded_only_by_pruned_consumers(self):
        # narrowing runs against pre-prune demand; once t2 is demand-pruned in
        # FIT, the plan must not force 'w' to materialise labels.b (rule (c))
        w = WildToy("w", produces={"labels.**": ts(kind="label")})
        t1 = Toy("t1", requires={"labels.a": ts(kind="label")}, produces={"out.a": ts()})
        t2 = Toy("t2", requires={"labels.b": ts(kind="label")}, produces={"out.b": ts()})
        sinks = {Mode.FIT: ["out.a"], Mode.TEST: ["out.b"]}
        plan = compile_plan(mods(w, t1, t2), Mode.FIT, {}, sinks=sinks)
        assert "t2" not in plan.module_names
        assert set(plan.step("w").produces) == {"labels.a"}
        assert not any(edge.key == "labels.b" for edge in plan.edges)
        # and t2's demand comes back in TEST, where it survives pruning
        test_plan = compile_plan(mods(w, t1, t2), Mode.TEST, {}, sinks=sinks)
        assert set(test_plan.step("w").produces) == {"labels.b"}

    def test_narrowed_key_kept_for_surviving_optional_consumer(self):
        # t2's non-optional demand narrows labels.b; t2 is pruned in FIT but the
        # alive optional consumer t1 still binds it -> the key must stay
        w = WildToy("w", produces={"labels.**": ts(kind="label")})
        t1 = Toy(
            "t1",
            requires={"labels.a": ts(kind="label"), "labels.b": ts(kind="label", optional=True)},
            produces={"out.a": ts()},
        )
        t2 = Toy("t2", requires={"labels.b": ts(kind="label")}, produces={"out.b": ts()})
        sinks = {Mode.FIT: ["out.a"], Mode.TEST: ["out.b"]}
        plan = compile_plan(mods(w, t1, t2), Mode.FIT, {}, sinks=sinks)
        assert "t2" not in plan.module_names
        assert set(plan.step("w").produces) == {"labels.a", "labels.b"}
        assert Edge("w", "labels.b", "t1") in plan.edges
        assert set(plan.step("t1").requires) == {"labels.a", "labels.b"}


# determinism and plan hashing


class TestDeterminism:
    # Golden plan_hash for the fixed chain_ab/SRC_X graph. Guards the canonical
    # serialisation: a regression that lets set/dict iteration order (or
    # PYTHONHASHSEED) leak into the payload, or that changes the payload
    # structure, breaks checkpoint/repro hashes across machines and
    # must be a conscious, reviewed change. The hash must stay byte-identical
    # across processes regardless of PYTHONHASHSEED.
    GOLDEN_CHAIN_HASH = "2fc174f6a6870b4cd914816a0ca06b2d01ade488a28afc54c212dffa7f972e32"

    @staticmethod
    def _modules():
        a = Toy("a", requires={"inputs.x": ts()}, produces={"embed.x": ts()})
        b = Toy("b", requires={"embed.x": ts()}, produces={"left.x": ts()})
        c = Toy("c", requires={"embed.x": ts()}, produces={"right.x": ts()})
        d = Toy("d", requires={"left.x": ts(), "right.x": ts()}, produces={"preds.x": ts()})
        return a, b, c, d

    def test_reordering_independent_modules_swaps_plan_positions(self):
        # documented loudly: reordering independent modules in YAML is the
        # supported way to nudge execution order, and the plan hash detects it
        a, b, c, d = self._modules()
        plan_bc = compile_plan({"a": a, "b": b, "c": c, "d": d}, Mode.FIT, SRC_X)
        a2, b2, c2, d2 = self._modules()
        plan_cb = compile_plan({"a": a2, "c": c2, "b": b2, "d": d2}, Mode.FIT, SRC_X)
        assert plan_bc.module_names == ("a", "b", "c", "d")
        assert plan_cb.module_names == ("a", "c", "b", "d")
        assert plan_bc.plan_hash != plan_cb.plan_hash  # order change is detectable
        assert plan_bc.edges == plan_cb.edges  # same graph, different schedule

    def test_same_declaration_order_gives_identical_plan_and_hash(self):
        a, b, c, d = self._modules()
        plan1 = compile_plan({"a": a, "b": b, "c": c, "d": d}, Mode.FIT, SRC_X)
        a2, b2, c2, d2 = self._modules()
        plan2 = compile_plan({"a": a2, "b": b2, "c": c2, "d": d2}, Mode.FIT, SRC_X)
        assert plan1.module_names == plan2.module_names
        assert plan1.edges == plan2.edges
        assert plan1.plan_hash == plan2.plan_hash

    def test_hash_stable_across_compilations(self):
        a, b, c, d = self._modules()
        modules = mods(a, b, c, d)
        hash1 = compile_plan(modules, Mode.FIT, SRC_X).plan_hash
        hash2 = compile_plan(modules, Mode.FIT, SRC_X).plan_hash
        assert hash1 == hash2

    def test_golden_hash_pins_canonical_serialisation(self):
        a, b = chain_ab()
        assert compile_plan(mods(a, b), Mode.FIT, SRC_X).plan_hash == self.GOLDEN_CHAIN_HASH


class TestHashSensitivity:
    def test_shape_change_changes_hash(self):
        def build(dim):
            a = Toy("a", requires={"inputs.x": ts()}, produces={"embed.x": ts(shape=("B", dim))})
            b = Toy("b", requires={"embed.x": ts(shape=("B", dim))}, produces={"preds.x": ts()})
            return mods(a, b)

        hash128 = compile_plan(build(128), Mode.FIT, SRC_X).plan_hash
        hash256 = compile_plan(build(256), Mode.FIT, SRC_X).plan_hash
        assert hash128 != hash256

    def test_module_rename_changes_hash(self):
        a1 = Toy("a1", requires={"inputs.x": ts()}, produces={"embed.x": ts()})
        a2 = Toy("a2", requires={"inputs.x": ts()}, produces={"embed.x": ts()})
        hash1 = compile_plan(mods(a1), Mode.FIT, SRC_X).plan_hash
        hash2 = compile_plan(mods(a2), Mode.FIT, SRC_X).plan_hash
        assert hash1 != hash2

    def test_structurally_identical_plans_share_hash_across_modes(self):
        # the hash is purely structural, so the FIT/VAL
        # plan-identity assertion is a cheap hash comparison (Plan.mode still
        # distinguishes the plans)
        a, b = chain_ab()
        fit_plan = compile_plan(mods(a, b), Mode.FIT, SRC_X)
        val_plan = compile_plan(mods(a, b), Mode.VAL, SRC_X)
        assert fit_plan.plan_hash == val_plan.plan_hash
        assert fit_plan.mode != val_plan.mode

    def test_mode_divergent_structure_changes_hash(self):
        a, b = chain_ab()
        loss = Toy(
            "loss",
            requires={"preds.x": ts(modes=Mode.TRAINING)},
            produces={"losses.x": ts(kind="loss", modes=Mode.TRAINING)},
        )
        fit_hash = compile_plan(mods(a, b, loss), Mode.FIT, SRC_X).plan_hash
        test_hash = compile_plan(mods(a, b, loss), Mode.TEST, SRC_X).plan_hash
        assert fit_hash != test_hash

    def test_added_module_changes_hash(self):
        a, b = chain_ab()
        aux = Toy("aux", requires={"inputs.x": ts()}, produces={"aux.y": ts()})
        hash1 = compile_plan(mods(a, b), Mode.FIT, SRC_X).plan_hash
        hash2 = compile_plan(mods(a, b, aux), Mode.FIT, SRC_X).plan_hash
        assert hash1 != hash2


# config validation


class TestConfigValidation:
    def test_composite_mode_rejected(self):
        a, b = chain_ab()
        with pytest.raises(ConfigError, match="primary mode"):
            compile_plan(mods(a, b), Mode.TRAINING, SRC_X)

    def test_name_mismatch_rejected(self):
        a = Toy("a", requires={"inputs.x": ts()}, produces={"embed.x": ts()})
        with pytest.raises(ConfigError, match="instance names"):
            compile_plan({"wrong": a}, Mode.FIT, SRC_X)

    def test_non_graphmodule_rejected(self):
        with pytest.raises(ConfigError, match="GraphModule"):
            compile_plan({"x": object()}, Mode.FIT, SRC_X)  # type: ignore[dict-item]

    def test_wildcard_source_rejected(self):
        src = unflatten_spec({"inputs.*": TensorSpec()})
        with pytest.raises(ConfigError, match="sources are concrete"):
            compile_plan({}, Mode.FIT, src)

    def test_wildcard_sink_rejected(self):
        a, b = chain_ab()
        with pytest.raises(ConfigError, match="sinks are concrete"):
            compile_plan(mods(a, b), Mode.FIT, SRC_X, sinks=["preds.*"])

    @pytest.mark.parametrize("reserved", [SOURCES, SINKS])
    def test_sentinel_module_name_rejected(self, reserved):
        # a module named '<sources>'/'<sinks>' would collide with the planner
        # sentinels and corrupt edge resolution — reject with a ConfigError
        weird = Toy(reserved, produces={"weird.x": ts()})
        consumer = Toy("c", requires={"weird.x": ts()}, produces={"preds.x": ts()})
        with pytest.raises(ConfigError, match="reserved planner sentinel"):
            compile_plan(mods(weird, consumer), Mode.FIT, SRC_X)


# dead-output analysis


class TestDeadcode:
    def test_unconsumed_leaf_reported(self):
        a, b = chain_ab()
        aux = Toy("aux", requires={"inputs.x": ts()}, produces={"aux.y": ts()})
        report = deadcode(mods(a, b, aux), Mode.FIT, SRC_X)
        found = {(d.module, d.key) for d in report}
        assert ("aux", "aux.y") in found
        assert ("b", "preds.x") in found  # nothing consumes preds.x without sinks
        assert ("a", "embed.x") not in found

    def test_sink_consumption_silences_report(self):
        a, b = chain_ab()
        report = deadcode(mods(a, b), Mode.FIT, SRC_X, sinks=["preds.x"])
        assert not any(d.key == "preds.x" for d in report)

    def test_pruned_module_reported_whole(self):
        a, b = chain_ab()
        aux = Toy("aux", requires={"inputs.x": ts()}, produces={"aux.y": ts()})
        report = deadcode(mods(a, b, aux), Mode.FIT, SRC_X, sinks=["preds.x"])
        entry = next(d for d in report if d.module == "aux")
        assert entry.key == "*"
        assert "pruned" in entry.reason

    def test_mode_gated_absence_not_reported(self):
        a, b = chain_ab()
        loss = Toy(
            "loss",
            requires={"preds.x": ts(modes=Mode.TRAINING)},
            produces={"losses.x": ts(kind="loss", modes=Mode.TRAINING)},
        )
        report = deadcode(mods(a, b, loss), Mode.TEST, SRC_X)
        assert not any(d.module == "loss" for d in report)

    def test_unconsumed_source_reported(self):
        a, b = chain_ab()
        report = deadcode(mods(a, b), Mode.FIT, SRC_XY)
        assert DeadOutput(SOURCES, "inputs.y", "source never consumed in mode FIT") in report

    def test_clean_graph_reports_nothing(self):
        a, b = chain_ab()
        report = deadcode(mods(a, b), Mode.FIT, SRC_X, sinks=["preds.x"])
        assert report == []

    def test_unconsumed_preds_in_test_is_error_severity(self):
        # an unconsumed preds.* port in TEST is an error by default
        a, b = chain_ab()
        report = deadcode(mods(a, b), Mode.TEST, SRC_X)
        finding = next(d for d in report if d.key == "preds.x")
        assert finding.severity == "error"
        assert "never persisted" in finding.reason

    def test_unconsumed_preds_outside_test_is_info_severity(self):
        # the normal no-metric-callback case is INFO, never promoted by --strict
        a, b = chain_ab()
        report = deadcode(mods(a, b), Mode.FIT, SRC_X)
        finding = next(d for d in report if d.key == "preds.x")
        assert finding.severity == "info"
        assert "metric callback" in finding.reason

    def test_unconsumed_non_preds_in_test_is_warning_severity(self):
        a, b = chain_ab()
        aux = Toy("aux", requires={"inputs.x": ts()}, produces={"aux.y": ts()})
        report = deadcode(mods(a, b, aux), Mode.TEST, SRC_X)
        finding = next(d for d in report if d.key == "aux.y")
        assert finding.severity == "warning"

    def test_pruned_module_in_onnx_is_info_severity(self):
        # unified manifest (fix-stage regression): ONNX sinks are the
        # writer-declared export-manifest ports, so a module narrowed out of
        # the export surface (onnx_streams/onnx_tasks) is LEGITIMATE — info,
        # never promoted, keeping `validate --strict --mode onnx` usable on
        # narrowed configs. Other modes keep the warning default.
        a, b = chain_ab()
        aux = Toy("aux", requires={"inputs.x": ts()}, produces={"aux.y": ts()})
        onnx_finding = next(
            d
            for d in deadcode(mods(a, b, aux), Mode.ONNX, SRC_X, sinks=["preds.x"])
            if d.module == "aux" and d.key == "*"
        )
        assert onnx_finding.severity == "info"
        assert "export surface" in onnx_finding.reason
        fit_finding = next(
            d
            for d in deadcode(mods(a, b, aux), Mode.FIT, SRC_X, sinks=["preds.x"])
            if d.module == "aux" and d.key == "*"
        )
        assert fit_finding.severity == "warning"


# plan immutability (frozen plan)


class TestPlanImmutability:
    def test_step_and_source_mappings_are_read_only(self):
        a, b = chain_ab()
        plan = compile_plan(mods(a, b), Mode.FIT, SRC_X)
        with pytest.raises(TypeError):
            plan.step("a").produces["sneak.x"] = ts()  # type: ignore[index]
        with pytest.raises(TypeError):
            plan.step("a").requires["sneak.x"] = ts()  # type: ignore[index]
        with pytest.raises(TypeError):
            plan.sources["sneak.x"] = ts()  # type: ignore[index]

    def test_plan_and_steps_are_hashable(self):
        a, b = chain_ab()
        plan = compile_plan(mods(a, b), Mode.FIT, SRC_X)
        assert isinstance(hash(plan), int)
        assert isinstance(hash(plan.step("a")), int)
