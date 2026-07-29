"""Tests for salt.graph.executor."""

import pytest
import torch
from torch import nn

from salt.graph.bundle import Bundle
from salt.graph.errors import (
    ConfigError,
    DeclarationError,
    KeyCollisionError,
    MutationError,
    UndeclaredAccessError,
)
from salt.graph.executor import (
    STEP_SCOPE_PREFIX,
    Executor,
    _StepRecording,
    record_steps,
)
from salt.graph.planner import compile_plan
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec

# toy fixtures (no physics — M1 scope)


def ts(**kwargs):
    return TensorSpec(**kwargs)


class ToyModule:
    """Minimal callable GraphModule: flat dotted spec dicts + a forward fn."""

    def __init__(self, name, requires=None, produces=None, fn=None):
        self.name = name
        self._requires = requires or {}
        self._produces = produces or {}
        self._fn = fn or (lambda b, mode: {})

    def declare_io(self, mode):
        return IO(unflatten_spec(self._requires), unflatten_spec(self._produces))

    def __call__(self, b, mode):
        return self._fn(b, mode)


class LinearToy(nn.Module):
    """nn.Module GraphModule: the call lands in forward(b, mode)."""

    def __init__(self, name, in_key, out_key, dim=4):
        super().__init__()
        self.name = name
        self._in = in_key
        self._out = out_key
        self.lin = nn.Linear(dim, dim)

    def declare_io(self, mode):
        return IO(
            unflatten_spec({self._in: TensorSpec()}),
            unflatten_spec({self._out: TensorSpec()}),
        )

    def forward(self, b, mode):
        return {self._out: self.lin(b.get(self._in))}


def mods(*modules):
    return {m.name: m for m in modules}


SRC_X = unflatten_spec({"inputs.x": TensorSpec(shape=("B", 4), dtype="float32")})
SINKS_BY_MODE = {Mode.TRAINING: ["loss.total"], Mode.TEST: ["preds.x"]}


def diamond_modules():
    """Diamond graph: embed -> (left, right) -> head, plus a TRAINING-gated loss sum."""
    embed = ToyModule(
        "embed",
        requires={"inputs.x": ts()},
        produces={"embed.x": ts()},
        fn=lambda b, mode: {"embed": {"x": b.get("inputs.x") * 2.0}},  # nested spelling
    )
    left = ToyModule(
        "left",
        requires={"embed.x": ts()},
        produces={"mid.left": ts()},
        fn=lambda b, mode: {"mid.left": b.get("embed.x") + 1.0},  # dotted spelling
    )
    right = ToyModule(
        "right",
        requires={"embed.x": ts()},
        produces={"mid.right": ts()},
        fn=lambda b, mode: {"mid.right": b.get("embed.x") - 1.0},
    )

    def head_fn(b, mode):
        y = b.get("mid.left") + b.get("mid.right")
        out = {"preds.x": y}
        if mode & Mode.TRAINING:
            out["losses.head"] = y.mean()
        return out

    head = ToyModule(
        "head",
        requires={"mid.left": ts(), "mid.right": ts()},
        produces={"preds.x": ts(), "losses.head": ts(kind="loss", modes=Mode.TRAINING)},
        fn=head_fn,
    )
    loss_sum = ToyModule(
        "loss_sum",
        requires={"losses.head": ts(kind="loss", modes=Mode.TRAINING)},
        produces={"loss.total": ts(kind="loss", modes=Mode.TRAINING)},
        fn=lambda b, mode: {"loss": {"total": b.get("losses.head") * 1.0}},
    )
    return embed, left, right, head, loss_sum


def input_bundle(x=None):
    return Bundle({"inputs": {"x": torch.ones(2, 4) if x is None else x}})


# happy path


class TestHappyPathDiamond:
    def test_fit_run_produces_all_declared_keys(self):
        plan = compile_plan(mods(*diamond_modules()), Mode.FIT, SRC_X, sinks=SINKS_BY_MODE)
        x = torch.ones(2, 4)
        b = input_bundle(x)
        out = Executor(plan).run(b)
        assert out is b  # run returns the bundle
        assert set(out.keys()) == {
            "inputs.x",
            "embed.x",
            "mid.left",
            "mid.right",
            "preds.x",
            "losses.head",
            "loss.total",
        }
        # (2x + 1) + (2x - 1) == 4x
        assert torch.allclose(out.get("preds.x"), 4.0 * x)
        assert torch.allclose(out.get("loss.total"), torch.tensor(4.0))

    def test_debug_run_of_clean_graph_passes(self):
        plan = compile_plan(mods(*diamond_modules()), Mode.FIT, SRC_X, sinks=SINKS_BY_MODE)
        out = Executor(plan).run(input_bundle(), debug=True)
        assert "loss.total" in out

    def test_dict_valued_declared_leaf_kept_whole(self):
        layout = {"streams": ["x"], "lengths": [4]}
        concat = ToyModule(
            "concat",
            requires={"inputs.x": ts()},
            produces={"seq.x": ts(), "seq.layout": ts(kind="meta")},
            fn=lambda b, mode: {"seq": {"x": b.get("inputs.x"), "layout": layout}},
        )
        plan = compile_plan(mods(concat), Mode.FIT, SRC_X)
        out = Executor(plan).run(input_bundle())
        assert out.get("seq.layout") == layout

    def test_missing_source_leaf_raises_keyerror(self):
        plan = compile_plan(mods(*diamond_modules()), Mode.FIT, SRC_X, sinks=SINKS_BY_MODE)
        with pytest.raises(KeyError, match="missing source leaves.*inputs.x"):
            Executor(plan).run(Bundle())


# declaration enforcement on merge (always on)


class TestDeclarationEnforcement:
    def test_undeclared_extra_key_raises_with_module_name(self):
        bad = ToyModule(
            "bad",
            requires={"inputs.x": ts()},
            produces={"embed.x": ts()},
            fn=lambda b, mode: {"embed": {"x": b.get("inputs.x"), "extra": b.get("inputs.x")}},
        )
        plan = compile_plan(mods(bad), Mode.FIT, SRC_X)
        with pytest.raises(DeclarationError, match="'bad'.*embed.extra"):
            Executor(plan).run(input_bundle())

    def test_omitted_declared_key_raises_with_module_name(self):
        forgetful = ToyModule(
            "forgetful",
            requires={"inputs.x": ts()},
            produces={"preds.x": ts(), "losses.x": ts(kind="loss")},
            fn=lambda b, mode: {"preds.x": b.get("inputs.x")},  # forgets losses.x
        )
        plan = compile_plan(mods(forgetful), Mode.FIT, SRC_X)
        with pytest.raises(DeclarationError, match="'forgetful'.*losses.x"):
            Executor(plan).run(input_bundle())

    def test_non_dict_return_raises(self):
        rogue = ToyModule(
            "rogue",
            requires={"inputs.x": ts()},
            produces={"preds.x": ts()},
            fn=lambda b, mode: b.get("inputs.x"),  # returns a tensor, not a dict
        )
        plan = compile_plan(mods(rogue), Mode.FIT, SRC_X)
        with pytest.raises(DeclarationError, match="'rogue' returned Tensor"):
            Executor(plan).run(input_bundle())

    def test_duplicate_spellings_of_one_key_raise(self):
        double = ToyModule(
            "double",
            requires={"inputs.x": ts()},
            produces={"preds.x": ts()},
            fn=lambda b, mode: {"preds.x": b.get("inputs.x"), "preds": {"x": 0}},
        )
        plan = compile_plan(mods(double), Mode.FIT, SRC_X)
        with pytest.raises(DeclarationError, match="'double'.*more than once"):
            Executor(plan).run(input_bundle())

    def test_write_once_collision_through_executor(self):
        plan = compile_plan(mods(*diamond_modules()), Mode.FIT, SRC_X, sinks=SINKS_BY_MODE)
        b = input_bundle()
        b.set("mid.left", torch.zeros(2, 4))  # pre-populated by a misbehaving caller
        with pytest.raises(KeyCollisionError, match="mid.left.*'left'"):
            Executor(plan).run(b)


# debug mode: read tracking


class TestDebugReadTracking:
    def snoop_plan(self):
        embed, *_ = diamond_modules()
        snoop = ToyModule(
            "snoop",
            requires={"embed.x": ts()},
            produces={"preds.snoop": ts()},
            # reads inputs.x without declaring it
            fn=lambda b, mode: {"preds.snoop": b.get("embed.x") + b.get("inputs.x")},
        )
        return compile_plan(mods(embed, snoop), Mode.FIT, SRC_X)

    def test_undeclared_read_caught_in_debug(self):
        with pytest.raises(UndeclaredAccessError) as exc:
            Executor(self.snoop_plan()).run(input_bundle(), debug=True)
        msg = str(exc.value)
        assert "'snoop'" in msg
        assert "'inputs.x'" in msg
        assert "declare_io" in msg  # names the declaration to amend

    def test_same_graph_passes_without_debug(self):
        out = Executor(self.snoop_plan()).run(input_bundle())
        assert "preds.snoop" in out

    def test_undeclared_probe_caught_in_debug(self):
        prober = ToyModule(
            "prober",
            requires={"inputs.x": ts()},
            produces={"preds.x": ts()},
            fn=lambda b, mode: {"preds.x": b.get("inputs.x") if "secret" in b else None},
        )
        plan = compile_plan(mods(prober), Mode.FIT, SRC_X)
        with pytest.raises(UndeclaredAccessError, match="'prober' probed.*'secret'"):
            Executor(plan).run(input_bundle(), debug=True)

    def test_subtree_with_undeclared_leaf_caught_in_debug(self):
        embed, left, right, *_ = diamond_modules()
        sub = ToyModule(
            "sub",
            requires={"mid.left": ts()},  # mid.right NOT declared
            produces={"preds.sub": ts()},
            fn=lambda b, mode: {"preds.sub": b.subtree("mid")["left"]},
        )
        plan = compile_plan(mods(embed, left, right, sub), Mode.FIT, SRC_X)
        executor = Executor(plan)
        assert "preds.sub" in executor.run(input_bundle())  # non-debug: fine
        with pytest.raises(UndeclaredAccessError, match="'sub'.*'mid.right'"):
            executor.run(input_bundle(), debug=True)

    def test_view_blocks_set_and_data(self):
        writer = ToyModule(
            "writer",
            requires={"inputs.x": ts()},
            produces={"preds.x": ts()},
            fn=lambda b, mode: b.set("preds.x", b.get("inputs.x")),
        )
        plan = compile_plan(mods(writer), Mode.FIT, SRC_X)
        with pytest.raises(UndeclaredAccessError, match="'writer' called set"):
            Executor(plan).run(input_bundle(), debug=True)

        grabber = ToyModule(
            "grabber",
            requires={"inputs.x": ts()},
            produces={"preds.x": ts()},
            fn=lambda b, mode: {"preds.x": b.data["inputs"]["x"]},
        )
        plan = compile_plan(mods(grabber), Mode.FIT, SRC_X)
        with pytest.raises(UndeclaredAccessError, match="'grabber'.*Bundle.data"):
            Executor(plan).run(input_bundle(), debug=True)


# debug mode: in-place mutation detection


class TestMutationDetection:
    @staticmethod
    def modules(mutator_fn):
        embed = ToyModule(
            "embed",
            requires={"inputs.x": ts()},
            produces={"embed.x": ts()},
            fn=lambda b, mode: {"embed.x": b.get("inputs.x") * 2.0},
        )
        mut = ToyModule(
            "mut",
            requires={"embed.x": ts()},
            produces={"preds.x": ts()},
            fn=mutator_fn,
        )
        return mods(embed, mut)

    def test_inplace_mutation_caught_in_debug(self):
        modules = self.modules(lambda b, mode: {"preds.x": b.get("embed.x").add_(1.0)})
        plan = compile_plan(modules, Mode.FIT, SRC_X)
        with pytest.raises(MutationError) as exc:
            Executor(plan).run(input_bundle(), debug=True)
        msg = str(exc.value)
        assert "'mut'" in msg
        assert "'embed.x'" in msg
        assert "clone" in msg  # names the fix

    def test_same_graph_passes_without_debug(self):
        # documented gap: mutation detection is debug-only (version snapshots)
        modules = self.modules(lambda b, mode: {"preds.x": b.get("embed.x").add_(1.0)})
        plan = compile_plan(modules, Mode.FIT, SRC_X)
        out = Executor(plan).run(input_bundle())
        assert "preds.x" in out

    def test_clone_before_mutating_is_legal_in_debug(self):
        modules = self.modules(lambda b, mode: {"preds.x": b.get("embed.x").clone().add_(1.0)})
        plan = compile_plan(modules, Mode.FIT, SRC_X)
        out = Executor(plan).run(input_bundle(), debug=True)
        assert torch.allclose(out.get("preds.x"), out.get("embed.x") + 1.0)


# optional ports (consumed if present, absent = omitted)


class TestOptionalPorts:
    def optional_consumer(self):
        def fn(b, mode):
            x = b.get("inputs.x")
            if "extras.bias" in b:  # the optional-port idiom
                x = x + b.get("extras.bias")
            return {"preds.x": x}

        return ToyModule(
            "opt",
            requires={"inputs.x": ts(), "extras.bias": ts(optional=True)},
            produces={"preds.x": ts()},
            fn=fn,
        )

    def test_absent_optional_is_omitted_and_probe_is_legal_in_debug(self):
        # no producer for extras.bias -> port dropped from the plan
        plan = compile_plan(mods(self.optional_consumer()), Mode.FIT, SRC_X)
        assert "extras.bias" not in plan.step("opt").requires
        x = torch.ones(2, 4)
        out = Executor(plan).run(input_bundle(x), debug=True)
        assert torch.equal(out.get("preds.x"), x)

    def test_bound_optional_is_consumed_in_debug(self):
        biaser = ToyModule(
            "biaser",
            requires={"inputs.x": ts()},
            produces={"extras.bias": ts()},
            fn=lambda b, mode: {"extras.bias": torch.ones(2, 4)},
        )
        plan = compile_plan(mods(self.optional_consumer(), biaser), Mode.FIT, SRC_X)
        assert "extras.bias" in plan.step("opt").requires
        x = torch.ones(2, 4)
        out = Executor(plan).run(input_bundle(x), debug=True)
        assert torch.equal(out.get("preds.x"), x + 1.0)


# per-mode execution (losses only in TRAINING plans)


class TestPerModeExecution:
    def test_fit_plan_computes_losses_test_plan_does_not(self):
        modules = mods(*diamond_modules())
        fit_plan = compile_plan(modules, Mode.FIT, SRC_X, sinks=SINKS_BY_MODE)
        test_plan = compile_plan(modules, Mode.TEST, SRC_X, sinks=SINKS_BY_MODE)
        assert "loss_sum" in fit_plan.module_names
        assert "loss_sum" not in test_plan.module_names

        fit_out = Executor(fit_plan).run(input_bundle())
        assert "losses.head" in fit_out
        assert "loss.total" in fit_out

        test_out = Executor(test_plan).run(input_bundle())
        assert "preds.x" in test_out
        assert "losses.head" not in test_out
        assert "loss.total" not in test_out


# determinism (frozen plan + seeded torch -> identical outputs)


class TestDeterminism:
    @staticmethod
    def build_and_run(seed):
        torch.manual_seed(seed)
        lin = LinearToy("lin", "inputs.x", "preds.x")
        plan = compile_plan(mods(lin), Mode.FIT, SRC_X)
        out = Executor(plan).run(input_bundle())
        return plan, out.get("preds.x")

    def test_same_seed_same_outputs(self):
        plan_a, out_a = self.build_and_run(seed=7)
        plan_b, out_b = self.build_and_run(seed=7)
        assert plan_a.plan_hash == plan_b.plan_hash
        assert torch.equal(out_a, out_b)

    def test_different_seed_different_outputs(self):
        _, out_a = self.build_and_run(seed=7)
        _, out_b = self.build_and_run(seed=8)
        assert not torch.equal(out_a, out_b)


# constructor validation


class TestExecutorConstruction:
    def test_superset_modules_mapping_is_fine(self):
        modules = mods(*diamond_modules())
        test_plan = compile_plan(modules, Mode.TEST, SRC_X, sinks=SINKS_BY_MODE)
        # loss_sum is not in the TEST plan but may stay in the mapping
        out = Executor(test_plan, modules).run(input_bundle())
        assert "preds.x" in out

    def test_missing_module_raises(self):
        modules = mods(*diamond_modules())
        plan = compile_plan(modules, Mode.FIT, SRC_X, sinks=SINKS_BY_MODE)
        incomplete = {name: mod for name, mod in modules.items() if name != "head"}
        with pytest.raises(ConfigError, match="no entry 'head'"):
            Executor(plan, incomplete)

    def test_name_mismatch_raises(self):
        a = ToyModule("a", requires={"inputs.x": ts()}, produces={"preds.x": ts()})
        plan = compile_plan(mods(a), Mode.FIT, SRC_X)
        imposter = ToyModule("zzz", requires={"inputs.x": ts()}, produces={"preds.x": ts()})
        with pytest.raises(ConfigError, match="name='zzz'"):
            Executor(plan, {"a": imposter})

    def test_non_callable_module_raises(self):
        class Inert:
            name = "a"

            def declare_io(self, mode):
                return IO(
                    unflatten_spec({"inputs.x": ts()}), unflatten_spec({"preds.x": ts()})
                )

        plan = compile_plan({"a": Inert()}, Mode.FIT, SRC_X)
        with pytest.raises(ConfigError, match="'a'.*not callable"):
            Executor(plan)


class TestRecordSteps:
    """The profiler step scopes (per-module attribution)."""

    @staticmethod
    def _run_under_profiler(enabled):
        modules = mods(*diamond_modules())
        plan = compile_plan(modules, Mode.FIT, SRC_X, sinks=SINKS_BY_MODE)
        executor = Executor(plan, modules)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            if enabled:
                with record_steps():
                    executor.run(input_bundle())
            else:
                executor.run(input_bundle())
        return {str(evt.key) for evt in prof.key_averages()}

    def test_scopes_absent_by_default(self):
        keys = self._run_under_profiler(enabled=False)
        assert not any(key.startswith(STEP_SCOPE_PREFIX) for key in keys)

    def test_scopes_named_after_plan_steps(self):
        keys = self._run_under_profiler(enabled=True)
        recorded = {
            key[len(STEP_SCOPE_PREFIX) :]
            for key in keys
            if key.startswith(STEP_SCOPE_PREFIX)
        }
        assert {"head", "loss_sum"} <= recorded

    def test_toggle_restores_previous_state(self):
        with record_steps():
            with record_steps():
                pass
            assert _StepRecording.enabled
        assert not _StepRecording.enabled
