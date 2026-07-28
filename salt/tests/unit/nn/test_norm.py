"""Unit tests for Normaliser and MaskedInputNormaliser (mirror of salt/model/modules/norm.py)."""

from __future__ import annotations

import pytest
import torch

from salt.graph import (
    Bundle,
    ConfigError,
    Mode,
    flatten_spec,
)
from salt.model.modules import (
    MaskedInputNormaliser,
    Normaliser,
    bind_all,
    resolve_bind_schema,
)
from salt.tests._fixtures.gn2v2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    build_gn2v2_modules,
    compile_gn2v2,
)
from salt.tests.unit.nn.conftest import fit_bundle

# construction + declare_io per module


class TestNormaliser:
    """Default fixed-norm-dict Normaliser: loads means/stds, preserves v1 parity."""

    def test_init_does_no_file_io(self, tmp_path):
        """__init__ records the path only — the file need not exist."""
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

    def test_materialise_fills_buffers_to_norm_dict_values(self, gn2v2):
        """Buffer values equal the parity norm dict's per-variable constants, in
        variable order (write_parity_norm_dict: mean 0.1*(i+1), std 1+0.05*(i+1))."""
        modules, _, _ = gn2v2
        norm = modules["norm"]
        exp_means_tracks = torch.tensor(
            [round(0.1 * (i + 1), 6) for i in range(len(TRACK_VARIABLES))], dtype=torch.float32
        )
        exp_stds_tracks = torch.tensor(
            [round(1.0 + 0.05 * (i + 1), 6) for i in range(len(TRACK_VARIABLES))],
            dtype=torch.float32,
        )
        exp_means_jets = torch.tensor(
            [round(0.1 * (i + 1), 6) for i in range(len(JET_VARIABLES))], dtype=torch.float32
        )
        assert torch.equal(norm.means_tracks, exp_means_tracks)
        assert torch.equal(norm.stds_tracks, exp_stds_tracks)
        assert torch.equal(norm.means_jets, exp_means_jets)
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

    def test_materialised_mirror_tracks_the_buffer(self, gn2v2, norm_paths):
        """The python mirror forward reads must agree with the buffer, always."""
        modules, _, _ = gn2v2
        norm = modules["norm"]
        assert norm._materialised_flag is bool(norm.materialised)  # noqa: SLF001

        fresh = build_gn2v2_modules(norm_paths[0])
        bind_all(fresh, resolve_bind_schema(compile_gn2v2(fresh, Mode.FIT)))
        assert fresh["norm"]._materialised_flag is bool(fresh["norm"].materialised)  # noqa: SLF001
        assert not fresh["norm"]._materialised_flag  # noqa: SLF001

    def test_materialised_mirror_restored_by_state_dict_load(self, gn2v2, norm_paths):
        """A checkpoint load must flip the mirror, not just the buffer."""
        modules, _, _ = gn2v2
        fresh = build_gn2v2_modules(norm_paths[0])
        bind_all(fresh, resolve_bind_schema(compile_gn2v2(fresh, Mode.FIT)))
        fresh["norm"].load_state_dict(modules["norm"].state_dict())
        assert bool(fresh["norm"].materialised)
        assert fresh["norm"]._materialised_flag  # noqa: SLF001
        # and forward no longer raises
        fresh["norm"](fit_bundle(), Mode.FIT)

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


class TestMaskedInputNormaliser:
    """Self-normalising MaskedInputNormaliser: online masked running stats (plan 01)."""

    @staticmethod
    def _bound_norm(streams, global_object=None, **kw):
        """Build + bind a standalone MaskedInputNormaliser over the GN2 fixture widths."""
        norm = MaskedInputNormaliser(streams=list(streams), global_object=global_object, **kw)
        norm.name = "norm"
        modules = build_gn2v2_modules("ignored.yaml")
        schema = resolve_bind_schema(compile_gn2v2(modules, Mode.FIT))
        norm.bind(schema)
        return norm

    def test_init_does_no_file_io(self, tmp_path):
        """__init__ records config only; norm_dict is ignored (no file read)."""
        norm = MaskedInputNormaliser(streams=["tracks"], norm_dict=tmp_path / "absent.yaml")
        norm.name = "norm"
        io = norm.declare_io(Mode.FIT)
        # FIT (training) declares the pad mask require alongside the input
        assert set(flatten_spec(io.requires)) == {"inputs.tracks", "masks.tracks"}
        assert set(flatten_spec(io.produces)) == {"normed.tracks"}

    def test_mask_require_is_training_only(self):
        """masks.<stream> is required in FIT/VAL but NOT TEST/ONNX (no mask at inference)."""
        norm = MaskedInputNormaliser(streams=["jets", "tracks"], global_object="jets")
        norm.name = "norm"
        flat = flatten_spec(norm.declare_io(Mode.FIT).requires)
        assert flat["masks.tracks"].modes == Mode.TRAINING
        assert flat["masks.tracks"].active_in(Mode.FIT)
        assert flat["masks.tracks"].active_in(Mode.VAL)
        assert not flat["masks.tracks"].active_in(Mode.TEST)
        assert not flat["masks.tracks"].active_in(Mode.ONNX)
        # the global object has no pad mask
        assert "masks.jets" not in flat

    def test_global_object_is_rank_two(self):
        norm = MaskedInputNormaliser(streams=["jets", "tracks"], global_object="jets")
        norm.name = "norm"
        flat = flatten_spec(norm.declare_io(Mode.FIT).requires)
        assert len(flat["inputs.jets"].shape) == 2
        assert len(flat["inputs.tracks"].shape) == 3

    def test_config_errors(self):
        with pytest.raises(ConfigError, match="non-empty"):
            MaskedInputNormaliser(streams=[])
        with pytest.raises(ConfigError, match="duplicate"):
            MaskedInputNormaliser(streams=["a", "a"])
        with pytest.raises(ConfigError, match="global_object"):
            MaskedInputNormaliser(streams=["a"], global_object="b")
        with pytest.raises(ConfigError, match="momentum"):
            MaskedInputNormaliser(streams=["a"], momentum=1.5)
        with pytest.raises(ConfigError, match="eps"):
            MaskedInputNormaliser(streams=["a"], eps=0.0)

    def test_buffers_init_to_identity(self):
        """A fresh (bound, untrained) Normaliser is identity: mean 0 / var 1, 0 batches."""
        norm = self._bound_norm(["jets", "tracks"], global_object="jets")
        assert torch.equal(norm.running_mean_tracks, torch.zeros_like(norm.running_mean_tracks))
        assert torch.equal(norm.running_var_tracks, torch.ones_like(norm.running_var_tracks))
        assert int(norm.num_batches_tracked_tracks) == 0
        assert int(norm.num_batches_tracked_jets) == 0

    def test_forward_produces_new_keys_and_never_mutates(self):
        """forward produces normed.* with running buffers and never touches inputs.*."""
        norm = self._bound_norm(["jets", "tracks"], global_object="jets")
        norm.eval()  # frozen apply, no update
        b = fit_bundle()
        before = b.get("inputs.tracks").clone()
        out = norm(b, Mode.TEST)
        assert set(out) == {"normed.jets", "normed.tracks"}
        assert torch.equal(b.get("inputs.tracks"), before)
        denom = torch.sqrt(norm.running_var_tracks + norm.eps)
        expected = (before - norm.running_mean_tracks) / denom
        assert torch.equal(out["normed.tracks"], expected)

    def test_padded_garbage_does_not_move_the_stats(self):
        """CORE REQUIREMENT: huge values in padded slots must NOT affect the stats."""
        norm_a = self._bound_norm(["tracks"])
        norm_b = self._bound_norm(["tracks"])
        torch.manual_seed(0)
        B_, T_, F_ = 8, 12, len(TRACK_VARIABLES)
        valid = torch.zeros(B_, T_, dtype=torch.bool)
        valid[:, :5] = True  # first 5 positions valid, rest padded
        x_valid = torch.randn(B_, T_, F_)
        pad_mask = ~valid

        xa = x_valid.clone()
        xa[pad_mask] = 0.0  # clean padding (like the real dump)
        xb = x_valid.clone()
        xb[pad_mask] = 1e6  # GARBAGE in the padded slots

        for norm, x in ((norm_a, xa), (norm_b, xb)):
            norm.train()
            b = Bundle()
            b.set("inputs.tracks", x)
            b.set("masks.tracks", pad_mask)
            norm(b, Mode.FIT)

        assert torch.allclose(norm_a.running_mean_tracks, norm_b.running_mean_tracks)
        assert torch.allclose(norm_a.running_var_tracks, norm_b.running_var_tracks)
        # and the stats equal the hand-computed valid-only moments (one EMA step from identity)
        flat_valid = x_valid[valid]  # [N_valid, F]
        batch_mean = flat_valid.mean(0)
        batch_var = flat_valid.var(0, unbiased=False)
        exp_mean = (1 - norm_a.momentum) * 0.0 + norm_a.momentum * batch_mean
        exp_var = (1 - norm_a.momentum) * 1.0 + norm_a.momentum * batch_var
        assert torch.allclose(norm_a.running_mean_tracks, exp_mean, atol=1e-5)
        assert torch.allclose(norm_a.running_var_tracks, exp_var, atol=1e-5)

    def test_fixed_momentum_follows_analytic_ema_trajectory(self):
        """Running buffers track the closed-form EMA over many stationary batches."""
        norm = self._bound_norm(["tracks"], momentum=0.1)
        norm.train()
        torch.manual_seed(1)
        B_, T_, F_ = 16, 10, len(TRACK_VARIABLES)
        valid = torch.ones(B_, T_, dtype=torch.bool)
        pad_mask = ~valid
        run_mean = torch.zeros(F_)
        run_var = torch.ones(F_)
        for _ in range(50):
            x = torch.randn(B_, T_, F_) * 2.0 + 3.0  # stationary-ish draws
            flat = x.reshape(-1, F_)
            run_mean = (1 - 0.1) * run_mean + 0.1 * flat.mean(0)
            run_var = (1 - 0.1) * run_var + 0.1 * flat.var(0, unbiased=False)
            b = Bundle()
            b.set("inputs.tracks", x)
            b.set("masks.tracks", pad_mask)
            norm(b, Mode.FIT)
        assert int(norm.num_batches_tracked_tracks) == 50
        assert torch.allclose(norm.running_mean_tracks, run_mean, atol=1e-5)
        assert torch.allclose(norm.running_var_tracks, run_var, atol=1e-5)

    def test_cumulative_converges_to_true_masked_dataset_stats(self):
        """momentum=None: running stats converge to the TRUE valid-only dataset mean/std."""
        norm = self._bound_norm(["tracks"], momentum=None)
        norm.train()
        torch.manual_seed(2)
        F_ = len(TRACK_VARIABLES)
        all_valid = []
        for _ in range(200):
            B_, T_ = 16, 10
            x = torch.randn(B_, T_, F_) * 1.5 - 0.5
            valid = torch.rand(B_, T_) > 0.4  # ~60% valid
            pad_mask = ~valid
            x[pad_mask] = 1e5  # garbage padding that MUST be excluded
            all_valid.append(x[valid])
            b = Bundle()
            b.set("inputs.tracks", x)
            b.set("masks.tracks", pad_mask)
            norm(b, Mode.FIT)
        truth = torch.cat(all_valid, dim=0)
        true_mean = truth.mean(0)
        true_var = truth.var(0, unbiased=False)
        assert torch.allclose(norm.running_mean_tracks, true_mean, atol=1e-3)
        assert torch.allclose(norm.running_var_tracks, true_var, rtol=1e-2, atol=1e-2)

    def test_eval_does_not_update(self):
        """In eval (or TEST/ONNX mode) the running stats are frozen."""
        norm = self._bound_norm(["tracks"])
        norm.train()
        b = fit_bundle()
        norm(b, Mode.FIT)  # one update from identity
        m0 = norm.running_mean_tracks.clone()
        v0 = norm.running_var_tracks.clone()
        n0 = int(norm.num_batches_tracked_tracks)
        # eval mode: no update even on a FIT-mode plan
        norm.eval()
        norm(fit_bundle(), Mode.FIT)
        assert torch.equal(norm.running_mean_tracks, m0)
        assert torch.equal(norm.running_var_tracks, v0)
        assert int(norm.num_batches_tracked_tracks) == n0
        # train() but TEST mode: still no update (mode-gated)
        norm.train()
        norm(fit_bundle(), Mode.TEST)
        assert torch.equal(norm.running_mean_tracks, m0)
        assert int(norm.num_batches_tracked_tracks) == n0

    def test_global_stream_no_mask_path(self):
        """The global object (no pad mask) updates over all rows."""
        norm = self._bound_norm(["jets", "tracks"], global_object="jets")
        norm.train()
        b = fit_bundle()
        jets = b.get("inputs.jets")
        norm(b, Mode.FIT)
        exp_mean = norm.momentum * jets.mean(0)
        exp_var = (1 - norm.momentum) * 1.0 + norm.momentum * jets.var(0, unbiased=False)
        assert torch.allclose(norm.running_mean_jets, exp_mean, atol=1e-6)
        assert torch.allclose(norm.running_var_jets, exp_var, atol=1e-6)
        assert int(norm.num_batches_tracked_jets) == 1

    def test_empty_valid_batch_is_a_noop(self):
        """A fully-padded stream skips the update (no NaNs, counter unchanged)."""
        norm = self._bound_norm(["tracks"])
        norm.train()
        B_, T_, F_ = 4, 6, len(TRACK_VARIABLES)
        x = torch.zeros(B_, T_, F_)
        pad_mask = torch.ones(B_, T_, dtype=torch.bool)  # everything padded
        b = Bundle()
        b.set("inputs.tracks", x)
        b.set("masks.tracks", pad_mask)
        out = norm(b, Mode.FIT)
        assert torch.equal(norm.running_mean_tracks, torch.zeros(F_))
        assert torch.equal(norm.running_var_tracks, torch.ones(F_))
        assert int(norm.num_batches_tracked_tracks) == 0
        assert torch.isfinite(out["normed.tracks"]).all()

    def test_checkpoint_round_trip_restores_buffers(self):
        """Buffers ride in the state_dict; reload restores them (no warmup needed)."""
        norm = self._bound_norm(["jets", "tracks"], global_object="jets")
        norm.train()
        for _ in range(5):
            norm(fit_bundle(), Mode.FIT)
        sd = norm.state_dict()
        assert "running_mean_tracks" in sd
        assert "running_var_tracks" in sd
        assert "num_batches_tracked_tracks" in sd

        fresh = self._bound_norm(["jets", "tracks"], global_object="jets")
        # before load: identity
        assert not torch.equal(fresh.running_mean_tracks, norm.running_mean_tracks)
        fresh.load_state_dict(sd)
        assert torch.equal(fresh.running_mean_tracks, norm.running_mean_tracks)
        assert torch.equal(fresh.running_var_tracks, norm.running_var_tracks)
        assert int(fresh.num_batches_tracked_tracks) == int(norm.num_batches_tracked_tracks)
        # frozen eval forward matches the original
        fresh.eval()
        norm.eval()
        b = fit_bundle()
        assert torch.equal(fresh(b, Mode.TEST)["normed.tracks"], norm(b, Mode.TEST)["normed.tracks"])
