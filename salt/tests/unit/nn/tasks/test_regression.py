"""Unit tests for RegressionTaskModule."""

from __future__ import annotations

import pytest
import torch

from salt.core.graph import (
    Bundle,
    ConfigError,
    Executor,
    Mode,
    flatten_spec,
)
from salt.core.nn import (
    ResolvedSchema,
    bind_all,
    materialise_all,
    resolve_bind_schema,
)
from salt.core.nn.tasks import (
    RegressionTaskModule,
)
from salt.tests._fixtures.gn2v2_fixture import (
    JET_VARIABLES,
    make_gn2_batch,
)
from salt.tests._fixtures.v2_builders import (
    build_regression_modules,
    compile_regression,
    make_regression_labels,
)
from salt.tests.unit.nn.conftest import B, T

# RegressionTaskModule (M5 sub-wave A: targets/denom/norm_params/scaler,
# custom_output_names, sequence, multi-output, mode-split de-scaling)


def _bind_reg_module(task: RegressionTaskModule, schema_widths: dict[str, int]) -> None:
    """Bind a standalone regression head against fixed widths (no fields)."""
    task.bind(ResolvedSchema(widths=schema_widths))


class TestRegressionTaskModule:
    def test_targets_required(self):
        with pytest.raises(ConfigError, match="targets is required"):
            RegressionTaskModule(stream="jets", targets=[])

    def test_single_scaling_method_guard(self):
        # v1 task.py:355 — at most one of denom/norm_params/scaler
        with pytest.raises(ConfigError, match="single scaling method"):
            RegressionTaskModule(
                stream="jets",
                targets="x",
                norm_params={"mean": 1.0, "std": 1.0},
                target_denominators="y",
            )

    def test_custom_output_name_count_mismatch(self):
        with pytest.raises(ConfigError, match="custom_output_names"):
            RegressionTaskModule(stream="jets", targets=["a", "b"], custom_output_names="only_one")

    def test_denominator_count_mismatch(self):
        with pytest.raises(ConfigError, match="target_denominators"):
            RegressionTaskModule(stream="jets", targets=["a", "b"], target_denominators="d")

    def test_norm_params_requires_mean_and_std(self):
        with pytest.raises(ConfigError, match="mean.*std|norm_params"):
            RegressionTaskModule(stream="jets", targets="x", norm_params={"mean": 1.0})

    def test_sequence_inference(self):
        seq = RegressionTaskModule(stream="tracks", targets="x")
        glob = RegressionTaskModule(stream="jets", targets="x", input="pooled.global")
        assert seq.sequence and not glob.sequence

    def test_output_suffixes_custom_override(self):
        task = RegressionTaskModule(
            stream="jets", targets=["mass", "pt"], custom_output_names=["truthMass", "truthPt"]
        )
        assert task.output_suffixes == ("truthMass", "truthPt")
        bare = RegressionTaskModule(stream="jets", targets=["mass", "pt"])
        assert bare.output_suffixes == ("mass", "pt")

    def test_declare_io_mode_gating_global(self):
        task = RegressionTaskModule(stream="jets", targets=["t1", "t2"], input="pooled.global")
        task.name = "regression"
        req = flatten_spec(task.declare_io(Mode.FIT).requires)
        assert set(req) == {"pooled.global", "labels.jets.t1", "labels.jets.t2"}
        assert req["labels.jets.t1"].kind == "label"
        assert req["labels.jets.t1"].modes == Mode.TRAINING
        produces = flatten_spec(task.declare_io(Mode.FIT).produces)
        assert produces["preds.jets.regression"].modes == Mode.ALL
        assert produces["preds.jets.regression"].shape == ("B", 2)
        assert produces["losses.regression"].modes == Mode.TRAINING

    def test_declare_io_sequence_shapes(self):
        task = RegressionTaskModule(stream="tracks", targets=["a", "b", "c"])
        task.name = "regression"
        req = flatten_spec(task.declare_io(Mode.FIT).requires)
        assert "masks.tracks" in req and req["masks.tracks"].kind == "pad_mask"
        pred = flatten_spec(task.declare_io(Mode.TEST).produces)["preds.tracks.regression"]
        assert pred.shape == ("B", "T:tracks", 3)

    def test_mode_split_denominator_dependency(self):
        """FIT|VAL|TEST demand labels.<denom>; ONNX adds the input Feature."""
        task = RegressionTaskModule(
            stream="jets",
            targets="HadronConeExclTruthLabelPt",
            input="pooled.global",
            target_denominators="pt_btagJes",
        )
        task.name = "regression"
        for mode in (Mode.FIT, Mode.TEST):
            req = flatten_spec(task.declare_io(mode).requires)
            assert "labels.jets.pt_btagJes" in req
            assert "inputs.jets" not in req
        onnx_req = flatten_spec(task.declare_io(Mode.ONNX).requires)
        assert "inputs.jets" in onnx_req
        # the denominator label leaf is excluded from ONNX (modes gate it)
        assert onnx_req["labels.jets.pt_btagJes"].modes == Mode.FIT | Mode.VAL | Mode.TEST

    def test_bind_infers_widths_and_output_size(self):
        task = RegressionTaskModule(
            stream="jets", targets=["a", "b"], input="pooled.global", context="ctx"
        )
        task.name = "regression"
        _bind_reg_module(task, {"pooled.global": 16, "ctx": 8})
        assert task.net.input_size == 16
        assert task.net.context_size == 8
        assert task.net.output_size == 2

    def test_bind_rejects_denominator_not_an_input_feature(self):
        """A ratio denominator absent from inputs.<stream> Features fails at bind."""
        task = RegressionTaskModule(
            stream="jets",
            targets="t",
            input="pooled.global",
            target_denominators="not_a_feature",
        )
        task.name = "regression"
        schema = ResolvedSchema(
            widths={"pooled.global": 16, "inputs.jets": len(JET_VARIABLES)},
            fields={"inputs.jets": tuple(JET_VARIABLES)},
        )
        with pytest.raises(ConfigError, match="ONNX export graph de-scales"):
            task.bind(schema)

    def test_fit_forward_parity_norm_params(self, norm_paths):
        """FIT preds are RAW/scaled and loss == v1 head, scaled space (norm_params)."""
        targets = ("R10TruthLabel_R22v1_TruthJetMass", "R10TruthLabel_R22v1_TruthJetPt")
        task = RegressionTaskModule(
            stream="jets",
            targets=list(targets),
            input="pooled.global",
            norm_params={"mean": [1.0, 2.0], "std": [3.0, 4.0]},
            weight=0.5,
        )
        modules = build_regression_modules(norm_paths[0], task)
        fit = compile_regression(modules, Mode.FIT, targets)
        bind_all(modules, resolve_bind_schema([fit]))
        materialise_all(modules)
        inputs, masks = make_gn2_batch(B, T)
        labels = make_regression_labels(B, targets)
        b = Bundle()
        for stream, x in inputs.items():
            b.set(f"inputs.{stream}", x)
        b.set("masks.tracks", masks["tracks"])
        for key, val in labels.items():
            b.set(key, val)
        out = Executor(fit).run(b, debug=True)
        v2_pred = out.get("preds.jets.regression")
        v2_loss = out.get("losses.regression")
        # call the head math directly (head_forward) for the reference
        pooled = out.get("pooled.global")
        tdict = {"jets": {t: labels[f"labels.jets.{t}"] for t in targets}}
        ref_pred, ref_loss = task.head_forward(pooled, tdict, None, context=None)
        assert torch.allclose(v2_pred, ref_pred, atol=1e-6)  # FIT preds are raw/scaled
        assert torch.allclose(v2_loss, ref_loss, atol=1e-6)

    def test_test_descale_uses_label_denominator(self, norm_paths):
        """TEST forward = RAW scaled preds (W34.3 flip); get_h5 de-scales via the LABEL denom."""
        targets, denoms = ("HadronConeExclTruthLabelPt",), ("pt_btagJes",)
        task = RegressionTaskModule(
            stream="jets",
            targets=list(targets),
            input="pooled.global",
            target_denominators=list(denoms),
            custom_output_names="pt",
        )
        modules = build_regression_modules(norm_paths[0], task)
        fit = compile_regression(modules, Mode.FIT, targets, denoms)
        test = compile_regression(modules, Mode.TEST, targets, denoms)
        onnx = compile_regression(modules, Mode.ONNX, targets, denoms)
        bind_all(modules, resolve_bind_schema([fit, test, onnx]))
        materialise_all(modules)
        inputs, masks = make_gn2_batch(B, T)
        labels = make_regression_labels(B, targets, denoms)
        b = Bundle()
        for stream, x in inputs.items():
            b.set(f"inputs.{stream}", x)
        b.set("masks.tracks", masks["tracks"])
        for key, val in labels.items():
            b.set(key, val)
        with torch.no_grad():
            b = Executor(test).run(b)
        v2_test = b.get("preds.jets.regression")
        pooled = b.get("pooled.global")
        with torch.no_grad():
            raw, _ = task.head_forward(pooled, {}, None, context=None)
        # W34.3: the TEST forward publishes the RAW scaled preds (NO de-scale)
        assert torch.allclose(v2_test, raw, atol=1e-6)
        # get_h5 now owns the de-scale (label-sourced denominator, v1 get_h5)
        h5 = task.get_h5(b, run_name="reg")
        with torch.no_grad():
            ref = task.run_inference(
                raw.clone(), labels={"jets": {"pt_btagJes": labels["labels.jets.pt_btagJes"]}}
            )
        assert torch.allclose(torch.as_tensor(h5["reg_pt"]), ref[..., 0], atol=1e-6)

    def test_onnx_descale_uses_input_feature_by_name(self, norm_paths):
        """ONNX de-scales with the denominator gathered BY NAME from inputs.<stream>."""
        targets, denoms = ("HadronConeExclTruthLabelPt",), ("pt_btagJes",)
        task = RegressionTaskModule(
            stream="jets",
            targets=list(targets),
            input="pooled.global",
            target_denominators=list(denoms),
            write_targets=False,  # prediction de-scale gate; labels: test_target_labels.py
        )
        modules = build_regression_modules(norm_paths[0], task)
        fit = compile_regression(modules, Mode.FIT, targets, denoms)
        test = compile_regression(modules, Mode.TEST, targets, denoms)
        onnx = compile_regression(modules, Mode.ONNX, targets, denoms)
        bind_all(modules, resolve_bind_schema([fit, test, onnx]))
        materialise_all(modules)
        inputs, masks = make_gn2_batch(B, T)
        labels = make_regression_labels(B, targets, denoms)

        def _run(plan, with_labels):
            bb = Bundle()
            for stream, x in inputs.items():
                bb.set(f"inputs.{stream}", x)
            bb.set("masks.tracks", masks["tracks"])
            if with_labels:
                for key, val in labels.items():
                    bb.set(key, val)
            with torch.no_grad():
                return Executor(plan).run(bb)

        b_onnx = _run(onnx, with_labels=False)
        b_test = _run(test, with_labels=True)
        v2_onnx = b_onnx.get("preds.jets.regression")
        v2_test = b_test.get("preds.jets.regression")
        # W34.3: BOTH forwards publish the RAW scaled preds (no de-scale in forward),
        # so the raw TEST and ONNX preds are IDENTICAL (same weights, same inputs).
        assert torch.allclose(v2_onnx, v2_test, atol=1e-6)
        # the de-scaling now lives in get_output (mode-split denominator source):
        # ONNX gathers the denominator BY NAME from inputs.jets; TEST from labels.
        col = JET_VARIABLES.index("pt_btagJes")
        denom_from_input = inputs["jets"][..., col]
        pooled = b_onnx.get("pooled.global")
        with torch.no_grad():
            raw, _ = task.head_forward(pooled, {}, None, context=None)
            ref_onnx = task.run_inference(
                raw.clone(), labels={"jets": {"pt_btagJes": denom_from_input}}
            )
        (onnx_field,) = task.get_output(b_onnx, Mode.ONNX, "reg")
        # ONNX get_output de-scales by-name from the input Feature; value is the
        # squeezed scalar at B>1 it stays [B] (squeeze drops no non-size-1 dim)
        assert torch.allclose(onnx_field.value, ref_onnx[..., 0], atol=1e-6)
        # the TEST get_output de-scales from the LABEL denominator -> a DIFFERENT
        # result than the ONNX (input-Feature) de-scale
        (test_field,) = task.get_output(b_test, Mode.TEST, "reg")
        assert not torch.allclose(test_field.value, onnx_field.value, atol=1e-6)

    def test_sequence_scaler_descale_parity_and_nan_padding(self):
        """Per-token (sequence) regression + functional scaler: RAW forward, get_output de-scale."""
        d = 16
        scaler = {"pt": {"op": "log", "op_scale": 0.2}, "mass": {"op": "linear", "op_scale": 10}}
        task = RegressionTaskModule(
            stream="tracks",
            targets=["pt", "mass"],
            scaler=scaler,
            write_targets=False,  # prediction de-scale gate; labels: test_target_labels.py
        )
        task.name = "regression"
        _bind_reg_module(task, {"encoded.tracks": d})
        assert task.scaler is not None
        x = torch.randn(B, T, d)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, T - 1] = True
        gen = torch.Generator().manual_seed(5)
        b = Bundle()
        b.set("encoded.tracks", x)
        b.set("masks.tracks", mask)
        b.set("labels.tracks.pt", 1.0 + torch.rand(B, T, generator=gen))
        b.set("labels.tracks.mass", 1.0 + torch.rand(B, T, generator=gen))
        fit_out = task.forward(b, Mode.FIT)
        assert fit_out["preds.tracks.regression"].shape == (B, T, 2)
        assert torch.isfinite(fit_out["losses.regression"])
        bt = Bundle()
        bt.set("encoded.tracks", x)
        bt.set("masks.tracks", mask)
        with torch.no_grad():
            test_out = task.forward(bt, Mode.TEST)
        pred = test_out["preds.tracks.regression"]
        assert pred.shape == (B, T, 2)
        # W34.3: the TEST forward emits the RAW scaled preds (no de-scale, no nan-pad)
        with torch.no_grad():
            raw, _ = task.head_forward(x, {}, {"tracks": mask}, context=None)
        assert torch.equal(pred, raw)
        # get_output now owns the scaler de-scale + the masked-position nan-pad
        bt2 = Bundle()
        bt2.set("preds.tracks.regression", pred)
        bt2.set("masks.tracks", mask)
        with torch.no_grad():
            fields = task.get_output(bt2, Mode.TEST, "reg")
            ref = task.run_inference(raw.clone(), labels=None, pad_mask=mask)
        for i, f in enumerate(fields):
            assert torch.isnan(f.value[:, T - 1]).all()  # masked positions nan-padded
            assert torch.equal(torch.nan_to_num(f.value), torch.nan_to_num(ref[..., i]))

    # -- Gaussian (mu/sigma, output==2R, NLL, stddev=sqrt) — A3 -----------------

    def test_gaussian_output_size_is_doubled(self):
        task = RegressionTaskModule(
            stream="jets",
            targets="pt",
            input="pooled.global",
            gaussian=True,
            norm_params={"mean": 1.0, "std": 2.0},
        )
        task.name = "gaussian_regression"
        _bind_reg_module(task, {"pooled.global": 16})
        assert task.net.output_size == 2  # 2 * 1 target
        assert task.output_suffixes == ("pt", "pt_stddev")
        produces = flatten_spec(task.declare_io(Mode.TEST).produces)
        assert produces["preds.jets.gaussian_regression"].shape == ("B", 2)

    def test_gaussian_rejects_functional_scaler(self):
        with pytest.raises(ConfigError, match="gaussian head cannot use a functional 'scaler'"):
            RegressionTaskModule(
                stream="jets",
                targets="pt",
                input="pooled.global",
                gaussian=True,
                scaler={"pt": {"op": "log", "op_scale": 0.2}},
            )

    def test_gaussian_requires_a_scaling_method_at_bind(self):
        task = RegressionTaskModule(
            stream="jets", targets="pt", input="pooled.global", gaussian=True
        )
        task.name = "gaussian_regression"
        with pytest.raises(ConfigError, match="gaussian head requires norm_params"):
            _bind_reg_module(task, {"pooled.global": 16})

    def test_gaussian_fit_forward_parity_and_nll_loss(self, norm_paths):
        """FIT preds are the RAW [B, 2R] head output; loss == v1 gaussian NLL."""
        targets = ("HadronConeExclTruthLabelPt",)
        task = RegressionTaskModule(
            stream="jets",
            targets=list(targets),
            input="pooled.global",
            gaussian=True,
            norm_params={"mean": 1.0, "std": 1.0},
            weight=0.5,
        )
        modules = build_regression_modules(norm_paths[0], task)
        fit = compile_regression(modules, Mode.FIT, targets)
        bind_all(modules, resolve_bind_schema([fit]))
        materialise_all(modules)
        inputs, masks = make_gn2_batch(B, T)
        labels = make_regression_labels(B, targets)
        b = Bundle()
        for stream, x in inputs.items():
            b.set(f"inputs.{stream}", x)
        b.set("masks.tracks", masks["tracks"])
        for key, val in labels.items():
            b.set(key, val)
        out = Executor(fit).run(b, debug=True)
        v2_pred = out.get("preds.jets.regression")
        v2_loss = out.get("losses.regression")
        assert v2_pred.shape == (B, 2)  # means ‖ raw variances
        pooled = out.get("pooled.global")
        tdict = {"jets": {t: labels[f"labels.jets.{t}"] for t in targets}}
        ref_pred, ref_loss = task.head_forward(pooled, tdict, None, context=None)
        assert torch.allclose(v2_pred, ref_pred, atol=1e-6)
        assert torch.allclose(v2_loss, ref_loss, atol=1e-6)

    def test_gaussian_test_descale_one_array_means_then_stddev(self, norm_paths):
        """TEST forward = RAW [B, 2R] (W34.3); get_output publishes means ‖ stddevs."""
        targets = ("HadronConeExclTruthLabelPt",)
        task = RegressionTaskModule(
            stream="jets",
            targets=list(targets),
            input="pooled.global",
            gaussian=True,
            norm_params={"mean": 2.0, "std": 3.0},
            write_targets=False,  # prediction de-scale gate; labels: test_target_labels.py
        )
        modules = build_regression_modules(norm_paths[0], task)
        fit = compile_regression(modules, Mode.FIT, targets)
        test = compile_regression(modules, Mode.TEST, targets)
        bind_all(modules, resolve_bind_schema([fit, test]))
        materialise_all(modules)
        inputs, masks = make_gn2_batch(B, T)
        b = Bundle()
        for stream, x in inputs.items():
            b.set(f"inputs.{stream}", x)
        b.set("masks.tracks", masks["tracks"])
        with torch.no_grad():
            b = Executor(test).run(b)
        v2_test = b.get("preds.jets.regression")
        assert v2_test.shape == (B, 2)
        pooled = b.get("pooled.global")
        with torch.no_grad():
            raw, _ = task.head_forward(pooled, {}, None, context=None)
        # W34.3: the gaussian TEST forward publishes the RAW [B, 2R] (NO de-scale)
        assert torch.allclose(v2_test, raw, atol=1e-6)
        # get_output owns the de-scale + the means‖stds one-array re-concat
        with torch.no_grad():
            ref_means, ref_stds = task.run_inference(raw.clone())
            fields = task.get_output(b, Mode.TEST, "reg")
        descaled = torch.stack([f.value for f in fields], dim=-1)
        assert torch.allclose(descaled, torch.cat([ref_means, ref_stds], dim=-1), atol=1e-6)
        # the stddev column IS sqrt(softplus(var)) * std (v1 task.py:762-764)
        assert (fields[1].value > 0).all()

    # -- sample_weight + NaN masking (inside composed v1 nan_loss) — A3 ---------

    def test_sample_weight_requires_reduction_none(self):
        with pytest.raises(ConfigError, match="sample_weight.*reduction"):
            RegressionTaskModule(
                stream="jets", targets="pt", input="pooled.global", sample_weight="w"
            )
        # explicit reduction: none is accepted
        RegressionTaskModule(
            stream="jets",
            targets="pt",
            input="pooled.global",
            sample_weight="w",
            loss={"class_path": "torch.nn.MSELoss", "init_args": {"reduction": "none"}},
        )

    def test_sample_weight_declares_the_weight_label(self):
        task = RegressionTaskModule(
            stream="jets",
            targets=["a", "b"],
            input="pooled.global",
            sample_weight="w",
            loss={"class_path": "torch.nn.MSELoss", "init_args": {"reduction": "none"}},
        )
        task.name = "regression"
        req = flatten_spec(task.declare_io(Mode.FIT).requires)
        assert "labels.jets.w" in req and req["labels.jets.w"].modes == Mode.TRAINING
        # the weight label leaf is TRAINING-gated (the planner drops it in
        # TEST/ONNX, same as the target labels — modes attribute, not absence)
        assert flatten_spec(task.declare_io(Mode.TEST).requires)["labels.jets.w"].modes == (
            Mode.TRAINING
        )

    @pytest.mark.parametrize("weights", [[1.0, 0.5, 2.0, 0.0, 1.5, 0.25], "zero"])
    def test_sample_weight_loss_parity_nonuniform_and_zero(self, norm_paths, weights):
        """Per-sample-weighted loss parity (nonuniform + all-zero) vs the v1 head."""
        targets = ("R10TruthLabel_R22v1_TruthJetMass", "R10TruthLabel_R22v1_TruthJetPt")
        w = torch.zeros(B) if weights == "zero" else torch.tensor(weights)
        task = RegressionTaskModule(
            stream="jets",
            targets=list(targets),
            input="pooled.global",
            sample_weight="w",
            norm_params={"mean": [1.0, 2.0], "std": [3.0, 4.0]},
            loss={"class_path": "torch.nn.MSELoss", "init_args": {"reduction": "none"}},
        )
        modules = build_regression_modules(norm_paths[0], task)
        fit = compile_regression(modules, Mode.FIT, targets, weight="w")
        bind_all(modules, resolve_bind_schema([fit]))
        materialise_all(modules)
        inputs, masks = make_gn2_batch(B, T)
        labels = make_regression_labels(B, targets)
        b = Bundle()
        for stream, x in inputs.items():
            b.set(f"inputs.{stream}", x)
        b.set("masks.tracks", masks["tracks"])
        for key, val in labels.items():
            b.set(key, val)
        b.set("labels.jets.w", w)
        out = Executor(fit).run(b, debug=True)
        v2_loss = out.get("losses.regression")
        # v1 reference: the same head, weight in the targets dict (task.py:430)
        pooled = out.get("pooled.global")
        tdict = {"jets": {t: labels[f"labels.jets.{t}"] for t in targets}}
        tdict["jets"]["w"] = w
        _, ref_loss = task.head_forward(pooled, tdict, None, context=None)
        assert torch.allclose(v2_loss, ref_loss, atol=1e-6)

    def test_nan_target_masking_loss_parity(self, norm_paths):
        """NaN targets are masked (→0) and reduced with nanmean; parity vs v1."""
        targets = ("HadronConeExclTruthLabelPt",)
        task = RegressionTaskModule(
            stream="jets",
            targets=list(targets),
            input="pooled.global",
            norm_params={"mean": 1.0, "std": 1.0},
            loss={"class_path": "torch.nn.MSELoss", "init_args": {"reduction": "none"}},
        )
        modules = build_regression_modules(norm_paths[0], task)
        fit = compile_regression(modules, Mode.FIT, targets)
        bind_all(modules, resolve_bind_schema([fit]))
        materialise_all(modules)
        inputs, masks = make_gn2_batch(B, T)
        labels = make_regression_labels(B, targets)
        # poison half the targets with NaN
        poisoned = labels[f"labels.jets.{targets[0]}"].clone()
        poisoned[::2] = torch.nan
        labels[f"labels.jets.{targets[0]}"] = poisoned
        b = Bundle()
        for stream, x in inputs.items():
            b.set(f"inputs.{stream}", x)
        b.set("masks.tracks", masks["tracks"])
        for key, val in labels.items():
            b.set(key, val)
        out = Executor(fit).run(b, debug=True)
        v2_loss = out.get("losses.regression")
        assert torch.isfinite(v2_loss)  # NaN targets did not poison the loss
        pooled = out.get("pooled.global")
        tdict = {"jets": {targets[0]: poisoned}}
        _, ref_loss = task.head_forward(pooled, tdict, None, context=None)
        assert torch.allclose(v2_loss, ref_loss, atol=1e-6)

    def test_declare_and_bind_are_file_free(self, monkeypatch):
        """The §2.3 no-I/O guard holds for the regression module too."""

        def _forbid(*args, **kwargs):
            raise AssertionError("file I/O during declare_io/bind (design §2.3)")

        monkeypatch.setattr("builtins.open", _forbid)
        task = RegressionTaskModule(
            stream="jets", targets=["a"], input="pooled.global", norm_params={"mean": 1, "std": 1}
        )
        task.name = "regression"
        for mode in (Mode.FIT, Mode.VAL, Mode.TEST, Mode.ONNX):
            task.declare_io(mode)
        _bind_reg_module(task, {"pooled.global": 16})
