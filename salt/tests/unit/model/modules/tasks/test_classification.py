"""Unit tests for ClassificationTaskModule."""

from __future__ import annotations

import math

import pytest
import torch

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import Mode, flatten_spec
from salt.model.modules import (
    ResolvedSchema,
    bind_all,
    materialise_all,
    resolve_bind_schema,
)
from salt.model.modules.tasks import (
    ClassificationTaskModule,
)
from salt.tests._fixtures.gn2v2_fixture import (
    ORIGIN_CLASSES,
    build_gn2v2_modules,
    compile_gn2v2,
)

# task modules


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

    def test_bind_builds_head_with_inferred_widths(self, gn2v2):
        modules, _, _ = gn2v2
        head = modules["track_origin"]
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
        """Class weights fill the CE buffer at materialise."""
        modules = build_gn2v2_modules(norm_paths[0], class_dict=norm_paths[1])
        bind_all(modules, resolve_bind_schema(compile_gn2v2(modules, Mode.FIT)))
        head = modules["track_origin"]
        assert torch.equal(head.loss.weight, torch.ones(8))  # bind: allocated, identity
        materialise_all(modules)
        expected = torch.tensor([4.2, 73.7, 1.0, 17.5, 12.3, 12.5, 141.7, 22.3])
        assert torch.allclose(head.loss.weight, expected)
        # the buffer is in the state dict (resume inherits it, cli.py:253-267)
        assert "loss.weight" in modules["track_origin"].state_dict()

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

    @staticmethod
    def _origin_bundle(head: ClassificationTaskModule, labels: torch.Tensor) -> Bundle:
        """Random FIT inputs for the bound track-origin head, with the given labels."""
        n_jets, n_tracks = labels.shape
        generator = torch.Generator().manual_seed(0)
        b = Bundle()
        b.set(head.input_key, torch.randn(n_jets, n_tracks, 16, generator=generator))
        b.set(head.context, torch.randn(n_jets, 16, generator=generator))
        b.set("masks.tracks", torch.zeros(n_jets, n_tracks, dtype=torch.bool))
        b.set(head.label_key, labels)
        return b

    def test_all_ignored_batch_gives_zero_loss(self, gn2v2):
        modules, _, _ = gn2v2
        head = modules["track_origin"]
        out = head(self._origin_bundle(head, torch.full((3, 5), -1)), Mode.FIT)
        loss = out[head.loss_key]
        assert loss.item() == 0.0
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)

    def test_partly_ignored_batch_keeps_the_loss_value(self, gn2v2):
        modules, _, _ = gn2v2
        head = modules["track_origin"]
        labels = torch.full((3, 5), -1)
        labels[0, :2] = torch.tensor([1, 3])
        out = head(self._origin_bundle(head, labels), Mode.FIT)
        expected = head.loss(out[head.pred_key].permute(0, 2, 1), labels)
        assert torch.isfinite(expected)
        torch.testing.assert_close(out[head.loss_key], expected * head.weight, atol=1e-6, rtol=0)


_CLASS_WEIGHTS = [None, [1.0, 3.0, 0.5]]
_N_ROWS = 12
_WIDTH = 16
# rows 0..6 are the head's own jet type; rows 7..11 are the "other" type (label -1)
_OTHER = torch.arange(_N_ROWS) >= 7
_OWN = ~_OTHER
_MASKED = torch.tensor([0, 1, 2] * 4).masked_fill(_OTHER, -1)


def _bound_classification_task(
    class_weight: list[float] | None = None,
    sample_weight: str | list[str] | None = "w",
    reduction: str = "none",
) -> ClassificationTaskModule:
    init_args: dict = {"reduction": reduction} | ({"weight": class_weight} if class_weight else {})
    task = ClassificationTaskModule(
        stream="jets",
        label="flavour_label",
        class_names=["bjets", "cjets", "ujets"],
        input="pooled.global",
        sample_weight=sample_weight,
        loss={"class_path": "torch.nn.CrossEntropyLoss", "init_args": init_args},
    )
    task.name = "jets_classification"
    task.bind(ResolvedSchema(widths={"pooled.global": 16}))
    return task


def _ref_ce(class_weight: list[float] | None) -> torch.nn.CrossEntropyLoss:
    weight = torch.as_tensor(class_weight) if class_weight is not None else None
    return torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=-1, weight=weight)


class TestClassificationTaskModuleSampleWeight:
    def test_sample_weight_requires_reduction_none(self):
        with pytest.raises(ConfigError, match="sample_weight.*reduction"):
            ClassificationTaskModule(
                stream="jets", label="flavour_label", class_names=["a", "b"], sample_weight="w"
            )
        _bound_classification_task()  # reduction: none is accepted

    def test_sample_weight_declares_the_weight_label_training_only(self):
        task = _bound_classification_task()
        for mode in (Mode.FIT, Mode.VAL, Mode.TEST):
            spec = flatten_spec(task.declare_io(mode).requires)["labels.jets.w"]
            assert spec.modes == Mode.TRAINING
            assert spec.kind == "label"
            assert spec.dtype == "float32"
        assert task.sample_weights == ("w",)
        assert task.weight_label_keys == ("labels.jets.w",)

    def test_sample_weight_absent_when_unset(self):
        task = _bound_classification_task(sample_weight=None)
        assert "labels.jets.w" not in flatten_spec(task.declare_io(Mode.FIT).requires)
        assert task.sample_weights == () and task.weight_label_keys == ()

    def test_sample_weight_list_is_the_per_row_product(self):
        two = _bound_classification_task(sample_weight=["w1", "w2"])
        one = _bound_classification_task(sample_weight="w")
        one.net.load_state_dict(two.net.state_dict())
        torch.manual_seed(5)
        x = torch.randn(9, _WIDTH)
        labels = torch.randint(0, 3, (9,))
        labels[3] = -1
        w1, w2 = torch.rand(9) + 0.1, torch.rand(9) + 0.1
        b2, b1 = Bundle(), Bundle()
        for b in (b2, b1):
            b.set("pooled.global", x)
            b.set(two.label_key, labels)
        b2.set("labels.jets.w1", w1)
        b2.set("labels.jets.w2", w2)
        b1.set("labels.jets.w", w1 * w2)
        loss2 = two.forward(b2, Mode.FIT)[two.loss_key]
        assert torch.allclose(loss2, one.forward(b1, Mode.FIT)[one.loss_key], atol=1e-6)
        req = flatten_spec(two.declare_io(Mode.FIT).requires)
        assert set(req) >= {"labels.jets.w1", "labels.jets.w2"}
        assert "labels.jets.w" not in req
        for key in ("labels.jets.w1", "labels.jets.w2"):
            spec = req[key]
            assert (spec.dtype, spec.kind, spec.modes) == ("float32", "label", Mode.TRAINING)
        assert two.sample_weights == ("w1", "w2")

    def test_sample_weight_list_rejects_empty_and_duplicates(self):
        with pytest.raises(ConfigError, match="at least one"):
            _bound_classification_task(sample_weight=[])
        with pytest.raises(ConfigError, match="duplicate"):
            _bound_classification_task(sample_weight=["w", "w"])
        with pytest.raises(ConfigError, match="reduction"):
            _bound_classification_task(sample_weight=["w1", "w2"], reduction="mean")

    @pytest.mark.parametrize("class_weight", _CLASS_WEIGHTS)
    def test_sample_weight_parity_unit_weight_matches_ce_mean(self, class_weight):
        """Unit weights: the weighted reduction equals CE(reduction='mean'), -1 rows included."""
        task = _bound_classification_task(class_weight=class_weight)
        torch.manual_seed(4)
        preds = torch.randn(9, 3)
        labels = torch.randint(0, 3, (9,))
        labels[[1, 6]] = -1
        result = task.calculate_loss(preds, labels, torch.ones(9))
        assert torch.allclose(result, _ref_ce(class_weight)(preds, labels), atol=1e-6)

    def test_sample_weight_nonuniform_hand_computed(self):
        task = _bound_classification_task()
        # softmax [.5, .5] label 0; [.75, .25] label 0; [.75, .25] label 1
        logits = torch.tensor([[0.0, 0.0], [math.log(3.0), 0.0], [math.log(3.0), 0.0]])
        w = torch.tensor([1.0, 2.0, 0.5])
        result = task.calculate_loss(logits, torch.tensor([0, 0, 1]), w)
        per_row = [math.log(2.0), math.log(4.0 / 3.0), math.log(4.0)]
        expected = sum(wi * li for wi, li in zip(w.tolist(), per_row, strict=True)) / 3.5
        assert result.item() == pytest.approx(expected, abs=1e-6)

    def test_sample_weight_zero_weight_rows_contribute_nothing_no_nan(self):
        task = _bound_classification_task()
        torch.manual_seed(1)
        preds = torch.randn(5, 3)
        labels = torch.randint(0, 3, (5,))
        w = torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0])
        keep = w > 0
        result = task.calculate_loss(preds, labels, w)
        assert not torch.isnan(result)
        expected = (w * task.loss(preds, labels))[keep].sum() / w[keep].sum()
        assert torch.allclose(result, expected, atol=1e-6)
        preds2 = torch.where(keep[:, None], preds, preds * 100)  # zero-weight rows are inert
        assert torch.allclose(result, task.calculate_loss(preds2, labels, w), atol=1e-6)


def _masking_task(path: str, class_weight: list[float] | None) -> ClassificationTaskModule:
    torch.manual_seed(11)
    sample_weight, reduction = (None, "mean") if path == "mean" else ("w", "none")
    return _bound_classification_task(class_weight, sample_weight, reduction)


def _fit_loss(task: ClassificationTaskModule, x: torch.Tensor, labels: torch.Tensor):
    b = Bundle()
    b.set("pooled.global", x)
    b.set(task.label_key, labels)
    for key in task.weight_label_keys:
        b.set(key, torch.ones(labels.shape[0]))
    return task.forward(b, Mode.FIT)[task.loss_key]


@pytest.mark.parametrize("class_weight", _CLASS_WEIGHTS)
class TestPerTypeMasking:
    @pytest.mark.parametrize("path", ["mean", "sample_weight"])
    def test_masked_loss_equals_own_type_subset(self, path, class_weight):
        task = _masking_task(path, class_weight)
        x = torch.randn(_N_ROWS, _WIDTH)
        subset = _fit_loss(task, x[_OWN], _MASKED[_OWN])
        assert torch.allclose(_fit_loss(task, x, _MASKED), subset, atol=1e-6)

    @pytest.mark.parametrize("path", ["mean", "sample_weight"])
    def test_other_type_rows_get_exactly_zero_gradient(self, path, class_weight):
        task = _masking_task(path, class_weight)
        x = torch.randn(_N_ROWS, _WIDTH, requires_grad=True)
        _fit_loss(task, x, _MASKED).backward()
        assert torch.all(x.grad[_OTHER] == 0)
        assert torch.isfinite(x.grad).all()
        assert torch.all(x.grad[_OWN].abs().sum(dim=-1) > 0)

    @pytest.mark.parametrize("path", ["mean", "sample_weight"])
    def test_all_ignored_batch_is_zero_with_finite_gradients(self, path, class_weight):
        task = _masking_task(path, class_weight)
        x = torch.randn(_N_ROWS, _WIDTH, requires_grad=True)
        loss = _fit_loss(task, x, torch.full((_N_ROWS,), -1))
        assert loss.item() == 0.0
        loss.backward()
        assert torch.all(x.grad == 0)
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in task.net.parameters()
        )

    def test_guarded_reduction_matches_torch_mean(self, class_weight):
        task = _masking_task("mean", class_weight)
        gen = torch.Generator().manual_seed(7)
        for _ in range(20):
            preds = torch.randn(_N_ROWS, 3, generator=gen)
            labels = torch.randint(-1, 3, (_N_ROWS,), generator=gen)
            labels[0] = torch.randint(0, 3, (1,), generator=gen)  # >=1 valid row
            guarded = task._masked_ce_mean(preds, labels)  # noqa: SLF001
            assert torch.allclose(guarded, _ref_ce(class_weight)(preds, labels), atol=1e-6)
        x = torch.randn(_N_ROWS, _WIDTH, generator=gen)
        expected = _ref_ce(class_weight)(task.net(x, None), _MASKED)
        assert torch.allclose(_fit_loss(task, x, _MASKED), expected, atol=1e-6)

    def test_sequence_head_matches_torch_mean_and_all_ignored_is_zero(self, class_weight):
        init_args: dict = {} if class_weight is None else {"weight": class_weight}
        loss_cfg = {"class_path": "torch.nn.CrossEntropyLoss", "init_args": init_args}
        task = ClassificationTaskModule("tracks", "lbl", ["a", "b", "c"], loss=loss_cfg)
        task.name = "track_origin"
        task.bind(ResolvedSchema(widths={"encoded.tracks": _WIDTH}))
        gen = torch.Generator().manual_seed(17)
        preds = torch.randn(4, 5, 3, generator=gen, requires_grad=True)  # [B, T, C]
        labels = torch.randint(-1, 3, (4, 5), generator=gen)
        labels[0, 0], labels[1, 2] = 1, -1  # >=1 valid and >=1 ignored token
        expected = _ref_ce(class_weight)(preds.permute(0, 2, 1), labels)
        torch.testing.assert_close(task.calculate_loss(preds, labels), expected)
        loss = task.calculate_loss(preds, torch.full((4, 5), -1))
        assert loss.item() == 0.0
        loss.backward()
        assert torch.isfinite(preds.grad).all()
        assert torch.all(preds.grad == 0)
