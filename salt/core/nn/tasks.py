"""Config-constructed task modules for the GN2v2 surface (design §3.3, §5.1, §9.2).

M2 porting policy (plan 05): each task module composes a FRESH v1 task head
(`salt.models.task.ClassificationTask` / `VertexingTask`) built at `bind`
from the resolved schema — loss math (ignore_index=-1, the label ``-2`` pad
fold, origin-weighted vertexing) stays verbatim v1; full absorption is M7.
What is new: the declared label/mask/context dependencies, the two-phase
bind, the per-mode output semantics, and the declarative class-weight source.

Per design §3.3, tasks consume PER-STREAM tensors produced by an explicit
`Split` (``encoded.<stream>``) — the v1 ``input_name_mask`` reconstruction
from pad-mask dict order (task.py:58-78) is gone. The composed v1 head is
therefore handed single-stream dicts, under which its internal slicing is
the identity; outputs are mathematically equal to v1's full-sequence path
but NOT guaranteed bitwise (the M2 gates use 1e-6/curve criteria, plan 05).

Mode semantics (design §3.3): ``preds.<stream>.<task>`` is published in ALL
modes — raw logits / raw edge scores in FIT|VAL (cheap, consumed by metrics
callbacks), converted physical values in TEST/ONNX for the classification
family (softmax / padded-aware softmax via the v1 ``run_inference``).
Vertexing is the documented per-family exception: TEST publishes per-node
vertex assignments (union-find, v1 writer semantics, task.py:985-986,1003);
ONNX publishes RAW edge scores so the export-side ``vertex_union_find``
reduce owns the in-graph union-find (v1 placement, to_onnx.py:426-431).
Labels are required and ``losses.<task>`` produced in FIT|VAL only.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.core.nn.bind import ResolvedSchema
from salt.core.nn.modules import _reject_width_keys, _stream_len
from salt.models.task import ClassificationTask as V1ClassificationTask
from salt.models.task import VertexingTask as V1VertexingTask

__all__ = ["ClassificationTaskModule", "VertexingTaskModule"]

_UNNAMED = "unnamed"
_WIDTH_KEYS = ("input_size", "output_size", "context_size")

_DEFAULT_CLS_LOSS: dict[str, Any] = {"class_path": "torch.nn.CrossEntropyLoss"}
_DEFAULT_VTX_LOSS: dict[str, Any] = {
    "class_path": "torch.nn.BCEWithLogitsLoss",
    "init_args": {"reduction": "none"},
}


class _TaskModuleBase(nn.Module):
    """Shared config capture + loss construction for the v2 task modules."""

    def __init__(
        self,
        stream: str,
        label: str,
        input: str | None,  # noqa: A002 - design §3.3 YAML surface name
        context: str | None,
        dense: dict[str, Any] | None,
        loss: str | dict[str, Any] | None,
        weight: float,
        default_loss: dict[str, Any],
    ) -> None:
        super().__init__()
        self.name = _UNNAMED
        _reject_width_keys(type(self).__name__, dense, _WIDTH_KEYS)
        self.stream = stream
        self.label = label
        self.input_key = input if input is not None else f"encoded.{stream}"
        self.context = context
        self.dense_cfg = dict(dense or {})
        self.loss_cfg = _loss_cfg(loss, default_loss)
        self.weight = float(weight)
        self.task: nn.Module | None = None

    @property
    def pred_key(self) -> str:
        """The published prediction key (design §3.3).

        Returns
        -------
        str
            ``preds.<stream>.<instance-name>``.
        """
        return f"preds.{self.stream}.{self.name}"

    @property
    def loss_key(self) -> str:
        """The published loss key, auto-collected by `LossSum` (design §3.3).

        Returns
        -------
        str
            ``losses.<instance-name>``.
        """
        return f"losses.{self.name}"

    @property
    def label_key(self) -> str:
        """The declared label dependency (demand-driven `Labels`, design §3.3).

        Returns
        -------
        str
            ``labels.<stream>.<label>``.
        """
        return f"labels.{self.stream}.{self.label}"


class ClassificationTaskModule(_TaskModuleBase):
    """Classification head over a pooled vector or a per-stream sequence (design §3.3).

    ``class_names`` is REQUIRED and explicit (no ``CLASS_NAMES`` fallback).
    Whenever the reader carries a schema artifact whose group attrs name the
    label's classes, the configured list is cross-checked — set AND order —
    by `salt.core.saltmodule.check_class_names`, default-on at
    `SaltModule.setup` and in ``salt2 graph validate`` (design §2.6, §4.1).
    The head width is ``len(class_names)``; the input/context widths are
    inferred at `bind` — no width keys in config (design §2.3).

    Declarative class weights (design §3.3, reproducing v1 ``use_class_dict``,
    cli.py:446-491): ``weight_source: {from_class_dict: <path>}`` allocates
    the CE ``weight`` buffer at `bind` (it lives in the loss's state_dict,
    ``torch.nn._WeightedLoss``), fills it at `materialise()` on fresh fits
    (the ONLY file I/O), and inherits it from the checkpoint on resume —
    v1's bake-on-resume contract (cli.py:253-267). Freezing the resolved
    values into the run-dir config is the config surface's job (M2 stage C).
    A literal ``loss.init_args.weight`` list remains valid and is exclusive
    with ``weight_source``.
    """

    def __init__(
        self,
        stream: str,
        label: str,
        class_names: Sequence[str],
        input: str | None = None,  # noqa: A002 - design §3.3 YAML surface name
        context: str | None = None,
        sequence: bool | None = None,
        dense: dict[str, Any] | None = None,
        loss: str | dict[str, Any] | None = None,
        weight: float = 1.0,
        weight_source: Mapping[str, str] | None = None,
        label_map: dict[int, int] | None = None,
    ) -> None:
        """Capture config only (design §2.3).

        Parameters
        ----------
        stream : str
            The labelled stream (``jets``, ``tracks``, ...).
        label : str
            Label field name, demanded as ``labels.<stream>.<label>``.
        class_names : Sequence[str]
            Ordered class names — REQUIRED, index-aligned with outputs.
        input : str | None, optional
            Input key, by default ``encoded.<stream>`` (the `Split` output).
        context : str | None, optional
            Context key (e.g. ``pooled.global``), by default None.
        sequence : bool | None, optional
            Whether the task is per-token, by default inferred: True when
            `input` is the per-stream default, False for an explicit input
            (e.g. ``pooled.global``). Set explicitly for per-token tasks on
            non-default inputs.
        dense : dict[str, Any] | None, optional
            Extra v1 `Dense` kwargs (no width keys), by default None.
        loss : str | dict[str, Any] | None, optional
            Loss config: a ``torch.nn`` class name or a
            ``{class_path, init_args}`` mapping, by default
            CrossEntropyLoss.
        weight : float, optional
            Scalar task-loss weight (applied INSIDE the composed v1 head,
            task.py:243), by default 1.0.
        weight_source : Mapping[str, str] | None, optional
            ``{"from_class_dict": <path>}`` declarative class-weight source
            (see class docstring), by default None.
        label_map : dict[int, int] | None, optional
            Integer label remap for training, by default None.

        Raises
        ------
        ConfigError
            On empty/duplicate class names, malformed `weight_source`, or a
            literal loss weight combined with `weight_source`.
        """
        super().__init__(stream, label, input, context, dense, loss, weight, _DEFAULT_CLS_LOSS)
        if not class_names:
            raise ConfigError(
                f"ClassificationTaskModule: class_names is required and explicit for label "
                f"{label!r} (design §3.3 — no CLASS_NAMES fallback)"
            )
        if len(set(class_names)) != len(tuple(class_names)):
            raise ConfigError(
                f"ClassificationTaskModule: duplicate class names in {tuple(class_names)}"
            )
        self.class_names = tuple(class_names)
        self.sequence = sequence if sequence is not None else input is None
        self.label_map = dict(label_map) if label_map is not None else None
        self.weight_source = _checked_weight_source(weight_source)
        if self.weight_source is not None and "weight" in self.loss_cfg.get("init_args", {}):
            raise ConfigError(
                "ClassificationTaskModule: class weights already specified in the loss config — "
                "remove them or drop weight_source (v1 use_class_dict contract, cli.py:476-479)"
            )

    def declare_io(self, mode: Mode) -> IO:
        """Declare input/context/masks (+ FIT|VAL labels) -> preds (+ FIT|VAL loss).

        Returns
        -------
        IO
            The declared requires/produces; label and loss ports carry
            ``modes=TRAINING`` and are filtered by the planner elsewhere.
        """
        del mode
        n_classes = len(self.class_names)
        width = sym_dim("D", self.name)
        if self.sequence:
            input_spec = TensorSpec(shape=("B", _stream_len(self.stream), width), dtype="float32")
            label_spec = TensorSpec(
                shape=("B", _stream_len(self.stream)),
                dtype="int64",
                kind="label",
                modes=Mode.TRAINING,
            )
            pred_spec = TensorSpec(
                shape=("B", _stream_len(self.stream), n_classes), dtype="float32"
            )
        else:
            input_spec = TensorSpec(shape=("B", width), dtype="float32")
            label_spec = TensorSpec(shape=("B",), dtype="int64", kind="label", modes=Mode.TRAINING)
            pred_spec = TensorSpec(shape=("B", n_classes), dtype="float32")
        requires: dict[str, TensorSpec] = {self.input_key: input_spec}
        if self.context is not None:
            requires[self.context] = TensorSpec(shape=None, dtype="float32")
        if self.sequence:
            requires[f"masks.{self.stream}"] = TensorSpec(
                shape=("B", _stream_len(self.stream)), dtype="bool", kind="pad_mask"
            )
        requires[self.label_key] = label_spec
        produces: dict[str, TensorSpec] = {
            self.pred_key: pred_spec,
            self.loss_key: TensorSpec(shape=(), kind="loss", modes=Mode.TRAINING),
        }
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def bind(self, schema: ResolvedSchema) -> None:
        """Build the composed v1 head with inferred widths (design §2.3, §3.3).

        When `weight_source` is set, the loss is constructed with a
        ones-initialised ``weight`` buffer sized ``len(class_names)`` so it
        lives in the state_dict; `materialise` fills it on fresh fits.
        """
        init_args = dict(self.loss_cfg.get("init_args", {}))
        if self.weight_source is not None:
            init_args["weight"] = torch.ones(len(self.class_names))
        elif isinstance(init_args.get("weight"), (list, tuple)):
            init_args["weight"] = torch.as_tensor(init_args["weight"], dtype=torch.float32)
        loss_module = _loss_class(self.loss_cfg)(**init_args)
        dense_config = {
            "input_size": schema.width(self.input_key),
            "output_size": len(self.class_names),
            **({"context_size": schema.width(self.context)} if self.context else {}),
            **self.dense_cfg,
        }
        self.task = V1ClassificationTask(
            name=self.name,
            input_name=self.stream,
            label=self.label,
            class_names=list(self.class_names),
            label_map=self.label_map,
            loss=loss_module,
            weight=self.weight,
            dense_config=dense_config,
        )

    def materialise(self) -> None:
        """Resolve ``weight_source`` from the class dict (the ONLY file I/O).

        Raises
        ------
        RuntimeError
            If called before `bind`.
        ValueError
            If the class dict lacks the stream/label or the weight count
            does not match ``class_names`` (v1 contract, cli.py:480-491).
        """
        if self.weight_source is None:
            return
        if self.task is None:
            raise RuntimeError(f"{type(self).__name__} {self.name!r}: materialise before bind()")
        path = self.weight_source["from_class_dict"]
        with open(path) as fh:
            class_dict = yaml.safe_load(fh)
        try:
            values = class_dict[self.stream][self.label]
        except (KeyError, TypeError):
            raise ValueError(
                f"Label {self.label!r} for stream {self.stream!r} not found in class dict "
                f"{path} — drop weight_source and specify class weights manually "
                f"(cli.py:483-488 semantics)"
            ) from None
        if len(values) != len(self.class_names):
            raise ValueError(
                f"class dict {path} has {len(values)} weights for {self.stream}.{self.label} "
                f"but the task declares {len(self.class_names)} class_names (design §3.3)"
            )
        with torch.no_grad():
            self.task.loss.weight.copy_(torch.as_tensor(values, dtype=torch.float32))

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Run the composed v1 head; raw logits + loss in FIT|VAL, probs in TEST/ONNX.

        The v1 head is handed SINGLE-STREAM label/mask dicts: its internal
        ``input_name_mask`` slicing is the identity on per-stream inputs,
        and the loss masking (pad fold to ``ignore_index=-1``, the ``-2``
        label fold, task.py:218-227) runs verbatim.

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only (design §2.5).
        """
        assert self.task is not None, "forward before bind()"
        x = b.get(self.input_key)
        ctx = b.get(self.context) if self.context is not None else None
        mask = b.get(f"masks.{self.stream}") if self.sequence else None
        if mode & Mode.TRAINING:
            labels_dict = {self.stream: {self.label: b.get(self.label_key)}}
            pad_masks = {self.stream: mask} if self.sequence else None
            preds, loss = self.task(x, labels_dict, pad_masks, context=ctx)
            return {self.pred_key: preds, self.loss_key: loss}
        # TEST|ONNX: identical, already-converted physical values (design §3.3)
        preds, _ = self.task(x, None, None, context=ctx)
        return {self.pred_key: self.task.run_inference(preds, mask)}


class VertexingTaskModule(_TaskModuleBase):
    """Edge-classification vertexing head (design §3.3, composes v1 `VertexingTask`).

    The origin-label dependency is DECLARED (``origin_label:`` config →
    ``labels.<stream>.<origin_label>``) instead of derived by string-replace
    and free-riding on another task's label collection (task.py:937).
    Origin weighting is explicit config: ``origin_weighting`` gives the
    heavy/fake origin *ids* (defaults reproduce v1's hardcoded 3,4,5 / 1,
    task.py:964 — bit-identical math). TODO(M3): name-based id resolution
    against the schema's origin class names (design §5.1 uses names).

    Mode exception (design §3.3): TEST publishes per-node vertex
    assignments (v1 ``run_inference``: union-find + ``mask_fill_flattened``,
    task.py:985-986,1003); ONNX publishes RAW edge scores for the
    export-side ``vertex_union_find`` reduce (to_onnx.py:426-431).
    """

    def __init__(
        self,
        stream: str,
        label: str,
        origin_label: str,
        input: str | None = None,  # noqa: A002 - design §3.3 YAML surface name
        context: str | None = None,
        dense: dict[str, Any] | None = None,
        loss: str | dict[str, Any] | None = None,
        weight: float = 1.0,
        origin_weighting: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        """Capture config only (design §2.3).

        Parameters
        ----------
        stream : str
            The track-like stream.
        label : str
            Vertex-index label (must contain ``"VertexIndex"`` — the
            composed v1 loss derives the origin key by string-replace,
            task.py:937; absorbed at M7).
        origin_label : str
            Declared origin-label dependency for edge weighting.
        input : str | None, optional
            Input key, by default ``encoded.<stream>``.
        context : str | None, optional
            Context key, by default None.
        dense : dict[str, Any] | None, optional
            Extra v1 `Dense` kwargs (no width keys), by default None.
        loss : str | dict[str, Any] | None, optional
            Loss config, by default ``BCEWithLogitsLoss(reduction="none")``
            (reduction MUST be ``none`` — per-edge weighting multiplies the
            unreduced loss, task.py:938-947).
        weight : float, optional
            Scalar task-loss weight, by default 1.0.
        origin_weighting : Mapping[str, Sequence[int]] | None, optional
            ``{"heavy": [...], "fake": [...]}`` origin ids, by default v1's
            ``{"heavy": [3, 4, 5], "fake": [1]}``.

        Raises
        ------
        ConfigError
            If `label` lacks ``"VertexIndex"`` or `origin_weighting` has
            unknown keys.
        """
        super().__init__(stream, label, input, context, dense, loss, weight, _DEFAULT_VTX_LOSS)
        if "VertexIndex" not in label:
            raise ConfigError(
                f"VertexingTaskModule: label {label!r} must contain 'VertexIndex' — the "
                "composed v1 loss derives the origin key as "
                "label.replace('VertexIndex', 'OriginLabel') (task.py:937; absorbed at M7)"
            )
        self.origin_label = origin_label
        weighting = dict(origin_weighting or {"heavy": [3, 4, 5], "fake": [1]})
        if unknown := sorted(set(weighting) - {"heavy", "fake"}):
            raise ConfigError(
                f"VertexingTaskModule: unknown origin_weighting keys {unknown} — expected "
                "'heavy' and 'fake' (design §3.3)"
            )
        try:
            self.heavy_ids = tuple(int(i) for i in weighting.get("heavy", (3, 4, 5)))
            self.fake_ids = tuple(int(i) for i in weighting.get("fake", (1,)))
        except (TypeError, ValueError) as err:
            raise ConfigError(
                f"VertexingTaskModule: origin_weighting entries must be INTEGER origin ids in "
                f"M2 (got {dict(weighting)!r}) — the design §5.1 name-based form (e.g. "
                "heavy: [FromB, FromBC, FromC]) lands with the M3 schema-attr resolution; "
                "v1's hardcoded ids are heavy: [3, 4, 5], fake: [1]"
            ) from err

    @property
    def origin_label_key(self) -> str:
        """The declared origin-label dependency (design §3.3).

        Returns
        -------
        str
            ``labels.<stream>.<origin_label>``.
        """
        return f"labels.{self.stream}.{self.origin_label}"

    def declare_io(self, mode: Mode) -> IO:
        """Declare input/context/mask (+ FIT|VAL vertex AND origin labels) -> preds/loss.

        The prediction spec is shape-unconstrained: the edge count is
        data-dependent in FIT|VAL|ONNX (``[E, 1]`` raw scores) and the TEST
        assignments are per-node (v1 ``mask_fill_flattened`` layout).

        Returns
        -------
        IO
            The declared requires/produces.
        """
        del mode
        width = sym_dim("D", self.name)
        label_spec = TensorSpec(
            shape=("B", _stream_len(self.stream)),
            dtype="int64",
            kind="label",
            modes=Mode.TRAINING,
        )
        requires: dict[str, TensorSpec] = {
            self.input_key: TensorSpec(
                shape=("B", _stream_len(self.stream), width), dtype="float32"
            ),
            f"masks.{self.stream}": TensorSpec(
                shape=("B", _stream_len(self.stream)), dtype="bool", kind="pad_mask"
            ),
            self.label_key: label_spec,
            self.origin_label_key: label_spec,
        }
        if self.context is not None:
            requires[self.context] = TensorSpec(shape=None, dtype="float32")
        produces: dict[str, TensorSpec] = {
            self.pred_key: TensorSpec(shape=None, dtype=None),
            self.loss_key: TensorSpec(shape=(), kind="loss", modes=Mode.TRAINING),
        }
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def bind(self, schema: ResolvedSchema) -> None:
        """Build the composed v1 vertexing head with inferred widths (design §2.3).

        ``input_size = 2 * width(input)`` (pair concat, task.py:894-896);
        ``context_size = width(context)`` (task.py:884-891).

        Raises
        ------
        ConfigError
            If the configured loss does not use ``reduction="none"``
            (per-edge weighting requires the unreduced loss).
        """
        init_args = dict(self.loss_cfg.get("init_args", {}))
        loss_module = _loss_class(self.loss_cfg)(**init_args)
        if getattr(loss_module, "reduction", "none") != "none":
            raise ConfigError(
                f"VertexingTaskModule {self.name!r}: loss reduction must be 'none' — the "
                "origin weighting multiplies the per-edge loss (task.py:938-947)"
            )
        width = schema.width(self.input_key)
        dense_config = {
            "input_size": 2 * width,
            "output_size": 1,
            **({"context_size": schema.width(self.context)} if self.context else {}),
            **self.dense_cfg,
        }
        self.task = _OriginWeightedVertexing(
            heavy_ids=self.heavy_ids,
            fake_ids=self.fake_ids,
            name=self.name,
            input_name=self.stream,
            label=self.label,
            loss=loss_module,
            weight=self.weight,
            dense_config=dense_config,
        )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Run the composed v1 head; per-mode outputs per design §3.3.

        The labels dict carries the origin labels under the v1-derived key
        (``label.replace("VertexIndex", "OriginLabel")``) so the composed
        loss finds them — the *graph* dependency is the declared
        ``origin_label`` port.

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only (design §2.5).
        """
        assert self.task is not None, "forward before bind()"
        x = b.get(self.input_key)
        ctx = b.get(self.context) if self.context is not None else None
        mask = b.get(f"masks.{self.stream}")
        pad_masks = {self.stream: mask}
        if mode & Mode.TRAINING:
            derived = self.label.replace("VertexIndex", "OriginLabel")
            labels_dict = {
                self.stream: {
                    self.label: b.get(self.label_key),
                    derived: b.get(self.origin_label_key),
                }
            }
            preds, loss = self.task(x, labels_dict, pad_masks, context=ctx)
            return {self.pred_key: preds, self.loss_key: loss}
        preds, _ = self.task(x, None, pad_masks, context=ctx)
        if mode & Mode.TEST:
            # per-node assignments — v1 writer semantics (task.py:985-986,1003)
            return {self.pred_key: self.task.run_inference(preds, mask)}
        # ONNX: raw edge scores; union-find lives in the export reduce (§3.3)
        return {self.pred_key: preds}


class _OriginWeightedVertexing(V1VertexingTask):
    """v1 `VertexingTask` with config-driven heavy/fake origin ids (design §3.3).

    With the default ids (3,4,5 / 1) `get_weights` is bit-identical to v1's
    hardcoded version (task.py:957-964): both build the heavy indicator via
    clipped sums and AND them pairwise over the adjacency.
    """

    def __init__(self, heavy_ids: Sequence[int], fake_ids: Sequence[int], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._heavy_ids = tuple(heavy_ids)
        self._fake_ids = tuple(fake_ids)

    def get_weights(self, labels: Tensor, adjmat: Tensor) -> Tensor:
        """Compute per-edge weights from configured origin ids.

        Returns
        -------
        Tensor
            Per-edge weights of shape ``[E]`` after adjacency compression.
        """
        heavy = torch.clip(sum(labels == i for i in self._heavy_ids), 0, 1)
        fake = torch.clip(sum(labels == i for i in self._fake_ids), 0, 1)
        weights = heavy - fake.int()
        weights = weights.unsqueeze(-1) & weights.unsqueeze(-2)
        weights = weights[adjmat]
        return 1 + weights


def _checked_weight_source(weight_source: Mapping[str, str] | None) -> dict[str, str] | None:
    """Validate a ``weight_source`` mapping (design §3.3).

    Returns
    -------
    dict[str, str] | None
        The validated mapping, or None.

    Raises
    ------
    ConfigError
        If the mapping is not exactly ``{"from_class_dict": <path>}``.
    """
    if weight_source is None:
        return None
    if set(weight_source) != {"from_class_dict"} or not isinstance(
        weight_source["from_class_dict"], (str, Path)
    ):
        raise ConfigError(
            f"weight_source must be {{'from_class_dict': <path>}}, got {dict(weight_source)!r} "
            "(design §3.3)"
        )
    return {"from_class_dict": str(weight_source["from_class_dict"])}


def _loss_cfg(loss: str | dict[str, Any] | None, default: dict[str, Any]) -> dict[str, Any]:
    """Normalise a loss config to ``{class_path, init_args}`` form.

    Returns
    -------
    dict[str, Any]
        The normalised config (a fresh dict).

    Raises
    ------
    ConfigError
        On malformed configs.
    """
    if loss is None:
        cfg: dict[str, Any] = {k: dict(v) if isinstance(v, dict) else v for k, v in default.items()}
        return cfg
    if isinstance(loss, str):
        return {"class_path": loss if "." in loss else f"torch.nn.{loss}"}
    if isinstance(loss, Mapping) and "class_path" in loss:
        return {
            "class_path": str(loss["class_path"]),
            "init_args": dict(loss.get("init_args", {})),
        }
    raise ConfigError(
        f"loss config must be a torch.nn class name or {{class_path, init_args}}, got {loss!r}"
    )


def _loss_class(cfg: Mapping[str, Any]) -> type[nn.Module]:
    """Resolve a loss ``class_path`` to its class (config-only, no instantiation).

    Returns
    -------
    type[nn.Module]
        The loss class.

    Raises
    ------
    ConfigError
        If the path does not resolve to an ``nn.Module`` subclass.
    """
    path = cfg["class_path"]
    module_path, _, cls_name = path.rpartition(".")
    try:
        cls = getattr(importlib.import_module(module_path), cls_name)
    except (ImportError, AttributeError, ValueError) as err:
        raise ConfigError(f"cannot resolve loss class_path {path!r}: {err}") from err
    if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
        raise ConfigError(f"loss class_path {path!r} is not an nn.Module subclass")
    return cls
