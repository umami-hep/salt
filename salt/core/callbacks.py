"""v2 Lightning callbacks: bundle-consuming metrics + run-dir artifacts.

Callbacks are configured through the dict-keyed top-level ``callbacks:``
section (deep-mergeable, null-deletable — design §5.3) and consume the step
**bundle** that `SaltModule`'s steps return (``{"loss": ..., "bundle": ...}``,
design §3.4 — the conscious break of v1 hidden contract #8). Unported v1
callbacks can keep using `salt.core.saltmodule.bundle_as_v1_outputs` during
migration; everything here is bundle-native.

Shipped callbacks:

- `ConfusionMatrix` — the v1 ``ConfusionMatrixCallback`` port (W5 gate):
  accumulates argmax predictions vs truth labels over validation batches and
  logs a confusion matrix to Comet at epoch end. Bundle requires are declared
  (`requires`), and the accumulated lists + computed matrix are stashed on
  the callback so gates can compare values logger-free.
- `GraphArtifacts` — design §4.4 run-dir artifacts, default-on via
  ``base2.yaml``: ``plan_<mode>.txt`` (the §4.4 ordered table with narrowed
  wildcards, per-group read columns and — TEST — the writer-sinks table),
  ``resolved_io.yaml`` (the machine-readable per-module requires/produces
  with resolved specs) plus ``graph_<stage>`` DOT + image (matplotlib
  layered DAG — `salt.core.render`). Written at fit start into the trainer
  log dir; at test start NEXT TO THE CHECKPOINT (with the eval H5) so an
  eval run never splits its outputs across two places nor litters the cwd
  (M3-review fix).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers.comet import CometLogger

from salt.core.graph.errors import ConfigError, GraphError
from salt.core.graph.spec import Mode, TensorSpec
from salt.core.render import dot_source, plan_table, render_graph

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lightning.pytorch.utilities.types import STEP_OUTPUT
    from torch import Tensor

    from salt.core.data.dataset import GraphDataset
    from salt.core.graph.bundle import Bundle
    from salt.core.graph.planner import Plan

__all__ = ["ConfusionMatrix", "GraphArtifacts"]


class ConfusionMatrix(Callback):
    """Log a per-epoch validation confusion matrix for one classification task.

    The v1 ``ConfusionMatrixCallback`` port (``confusion_matrix.py``),
    consuming the step bundle: predictions are read from
    ``preds.<stream>.<task_name>`` (RAW logits in VAL per design §3.3 —
    ``argmax`` is conversion-invariant, so values match v1 exactly) and truth
    from ``labels.<stream>.<label>``. Stream / label / class names are
    resolved from the named task module by duck-typing (the
    ``ClassificationTaskModule`` surface: ``stream``/``label``/
    ``class_names`` — same protocol as `check_class_names`), so user task
    modules participate too.

    The declared bundle requires are exposed via `requires` after ``setup``.
    No extra dataset-demand wiring is needed: VAL labels are already demanded
    by the task itself (``modes=TRAINING``, design §3.3).

    v1-deviation (documented): a missing/incompatible task module is a
    `ConfigError` at setup — v1 silently skipped setup and crashed later
    with an `AttributeError` on the first validation batch.

    At each validation epoch end the matrix is logged to Comet when a
    `CometLogger` is attached (v1 gate kept), and — logger or not — the
    accumulated lists and the computed counts matrix are stashed on the
    callback (``last_truth_labels`` / ``last_pred_labels`` /
    ``last_matrix`` / ``last_ignored``) so the W5 gate can compare v1 vs v2
    values without any logger.

    Parameters
    ----------
    task_name : str
        Instance name of the classification task module (the
        ``model.modules`` dict key, e.g. ``jets_classification``).
    class_names_override : list[str] | dict[str, str] | None, optional
        Class names for logging: a full replacement list, or a mapping from
        existing to new names (v1 semantics), by default None (the task's
        ``class_names``).
    """

    def __init__(
        self, task_name: str, class_names_override: list[str] | dict[str, str] | None = None
    ) -> None:
        self.task_name = task_name
        self.class_names_override = class_names_override
        # resolved at setup
        self.task_stream: str | None = None
        self.task_label_name: str | None = None
        self.task_class_names: list[str] = []
        self.requires: tuple[str, ...] = ()
        # per-epoch accumulators (v1 list-of-tensors semantics)
        self.truth_labels: list[Tensor] = []
        self.pred_labels: list[Tensor] = []
        # stashed at epoch end for logger-free value comparison (W5)
        self.last_truth_labels: list[Tensor] = []
        self.last_pred_labels: list[Tensor] = []
        self.last_matrix: Tensor | None = None
        self.last_ignored: int = 0

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Resolve stream/label/class names from the named task module (fit only).

        Raises
        ------
        ConfigError
            When `pl_module` carries no graph-module dict, or `task_name`
            does not resolve to a module with the classification-task
            surface (candidates are listed).
        """
        del trainer
        if stage != "fit":
            return
        self.truth_labels = []
        self.pred_labels = []
        modules = getattr(pl_module, "_graph_modules", None)
        if not isinstance(modules, dict):
            raise ConfigError(
                f"ConfusionMatrix needs a SaltModule-style LightningModule with a graph-module "
                f"dict, got {type(pl_module).__name__} (design §3.4)"
            )
        module = modules.get(self.task_name)
        stream = getattr(module, "stream", None)
        label = getattr(module, "label", None)
        class_names = getattr(module, "class_names", None)
        if module is None or stream is None or label is None or class_names is None:
            surface = ("stream", "label", "class_names")
            candidates = sorted(
                name
                for name, mod in modules.items()
                if all(getattr(mod, attr, None) is not None for attr in surface)
            )
            raise ConfigError(
                f"ConfusionMatrix: task_name {self.task_name!r} does not name a classification "
                f"task module (needs stream/label/class_names — design §3.3). "
                f"Configured candidates: {candidates or '<none>'}"
            )
        self.task_stream = stream
        self.task_label_name = label
        if isinstance(self.class_names_override, dict):
            self.task_class_names = [
                self.class_names_override.get(name, name) for name in class_names
            ]
        else:
            self.task_class_names = list(self.class_names_override or class_names)
        self.requires = (
            f"preds.{stream}.{self.task_name}",
            f"labels.{stream}.{label}",
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate argmax predictions and truth labels from the step bundle."""
        del trainer, pl_module, batch, batch_idx, dataloader_idx
        bundle: Bundle = outputs["bundle"]
        pred_key, label_key = self.requires
        pred_labels_batch = torch.argmax(bundle.get(pred_key), dim=-1)
        truth_labels_batch = bundle.get(label_key)
        # v1 verbatim: extend iterates dim 0 (scalars for [B], rows for [B, T])
        self.truth_labels.extend(truth_labels_batch)
        self.pred_labels.extend(pred_labels_batch)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Stash the epoch's values, log to Comet when attached, reset."""
        del pl_module
        self.last_truth_labels = self.truth_labels
        self.last_pred_labels = self.pred_labels
        self.last_matrix, self.last_ignored = self.confusion_counts(
            self.truth_labels, self.pred_labels, len(self.task_class_names)
        )
        if isinstance(trainer.logger, CometLogger):
            trainer.logger.experiment.log_confusion_matrix(
                y_true=self.truth_labels,
                y_predicted=self.pred_labels,
                labels=self.task_class_names,
                epoch=trainer.current_epoch,
            )
        self.truth_labels = []
        self.pred_labels = []

    @staticmethod
    def confusion_counts(
        truth: list[Tensor] | Tensor,
        preds: list[Tensor] | Tensor,
        num_classes: int,
    ) -> tuple[Tensor, int]:
        """Compute an integer confusion-counts matrix from accumulated labels.

        Rows are truth classes, columns predicted classes. Entries whose
        truth value lies outside ``[0, num_classes)`` — e.g. the ``-1``
        padding of per-track labels — are dropped and counted in the second
        return value (the transparent reduction the W5 gate applies to BOTH
        the v1 and v2 accumulated lists).

        Parameters
        ----------
        truth : list[Tensor] | Tensor
            Accumulated truth labels (scalars or per-token rows).
        preds : list[Tensor] | Tensor
            Accumulated argmax predictions, same layout as `truth`.
        num_classes : int
            The number of classes ``C``.

        Returns
        -------
        tuple[Tensor, int]
            The ``[C, C]`` int64 counts matrix and the number of dropped
            (out-of-range truth) entries.

        Raises
        ------
        ValueError
            When the flattened truth/pred shapes differ.
        """
        if len(truth) == 0:
            return torch.zeros(num_classes, num_classes, dtype=torch.int64), 0
        t = torch.cat([torch.as_tensor(x).flatten() for x in truth]).long()
        p = torch.cat([torch.as_tensor(x).flatten() for x in preds]).long()
        if t.shape != p.shape:
            raise ValueError(f"truth/pred shapes differ after flattening: {t.shape} vs {p.shape}")
        valid = (t >= 0) & (t < num_classes) & (p >= 0) & (p < num_classes)
        matrix = torch.bincount(
            t[valid] * num_classes + p[valid], minlength=num_classes * num_classes
        ).reshape(num_classes, num_classes)
        return matrix, int((~valid).sum())


class GraphArtifacts(Callback):
    """Write the design §4.4 run-dir artifacts at fit/test start (rank zero).

    Per stage start (after `SaltModule.setup` compiled the plans):

    - ``plan_<mode>.txt`` for every stage mode (``fit``+``val`` / ``test``):
      the ordered §4.4 step table for BOTH the dataset plan and the model
      plan, including narrowed wildcard results (the full label list) and
      the demand-narrowed per-group read columns (design §6.1). The TEST
      table appends the writer-sinks section (writer instance -> consumed
      keys, design §8 — which writer consumes which ``preds.*``).
    - ``resolved_io.yaml`` — the §4.4 machine-readable artifact: per mode,
      the plan sources and every module's flattened requires/produces with
      the resolved specs (shape/dtype/kind, declared fields).
    - ``graph_<stage>.dot`` + ``graph_<stage>.<image_format>`` (and
      ``graph_<stage>_dataset.*`` when a `GraphDataModule` is attached):
      the matplotlib layered DAG (`salt.core.render.render_graph` — the
      container has no graphviz binary), DOT alongside for manual
      re-rendering.

    Never fails a run: a LightningModule without compiled plans, or a
    missing matplotlib, degrade to a warning / DOT-only output.

    Parameters
    ----------
    output_dir : str | None, optional
        Artifact directory, by default None — the trainer log dir on the
        fit path (``default_root_dir`` until the M6 run-dir layout lands)
        and the CHECKPOINT's directory on the test path, where the eval H5
        goes (M3-review fix: the old bare-cwd default split eval outputs
        across two places and polluted the invocation cwd).
    image_format : str, optional
        Graph image suffix (``svg``/``png``/``pdf``), by default ``svg``.
    """

    def __init__(self, output_dir: str | None = None, image_format: str = "svg") -> None:
        self.output_dir = output_dir
        self.image_format = image_format.lstrip(".")

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Write fit+val plan tables and the FIT graph before training starts."""
        self._write(trainer, pl_module, stage="fit", mode_names=("fit", "val"))

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Write the test plan table and TEST graph before evaluation starts."""
        self._write(trainer, pl_module, stage="test", mode_names=("test",))

    # -- internals -----------------------------------------------------------

    def _write(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str,
        mode_names: tuple[str, ...],
    ) -> None:
        """Write all artifacts for one stage (rank zero, never raising)."""
        if not trainer.is_global_zero:
            return
        plans: Mapping[Mode, Plan] = getattr(pl_module, "plans", None) or {}
        if not plans:
            warnings.warn(
                f"GraphArtifacts: {type(pl_module).__name__} has no compiled plans — "
                "run-dir artifacts skipped (design §4.4 expects a SaltModule)",
                stacklevel=2,
            )
            return
        out_dir = Path(self.output_dir) if self.output_dir else self._default_dir(trainer, stage)
        out_dir.mkdir(parents=True, exist_ok=True)
        writer_sinks = self._writer_sinks(trainer, pl_module) if stage == "test" else None
        for mode_name in mode_names:
            plan = plans.get(Mode[mode_name.upper()])
            if plan is None:
                continue
            dataset = self._stage_dataset(trainer, mode_name)
            (out_dir / f"plan_{mode_name}.txt").write_text(
                self._plan_text(plan, dataset, writer_sinks if mode_name == "test" else None)
            )
        (out_dir / "resolved_io.yaml").write_text(self._resolved_io(trainer, plans, mode_names))
        # one graph per stage: FIT (VAL is contractually identical, §3.4) / TEST
        plan = plans[Mode[stage.upper()]]
        modules = getattr(pl_module, "_graph_modules", None) or {}
        pruned = sorted(set(modules) - set(plan.module_names))
        self._render(plan, modules, pruned, out_dir / f"graph_{stage}", f"model graph — {stage}")
        dataset = self._stage_dataset(trainer, stage)
        if dataset is not None:
            self._render(
                dataset.plan,
                dataset.modules,
                sorted(set(dataset.modules) - set(dataset.plan.module_names)),
                out_dir / f"graph_{stage}_dataset",
                f"dataset graph — {stage}",
            )
        print(f"wrote graph/plan artifacts to {out_dir} (design §4.4)")

    @staticmethod
    def _default_dir(trainer: Trainer, stage: str) -> Path:
        """The stage's default artifact directory (class docstring rationale).

        Returns
        -------
        Path
            Test: the checkpoint's directory (where the eval H5 goes) when
            ``trainer.ckpt_path`` is known; otherwise — and always on the
            fit path — the trainer log dir.
        """
        ckpt_path = getattr(trainer, "ckpt_path", None)
        if stage == "test" and ckpt_path:
            return Path(ckpt_path).parent
        return Path(trainer.log_dir or trainer.default_root_dir)

    @staticmethod
    def _writer_sinks(trainer: Trainer, pl_module: LightningModule) -> list[str] | None:
        """Per-writer consumed-keys lines for the TEST plan table (design §8).

        Returns
        -------
        list[str] | None
            ``"name (Class): key, key"`` lines from the attached writer
            callback's `per_writer_demand`, or None when no writer callback
            / reader / graph-module dict is attached (artifacts never fail
            a run).
        """
        callbacks = getattr(trainer, "callbacks", None) or []
        cb = next((c for c in callbacks if callable(getattr(c, "per_writer_demand", None))), None)
        reader = getattr(getattr(trainer, "datamodule", None), "reader", None)
        modules = getattr(pl_module, "_graph_modules", None)
        if cb is None or reader is None or not modules:
            return None
        try:
            per_writer = cb.per_writer_demand(modules, reader)
        except GraphError:
            return None  # a broken writers block fails the run elsewhere
        return [
            f"  {name} ({type(cb.writers[name]).__name__}): " + ", ".join(keys)
            for name, keys in per_writer.items()
        ]

    @staticmethod
    def _plan_text(
        plan: Plan, dataset: GraphDataset | None, writer_sinks: list[str] | None = None
    ) -> str:
        """Build one ``plan_<mode>.txt`` payload: dataset plan + model plan.

        Returns
        -------
        str
            The artifact text (trailing newline included); `writer_sinks`
            lines (TEST only) append as the writer-sinks section.
        """
        sections: list[str] = []
        if dataset is not None:
            sections.append(f"# dataset plan (design §6.1)\n{plan_table(dataset.plan)}")
            columns = [
                f"  {stream}: " + ", ".join(f"{field} ({who})" for field, who in fields.items())
                for stream, fields in dataset.read_fields.items()
            ]
            if columns:
                sections.append(
                    "read columns (demand-narrowed, design §6.1):\n" + "\n".join(columns)
                )
        sections.append(f"# model plan (design §3.1)\n{plan_table(plan)}")
        if writer_sinks:
            sections.append("# writer sinks (design §8)\n" + "\n".join(writer_sinks))
        return "\n\n".join(sections) + "\n"

    def _resolved_io(
        self, trainer: Trainer, plans: Mapping[Mode, Plan], mode_names: tuple[str, ...]
    ) -> str:
        """Build the ``resolved_io.yaml`` payload (design §4.4, machine-readable).

        Returns
        -------
        str
            YAML: per mode, the plan ``sources`` and each module's flattened
            ``requires``/``produces`` with resolved specs — dataset-plan
            modules included when a `GraphDataModule` stage dataset exists.
        """
        payload: dict[str, Any] = {}
        for mode_name in mode_names:
            plan = plans.get(Mode[mode_name.upper()])
            if plan is None:
                continue
            model_io = _plan_io(plan)
            section: dict[str, Any] = {
                "plan_hash": plan.plan_hash,
                "sources": model_io["sources"],
                "modules": model_io["modules"],
            }
            dataset = self._stage_dataset(trainer, mode_name)
            if dataset is not None:
                section["dataset_modules"] = _plan_io(dataset.plan)["modules"]
            payload[mode_name] = section
        return yaml.safe_dump(payload, sort_keys=False)

    def _render(
        self,
        plan: Plan,
        modules: Mapping[str, Any],
        pruned: list[str],
        base: Path,
        title: str,
    ) -> None:
        """Write DOT + image for one plan; image degrades to a hint on ImportError."""
        dot_path = base.with_suffix(".dot")
        dot_path.write_text(dot_source(plan, modules, pruned))
        img_path = base.with_suffix(f".{self.image_format}")
        try:
            render_graph(plan, img_path, title=f"{title} [mode={plan.mode.name}]", pruned=pruned)
        except ImportError:
            print(
                f"matplotlib not importable — wrote {dot_path} only; render manually with: "
                f"dot -T{self.image_format} {dot_path} -o {img_path}"
            )

    @staticmethod
    def _stage_dataset(trainer: Trainer, mode_name: str) -> GraphDataset | None:
        """The stage's `GraphDataset` from the attached datamodule, if any.

        Returns
        -------
        GraphDataset | None
            The dataset whose plan matches `mode_name`, or None (duck-typed:
            non-Graph datamodules yield None).
        """
        dm = getattr(trainer, "datamodule", None)
        attr = {"fit": "train_dset", "val": "val_dset", "test": "test_dset"}[mode_name]
        dataset = getattr(dm, attr, None)
        return dataset if hasattr(dataset, "plan") and hasattr(dataset, "read_fields") else None


def _spec_dict(spec: TensorSpec) -> dict[str, Any]:
    """A `TensorSpec` as plain YAML-able data (``resolved_io.yaml``, design §4.4).

    Returns
    -------
    dict[str, Any]
        ``shape``/``dtype``/``kind`` always; ``fields`` when declared.
    """
    out: dict[str, Any] = {
        "shape": list(spec.shape) if spec.shape is not None else None,
        "dtype": spec.dtype,
        "kind": spec.kind,
    }
    if spec.fields is not None:
        out["fields"] = list(spec.fields)
    return out


def _plan_io(plan: Plan) -> dict[str, Any]:
    """One plan's sources + per-module resolved requires/produces (design §4.4).

    Returns
    -------
    dict[str, Any]
        ``{"sources": {key: spec}, "modules": {name: {"class", "requires",
        "produces"}}}`` in plan (topological) order.
    """
    return {
        "sources": {key: _spec_dict(spec) for key, spec in plan.sources.items()},
        "modules": {
            step.name: {
                "class": type(step.module).__name__,
                "requires": {key: _spec_dict(spec) for key, spec in step.requires.items()},
                "produces": {key: _spec_dict(spec) for key, spec in step.produces.items()},
            }
            for step in plan.steps
        },
    }
