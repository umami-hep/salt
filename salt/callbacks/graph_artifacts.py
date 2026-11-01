"""`GraphArtifacts` — run-dir plan/graph artifact writer."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from lightning import Callback, LightningModule, Trainer

from salt.data.dataset import SaltDataset
from salt.graph.errors import GraphError
from salt.graph.planner import Plan
from salt.graph.render import graph_dot, plan_table, pruned_modules, render_image
from salt.graph.spec import Mode, TensorSpec
from salt.model.bind import resolve_bind_schema
from salt.utils.logging import get_logger

_LOG = get_logger(__name__)


class GraphArtifacts(Callback):
    """Write run-dir artifacts (plan tables, resolved I/O, graph images) at fit/test start.

    Per stage start (after `SaltModule.setup` compiled the plans), writes at
    rank zero:

    - ``plan_<mode>.txt`` for every stage mode (``fit``+``val`` / ``test``):
      the ordered step table for both the dataset plan and the model plan.
    - ``resolved_io.yaml`` — machine-readable: per mode, the plan sources
      and every module's flattened requires/produces with resolved specs.
    - ``graph_<stage>.{dot,png,pdf}`` (``<stage>`` = ``fit`` / ``test``; VAL
      shares the FIT graph), plus ``graph_<stage>_dataset.{dot,png,pdf}`` when
      a `SaltDataModule` is attached: the same render as ``salt graph plot``
      (`salt.graph.render.graph_dot` with the statically resolved feature
      widths, PNG at 150 dpi + PDF via `salt.graph.render.render_image`); the
      DOT is kept for manual re-rendering.

    Never fails a run: a LightningModule without compiled plans, unresolvable
    widths, or a missing/failing ``dot`` binary degrade to a warning (DOT
    without widths / DOT-only output).

    Parameters
    ----------
    output_dir : str | None, optional
        Artifact directory, by default None — the trainer log dir on the
        fit path, and the checkpoint's directory on the test path (so eval
        outputs land next to the checkpoint and the eval H5).
    """

    def __init__(self, output_dir: str | None = None) -> None:
        self.output_dir = output_dir

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Write fit+val plan tables and the FIT graph before training starts."""
        self._write(trainer, pl_module, stage="fit", mode_names=("fit", "val"))

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Write the test plan table and TEST graph before evaluation starts."""
        self._write(trainer, pl_module, stage="test", mode_names=("test",))

    # -- internals -----------------------------------------------------------

    def _write(
        self, trainer: Trainer, pl_module: LightningModule, stage: str, mode_names: tuple[str, ...]
    ) -> None:
        """Write all artifacts for one stage (rank zero, never raising)."""
        if not trainer.is_global_zero:
            return
        plans: Mapping[Mode, Plan] = getattr(pl_module, "plans", None) or {}
        if not plans:
            warnings.warn(
                f"GraphArtifacts: {type(pl_module).__name__} has no compiled plans — "
                "run-dir artifacts skipped (expected a SaltModule)",
                stacklevel=2,
            )
            return
        out_dir = Path(self.output_dir) if self.output_dir else self._default_dir(trainer, stage)
        out_dir.mkdir(parents=True, exist_ok=True)
        for mode_name in mode_names:
            plan = plans.get(Mode[mode_name.upper()])
            if plan is None:
                continue
            dataset = self._stage_dataset(trainer, mode_name)
            (out_dir / f"plan_{mode_name}.txt").write_text(self._plan_text(plan, dataset))
        (out_dir / "resolved_io.yaml").write_text(self._resolved_io(trainer, plans, mode_names))
        # one graph per stage: FIT (VAL is identical) / TEST
        widths = _widths(plans)
        plan = plans[Mode[stage.upper()]]
        modules = getattr(pl_module, "_graph_modules", None) or {}
        self._render(plan, modules, widths, out_dir / f"graph_{stage}")
        dataset = self._stage_dataset(trainer, stage)
        if dataset is not None:
            self._render(dataset.plan, dataset.modules, widths, out_dir / f"graph_{stage}_dataset")
        _LOG.info(f"wrote graph/plan artifacts to {out_dir}")

    @staticmethod
    def _default_dir(trainer: Trainer, stage: str) -> Path:
        """Test: the checkpoint's directory when ``trainer.ckpt_path`` is known;
        otherwise (and always on fit) the trainer log dir.
        """
        ckpt_path = getattr(trainer, "ckpt_path", None)
        if stage == "test" and ckpt_path:
            return Path(ckpt_path).parent
        return Path(trainer.log_dir or trainer.default_root_dir)

    @staticmethod
    def _plan_text(plan: Plan, dataset: SaltDataset | None) -> str:
        """Build one ``plan_<mode>.txt`` payload: dataset plan + model plan."""
        sections: list[str] = []
        if dataset is not None:
            sections.append(f"# dataset plan\n{plan_table(dataset.plan)}")
            columns = [
                f"  {stream}: " + ", ".join(f"{field} ({who})" for field, who in fields.items())
                for stream, fields in dataset.read_fields.items()
            ]
            if columns:
                sections.append("read columns (demand-narrowed):\n" + "\n".join(columns))
        sections.append(f"# model plan\n{plan_table(plan)}")
        return "\n\n".join(sections) + "\n"

    def _resolved_io(
        self, trainer: Trainer, plans: Mapping[Mode, Plan], mode_names: tuple[str, ...]
    ) -> str:
        """The ``resolved_io.yaml`` YAML payload: per mode, plan ``sources`` + each
        module's flattened requires/produces, plus dataset-plan modules when a
        `SaltDataModule` stage dataset exists.
        """
        payload: dict[str, Any] = {}
        for mode_name in mode_names:
            plan = plans.get(Mode[mode_name.upper()])
            if plan is None:
                continue
            section: dict[str, Any] = {"plan_hash": plan.plan_hash, **_plan_io(plan)}
            dataset = self._stage_dataset(trainer, mode_name)
            if dataset is not None:
                section["dataset_modules"] = _plan_io(dataset.plan)["modules"]
            payload[mode_name] = section
        return yaml.safe_dump(payload, sort_keys=False)

    @staticmethod
    def _render(
        plan: Plan, modules: Mapping[str, Any], widths: Mapping[str, int] | None, base: Path
    ) -> None:
        """Write ``<base>.dot`` and render ``<base>.{png,pdf}`` (non-strict: never raises)."""
        dot_path = base.with_name(f"{base.name}.dot")
        pruned = pruned_modules(modules, plan)
        dot_path.write_text(graph_dot(plan, modules, pruned=pruned, widths=widths))
        render_image(dot_path, base, strict=False)

    @staticmethod
    def _stage_dataset(trainer: Trainer, mode_name: str) -> SaltDataset | None:
        """The stage's `SaltDataset` from the attached datamodule matching `mode_name`,
        or None for non-Graph datamodules.
        """
        dm = getattr(trainer, "datamodule", None)
        attr = {"fit": "train_dset", "val": "val_dset", "test": "test_dset"}[mode_name]
        dataset = getattr(dm, attr, None)
        return dataset if hasattr(dataset, "plan") and hasattr(dataset, "read_fields") else None


def _widths(plans: Mapping[Mode, Plan]) -> dict[str, int] | None:
    """Static feature widths unified over every compiled plan (as ``salt graph plot``); None
    with a warning when they do not resolve — the graphs then keep symbolic feature dims.
    """
    try:
        return dict(resolve_bind_schema(list(plans.values())).widths)
    except GraphError as err:
        _LOG.warning(f"GraphArtifacts: widths not resolved, graphs keep symbolic dims: {err}")
        return None


def _spec_dict(spec: TensorSpec) -> dict[str, Any]:
    """A `TensorSpec` as plain YAML-able data: ``shape``/``dtype``/``kind`` always,
    ``fields`` when declared.
    """
    shape = list(spec.shape) if spec.shape is not None else None
    out: dict[str, Any] = {"shape": shape, "dtype": spec.dtype, "kind": spec.kind}
    if spec.fields is not None:
        out["fields"] = list(spec.fields)
    return out


def _plan_io(plan: Plan) -> dict[str, Any]:
    """One plan's ``{"sources": {key: spec}, "modules": {name: {"class", "requires",
    "produces"}}}``, in plan (topological) order.
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
