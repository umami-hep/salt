"""`GraphArtifacts` — run-dir plan/graph artifact writer."""

from __future__ import annotations

import shutil
import subprocess
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from lightning import Callback, LightningModule, Trainer

from salt.core.data.dataset import GraphDataset
from salt.core.graph.errors import GraphError
from salt.core.graph.planner import Plan
from salt.core.graph.spec import Mode, TensorSpec
from salt.core.render import dot_source, plan_table


class GraphArtifacts(Callback):
    """Write run-dir artifacts (plan tables, resolved I/O, graph images) at fit/test start.

    Per stage start (after `SaltModule.setup` compiled the plans), writes at
    rank zero:

    - ``plan_<mode>.txt`` for every stage mode (``fit``+``val`` / ``test``):
      the ordered step table for both the dataset plan and the model plan.
      The TEST table appends the writer-sinks section (writer instance ->
      consumed ``preds.*`` keys).
    - ``resolved_io.yaml`` — machine-readable: per mode, the plan sources
      and every module's flattened requires/produces with resolved specs.
    - ``graph_<stage>.dot`` + ``graph_<stage>.<image_format>`` (and
      ``graph_<stage>_dataset.*`` when a `GraphDataModule` is attached):
      Graphviz DOT rasterised via the ``dot`` binary, kept alongside for
      manual re-rendering.

    Never fails a run: a LightningModule without compiled plans, or a
    missing/failing ``dot`` binary, degrade to a warning / DOT-only output.

    Parameters
    ----------
    output_dir : str | None, optional
        Artifact directory, by default None — the trainer log dir on the
        fit path, and the checkpoint's directory on the test path (so eval
        outputs land next to the checkpoint and the eval H5).
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
        # one graph per stage: FIT (VAL is identical) / TEST
        plan = plans[Mode[stage.upper()]]
        modules = getattr(pl_module, "_graph_modules", None) or {}
        pruned = sorted(set(modules) - set(plan.module_names))
        self._render(plan, modules, pruned, out_dir / f"graph_{stage}")
        dataset = self._stage_dataset(trainer, stage)
        if dataset is not None:
            self._render(
                dataset.plan,
                dataset.modules,
                sorted(set(dataset.modules) - set(dataset.plan.module_names)),
                out_dir / f"graph_{stage}_dataset",
            )
        print(f"wrote graph/plan artifacts to {out_dir} (design §4.4)")

    @staticmethod
    def _default_dir(trainer: Trainer, stage: str) -> Path:
        """The stage's default artifact directory.

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
        """Per-writer consumed-keys lines for the TEST plan table.

        Returns
        -------
        list[str] | None
            ``"name (Class): key, key"`` lines from the attached writer
            callback's `per_writer_demand`, or None when no writer callback
            / reader / graph-module dict is attached.
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
            The artifact text; `writer_sinks` lines (TEST only) append as
            the writer-sinks section.
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
        """Build the ``resolved_io.yaml`` payload.

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
    ) -> None:
        """Write DOT + image for one plan via the `dot` binary; degrade to DOT-only.

        Best-effort: a missing/failing ``dot`` degrades to a DOT-only hint
        and never crashes a training run.
        """
        dot_path = base.with_suffix(".dot")
        dot_path.write_text(dot_source(plan, modules, pruned))
        img_path = base.with_suffix(f".{self.image_format}")
        hint = f"render manually with: dot -T{self.image_format} {dot_path} -o {img_path}"
        dot_bin = shutil.which("dot")
        if dot_bin is None:
            print(f"graphviz `dot` not on PATH — wrote {dot_path} only; {hint}")
            return
        result = subprocess.run(
            [dot_bin, f"-T{self.image_format}", str(dot_path), "-o", str(img_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"`dot` failed to render {img_path} (exit {result.returncode}): "
                f"{result.stderr.strip() or '(no stderr)'} — wrote {dot_path} only; {hint}"
            )

    @staticmethod
    def _stage_dataset(trainer: Trainer, mode_name: str) -> GraphDataset | None:
        """The stage's `GraphDataset` from the attached datamodule, if any.

        Returns
        -------
        GraphDataset | None
            The dataset whose plan matches `mode_name`, or None for
            non-Graph datamodules.
        """
        dm = getattr(trainer, "datamodule", None)
        attr = {"fit": "train_dset", "val": "val_dset", "test": "test_dset"}[mode_name]
        dataset = getattr(dm, attr, None)
        return dataset if hasattr(dataset, "plan") and hasattr(dataset, "read_fields") else None


def _spec_dict(spec: TensorSpec) -> dict[str, Any]:
    """A `TensorSpec` as plain YAML-able data (for ``resolved_io.yaml``).

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
    """One plan's sources + per-module resolved requires/produces.

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
