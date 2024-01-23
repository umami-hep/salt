import json
from datetime import datetime
from pathlib import Path

from lightning import Callback, LightningModule, Trainer


class PerformanceWriter(Callback):
    def __init__(
        self,
        dir_path: str | None = None,
        add_metrics: list | None = None,
        stdOut: bool = False,
    ) -> None:
        """Write performance metrics to json file.

        Parameters
        ----------
        dir_path : str
            Path to save json file
        add_metrics: list of str
            Optional additional metrics to add to the logs
        stdOut: bool
            Optional: whether to print the performance to stdOut
        """
        super().__init__()
        self.dir_path = dir_path
        self.metrics = ["train_loss", "val_loss", "val_jet_classification_loss"]
        self.metrics += add_metrics if add_metrics is not None else []
        self.stdOut = stdOut

    def setup(self, trainer: Trainer, module: LightningModule, stage: str) -> None:  # noqa: ARG002
        if trainer.fast_dev_run:
            return

        out_dir = Path(trainer.log_dir) if self.dir_path is None else Path(self.dir_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = Path(out_dir / "performance_metric.json")
        with open(self.path, "w") as _:
            pass

    def on_validation_epoch_end(self, trainer, module) -> None:
        if trainer.state.stage != "validate" or trainer.fast_dev_run:
            return

        with open(self.path, "a") as perf_file:
            timestamp = datetime.now()
            addMetrics = {
                "epoch": module.current_epoch,
                **{
                    metric: f"{trainer.callback_metrics[metric]:.5f}"
                    for metric in self.metrics
                    if metric in trainer.callback_metrics
                },
                "timestamp": timestamp.isoformat(),
            }
            perf_file.write(json.dumps(addMetrics, default=str))
            perf_file.write("\n")  # need for the njson format

            if self.stdOut:
                phraseMetrics = [
                    f"{metric}={trainer.callback_metrics[metric]:.5f}\n"
                    for metric in self.metrics
                    if metric in trainer.callback_metrics
                ]
                content = (
                    f"epoch {module.current_epoch}:\n"
                    f"{''.join(phraseMetrics)}timestamp={timestamp}\n"
                )
                print(content)
