import json
from datetime import datetime
from pathlib import Path

from lightning import Callback, LightningModule, Trainer


class PerformanceWriter(Callback):
    """Write performance metrics to json file.

    Parameters
    ----------
    dir_path : str | None, optional
        Path to save json file, by default None
    add_metrics : list | None, optional
        Optional additional metrics to add to the logs, by default None
    std_out : bool, optional
        Whether to print the performance to std_out, by default False
    """

    def __init__(
        self,
        dir_path: str | None = None,
        add_metrics: list | None = None,
        std_out: bool = False,
    ) -> None:
        super().__init__()
        self.dir_path = dir_path
        self.metrics = ["train_loss", "val_loss", "val_jet_classification_loss"]
        self.metrics += add_metrics if add_metrics is not None else []
        self.std_out = std_out

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
            metrics_dict = {
                "epoch": module.current_epoch,
                **{
                    metric: f"{trainer.callback_metrics[metric]:.5f}"
                    for metric in self.metrics
                    if metric in trainer.callback_metrics
                },
                "timestamp": timestamp.isoformat(),
            }
            perf_file.write(json.dumps(metrics_dict, default=str))
            perf_file.write("\n")  # need for the njson format

            if self.std_out:
                phrase_metrics_list = [
                    f"{metric}={trainer.callback_metrics[metric]:.5f}\n"
                    for metric in self.metrics
                    if metric in trainer.callback_metrics
                ]
                content = (
                    f"epoch {module.current_epoch}:\n"
                    f"{''.join(phrase_metrics_list)}timestamp={timestamp}\n"
                )
                print(content)
