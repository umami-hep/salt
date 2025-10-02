import torch
from torch import Tensor, nn
from torch.nn import ModuleList


class R21Xbb(nn.Module):
    """Wrapper model for running multiple jet-tagging tasks on flattened track inputs.

    This module takes track-level inputs from a dict, flattens them, and
    forwards them to each task head specified in ``self.tasks``. Each task head
    is expected to implement a callable returning both predictions and a loss
    dictionary when given the flattened input.

    Parameters
    ----------
    tasks : ModuleList
        List of task modules. Each task module should define at least:
        - ``task.input_name`` (used to determine input source),
        - ``task.name`` (used as key for predictions/losses),
        - a ``__call__`` signature like
            ``task(x: Tensor, labels: dict | None, pad_mask, context=None)``
            returning ``(task_preds, task_loss)``.
    """

    def __init__(self, tasks: ModuleList):
        super().__init__()
        self.tasks = tasks
        self.init_nets: list = []

    def forward(
        self,
        inputs: dict,
        pad_masks: dict | None = None,
        labels: dict | None = None,
    ) -> tuple[dict, dict]:
        """Forward pass through the R21Xbb wrapper.

        Flattens the track input from ``inputs["track"]`` and feeds it to all tasks.

        Parameters
        ----------
        inputs : dict
            Dictionary containing at least ``"track"`` with shape ``[B, L, D]``.
        pad_masks : dict | None, optional
            Optional padding masks. Not used in this implementation but included
            for API consistency. The default is ``None``.
        labels : dict | None, optional
            Optional ground-truth labels forwarded to each task. The default is ``None``.

        Returns
        -------
        dict
            Dictionary of predictions from each task keyed by ``task.name``.
        dict
            Dictionary of loss values from each task keyed by ``task.name``.
        """
        _ = pad_masks
        x = inputs["track"]
        x = torch.flatten(x, start_dim=1)
        preds, loss = self.run_tasks(x, labels)
        return preds, loss

    def run_tasks(
        self,
        x: Tensor,
        labels: dict | None = None,
    ) -> tuple[dict, dict]:
        """Run all configured tasks on the given flattened input.

        Parameters
        ----------
        x : Tensor
            Flattened input tensor of shape ``[B, F]`` where ``F`` is the
            total flattened feature size of ``inputs["track"]``.
        labels : dict | None, optional
            Optional ground-truth labels forwarded to each task. The default is ``None``.

        Returns
        -------
        dict
            Dictionary of predictions from each task keyed by ``task.name``.
        dict
            Dictionary of loss values from each task keyed by ``task.name``.

        Notes
        -----
        - Currently assumes all tasks take the same flattened track input.
          Warnings are printed if ``task.input_name`` differs from
          ``self.tasks[0].global_object``.
        """
        preds: dict = {}
        loss: dict = {}
        for task in self.tasks:
            if task.input_name == self.tasks[0].global_object:
                task_input = x
            else:
                print("WARNING: R21Xbb is not configured for input_name other than jet")
                print("WARNING: R21Xbb shall go on jet tagging")
                task_input = x
            task_preds, task_loss = task(task_input, labels, None, context=None)
            preds[task.name] = task_preds
            loss[task.name] = task_loss

        return preds, loss
