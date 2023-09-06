import torch
from torch import Tensor, nn
from torch.nn import ModuleList


class R21Xbb(nn.Module):
    def __init__(self, tasks: ModuleList):
        super().__init__()
        self.tasks = tasks
        self.init_nets = False

    def forward(self, inputs: dict, mask=None, labels: dict | None = None):
        x = inputs["track"]
        x = torch.flatten(x, start_dim=1)
        preds, loss = self.tasks_forward(x, labels)
        return preds, loss

    def tasks_forward(
        self,
        x: Tensor,
        labels: dict | None = None,
    ):
        preds = {}
        loss = {}
        for task in self.tasks:
            if task.input_type == "jet":
                task_input = x
            else:
                print("WARNING: R21Xbb is not configured for input_type other than jet")
                print("WARNING: R21Xbb shall go on jet tagging")
                task_input = x
            task_preds, task_loss = task(task_input, labels, None, context=None)
            preds[task.name] = task_preds
            loss[task.name] = task_loss

        return preds, loss
