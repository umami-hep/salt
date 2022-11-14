import torch.nn as nn

# these need to be directly imported from their modules
from salt.models.dense import Dense
from salt.models.pooling import Pooling
from salt.models.task import Task


class JetTagger(nn.Module):
    def __init__(
        self,
        init_net: Dense = None,
        gnn: nn.Module = None,
        pool_net: Pooling = None,
        jet_net: Task = None,
        track_net: Task = None,
    ):
        """Jet constituent tagger model.

        # TODO: add option to pool separately for each task

        Parameters
        ----------
        init_net : Dense
            Initialisation network
        gnn : nn.Module
            Graph neural network
        pool_net : nn.Module
            Pooling network
        jet_net : Task
            Jet classification task
        track_net : Task
            Track classification task
        """
        super().__init__()

        self.init_net = init_net
        self.gnn = gnn
        self.jet_net = jet_net
        self.pool_net = pool_net
        self.track_net = track_net

    def forward(self, x, mask, labels):
        embd_x = self.init_net(x)
        if self.gnn:
            embd_x = self.gnn(embd_x, mask=mask)
        pooled = self.pool_net(embd_x, mask=mask)

        # run tasks
        preds, loss = self.tasks(pooled, embd_x, mask, labels)

        return preds, loss

    def tasks(self, pooled, embd_x, mask, labels):
        preds = {}
        loss = {}

        for task in self.get_tasks():
            # TODO: make robust with a flag to say what the task is on
            if "jet" in task.name:
                task_input = pooled
                task_mask = None
            else:
                task_input = embd_x
                task_mask = mask
            task_labels = labels[task.name] if labels is not None else None
            task_preds, task_loss = task(task_input, task_labels, task_mask)
            preds[task.name] = task_preds
            loss[task.name] = task_loss

        return preds, loss

    def get_tasks(self):
        tasks = []
        for n in dir(self):
            task = getattr(self, n)
            if not isinstance(task, Task) or task is None:
                continue
            tasks.append(task)
        return tasks
