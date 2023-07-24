from torch import Tensor, nn
from torch.nn import ModuleList


class MLPTagger(nn.Module):
    def __init__(
        self,
        init_nets: ModuleList,
        tasks: ModuleList,
    ):
        """Jet level tagger, similar to DL1.

        Parameters
        ----------
        init_nets: ModuleList
            Initialisation / dense network
        tasks : ModuleList
            Task networks
        """
        super().__init__()

        self.init_nets = init_nets
        self.tasks = tasks

        if len(self.init_nets) != 1:
            raise ValueError("MLPTagger can only handle one init_net")
        self.init_net = self.init_nets[0]

        if self.init_net.name != "jet":
            raise ValueError(
                f"MLPTagger can only handle init_nets with name 'jet', not '{self.init_net.name}'"
            )
        for task in self.tasks:
            if task.input_type != "jet":
                raise ValueError(
                    "MLPTagger can only handle tasks with input_type 'jet', not"
                    f" '{task.input_type}'"
                )

    def forward(self, inputs: dict, mask: dict | None = None, labels: dict | None = None):
        return self.tasks_forward(self.init_net(inputs), labels)

    def tasks_forward(self, x: Tensor, labels: dict | None = None):
        preds = {}
        loss = {}
        for task in self.tasks:
            preds[task.name], loss[task.name] = task(x, labels)
        return preds, loss
