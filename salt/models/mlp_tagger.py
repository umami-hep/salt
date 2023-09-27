from torch import Tensor, nn
from torch.nn import ModuleList

from salt.models import InitNet


class MLPTagger(nn.Module):
    def __init__(
        self,
        init_nets: list[dict],
        tasks: ModuleList,
    ):
        """Jet level tagger, similar to DL1.

        Parameters
        ----------
        init_nets: ModuleList
            Initialisation networks configuration
        tasks : ModuleList
            Task networks
        """
        super().__init__()

        self.init_nets = nn.ModuleList([InitNet(**init_net) for init_net in init_nets])
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
        preds: dict[str, dict[str, Tensor]] = {}
        loss = {}
        for task in self.tasks:
            if task.input_type not in preds:
                preds[task.input_type] = {}
            preds[task.input_type][task.name], loss[task.name] = task(x, labels)
        return preds, loss
