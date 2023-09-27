from torch import Tensor, nn
from torch.nn import ModuleList

from salt.models import InitNet


class MLPTagger(nn.Module):
    def __init__(
        self,
        init_nets: list[dict],
        tasks: ModuleList,
    ):
        """Per-object tagger, can be used to implement DL1.

        The model consists of a single initial dense embedding network, followed by a
        number of tasks.

        Parameters
        ----------
        init_nets: list[dict]
            Keyword arguments used to instantiate a [salt.models.InitNet][salt.models.InitNet].
            Only one input type is supported for a dense tagger, so the list must be of length 1.
        tasks : ModuleList
            List of tasks. Each task inherits from
            [salt.models.task.TaskBase][salt.models.task.TaskBase].
        """
        super().__init__()

        if len(init_nets) != 1:
            raise ValueError(f"MLPTagger can only handle a single init_net, not {len(init_nets)}")
        self.init_nets = nn.ModuleList([InitNet(**init_net) for init_net in init_nets])
        self.init_net = self.init_nets[0]
        self.tasks = tasks

        if self.init_net.name != "jet":
            raise ValueError(
                f"MLPTagger can only handle init_net with name 'jet', not '{self.init_net.name}'"
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
