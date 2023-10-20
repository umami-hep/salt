from torch import Tensor, nn
from torch.nn import ModuleList

from salt.models import InitNet
from salt.utils.typing import BoolTensors, NestedTensors, Tensors


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
            Keyword arguments used to instantiate a [`salt.models.InitNet`][salt.models.InitNet].
            Only one input type is supported for a dense tagger, so the list must be of length 1.
        tasks : ModuleList
            List of tasks. Each task inherits from
            [`salt.models.task.TaskBase`][salt.models.task.TaskBase].
        """
        super().__init__()

        if len(init_nets) != 1:
            raise ValueError(f"MLPTagger can only handle one init_net, not {len(init_nets)}")

        # keep self.init_nets for compatibility with other models
        self.init_nets = nn.ModuleList([InitNet(**init_net) for init_net in init_nets])
        self.init_net = self.init_nets[0]
        self.tasks = tasks

    def forward(
        self, inputs: Tensors, mask: BoolTensors | None = None, labels: NestedTensors | None = None
    ):
        assert not mask, "MLPTagger does not support masking"
        return self.tasks_forward(self.init_net(inputs), labels)

    def tasks_forward(self, x: Tensor, labels: NestedTensors | None = None):
        preds: NestedTensors = {}
        loss = {}
        for task in self.tasks:
            if task.input_name not in preds:
                preds[task.input_name] = {}
            preds[task.input_name][task.name], loss[task.name] = task(x, labels)
        return preds, loss
