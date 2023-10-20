from typing import TypeAlias

from torch import BoolTensor, Tensor

Vars: TypeAlias = dict[str, list[str]]
Tensors: TypeAlias = dict[str, Tensor]
BoolTensors: TypeAlias = dict[str, BoolTensor]
NestedTensors: TypeAlias = dict[str, dict[str, Tensor]]
