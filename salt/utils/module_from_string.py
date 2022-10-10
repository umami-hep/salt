import torch.nn as nn


def get_module(s: str = None):
    if s is None:
        return s

    if type(s) != str:
        raise ValueError(f"Should provide a string, not {s}")

    if s == "relu":
        return nn.ReLU
    elif s == "silu":
        return nn.SiLU

    elif s == "layernorm":
        return nn.LayerNorm
    elif s == "batchnorm":
        return nn.BatchNorm

    raise ValueError(f"Module {s} not recognised.")
