"""FeaturewiseTransformation (FiLM) GraphModule."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import _UNNAMED
from salt.core.nn.dense import Dense, _reject_width_keys

_FEATUREWISE_LAYERS: frozenset[str] = frozenset({"input", "encoder", "global"})
"""Valid FiLM ``layer`` placements: ``input`` applies scale/bias before a
`StreamEmbed`'s projection; ``encoder`` at the start of every encoder layer;
``global`` to the pooled/encoded representation before pooling."""


class FeaturewiseTransformation(nn.Module):
    """Feature-wise (FiLM) scale/bias from per-event ``parameters``.

    https://distill.pub/2018/feature-wise-transformations/. ``layer`` is one
    of ``{"input", "encoder", "global"}``; a scale/bias net is built iff its
    ``dense_config_*`` is truthy — at least one of the two is required.
    """

    def __init__(
        self,
        layer: str,
        num_params: int,
        num_features: int,
        dense_config_scale: dict | None = None,
        dense_config_bias: dict | None = None,
        apply_norm: bool = False,
    ) -> None:
        super().__init__()
        self.name = _UNNAMED
        if layer not in _FEATUREWISE_LAYERS:
            raise ConfigError(
                f"FeaturewiseTransformation: layer must be one of {sorted(_FEATUREWISE_LAYERS)}, "
                f"got {layer!r} (v1 featurewise.py:44)"
            )
        if num_params < 1:
            raise ConfigError(
                f"FeaturewiseTransformation: num_params must be >= 1, got {num_params}"
            )
        if num_features < 1:
            raise ConfigError(
                f"FeaturewiseTransformation: num_features must be >= 1, got {num_features}"
            )
        scale_cfg = dict(dense_config_scale or {})
        bias_cfg = dict(dense_config_bias or {})
        for cfg in (scale_cfg, bias_cfg):
            _reject_width_keys(
                "FeaturewiseTransformation", cfg, ("input_size", "output_size", "context_size")
            )
        # a net is built iff its dense config is truthy — None or {} builds nothing
        build_scale = bool(dense_config_scale)
        build_bias = bool(dense_config_bias)
        if not build_scale and not build_bias:
            raise ConfigError(
                "FeaturewiseTransformation: specify at least one (non-empty) dense_config_scale "
                "or dense_config_bias (v1 featurewise.py:63-66)"
            )
        self.layer = layer
        self.num_params = int(num_params)
        self.num_features = int(num_features)
        self._build_scale = build_scale
        self._build_bias = build_bias
        self.scale_cfg = scale_cfg
        self.bias_cfg = bias_cfg
        self.apply_norm = bool(apply_norm)
        self.scale_net: nn.Module | None = None
        self.bias_net: nn.Module | None = None
        self.norm: nn.Module | None = None
        self._built = False

    def build(self) -> None:
        """Construct the scale/bias `Dense` nets + optional norm (idempotent)."""
        if self._built:
            return
        if self._build_scale:
            self.scale_net = Dense(
                input_size=self.num_params, output_size=self.num_features, **self.scale_cfg
            )
        if self._build_bias:
            self.bias_net = Dense(
                input_size=self.num_params, output_size=self.num_features, **self.bias_cfg
            )
        if self.apply_norm:
            self.norm = nn.LayerNorm(self.num_features)
        self._built = True

    def forward(self, params: Tensor, features: Tensor) -> Tensor:
        """Apply the FiLM scale/bias (and optional norm) to ``features``."""
        assert self._built, "FeaturewiseTransformation.forward before build()"
        if self.scale_net is not None:
            features = self.scale_net(params).unsqueeze(1) * features
        if self.bias_net is not None:
            features = torch.add(features, self.bias_net(params).unsqueeze(1))
        if self.norm is not None:
            features = self.norm(features)
        return features
