"""Checkpoint-load guards on `SaltModule.on_load_checkpoint`."""

from __future__ import annotations

import pytest
import torch

from salt.core.graph.errors import ConfigError
from salt.core.saltmodule import SaltModule


def _on_load(checkpoint: dict) -> None:
    """Run the real ``on_load_checkpoint`` on a bare instance (no fit needed)."""
    module = SaltModule.__new__(SaltModule)
    SaltModule.on_load_checkpoint(module, checkpoint)


class TestV1CheckpointRejected:
    def test_v1_layout_raises_explicit_config_error(self):
        """A v1 (ModelWrapper) state-dict layout fails EARLY with a clear message.

        v1 checkpoints carry ``model.pool_net.*`` keys — salt.core must reject
        them with one explicit error (pin 29c67a1 / offline conversion), not a
        cascade of hundreds of missing-key failures from the strict load.
        """
        checkpoint = {
            "state_dict": {
                "model.pool_net.gate_nn.weight": torch.zeros(1, 16),
                "model.encoder.layers.0.attn.in_proj_weight": torch.zeros(48, 16),
                "norm.jets_means": torch.zeros(2),
            }
        }
        with pytest.raises(ConfigError, match=r"v1 .*not supported.*29c67a1"):
            _on_load(checkpoint)

    def test_v2_layout_passes_the_guard(self):
        """A v2-shaped state dict sails past the v1 guard (payload warning only)."""
        checkpoint = {"state_dict": {"net.pool.gate_nn.weight": torch.zeros(1, 16)}}
        with pytest.warns(UserWarning, match="no 'salt_core' payload"):
            _on_load(checkpoint)
