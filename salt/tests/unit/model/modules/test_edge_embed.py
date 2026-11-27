"""Unit tests for the edge-feature helpers (mirror of salt/model/modules/edge_embed.py)."""

from __future__ import annotations

import pytest
import torch

from salt.model.modules.edge_embed import calculate_edge_features, check_edge_config


def test_check_edge_config_missing_variable():
    with pytest.raises(ValueError, match="required for edge features"):
        check_edge_config(["kt"], ["eta", "phi"])


def test_check_edge_config_missing_angular():
    with pytest.raises(ValueError, match="require either"):
        check_edge_config(["dR"], ["pt"])
    check_edge_config(["dR"], ["deta", "dphi"])


def test_edge_features_relative_angular_matches_absolute():
    """Jet-relative deta/dphi give the same dR/kt as absolute eta/phi."""
    torch.manual_seed(0)
    batch_size, seq_len = 2, 6
    feats = ["dR", "kt", "z", "isSelfLoop"]

    pt = torch.rand(batch_size, seq_len) + 0.5
    eta = torch.randn(batch_size, seq_len)
    phi = torch.randn(batch_size, seq_len)
    jet_eta, jet_phi = 0.3, -0.7
    deta, dphi = eta - jet_eta, phi - jet_phi

    check_edge_config(feats, ["pt", "eta", "phi"])
    e_abs = calculate_edge_features(
        torch.stack([pt, eta, phi], dim=-1), {"pt": 0, "eta": 1, "phi": 2}, feats
    )
    check_edge_config(feats, ["pt", "deta", "dphi"])
    e_rel = calculate_edge_features(
        torch.stack([pt, deta, dphi], dim=-1), {"pt": 0, "deta": 1, "dphi": 2}, feats
    )

    assert e_abs.shape == (batch_size, seq_len, seq_len, len(feats))
    assert torch.allclose(e_abs, e_rel, atol=1e-6)
    assert not torch.isnan(e_rel).any()
