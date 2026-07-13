"""Shared helpers for the callbacks test mirrors (split from test_callbacks.py, W45.2c)."""

from __future__ import annotations

import torch

from salt.core.graph.bundle import Bundle


LRS = {"initial": 1e-4, "max": 1e-3, "end": 1e-5, "pct_start": 0.1}


def make_matched_bundle(seed: int = 5, batch: int = 6, m: int = 5, n_cls: int = 3, t: int = 10):
    """A VAL step bundle carrying the matcher-permuted ``matched.objects.*`` keys."""
    from salt.tests._fixtures.regression_fixture import MASKFORMER_WRITER_REG_TARGETS

    gen = torch.Generator().manual_seed(seed)
    bundle = Bundle()
    bundle.set("matched.objects.class_logits", torch.randn(batch, m, n_cls, generator=gen))
    bundle.set("matched.objects.object_class", torch.randint(0, n_cls, (batch, m), generator=gen))
    bundle.set("matched.objects.masks", torch.randn(batch, m, t, generator=gen))
    bundle.set("matched.objects.target_masks", torch.rand(batch, m, t, generator=gen) > 0.5)
    bundle.set(
        "matched.objects.regression",
        torch.randn(batch, m, len(MASKFORMER_WRITER_REG_TARGETS), generator=gen),
    )
    bundle.set(
        "matched.objects.target_regression",
        torch.randn(batch, m, len(MASKFORMER_WRITER_REG_TARGETS), generator=gen),
    )
    bundle.set("masks.tracks", torch.zeros(batch, t, dtype=torch.bool))
    return bundle
