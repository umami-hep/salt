"""`undo_padding`/`redo_padding` under ``torch.compile`` (the flash-varlen unpad seam)."""

from __future__ import annotations

import torch

from salt.utils.tensor_utils import redo_padding, undo_padding

B, L, D = 6, 12, 8


def _batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """A padded ``[B, L, D]`` batch and its ``True == padded`` mask.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The batch and its padding mask.
    """
    generator = torch.Generator().manual_seed(seed)
    seq = torch.randn(B, L, D, generator=generator)
    lengths = torch.randint(1, L, (B,), generator=generator)
    mask = torch.arange(L)[None, :] >= lengths[:, None]
    return seq, mask


def _roundtrip(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    packed, culens, maxlen = undo_padding(seq, mask)
    del culens, maxlen
    return redo_padding(packed, mask)


class TestUnpadRepadCompile:
    def test_eager_roundtrip_zeroes_padding(self):
        seq, mask = _batch()
        out = _roundtrip(seq, mask)
        assert out.shape == seq.shape
        # valid positions survive untouched, padded positions are zeroed
        assert torch.equal(out[~mask], seq[~mask])
        assert torch.equal(out[mask], torch.zeros_like(out[mask]))

    def test_metadata_matches_mask(self):
        seq, mask = _batch(seed=3)
        valid = (~mask).sum(dim=-1)
        packed, culens, maxlen = undo_padding(seq, mask)
        assert packed.shape[0] == int(valid.sum())
        assert maxlen == int(valid.max())
        expected = torch.cat([torch.zeros(1, dtype=torch.int32), valid.cumsum(0).int()])
        assert torch.equal(culens, expected)

    def test_compiled_matches_eager_bitwise(self):
        """The compiler-disabled seam must not change a single bit of the result."""
        seq, mask = _batch(seed=7)
        expected = _roundtrip(seq, mask)
        torch._dynamo.reset()  # noqa: SLF001 - the dynamo test surface
        compiled = torch.compile(_roundtrip, backend="eager")
        assert torch.equal(compiled(seq, mask), expected)

    def test_compiled_handles_a_second_batch_of_different_length(self):
        """The packed dim is marked dynamic — a new token count must not blow up."""
        torch._dynamo.reset()  # noqa: SLF001 - the dynamo test surface
        compiled = torch.compile(_roundtrip, backend="eager")
        for seed in (1, 2, 3):
            seq, mask = _batch(seed=seed)
            assert torch.equal(compiled(seq, mask), _roundtrip(seq, mask))

    def test_helpers_stay_outside_the_graph(self):
        """Dynamo must break at the seam rather than trace the unpad/repad.

        Behavioural, not attribute-sniffing: if the ``torch.compiler.disable``
        markers are dropped, dynamo traces the boolean indexing into the graph and
        the break count goes to zero — which is precisely the state that makes
        inductor reject ``aten.nonzero`` on CUDA (investigation 01, F9).
        """
        seq, mask = _batch(seed=11)
        torch._dynamo.reset()  # noqa: SLF001 - the dynamo test surface
        explanation = torch._dynamo.explain(_roundtrip)(seq, mask)  # noqa: SLF001 - dynamo API
        assert explanation.graph_break_count > 0, (
            "no graph break at the unpad/repad seam — the compiler-disable markers on "
            "undo_padding/redo_padding have been lost"
        )
