"""Pinned snapshot of upstream salt MaskFormer at commit 6570e85.

Vendored, verbatim copies of the three MaskFormer-core modules as they exist
upstream at the pinned commit 6570e85. (The three files are unchanged through the
current upstream HEAD 21c90b6 — ``git log 6570e85..upstream/main -- <files>`` is
empty — so the snapshot stays faithful; do NOT assume ``upstream/main`` itself
equals 6570e85 in general, only these three files do.)

- ``maskformer.py``       (was ``salt/models/maskformer.py``)
- ``maskformer_loss.py``  (was ``salt/models/maskformer_loss.py``)
- ``matcher.py``          (was ``salt/models/matcher.py``)

These are the INDEPENDENT v1 oracle for the MaskFormer parity gates (MF1a, MF1c
in ``salt/tests/integration/gates_m5.py``). They are pinned here so the oracle
stays bitwise-faithful to upstream 6570e85 even as the live
``salt/models/maskformer*.py`` / ``matcher.py`` diverge under the MFU absorption
work (the live worktree copies already carry extra features such as
``constituent_name``, ``class_weights`` and the scipy matcher backend).

Only the two cross-module imports were edited to relative form so the snapshot
is self-contained:

- ``maskformer.py``:      ``from salt.models.maskformer_loss import MaskFormerLoss``
                            -> ``from .maskformer_loss import MaskFormerLoss``
- ``maskformer_loss.py``: ``from salt.models.matcher import HungarianMatcher``
                            -> ``from .matcher import HungarianMatcher``

All remaining imports point at STABLE, shared v1-oracle infra that is byte-
identical between upstream 6570e85 and the worktree (verified): ``GLU`` /
``Attention`` from ``salt.models.transformer``, ``Tensors`` from ``salt.stypes``,
``indices_from_mask`` from ``salt.utils.mask_utils``, and ``Solvers`` from the
external ``py_lap_solver`` package.
"""
