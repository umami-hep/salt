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

- ``maskformer.py``:      ``from <v1 models pkg>.maskformer_loss import MaskFormerLoss``
                            -> ``from .maskformer_loss import MaskFormerLoss``
- ``maskformer_loss.py``: ``from <v1 models pkg>.matcher import HungarianMatcher``
                            -> ``from .matcher import HungarianMatcher``

DEL-1 (single further deviation from verbatim): with the v1 tree retired,
``maskformer.py``'s three shared-infra imports were repointed to the
interface-identical salt.core ports — ``GLU`` (``salt.core.nn.glu``),
``Attention`` (``salt.core.nn.attention``), ``indices_from_mask``
(``salt.core.utils.mask_utils``) — so the MF oracle semantics are unchanged.
``Tensors`` still comes from ``salt.stypes`` and ``Solvers`` from the external
``py_lap_solver`` package.
"""
