"""``GenModule`` base for the modular test-data generator pipeline.

A pipeline is an ordered ``list[GenModule]``. Each module declares its data
contract as instance-level lists of *contract keys* (``requires`` /
``produces`` / ``mutates``) and implements ``__call__(data, rng) -> data``.

A contract key is either a GROUP key (``"tracks"``) or a FIELD key
(``"tracks.ftagTruthParentBarcode"``) -- a naming convention over a structured
array's ``dtype.names``, not a change in storage.

See ``design/03_modular_generator.md`` §1.
"""

from __future__ import annotations

import numpy as np


class GenModule:
    """Base for all generation modules.

    Subclasses set their contract in ``__init__`` (the produced group name is an
    init-arg, e.g. ``Constituents(name="tracks")`` produces ``"tracks"``), so the
    contract is instance-level rather than class-level.

    The ``n_samples`` / ``flags`` attributes default to ``None`` so the
    ``Pipeline`` can inject its pipeline-level values at run time unless the
    module set an explicit override (see §2.7).
    """

    # Contract -- populated by __init__. Defaults empty.
    requires: list[str] = []
    produces: list[str] = []
    mutates: list[str] = []

    # Pipeline-injected; None means "use the pipeline value".
    n_samples: int | None = None
    flags: dict[str, bool] | None = None

    def __call__(
        self, data: dict[str, np.ndarray], rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """Mutate/extend ``data`` and return it. Must honour the declared contract."""
        raise NotImplementedError

    # -- helpers shared by concrete modules ------------------------------- #
    @property
    def _flags(self) -> dict[str, bool]:
        return self.flags or {}

    def group_spec(self):
        """Return this module's ``GroupSpec`` (parsed), or ``None`` for writers.

        Producer modules override this so the pipeline can collect every group's
        spec and hand a reconstructed thin ``Schema`` to the writers (design §6
        decided default 2). Writers return ``None``.
        """
        return None
