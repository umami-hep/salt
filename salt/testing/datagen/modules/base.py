"""``GenModule`` base for the modular test-data generator pipeline: modules
declare ``requires``/``produces``/``mutates`` contract keys and implement
``__call__(data, rng) -> data``.
"""

from __future__ import annotations

import numpy as np

from ..engine import _build_group_array
from ..schema import parse_schema
from ._util import infer_n


class GenModule:
    """Base for all generation modules.

    Subclasses set their contract in ``__init__`` (the produced group name is an
    init-arg, e.g. ``Constituents(name="tracks")`` produces ``"tracks"``), so the
    contract is instance-level rather than class-level.

    The ``n_samples`` / ``flags`` attributes default to ``None`` so the
    ``Pipeline`` can inject its pipeline-level values at run time unless the
    module set an explicit override.
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
        spec and hand a reconstructed thin ``Schema`` to the writers. Writers
        return ``None``.
        """
        return


class _GroupModule(GenModule):
    """Shared body of the single-group producers: subclasses set ``name``,
    ``_group_dict``, ``_fill``, ``n_samples``, ``flags`` and ``_group_spec``.
    """

    def _schema(self, data):
        sch = parse_schema({
            "n_samples": infer_n(data, self.n_samples),
            "groups": [self._group_dict],
            **self._fill,
        })
        self._group_spec = sch.group(self.name)
        return sch

    def __call__(self, data, rng):
        sch = self._schema(data)
        group = sch.group(self.name)
        arr, _ = _build_group_array(rng, sch, group, self.flags or {})
        data[self.name] = arr
        return data

    def group_spec(self):
        if self._group_spec is None:
            # build (no data needed: n_samples irrelevant for spec)
            self._schema({})
        return self._group_spec
