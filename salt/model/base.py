"""`SaltModelModule` — base class for model-side graph participants.

Every ``nn.Module`` that plans/executes as a graph node (embeds, encodes,
pools, computes a loss, or writes an ``outputs:`` leaf) subclasses this. It
owns the `GraphModule` contract (``name`` + `declare_io`) plus the optional
two-phase `bind`/`materialise`/`derived_widths` hooks and the `is_sink`
marker — each documented ONCE here (docstring policy rule 1); concrete
overrides state only what differs. Raw torch building blocks (`Dense`,
`Attention`, transformer internals, ...) are NOT graph participants — they
stay plain ``nn.Module``, owned internally by a `SaltModelModule`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from torch import Tensor, nn

from salt.graph.spec import _UNNAMED, IO, Mode

if TYPE_CHECKING:
    from salt.graph.bundle import Bundle
    from salt.model.bind import ResolvedSchema

__all__ = ["SaltModelModule"]


class SaltModelModule(nn.Module):
    """Base for model-side graph participants.

    Two-phase construction: config-only ``__init__`` -> `declare_io`
    (config-only, no data/tensors) -> `bind` (schema-driven, builds inner
    torch modules) -> optional `materialise` (fresh-fit-only file I/O) ->
    per-batch `forward`. Subclasses own their raw torch building blocks
    internally; a `SaltModelModule` is never itself a raw block.

    Not a Python ``ABC``: `declare_io` and `forward` are each documented as
    the contract every FORWARD-REACHABLE participant must implement, but
    neither is a hard `abstractmethod` — a manifest-only ``outputs:`` section
    writer (e.g. `InputCopyWriter`) legitimately implements no ``forward``
    (its serialisation surface is `copy_spec`/`columns` instead) because it
    never enters the executor's forward loop. Instantiation-time validation
    (`SaltModule.__init__`/`SaltDataModule.__init__`) checks
    ``isinstance(m, SaltModelModule)``, not method presence.
    """

    name: str
    """The instance name — the config dict key, assigned before compile."""

    def __init__(self) -> None:
        """Initialise the instance name placeholder (assigned from the config key)."""
        super().__init__()
        self.name: str = _UNNAMED

    def declare_io(self, mode: Mode) -> IO:
        """Return the declared requires/produces for the given mode.

        A function of the module's own config only — must not touch data
        files, the network, or tensors. NOT abstract: see the class
        docstring's manifest-only carve-out.

        Raises
        ------
        NotImplementedError
            If a concrete graph participant that DOES enter the plan never
            overrides `declare_io`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no declare_io() — either it is a manifest-only "
            "outputs: section writer (never entered into the plan) or it is missing "
            "a declare_io() override"
        )

    def bind(self, schema: ResolvedSchema) -> None:
        """Build inner torch modules from resolved widths; default no-op.

        Config-only — the schema is built from compiled plans, which are
        themselves config-derived, so this stays a pure function of config.
        The ONLY place a graph module may construct submodules sized from
        resolved bundle widths/fields. Called once per model, after every
        mode's plan compiles and before any forward.
        """

    def materialise(self) -> None:
        """Load file-touching values (e.g. normalisation constants); default no-op.

        Runs ONLY before a fresh fit, after `bind` — skipped on checkpoint
        load, where values arrive via the state_dict. The only hook
        permitted to touch the filesystem beyond config.
        """

    def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
        """Contribute produced-key widths derived from resolved input widths; default none.

        Some modules' output width is not a single shared symbol the
        bind-time dim table can unify (e.g. a concat's summed width); such a
        module contributes it here instead. Runs to a fixpoint alongside
        symbol back-binding (`salt.model.bind.resolve_bind_schema`) — a
        contribution that disagrees with an already-resolved width raises
        `BindError` there.

        Returns
        -------
        dict[str, int]
            ``{produced_key: width}``; empty by default.
        """
        del widths
        return {}

    def is_sink(self) -> bool:
        """Whether this module is a terminal sink; default False.

        Model-side graph participants are never sinks in practice — terminal
        sinks (`H5OutputSink`, `OnnxExportSink`) are Lightning `Callback`s
        satisfying the separate `SinkModule` Protocol
        (`salt.graph.spec`), not `SaltModelModule`. Kept here only as a
        defensive default so an `isinstance(m, SinkModule)` structural check
        never accidentally matches an unrelated model module.

        Returns
        -------
        bool
            Always ``False``.
        """
        return False

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Run this module's per-batch math over `b`; return the produced leaves.

        Reads only its declared ``declare_io(mode).requires`` off `b`,
        returns exactly its declared ``produces`` (flat or nested dotted
        dict — the executor canonicalises either spelling). NOT abstract: a
        manifest-only ``outputs:`` section writer (e.g. `InputCopyWriter`)
        never reaches the executor's per-batch forward loop, so it inherits
        this default undisturbed.

        Raises
        ------
        NotImplementedError
            If a concrete graph participant that DOES reach the forward loop
            never overrides `forward`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no forward() — either it is a manifest-only "
            "outputs: section writer (never called by the executor) or it is missing "
            "a forward() override"
        )
