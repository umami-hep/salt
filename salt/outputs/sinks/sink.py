"""The output-sink base classes: `Node` (declare-only) and `RuntimeSink`."""

from __future__ import annotations

import fnmatch
import functools
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import IO, PRIMARY_MODES, Mode, flatten_spec

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps this module import-light
    from salt.outputs.output_schema import OutputField

__all__ = [
    "Node",
    "OutputSink",
    "RuntimeSink",
    "SinkContext",
    "collect_manifest_fields",
    "is_test_persistence_sink",
    "parse_modes",
]

ALL_MODES: frozenset[Mode] = frozenset({Mode.FIT, Mode.VAL, Mode.TEST, Mode.ONNX})
"""The permissive `Node.allowed_modes` default; concrete sinks narrow it."""


def _mode_key(mode: Mode) -> str:
    """The config spelling of an ATOMIC `Mode` — its member name, lowercased.

    `Mode` is a `Flag`, so ``.name`` is None for a composite (``TRAINING``,
    ``ALL``); those have no config spelling and are refused rather than
    rendered as a misleading ``mode.training``.
    """
    if mode.name is None:
        raise ValueError(f"{mode!r} is a composite Mode and has no config spelling")
    return mode.name.lower()


_MODE_BY_NAME: dict[str, Mode] = {_mode_key(mode): mode for mode in PRIMARY_MODES}
"""The config vocabulary for ``modes:`` — the planner's own `Mode` names, lowercased."""

_MODE_BY_NAME["export"] = Mode.ONNX
"""``export`` is accepted for `Mode.ONNX`: it is what a section WRITER's own
``modes:`` list calls that mode (`salt.outputs.run_task_output.parse_output_modes`),
and writers and sinks share one section — one spelling per mode across the surface."""


def _fmt_modes(modes: frozenset[Mode]) -> str:
    """The canonical rendering of a mode set in an error message."""
    return "[" + ", ".join(_mode_key(m) for m in PRIMARY_MODES if m in modes) + "]"


def parse_modes(modes: Sequence[str | Mode], owner: str) -> frozenset[Mode]:
    """Parse a config ``modes:`` list into planner `Mode` members.

    The vocabulary is the planner's own — ``fit``/``val``/``test``/``onnx``,
    case-insensitive, plus ``export`` for ``onnx`` (the spelling a section
    WRITER's ``modes:`` list uses, and writers and sinks share one section).

    It deliberately has no entry for `salt inference`: that driver calls a
    sink's lifecycle directly, with no planner mode and no registration, so it
    is outside what ``modes:`` selects (see ``docs/outputs.md``).

    Parameters
    ----------
    modes : Sequence[str | Mode]
        The configured mode names. Must name at least one mode — to switch a
        sink off, delete its section entry (``<key>: null``) rather than
        declaring it with no modes.
    owner : str
        The declaring class name, for the error message.

    Returns
    -------
    frozenset[Mode]
        The parsed modes.

    Raises
    ------
    ConfigError
        On an empty list, or a name that is not a planner mode.
    """
    if not modes:
        raise ConfigError(
            f"{owner}: `modes:` is an empty list — name at least one of "
            f"{_fmt_modes(ALL_MODES)}, or delete the sink from the section entirely "
            "with `<key>: null`."
        )
    parsed: set[Mode] = set()
    for entry in modes:
        if isinstance(entry, Mode):
            parsed.add(entry)
            continue
        key = str(entry).strip().lower()
        if key not in _MODE_BY_NAME:
            raise ConfigError(
                f"{owner}: {entry!r} is not a mode — `modes:` takes the planner's own mode "
                f"names {_fmt_modes(ALL_MODES)}. (`salt inference` drives a sink directly, "
                "outside the planner's modes, and is not selectable here.)"
            )
        parsed.add(_MODE_BY_NAME[key])
    return frozenset(parsed)


def collect_manifest_fields(sources: Iterable[Any], mode: Mode) -> list[tuple[str, OutputField]]:
    """Collect the ``(leaf_key, OutputField)`` manifest a set of producers declares.

    The one place a sink learns WHAT to serialise. Any graph module minting
    ``outputs.*`` leaves names them by implementing ``manifest_fields(mode)``
    — the ``outputs:`` section's `RunTaskOutput`, a conversion producer, a
    reconstruction node. Sources are walked in order and their fields
    concatenated, so the caller's source order is the column/tuple order.

    A field marked ``final=False`` is an intermediate leaf a downstream node
    consumes; it is declared (so it is visible) and dropped here (so no sink
    serialises it).

    Parameters
    ----------
    sources : Iterable[Any]
        The candidate producers. An entry without a callable
        ``manifest_fields`` is skipped, so a source list may mix producers
        with plain graph modules.
    mode : Mode
        The mode to collect for.

    Returns
    -------
    list[tuple[str, OutputField]]
        The concatenated final fields, each tagged with its ``outputs.*`` leaf key.
    """
    out: list[tuple[str, OutputField]] = []
    for source in sources:
        manifest = getattr(source, "manifest_fields", None)
        if not callable(manifest):
            continue
        out.extend((key, field) for key, field in manifest(mode) if field.final)
    return out


@dataclass(frozen=True)
class SinkContext:
    """The run facts a sink needs when it opens its output.

    A sink is driven by whichever command is running — the Lightning test
    loop, or `salt inference`, which runs no Lightning at all. `SinkContext`
    is the driver-independent handle that carries the few facts a sink
    actually reads at open time, so a sink never touches a `Trainer` and
    every driver can supply the context honestly.

    Parameters
    ----------
    run_name : str
        The model's run name, used to prefix output column names.
    datamodule : Any
        The datamodule whose ``test_dset`` carries the reader — the source
        schema, sequence lengths for pad-back, and the source filename the
        output is named after.
    ckpt_path : str | None
        The checkpoint being evaluated. Output paths are templated on it, so
        a sink that names files rejects None.
    num_test_batches : Any, optional
        Lightning's per-dataloader batch counts, used to size a
        ``limit_test_batches``-capped run. None (the default, and what a
        non-Lightning driver supplies) means "the whole dataset".
    world_size : int, optional
        Devices taking part, by default 1. Multi-device test writing is out
        of scope, so anything else is refused at wiring time.
    """

    run_name: str
    datamodule: Any
    ckpt_path: str | None
    num_test_batches: Any = None
    world_size: int = 1

    @classmethod
    def from_trainer(cls, trainer: Any) -> SinkContext:
        """Build the context from a live Lightning `Trainer`.

        Duck-typed on purpose: this reads only the handful of attributes
        above, which is what lets the same sink run under a driver that has
        no trainer at all — and what keeps this module free of any Lightning
        import.
        """
        module = getattr(trainer, "lightning_module", None)
        ckpt_path = getattr(trainer, "ckpt_path", None)
        return cls(
            run_name=getattr(module, "name", None) or "salt",
            datamodule=getattr(trainer, "datamodule", None),
            ckpt_path=None if ckpt_path is None else str(ckpt_path),
            num_test_batches=getattr(trainer, "num_test_batches", None),
            world_size=int(getattr(trainer, "world_size", 1) or 1),
        )

    @property
    def reader(self) -> Any:
        """The test dataset's reader, or None when no dataset is built."""
        return getattr(getattr(self.datamodule, "test_dset", None), "reader", None)


def is_test_persistence_sink(sink: Any) -> bool:
    """Whether a ``writer_demand``-exposing sink is THE TEST persistence sink.

    The one selector both the runtime (`SaltModule._attached_writer`) and the
    static graph tooling (`salt.cli`) use, so a config resolves the same sink
    either way. True for a primary sink like `H5OutputSink`; False for an
    ONNX-only sink (`OnnxExportSink`, empty TEST requires) and for an
    auxiliary sink that opts out (`JSONLOutputSink`). A duck-typed object
    without ``is_test_sink`` counts as one.

    Parameters
    ----------
    sink : Any
        A sink node, typically one exposing ``writer_demand``.

    Returns
    -------
    bool
        Whether `sink` should anchor the TEST boundary demand.
    """
    is_test_sink = getattr(sink, "is_test_sink", None)
    return True if not callable(is_test_sink) else bool(is_test_sink())


class Node:
    """Base class for a terminal output node: DECLARE-ONLY, no lifecycle.

    A node is the last thing in a mode's graph. It declares which
    ``outputs.*`` leaves it consumes and nothing else — that single
    declaration drives both the planner (demand-gating keeps exactly the
    required producers alive) and `writer_demand`, so the two can never
    disagree.

    Subclass `Node` directly when the node has no run-time work at all —
    `OnnxExportSink` is the case in the tree: export never executes a test
    loop, so naming the leaves at compile time is its whole job. Subclass
    `RuntimeSink` when the node consumes batches as they are produced.

    A node is NOT a Lightning callback. The Lightning test loop drives a
    `RuntimeSink` through a generated adapter (`salt.callbacks.SinkAdapter`),
    and `salt inference` drives the same methods directly — so a sink author
    never writes a Lightning hook and no driver is privileged.

    What a subclass provides:

    ``name``
        The graph-node instance name, unique across the graph.
    ``allowed_modes``
        The modes this class may be configured to run in. A config `modes:`
        list must be a subset; the default is permissive and concrete
        classes narrow it.
    ``declare_io(mode)``
        The node's requires/produces for `mode`. A sink requires the
        ``outputs.*`` leaves it serialises (plus anything else it needs, e.g.
        ``meta.rows``) and produces nothing. Whatever a subclass returns here
        is gated by `effective_modes` first: outside them the node declares
        empty IO, so the planner prunes it and every producer kept alive only
        for it. Overriding `declare_io` is all a subclass does — the gate is
        applied to the override automatically, so `modes:` works for a
        third-party sink exactly as it does for the shipped ones.

    What a node consumes is resolved from two MANIFEST SOURCES, bound before
    any ``declare_io`` resolution: the ``outputs:`` section
    (`bind_output_section`) and the model's graph modules
    (`bind_model_modules`). Every entry exposing ``manifest_fields(mode)``
    contributes its ``outputs.*`` leaves — see `collect_manifest_fields`.
    ``modes:`` picks WHEN a node runs; ``consumes:`` picks WHAT it takes out
    of that manifest. Narrowing composes with demand-gating rather than
    replacing it: a leaf no sink consumes is still the existing dead-prediction
    hard error.

    Two predicates control how the graph machinery treats the node:

    - `is_sink` (True here, rarely overridden) marks the node terminal, so
      the executor excludes it from the per-batch forward loop and the
      planner renders it as a sink card.
    - `is_test_sink` marks the node as *the* TEST persistence sink. Exactly
      one registered sink holds that role: it anchors the TEST boundary
      demand (`SaltModule._attached_writer` picks the first one). The base
      implementation answers True whenever ``declare_io(Mode.TEST).requires``
      is non-empty, which is right for a primary sink and for an ONNX-only
      node (empty TEST requires -> False). An AUXILIARY sink that rides
      alongside the primary H5 sink must override it to return False — see
      `JSONLOutputSink`.

    Attributes
    ----------
    name : str
        The graph-node instance name, unique across the graph.
    allowed_modes : ClassVar[frozenset[Mode]]
        The modes this class may be configured to run in.
    """

    name: str
    """The graph-node instance name, unique across the graph."""

    allowed_modes: ClassVar[frozenset[Mode]] = ALL_MODES
    """The modes a config may select for this class (concrete classes narrow it)."""

    _configured_modes: frozenset[Mode] | None = None
    """The config's ``modes:`` selection; None means "the class default"."""

    _consumes: tuple[str, ...] | None = None
    """The config's ``consumes:`` patterns; None means "every declared leaf"."""

    _output_section: Mapping[str, Any] | None = None
    """The bound ``outputs:`` section — the FIRST manifest source."""

    _manifest_modules: Mapping[str, Any] | None = None
    """The bound model graph modules — the SECOND manifest source."""

    def __init__(
        self,
        modes: Sequence[str | Mode] | None = None,
        consumes: Sequence[str] | None = None,
    ) -> None:
        """Select which planner modes this node runs in and what it takes.

        Parameters
        ----------
        modes : Sequence[str | Mode] | None, optional
            The modes to run in, from ``fit``/``val``/``test``/``onnx``.
            Omitted (the default) means `allowed_modes` — for every shipped
            sink that is exactly the set its `declare_io` already gates on,
            so an omitted list changes nothing.
        consumes : Sequence[str] | None, optional
            fnmatch patterns over the dotted ``outputs.*`` leaf key (e.g.
            ``outputs.jets.*``, ``outputs.tracks.HadronIndex``) narrowing what
            this node takes from the collected manifest. Omitted (the default)
            means everything the manifest declares. A pattern matching no
            available leaf is a `ConfigError`, so a typo fails loudly.

        Raises
        ------
        ConfigError
            When a requested mode is outside `allowed_modes`, or ``consumes:``
            is an empty list.
        """
        if consumes is not None:
            if not list(consumes):
                raise ConfigError(
                    f"{type(self).__name__}: `consumes:` is an empty list — name at least one "
                    "leaf-key pattern, or omit it entirely to consume every declared leaf "
                    "(to switch the sink off, delete it from the section with `<key>: null`)."
                )
            self._consumes = tuple(str(p) for p in consumes)
        if modes is None:
            return
        selected = parse_modes(modes, type(self).__name__)
        if extra := selected - self.allowed_modes:
            raise ConfigError(
                f"{type(self).__name__}: modes {_fmt_modes(extra)} are not allowed for this "
                f"sink — requested modes={_fmt_modes(selected)}, "
                f"allowed_modes={_fmt_modes(self.allowed_modes)}"
            )
        self._configured_modes = selected

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Gate a subclass's own `declare_io` on `effective_modes`.

        Applied here rather than at wiring time so the gate reaches EVERY
        sink, including a third-party one the wiring never special-cases —
        a `modes:` list that some sinks ignored would be a config surface
        that lies. Discovery is untouched: `salt.outputs.iter_sinks` stays
        the single place a sink is found.
        """
        super().__init_subclass__(**kwargs)
        declared = cls.__dict__.get("declare_io")
        if declared is None or getattr(declared, "_salt_mode_gated", False):
            return

        @functools.wraps(declared)
        def _mode_gated(self: Node, mode: Mode, _inner: Any = declared) -> IO:
            if not any(mode & selected for selected in self.effective_modes):
                return IO(requires={}, produces={})
            return _inner(self, mode)

        _mode_gated._salt_mode_gated = True  # type: ignore[attr-defined]  # noqa: SLF001
        cls.declare_io = _mode_gated  # type: ignore[method-assign]

    @property
    def modes_configured(self) -> bool:
        """Whether the config selected `modes:` explicitly (vs taking the class default)."""
        return self._configured_modes is not None

    @property
    def effective_modes(self) -> frozenset[Mode]:
        """The modes this node actually runs in: `allowed_modes` narrowed by `modes:`."""
        return self.allowed_modes if self._configured_modes is None else self._configured_modes

    # -- manifest sources ---------------------------------------------------

    def bind_model_modules(self, model_modules: Mapping[str, Any]) -> None:
        """Capture the model's graph modules — the SECOND manifest source.

        The counterpart of `bind_output_section`: a producer minting
        ``outputs.*`` leaves from the MODEL graph (a conversion node, a
        reconstruction node) names them through its own
        ``manifest_fields(mode)``, which this dict is how the node reaches.
        Bound before any ``declare_io`` / ``writer_demand`` resolution.
        """
        self._manifest_modules = model_modules
        self._invalidate_manifest()

    def _invalidate_manifest(self) -> None:
        """Drop any cached manifest resolution (subclass hook, no-op here)."""

    def _manifest_source_groups(self) -> list[list[Any]]:
        """The bound producers as one list per BINDING, section group then model group.

        The group boundary is what keeps a sink's serialisation order stable: a
        consumer that orders leaves within a group (`OnnxExportSink` sorts
        globals ahead of per-token) must not let the model group's leaves
        migrate into the section group's block, so the two are never flattened
        before ordering. Within a group, declaration order. A section writer
        folded into the model graph appears in both dicts and is taken once, at
        its section position; this node itself is never a source.
        """
        groups: list[list[Any]] = []
        seen: set[int] = {id(self)}
        for bound in (self._output_section, self._manifest_modules):
            group: list[Any] = []
            for entry in (bound or {}).values():
                if id(entry) in seen:
                    continue
                seen.add(id(entry))
                group.append(entry)
            groups.append(group)
        return groups

    def _manifest_sources(self) -> list[Any]:
        """Every bound producer, section entries first then model modules, deduplicated."""
        return [entry for group in self._manifest_source_groups() for entry in group]

    def _manifest_source_names(self) -> list[str]:
        """The instance names of the bound manifest sources, for error messages."""
        return [*(self._output_section or {}), *(self._manifest_modules or {})]

    def _filter_consumed(
        self, fields: Sequence[tuple[str, OutputField]]
    ) -> list[tuple[str, OutputField]]:
        """Narrow a collected manifest to the ``consumes:`` patterns.

        Every pattern is validated (so a typo in ANY of them is loud) before
        the filter runs; a field survives when at least one pattern matches
        its leaf key. No ``consumes:`` -> the fields unchanged.

        Raises
        ------
        ConfigError
            When a pattern matches none of the available leaf keys.
        """
        patterns = self._consumes
        if patterns is None:
            return list(fields)
        available = sorted({key for key, _ in fields})
        for pattern in patterns:
            if not any(fnmatch.fnmatchcase(key, pattern) for key in available):
                raise ConfigError(
                    f"{type(self).__name__}: `consumes:` pattern {pattern!r} matches none of "
                    f"the declared output leaves — available leaf keys are {available}"
                )
        return [
            (key, field)
            for key, field in fields
            if any(fnmatch.fnmatchcase(key, pattern) for pattern in patterns)
        ]

    def is_sink(self) -> bool:
        """Mark this module a terminal sink, excluded from the executor forward loop."""
        return True

    def is_test_sink(self) -> bool:
        """Whether this node is the TEST persistence sink.

        Discriminator among ``writer_demand``-exposing sinks: True when
        ``declare_io(Mode.TEST).requires`` is non-empty (e.g. `H5OutputSink`);
        an ONNX-only node (`OnnxExportSink`) returns False regardless of
        declaration order.
        """
        return bool(flatten_spec(self.declare_io(Mode.TEST).requires))

    def declare_io(self, mode: Mode) -> IO:  # pragma: no cover - overridden by subclasses
        """Declare the node's requires/produces for `mode` (subclass override)."""
        del mode
        return IO(requires={}, produces={})

    def writer_demand(
        self, model_modules: Mapping[str, Any], reader: Any
    ) -> dict[str, str]:  # pragma: no cover - overridden
        """The TEST demand this node anchors, GENERATED from `declare_io` (subclass override)."""
        del model_modules, reader
        return {}


class RuntimeSink(Node):
    """A terminal node that also consumes batches: the lifecycle lives here.

    Adds the four driver-called methods to `Node`. `H5OutputSink` writes the
    eval H5, and `JSONLOutputSink` (in ``salt/outputs/sinks/jsonl_sink.py``) is the
    worked example of a third format. Subclass this to add a format of your
    own; see ``docs/outputs.md``.

    The lifecycle is driver-independent — the Lightning test loop reaches it
    through a generated `salt.callbacks.SinkAdapter`, `salt inference` calls
    it directly:

    ``open_schema(ctx)``
        Called once before the first batch with a `SinkContext` — open the
        file, write the header, resolve the column schema. The context, not
        a `Trainer`, is what carries the run facts, so the same sink works
        under any driver.
    ``consume(bundle)``
        Called once per batch with the executed `Bundle`. Read each required
        leaf with ``bundle.get(key)`` and append it.
    ``flush()``
        Called after the last batch — close the handle, report.
    ``close_if_open()``
        Called on any exit path, including an exception mid-run. Must be
        idempotent: it is what guarantees a crashed test still closes the
        file.

    Notes
    -----
    Multi-device TEST is out of scope: the adapter raises `ConfigError` when
    ``world_size != 1``, so every sink writes from a single process and no
    rank-zero guard is needed in `consume` / `flush`.

    Examples
    --------
    A minimal sink counting the rows it saw::

        from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
        from salt.logging import console
        from salt.outputs import RuntimeSink


        class RowCountSink(RuntimeSink):
            name = "row_count"

            def declare_io(self, mode: Mode) -> IO:
                if not (mode & Mode.TEST):
                    return IO(requires={}, produces={})
                req = {"meta.rows": TensorSpec(shape=None, dtype="int64", kind="meta")}
                return IO(requires=unflatten_spec(req), produces={})

            def is_test_sink(self) -> bool:
                return False  # auxiliary: the H5 sink stays the demand anchor

            def open_schema(self, ctx):
                self.rows = 0

            def consume(self, bundle):
                start, stop = bundle.get("meta.rows")
                self.rows += int(stop) - int(start)

            def flush(self):
                console(f"saw {self.rows} rows")
    """

    allowed_modes: ClassVar[frozenset[Mode]] = frozenset({Mode.TEST})
    """A runtime sink runs in the TEST loop; export/fit never drive one."""

    def open_schema(self, ctx: SinkContext) -> None:  # pragma: no cover - overridden
        """Open the sink's output schema before the first batch."""

    def consume(self, bundle: Bundle) -> None:  # pragma: no cover - overridden
        """Consume one executed bundle."""

    def flush(self) -> None:  # pragma: no cover - overridden
        """Finalise the sink."""

    def close_if_open(self) -> None:  # pragma: no cover - overridden
        """Idempotently close any open handle (failure-cleanup)."""


class OutputSink(RuntimeSink):
    """Deprecated alias of `RuntimeSink`, kept for third-party subclasses.

    The sink base used to BE a `lightning.Callback`; it was split into
    `Node` (declare-only) and `RuntimeSink` (lifecycle) so a sink stops
    privileging one driver. An existing ``class MySink(OutputSink)`` keeps
    working unchanged — the lifecycle methods and their contracts are
    identical — but subclassing warns, and the name is removed after the
    deprecation window. Subclass `RuntimeSink` instead.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Warn once per subclass that `OutputSink` is deprecated."""
        super().__init_subclass__(**kwargs)
        warnings.warn(
            f"{cls.__name__} subclasses OutputSink, which is a deprecated alias of "
            "RuntimeSink — subclass salt.outputs.RuntimeSink instead (a sink is no "
            "longer a lightning Callback; the test loop drives it through a generated "
            "adapter).",
            DeprecationWarning,
            stacklevel=2,
        )
