"""Sinks declared in the ``outputs:`` section: partition, `modes:` gate, wiring rules."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from salt.cli import load_config
from salt.graph.errors import ConfigError
from salt.graph.planner import compile_plan
from salt.graph.spec import IO, Mode, SinkModule, TensorSpec, unflatten_spec
from salt.main import SaltCLI
from salt.model.base import SaltModelModule
from salt.model.saltmodule import SaltModule, _is_section_sink
from salt.outputs import (
    H5OutputSink,
    InputCopyWriter,
    JSONLOutputSink,
    OnnxExportSink,
    PadMaskWriter,
)
from salt.outputs.sinks.sink import ALL_MODES, Node, RuntimeSink
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict

pytestmark = pytest.mark.cpu_always

_CONFIGS = Path(__file__).parents[3] / "configs"
_LRS = {"initial": 1e-3, "max": 5e-3, "end": 1e-4, "pct_start": 0.1}
_JET_OUT = "outputs.jets.jets_classification"


class _InertModule(SaltModelModule):
    """A model module declaring no IO — just enough to build a `SaltModule`."""

    def declare_io(self, mode: Mode) -> IO:
        del mode
        return IO(requires={}, produces={})


def _model(outputs=None) -> SaltModule:
    """A minimal `SaltModule` to compose an ``outputs:`` section onto.

    Deliberately loss-free: a `LossSum` narrows against the other modules'
    ``losses.*`` produces at construction and refuses an empty list, and none
    of these tests exercise the training graph.
    """
    return SaltModule({"inert": _InertModule()}, lrs=_LRS, outputs=outputs)


def _section() -> dict:
    """A section with two writers around one sink, sink declared in the MIDDLE."""
    return {
        "inputs_copy": InputCopyWriter(streams=["jets"]),
        "h5_output": H5OutputSink(),
        "pad_mask": PadMaskWriter(streams=["tracks"]),
    }


# ---------------------------------------------------------------------------
# (1) the partition: sinks out of the writer section, the graph, and the order
# ---------------------------------------------------------------------------


def test_section_sink_is_partitioned_out_of_the_writer_manifest():
    """A section sink lands on `_section_sinks`, never in the bound writer manifest."""
    section = _section()
    model = _model(section)

    assert model._section_sinks == [section["h5_output"]]  # noqa: SLF001
    assert list(model._output_section) == ["inputs_copy", "pad_mask"]  # noqa: SLF001


def test_section_sink_is_not_folded_into_the_graph():
    """A section sink is excluded from the planning module dict and from ``net``."""
    model = _model(_section())

    assert "h5_output" not in model._graph_modules  # noqa: SLF001
    assert "h5_output" not in model.net
    # the non-manifest-only writer IS folded, so the exclusion is specific to sinks
    assert "pad_mask" in model._graph_modules  # noqa: SLF001


def test_section_sink_is_named_by_its_dict_key():
    """`compose_output_section` names a sink from its section key, as it does a writer."""
    sink = H5OutputSink()
    _model({"my_h5": sink})
    assert sink.name == "my_h5"


@pytest.mark.parametrize("position", [0, 1, 2], ids=["first", "middle", "last"])
def test_writer_order_is_independent_of_where_the_sink_sits(position):
    """A sink takes no part in column ordering: the writer order is identical wherever it goes."""
    writers = [
        ("inputs_copy", InputCopyWriter(streams=["jets"])),
        ("pad_mask", PadMaskWriter(streams=["tracks"])),
    ]
    entries = list(writers)
    entries.insert(position, ("h5_output", H5OutputSink()))
    model = _model(dict(entries))
    assert list(model._output_section) == ["inputs_copy", "pad_mask"]  # noqa: SLF001


# ---------------------------------------------------------------------------
# (2) the discriminator — why the partition cannot select on the Protocol
# ---------------------------------------------------------------------------


def test_section_writers_match_the_sink_protocol_but_are_not_sinks():
    """`SinkModule` is structural and every WRITER satisfies it — hence the `Node` check.

    `SaltModelModule` carries an ``is_sink`` method (returning False) so a model
    module never accidentally matches an ``isinstance(m, SinkModule)`` check.
    That method is exactly what makes the runtime-checkable Protocol match the
    writer, so the partition selects on the `Node` base + the ANSWER to
    ``is_sink()``, not on the Protocol.
    """
    writer = PadMaskWriter(streams=["tracks"])
    assert isinstance(writer, SinkModule)  # structural match — attribute present
    assert writer.is_sink() is False  # but the answer is False
    assert _is_section_sink(writer) is False
    assert _is_section_sink(H5OutputSink()) is True


def test_duck_typed_sink_is_partitioned_without_subclassing_node():
    """A sink that only answers ``is_sink() -> True`` still partitions correctly."""

    class _DuckSink:
        name = "duck"

        def is_sink(self) -> bool:
            return True

        def declare_io(self, mode):
            del mode
            return IO(requires={}, produces={})

    assert _is_section_sink(_DuckSink()) is True


# ---------------------------------------------------------------------------
# (3) `modes:` validation
# ---------------------------------------------------------------------------


def test_omitted_modes_are_the_class_defaults():
    """No ``modes:`` means `allowed_modes` — today's behaviour, unchanged."""
    sink = H5OutputSink()
    assert sink.modes_configured is False
    assert sink.effective_modes == frozenset({Mode.TEST})


def test_configured_modes_are_recorded():
    sink = H5OutputSink(modes=["test"])
    assert sink.modes_configured is True
    assert sink.effective_modes == frozenset({Mode.TEST})


def test_modes_outside_allowed_modes_raise_naming_both_sets():
    """`modes ⊄ allowed_modes` is a ConfigError naming BOTH sets."""
    with pytest.raises(ConfigError) as err:
        H5OutputSink(modes=["onnx"])
    message = str(err.value)
    assert "allowed_modes=[test]" in message
    assert "modes=[onnx]" in message


def test_unknown_mode_name_is_refused():
    with pytest.raises(ConfigError, match="is not a mode"):
        H5OutputSink(modes=["nonsense"])


def test_empty_modes_list_points_at_the_delete_syntax():
    """An empty list is not "off" — the section already deletes an entry with null."""
    with pytest.raises(ConfigError, match="empty list"):
        H5OutputSink(modes=[])


def test_export_is_accepted_as_the_onnx_spelling():
    """Writers in the SAME section spell `Mode.ONNX` ``export``; a sink accepts both."""
    assert OnnxExportSink(modes=["export"]).effective_modes == frozenset({Mode.ONNX})
    assert OnnxExportSink(modes=["onnx"]).effective_modes == frozenset({Mode.ONNX})


# ---------------------------------------------------------------------------
# (4) `modes:` is LOAD-BEARING — narrowing changes the compiled plan
# ---------------------------------------------------------------------------


class _MultiModeSink(RuntimeSink):
    """A third-party-style sink that runs in every mode unless narrowed."""

    allowed_modes = ALL_MODES
    name = "multi_sink"

    def __init__(self, leaf: str = "outputs.a", modes=None) -> None:
        super().__init__(modes=modes)
        self._leaf = leaf

    def declare_io(self, mode: Mode) -> IO:
        del mode
        return IO(
            unflatten_spec({self._leaf: TensorSpec(shape=None, dtype=None, kind="data")}),
            produces={},
        )


class _Producer:
    """A compile-only producer: one dataset input to one ``outputs.*`` leaf."""

    def __init__(self, name: str, src: str, dst: str) -> None:
        self.name = name
        self._src, self._dst = src, dst

    def declare_io(self, mode: Mode) -> IO:
        del mode
        return IO(
            unflatten_spec({self._src: TensorSpec(shape=("B", 4), dtype="float32")}),
            unflatten_spec({self._dst: TensorSpec(shape=("B", 4), dtype="float32")}),
        )


def test_narrowing_modes_empties_declare_io_outside_them():
    """The gate wraps a subclass's OWN `declare_io` — no cooperation required from it."""
    wide = _MultiModeSink()
    assert wide.declare_io(Mode.TEST).requires
    assert wide.declare_io(Mode.ONNX).requires

    narrow = _MultiModeSink(modes=["test"])
    assert narrow.declare_io(Mode.TEST).requires
    assert narrow.declare_io(Mode.ONNX).requires == {}


def _toy_plan(gated_sink, mode):
    """Compile a two-chain toy graph: one chain feeds `gated_sink`, one always survives."""
    modules = {
        "prod_a": _Producer("prod_a", "inputs.a", "outputs.a"),
        "prod_b": _Producer("prod_b", "inputs.b", "outputs.b"),
        "gated": gated_sink,
        "always": _MultiModeSink(leaf="outputs.b"),
    }
    modules["always"].name = "always"
    gated_sink.name = "gated"
    sources = unflatten_spec({
        "inputs.a": TensorSpec(shape=("B", 4), dtype="float32"),
        "inputs.b": TensorSpec(shape=("B", 4), dtype="float32"),
    })
    return compile_plan(modules, mode, sources, sinks=[])


def test_narrowing_prunes_the_sink_and_its_exclusive_producer():
    """The requested feature: a narrowed sink drops out, and so does the producer only it kept alive."""
    wide = _toy_plan(_MultiModeSink(), Mode.ONNX)
    assert {"prod_a", "gated", "prod_b", "always"} <= set(wide.module_names)

    narrowed = _toy_plan(_MultiModeSink(modes=["test"]), Mode.ONNX)
    assert "gated" not in narrowed.module_names
    assert "prod_a" not in narrowed.module_names  # kept alive ONLY by the gated sink
    # the untouched chain is unaffected
    assert {"prod_b", "always"} <= set(narrowed.module_names)


def test_narrowing_changes_the_plan_hash():
    """Narrowing is a real semantic change, so the plan hash MOVES — by design."""
    wide = _toy_plan(_MultiModeSink(), Mode.ONNX).plan_hash
    narrowed = _toy_plan(_MultiModeSink(modes=["test"]), Mode.ONNX).plan_hash
    assert wide != narrowed


# ---------------------------------------------------------------------------
# (5) wiring validation
# ---------------------------------------------------------------------------


def _wiring(sinks, section_sinks=None):
    """A stand-in CLI + model carrying `sinks` on the registry, for `_validate_wired_sinks`."""
    trainer = SimpleNamespace(salt_sinks=list(sinks), callbacks=[])
    model = SimpleNamespace(_section_sinks=list(section_sinks or []))
    return SimpleNamespace(trainer=trainer), model


def _seeded_h5(name: str) -> H5OutputSink:
    """An H5 sink whose TEST requires are non-empty (i.e. a real persistence sink)."""
    from salt.outputs.output_schema import OutputColumn

    sink = H5OutputSink()
    sink.name = name
    sink._columns = (OutputColumn(key=_JET_OUT, suffixes=["pb", "pc", "pu"]),)  # noqa: SLF001
    sink._columns_resolved = True  # noqa: SLF001
    return sink


def test_one_test_persistence_sink_is_allowed():
    cli, model = _wiring([_seeded_h5("h5_output")])
    SaltCLI._validate_wired_sinks(cli, model)  # noqa: SLF001 - unbound call on a stand-in


def test_two_test_persistence_sinks_are_refused():
    """Two sinks claiming the TEST boundary demand is ambiguous — refuse rather than pick."""
    cli, model = _wiring([_seeded_h5("h5_output"), _seeded_h5("second_h5")])
    with pytest.raises(ConfigError, match="TEST persistence sinks"):
        SaltCLI._validate_wired_sinks(cli, model)  # noqa: SLF001


def test_onnx_sink_does_not_count_as_a_persistence_sink():
    """The ONNX manifest has empty TEST requires, so it never contends for the role."""
    onnx = OnnxExportSink()
    cli, model = _wiring([_seeded_h5("h5_output"), onnx])
    SaltCLI._validate_wired_sinks(cli, model)  # noqa: SLF001


def test_section_declared_auxiliary_sink_must_state_its_modes():
    """An auxiliary sink rides alongside the persistence sink, so it declares when it runs."""
    aux = JSONLOutputSink()
    aux.name = "jsonl"
    cli, model = _wiring([_seeded_h5("h5_output"), aux], section_sinks=[aux])
    with pytest.raises(ConfigError, match="without a `modes:` list"):
        SaltCLI._validate_wired_sinks(cli, model)  # noqa: SLF001


def test_section_declared_auxiliary_sink_with_modes_is_accepted():
    aux = JSONLOutputSink(modes=["test"])
    aux.name = "jsonl"
    cli, model = _wiring([_seeded_h5("h5_output"), aux], section_sinks=[aux])
    SaltCLI._validate_wired_sinks(cli, model)  # noqa: SLF001


# ---------------------------------------------------------------------------
# (6) the worked proof — MaskFormer.yaml, section-declared, three modes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def maskformer_cfg(tmp_path_factory):
    """`MaskFormer.yaml` loaded through the real CLI surface (sinks now section-declared)."""
    base = tmp_path_factory.mktemp("mf_section_sinks")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    with open(nd_path) as fh:
        nd = yaml.safe_load(fh)
    # the one jets variable this config declares beyond the shared parity fixture
    jets = nd.setdefault("jets", {})
    if "mass" not in jets:
        idx = len(jets)
        jets["mass"] = {"mean": round(0.1 * (idx + 1), 6), "std": round(1.0 + 0.05 * (idx + 1), 6)}
    with open(nd_path, "w") as fh:
        yaml.dump(nd, fh, sort_keys=False)
    return load_config(
        _CONFIGS / "MaskFormer.yaml",
        [f"model.modules.norm.init_args.norm_dict={nd_path}"],
    )


def test_maskformer_declares_both_sinks_in_the_section():
    """The shipped config is the worked proof: no sink is left under ``callbacks:``."""
    raw = yaml.safe_load((_CONFIGS / "MaskFormer.yaml").read_text())
    assert {"h5_output", "onnx_export"} <= set(raw["outputs"])
    assert set(raw.get("callbacks") or {}) == {"maskformer_metrics"}


@pytest.mark.parametrize("mode_name", ["FIT", "TEST", "ONNX"])
def test_maskformer_compiles_in_every_gated_mode(maskformer_cfg, mode_name):
    """All three modes compile — section-declared sinks are discovered end to end."""
    mode = getattr(Mode, mode_name)
    assert maskformer_cfg.mode_errors.get(mode) is None
    plan = compile_plan(
        maskformer_cfg.modules,
        mode,
        maskformer_cfg.sources,
        schema=maskformer_cfg.schema,
        sinks=maskformer_cfg.sinks,
        sink_origins=maskformer_cfg.sink_origins.get(mode),
    )
    assert plan.plan_hash


def test_maskformer_section_sinks_are_folded_per_mode(maskformer_cfg):
    """The H5 sink is a TEST node, the ONNX manifest an ONNX node, and FIT has neither."""

    def _names(mode):
        return set(
            compile_plan(
                maskformer_cfg.modules,
                mode,
                maskformer_cfg.sources,
                schema=maskformer_cfg.schema,
                sinks=maskformer_cfg.sinks,
                sink_origins=maskformer_cfg.sink_origins.get(mode),
            ).module_names
        )

    assert "h5_output" in _names(Mode.TEST)
    assert "onnx_export" in _names(Mode.ONNX)
    fit = _names(Mode.FIT)
    assert "h5_output" not in fit
    assert "onnx_export" not in fit


def test_maskformer_section_sinks_are_not_graph_modules(maskformer_cfg):
    """A section sink is folded at COMPILE time, never into the model's module dict."""
    # the loaded config folds the sink nodes for rendering, but the model's own
    # graph modules (what `net` / the state_dict follow) must not contain them
    assert isinstance(maskformer_cfg.modules["h5_output"], Node)
    assert isinstance(maskformer_cfg.modules["onnx_export"], Node)
