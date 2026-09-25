"""Stub-based mirror tests for `salt.model.sink_prep` (no Trainer, no data file)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from salt.graph.errors import ConfigError
from salt.graph.planner import compile_plan
from salt.graph.spec import IO, PRIMARY_MODES, Mode, TensorSpec, unflatten_spec
from salt.model.sink_prep import (
    ModeSinks,
    PreparedSinks,
    assert_no_dead_preds,
    bind_manifests,
    boundary_demand,
    callback_demand,
    dead_preds_message,
    fold_sink_node,
    is_section_sink,
    model_sinks,
    prepare_sinks,
    select_fitval_callbacks,
    select_onnx_sink,
    select_test_sink,
)
from salt.outputs import (
    H5OutputSink,
    OnnxExportSink,
    OutputColumn,
    OutputField,
    PadMaskWriter,
    TaskOutput,
)
from salt.outputs.sinks.onnx.config import sanitised_model_name

# stub helpers


class _Stub:
    """A minimal `GraphModule` declaring the given requires/produces (compile-only)."""

    def __init__(self, name, requires=None, produces=None):
        self.name = name
        self._req = requires or {}
        self._prod = produces or {}

    def declare_io(self, mode):
        del mode
        return IO(unflatten_spec(self._req), unflatten_spec(self._prod))


class _ColumnProducer:
    """A minimal ``outputs:`` section producer declaring TEST-mode H5 fields."""

    def __init__(self, columns: list[OutputColumn]):
        self._fields = [(col.key, OutputField(h5_name=s)) for col in columns for s in col.suffixes]

    def manifest_fields(self, mode):
        return list(self._fields) if mode & Mode.TEST else []


def _seed_columns(sink: H5OutputSink, columns: list[OutputColumn]) -> H5OutputSink:
    """Bind an ``outputs:`` section whose producer mints `columns` (survives manifest rebinds)."""
    sink.bind_output_section({"run_tasks": _ColumnProducer(columns)})
    return sink


def _h5_sink() -> H5OutputSink:
    """A real H5 persistence sink, seeded with one resolved column."""
    sink = _seed_columns(
        H5OutputSink(copy_inputs={"jets": [], "tracks": []}, write_pad_mask=["tracks"]),
        [OutputColumn(key="outputs.jets.jets_classification", suffixes=["pb", "pc", "pu"])],
    )
    if sink.name != "h5_output":
        sink.name = "h5_output"
    return sink


def _loss_stub() -> _Stub:
    return _Stub("loss", produces={"loss.total": TensorSpec(shape=(), dtype="float32")})


def _preds_stub(name: str = "task") -> _Stub:
    return _Stub(name, produces={"preds.jets.x": TensorSpec(shape=("B", 3), dtype="float32")})


def _pooled_stub() -> _Stub:
    return _Stub("pooled", produces={"pooled.global": TensorSpec(shape=("B", 8), dtype="float32")})


def _requiring_stub() -> _Stub:
    """Requires ``labels.jets.x`` (dataset-only) + ``pooled.global`` (sibling-produced)."""
    return _Stub(
        "consumer",
        requires={
            "labels.jets.x": TensorSpec(shape=("B",), dtype="int64"),
            "pooled.global": TensorSpec(shape=("B", 8), dtype="float32"),
        },
    )


def _mode_sinks_map() -> dict[Mode, ModeSinks]:
    return {m: ModeSinks(anchors=[], anchor_origins=None) for m in (Mode.FIT, Mode.VAL, Mode.TEST)}


class _RaisingCallback:
    """A FIT/VAL callback whose ``fit_val_demand`` always raises."""

    def fit_val_demand(self, model_modules):
        raise ConfigError("boom")


# is_section_sink


def test_is_section_sink_true_for_node_instance():
    """A `Node` instance (e.g. `H5OutputSink`) is a section sink."""
    assert is_section_sink(_h5_sink()) is True


def test_is_section_sink_true_for_duck_typed_is_sink():
    """A duck object whose ``is_sink()`` is truthy is a section sink."""
    duck = SimpleNamespace(is_sink=lambda: True)
    assert is_section_sink(duck) is True


def test_is_section_sink_false_for_pad_mask_writer():
    """A `PadMaskWriter` — a real section WRITER, not a sink — is False."""
    assert is_section_sink(PadMaskWriter(streams=["tracks"])) is False


# select_test_sink


def test_select_test_sink_picks_h5_regardless_of_order():
    """The H5 sink is selected first regardless of callback order."""
    h5, onnx = _h5_sink(), OnnxExportSink(model_name="X")
    assert select_test_sink([onnx, h5]) is h5
    assert select_test_sink([h5, onnx]) is h5


def test_select_test_sink_none_when_only_onnx():
    """An ONNX-only sink list selects nothing (no TEST persistence)."""
    assert select_test_sink([OnnxExportSink(model_name="X")]) is None


def test_select_test_sink_none_when_empty():
    """No sinks -> None."""
    assert select_test_sink([]) is None


def test_select_test_sink_raises_on_non_node_sink():
    """A duck sink without ``declare_io``/``is_sink`` is not a graph node."""
    duck = SimpleNamespace(name="duck", writer_demand=lambda *_a, **_k: {})
    with pytest.raises(ConfigError, match="not a graph node"):
        select_test_sink([duck])


# select_onnx_sink


def test_select_onnx_sink_either_order():
    """The ONNX sink is selected regardless of callback order."""
    h5, onnx = _h5_sink(), OnnxExportSink(model_name="X")
    assert select_onnx_sink([onnx, h5]) is onnx
    assert select_onnx_sink([h5, onnx]) is onnx


def test_select_onnx_sink_none_when_absent():
    """No ONNX sink present -> None."""
    assert select_onnx_sink([_h5_sink()]) is None


# select_fitval_callbacks


def test_select_fitval_callbacks_filters_on_callable_fit_val_demand():
    """Only callbacks with a callable ``fit_val_demand`` survive the filter."""
    has_demand = SimpleNamespace(fit_val_demand=lambda modules: {})
    no_demand = SimpleNamespace()
    assert select_fitval_callbacks([has_demand, no_demand]) == [has_demand]


def test_select_fitval_callbacks_none_returns_empty():
    """``None`` -> ``[]``."""
    assert select_fitval_callbacks(None) == []


# bind_manifests


class _RecordingSink:
    """Records the arguments passed to its bind_* methods."""

    def __init__(self) -> None:
        self.bound_modules = None
        self.bound_section = None

    def bind_model_modules(self, model_modules):
        self.bound_modules = model_modules

    def bind_output_section(self, section):
        self.bound_section = section


def test_bind_manifests_calls_both_when_available():
    """A sink with both bind methods receives both manifests."""
    sink = _RecordingSink()
    modules, section = {"a": object()}, {"b": object()}
    bind_manifests([sink], modules, section)
    assert sink.bound_modules is modules
    assert sink.bound_section is section


def test_bind_manifests_skips_empty_section():
    """An empty ``output_section`` does NOT call ``bind_output_section``."""
    sink = _RecordingSink()
    bind_manifests([sink], {"a": object()}, {})
    assert sink.bound_modules is not None
    assert sink.bound_section is None


def test_bind_manifests_leaves_object_with_neither_method_alone():
    """An object exposing neither bind method is left untouched (no crash)."""
    bind_manifests([object()], {"a": object()}, {"b": object()})


# fold_sink_node


def test_fold_sink_node_returns_new_dict_leaves_input_untouched():
    """`fold_sink_node` returns a NEW dict; the input mapping is untouched."""
    modules = {"task": _preds_stub()}
    sink = _h5_sink()
    folded = fold_sink_node(modules, sink)
    assert folded is not modules
    assert folded[sink.name] is sink
    assert sink.name not in modules


def test_fold_sink_node_collision_raises():
    """A name collision with an existing model module raises `ConfigError`."""
    sink = _h5_sink()
    modules = {sink.name: _preds_stub(sink.name)}
    with pytest.raises(ConfigError, match="collides with a model module"):
        fold_sink_node(modules, sink)


# callback_demand


def test_callback_demand_empty_for_test_and_onnx():
    """No callback demand outside TRAINING modes, regardless of callbacks."""
    cb = SimpleNamespace(fit_val_demand=lambda modules: ["labels.jets.x"])
    assert callback_demand({}, Mode.TEST, [cb]) == {}
    assert callback_demand({}, Mode.ONNX, [cb]) == {}


def test_callback_demand_setdefault_first_seen_wins():
    """The first callback to demand a key wins the attribution (``setdefault``)."""

    class _CBOne:
        def fit_val_demand(self, model_modules):
            return ["labels.jets.x"]

    class _CBTwo:
        def fit_val_demand(self, model_modules):
            return ["labels.jets.x"]

    demand = callback_demand({}, Mode.FIT, [_CBOne(), _CBTwo()])
    assert demand == {"labels.jets.x": "callback '_CBOne'"}


# model_sinks


def test_model_sinks_training_loss_only():
    """TRAINING with no callback keys anchors on ``loss.total`` alone."""
    assert model_sinks({"loss": _loss_stub()}, Mode.FIT) == ["loss.total"]


def test_model_sinks_training_with_callback_keys():
    """Callback keys not already present are appended after ``loss.total``."""
    sinks = model_sinks(
        {"loss": _loss_stub()}, Mode.FIT, callback_keys=("loss.total", "preds.x")
    )
    assert sinks == ["loss.total", "preds.x"]


def test_model_sinks_training_missing_loss_raises():
    """No module producing ``loss.total`` in a TRAINING mode is a hard error."""
    with pytest.raises(ConfigError, match="no module produces 'loss.total'"):
        model_sinks({}, Mode.FIT)


def test_model_sinks_test_all_preds_in_declared_order():
    """TEST/ONNX anchors every TEST-active ``preds.*`` key, in produced order."""
    modules = {
        "a": _Stub("a", produces={"preds.jets.a": TensorSpec(shape=("B",), dtype="float32")}),
        "b": _Stub("b", produces={"preds.jets.b": TensorSpec(shape=("B",), dtype="float32")}),
    }
    assert model_sinks(modules, Mode.TEST) == ["preds.jets.a", "preds.jets.b"]


def test_model_sinks_test_no_preds_raises():
    """No module producing any ``preds.*`` key in TEST/ONNX is a hard error."""
    with pytest.raises(ConfigError, match="anchor on predictions"):
        model_sinks({}, Mode.TEST)


# assert_no_dead_preds / dead_preds_message


def _dead_preds_plan(*, include_dead: bool):
    """A TEST plan folding one converted pred + optionally one unconsumed one."""
    out_key = "outputs.tracks.track_origin"
    track_task = _Stub(
        "track_task",
        produces={"preds.tracks.track_origin": TensorSpec(shape=("B", "L", 5), dtype="float32")},
    )
    producer = TaskOutput(task="track_origin", stream="tracks")
    producer.name = "track_origin_output"

    class _Sink:
        name = "sink"

        def is_sink(self) -> bool:
            return True

        def declare_io(self, mode):
            if not (mode & Mode.TEST):
                return IO(requires={}, produces={})
            return IO(
                unflatten_spec({out_key: TensorSpec(shape=None, dtype=None, kind="data")}),
                produces={},
            )

    sink = _Sink()
    modules = {track_task.name: track_task, producer.name: producer, sink.name: sink}
    if include_dead:
        modules["jet_task"] = _Stub(
            "jet_task",
            produces={"preds.jets.jets_classification": TensorSpec(shape=("B", 3), dtype="float32")},
        )
    sources = unflatten_spec({"meta.rows": TensorSpec(shape=(2,), dtype="int64", kind="meta")})
    plan = compile_plan(modules, Mode.TEST, sources, sinks=[])
    model_modules = {k: v for k, v in modules.items() if k != sink.name}
    return model_modules, plan


def test_assert_no_dead_preds_passes_when_all_consumed():
    """No dead pred when every produced ``preds.*`` feeds a demanded output."""
    model_modules, plan = _dead_preds_plan(include_dead=False)
    assert_no_dead_preds(model_modules, plan)


def test_assert_no_dead_preds_raises_naming_dead_key():
    """A computed-but-never-persisted ``preds.*`` is a hard error naming the key."""
    model_modules, plan = _dead_preds_plan(include_dead=True)
    with pytest.raises(ConfigError) as excinfo:
        assert_no_dead_preds(model_modules, plan)
    message = str(excinfo.value)
    assert "preds.jets.jets_classification" in message
    assert "expose=[fit,val]" in message
    assert "--model.modules.jet_task=null" in message


def test_dead_preds_message_contains_expose_and_null_hints():
    """`dead_preds_message` names the dead key and both fix hints."""
    message = dead_preds_message(["preds.jets.x"], {"preds.jets.x": "jet_task"})
    assert "preds.jets.x" in message
    assert "expose=[fit,val]" in message
    assert "--model.modules.jet_task=null" in message


# boundary_demand


def test_boundary_demand_fit_contains_dataset_key_not_sibling_produced():
    """FIT demand contains the dataset-only require, not the sibling-produced one."""
    model_modules = {"consumer": _requiring_stub(), "pooled": _pooled_stub()}
    prepared = PreparedSinks(
        modules={}, test_sink=None, onnx_sink=None, writer_demand=None, by_mode=_mode_sinks_map()
    )
    demand, origins = boundary_demand(model_modules, prepared)[Mode.FIT]
    assert "labels.jets.x" in demand
    assert "pooled.global" not in demand
    assert origins["labels.jets.x"] == "'consumer' (config: model.modules.consumer)"


def test_boundary_demand_test_appends_writer_demand_then_meta_rows_last():
    """TEST appends writer-demand keys (attributed) then ``meta.rows`` LAST."""
    prepared = PreparedSinks(
        modules={},
        test_sink=None,
        onnx_sink=None,
        writer_demand={"labels.jets.w": "sink 'X' demanding labels.jets.w"},
        by_mode=_mode_sinks_map(),
    )
    demand, origins = boundary_demand({}, prepared)[Mode.TEST]
    assert demand[-1] == "meta.rows"
    assert "labels.jets.w" in demand
    assert origins["labels.jets.w"] == "sink 'X' demanding labels.jets.w"


def test_boundary_demand_wildcard_writer_key_raises():
    """A wildcard writer-demand key is refused: writer requires must be concrete."""
    prepared = PreparedSinks(
        modules={},
        test_sink=None,
        onnx_sink=None,
        writer_demand={"labels.jets.*": "sink 'X'"},
        by_mode=_mode_sinks_map(),
    )
    with pytest.raises(ConfigError, match="writer requires are concrete keys"):
        boundary_demand({}, prepared)


def test_boundary_demand_training_callback_key_outside_namespaces_raises():
    """A TRAINING callback origin key outside the dataset namespaces is refused."""

    class _BogusCallback:
        def fit_val_demand(self, model_modules):
            return ["bogus.namespace.key"]

    prepared = PreparedSinks(
        modules={},
        test_sink=None,
        onnx_sink=None,
        writer_demand=None,
        by_mode=_mode_sinks_map(),
        fitval_callbacks=(_BogusCallback(),),
    )
    with pytest.raises(ConfigError, match="correct the callback's requires"):
        boundary_demand({}, prepared)


# prepare_sinks


def test_prepare_sinks_no_reader_disables_test_persistence():
    """Without a reader there is no TEST sink/writer demand; TEST anchors all preds."""
    modules = {"loss": _loss_stub(), "task": _preds_stub()}
    prep = prepare_sinks(modules, [_h5_sink()], output_section={}, reader=None, callbacks=[])
    assert prep.test_sink is None
    assert prep.writer_demand is None
    assert prep.by_mode[Mode.TEST].anchors == ["preds.jets.x"]


def test_prepare_sinks_with_reader_and_h5_folds_the_sink():
    """A reader + H5 sink folds it as the TEST anchor; FIT still anchors ``loss.total``."""
    modules = {"loss": _loss_stub(), "task": _preds_stub()}
    h5 = _h5_sink()
    prep = prepare_sinks(modules, [h5], output_section={}, reader=object(), callbacks=[])
    assert prep.test_sink is h5
    assert prep.modules["h5_output"] is h5
    assert prep.by_mode[Mode.TEST].anchors == []
    assert prep.by_mode[Mode.FIT].anchors == ["loss.total"]


def test_prepare_sinks_anchor_meta_rows_flag_controls_test_meta_rows_anchor():
    """No sink: ``anchor_meta_rows`` appends ``meta.rows`` to TEST anchors, or not."""
    modules = {"loss": _loss_stub(), "task": _preds_stub()}
    with_flag = prepare_sinks(
        modules, [], output_section={}, reader=None, callbacks=[], anchor_meta_rows=True
    )
    without_flag = prepare_sinks(
        modules, [], output_section={}, reader=None, callbacks=[], anchor_meta_rows=False
    )
    assert with_flag.by_mode[Mode.TEST].anchors[-1] == "meta.rows"
    assert "meta.rows" not in without_flag.by_mode[Mode.TEST].anchors


def test_prepare_sinks_modes_param_excludes_onnx_when_not_listed():
    """With ``modes=(FIT, VAL, TEST)``, the ONNX sink is never looked up or folded."""
    modules = {"loss": _loss_stub(), "task": _preds_stub()}
    onnx = OnnxExportSink(model_name="X")
    prep = prepare_sinks(
        modules,
        [onnx],
        output_section={},
        reader=None,
        callbacks=[],
        modes=(Mode.FIT, Mode.VAL, Mode.TEST),
    )
    assert prep.onnx_sink is None
    assert "onnx_export" not in prep.modules


def test_prepare_sinks_primary_modes_folds_onnx_sink():
    """With the default `PRIMARY_MODES`, the ONNX sink IS looked up and folded."""
    modules = {"loss": _loss_stub(), "task": _preds_stub()}
    onnx = OnnxExportSink(model_name="X")  # no inputs: contract check errors, collected below
    assert PRIMARY_MODES == (Mode.FIT, Mode.VAL, Mode.TEST, Mode.ONNX)
    prep = prepare_sinks(
        modules, [onnx], output_section={}, reader=None, callbacks=[], collect_errors=True
    )
    assert prep.onnx_sink is onnx
    assert "onnx_export" in prep.modules


def test_prepare_sinks_collect_errors_true_records_per_mode_error():
    """A raising ``fit_val_demand`` callback's error is captured per-mode, not raised."""
    modules = {"loss": _loss_stub(), "task": _preds_stub()}
    prep = prepare_sinks(
        modules,
        [],
        output_section={},
        reader=None,
        callbacks=[_RaisingCallback()],
        collect_errors=True,
    )
    fit = prep.by_mode[Mode.FIT]
    assert fit.error == "boom"
    assert fit.anchors == ["loss.total"]
    assert fit.anchor_origins is None


def test_prepare_sinks_collect_errors_false_raises():
    """Without ``collect_errors``, the same callback error propagates."""
    modules = {"loss": _loss_stub(), "task": _preds_stub()}
    with pytest.raises(ConfigError, match="boom"):
        prepare_sinks(
            modules,
            [],
            output_section={},
            reader=None,
            callbacks=[_RaisingCallback()],
            collect_errors=False,
        )


def test_prepare_sinks_onnx_sink_without_contract_warns_and_defaults_model_name():
    """An `OnnxExportSink` with no inputs/model_name warns and gets a default model_name."""
    modules = {"loss": _loss_stub(), "task": _preds_stub()}
    onnx = OnnxExportSink()
    prep = prepare_sinks(
        modules, [onnx], output_section={}, reader=None, callbacks=[], run_name="salt"
    )
    assert "declares no export contract" in prep.by_mode[Mode.ONNX].warning
    assert onnx.model_name == sanitised_model_name("salt")


def test_prepare_sinks_no_onnx_sink_warns():
    """No ONNX sink present: the ONNX mode warns it has nothing to export."""
    modules = {"loss": _loss_stub(), "task": _preds_stub()}
    prep = prepare_sinks(modules, [], output_section={}, reader=None, callbacks=[])
    assert "declares no OnnxExportSink" in prep.by_mode[Mode.ONNX].warning


def test_prepare_sinks_training_callback_sets_anchor_origins():
    """A FIT/VAL callback's demand key is attributed to it in ``anchor_origins``."""

    class _MetricsCallback:
        def fit_val_demand(self, model_modules):
            return ["labels.jets.aux"]

    modules = {"loss": _loss_stub(), "task": _preds_stub()}
    prep = prepare_sinks(
        modules, [], output_section={}, reader=None, callbacks=[_MetricsCallback()]
    )
    fit = prep.by_mode[Mode.FIT]
    assert fit.anchor_origins == {"labels.jets.aux": "callback '_MetricsCallback'"}
    assert "labels.jets.aux" in fit.anchors


def test_prepare_sinks_empty_modes_derives_no_anchors():
    """``modes=()`` selects/binds/records callbacks but derives no per-mode anchors."""
    modules = {"pooled": _pooled_stub()}
    prep = prepare_sinks(modules, [], output_section={}, reader=None, callbacks=[], modes=())
    assert prep.by_mode == {}
    assert prep.fitval_callbacks == ()
    with pytest.raises(ConfigError, match="no module produces 'loss.total'"):
        prepare_sinks(modules, [], output_section={}, reader=None, callbacks=[], modes=(Mode.FIT,))
    with pytest.raises(ConfigError, match="anchor on predictions"):
        prepare_sinks(modules, [], output_section={}, reader=None, callbacks=[], modes=(Mode.TEST,))


def test_prepare_sinks_test_only_needs_no_loss_total():
    """``modes=(TEST,)`` on a preds-only model derives TEST anchors without FIT's loss."""
    modules = {"task": _preds_stub()}
    prep = prepare_sinks(
        modules, [], output_section={}, reader=None, callbacks=[], modes=(Mode.TEST,)
    )
    assert prep.by_mode[Mode.TEST].anchors == ["preds.jets.x"]
    assert Mode.FIT not in prep.by_mode


def test_prepare_sinks_training_only_needs_no_preds():
    """``modes=(FIT, VAL)`` on a loss-only model derives FIT/VAL anchors without TEST's preds."""
    modules = {"loss": _loss_stub()}
    prep = prepare_sinks(
        modules, [], output_section={}, reader=None, callbacks=[], modes=(Mode.FIT, Mode.VAL)
    )
    assert prep.by_mode[Mode.FIT].anchors == ["loss.total"]
    assert prep.by_mode[Mode.VAL].anchors == ["loss.total"]
    assert Mode.TEST not in prep.by_mode


def test_prepare_sinks_writer_demand_false_skips_column_resolution():
    """``writer_demand=False`` folds the TEST sink but never calls its ``writer_demand``."""
    sink = H5OutputSink()
    sink.name = "h5_output"
    prep = prepare_sinks(
        {"task": _preds_stub()},
        [sink],
        output_section={},
        reader=object(),
        callbacks=[],
        modes=(Mode.TEST,),
        writer_demand=False,
    )
    assert prep.test_sink is sink
    assert prep.writer_demand is None
    assert "h5_output" in prep.modules
    assert prep.by_mode[Mode.TEST].anchors == []
    with pytest.raises(ConfigError, match="found no RunTaskOutput task with a final H5 column"):
        prepare_sinks(
            {"task": _preds_stub()},
            [sink],
            output_section={},
            reader=object(),
            callbacks=[],
            modes=(Mode.TEST,),
        )


def test_prepare_sinks_records_fitval_callbacks():
    """`fitval_callbacks` keeps only the demand-declaring callback, by identity."""

    class _MetricsCallback:
        def fit_val_demand(self, model_modules):
            return ["labels.jets.aux"]

    cb = _MetricsCallback()
    modules = {"loss": _loss_stub(), "task": _preds_stub()}
    prep = prepare_sinks(
        modules, [], output_section={}, reader=None, callbacks=[cb, object()], modes=()
    )
    assert prep.fitval_callbacks == (cb,)
    assert prep.by_mode == {}
    demand = boundary_demand(modules, prep)
    assert demand[Mode.FIT][1]["labels.jets.aux"] == "callback '_MetricsCallback'"
    assert "labels.jets.aux" in demand[Mode.FIT][0]


# frozen dataclasses


def test_mode_sinks_and_prepared_sinks_are_frozen():
    """Both dataclasses reject attribute assignment after construction."""
    mode_sinks_obj = ModeSinks(anchors=[], anchor_origins=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        mode_sinks_obj.anchors = ["x"]  # type: ignore[misc]

    prepared = PreparedSinks(
        modules={}, test_sink=None, onnx_sink=None, writer_demand=None, by_mode={}
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        prepared.modules = {}  # type: ignore[misc]
