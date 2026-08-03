"""salt.outputs.sinks — the terminal sink nodes and the machinery behind them.

The producers live one level up in `salt.outputs`; everything that CONSUMES
their ``outputs.*`` leaves and puts them somewhere lives here.
"""

from __future__ import annotations

from salt.outputs.sinks.h5_sink import H5OutputSink, H5OutputWriter
from salt.outputs.sinks.jsonl_sink import JSONLOutputSink
from salt.outputs.sinks.onnx_sink import OnnxExportLeaf, OnnxExportSink
from salt.outputs.sinks.registry import iter_sinks, register_sink, sink_registry
from salt.outputs.sinks.sink import (
    Node,
    OutputSink,
    RuntimeSink,
    SinkContext,
    collect_manifest_fields,
    is_test_persistence_sink,
    parse_modes,
)

__all__ = [
    "H5OutputSink",
    "H5OutputWriter",
    "JSONLOutputSink",
    "Node",
    "OnnxExportLeaf",
    "OnnxExportSink",
    "OutputSink",
    "RuntimeSink",
    "SinkContext",
    "collect_manifest_fields",
    "is_test_persistence_sink",
    "iter_sinks",
    "parse_modes",
    "register_sink",
    "sink_registry",
]
