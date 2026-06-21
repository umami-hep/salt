"""salt.core.outputs — the two-layer output-writing system (design §2).

Replaces the M4.5 single-manifest writers (``salt/core/writers/``) with a
clean split mirroring the model's dataflow:

- **producers** (`salt.core.outputs.producers`) — in-graph `GraphModule`s that
  read raw ``preds.<stream>.<task>`` and write eval/export-ready
  ``outputs.<stream>.<name>`` leaves. They live in ``model.modules`` and are
  carried by the existing planner / executor / bind machinery. Active in
  TEST/ONNX only, so the planner demand-prunes them from FIT/VAL (the §4
  risk 4 keystone) and the training path is unperturbed.
- **sinks** (`salt.core.outputs.sinks`) — terminal Lightning callbacks that
  consume the producers' ``outputs.*`` leaves. Each is told which output names
  it wants; its declared demand anchors the TEST plan's sinks, keeping the
  producers alive in TEST.

P0 ships the generic ``TaskOutput`` producer (identity copy) and the minimal
``CollectOutputs`` no-op sink — the smallest slice proving
producer -> ``outputs.*`` -> sink with demand-gating. Conversion producers
(P1: softmax/de-scale) and the ``H5OutputWriter`` sink build on this contract.
"""

from __future__ import annotations

from salt.core.outputs.producers import TaskOutput
from salt.core.outputs.sinks import CollectOutputs

__all__ = ["CollectOutputs", "TaskOutput"]
