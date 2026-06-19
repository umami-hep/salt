"""salt.core.data — the v2 dataset pipeline (design §2.4, §6).

Reader -> processors -> numpy→torch boundary, compiled and executed by the
same graph kernel as the model side. Public surface:

- base classes: `DatasetModule`, `Reader`, `Processor`, `WorkerCtx`
- the throughput-preserving reader: `H5StructuredReader` (+ `GroupConfig`)
- shipped processors: `Features`, `Labels`, `MultiTarget`
- the runtime: `GraphDataset` (per-batch plan execution + torch boundary)
  and `GraphDataModule` (Lightning wiring, ``data:`` YAML block)
- VDS helpers: `create_vds`, `default_vds_path`, `has_wildcard`
"""

from __future__ import annotations

from salt.core.data.base import DatasetModule, Processor, Reader, WorkerCtx
from salt.core.data.datamodule import GraphDataModule
from salt.core.data.dataset import MODEL_VISIBLE_NAMESPACES, GraphDataset
from salt.core.data.easyjet_reader import EasyjetGroupConfig, EasyjetReader
from salt.core.data.processors import Features, Labels, MaskFormerTargets, MultiTarget
from salt.core.data.reader import GroupConfig, H5StructuredReader
from salt.core.data.vds import create_vds, default_vds_path, has_wildcard

__all__ = [
    "MODEL_VISIBLE_NAMESPACES",
    "DatasetModule",
    "EasyjetGroupConfig",
    "EasyjetReader",
    "Features",
    "GraphDataModule",
    "GraphDataset",
    "GroupConfig",
    "H5StructuredReader",
    "Labels",
    "MaskFormerTargets",
    "MultiTarget",
    "Processor",
    "Reader",
    "WorkerCtx",
    "create_vds",
    "default_vds_path",
    "has_wildcard",
]
