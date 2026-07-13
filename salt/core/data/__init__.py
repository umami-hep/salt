"""salt.core.data — the v2 dataset pipeline: readers, processors, the
`GraphDataset` runtime and `GraphDataModule` Lightning wiring, VDS helpers.
"""

from __future__ import annotations

from salt.core.data.base import DatasetModule, Processor, Reader, WorkerCtx
from salt.core.data.cuts import Cut, CutSpec
from salt.core.data.datamodule import GraphDataModule
from salt.core.data.dataset import MODEL_VISIBLE_NAMESPACES, GraphDataset
from salt.core.data.easyjet_reader import EasyjetGroupConfig, EasyjetReader
from salt.core.data.features import Features
from salt.core.data.ftag1lite_reader import FTAG1LiteGroupConfig, FTAG1LiteReader
from salt.core.data.ftag_labeller import FtagLabeller
from salt.core.data.input_samples import InputSamples
from salt.core.data.labels import Labels
from salt.core.data.maskformer_targets import MaskFormerTargets
from salt.core.data.multi_target import MultiTarget
from salt.core.data.multisample_reader import MultiSampleReader, SampleConfig
from salt.core.data.reader import GroupConfig, H5StructuredReader
from salt.core.data.stream import OffsetIndex, StreamConfig
from salt.core.data.vds import create_vds, default_vds_path, has_wildcard
from salt.core.data.vds_module import VDS

__all__ = [
    "MODEL_VISIBLE_NAMESPACES",
    "VDS",
    "Cut",
    "CutSpec",
    "DatasetModule",
    "EasyjetGroupConfig",
    "EasyjetReader",
    "FTAG1LiteGroupConfig",
    "FTAG1LiteReader",
    "Features",
    "FtagLabeller",
    "GraphDataModule",
    "GraphDataset",
    "GroupConfig",
    "H5StructuredReader",
    "InputSamples",
    "Labels",
    "MaskFormerTargets",
    "MultiSampleReader",
    "MultiTarget",
    "OffsetIndex",
    "Processor",
    "Reader",
    "SampleConfig",
    "StreamConfig",
    "WorkerCtx",
    "create_vds",
    "default_vds_path",
    "has_wildcard",
]
