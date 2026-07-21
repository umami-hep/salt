"""salt.data — the v2 dataset pipeline: readers, processors, the
`GraphDataset` runtime and `GraphDataModule` Lightning wiring, VDS helpers.
"""

from __future__ import annotations

from salt.data.base import Processor, Reader, SaltDatasetModule, WorkerCtx
from salt.data.readers.cuts import Cut, CutSpec
from salt.data.datamodule import GraphDataModule
from salt.data.dataset import MODEL_VISIBLE_NAMESPACES, GraphDataset
from salt.data.readers.easyjet_reader import EasyjetGroupConfig, EasyjetReader
from salt.data.processors.features import Features
from salt.data.readers.ftag1lite_reader import FTAG1LiteGroupConfig, FTAG1LiteReader
from salt.data.processors.ftag_labeller import FtagLabeller
from salt.data.input_samples import InputSamples
from salt.data.processors.labels import Labels
from salt.data.processors.maskformer_targets import MaskFormerTargets
from salt.data.processors.multi_target import MultiTarget
from salt.data.readers.multisample_reader import MultiSampleReader, SampleConfig
from salt.data.readers.physlite_reader import PhysliteGroupConfig, PhysliteReader
from salt.data.readers.reader import GroupConfig, H5StructuredReader
from salt.data.readers.stream import OffsetIndex, StreamConfig
from salt.data.readers.uproot_reader import UprootGroupConfig, UprootReader
from salt.data.readers.xaod_reader import XAODReader
from salt.data.readers.vds import create_vds, default_vds_path, has_wildcard
from salt.data.readers.vds_module import VDS

__all__ = [
    "MODEL_VISIBLE_NAMESPACES",
    "VDS",
    "Cut",
    "CutSpec",
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
    "PhysliteGroupConfig",
    "PhysliteReader",
    "Processor",
    "Reader",
    "SaltDatasetModule",
    "SampleConfig",
    "StreamConfig",
    "UprootGroupConfig",
    "UprootReader",
    "WorkerCtx",
    "XAODReader",
    "create_vds",
    "default_vds_path",
    "has_wildcard",
]
