"""salt.data — the v2 dataset pipeline: readers, processors, the
`SaltDataset` runtime and `SaltDataModule` Lightning wiring, VDS helpers.
"""

from __future__ import annotations

from salt.data.base import Processor, Reader, RowBlock, SaltDatasetModule, WorkerCtx
from salt.data.datamodule import SaltDataModule
from salt.data.dataset import MODEL_VISIBLE_NAMESPACES, SaltDataset
from salt.data.input_samples import InputSamples
from salt.data.iterable_dataset import DEFAULT_BLOCK_ROWS, IterableSaltDataset
from salt.data.manifest import CorpusManifest, ManifestEntry, build_manifest
from salt.data.processors.features import Features
from salt.data.processors.ftag_labeller import FtagLabeller
from salt.data.processors.labels import Labels
from salt.data.processors.maskformer_targets import MaskFormerTargets
from salt.data.processors.multi_target import MultiTarget
from salt.data.readers.cuts import ConstituentCuts, Cut, CutSpec, GlobalObjectCuts
from salt.data.readers.multisample_reader import MultiSampleReader, SampleConfig
from salt.data.readers.reader import GroupConfig, H5StructuredReader
from salt.data.readers.stream import OffsetIndex, StreamConfig
from salt.data.readers.uproot_reader import UprootGroupConfig, UprootReader
from salt.data.readers.vds import VDS, create_vds, default_vds_path, has_wildcard

__all__ = [
    "DEFAULT_BLOCK_ROWS",
    "MODEL_VISIBLE_NAMESPACES",
    "VDS",
    "ConstituentCuts",
    "CorpusManifest",
    "Cut",
    "CutSpec",
    "Features",
    "FtagLabeller",
    "GlobalObjectCuts",
    "GroupConfig",
    "H5StructuredReader",
    "InputSamples",
    "IterableSaltDataset",
    "Labels",
    "ManifestEntry",
    "MaskFormerTargets",
    "MultiSampleReader",
    "MultiTarget",
    "OffsetIndex",
    "Processor",
    "Reader",
    "RowBlock",
    "SaltDataModule",
    "SaltDataset",
    "SaltDatasetModule",
    "SampleConfig",
    "StreamConfig",
    "UprootGroupConfig",
    "UprootReader",
    "WorkerCtx",
    "build_manifest",
    "create_vds",
    "default_vds_path",
    "has_wildcard",
]
