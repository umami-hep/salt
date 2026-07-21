"""Pre-de-core checkpoint/config class_path compatibility (Plan 61 W5).

Checkpoints and their saved ``config.yaml`` written before the de-core rename
carry ``salt.core.*`` class_paths; the load-time remapper resolves them to the
flat ``salt.*`` namespace. These are the ONLY sanctioned ``salt.core.*`` strings
in the codebase outside the remapper table itself and migration notes.
"""

from __future__ import annotations

import pytest

import salt.main as m  # importing installs the jsonargparse import_object patch

# (pre-de-core class_path, expected salt.* home) — covers the three-way nn
# fan-out (facade -> modules, submodule -> modules/nn/model), the data fan-out
# (facade -> salt.data), and the flat top-level singletons.
REMAP_CASES = [
    ("salt.core.SaltModule", "salt.model.SaltModule"),
    ("salt.core.saltmodule.SaltModule", "salt.model.saltmodule.SaltModule"),
    ("salt.core.nn.StreamEmbed", "salt.model.modules.StreamEmbed"),
    ("salt.core.nn.TransformerEncoder", "salt.model.modules.TransformerEncoder"),
    ("salt.core.nn.Normaliser", "salt.model.modules.Normaliser"),
    ("salt.core.nn.tasks.ClassificationTaskModule", "salt.model.modules.tasks.ClassificationTaskModule"),
    ("salt.core.nn.base.SaltModelModule", "salt.model.base.SaltModelModule"),
    ("salt.core.nn.bind.bind_all", "salt.model.bind.bind_all"),
    ("salt.core.nn.dense.Dense", "salt.model.nn.dense.Dense"),
    ("salt.core.nn.transformer.Transformer", "salt.model.nn.transformer.Transformer"),
    ("salt.core.outputs.H5OutputSink", "salt.outputs.H5OutputSink"),
    ("salt.core.outputs.OnnxExportSink", "salt.outputs.OnnxExportSink"),
    ("salt.core.data.H5StructuredReader", "salt.data.H5StructuredReader"),
    ("salt.core.data.reader.H5StructuredReader", "salt.data.readers.reader.H5StructuredReader"),
    ("salt.core.data.features.Features", "salt.data.processors.features.Features"),
    ("salt.core.data.datamodule.GraphDataModule", "salt.data.datamodule.GraphDataModule"),
    ("salt.core.callbacks.Checkpoint", "salt.callbacks.Checkpoint"),
    ("salt.core.graph.planner.compile_plan", "salt.graph.planner.compile_plan"),
    ("salt.core.optim.HybridMuonAdamW", "salt.optim.HybridMuonAdamW"),
    ("salt.core.loss_history.LossHistoryWriter", "salt.utils.loss_history.LossHistoryWriter"),
    ("salt.core.mup.setup_mup", "salt.model.mup.setup_mup"),
]

NOOP_CASES = [
    "torch.nn.BCEWithLogitsLoss",
    "lightning.pytorch.callbacks.LearningRateMonitor",
    "salt.model.SaltModule",  # already-new path is untouched
    "salt.data.H5StructuredReader",
    "saltcore.thing",  # not the salt.core package (no false prefix match)
]


@pytest.mark.parametrize(("old", "new"), REMAP_CASES)
def test_remap_class_path(old, new):
    assert m._remap_class_path(old) == new


@pytest.mark.parametrize("name", NOOP_CASES)
def test_remap_is_noop_for_non_pre_decore(name):
    assert m._remap_class_path(name) == name


def test_longest_prefix_wins_over_facade():
    # salt.core.nn.tasks.* and salt.core.nn.base.* must NOT collapse to the bare
    # salt.core.nn -> salt.model.modules facade rule.
    assert m._remap_class_path("salt.core.nn.tasks.X") == "salt.model.modules.tasks.X"
    assert m._remap_class_path("salt.core.nn.base.X") == "salt.model.base.X"
    assert m._remap_class_path("salt.core.nn.dense.X") == "salt.model.nn.dense.X"
    assert m._remap_class_path("salt.core.nn.X") == "salt.model.modules.X"


def test_resolve_class_path_accepts_pre_decore():
    from salt.model.saltmodule import SaltModule

    assert m._resolve_class_path("salt.core.SaltModule") is SaltModule
    assert m._resolve_class_path("salt.model.SaltModule") is SaltModule  # new path still works


def test_jsonargparse_import_object_remaps_subclass_path():
    # the exact call jsonargparse makes when resolving a subclass class_path in a
    # saved config — must transparently accept the pre-de-core path.
    from jsonargparse import _typehints as th

    assert getattr(th.import_object, "_salt_decore_remap", False)
    from salt.model.modules.stream_embed import StreamEmbed

    assert th.import_object("salt.core.nn.StreamEmbed") is StreamEmbed


def test_every_remap_target_is_importable():
    # each remapped path must resolve to a real object (guards against a stale
    # entry after future refactors).
    for old, _new in REMAP_CASES:
        assert m._resolve_class_path(old) is not None
