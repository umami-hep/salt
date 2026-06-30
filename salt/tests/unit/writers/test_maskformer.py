"""Tests for `salt.core.writers.maskformer` — the MaskFormerObjectWriter.

TEST byte-parity vs the v1 op chain + the extra-group plumbing
(predictionwriter.py:267-308; M5 sub-wave C, plan 10): decoder-preds/truth
requires, the objects/object_masks extra groups, v1 column naming, the
ONNX two-object-reduce manifest, and the OBJECT_INDEX constant single-ownership.

(Split out of the former monolithic ``test_writers.py``; shared fixtures /
toy-writers / constants come from ``salt.tests._fixtures.writers_common``.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.writers import (
    OBJECT_INDEX,
    MaskFormerObjectWriter,
    WriteCtx,
    WriterCallback,
)
from salt.tests._fixtures.writers_common import (  # noqa: F401  (data/modules are fixtures)
    L_FILE,
    data,
    declare_ctx,
    modules,
)

# ---------------------------------------------------------------------------
# MaskFormerObjectWriter: TEST byte-parity vs v1 + the extra-group plumbing
# (predictionwriter.py:267-308; M5 sub-wave C, plan 10)
# ---------------------------------------------------------------------------


OBJECT_CLASSES = ["b", "c", "null"]


def mf_writer_modules(nd):
    from salt.tests._fixtures.regression_fixture import build_maskformer_writer_modules

    return build_maskformer_writer_modules(nd)


def mf_write_ctx(data, modules, n_tracks: int = 10, total: int = 6) -> WriteCtx:
    return WriteCtx(
        output_path=data["dir"] / "mf.h5",
        total=total,
        run_name="MFrun",
        source_path=data["h5"],
        streams=("jets", "tracks"),
        sequence_streams=("tracks",),
        group_datasets={"jets": "jets", "tracks": "tracks"},
        seq_lengths={"tracks": n_tracks},
        model_modules=modules,
        batch_size=total,
    )


class TestMaskFormerObjectWriter:
    def _writer(self):
        w = MaskFormerObjectWriter(object_classes=OBJECT_CLASSES, regression_task="regression")
        w.name = "object_writer"
        return w

    def test_requires_decoder_preds_and_truth_labels(self, modules):
        # the truth requires keep MaskFormerTargets alive in the TEST plan (amendment §3);
        # masks.tracks is the constituent pad mask `write` reads for the MaskIndex padding
        keys = sorted(self._writer().requires(declare_ctx(modules)))
        assert keys == [
            "labels.objects.masks",
            "labels.objects.object_class",
            "masks.tracks",
            "objects.class_probs",
            "objects.masks",
        ]

    def test_extra_groups_object_and_object_masks(self, data, modules):
        mods = mf_writer_modules(data["nd"])
        groups = self._writer().extra_groups(mf_write_ctx(data, mods, n_tracks=L_FILE))
        # objects [M], object_masks [M, T] (M from the decoder, T from the file length)
        assert groups == {"objects": (5,), "object_masks": (5, L_FILE)}

    def test_columns_v1_naming(self, data, modules):
        mods = mf_writer_modules(data["nd"])
        cols = self._writer().columns(mf_write_ctx(data, mods))
        # objects: per-class p{name} probs + the class_label truth (v1 :276-285)
        assert list(cols["objects"].names) == ["MFrun_pb", "MFrun_pc", "MFrun_pnull", "class_label"]
        # the constituent stream gains the MaskIndex column (the PINNED test suffix)
        assert list(cols["tracks"].names) == [f"MFrun_{OBJECT_INDEX.test}"]
        # object_masks: truth + logits (v1 :300-308)
        assert list(cols["object_masks"].names) == ["truth_mask", "mask_logits"]

    def test_write_byte_parity_vs_v1_opchain(self, data, modules):
        from salt.tests._fixtures.regression_fixture import make_maskformer_writer_batch
        from salt.utils.mask_utils import indices_from_mask

        mods = mf_writer_modules(data["nd"])
        writer = self._writer()
        writer.setup(mf_write_ctx(data, mods))
        batch = make_maskformer_writer_batch(batch_size=6, n_tracks=10)
        bundle = Bundle()
        for key, value in batch.items():
            bundle.set(key, value)
        out = writer.write(bundle, slice(0, 6))

        cp, masks = batch["objects.class_probs"], batch["objects.masks"]
        oc, tm, pad = (
            batch["labels.objects.object_class"],
            batch["labels.objects.masks"],
            batch["masks.tracks"],
        )
        # objects group: probs + remapped truth class (v1 op chain, predictionwriter.py:276-285)
        v1_probs = u2s(cp.numpy(), np.dtype([(f"MFrun_p{c}", "f4") for c in OBJECT_CLASSES]))
        for n in v1_probs.dtype.names:
            assert out["objects"][n].tobytes() == v1_probs[n].tobytes()
        v1_class = u2s(oc.unsqueeze(-1).numpy(), np.dtype([("class_label", "i8")]))
        assert out["objects"]["class_label"].tobytes() == v1_class["class_label"].tobytes()
        # MaskIndex: indices_from_mask(sigmoid > 0.5) (-2 no object), -1 padded (v1 :287-297)
        v1_idx = indices_from_mask(masks.sigmoid() > 0.5).int().numpy()
        v1_idx = np.where(~pad.numpy(), v1_idx, -1)
        col = f"MFrun_{OBJECT_INDEX.test}"
        assert (
            out["tracks"][col].tobytes()
            == u2s(np.expand_dims(v1_idx, -1), np.dtype([(col, "i8")]))[col].tobytes()
        )
        assert (out["tracks"][col] == -1).any()  # padded sentinel
        assert (out["tracks"][col] == -2).any()  # no-object sentinel
        # object_masks group: truth mask + logits (v1 :300-308)
        assert (
            out["object_masks"]["truth_mask"].tobytes()
            == u2s(tm.unsqueeze(-1).numpy(), np.dtype([("truth_mask", "i8")]))[
                "truth_mask"
            ].tobytes()
        )
        assert (
            out["object_masks"]["mask_logits"].tobytes()
            == u2s(masks.float().unsqueeze(-1).numpy(), np.dtype([("mask_logits", "f4")]))[
                "mask_logits"
            ].tobytes()
        )

    def test_onnx_manifest_two_object_reduces(self, data):
        from salt.core.onnx.config import ExportOutput

        mods = mf_writer_modules(data["nd"])
        manifest = self._writer().onnx_outputs(declare_ctx(mods))
        assert all(type(e) is ExportOutput for e in manifest)
        # leading_object (5/3 split_scalars-style names) then object_index (HadronIndex)
        assert manifest[0].port == "preds.objects.regression"
        assert manifest[0].reduce == "leading_object"
        assert manifest[0].names == [
            "leading_objects_pt",
            "leading_objects_Lxy",
            "leading_objects_mass",
        ]
        assert manifest[1].port == "objects.masks"
        assert manifest[1].reduce == "object_index"
        assert manifest[1].name == OBJECT_INDEX.onnx  # HadronIndex (pinned divergence)

    def test_onnx_false_is_eval_only(self, data):
        mods = mf_writer_modules(data["nd"])
        w = MaskFormerObjectWriter(
            object_classes=OBJECT_CLASSES, regression_task="regression", onnx=False
        )
        w.name = "object_writer"
        assert w.onnx_outputs(declare_ctx(mods)) == []
        assert "objects.class_probs" in w.requires(declare_ctx(mods))  # TEST role intact

    def test_empty_object_classes_rejected(self):
        with pytest.raises(ConfigError, match="non-empty"):
            MaskFormerObjectWriter(object_classes=[])

    def test_object_index_imported_not_redeclared(self):
        # merge condition 4: the strings live ONLY in salt.core.writers.names
        import salt.core.writers.maskformer as src

        source = Path(src.__file__).read_text()
        assert "from salt.core.writers.names import OBJECT_INDEX" in source
        assert '"MaskIndex"' not in source and "'MaskIndex'" not in source
        assert '"HadronIndex"' not in source and "'HadronIndex'" not in source

    def test_extra_group_merges_through_writer_callback(self, data):
        # the extra-group plumbing: the writer's objects/object_masks groups flow
        # through WriterCallback._merge_columns with their writer-declared shapes
        mods = mf_writer_modules(data["nd"])
        cb = WriterCallback(modules={"object_writer": self._writer()})
        ctx = mf_write_ctx(data, mods)
        for writer in cb.writers.values():
            writer.setup(ctx)
        dtypes, shapes = cb._merge_columns(ctx)
        assert "objects" in dtypes and "object_masks" in dtypes
        assert shapes["objects"] == (6, 5)  # (total, M)
        assert shapes["object_masks"] == (6, 5, 10)  # (total, M, T=n_tracks of the ctx)
        # the reader-stream MaskIndex column rides on the tracks group
        assert f"MFrun_{OBJECT_INDEX.test}" in dtypes["tracks"].names
