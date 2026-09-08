"""Unit mirror for `salt/scripts/regen_gn3large_config.py`.

The script is a TEXTUAL rewrite of the stale public GN3Large bundle config (no
YAML round-trip), so these tests drive it with a minimal stale-shaped fixture
and assert each of the four rewrites plus the fail-loud paths. The shipped
mirror itself is checked against the real bundle by
`salt/tests/unit/test_finetune_configs.py` and the `--check` CLI.
"""

from __future__ import annotations

import pytest

from salt.scripts import regen_gn3large_config as regen

# A minimal stale-shaped config: every anchor the rewrites key on, in the
# order the real bundle has them (callbacks: before export: before trainer:).
STALE = """\
name: GN4_big
data:
  modules:
    reader:
      class_path: salt.core.data.H5StructuredReader
model:
  class_path: salt.core.SaltModule
  init_args:
    modules:
      norm:
        class_path: salt.core.nn.Normaliser
        init_args:
          norm_dict: /data/atlas_samples/somewhere/norm_dict_v2.yaml
      jets_classification:
        class_path: salt.core.nn.tasks.ClassificationTaskModule
outputs:
  pad_mask:
    class_path: salt.core.outputs.PadMaskWriter
callbacks:
  h5_output:
    class_path: salt.outputs.H5OutputSink
  onnx_export:
    class_path: salt.outputs.OnnxExportSink
export:
  model_name: GN3Large
  inputs:
  - {port: inputs.jets, name: jets_features}
trainer:
  precision: 32-true
"""


class TestRegenerateBody:
    def test_no_salt_core_survives(self):
        body = regen.regenerate_body(STALE)
        assert "salt.core." not in body

    def test_class_paths_mapped(self):
        body = regen.regenerate_body(STALE)
        assert "class_path: salt.data.H5StructuredReader\n" in body
        assert "class_path: salt.model.SaltModule\n" in body
        assert "class_path: salt.model.modules.Normaliser\n" in body
        assert "class_path: salt.model.modules.tasks.ClassificationTaskModule\n" in body
        assert "class_path: salt.outputs.PadMaskWriter\n" in body

    def test_norm_dict_relative_sibling(self):
        body = regen.regenerate_body(STALE)
        assert "          norm_dict: norm_dict_v2.yaml\n" in body
        assert "/data/atlas_samples" not in body

    def test_export_folded_into_outputs_and_callbacks_dropped(self):
        body = regen.regenerate_body(STALE)
        assert "callbacks:" not in body
        assert "\nexport:\n" not in body
        # the export body is re-indented under outputs.onnx_export.init_args
        assert "  onnx_export:\n    class_path: salt.outputs.OnnxExportSink\n    init_args:\n" in body
        assert "      model_name: GN3Large\n" in body
        # untouched tail
        assert body.endswith("trainer:\n  precision: 32-true\n")

    def test_everything_else_byte_identical(self):
        body = regen.regenerate_body(STALE)
        # lines the rewrites do not target survive verbatim
        for line in ("name: GN4_big\n", "data:\n", "  modules:\n", "    reader:\n"):
            assert line in body

    def test_idempotent_on_its_own_output_fails_loud(self):
        # a second pass finds no callbacks:/export: anchors -> fail-loud, not silent
        body = regen.regenerate_body(STALE)
        with pytest.raises(SystemExit):
            regen.regenerate_body(body)


class TestFailLoud:
    def test_unmapped_salt_core_class_path(self):
        stale = STALE.replace(
            "salt.core.data.H5StructuredReader", "salt.core.data.NoSuchReader"
        )
        with pytest.raises(SystemExit):
            regen.regenerate_body(stale)

    def test_missing_export_block(self):
        stale = STALE.replace(
            "export:\n  model_name: GN3Large\n  inputs:\n  - {port: inputs.jets, name: jets_features}\n",
            "",
        )
        with pytest.raises(SystemExit):
            regen.regenerate_body(stale)

    def test_unexpected_callbacks_block(self):
        stale = STALE.replace("  h5_output:\n", "  h5_output_renamed:\n")
        with pytest.raises(SystemExit):
            regen.regenerate_body(stale)

    def test_norm_dict_must_appear_once(self):
        stale = STALE.replace(
            "          norm_dict: /data/atlas_samples/somewhere/norm_dict_v2.yaml\n",
            "",
        )
        with pytest.raises(SystemExit):
            regen.regenerate_body(stale)


class TestHeader:
    def test_build_and_split_round_trip(self):
        body = regen.regenerate_body(STALE)
        header = regen.build_header("a" * 64, "b" * 64, "c" * 64, regen.sha256_bytes(body.encode()))
        assert header.startswith("#")
        assert "a" * 64 in header
        assert regen.sha256_bytes(body.encode()) in header
        head, split_body = regen._split_header(header + body)
        assert head == header
        assert split_body == body
        assert split_body.startswith("name: GN4_big")

    def test_split_requires_header(self):
        with pytest.raises(SystemExit):
            regen._split_header(regen.regenerate_body(STALE))

    def test_sha256_helpers(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_bytes(b"hello")
        assert regen.sha256_bytes(b"hello") == regen.sha256_file(p)
        assert len(regen.sha256_bytes(b"hello")) == 64
