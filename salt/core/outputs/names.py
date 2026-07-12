"""Shared output-suffix constants — one owner per logical output name.

Writers are the single source of output names for both modes: a writer
declares a logical *suffix*; the TEST column name is ``{run_name}_{suffix}``
and the ONNX output name is ``{export.model_name}_{suffix}`` (the exporter
owns the prefix — writers never see the Athena name). The constants here are
the suffixes shared (or explicitly pinned as divergent) across modes, so a
logical output can never be named independently in two different files —
every cross-mode name decision lives in this module.

Two kinds of entries:

- **Shared constants** (`VERTEX_INDEX`): the SAME suffix in TEST and ONNX.
  The remaining v1 asymmetry (TEST writes the bare column while
  ``prefix_vertex_column=false``) is a *prefix policy* on the TaskWriter, not
  a second name — see ``modules.py``.
- **Pinned divergences** (`OBJECT_INDEX`): v1 shipped two different suffixes
  for one logical output and byte-parity gating forces v2 to keep both —
  recorded as one explicit, documented pair instead of two accidental
  literals. `MaskFormerObjectWriter` must import `OBJECT_INDEX` — never
  re-declare the strings.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["OBJECT_INDEX", "VERTEX_INDEX", "ModeSplitSuffix", "pascal_case"]

VERTEX_INDEX = "VertexIndex"
"""The vertexing output suffix — shared by TEST and ONNX.

TEST: the `TaskWriter` column (bare while ``prefix_vertex_column=false``,
v1 byte parity; ``{run_name}_VertexIndex`` once that flag flips). ONNX:
``{model_name}_VertexIndex``. One constant, one per-mode prefix rule, one
compat flag.
"""


@dataclass(frozen=True)
class ModeSplitSuffix:
    """A pinned cross-mode suffix divergence.

    v1 shipped some logical outputs under different suffixes in eval and
    ONNX; byte-parity gates force v2 to reproduce both. Instead of two
    accidental literals in two files, the pair is declared once here with
    the divergence on record.

    Parameters
    ----------
    test : str
        The TEST (eval H5 column) suffix.
    onnx : str
        The ONNX (Athena output) suffix.
    why : str
        The v1 evidence pinning the divergence.
    """

    test: str
    onnx: str
    why: str


OBJECT_INDEX = ModeSplitSuffix(
    test="MaskIndex",
    onnx="HadronIndex",
    why=(
        "v1 shipped the MaskFormer track-to-object index as the eval column "
        "'{run_name}_MaskIndex' (predictionwriter.py:289) but the ONNX output "
        "'{model_name}_HadronIndex' (to_onnx.py object outputs) — byte-parity gating (M5 "
        "W-gates / O-gates) pins both until an M7-style adjudication renames one side"
    ),
)
"""The MaskFormer track-to-object index suffix pair.

`MaskFormerObjectWriter` uses ``OBJECT_INDEX.test`` for its eval column and
``OBJECT_INDEX.onnx`` for its `ExportOutput` entry.
"""


def pascal_case(name: str) -> str:
    """Snake-case to Pascal-case — the default ONNX aux-output suffix rule.

    Reproduces v1's hand-built aux names: ``track_origin -> TrackOrigin``,
    ``track_type -> TrackType``. Overridable per task via the `TaskWriter`
    ``onnx_names:`` mapping. Each ``_``-separated part is upper-cased at its
    first letter only, so e.g. ``track_pVtx`` stays ``TrackPVtx``-free
    (that would require an explicit ``onnx_names`` entry).
    """
    return "".join(part[:1].upper() + part[1:] for part in name.split("_"))
