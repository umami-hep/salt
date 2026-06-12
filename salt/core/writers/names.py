"""Shared output-suffix constants — one owner per logical output name.

The M4.5 unified-manifest amendment (amendment §5, merge condition 4) makes
writers the single source of output names for BOTH modes: a writer declares
a logical *suffix*; the TEST column name is ``{run_name}_{suffix}`` and the
ONNX output name is ``{export.model_name}_{suffix}`` (the exporter owns the
prefix — writers never see the Athena name). The constants here are the
suffixes shared (or explicitly pinned as divergent) across modes, so the
v1 vertex-drift failure shape — one logical output named independently in
two files (``task.py:1004`` vs ``to_onnx.py:287``) — is structurally
unrepresentable: every cross-mode name decision lives in THIS module.

Two kinds of entries:

- **Shared constants** (`VERTEX_INDEX`): the SAME suffix in TEST and ONNX.
  The remaining v1 asymmetry (TEST writes the bare column while
  ``prefix_vertex_column=false``) is a *prefix policy* on the TaskWriter,
  not a second name — see ``modules.py`` and the amendment §5 rule 6.
- **Pinned divergences** (`OBJECT_INDEX`): v1 shipped two different
  suffixes for one logical output and byte-parity gating forces v2 to keep
  both — recorded as ONE explicit, documented pair instead of two
  accidental literals (amendment merge condition 4: "must not reproduce
  the vertex failure shape"). M5's ``MaskFormerObjectWriter`` MUST import
  `OBJECT_INDEX` — never re-declare the strings.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["OBJECT_INDEX", "VERTEX_INDEX", "ModeSplitSuffix", "pascal_case"]

VERTEX_INDEX = "VertexIndex"
"""The vertexing output suffix — SHARED by TEST and ONNX (amendment §2.2).

TEST: the `TaskWriter` column (bare while ``prefix_vertex_column=false``,
v1 byte parity ``task.py:1003``; ``{run_name}_VertexIndex`` once the M7
adjudication flips the flag). ONNX: ``{model_name}_VertexIndex`` (v1
``to_onnx.py:287-288``). One constant, one per-mode prefix rule, one compat
flag — the v1 drift class reduced to a declared, single-file property.
"""


@dataclass(frozen=True)
class ModeSplitSuffix:
    """A PINNED cross-mode suffix divergence (amendment merge condition 4).

    v1 shipped some logical outputs under DIFFERENT suffixes in eval and
    ONNX; byte-parity gates force v2 to reproduce both. Instead of two
    accidental literals in two files, the pair is declared once here with
    the divergence on record. When an adjudication unifies the names, the
    fix is this one declaration.

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
"""The MaskFormer track-to-object index suffix pair (M5 imports this).

`MaskFormerObjectWriter` (M5) uses ``OBJECT_INDEX.test`` for its eval
column and ``OBJECT_INDEX.onnx`` for its `ExportOutput` entry — the
amendment's §3 row obeying its own discipline (merge condition 4).
"""


def pascal_case(name: str) -> str:
    """Snake-case to Pascal-case — the default ONNX aux-output suffix rule.

    Reproduces v1's hand-built aux names for the goldens
    (``to_onnx.py:283-292``): ``track_origin -> TrackOrigin``,
    ``track_type -> TrackType``. Overridable per task via the
    `TaskWriter` ``onnx_names:`` mapping (amendment §2.2, merge
    condition 6).

    Returns
    -------
    str
        Each ``_``-separated part upper-cased at its first letter (the
        rest untouched, so ``track_pVtx`` stays ``TrackPVtx``-free:
        ``TrackPVtx`` would require an explicit ``onnx_names`` entry).
    """
    return "".join(part[:1].upper() + part[1:] for part in name.split("_"))
