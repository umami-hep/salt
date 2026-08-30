"""One model, two readers — the ``ttbar_vs_hh4b_event_tagger`` cross-format gate.

``ttbar_vs_hh4b_event_tagger.yaml`` declares no reader; it is paired with exactly
one of ``readers/easyjet_events.yaml`` or ``readers/physlite_events.yaml``. The
whole claim of that arrangement is that the two fragments are interchangeable, so
this module gates the claim from both sides:

**Statically** (no data at all) the two fragments must instantiate and must serve
the same stream names, the same field names and the same jaggedness — and between
them must cover every variable the model declares. A drift on either side is a
config error, not a training failure discovered hours later.

**Dynamically** the easyjet leg trains — as matrix row 6 (`event_tagger_easyjet`)
in ``pipeline.py``/``test_pipeline.py``, on the same synthetic
``AnalysisMiniTree`` fixture this module's static gates use. PHYSLITE
deliberately has no dynamic leg, here or in the matrix: its xAOD POOL layout
(``ElementLink`` structs resolving into separate containers) is not worth
faking, so a real file is needed and the static contract below is what CI can
hold. That asymmetry is recorded here rather than left as a silent gap.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from salt.main import CONFIG_DIR

pytestmark = pytest.mark.cpu_always

MODEL = "ttbar_vs_hh4b_event_tagger"
EASYJET = "readers/easyjet_events"
PHYSLITE = "readers/physlite_events"

# Truth-level ttbar heavy-flavour categorisation. easyjet writes it only for the
# DSIDs in `DSID_HF_class_samples` (601237 is listed, 603404 is not) and the ttbar
# PHYSLITE carries `EventInfoAuxDyn.HF_Classification`, so on a ttbar-vs-HH4b
# problem the branch's mere PRESENCE is the class label. Never a feature.
LABEL_LEAK_BRANCHES = ("HF_Classification", "HF_SimpleClassification")


def _load(config: str) -> dict:
    """A shipped config, include-expanded."""
    from salt.config_utils import expand_includes  # noqa: PLC0415

    path = CONFIG_DIR / f"{config}.yaml"
    assert path.is_file(), f"missing shipped config {path}"
    return yaml.safe_load(Path(expand_includes(str(path))).read_text()) or {}


def _reader_node(config: str) -> dict:
    return _load(config)["data"]["modules"]["reader"]


def _instantiate_reader(config: str):
    """Build the fragment's reader through jsonargparse, exactly as the CLI does.

    This is the gate a reader fragment gets INSTEAD of ``graph validate``: a
    fragment has no model, so there is no plan to compile, but every invariant a
    reader enforces (`unroll` naming a scalar group, link/target pairing,
    constituent cuts on a jagged stream naming configured branches, ...) is
    raised from its constructor and is caught here.
    """
    from jsonargparse import ArgumentParser  # noqa: PLC0415

    from salt.data.base import Reader  # noqa: PLC0415

    parser = ArgumentParser(exit_on_error=False)
    parser.add_subclass_arguments(Reader, "reader")
    cfg = parser.parse_object({"reader": _reader_node(config)})
    return parser.instantiate_classes(cfg).reader


def _served(reader) -> dict[str, tuple[tuple[str, ...], bool]]:
    """``{stream: (field names, jagged)}`` for a MultiSampleReader's sub-readers.

    Every sub-reader must agree (``MultiSampleReader.prepare`` enforces it on real
    data); asserted here so a mismatch is caught with no file present. Reads
    ``served_branches``, not ``branches``, so a sub-reader that reaches some of
    its fields through a 1:1 ElementLink join still has to serve the same names
    in the same order as one that reads them flat — that equivalence is the whole
    reason the join exists.
    """
    per_sample = [
        {
            stream: (tuple(cfg.served_branches), cfg.jagged)
            for stream, cfg in sample.reader.groups.items()
        }
        for sample in reader.samples
    ]
    assert per_sample, "reader declares no samples"
    first = per_sample[0]
    for other in per_sample[1:]:
        assert other == first, f"sub-readers disagree on the served schema: {first} vs {other}"
    return first


# ---------------------------------------------------------------- static gates


@pytest.mark.parametrize("fragment", [EASYJET, PHYSLITE])
def test_fragment_is_a_pure_reader(fragment):
    """A reader fragment declares a reader and NO model — it is a component."""
    raw = _load(fragment)
    assert "reader" in raw["data"]["modules"], f"{fragment} declares no reader"
    assert "model" not in raw, (
        f"{fragment} declares a model. A reader fragment is a component: giving it "
        "a model (or an include: pointing at one) so that it validates standalone "
        "reverse-engineers the test instead of modelling the domain."
    )


def test_model_declares_no_reader():
    """The model half names no format — that is what makes the pairing meaningful."""
    raw = _load(MODEL)
    assert "model" in raw, f"{MODEL} declares no model"
    assert "reader" not in raw["data"]["modules"], (
        f"{MODEL} declares a reader. It is the model half of a two-part stack; "
        "binding it to one format defeats the whole demonstration."
    )


def test_both_fragments_serve_the_same_contract():
    """easyjet and PHYSLITE serve identical stream names, field names, jaggedness.

    This is the entire claim of the one-model-two-readers arrangement, and it is
    checkable with no data on either side.
    """
    pytest.importorskip("awkward", reason="reader fragments need `pip install 'salt[root]'`")
    easyjet = _served(_instantiate_reader(EASYJET))
    physlite = _served(_instantiate_reader(PHYSLITE))

    assert set(easyjet) == set(physlite), (
        "the two fragments serve different streams — the model cannot bind to both.\n"
        f"  easyjet : {sorted(easyjet)}\n  physlite: {sorted(physlite)}"
    )
    for stream in sorted(easyjet):
        ej_fields, ej_jagged = easyjet[stream]
        pl_fields, pl_jagged = physlite[stream]
        assert ej_jagged == pl_jagged, (
            f"stream {stream!r}: jaggedness differs (easyjet={ej_jagged}, physlite={pl_jagged})"
        )
        # PHYSLITE additionally reads its cut variables (NNJvtPass), which easyjet
        # does not need because the ntupler already applied that selection. So the
        # easyjet field set must be a SUBSET, not equal.
        missing = set(ej_fields) - set(pl_fields)
        assert not missing, (
            f"stream {stream!r}: PHYSLITE does not serve {sorted(missing)}, which "
            "easyjet does — the model would bind on one leg and not the other"
        )


def test_model_variables_are_served_by_both_fragments():
    """Every variable the model consumes exists on both legs."""
    pytest.importorskip("awkward", reason="reader fragments need `pip install 'salt[root]'`")
    variables = _load(MODEL)["data"]["modules"]["features"]["init_args"]["variables"]
    for fragment in (EASYJET, PHYSLITE):
        served = _served(_instantiate_reader(fragment))
        for stream, wanted in variables.items():
            assert stream in served, f"{fragment} serves no {stream!r} stream"
            missing = sorted(set(wanted) - set(served[stream][0]))
            assert not missing, f"{fragment}: {stream} is missing {missing}"


def test_no_shipped_config_uses_a_label_leaking_branch():
    """`HF_Classification`/`HF_SimpleClassification` appear in no shipped config.

    They exist on the ttbar side only, so their presence alone IS the class label
    on this problem. This is a correctness gate, not a style preference.
    """
    # a config may DISCUSS the branch in a comment (this pair is documented in
    # ttbar_vs_hh4b_event_tagger.yaml's header); only a non-comment line counts
    offenders = [
        (path.relative_to(CONFIG_DIR).as_posix(), line.strip())
        for path in sorted(CONFIG_DIR.rglob("*.yaml"))
        for line in path.read_text().splitlines()
        if any(branch in line.split("#", 1)[0] for branch in LABEL_LEAK_BRANCHES)
    ]
    assert not offenders, f"label-leaking branch named in a shipped config: {offenders}"

# The dynamic gate — the model + easyjet fragment training end-to-end on the
# synthetic minitree pair — now lives as matrix row 6 (`event_tagger_easyjet`)
# in pipeline.py/test_pipeline.py (`_paired_root_context`, same
# write_sample_pair/write_sourced_fragment fixtures). No dynamic PHYSLITE leg
# exists there either, for the same reason given in the module docstring.
