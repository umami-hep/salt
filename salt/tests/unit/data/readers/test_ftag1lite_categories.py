"""Tests for the FTAG1LITE container-name categories, `ContainerOneHot` and the reader's
injected category code columns.
"""

from __future__ import annotations

import numpy as np
import pytest

from salt.data import ContainerOneHot
from salt.data.base import WorkerCtx
from salt.data.readers import ftag1lite_categories
from salt.data.readers.ftag1lite_categories import (
    MERGE_RTAGS,
    RTAG_SUBCAMPAIGN,
    SHOWERS,
    SIM_TYPES,
    SUBCAMPAIGNS,
    category_codes,
    onehot_fields,
    parse_container_categories,
)
from salt.graph.errors import ConfigError
from salt.graph.spec import Mode, flatten_spec
from salt.tests._fixtures.joint_ftag1lite_corpus import CONTAINERS, build_joint_corpus, make_reader

_W = "mc20_13TeV.361100.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Wplusenu.deriv.DAOD_FTAG1LITE.e3601"
_T = "mc23_13p6TeV.601348.PhPy8EG_tb_lep_antitop.deriv.DAOD_FTAG1LITE.e8514"
_Z = "mc23_13p6TeV.601190.PhPy8EG_AZNLO_Zmumu.deriv.DAOD_FTAG1LITE.e8514"
_SH = "mc20_13TeV.700320.Sh_2211_Zee_maxHTpTV2_BFilter.deriv.DAOD_FTAG1LITE.e8351_e7400"
_DM = "mc23_13p6TeV.801991.Py8EG_A14NNPDF23LO_Dimuon_12_3_bbAndcc.deriv.DAOD_FTAG1LITE.e8591"

# one real-style DID per table r-tag: (DID, r-tag, subcampaign)
SINGLE_RTAG_DIDS = [
    (f"{_W}_s3681_r13167_p7356", "r13167", "mc20a"),
    (f"{_W}_s3681_r13144_p7356", "r13144", "mc20d"),
    (f"{_W}_s3681_r13145_p7356", "r13145", "mc20e"),
    (f"{_T}_s4162_r14622_p7356", "r14622", "mc23a"),
    (f"{_T}_s4159_r15224_p7356", "r15224", "mc23d"),
    (f"{_Z}_s4162_r15540_p7356", "r15540", "mc23a"),
    (f"{_Z}_s4159_r15530_p7356", "r15530", "mc23d"),
    (f"{_Z}_s4369_r16083_p7356", "r16083", "mc23e"),
]

# two-r-tag DIDs (reco first, merge second) and an a-tag before an s-tag
MULTI_TAG_DIDS = [
    (f"{_SH}_s3681_r13167_r13146_p7356", ("mc20a", "Sherpa", "FullSim")),
    (f"{_DM}_e8586_a910_s4114_r15540_r15516_p7356", ("mc23a", "Pythia8", "AF3")),
    (f"{_DM}_e8586_a934_s4370_r16083_r15970_p7356", ("mc23e", "Pythia8", "AF3")),
]

SHOWER_CASES = [
    ("601190.PhPy8EG_AZNLO_Zmumu", "Pythia8"),
    ("364700.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ0WithSW", "Pythia8"),
    ("364541.Sherpa_222_NNPDF30NNLO_SinglePhoton_pty_17_35", "Sherpa"),
    ("999001.H7EG_synthetic_ttbar", "Herwig7"),
    ("999003.MGPy8EG_synthetic_other", "other"),
    ("999004.PhPy8EGx_synthetic_other", "other"),
]

FIELDS_11 = (
    *(f"subcampaign_{s}" for s in ("mc20a", "mc20d", "mc20e", "mc23a", "mc23c", "mc23d", "mc23e")),
    *(f"shower_{s}" for s in ("Pythia8", "Herwig7", "Sherpa", "other")),
)
FIELDS_13 = (*FIELDS_11, "sim_type_FullSim", "sim_type_AF3")

UNKNOWN_RTAG = "mc23_13p6TeV.601229.PhPy8EG_A14_ttbar.deriv.DAOD_FTAG1LITE.e8514_s4159_r99999_p7356"
DATA_DID = "data23_13p6TeV.00456789.physics_Main.deriv.DAOD_FTAG1LITE.r14765_p7356"
LOCAL_DIR = "my_local_ftag1lite_folder"

# the fixture containers, written out by hand: (subcampaign, shower, sim type)
EXPECTED = {
    CONTAINERS[0]: ("mc23d", "Pythia8", "FullSim"),
    CONTAINERS[1]: ("mc23a", "Pythia8", "FullSim"),
}

_CODE_DEMAND = {"jets": dict.fromkeys(("subcampaign", "shower", "sim_type"), "t")}


def _forms(did: str) -> list[str]:
    """The DIAS ``<scope>__<did>``, local ``<scope>_<did>`` and bare forms of `did`."""
    scope = did.split(".")[0]
    return [f"{scope}__{did}", f"{scope}_{did}", did]


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("form", [0, 1, 2], ids=["dias", "local", "bare"])
@pytest.mark.parametrize(("did", "rtag", "subcampaign"), SINGLE_RTAG_DIDS)
def test_every_table_rtag_maps_in_every_directory_form(
    form: int, did: str, rtag: str, subcampaign: str
) -> None:
    """Each table r-tag gives its subcampaign for DIAS, local and bare names."""
    name = _forms(did)[form]
    cats = parse_container_categories(f"/data/corpus/{name}")
    assert (cats.name, cats.rtag, cats.subcampaign) == (name, rtag, subcampaign)
    assert (cats.shower, cats.sim_type) == ("Pythia8", "FullSim")
    assert (cats.dsid, cats.physics_short) == tuple(did.split(".")[1:3])


def test_table_covers_exactly_the_accepted_rtags() -> None:
    """The r-tag table is the accepted one and no merge tag is in it."""
    assert {d[1]: d[2] for d in SINGLE_RTAG_DIDS} == RTAG_SUBCAMPAIGN
    assert MERGE_RTAGS == {"r13146", "r15514", "r15516", "r15970"}
    assert not MERGE_RTAGS & set(RTAG_SUBCAMPAIGN)


@pytest.mark.parametrize(("did", "expected"), MULTI_TAG_DIDS)
def test_first_rtag_and_first_sim_tag_decide(did: str, expected: tuple[str, ...]) -> None:
    """Two-r-tag DIDs use the reco tag; the first a/s tag sets the simulation type."""
    cats = parse_container_categories(_forms(did)[0], require_sim_type=True)
    assert (cats.subcampaign, cats.shower, cats.sim_type) == expected


@pytest.mark.parametrize(
    ("name", "token"),
    [
        (f"mc20_13TeV__{_W}_s3681_r99999_p7356", "r99999"),
        (f"mc20_13TeV_{_SH}_s3681_r13146_p7356", "merge tag"),
        ("user.someone.not_a_container", "DSID"),
        ("mc20_13TeV.361100.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Wplusenu", "AMI tag"),
        ("mc20_13TeV.361100..deriv.DAOD_FTAG1LITE.e3601_s3681_r13144_p7356", "physics-short"),
        (f"{_W}_s3681_p7356", "e3601_s3681_p7356"),
    ],
    ids=["unknown-rtag", "merge-first", "no-dsid", "no-ami-tags", "no-physics-short", "no-rtag"],
)
def test_bad_names_raise_naming_container_and_token(name: str, token: str) -> None:
    """Unknown first r-tag, merge tag first or a malformed name is a ConfigError."""
    with pytest.raises(ConfigError) as exc:
        parse_container_categories(name)
    assert name in str(exc.value)
    assert token in str(exc.value)


@pytest.mark.parametrize(("physics", "shower"), SHOWER_CASES)
def test_shower_from_first_physics_short_token(physics: str, shower: str) -> None:
    """The first physics-short token maps by exact match; anything else is other."""
    did = f"mc23_13p6TeV.{physics}.deriv.DAOD_FTAG1LITE.e8514_s4162_r15540_p7356"
    assert parse_container_categories(f"mc23_13p6TeV__{did}").shower == shower


def test_missing_sim_tag_is_none_unless_required() -> None:
    """No a/s tag gives sim_type None (code -1), or a ConfigError when it is required."""
    name = f"{_Z}_r15540_p7356"
    cats = parse_container_categories(name)
    assert cats.sim_type is None
    assert category_codes(cats)["sim_type"] == -1
    with pytest.raises(ConfigError, match="simulation tag") as exc:
        parse_container_categories(name, require_sim_type=True)
    assert name in str(exc.value)


def test_onehot_fields_order_and_lengths() -> None:
    """11 names without the sim block, 13 with it, in the fixed order."""
    assert onehot_fields(False) == FIELDS_11
    assert onehot_fields(True) == FIELDS_13
    assert (len(FIELDS_11), len(FIELDS_13)) == (11, 13)


# --------------------------------------------------------------------------- #
# ContainerOneHot
# --------------------------------------------------------------------------- #


class _Batch:
    """The minimal bundle `process` reads: ``get(key)``."""

    def __init__(self, data: dict[str, np.ndarray]) -> None:
        self.data = data

    def get(self, key: str) -> np.ndarray:
        return self.data[key]


def _raw(names: list[str]) -> _Batch:
    """A structured ``raw.jets`` batch carrying each container's codes."""
    raw = np.zeros(len(names), dtype=[("pt", "f4"), *((f, "i8") for f in _CODE_DEMAND["jets"])])
    for i, name in enumerate(names):
        for col, code in category_codes(parse_container_categories(name)).items():
            raw[col][i] = code
    return _Batch({"raw.jets": raw})


def _module(include_sim: bool = False) -> ContainerOneHot:
    module = ContainerOneHot(include_sim_type=include_sim)
    module.name = "container_onehot"
    return module


@pytest.mark.parametrize("include_sim", [False, True])
@pytest.mark.parametrize("mode", [Mode.FIT, Mode.VAL, Mode.TEST])
def test_declare_io_and_read_fields(mode: Mode, include_sim: bool) -> None:
    """``raw.jets`` -> ``inputs.container_onehot`` [B, F] with named columns in every mode."""
    module = _module(include_sim)
    io = module.declare_io(mode)
    spec = flatten_spec(io.produces)["inputs.container_onehot"]
    fields = FIELDS_13 if include_sim else FIELDS_11
    assert spec.fields == fields == module.fields
    assert (spec.shape, spec.dtype) == (("B", len(fields)), "float32")
    assert spec.active_in(mode)
    assert set(flatten_spec(io.requires)) == {"raw.jets"}
    codes = ["subcampaign", "shower", *(["sim_type"] if include_sim else [])]
    assert module.read_fields(None) == {"jets": dict.fromkeys(codes, "container_onehot")}


@pytest.mark.parametrize("include_sim", [False, True])
def test_process_builds_one_hot_rows(include_sim: bool) -> None:
    """Each row gets exactly one 1 per block, at its container's categories."""
    names = [f"mc20_13TeV__{MULTI_TAG_DIDS[0][0]}", f"mc23_13p6TeV_{MULTI_TAG_DIDS[1][0]}"]
    names = [*names, names[1], names[0]]
    out = _module(include_sim).process(_raw(names), slice(0, 4), Mode.FIT)
    onehot = out["inputs.container_onehot"]
    fields = FIELDS_13 if include_sim else FIELDS_11
    assert onehot.dtype == np.float32
    assert onehot.shape == (4, len(fields))
    for row, name in zip(onehot, names, strict=True):
        cats = parse_container_categories(name)
        expected = np.zeros(len(fields), dtype=np.float32)
        expected[fields.index(f"subcampaign_{cats.subcampaign}")] = 1.0
        expected[fields.index(f"shower_{cats.shower}")] = 1.0
        if include_sim:
            expected[fields.index(f"sim_type_{cats.sim_type}")] = 1.0
        np.testing.assert_array_equal(row, expected, err_msg=name)


@pytest.mark.parametrize(
    ("col", "bad", "include_sim"),
    [
        ("subcampaign", len(SUBCAMPAIGNS), False),
        ("shower", -1, False),
        ("sim_type", -1, True),
        ("sim_type", len(SIM_TYPES), True),
    ],
)
def test_out_of_range_codes_raise_value_error(col: str, bad: int, include_sim: bool) -> None:
    """A code outside its vocabulary (a -1 sim code with the sim block) is a ValueError."""
    batch = _raw([f"mc23_13p6TeV__{SINGLE_RTAG_DIDS[5][0]}"] * 2)
    batch.data["raw.jets"][col][1] = bad
    with pytest.raises(ValueError, match=rf"jets\.{col}") as exc:
        _module(include_sim).process(batch, slice(0, 2), Mode.FIT)
    assert f"[{bad}]" in str(exc.value)


def test_missing_sim_code_is_ignored_without_the_sim_block() -> None:
    """``sim_type == -1`` is fine when the sim block is off."""
    batch = _raw([f"{_Z}_r15540_p7356"])
    out = _module(False).process(batch, slice(0, 1), Mode.FIT)["inputs.container_onehot"]
    assert out.sum() == 2
    assert out[0, len(SUBCAMPAIGNS) + SHOWERS.index("Pythia8")] == 1


# --------------------------------------------------------------------------- #
# reader-injected code columns on the synthetic corpus
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def corpus_root(tmp_path_factory):  # noqa: ANN001, ANN201
    pytest.importorskip("uproot")
    pytest.importorskip("awkward")
    return build_joint_corpus(tmp_path_factory.mktemp("categories_corpus")).root


@pytest.mark.parametrize("jets", ["both", "small_r"])
def test_reader_injects_each_files_container_codes(corpus_root, jets: str) -> None:  # noqa: ANN001
    """Every served row carries its own file's container codes; fingerprint omits them."""
    reader = make_reader(corpus_root, jets)
    reader.prepare()
    reader.bind(WorkerCtx(mode=Mode.FIT, read_fields=_CODE_DEMAND, seed=0))
    blocks = reader.row_blocks()
    assert {reader.group_names[b.group] for b in blocks} == set(CONTAINERS)
    for block in blocks:
        name = reader.group_names[block.group]
        jets_raw = reader.read_block(block, Mode.FIT)["raw.jets"]
        expected = category_codes(parse_container_categories(name))
        sub, shower, sim = EXPECTED[name]
        assert expected == {
            "subcampaign": SUBCAMPAIGNS.index(sub),
            "shower": SHOWERS.index(shower),
            "sim_type": SIM_TYPES.index(sim),
        }
        assert len(jets_raw) == block.n_rows
        for col, code in expected.items():
            assert jets_raw[col].dtype == np.dtype("int64")
            assert (jets_raw[col] == code).all(), f"{name} {col}"
    fp = reader.config_fingerprint()
    assert "special_fields" not in fp
    keys = "reader jets jet_type_field small_type large_type num stage row_blocks sides"
    assert set(fp) == set(keys.split())


def _mixed_corpus(tmp_path):  # noqa: ANN001, ANN201
    """A corpus of one data DID, one unknown-r-tag DID and one non-DID folder."""
    pytest.importorskip("uproot")
    pytest.importorskip("awkward")
    return build_joint_corpus(
        tmp_path / "mixed", containers=(DATA_DID, UNKNOWN_RTAG, LOCAL_DIR)
    ).root


@pytest.mark.parametrize("jets", ["small_r", "both"])
def test_unparseable_containers_read_without_code_demand(tmp_path, monkeypatch, jets: str) -> None:  # noqa: ANN001
    """Without a code demand nothing is parsed: data / unknown-r-tag / non-DID dirs read fine."""
    root = _mixed_corpus(tmp_path)

    def _never(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError(f"parse_container_categories called: {args} {kwargs}")

    monkeypatch.setattr(ftag1lite_categories, "parse_container_categories", _never)
    reader = make_reader(root, jets)
    reader.prepare()
    assert len(reader) > 0
    assert set(reader.group_names) == {DATA_DID, UNKNOWN_RTAG, LOCAL_DIR}
    reader.bind(WorkerCtx(mode=Mode.FIT, read_fields={}, seed=0))
    for block in reader.row_blocks():
        jets_raw = reader.read_block(block, Mode.FIT)["raw.jets"]
        assert len(jets_raw) == block.n_rows
        names = set(jets_raw.dtype.names)
        assert not names & {"subcampaign", "shower", "sim_type"}
        assert ("jet_type" in names) == (jets == "both")


def test_unparseable_containers_fail_once_when_codes_demanded(tmp_path) -> None:  # noqa: ANN001
    """With a code demand, bind raises ONE ConfigError naming every unparseable container."""
    reader = make_reader(_mixed_corpus(tmp_path), "small_r")
    reader.prepare()
    with pytest.raises(ConfigError) as exc:
        reader.bind(WorkerCtx(mode=Mode.FIT, read_fields=_CODE_DEMAND, seed=0))
    msg = str(exc.value)
    assert "3 container(s)" in msg
    assert "r99999" in msg
    for name in (DATA_DID, UNKNOWN_RTAG, LOCAL_DIR):
        assert name in msg
