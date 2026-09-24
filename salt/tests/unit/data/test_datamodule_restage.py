"""Datamodule-level restaging: move_files_temp -> _stage -> restage -> teardown."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from salt.data import Features, InputSamples, Labels, SaltDataModule, UprootReader
from salt.graph.spec import Mode

uproot = pytest.importorskip("uproot")
awkward = pytest.importorskip("awkward")

from salt.tests._fixtures.easyjet_minitree import (  # noqa: E402
    TREE_NAME,
    build_fixture_arrays,
    write_minitree,
)

FIT_SINKS = ["inputs.jets", "masks.jets", "labels.jets.HadronConeExclTruthLabelID"]
TEST_SINKS = ["inputs.jets", "masks.jets", "meta.rows"]
SINKS = {Mode.FIT: FIT_SINKS, Mode.VAL: FIT_SINKS, Mode.TEST: TEST_SINKS}


@pytest.fixture
def two_file_dir(tmp_path: Path) -> Path:
    """A directory with two deterministic easyjet minitrees, `a.root` + `b.root`."""
    src = tmp_path / "src"
    src.mkdir()
    write_minitree(src / "a.root", build_fixture_arrays(seed=99))
    write_minitree(src / "b.root", build_fixture_arrays(seed=7))
    return src


def _reader_proto() -> UprootReader:
    """A jets(jagged)+event(scalar) UprootReader prototype — no filename, no unroll."""
    return UprootReader(
        groups={
            "jets": {
                "branches": {
                    "pt": "recojet_antikt4PFlow_pt_NOSYS",
                    "eta": "recojet_antikt4PFlow_eta",
                    "HadronConeExclTruthLabelID": (
                        "recojet_antikt4PFlow_HadronConeExclTruthLabelID"
                    ),
                },
                "jagged": True,
                "pad_max": 8,
            },
            "event": {"branches": {"eventNumber": "eventNumber"}, "jagged": False},
        },
        tree=TREE_NAME,
        unroll=None,
    )


def make_dm(src_dir: Path, stage_root: str | None) -> SaltDataModule:
    """A datamodule over `src_dir` with optional `move_files_temp` staging."""
    return SaltDataModule(
        modules={
            "input_samples": InputSamples(
                files={"train": src_dir, "val": src_dir, "test": src_dir}
            ),
            "reader": _reader_proto(),
            "features": Features(variables={"jets": ["pt", "eta"]}),
            "labels": Labels(dtype_policy="int64-for-int"),
        },
        sinks=SINKS,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        move_files_temp=stage_root,
    )


def _assert_batches_equal(a, b, path: str = "batch") -> None:
    """Recursively assert two nested batch dicts are byte-identical leaf-for-leaf."""
    assert set(a) == set(b), path
    for key, av in a.items():
        bv = b[key]
        sub = f"{path}.{key}"
        if isinstance(av, dict):
            _assert_batches_equal(av, bv, sub)
            continue
        assert av.dtype == bv.dtype, sub
        assert av.shape == bv.shape, sub
        assert av.numpy().tobytes() == bv.numpy().tobytes(), sub


def test_fit_setup_restages_both_stage_readers_under_one_digest_dir(two_file_dir, tmp_path):
    """setup('fit') stages both train+val readers under one shared 16-hex digest dir."""
    root = tmp_path / "stage"
    dm = make_dm(two_file_dir, str(root))
    dm.setup("fit")
    parents = set()
    for ds in (dm.train_dset, dm.val_dset):
        srcs = ds.reader.sources()
        assert len(srcs) == 2
        assert {p.name for p in srcs} == {"a.root", "b.root"}
        for p in srcs:
            assert root in p.parents
            assert (p.parent / f"{p.name}.done").exists()
        parent = {p.parent for p in srcs}
        assert len(parent) == 1
        parent = parent.pop()
        assert len(parent.name) == 16
        int(parent.name, 16)  # parses as a hex digest
        parents.add(parent)
    assert len(parents) == 1  # train and val resolve to the SAME digest dir
    for name in ("a.root", "b.root"):
        orig = two_file_dir / name
        staged = next(p for p in dm.train_dset.reader.sources() if p.name == name)
        assert orig.exists()
        assert orig.stat().st_size == staged.stat().st_size


def test_staged_batches_are_byte_identical_to_unstaged(two_file_dir, tmp_path):
    dm_s = make_dm(two_file_dir, str(tmp_path / "stage"))
    dm_u = make_dm(two_file_dir, None)
    dm_s.setup("fit")
    dm_u.setup("fit")
    for ds_s, ds_u in ((dm_s.train_dset, dm_u.train_dset), (dm_s.val_dset, dm_u.val_dset)):
        for rows in (np.s_[0:6], np.s_[6:12]):
            _assert_batches_equal(ds_s[rows], ds_u[rows])


def test_stage_is_identity_when_move_files_temp_is_none(two_file_dir):
    """No `move_files_temp` root -> `_stage` returns the reader object unchanged."""
    dm = make_dm(two_file_dir, None)
    assert dm._resolve_stage_root("fit") is None
    r = dm._reader_proto.with_source(filename=two_file_dir)
    assert dm._stage(r) is r


def test_test_stage_is_never_staged(two_file_dir, tmp_path):
    """`setup('test')` never stages, even when `move_files_temp` is configured."""
    root = tmp_path / "stage"
    dm = make_dm(two_file_dir, str(root))
    assert dm._resolve_stage_root("test") is None
    dm.setup("test")
    srcs = dm.test_dset.reader.sources()
    assert srcs
    for p in srcs:
        assert two_file_dir in p.parents
    assert not root.exists() or not list(root.rglob("*.root"))


def test_teardown_fit_removes_the_stage_root(two_file_dir, tmp_path):
    """`teardown('fit')` removes the staging root; `teardown('test')` never does."""
    root = tmp_path / "stage"
    dm = make_dm(two_file_dir, str(root))
    dm.setup("fit")
    assert root.exists()
    assert len(list(root.rglob("*.root"))) == 2
    dm.teardown("test")
    assert root.exists()  # teardown("test") must not remove the fit staging root
    dm.teardown("fit")
    assert not root.exists()
