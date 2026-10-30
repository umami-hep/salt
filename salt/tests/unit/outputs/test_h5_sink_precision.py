"""H5OutputSink dtype-policy coverage on the MAIN column path (input copies,
task columns, pad mask) — half_precision downcasts floats only, ints/bools
untouched (re-anchored from the retired ``test_half_precision_v1_flag`` e2e).

Also pins the legacy ``copy_inputs`` / ``write_pad_mask`` ctor kwargs: accepted
(and ignored) at their defaults so pre-round-5 saved configs still load, and
rejected with a `ConfigError` naming the replacement writer otherwise.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import torch

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.outputs import InputCopyWriter, PadMaskWriter
from salt.outputs.output_schema import OutputColumn
from salt.outputs.run_task_output import OutputSectionWriter
from salt.outputs.sinks.h5_sink import H5OutputSink
from salt.outputs.sinks.sink import Node, SinkContext

pytestmark = pytest.mark.cpu_always

_N = 128  # rows (>= the ftag H5Writer default chunk size of 100)
_T = 5  # file token length


@dataclass
class _Group:
    dataset: str
    global_object: bool


class _Reader:
    def __init__(self, source_path: Path) -> None:
        self.source_path = str(source_path)
        self.streams = ("jets", "tracks")
        self.groups = {
            "jets": _Group("jets", global_object=True),
            "tracks": _Group("tracks", global_object=False),
        }

    @property
    def h5_source(self) -> str:
        """Advertise the h5py-openable source (mirrors H5StructuredReader)."""
        return self.source_path


class _Dset:
    def __init__(self, reader: _Reader) -> None:
        self.reader = reader

    def __len__(self) -> int:
        return _N


class _DataModule:
    def __init__(self, reader: _Reader) -> None:
        self.test_dset = _Dset(reader)
        self.batch_size = _N
        self.test_suff = None


def _source_file(path: Path) -> np.ndarray:
    """A tiny source H5: float + int jets columns, per-token tracks group.

    Returns the jets structured array so the caller can seed matching
    bundle-sourced copy columns (input copies flow through the bundle, never
    indexed back out of the source file).
    """  # noqa: DOC201 - test helper, extra return note above
    rng = np.random.default_rng(11)
    jets = np.zeros(_N, dtype=np.dtype([("pt", "f4"), ("n_trks", "i8")]))
    jets["pt"] = rng.random(_N).astype("f4") * 100
    jets["n_trks"] = rng.integers(0, _T, _N)
    tracks = np.zeros((_N, _T), dtype=np.dtype([("d0", "f4")]))
    tracks["d0"] = rng.random((_N, _T)).astype("f4")
    with h5py.File(path, "w") as f:
        f.create_dataset("jets", data=jets)
        f.create_dataset("tracks", data=tracks)
    return jets


def _seed_columns(sink: H5OutputSink) -> None:
    """Two bundle-sourced copy columns (pt, n_trks) + one float prob column
    (jets) + one bare int column (tracks).
    """
    sink._columns = (  # noqa: SLF001 - OutputColumn is the sink's internal value object; seeded directly for this white-box test
        OutputColumn(key="outputs.jets.pt", suffixes=["pt"], dtype=None, prefix=False),
        OutputColumn(key="outputs.jets.n_trks", suffixes=["n_trks"], dtype=None, prefix=False),
        OutputColumn(key="outputs.jets.cls", suffixes=["pb", "pc"]),
        OutputColumn(key="outputs.tracks.vtx", suffixes=["VertexIndex"], dtype="i8", prefix=False),
    )
    sink._columns_resolved = True  # noqa: SLF001


def _section() -> dict[str, Any]:
    """The ``outputs:`` writers every driven sink binds: jets input copy + tracks pad mask."""
    return {
        "inputs_copy": InputCopyWriter(streams=["jets"], variables={"jets": ["pt", "n_trks"]}),
        "pad_mask": PadMaskWriter(streams=["tracks"]),
    }


def _drive(sink: H5OutputSink, tmp_path: Path, section: dict[str, Any]) -> Path:
    """Bind `section`, seed columns, then drive open_schema -> consume -> flush."""  # noqa: DOC201 - test helper, no Returns block per docstring policy
    src = tmp_path / "src.h5"
    jets_src = _source_file(src)
    out = Path(sink.output)
    # bind BEFORE seeding: binding resets the resolved column table
    sink.bind_output_section(section)
    _seed_columns(sink)
    ctx = SinkContext(
        run_name="salt",
        datamodule=_DataModule(_Reader(src)),
        ckpt_path=str(tmp_path / "e0-loss=0.1.ckpt"),
    )
    sink.open_schema(ctx)
    torch.manual_seed(3)
    mask = torch.zeros(_N, _T, dtype=torch.bool)
    mask[:, 3:] = True  # last two positions padded
    bundle = Bundle({
        "meta": {"rows": torch.tensor([0, _N])},
        "outputs": {
            "jets": {
                "cls": torch.rand(_N, 2),
                # bundle-sourced input copies: equal to the source file's values
                "pt": torch.from_numpy(jets_src["pt"].copy()),
                "n_trks": torch.from_numpy(jets_src["n_trks"].copy()),
            },
            # the vertexing leaf reaches the sink as the int-cast union-find
            # value (int32; -inf padding -> int32 min)
            "tracks": {
                "vtx": torch.full((_N, _T), -(2**31), dtype=torch.int32).masked_fill(~mask, 4),
                "mask": mask,
            },
        },
        "masks": {"tracks": mask},
    })
    sink.consume(bundle)
    sink.flush()
    assert out.exists()
    return out


def _run_sink(
    tmp_path: Path,
    *,
    half_precision: bool,
    legacy: dict[str, Any] | None = None,
    name: str | None = None,
) -> Path:
    """Build a sink (plus any `legacy` ctor kwargs) writing `name` and drive it."""  # noqa: DOC201 - test helper, no Returns block per docstring policy
    out = tmp_path / (name or ("half.h5" if half_precision else "full.h5"))
    sink = H5OutputSink(output=str(out), half_precision=half_precision, **(legacy or {}))
    return _drive(sink, tmp_path, _section())


def _read_all(path: Path) -> dict[str, np.ndarray]:
    """Every top-level dataset of an H5 file, read into memory."""  # noqa: DOC201 - test helper, no Returns block per docstring policy
    with h5py.File(path) as f:
        return {key: f[key][:] for key in f}


def _assert_same_h5(left: Path, right: Path) -> None:
    """Same dataset names, dtypes and values (incl. the 'jets' / 'tracks' groups)."""
    a, b = _read_all(left), _read_all(right)
    assert sorted(a) == sorted(b)
    assert {"jets", "tracks"} <= set(a)
    for key, arr in a.items():
        assert arr.dtype == b[key].dtype, key
        assert np.array_equal(arr, b[key]), key


def test_half_precision_downcasts_floats_only(tmp_path):
    """half_precision: float copies AND prob columns -> f2; int/bool untouched."""
    out = _run_sink(tmp_path, half_precision=True)
    with h5py.File(out) as f:
        jets_dtype, tracks_dtype = f["jets"].dtype, f["tracks"].dtype
    assert jets_dtype["pt"] == np.dtype("f2")  # float input copy downcast
    assert jets_dtype["salt_pb"] == np.dtype("f2")  # prob column downcast
    assert jets_dtype["n_trks"] == np.dtype("i8")  # int copy untouched
    assert tracks_dtype["VertexIndex"] == np.dtype("i8")  # int task column untouched
    assert tracks_dtype["mask"] == np.dtype("?")  # bool pad mask untouched


def test_full_precision_keeps_f4(tmp_path):
    """Default (full) precision: the v1 f4 policy on floats, ints/bools as-is."""
    out = _run_sink(tmp_path, half_precision=False)
    with h5py.File(out) as f:
        jets_dtype, tracks_dtype = f["jets"].dtype, f["tracks"].dtype
        tracks = f["tracks"][:]
    assert jets_dtype["pt"] == np.dtype("f4")
    assert jets_dtype["salt_pb"] == np.dtype("f4")
    assert jets_dtype["n_trks"] == np.dtype("i8")
    assert tracks_dtype["VertexIndex"] == np.dtype("i8")
    assert tracks_dtype["mask"] == np.dtype("?")
    # value spot-checks through the packing path: padded sentinel + mask polarity
    padded = tracks["mask"]
    assert (padded == np.tile([False] * 3 + [True] * 2, (_N, 1))).all()
    assert (tracks["VertexIndex"][padded] == np.int64(-(2**31))).all()
    assert (tracks["VertexIndex"][~padded] == 4).all()


def test_legacy_kwargs_at_defaults_are_accepted_and_ignored(tmp_path):
    """copy_inputs=None / write_pad_mask=False (the saved-config values) are no-ops."""
    # signature order is the saved-config dump order: the legacy kwargs stay first
    params = list(inspect.signature(H5OutputSink.__init__).parameters)
    assert params[1:4] == ["copy_inputs", "write_pad_mask", "output"]
    sink = H5OutputSink(copy_inputs=None, write_pad_mask=False, output=str(tmp_path / "x.h5"))
    # ignored: the section-driven state stays at its unbound defaults
    assert sink.copy_inputs == {}
    assert sink.write_pad_mask is False
    legacy = _run_sink(
        tmp_path,
        half_precision=False,
        legacy={"copy_inputs": None, "write_pad_mask": False},
        name="legacy.h5",
    )
    plain = _run_sink(tmp_path, half_precision=False, name="plain.h5")
    _assert_same_h5(legacy, plain)


def test_legacy_saved_config_round_trip(tmp_path):
    """A pre-round-5 saved ``outputs:`` block (legacy null/false keys) parses,
    dumps/re-parses identically, instantiates, and writes the same H5 as a
    sink built without the legacy keys.
    """
    from jsonargparse import ArgumentParser

    rt = tmp_path / "rt.h5"
    saved = f"""
outputs:
  inputs_copy:
    class_path: salt.outputs.input_copy_writer.InputCopyWriter
    init_args:
      streams: [jets]
      variables:
        jets: [pt, n_trks]
  pad_mask:
    class_path: salt.outputs.pad_mask_writer.PadMaskWriter
    init_args:
      streams: [tracks]
  h5_output:
    class_path: salt.outputs.sinks.h5_sink.H5OutputSink
    init_args:
      copy_inputs: null
      write_pad_mask: false
      output: '{rt}'
"""
    parser = ArgumentParser(exit_on_error=False)
    # the same type as salt/main.py's top-level `outputs:` section argument
    parser.add_argument(
        "--outputs",
        type=dict[str, OutputSectionWriter | Node | None] | None,
        default=None,
    )
    cfg = parser.parse_string(saved)
    h5_args = cfg.outputs["h5_output"].init_args
    assert h5_args.copy_inputs is None
    assert h5_args.write_pad_mask is False
    dumped = parser.dump(cfg)
    reparsed = parser.parse_string(dumped)
    assert parser.dump(reparsed) == dumped
    assert reparsed.outputs["h5_output"].init_args.copy_inputs is None
    assert reparsed.outputs["h5_output"].init_args.write_pad_mask is False
    # instantiate exactly as the CLI does; the section bound is the writers only
    outputs = parser.instantiate_classes(reparsed).outputs
    sink = outputs["h5_output"]
    assert isinstance(sink, H5OutputSink)
    assert sink.copy_inputs == {}
    assert sink.write_pad_mask is False
    section = {key: w for key, w in outputs.items() if not isinstance(w, Node)}
    assert list(section) == ["inputs_copy", "pad_mask"]
    out = _drive(sink, tmp_path, section)
    assert out == rt
    plain = _run_sink(tmp_path, half_precision=False, name="plain.h5")
    _assert_same_h5(out, plain)


@pytest.mark.parametrize("value", [{"jets": ["pt"]}, {}])
def test_legacy_copy_inputs_non_default_raises(tmp_path, value):
    """Anything but None for copy_inputs is rejected, naming InputCopyWriter."""
    with pytest.raises(ConfigError, match="InputCopyWriter"):
        H5OutputSink(copy_inputs=value, output=str(tmp_path / "x.h5"))


@pytest.mark.parametrize("value", [True, ["tracks"]])
def test_legacy_write_pad_mask_non_default_raises(tmp_path, value):
    """Anything but False for write_pad_mask is rejected, naming PadMaskWriter."""
    with pytest.raises(ConfigError, match="PadMaskWriter"):
        H5OutputSink(write_pad_mask=value, output=str(tmp_path / "x.h5"))
