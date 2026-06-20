"""One-batch SHAPE PROBE for the salt v2 graph pipeline (design §4.3 annotation).

``salt2 graph plot`` renders a compiled plan with the DECLARED (symbolic) shape
on every port-card row — ``inputs.tracks (B, T:tracks, 19)``. The probe runs ONE
real batch through the compiled plan and records every bundle leaf's CONCRETE
torch shape, so the renderer can show ``inputs.tracks (16, 40, 19)`` instead.
``--probe`` wires this into ``salt2 graph plot``.

Two batch sources, ONE execution path
--------------------------------------
Both sources end up running the EXACT live ``salt2 fit``/``salt2 test`` wiring
(run-free `Salt2CLI` -> attach trainer/model/datamodule -> ``dm.setup`` ->
``model.setup`` -> `materialise_all` -> one ``dset[slice]`` batch ->
``Executor(plan).run(bundle)``), so the shapes a probe reports are the shapes a
real run produces, not a re-derivation:

REAL (``data_file`` given)
    The file is wired into the datamodule (``train/val/test_file``) exactly as
    `salt2 fit` does; the model's real ``norm_dict`` (passed via ``--set``)
    materialises the `Normaliser` buffers. This is the must-have — it proves the
    concrete shapes off a live batch of the user's own data.

SYNTHETIC (``data_file`` is None)
    Works for ANY config with NO data. We compile the dataset plan once to learn
    the demand-narrowed per-stream read set (the EXACT fields the reader would
    pull — Features variables, the narrowed label fields every task/Labeller
    demands, MultiTarget/MaskFormer raw columns), then synthesise a tiny HDF5
    file whose structured groups carry precisely those fields (plus the ``valid``
    pad field for sequence streams), pick a concrete ``B`` (16) and ``T`` (40)
    per stream, and feed THAT file through the identical real path. A matching
    synthetic ``norm_dict`` (mean 0 / std 1 for every field) is written to a temp
    file and injected via ``--set ...norm_dict=<tmp>`` so the `Normaliser`
    materialises. Values are random/zeros — irrelevant to shapes.

The captured map is ``{flat_key: tuple(shape)}`` for every torch tensor in the
post-run bundle. Keys that never become torch tensors (``raw.*`` never crosses
the boundary, ``seq.layout`` is a dict leaf) are simply ABSENT — the renderer
falls back to the declared/symbolic shape for those.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from salt.core.graph.spec import Mode

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["SYNTHETIC", "probe_shapes"]

# Sentinel meaning "no data file — synthesise a batch" (the bare ``--probe``
# const). A distinct object so an empty-string path can never be mistaken for it.
SYNTHETIC = object()

_DEFAULT_BATCH = 16
_DEFAULT_TOKENS = 40  # concrete T per sequence stream in synthetic mode

# Lightning stage string per primary mode (the datamodule/model setup key).
_STAGE_OF_MODE: dict[Mode, str] = {
    Mode.FIT: "fit",
    Mode.VAL: "fit",  # VAL datasets are built by setup("fit")
    Mode.TEST: "test",
}


def probe_shapes(
    config_paths: Sequence[str | Path] | str | Path,
    set_overrides: Sequence[str] | None,
    data_file: str | Path | object | None,
    mode: Mode,
    batch: int = _DEFAULT_BATCH,
) -> dict[str, tuple[int, ...]]:
    """Run one batch through `mode`'s compiled plan and record every leaf's shape.

    Parameters
    ----------
    config_paths : Sequence[str | Path] | str | Path
        The trainer config (stack) — the same ``-c`` argument `salt2 graph plot`
        parses.
    set_overrides : Sequence[str] | None
        ``KEY=VALUE`` trainer overrides (the ``--set`` flag). In synthetic mode
        the synthetic ``norm_dict`` paths are appended automatically.
    data_file : str | Path | object | None
        A real HDF5 file to seed from (REAL mode), or `SYNTHETIC` / ``None`` to
        synthesise a batch from the reader schema (SYNTHETIC mode).
    mode : Mode
        The primary mode whose plan to run (``Mode.FIT`` / ``VAL`` / ``TEST``).
    batch : int, optional
        Batch size B, by default 16.

    Returns
    -------
    dict[str, tuple[int, ...]]
        ``{dotted_key: tuple(shape)}`` for every torch tensor in the post-run
        bundle (``inputs.* / normed.* / embed.* / encoded.* / preds.* /
        losses.* / labels.* / masks.*``). Scalars map to ``()``.

    Raises
    ------
    ValueError
        When `mode` is ONNX (no dataset batch to seed) or not a primary mode.
    """
    if mode not in _STAGE_OF_MODE:
        raise ValueError(
            f"probe mode {mode.name!r} is not supported — use fit/val/test "
            "(onnx has no dataset batch to seed)"
        )
    raw_paths = [config_paths] if isinstance(config_paths, (str, Path)) else list(config_paths)
    paths = [Path(p) for p in raw_paths]
    synthetic = data_file is None or data_file is SYNTHETIC

    with tempfile.TemporaryDirectory(prefix="salt_probe_") as tmp:
        tmpdir = Path(tmp)
        overrides = list(set_overrides or [])
        if synthetic:
            real_file, extra_overrides = _synthesise_inputs(
                paths, overrides, mode, tmpdir, batch
            )
            overrides += extra_overrides
        else:
            real_file = Path(data_file)  # type: ignore[arg-type]
        return _run_one_batch(paths, overrides, real_file, mode, batch)


# ---------------------------------------------------------------------------
# the shared real-data execution path (both sources land here)
# ---------------------------------------------------------------------------


def _run_one_batch(
    paths: Sequence[Path],
    overrides: Sequence[str],
    data_file: Path,
    mode: Mode,
    batch: int,
) -> dict[str, tuple[int, ...]]:
    """Build the run-free CLI, wire the file in, run one batch, record shapes.

    Mirrors `salt2 fit`/`test`: attach trainer<->model<->datamodule, adopt the
    model's sink demand, ``dm.setup``/``model.setup`` (compile + bind), run the
    preflights + `materialise_all` (loads the norm_dict), pull one batch from the
    stage dataset and execute the compiled plan over it.

    Returns
    -------
    dict[str, tuple[int, ...]]
        Concrete shapes by dotted key for the post-run bundle's torch tensors.
    """
    import torch  # noqa: PLC0415 - heavy import, lazy

    from salt.core.cli import _parse_trainer_cli  # noqa: PLC0415 - same-package surface
    from salt.core.graph.bundle import Bundle  # noqa: PLC0415
    from salt.core.graph.executor import Executor  # noqa: PLC0415
    from salt.core.nn.bind import materialise_all  # noqa: PLC0415

    # CPU-friendly + deterministic: torch-math attention already in the configs,
    # 0 workers (bind in-process), single CPU device, 32-bit. The fit stage builds
    # BOTH the train and val datasets, so point both files at the seed file (one
    # is enough — we only read one batch from the mode's own dataset).
    file_overrides = [f"data.{attr}={data_file}" for attr in _dm_file_attrs(mode)]
    overrides = [
        *overrides,
        *file_overrides,
        "data.num_workers=0",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "trainer.precision=32-true",
    ]
    cli = _parse_trainer_cli(paths, overrides)
    model, dm, trainer = cli.model, cli.datamodule, cli.trainer

    # attach run-free trainer<->model<->datamodule (what trainer.fit wires up).
    # Private-member writes/reads mirror salt.core.cli's run-free adapters: the
    # probe drives the same SaltModule/datamodule surface from the same package.
    trainer.datamodule = dm
    model._trainer = trainer  # noqa: SLF001 - SaltModule reads self._trainer (same-package)
    dm.trainer = trainer
    dm.set_sinks(model.sink_demand())

    stage = _STAGE_OF_MODE[mode]
    dm.setup(stage)
    model.setup(stage)

    dset = _stage_dataset(dm, mode)
    plan = model.plans[mode]

    # materialise file-backed values (the norm_dict mean/std) — what on_fit_start
    # does on a fresh fit; preflights validate the norm dict first.
    model._run_preflights()  # noqa: SLF001 - same-package run-free on_fit_start mirror
    materialise_all(model._graph_modules)  # noqa: SLF001 - same-package adapter

    b = min(batch, len(dset))
    seed_bundle = Bundle(dict(dset[slice(0, b)]))

    model.eval()
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        out = Executor(plan).run(seed_bundle)

    shapes: dict[str, tuple[int, ...]] = {}
    for key in out.keys():  # noqa: SIM118 - Bundle.keys() is a method, not a dict
        value = out.get(key)
        if isinstance(value, torch.Tensor):
            shapes[key] = tuple(value.shape)
        elif hasattr(value, "shape"):  # ndarray boundary leaves, defensive
            shapes[key] = tuple(int(d) for d in value.shape)
    return shapes


def _dm_file_attrs(mode: Mode) -> tuple[str, ...]:
    """The datamodule file attributes that `mode`'s ``setup`` needs populated.

    ``setup("fit")`` builds the train AND val datasets, so both files must point
    at the seed file (only the mode's own dataset is read); ``setup("test")``
    builds only the test dataset.

    Returns
    -------
    tuple[str, ...]
        The ``GraphDataModule`` file attribute names to set.
    """
    if mode in {Mode.FIT, Mode.VAL}:
        return ("train_file", "val_file")
    return ("test_file",)


def _stage_dataset(dm: Any, mode: Mode) -> Any:
    """The built `GraphDataset` for `mode` (train/val/test).

    Returns
    -------
    Any
        The stage `GraphDataset` (``train_dset`` / ``val_dset`` / ``test_dset``).
    """
    return {Mode.FIT: dm.train_dset, Mode.VAL: dm.val_dset, Mode.TEST: dm.test_dset}[mode]


# ---------------------------------------------------------------------------
# synthetic batch: build a fake H5 + schema + norm_dict from the reader schema
# ---------------------------------------------------------------------------


def _synthesise_inputs(
    paths: Sequence[Path],
    overrides: Sequence[str],
    mode: Mode,
    tmpdir: Path,
    batch: int,
) -> tuple[Path, list[str]]:
    """Synthesise a fake H5 file + norm_dict from the config's reader schema.

    Compiles the dataset plan once (config-only, no file) to learn the
    demand-narrowed per-stream read set, then writes a structured-group HDF5
    file carrying exactly those fields (plus ``valid`` for sequence streams) and
    a matching mean-0/std-1 norm_dict for every `Normaliser`. Returns the file
    path and the extra ``--set ...norm_dict=<tmp>`` overrides.

    Returns
    -------
    tuple[Path, list[str]]
        ``(h5_file, norm_dict_overrides)``.
    """
    import h5py  # noqa: PLC0415

    from salt.core.cli import _parse_trainer_cli  # noqa: PLC0415
    from salt.core.data.dataset import GraphDataset  # noqa: PLC0415
    from salt.core.data.processors import Labels  # noqa: PLC0415

    # First parse: inject placeholder norm_dict paths for every Normaliser so the
    # trainer config can be constructed data-free (their values are irrelevant
    # while we only need the dataset plan). The norm_dicts are written for real
    # below and re-injected on the execution parse.
    norm_dict_path = tmpdir / "norm_dict.yaml"
    norm_overrides = _norm_dict_overrides(paths, overrides, norm_dict_path)

    cli = _parse_trainer_cli(paths, [*overrides, *norm_overrides])
    dm = cli.datamodule
    model = cli.model
    reader = dm.reader

    # the demand-narrowed read set needs a compiled dataset plan; build a
    # throwaway GraphDataset for the mode (config-only — no file touched until
    # __len__/__getitem__). Labels narrowing comes from the model's sink demand.
    sinks = model.sink_demand()
    for module in dm.modules.values():
        if isinstance(module, Labels):
            module.bind_streams(reader.streams)
    probe_dset = GraphDataset(dm.modules, mode=mode, sinks=sinks)
    read_fields = probe_dset.read_fields  # {stream: {field: who}}
    label_fields = _label_fields(probe_dset.plan)  # {stream: {int-typed label fields}}

    # per-stream: every group the reader serves needs a dataset, even if no
    # field is demanded (the reader reads raw.<stream> for ALL configured
    # groups). Default to a single synthetic field when nothing is demanded.
    streams = reader.streams
    h5_file = tmpdir / "synthetic.h5"
    t = _DEFAULT_TOKENS
    rng = np.random.default_rng(0)
    with h5py.File(h5_file, "w") as f:
        for stream in streams:
            cfg = reader.groups[stream]
            fields = sorted(read_fields.get(stream, {}))
            if not fields:
                fields = ["_probe_dummy"]
            arr = _synthesise_group(
                stream, cfg, fields, label_fields.get(stream, frozenset()), batch, t, rng
            )
            f.create_dataset(cfg.dataset, data=arr)
        # global file attr so the schema artifact has something to scrape
        f.attrs["probe"] = "synthetic"

    _write_norm_dict(paths, overrides, norm_overrides, model, norm_dict_path)
    return h5_file, norm_overrides


def _synthesise_group(
    stream: str,
    cfg: Any,
    fields: Sequence[str],
    label_fields: frozenset[str],
    batch: int,
    tokens: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build one structured array for a stream: ``[B]`` global, ``[B, T]`` sequence.

    Feature fields are float32 random noise; demanded LABEL fields are int32
    zeros (a valid class index / vertex index for any task — the `Labels`
    processor casts integer columns to int64, which classification/vertexing
    losses require). Sequence streams also carry a ``valid`` bool field (the
    reader derives ``masks.<stream> = ~valid``). Only shapes matter — values are
    irrelevant.

    Returns
    -------
    np.ndarray
        The structured batch array.
    """
    del stream
    is_global = bool(cfg.global_object)
    shape: tuple[int, ...] = (batch,) if is_global else (batch, tokens)
    names = list(dict.fromkeys(fields))
    if "valid" in names:
        names.remove("valid")
    dtype_fields: list[tuple[str, str]] = [
        (name, "i4" if name in label_fields else "f4") for name in names
    ]
    if not is_global:
        dtype_fields.append(("valid", "bool"))
    arr = np.zeros(shape, dtype=np.dtype(dtype_fields))
    for name in names:
        if name in label_fields:
            arr[name] = 0  # valid class/vertex index 0; int32 -> Labels casts to int64
        else:
            # small finite noise (Features asserts finiteness)
            arr[name] = rng.standard_normal(shape).astype("f4")
    if not is_global:
        # mark all constituents valid so masks are all-False (no padding) — the
        # shape is identical either way, and avoids empty-token degeneracies.
        arr["valid"] = True
    return arr


def _label_fields(plan: Any) -> dict[str, frozenset[str]]:
    """Per-stream raw fields that feed a ``labels.*`` produce (made int-typed).

    A field is treated as a label when it is read directly off the file to
    populate a ``labels.<stream>.<field>`` leaf — i.e. demanded by a step that
    produces under the ``labels`` namespace (`Labels`, `MultiTarget`,
    `MaskFormerTargets`). Such columns must be integer so the label processors'
    int64 cast fires and classification/vertexing losses get ``Long`` targets.

    Returns
    -------
    dict[str, frozenset[str]]
        ``{stream: {field, ...}}`` of label-typed raw fields.
    """
    out: dict[str, set[str]] = {}
    for step in plan.steps:
        produces_label = any(key.split(".", 1)[0] == "labels" for key in step.produces)
        collector = getattr(step.module, "read_fields", None)
        if not produces_label or collector is None:
            continue
        for stream, fields in collector(step).items():
            out.setdefault(stream, set()).update(fields)
    return {stream: frozenset(fields) for stream, fields in out.items()}


def _norm_dict_overrides(
    paths: Sequence[Path],
    overrides: Sequence[str],
    norm_dict_path: Path,
) -> list[str]:
    """``--set`` overrides pointing every config Normaliser at the synthetic norm dict.

    Discovers the Normaliser module names from a throwaway parse (the configs
    already supply placeholder norm_dict paths via the user's ``--set``, so this
    parse succeeds).

    Returns
    -------
    list[str]
        ``["model.modules.<norm>.init_args.norm_dict=<tmp>", ...]``.
    """
    from salt.core.cli import _parse_trainer_cli  # noqa: PLC0415
    from salt.core.nn.modules import Normaliser  # noqa: PLC0415

    cli = _parse_trainer_cli(paths, overrides)
    graph_modules = cli.model._graph_modules  # noqa: SLF001 - same-package adapter
    names = [name for name, module in graph_modules.items() if isinstance(module, Normaliser)]
    return [f"model.modules.{name}.init_args.norm_dict={norm_dict_path}" for name in names]


def _write_norm_dict(
    paths: Sequence[Path],
    overrides: Sequence[str],
    norm_overrides: Sequence[str],
    model: Any,
    norm_dict_path: Path,
) -> None:
    """Write a mean-0/std-1 norm dict covering every Normaliser stream + field.

    The Normaliser captures its per-stream field list only at ``bind`` (it has
    none at construction), so we read the configured Features variables off the
    parsed config: a Normaliser stream is normalised over the same columns the
    Features module materialises for that stream.
    """
    import yaml  # noqa: PLC0415

    from salt.core.cli import _parse_trainer_cli  # noqa: PLC0415
    from salt.core.data.processors import Features  # noqa: PLC0415
    from salt.core.nn.modules import Normaliser  # noqa: PLC0415

    cli = _parse_trainer_cli(paths, [*overrides, *norm_overrides])
    features = next(
        (m for m in cli.datamodule.modules.values() if isinstance(m, Features)), None
    )
    var_map: dict[str, list[str]] = dict(features.variables) if features is not None else {}

    norm_dict: dict[str, dict[str, dict[str, float]]] = {}
    for module in cli.model._graph_modules.values():  # noqa: SLF001 - same-package adapter
        if not isinstance(module, Normaliser):
            continue
        for stream in module.streams:
            entries = norm_dict.setdefault(stream, {})
            for field in var_map.get(stream, []):
                entries[field] = {"mean": 0.0, "std": 1.0}
    with open(norm_dict_path, "w") as fh:
        yaml.safe_dump(norm_dict, fh, sort_keys=False)
    del model  # kept in the signature for symmetry / future per-bind field lookup
