r"""muP (maximal update parametrization) tooling: shape generation and coord-check.

Provides :func:`generate_shapes` (``salt mup-shapes``), :func:`coord_check`
(``salt mup-coord-check``), and :func:`setup_mup`, a console entry point.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.executor import Executor
from salt.graph.planner import compile_plan
from salt.graph.spec import Mode
from salt.model.bind import bind_all, materialise_all, resolve_bind_schema
from salt.utils.logging import console

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "build_model_at_widths",
    "coord_check",
    "generate_shapes",
    "plot_coord_data",
    "setup_mup",
]


# ---------------------------------------------------------------------------
# helpers — build a SaltModule from a config at given apply_to widths
# ---------------------------------------------------------------------------


def _parse_cli(configs: Sequence[str | Path], set_overrides: Sequence[str]) -> Any:
    """Parse a trainer config stack through the real salt surface, run-free,
    returning the constructed (un-setup) `SaltCLI` (``cli.model`` +
    ``cli.datamodule``). Raises `ConfigError` on a parse failure.
    """
    import warnings

    from salt.main import SaltCLI
    from salt.utils.config_utils import disable_logger_in_config

    args: list[str] = []
    for cfg in configs:
        # disable the logger in keyless envs (no COMET_API_KEY) so run-free
        # parsing doesn't fail at instantiate_classes
        cfg_no_logger = disable_logger_in_config(str(cfg))
        args.extend(["--config", cfg_no_logger])
    for entry in set_overrides:
        if "=" not in entry:
            raise ConfigError(f"--set entries must be KEY=VALUE, got {entry!r}")
        args.append(f"--{entry}")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*args parameter is intended to run from within Python.*"
            )
            cli = SaltCLI(args=args, run=False)
    except SystemExit as err:
        raise ConfigError(
            f"trainer config {' '.join(str(c) for c in configs)} failed to parse through the "
            f"salt surface (parser exit {err.code}). Required init_args left as overrides in the "
            "YAML can be supplied data-free via --set, e.g. "
            "--set model.modules.norm.init_args.norm_dict=unused.yaml"
        ) from err
    return cli


def _parse_model(configs: Sequence[str | Path], set_overrides: Sequence[str]) -> Any:
    """Parse and return just the constructed `SaltModule` (the apply_to-width probe)."""
    return _parse_cli(configs, set_overrides).model


def _width_overrides(model: Any, width: int) -> list[str]:
    """``--set`` width overrides for every ``apply_to`` module's declared
    ``MUP_WIDTH_ARG`` (e.g. `StreamEmbed.MUP_WIDTH_ARG == "out_dim"`), all set
    to `width`. Raises `ConfigError` if a module declares none.
    """
    cfg = _require_mup_cfg(model)
    overrides: list[str] = []
    for name in cfg["apply_to"]:
        module = model.net[name]
        width_arg = getattr(module, "MUP_WIDTH_ARG", None)
        if width_arg is None:
            raise ConfigError(
                f"module {name!r} ({type(module).__name__}) is in mup.apply_to but declares no "
                "MUP_WIDTH_ARG — the shape-generation tooling cannot sweep its width "
                "(add MUP_WIDTH_ARG to the module class)"
            )
        overrides.append(f"model.modules.{name}.init_args.{width_arg}={width}")
    return overrides


def _require_mup_cfg(model: Any) -> dict[str, Any]:
    """The model's validated mup config; raises `ConfigError` when the config
    declares no ``model.init_args.mup`` block.
    """
    cfg = getattr(model, "mup_cfg", None)
    if cfg is None:
        raise ConfigError(
            "config declares no model.init_args.mup block — muP tooling needs "
            "mup: {apply_to: [...], shape_path: ...}"
        )
    return cfg


def build_model_at_widths(
    configs: Sequence[str | Path],
    width: int,
    set_overrides: Sequence[str] = (),
    *,
    bind: bool = True,
    materialise: bool = False,
) -> Any:
    """Build a `SaltModule` with every ``apply_to`` module at `width`.

    Re-parses the config with the per-``apply_to`` width overrides applied,
    then (by default) compiles the FIT plan, binds, and — optionally —
    materialises so the model is forward-runnable.

    Parameters
    ----------
    configs : Sequence[str | Path]
        The trainer config stack (deep-merged left-to-right).
    width : int
        The width to set on every ``apply_to`` module's ``MUP_WIDTH_ARG``.
    set_overrides : Sequence[str], optional
        Extra ``--set`` overrides (e.g. the data-free norm_dict), by default ().
    bind : bool, optional
        Whether to compile + bind the model (needed for shape extraction and
        coord-check), by default True.
    materialise : bool, optional
        Whether to materialise file-backed values (Normaliser buffers) so the
        model can run a forward, by default False.

    Returns
    -------
    SaltModule
        The width-set (and, by default, bound) model.
    """
    # parse once to read the apply_to width-arg list, then re-parse with the
    # width overrides applied — the second parse is the model we return
    probe = _parse_model(configs, set_overrides)
    overrides = [*set_overrides, *_width_overrides(probe, width)]
    cli = _parse_cli(configs, overrides)
    model = cli.model
    if bind:
        combined = _combined_graph(cli)
        # reader.declare_io is data-free, so compile the combined data+model
        # graph with empty sources and it resolves without touching a file.
        # Bind only the model-side modules: the reader's bind DOES touch the
        # file, which this data-free shape/coord path must avoid.
        fit_plan = compile_plan(combined, Mode.FIT, sources={}, sinks=["loss.total"])
        schema = resolve_bind_schema([fit_plan])
        bind_all(model._graph_modules, schema)  # noqa: SLF001 - same-package tooling
        model.schema = schema
        model._bound = True  # noqa: SLF001 - same-package tooling
        if materialise:
            # materialise only the model side; the dataset modules' file-touching
            # materialise is skipped — the coord-check synthesises its own batch
            materialise_all(model._graph_modules)  # noqa: SLF001 - same-package tooling
            model._materialised = True  # noqa: SLF001 - same-package tooling
        # store the combined plan + the model-only coord-check plan (run
        # data-free from a synthesised boundary batch) for the coord-check executor
        model._mup_combined = combined  # noqa: SLF001 - same-package tooling
        model._mup_fit_plan = fit_plan  # noqa: SLF001 - same-package tooling
        coord_sources = _model_boundary_sources(cli)
        model._mup_coord_plan = compile_plan(  # noqa: SLF001 - same-package tooling
            model._graph_modules,  # noqa: SLF001 - same-package tooling
            Mode.FIT,
            sources=coord_sources,
            sinks=["loss.total"],
        )
    return model


def _model_boundary_sources(cli: Any) -> Any:
    """The MODEL-side boundary sources (``inputs.*``/``masks.*``/``labels.*``) the
    coord-check needs, derived data-free from `Features` variables, the
    reader's ``global_object`` flag, and the model's FIT label demand.
    Raises `ConfigError` when no `Features` module is configured.
    """
    from salt.data.processors.features import Features
    from salt.graph.spec import TensorSpec, sym_dim, unflatten_spec

    model, dm = cli.model, cli.datamodule
    reader = dm.reader
    features = next((m for m in dm.modules.values() if isinstance(m, Features)), None)
    if features is None:
        raise ConfigError(
            "coord-check needs a Features module in data.modules to derive input field counts"
        )
    flat: dict[str, TensorSpec] = {}
    for stream, names in features.variables.items():
        is_global = bool(getattr(reader.groups.get(stream), "global_object", False))
        n_fields = len(names)
        if is_global:
            flat[f"inputs.{stream}"] = TensorSpec(
                shape=("B", n_fields), dtype="float32", fields=tuple(names)
            )
        else:
            t_dim = sym_dim("T", stream)
            flat[f"inputs.{stream}"] = TensorSpec(
                shape=("B", t_dim, n_fields), dtype="float32", fields=tuple(names)
            )
            flat[f"masks.{stream}"] = TensorSpec(shape=("B", t_dim), dtype="bool", kind="pad_mask")
    # label demand (FIT): labels.<stream>.<label> is per-token [B, T] for a
    # sequence stream, per-jet [B] for a global-object stream
    for key in model.sink_demand().get(Mode.FIT, []):
        parts = key.split(".")
        if parts[0] != "labels" or key in flat:
            continue
        stream = parts[1] if len(parts) > 1 else ""
        is_global = bool(getattr(reader.groups.get(stream), "global_object", True))
        shape = ("B",) if is_global else ("B", sym_dim("T", stream))
        flat[key] = TensorSpec(shape=shape, dtype="int64", kind="label", modes=Mode.TRAINING)
    return unflatten_spec(flat)


def _combined_graph(cli: Any) -> dict[str, Any]:
    """The combined data + model module dict; binds the dataset `Labels`
    processors to the reader streams so their label-key universe resolves.
    """
    from salt.data.processors.labels import Labels

    model, dm = cli.model, cli.datamodule
    data_modules = dm.modules
    reader = dm.reader
    for module in data_modules.values():
        if isinstance(module, Labels):
            module.bind_streams(reader.streams)
    return {**data_modules, **model._graph_modules}  # noqa: SLF001 - same-package tooling


# ---------------------------------------------------------------------------
# salt mup-shapes — base/delta infshape generation
# ---------------------------------------------------------------------------


def generate_shapes(
    configs: Sequence[str | Path],
    save_path: str | Path | None = None,
    base_width: int | None = None,
    delta_width: int | None = None,
    set_overrides: Sequence[str] = (),
) -> tuple[Path, Any]:
    """Generate the base/delta muP infshapes for a config.

    Builds a BASE model with every ``apply_to`` module at `base_width` and a
    DELTA model at `delta_width` (which must differ — the two widths fix the
    infinite-width directions), then ``mup.make_base_shapes(base, delta,
    savefile)`` writes the infshape file. This is exactly what
    ``SaltModule.mup.shape_path`` loads at bind, and what
    `MuAdamW`/`MuReadout.width_mult()` resolve against.

    Parameters
    ----------
    configs : Sequence[str | Path]
        The trainer config stack.
    save_path : str | Path | None, optional
        Where to write the infshape file; defaults to the config's
        ``mup.shape_path`` (and errors if neither is set).
    base_width : int | None, optional
        The base (narrow) ``apply_to`` width; defaults to half the config's
        configured width (read from the first ``apply_to`` module).
    delta_width : int | None, optional
        The delta (wider) ``apply_to`` width; defaults to the config's
        configured width.
    set_overrides : Sequence[str], optional
        Extra ``--set`` overrides (e.g. data-free norm_dict), by default ().

    Returns
    -------
    tuple[Path, Any]
        ``(written_path, base_shapes)`` — the infshape file and the in-memory
        base-shapes dict (for programmatic callers / tests).

    Raises
    ------
    ConfigError
        On a missing mup block, equal base/delta widths, or no save path.
    """
    from mup import make_base_shapes

    probe = _parse_model(configs, set_overrides)
    cfg = _require_mup_cfg(probe)
    first = probe.net[cfg["apply_to"][0]]
    configured = int(getattr(first, getattr(first, "MUP_WIDTH_ARG")))  # noqa: B009
    base_w = int(base_width) if base_width is not None else max(1, configured // 2)
    delta_w = int(delta_width) if delta_width is not None else configured
    if base_w == delta_w:
        raise ConfigError(
            f"mup base_width and delta_width must differ (both {base_w}) — the two widths fix the "
            "infinite-width directions (v1 parameter_base != parameter_delta, "
            "GN2_muP.yaml:14-15)"
        )
    out = save_path if save_path is not None else cfg.get("shape_path")
    if out is None:
        raise ConfigError(
            "no save path: pass --save-path or set model.init_args.mup.shape_path in the config"
        )
    out_path = Path(out)
    base_model = build_model_at_widths(configs, base_w, set_overrides, bind=True)
    delta_model = build_model_at_widths(configs, delta_w, set_overrides, bind=True)
    base_shapes = make_base_shapes(base_model.net, delta_model.net, savefile=str(out_path))
    console(
        f"salt mup-shapes: wrote infshapes to {out_path} "
        f"(apply_to={cfg['apply_to']}, base_width={base_w}, delta_width={delta_w})"
    )
    return out_path, base_shapes


# ---------------------------------------------------------------------------
# salt mup-coord-check — coordinate-check data + plot
# ---------------------------------------------------------------------------


def _record_l1_hook(records: list[dict[str, Any]], width: int, name: str, step: int):
    """Forward hook recording the output L1 coordinate norm as
    ``{width, module, t, l1}`` per output tensor.
    """

    def hook(_module: Any, _inp: Any, output: Any) -> None:
        with torch.no_grad():
            tensors: list[torch.Tensor] = []
            if isinstance(output, torch.Tensor):
                tensors = [output]
            elif isinstance(output, (tuple, list)):
                tensors = [o for o in output if isinstance(o, torch.Tensor)]
            elif isinstance(output, dict):
                tensors = [o for o in output.values() if isinstance(o, torch.Tensor)]
            for tensor in tensors:
                if tensor.is_floating_point():
                    records.append({  # noqa: PERF401 - hook accumulates into a shared records list
                        "width": width,
                        "module": name,
                        "t": step,
                        "l1": float(tensor.abs().mean().item()),
                    })

    return hook


def coord_check(
    configs: Sequence[str | Path],
    widths: Sequence[int],
    *,
    batch: Bundle | None = None,
    nsteps: int = 3,
    nseeds: int = 1,
    lr: float = 1e-2,
    shape_file: str | Path | None = None,
    set_overrides: Sequence[str] = (),
) -> pd.DataFrame:
    """Run the muP coord-check at several widths.

    For each ``width`` build a muP model, apply a SHARED base/delta shape file,
    then train it for `nsteps` steps on a FIXED batch, recording each
    ``apply_to`` submodule's output L1 coordinate norm via a forward hook. A
    muP-correct net has these norms roughly invariant across widths
    ("muP-flat").

    **Shared-base protocol**: the coord-check is only meaningful when every
    swept width is parametrised against ONE base/delta infshape file. A
    per-width self-base (``set_base_shapes(model.net, model.net)``) forces
    ``width_mult == 1`` at every width, so `MuReadout` is never damped and
    produces a misleading non-flat curve. This function generates (or
    accepts) a base (narrowest swept width) + delta (a wider reference)
    infshape file ONCE and applies it to ALL widths, so wider models see
    ``width_mult > 1`` and the readout is correctly damped.

    Parameters
    ----------
    configs : Sequence[str | Path]
        The trainer config stack (must declare a ``mup`` block).
    widths : Sequence[int]
        The ``apply_to`` widths to sweep (e.g. ``[16, 32, 64, 128]``).
    batch : Bundle | None, optional
        A FIXED input batch (``inputs.*``/``masks.*``/``labels.*`` keys). When
        None a synthetic random batch is generated from the FIT plan's
        boundary, by default None.
    nsteps : int, optional
        Training steps per width, by default 3.
    nseeds : int, optional
        Random-seed repeats, by default 1.
    lr : float, optional
        The (large) coord-check learning rate, by default 1e-2.
    shape_file : str | Path | None, optional
        A pre-generated base/delta infshape file to apply at EVERY width. When
        None, one is generated on the fly (base = min(widths), delta = a wider
        reference), by default None.
    set_overrides : Sequence[str], optional
        Extra ``--set`` overrides, by default ().

    Returns
    -------
    pandas.DataFrame
        Columns ``width, module, t, l1`` — the coord-check data.
    """
    import pandas as pd
    from mup import set_base_shapes
    from mup.optim import MuAdamW

    shared_shapes = _resolve_coord_shape_file(configs, widths, shape_file, set_overrides)
    records: list[dict[str, Any]] = []
    for seed in range(nseeds):
        for width in widths:
            torch.manual_seed(seed)
            model = build_model_at_widths(
                configs, width, set_overrides, bind=True, materialise=batch is None
            )
            cfg = _require_mup_cfg(model)
            # shared base shapes (not a per-width self-base): resolves
            # MuReadout.width_mult() = width/base_width so the readout is
            # correctly damped, instead of forcing width_mult == 1 at every width
            set_base_shapes(model.net, str(shared_shapes), rescale_params=False)
            coord_plan = model._mup_coord_plan  # noqa: SLF001 - same-package tooling
            fixed = _coord_batch(coord_plan) if batch is None else batch
            optimizer = MuAdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            handles: list[Any] = []
            model.train()
            try:
                for step in range(1, nsteps + 1):
                    for handle in handles:
                        handle.remove()
                    handles = _attach_apply_to_hooks(
                        model, cfg["apply_to"], records, width, step=step
                    )
                    out = Executor(coord_plan).run(_clone_batch(fixed))
                    loss = out.get("loss.total")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            finally:
                for handle in handles:
                    handle.remove()
    return pd.DataFrame.from_records(records, columns=["width", "module", "t", "l1"])


def _resolve_coord_shape_file(
    configs: Sequence[str | Path],
    widths: Sequence[int],
    shape_file: str | Path | None,
    set_overrides: Sequence[str],
) -> Path:
    """Resolve the SHARED base/delta infshape file the coord-check applies at
    every width: `shape_file` used as-is if given, else generated once via
    :func:`generate_shapes` (base = narrowest swept width, delta = a
    strictly-wider reference) into a temp file — so wider models resolve
    ``width_mult > 1`` and the readout is damped. Raises `ConfigError` when
    `widths` is empty.
    """
    if shape_file is not None:
        path = Path(shape_file)
        if not path.is_file():
            raise ConfigError(
                f"mup-coord-check --shape-file {path} does not exist — generate it with "
                "salt mup-shapes first, or omit --shape-file to auto-generate a shared base"
            )
        return path
    if not widths:
        raise ConfigError("mup-coord-check needs at least one width to sweep")
    base_w = min(widths)
    delta_w = max(widths) if max(widths) != base_w else base_w * 2
    import tempfile

    out = Path(tempfile.mkdtemp(prefix="salt_coord_shapes_")) / "coord_check.bsh"
    generate_shapes(
        configs,
        save_path=out,
        base_width=base_w,
        delta_width=delta_w,
        set_overrides=set_overrides,
    )
    console(
        f"salt mup-coord-check: generated SHARED base/delta infshapes (base_width={base_w}, "
        f"delta_width={delta_w}) at {out} — applied at EVERY swept width so width_mult is "
        "correct (the MU-HUMAN shared-base protocol, not a per-width self-base)"
    )
    return out


def _attach_apply_to_hooks(
    model: Any, apply_to: Sequence[str], records: list[dict[str, Any]], width: int, step: int = 1
) -> list[Any]:
    """Register output-L1 forward hooks on every ``apply_to`` submodule subtree,
    returning the handles.
    """
    handles: list[Any] = []
    for name in apply_to:
        root = model.net[name]
        for sub_name, submodule in root.named_modules():
            label = name if not sub_name else f"{name}.{sub_name}"
            handles.append(
                submodule.register_forward_hook(_record_l1_hook(records, width, label, step))
            )
    return handles


def _coord_batch(plan: Any) -> Bundle:
    """Synthesise a random FIT batch from the plan's boundary sources (smoke
    path, used when no real ``--train-file`` batch is given): B=4, T=5;
    ``pad_mask`` all-valid, ``label`` random class ids, else random float.
    """
    b = Bundle()
    n_batch, n_tracks = 4, 5
    for key, spec in plan.sources.items():
        concrete = tuple(
            n_batch if dim in {"B", -1} else n_tracks if isinstance(dim, str) else int(dim)
            for dim in spec.shape
        )
        if spec.kind == "pad_mask":
            b.set(key, torch.zeros(concrete, dtype=torch.bool))
        elif spec.kind == "label" or (spec.dtype and "int" in str(spec.dtype)):
            b.set(key, torch.randint(0, 3, concrete or (n_batch,)))
        else:
            b.set(key, torch.randn(concrete))
    return b


def _clone_batch(batch: Bundle) -> Bundle:
    """Deep-clone a bundle (``copy.deepcopy`` is not enough for the
    nested-dict-of-tensors batch) so a fixed batch survives in-place forward
    mutation.
    """
    clone = Bundle()
    for key in batch.keys():  # noqa: SIM118 - Bundle.keys() is the public flat-key API, not a dict
        value = batch.get(key)
        clone.set(key, value.clone().detach() if isinstance(value, torch.Tensor) else value)
    return clone


def plot_coord_data(df: pd.DataFrame, save_to: str | Path, *, title: str | None = None) -> Any:
    """Plot the coord-check data.

    One log-log subplot per training step: per-module output L1 norm vs
    width. A muP-flat net shows roughly horizontal lines (norms invariant
    across width); a non-muP net's lines slope with width.

    Parameters
    ----------
    df : pandas.DataFrame
        The :func:`coord_check` output (``width, module, t, l1``).
    save_to : str | Path
        Output image path (PNG/PDF).
    title : str | None, optional
        Figure suptitle, by default None.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (also saved to `save_to`).
    """
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    steps = sorted(df["t"].unique()) if not df.empty else [1]
    fig, axes = plt.subplots(1, len(steps), figsize=(5 * len(steps), 4), squeeze=False)
    for ax, step in zip(axes[0], steps, strict=False):
        sub = df[df["t"] == step]
        for module, grp in sub.groupby("module"):
            ordered = grp.sort_values("width")
            ax.plot(ordered["width"], ordered["l1"], marker="o", label=str(module))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("width")
        ax.set_ylabel("output L1 norm")
        ax.set_title(f"t={step}")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="upper right", fontsize="x-small")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    save_to = Path(save_to)
    if save_to.parent != Path():
        save_to.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_to)
    console(f"salt mup-coord-check: wrote coord-check plot to {save_to}")
    return fig


# ---------------------------------------------------------------------------
# setup_mup — console entry point
# ---------------------------------------------------------------------------


def setup_mup(args: Sequence[str] | None = None) -> int:
    """The ``setup_mup`` console entry point — a thin alias for ``salt mup-shapes``.

    Forwards to the salt ``mup-shapes`` subcommand so the two surfaces
    share one implementation.

    Parameters
    ----------
    args : Sequence[str] | None, optional
        argv (without the program name); console use reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code.
    """
    from salt.cli import main as graph_main

    argv = list(sys.argv[1:] if args is None else args)
    return graph_main(["mup-shapes", *argv])


# ---------------------------------------------------------------------------
# argparse command handlers (wired into salt graph CLI, cli.py)
# ---------------------------------------------------------------------------


def cmd_mup_shapes(args: Any) -> int:
    """``salt mup-shapes`` handler: generate base/delta infshapes."""
    generate_shapes(
        args.config,
        save_path=args.save_path,
        base_width=args.base_width,
        delta_width=args.delta_width,
        set_overrides=args.set or [],
    )
    return 0


def cmd_mup_coord_check(args: Any) -> int:
    """``salt mup-coord-check`` handler: writes coord-check data (CSV) + plot."""
    widths = [int(w) for w in args.widths]
    df = coord_check(
        args.config,
        widths,
        nsteps=args.nsteps,
        nseeds=args.nseeds,
        lr=args.lr,
        shape_file=getattr(args, "shape_file", None),
        set_overrides=args.set or [],
    )
    out = Path(args.output)
    if out.parent != Path():
        out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    console(f"salt mup-coord-check: wrote coord-check data to {csv_path}")
    plot_coord_data(df, out, title=f"muP coord-check (widths {widths})")
    return 0
