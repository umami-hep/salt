"""v1 -> v2 state-dict mapping.

Maps a v1 ``ModelWrapper`` state_dict onto a v2 module dict built from
config, so a v2 model can be trained from transferred v1 weights.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from salt.core.graph.spec import GraphModule
from salt.core.nn.modules import (
    Concat,
    GlobalAttentionPooling,
    MaskedInputNormaliser,
    Normaliser,
    StreamEmbed,
    TransformerEncoder,
)
from salt.core.nn.tasks import ClassificationTaskModule, VertexingTaskModule

__all__ = ["map_v1_state_dict"]

_INIT_NET_RE = re.compile(r"^model\.init_nets\.(\d+)\.(.+)$")
_TASK_RE = re.compile(r"^model\.tasks\.(\d+)\.(.+)$")
_NORM_RE = re.compile(r"^norm\.(.+)_(means|stds)$")


def map_v1_state_dict(
    v1_sd: Mapping[str, Tensor],
    modules: Mapping[str, GraphModule],
    task_order: Sequence[str] | None = None,
) -> dict[str, Tensor]:
    """Map a v1 `ModelWrapper` state_dict onto a v2 module dict.

    The returned keys are relative to the MODULE DICT — loadable via
    ``nn.ModuleDict(modules).load_state_dict(mapped)`` after every module's
    `bind` has run (bind allocates the buffers/layers the load fills).
    Inside `SaltModule` (``self.net = nn.ModuleDict(...)``) the same keys
    gain the ``net.`` prefix. Tensor values are shared, not cloned
    (``load_state_dict`` copies into the target parameters).

    v1 ``init_nets`` order IS the concat order, so init-net ``i`` maps to
    the `StreamEmbed` for ``Concat.streams[i]``. v1 ``tasks`` order is the
    construction order, which the state_dict alone does not name — by
    default task index ``i`` maps to the ``i``-th task module in
    module-dict order; pass ``task_order`` to override.

    Parameters
    ----------
    v1_sd : Mapping[str, Tensor]
        ``ModelWrapper.state_dict()`` of the v1 model.
    modules : Mapping[str, GraphModule]
        The v2 module dict (instance name -> module), containing exactly one
        normaliser (`Normaliser` or `MaskedInputNormaliser`), `Concat`,
        `TransformerEncoder`, `GlobalAttentionPooling`, one `StreamEmbed`
        per concat stream, and the task modules.
    task_order : Sequence[str] | None, optional
        v2 instance names in v1 ``model.tasks`` index order, by default the
        task modules' module-dict order.

    Returns
    -------
    dict[str, Tensor]
        The mapped v2 state_dict.

    Raises
    ------
    ValueError
        If the module dict lacks (or duplicates) a required module, an
        index/stream in the v1 state_dict has no v2 counterpart, or any v1
        key cannot be mapped (nothing is dropped silently).
    """
    norm = _single(modules, (Normaliser, MaskedInputNormaliser))
    self_normalising = isinstance(norm[1], MaskedInputNormaliser)
    concat = _single(modules, Concat)
    encoder_name = _single(modules, TransformerEncoder)[0]
    pool_name = _single(modules, GlobalAttentionPooling)[0]

    embed_by_stream: dict[str, str] = {}
    for name, module in modules.items():
        if isinstance(module, StreamEmbed):
            if module.stream in embed_by_stream:
                raise ValueError(
                    f"map_v1_state_dict: two StreamEmbed modules for stream "
                    f"{module.stream!r} ({embed_by_stream[module.stream]!r}, {name!r})"
                )
            embed_by_stream[module.stream] = name
    embed_names: list[str] = []
    for stream in concat[1].streams:
        if stream not in embed_by_stream:
            raise ValueError(
                f"map_v1_state_dict: no StreamEmbed for concat stream {stream!r} "
                f"(streams {concat[1].streams})"
            )
        embed_names.append(embed_by_stream[stream])

    if task_order is None:
        task_names = [
            name
            for name, module in modules.items()
            if isinstance(module, (ClassificationTaskModule, VertexingTaskModule))
        ]
    else:
        task_names = list(task_order)
        for name in task_names:
            if name not in modules:
                raise ValueError(f"map_v1_state_dict: task_order names unknown module {name!r}")

    out: dict[str, Tensor] = {}
    unmapped: list[str] = []
    norm_streams: set[str] = set()
    for key, value in v1_sd.items():
        if (match := _NORM_RE.match(key)) is not None:
            stream, kind = match.groups()
            if stream not in norm[1].streams:
                raise ValueError(
                    f"map_v1_state_dict: v1 norm buffer for stream {stream!r} has no v2 "
                    f"counterpart — Normaliser streams are {norm[1].streams}"
                )
            norm_streams.add(stream)
            if not self_normalising:
                # Default fixed-norm `Normaliser`: v1 means/stds map straight to
                # the v2 means_<stream>/stds_<stream> buffers (bitwise v1 parity).
                out[f"{norm[0]}.{kind}_{stream}"] = value
            # MaskedInputNormaliser stores running_mean/running_var, not means/stds:
            # v1's `(x - mean) / std` maps to v2's
            # `(x - running_mean) / sqrt(running_var + eps)`, so running_mean =
            # v1.means and running_var = v1.stds**2 (the eps is a deliberate,
            # non-bitwise departure from v1 parity).
            elif kind == "means":
                out[f"{norm[0]}.running_mean_{stream}"] = value
            else:  # stds -> variance
                out[f"{norm[0]}.running_var_{stream}"] = value * value
        elif (match := _INIT_NET_RE.match(key)) is not None:
            index, rest = int(match.group(1)), match.group(2)
            if index >= len(embed_names):
                raise ValueError(
                    f"map_v1_state_dict: v1 init net index {index} out of range — concat "
                    f"declares {len(embed_names)} streams ({concat[1].streams})"
                )
            if not rest.startswith("net."):
                # pos_enc / featurewise parameters — not mapped
                unmapped.append(key)
                continue
            out[f"{embed_names[index]}.{rest}"] = value
        elif (match := _TASK_RE.match(key)) is not None:
            index, rest = int(match.group(1)), match.group(2)
            if index >= len(task_names):
                raise ValueError(
                    f"map_v1_state_dict: v1 task index {index} out of range — the module "
                    f"dict has {len(task_names)} task modules ({task_names})"
                )
            out[f"{task_names[index]}.task.{rest}"] = value
        elif key.startswith("model.encoder."):
            out[f"{encoder_name}.encoder.{key[len('model.encoder.') :]}"] = value
        elif key.startswith("model.pool_net."):
            out[f"{pool_name}.pool_net.{key[len('model.pool_net.') :]}"] = value
        else:
            unmapped.append(key)
    if unmapped:
        raise ValueError(
            f"map_v1_state_dict: {len(unmapped)} v1 keys have no v2 mapping (nothing is "
            f"dropped silently): {sorted(unmapped)}"
        )
    # Synthesise the chosen normaliser's v2-only bookkeeping buffers, which v1
    # has no counterpart for.
    if not self_normalising:
        out[f"{norm[0]}.materialised"] = torch.tensor(True)
    else:
        # Transferred stats count as "seen": num_batches_tracked=1. The exact
        # object count is unknown, so num_objects_seen is left at 0 — only the
        # cumulative momentum=None path would consume it.
        for stream in norm_streams:
            out[f"{norm[0]}.num_batches_tracked_{stream}"] = torch.tensor(1, dtype=torch.long)
            out[f"{norm[0]}.num_objects_seen_{stream}"] = torch.tensor(0, dtype=torch.long)
    return out


def _single(
    modules: Mapping[str, GraphModule], cls: type | tuple[type, ...]
) -> tuple[str, object]:
    """Find the single instance of `cls` (one class or a tuple) in the module dict.

    Raises
    ------
    ValueError
        If there is no instance, or more than one.
    """
    found = [(name, module) for name, module in modules.items() if isinstance(module, cls)]
    if len(found) != 1:
        label = (
            cls.__name__
            if isinstance(cls, type)
            else " | ".join(c.__name__ for c in cls)
        )
        raise ValueError(
            f"map_v1_state_dict: expected exactly one {label} in the module dict, "
            f"found {[name for name, _ in found]}"
        )
    return found[0]
