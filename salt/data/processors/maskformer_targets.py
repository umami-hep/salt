"""`MaskFormerTargets` — MaskFormer object targets: class, truth masks, regression labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from salt.data.base import Processor
from salt.graph.errors import ConfigError, SchemaError
from salt.graph.spec import IO, OBJECT_STREAM, Mode, TensorSpec, sym_dim, unflatten_spec


@dataclass(frozen=True)
class _ObjectCut:
    """A single field-bound cut on MaskFormer truth objects: ``min <= batch[field] <= max``.

    NaN handling / PV exemption / drop semantics live in
    `MaskFormerTargets._select_objects`.
    """

    field: str
    min: float | None = None
    max: float | None = None

    def __post_init__(self) -> None:
        if self.min is None and self.max is None:
            raise ConfigError(
                f"MaskFormerTargets: ObjectCut on field {self.field!r} must set at least one of "
                "min/max (v1 ObjectCut.__post_init__, configs.py)"
            )


class MaskFormerTargets(Processor):
    """MaskFormer object targets: ``object_class`` + per-constituent ``masks``.

    A single demand-gated processor producing, under the ``objects`` bundle
    stream, from the truth-object group:

    - ``labels.objects.object_class`` ``[B, M]`` int64 — the raw object
      class label remapped through ``class_map``, with the **null class
      validated LAST**.
    - ``labels.objects.masks`` ``[B, M, T]`` bool — the per-object-by-
      constituent truth mask: ``constituent_id == object_id`` over the
      truncated constituent stream.
    - ``labels.objects.<target>`` ``[B, M]`` float32 per
      ``regression_targets`` entry — the raw per-object regression labels
      the object-regression task stacks + scales.

    Class remapping is atomic and collision-proof: the mapped column is
    built from a single vectorised ``np.select`` against the ORIGINAL raw
    values (not a sequential in-place walk). Mask construction never
    mutates the published id columns: the sentinel substitution needed for
    the equality runs on a private copy.

    DEMAND-gated, not mode-gated: all three product families are declared
    in ALL modes; ordinary sink pruning removes the module from a plan only
    when nothing demands its outputs. The `H5OutputSink` object group demands
    ``labels.objects.{object_class,masks}`` in TEST (truth columns), so
    this module IS in the test plan. It is pruned from ONNX (nothing
    demands truth there).

    Parameters
    ----------
    object_class : str
        Object class label field in the object group (e.g. ``flavour``).
        Remapped through ``class_map`` to ``object_class``.
    object_id : str
        Object identity field used to build masks (e.g. ``barcode``).
    constituent_id : str
        Constituent identity field tested against ``object_id`` to build
        masks (e.g. ``ftagTruthParentBarcode``).
    class_map : Mapping[str, Mapping[str, Any]]
        ``{name: {raw: int | list[int], mapped: int, weight?: float}}``.
        MUST contain a ``null`` entry mapped LAST
        (``mapped == len(class_map) - 1``), and the ``mapped`` values MUST
        be exactly ``range(len(class_map))``. jsonargparse may parse the
        YAML ``null:`` key as a Python ``None`` — both spellings are
        accepted, normalised to ``"null"``. ``raw`` may be a single int
        (the common case) or a list/tuple of ints that all map to the same
        ``mapped`` index — a class *merge*. An optional scalar ``weight``
        per class feeds :attr:`object_weights` (default ``1.0``).
    object_stream : str
        File group holding the object features (e.g. ``truth_hadrons``).
        The reader serves it as ``raw.<object_stream>``.
    constituent_stream : str
        File group holding the constituent features (e.g. ``tracks``). The
        mask's last dim aligns with this stream's token count.
    regression_targets : Sequence[str] | None, optional
        Per-object regression label fields published under
        ``labels.objects.<target>``, by default None (no regression labels).
    num_objects : int | None, optional
        The number of object queries ``M``. When set, the produced shapes
        carry it as a concrete dim (a static check that the file's object
        count matches the decoder's query bank); None leaves ``M``
        symbolic. Bridged to ``max_objects`` when only one is set (see
        below).
    cuts : Sequence[_ObjectCut | Mapping[str, Any]] | None, optional
        Per-jet field cuts for object selection; dicts are coerced to
        :class:`_ObjectCut`. Default None.
    sort_by : str | None, optional
        Object-group field to sort survivors by before truncation. Default
        None (file slot order).
    sort_descending : bool, optional
        Sort direction for ``sort_by``. Default True.
    pv_class : int | None, optional
        Mapped class index identifying the primary vertex pinned at slot 0
        (validated to a non-null mapped index); ``None`` disables PV
        pinning. Default 0.
    max_objects : int | None, optional
        Max object slots retained per jet after selection. ``None``
        auto-links to ``num_objects`` (the decoder query bank).
    max_lxy_mm : float | None, optional
        |Lxy| threshold (mm) above which a vertex is re-labelled to null.
        Default None (disabled).
    lxy_field : str, optional
        Name of the Lxy field used by ``max_lxy_mm``. Default ``"Lxy"``.

    Attributes
    ----------
    object_weights : list[float]
        Per-class loss weights ordered by mapped index, derived from each
        class's optional ``weight``.

    Raises
    ------
    ConfigError
        On a missing ``null`` class, a null not mapped last, ``mapped``
        values that are not ``range(len(class_map))``, a raw value shared
        across mapped indices, a non-scalar class weight, a duplicate
        regression target, a non-positive ``num_objects`` / ``max_objects``,
        an out-of-range ``pv_class``, or an ``_ObjectCut`` with neither min
        nor max.
    """

    def __init__(
        self,
        object_class: str,
        object_id: str,
        constituent_id: str,
        class_map: Mapping[str, Mapping[str, int]],
        object_stream: str,
        constituent_stream: str,
        regression_targets: Sequence[str] | None = None,
        num_objects: int | None = None,
        cuts: Sequence[_ObjectCut | Mapping[str, Any]] | None = None,
        sort_by: str | None = None,
        sort_descending: bool = True,
        pv_class: int | None = 0,
        max_objects: int | None = None,
        max_lxy_mm: float | None = None,
        lxy_field: str = "Lxy",
    ) -> None:
        super().__init__()
        self.object_class = str(object_class)
        self.object_id = str(object_id)
        self.constituent_id = str(constituent_id)
        self.object_stream = str(object_stream)
        self.constituent_stream = str(constituent_stream)
        self._raw_to_mapped = self._checked_class_map(class_map)
        # per-class loss weights, ordered by mapped index (v1 derived property
        # MaskformerObjectConfig.object_weights). Defaults to 1.0 per class.
        self.object_weights: list[float] = self._class_weights(class_map)
        self.regression_targets: tuple[str, ...] = tuple(regression_targets or ())
        if len(set(self.regression_targets)) != len(self.regression_targets):
            raise ConfigError(
                f"MaskFormerTargets: duplicate regression targets in {self.regression_targets}"
            )

        # --- object-selection config surface -------------------------------
        # Generic per-jet field cuts. jsonargparse may hand dicts -> coerce to
        # _ObjectCut (v1 MaskformerObjectConfig.__post_init__ dict->ObjectCut).
        self.cuts: tuple[_ObjectCut, ...] = tuple(
            c if isinstance(c, _ObjectCut) else _ObjectCut(**dict(c)) for c in (cuts or ())
        )
        self.sort_by = sort_by
        self.sort_descending = bool(sort_descending)
        self.max_lxy_mm = max_lxy_mm
        self.lxy_field = str(lxy_field)

        # Object selection — and crucially the pv_class PV-pin REORDER — runs
        # ONLY when the user EXPLICITLY configured cuts / sort_by / max_objects.
        # It MUST key on the PRE-bridge `max_objects` argument: the
        # num_objects -> max_objects bridge below sets self.max_objects from
        # the decoder query bank for EVERY MaskFormer config, so gating on
        # self.max_objects would fire selection unconditionally and break
        # byte-identity (pv_class defaults to 0, so the PV-pin would silently
        # reorder slot 0). max_lxy relabel is a SEPARATE gate
        # (self.max_lxy_mm is not None).
        self._should_select: bool = (
            bool(self.cuts) or sort_by is not None or max_objects is not None
        )

        # In v2 num_objects is the decoder query bank (the declared label M
        # dim, set from mask_decoder.num_objects in convert.py) and
        # max_objects is the selection truncation count. They MUST agree:
        # declare_io produces object_class with M == num_objects while
        # _select_objects truncates to max_objects, so a config that sets BOTH
        # to different values would emit a [B, max_objects] array against a
        # declared [B, num_objects] shape. The bridge below only equalises
        # them when one is None, so guard the both-set case explicitly.
        if num_objects is not None and max_objects is not None and num_objects != max_objects:
            raise ConfigError(
                f"MaskFormerTargets: num_objects ({num_objects}) and max_objects "
                f"({max_objects}) are both set but differ — num_objects is the decoder "
                "query bank (declared label M) and max_objects is the selection "
                "truncation count; they must be equal (set only one, or set both equal)"
            )

        # legacy num_objects <-> max_objects bridge (v1 MaskformerObjectConfig
        # __post_init__): num_objects is ALSO the decoder query bank, so this
        # auto-links the selection truncation count (max_objects) to it when
        # the config leaves it unset. max_objects wins when both are set.
        if max_objects is None and num_objects is not None:
            max_objects = num_objects
        if num_objects is None and max_objects is not None:
            num_objects = max_objects
        if num_objects is not None and num_objects < 1:
            raise ConfigError(f"MaskFormerTargets: num_objects must be >= 1, got {num_objects}")
        if max_objects is not None and max_objects < 1:
            raise ConfigError(f"MaskFormerTargets: max_objects must be >= 1, got {max_objects}")
        self.num_objects = num_objects
        self.max_objects = max_objects

        # PV class must be a valid non-null mapped index (v1
        # MaskformerObjectConfig __post_init__). null_index == n_non_null.
        self.pv_class = pv_class
        if pv_class is not None:
            n_non_null = self.null_index
            if not (0 <= pv_class < n_non_null):
                raise ConfigError(
                    f"MaskFormerTargets: pv_class={pv_class} must be in [0, {n_non_null - 1}] "
                    "(non-null mapped indices; v1 configs.py)"
                )

        # Derived raw-id sets (mapped-index world -> raw-id world). PV raws:
        # every raw whose mapped == pv_class — these classes get pinned at
        # slot 0, exempt from cuts/sorts. null raw: the raw whose mapped ==
        # null_index — the sentinel written to pad slots' class field so the
        # np.select class-map maps them cleanly back to null_index.
        self._pv_raw_values: tuple[int, ...] = (
            tuple(r for r, m in self._raw_to_mapped.items() if m == self.pv_class)
            if self.pv_class is not None
            else ()
        )
        self._null_raw_value: int = next(
            r for r, m in self._raw_to_mapped.items() if m == self.null_index
        )

    @staticmethod
    def _checked_class_map(class_map: Mapping[str, Mapping[str, Any]]) -> dict[int, int]:
        """Validate the class map (null LAST, mapped == range) and return raw->mapped.

        ``null`` present (``None`` or the string ``"null"`` accepted), null
        mapped LAST, and the ``mapped`` set exactly ``range(len(class_map))``.
        A class entry's ``raw`` may be a single int or a list/tuple of ints
        that all map to the SAME ``mapped`` index — a class *merge*; merged
        raw values MUST be disjoint across classes.
        """
        if not class_map:
            raise ConfigError("MaskFormerTargets: class_map must not be empty (FD 1108)")
        # jsonargparse may parse a YAML ``null:`` key as Python None — accept both.
        names = {
            ("null" if name is None else str(name)): dict(spec) for name, spec in class_map.items()
        }
        if "null" not in names:
            raise ConfigError(
                "MaskFormerTargets: class_map must contain a 'null' (no-object) class "
                "(v1 MaskformerObjectConfig, configs.py:36)"
            )
        n = len(names)
        raw_to_mapped: dict[int, int] = {}
        for name, spec in names.items():
            if "mapped" not in spec:
                raise ConfigError(
                    f"MaskFormerTargets: class_map[{name!r}] is missing a 'mapped' index"
                )
            if name != "null" and "raw" not in spec:
                raise ConfigError(
                    f"MaskFormerTargets: class_map[{name!r}] is missing a 'raw' index "
                    "(only the 'null' class may omit it; v1 object_classes)"
                )
            mapped = int(spec["mapped"])
            # the null class's raw id defaults to -1 (v1 MaskFormer.yaml:177: null raw -1).
            # raw may be a scalar (the common case) or a list/tuple → class merge.
            raw_spec = spec.get("raw", -1)
            if isinstance(raw_spec, (list, tuple)):
                if not raw_spec:
                    raise ConfigError(
                        f"MaskFormerTargets: class_map[{name!r}] 'raw' list must not be empty"
                    )
                raws = [int(r) for r in raw_spec]
            else:
                raws = [int(raw_spec)]
            for r in raws:
                if r in raw_to_mapped and raw_to_mapped[r] != mapped:
                    raise ConfigError(
                        f"MaskFormerTargets: raw class id {r} is mapped to multiple classes "
                        f"({raw_to_mapped[r]} and {mapped}) — merged raws must be disjoint "
                        "across mapped indices (v1 object_classes)"
                    )
                raw_to_mapped[r] = mapped
        if names["null"]["mapped"] != n - 1:
            raise ConfigError(
                f"MaskFormerTargets: the 'null' class must be mapped LAST (to {n - 1}), got "
                f"{names['null']['mapped']} (v1 configs.py:39 'Null class must be last')"
            )
        if set(raw_to_mapped.values()) != set(range(n)):
            raise ConfigError(
                f"MaskFormerTargets: mapped class indices {sorted(set(raw_to_mapped.values()))} "
                f"must be exactly range({n}) (v1 configs.py:42)"
            )
        return raw_to_mapped

    @staticmethod
    def _class_weights(class_map: Mapping[str, Mapping[str, Any]]) -> list[float]:
        """Per-class loss weights ordered by mapped index; each class carries an optional
        scalar ``weight`` (default ``1.0``), ordered so the list index aligns with the
        class index the loss expects.
        """
        names = {
            ("null" if name is None else str(name)): dict(spec) for name, spec in class_map.items()
        }
        by_mapped: dict[int, float] = {}
        for name, spec in names.items():
            w = spec.get("weight", 1.0)
            if isinstance(w, (list, tuple)):
                raise ConfigError(
                    f"MaskFormerTargets: class_map[{name!r}] 'weight' must be a scalar, got {w!r} "
                    "(v1 object_weights, configs.py)"
                )
            by_mapped[int(spec["mapped"])] = float(w)
        return [by_mapped[i] for i in range(len(by_mapped))]

    @property
    def null_index(self) -> int:
        """The mapped index of the null/no-object class (max mapped value; correct
        under class merges).
        """
        return max(self._raw_to_mapped.values())

    def declare_io(self, mode: Mode) -> IO:
        """Declare object/constituent fields -> ``labels.objects.*`` (declared in ALL
        modes — demand-gated, not mode-gated; the planner prunes an unused module).
        """
        del mode
        # base fields always read. When selection is active, the cut/sort
        # fields must also be read so _select_objects can evaluate them; when
        # the Lxy relabel is active, lxy_field too. Both are added ONLY under
        # their gate, so the unconfigured path declares the IDENTICAL
        # requires as the no-selection case (byte-identity). dict.fromkeys
        # dedupes while preserving first-seen order.
        obj_field_list = [self.object_class, self.object_id, *self.regression_targets]
        if self._should_select:
            obj_field_list += [c.field for c in self.cuts]
            if self.sort_by is not None:
                obj_field_list.append(self.sort_by)
        if self.max_lxy_mm is not None:
            obj_field_list.append(self.lxy_field)
        obj_fields = tuple(dict.fromkeys(obj_field_list))
        requires = {
            f"raw.{self.object_stream}": TensorSpec(kind="data", fields=obj_fields),
            f"raw.{self.constituent_stream}": TensorSpec(
                kind="data", fields=(self.constituent_id,)
            ),
        }
        m: int | str = self.num_objects if self.num_objects is not None else sym_dim("M", self.name)
        tok = sym_dim("T", self.constituent_stream)
        produces: dict[str, TensorSpec] = {
            f"labels.{OBJECT_STREAM}.object_class": TensorSpec(
                shape=("B", m), dtype="int64", kind="label"
            ),
            f"labels.{OBJECT_STREAM}.masks": TensorSpec(
                shape=("B", m, tok), dtype="bool", kind="label"
            ),
        }
        for target in self.regression_targets:
            produces[f"labels.{OBJECT_STREAM}.{target}"] = TensorSpec(
                shape=("B", m), dtype="float32", kind="label"
            )
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def _select_objects(self, obj: np.ndarray) -> np.ndarray:
        """Per-jet object selection: cuts -> PV-pin -> sort -> truncate.

        1. CUTS — drop vertices failing any :class:`_ObjectCut`
           (``min <= field <= max``); NaN FAILS the cut; AND across cuts;
           PV exempt.
        2. PV-PIN — vertices whose raw class maps to ``pv_class`` go to
           slot 0 in input order, exempt from cuts/sorts.
        3. SORT — stable ``np.argsort`` of :attr:`sort_by` (reversed if
           :attr:`sort_descending`).
        4. TRUNCATE — ``concat([pv, non_pv])[:max_objects]``.

        Pad convention: ``valid`` is used when present, else the pad
        sentinel ``object_id == -1``; output pad slots are sentinel-filled
        (``id -> -1``, class field -> ``_null_raw_value`` -> ``null_index``).
        """
        n_jets, n_in = obj.shape
        n_out = self.max_objects if self.max_objects is not None else n_in

        # required fields must be present in the structured dtype (declare_io
        # adds them to the read set; this guards a schema-less / mis-wired read).
        needed: set[str] = {c.field for c in self.cuts}
        if self.sort_by is not None:
            needed.add(self.sort_by)
        if self._pv_raw_values:
            needed.add(self.object_class)
        missing = needed - set(obj.dtype.names or ())
        if missing:
            raise SchemaError(
                f"MaskFormerTargets: object selection needs fields {sorted(missing)} but "
                f"raw.{self.object_stream} has {obj.dtype.names}. Add them to the object "
                "stream variables/read set."
            )

        # candidate (non-pad) slots. The production v2 reader appends an
        # authoritative `valid` bool field to every assembled stream, keyed
        # exactly as upstream's `valid`; prefer it so a legitimate object
        # with object_id == -1 is NOT silently dropped. The hand-built unit
        # fixtures omit `valid`, so fall back to the v2 pad sentinel
        # object_id == -1 (INT_PAD_SENTINEL) == ~valid.
        if "valid" in (obj.dtype.names or ()):
            valid = np.asarray(obj["valid"]).astype(bool)
        else:
            valid = np.asarray(obj[self.object_id]) != -1

        # per-cut keep mask; NaN (float) FAILS the cut (strict); AND across cuts.
        keep = valid.copy()
        for cut in self.cuts:
            vals = np.asarray(obj[cut.field])
            ok = np.ones(vals.shape, dtype=bool)
            if np.issubdtype(vals.dtype, np.floating):
                ok &= ~np.isnan(vals)
            if cut.min is not None:
                ok &= vals >= cut.min
            if cut.max is not None:
                ok &= vals <= cut.max
            keep &= ok

        # PV identification (vectorised). PV is pinned at slot 0 and exempt
        # from cuts/sorts. Restricted to non-pad slots.
        if self._pv_raw_values:
            pv_mask = np.isin(np.asarray(obj[self.object_class]), self._pv_raw_values) & valid
        else:
            pv_mask = np.zeros((n_jets, n_in), dtype=bool)

        out = np.zeros((n_jets, n_out), dtype=obj.dtype)  # pad slots default 0/False
        filled = np.zeros((n_jets, n_out), dtype=bool)
        sort_vals = np.asarray(obj[self.sort_by]) if self.sort_by is not None else None
        for j in range(n_jets):
            is_pv_j = pv_mask[j]
            pv_idx = np.where(is_pv_j)[0]
            non_pv_idx = np.where(keep[j] & ~is_pv_j)[0]
            if sort_vals is not None and non_pv_idx.size:
                order = np.argsort(sort_vals[j][non_pv_idx], kind="stable")
                if self.sort_descending:
                    order = order[::-1]
                non_pv_idx = non_pv_idx[order]
            chosen = np.concatenate([pv_idx, non_pv_idx])[:n_out]
            if chosen.size:
                out[j, : chosen.size] = obj[j][chosen]
                filled[j, : chosen.size] = True

        # sentinel-fill pad slots so they don't collide with real ids/classes
        # (v1 datasets.py:148-164). 'filled' (not a valid field) marks the pads.
        pad = ~filled
        if pad.any():
            out[self.object_id][pad] = -1
            out[self.object_class][pad] = self._null_raw_value
        return out

    def process(self, batch, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        """Build the object class (atomic ``np.select`` over original raw values), the
        truth masks (``constituent_id == object_id`` on a private id copy), and the raw
        regression labels.
        """
        del rows, mode
        obj = batch.get(f"raw.{self.object_stream}")
        con = batch.get(f"raw.{self.constituent_stream}")
        out: dict[str, np.ndarray] = {}

        # Object selection (cuts -> PV-pin -> sort -> truncate). GATED: only
        # when the user explicitly configured cuts/sort_by/max_objects. When
        # OFF, `obj` is the raw stream untouched -> the blocks below are
        # byte-identical to the no-selection case. The selection rebuilds a
        # NEW [B, n_out] structured array so EVERY field (class, id,
        # regression, lxy) is permuted/truncated IN LOCKSTEP.
        if self._should_select:
            obj = self._select_objects(obj)

        # object_class: map raw -> mapped from the ORIGINAL values (atomic, no
        # sequential x[x==k]=v mutation; v1 datasets.py:641-643). Unmapped raw
        # values fall through to the null index (v1 reads only configured
        # classes, and any object whose raw class is not in the map is a
        # no-object slot).
        raw_class = np.asarray(obj[self.object_class])
        conds = [raw_class == raw for raw in self._raw_to_mapped]
        choices = [self._raw_to_mapped[raw] for raw in self._raw_to_mapped]
        object_class = np.select(conds, choices, default=self.null_index).astype(np.int64)

        # Lxy relabel (SEPARATE gate, self.max_lxy_mm is not None; v1
        # datasets.py:814-820). AFTER the class-map: vertices with
        # |Lxy| > max_lxy_mm cannot be reconstructed by the tracker -> re-label
        # to null. NaN-safe: np.abs(nan) > thr is always False, so null/pad
        # slots (NaN or 0 Lxy) are never accidentally relabelled.
        if self.max_lxy_mm is not None:
            lxy = np.asarray(obj[self.lxy_field])
            object_class[np.abs(lxy) > self.max_lxy_mm] = self.null_index
        out[f"labels.{OBJECT_STREAM}.object_class"] = object_class

        # masks: constituent_id == object_id, [B, M] x [B, T] -> [B, M, T]. v1
        # build_target_masks substitutes -1 ids with -999 IN PLACE before the
        # equality; we do it on a private COPY so the published object_class /
        # ids are untouched. The substitution makes invalid (-1) objects never
        # match a constituent (constituent ids are non-negative).
        object_ids = np.array(obj[self.object_id], copy=True)
        object_ids[object_ids == -1] = -999
        constituent_ids = np.asarray(con[self.constituent_id])
        # [B, M, 1] == [B, 1, T] -> [B, M, T]
        out[f"labels.{OBJECT_STREAM}.masks"] = object_ids[:, :, None] == constituent_ids[:, None, :]

        # raw per-object regression labels (float32; the task stacks + scales them)
        for target in self.regression_targets:
            out[f"labels.{OBJECT_STREAM}.{target}"] = np.asarray(obj[target], dtype=np.float32)
        return out
