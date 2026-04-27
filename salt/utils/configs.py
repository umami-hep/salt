import warnings
from dataclasses import dataclass


@dataclass
class ObjectCut:
    """A single field-bound cut applied to MaskFormer truth objects.

    A vertex is *kept* if ``min <= batch[field] <= max``. NaN values FAIL
    the cut (the vertex is dropped). PV (identified via ``pv_class`` on
    :class:`MaskformerObjectConfig`) is exempt from cuts.

    Attributes
    ----------
    field : str
        Name of the field in the objects structured array to cut on. Must
        appear in ``data.variables.objects`` so it is present in the
        loaded dtype.
    min : float | None
        Inclusive lower bound. ``None`` disables the lower bound.
    max : float | None
        Inclusive upper bound. ``None`` disables the upper bound.
    """

    field: str
    min: float | None = None
    max: float | None = None

    def __post_init__(self) -> None:
        if self.min is None and self.max is None:
            raise ValueError(
                f"ObjectCut on field '{self.field}' must set at least one of min/max"
            )


@dataclass
class MaskformerObjectConfig:
    """Sub-configuration for Maskformer describing the object to reconstruct.

    Attributes
    ----------
    name : str
        Name of the container containing the object.
    id_label : str
        Name of the variable in the global object to use as an ID.
    class_label : str | None
        Name of the variable in the global object to use as a class label.
    object_classes : dict[str | None, dict[str, int]] | None
        Mapping of class names to raw/mapped indices. Expected format::

            {
                "class_name_1": {"raw": int, "mapped": int},
                "class_name_2": {"raw": int, "mapped": int},
                "null": {"mapped": len(object_classes) - 1}
            }
    max_lxy_mm : float | None
        Optional threshold on |Lxy| (mm). Vertices failing this cut are
        *re-labelled to null* in :meth:`process_labels` (legacy behaviour
        kept for backward compatibility). For *physical exclusion* (drop
        the vertex and free its slot) use ``cuts`` instead, e.g.
        ``cuts: [{field: Lxy, max: 200}]``. ``None`` (default) disables
        the cut.
    lxy_field : str
        Name of the Lxy field in the objects structured array. Defaults to
        ``"Lxy"``. Must appear in ``data.variables.objects`` in the training
        config.
    cuts : list[ObjectCut] | None
        Generic per-jet field cuts applied during data loading. A vertex
        is *dropped* if any cut fails — survivors are compacted into the
        leading output slots. NaN values fail (strict). PV is exempt.
        ``None`` (default) disables generic cuts. Complements (does not
        replace) ``max_lxy_mm``: ``max_lxy_mm`` *relabels* to null (keeps
        the slot), ``cuts`` *drops* the vertex (frees the slot).
    sort_by : str | None
        Field name to sort surviving non-PV vertices by, before applying
        truncation. Must appear in ``data.variables.objects``. ``None``
        (default) preserves the original HDF5 slot order.
    sort_descending : bool
        Whether to sort by ``sort_by`` in descending order (largest first).
        Defaults to ``True``.
    pv_class : int | None
        Mapped class index that identifies the primary vertex (PV).
        Vertices whose ``class_label`` raw value maps to this index are
        always pinned at output slot 0 and exempt from cuts/sorts. Set to
        ``None`` to disable PV pinning entirely (no slot is special).
        Defaults to ``0``.
    max_objects : int | None
        Maximum number of object slots to retain per jet, applied after
        cuts, PV-pinning, and sorting. ``None`` (default) keeps all
        original slots. May be auto-linked from
        ``model.mask_decoder.num_objects`` by the CLI when left ``None``.
    num_objects : int | None
        DEPRECATED alias for ``max_objects``, kept for backward
        compatibility with older configs. If both are set, ``max_objects``
        wins. Will be removed in a future release.
    """

    name: str
    id_label: str
    class_label: str | None = None
    object_classes: dict[str | None, dict[str, float | int | list[int]]] | None = None
    max_lxy_mm: float | None = None
    lxy_field: str = "Lxy"
    cuts: list[ObjectCut] | None = None
    sort_by: str | None = None
    sort_descending: bool = True
    pv_class: int | None = 0
    max_objects: int | None = None
    num_objects: int | None = None  # legacy alias for max_objects

    def __post_init__(self) -> None:
        """Ensure that the object_classes dictionary is valid."""
        if self.object_classes:
            # When reading in 'null', jsonargparse may cast it to None, so cast back
            if None in self.object_classes:
                self.object_classes["null"] = self.object_classes.pop(None)
            elif "null" not in self.object_classes:
                raise ValueError("Null class must be present in object_classes")

            assert self.object_classes["null"]["mapped"] == len(self.object_classes) - 1, (
                "Null class must be last"
            )
            assert set(self.class_map.values()) == set(range(len(self.object_classes)))

            for k, v in self.object_classes.items():
                if isinstance(v["raw"], float):
                    v["raw"] = int(v["raw"])

        # jsonargparse may pass dicts → coerce to ObjectCut
        if self.cuts:
            self.cuts = [
                ObjectCut(**c) if isinstance(c, dict) else c for c in self.cuts
            ]

        # legacy num_objects → max_objects bridge (max_objects wins if both set)
        if self.max_objects is None and self.num_objects is not None:
            self.max_objects = self.num_objects
        # keep num_objects in sync so any downstream code that still reads
        # it sees the resolved value.
        if self.num_objects is None and self.max_objects is not None:
            self.num_objects = self.max_objects

        # Validate pv_class is a valid non-null mapped index
        if self.pv_class is not None and self.object_classes:
            n_non_null = len(self.object_classes) - 1
            if not (0 <= self.pv_class < n_non_null):
                raise ValueError(
                    f"pv_class={self.pv_class} must be in [0, {n_non_null - 1}] "
                    f"(non-null mapped indices)"
                )

    @property
    def num_classes(self) -> int:
        """Return the total number of classes including null.

        Returns
        -------
        int
            Total number of classes.

        Raises
        ------
        ValueError
            If no object classes are defined.
        """
        if not self.object_classes:
            raise ValueError("No object classes defined")
        return len(self.object_classes)

    @property
    def object_weights(self) -> list[float]:
        """Returns a list of weights for each class, ordered by the mapped value.

        Raises
        ------
        ValueError
            If object classes are not defined
        """
        if not self.object_classes:
            raise ValueError("No object classes defined")
        return [v.get("weight", 1.0) for v in self.object_classes.values()]

    @property
    def num_not_null_classes(self) -> int:
        """Return the number of non-null classes.

        Returns
        -------
        int
            Number of non-null classes.

        Raises
        ------
        ValueError
            If no object classes are defined.
        """
        if not self.object_classes:
            raise ValueError("No object classes defined")
        return len(self.object_classes) - 1

    @property
    def null_index(self) -> int:
        """Return the mapped index of the null class.

        Returns
        -------
        int
            Index of the null class.
        """
        return self.num_not_null_classes

    @property
    def class_map(self) -> dict[int, int]:
        """Return mapping from raw class labels to mapped class labels.

        Returns
        -------
        dict[int, int]
            Dictionary mapping raw indices to mapped indices.

        Raises
        ------
        ValueError
            If no object classes are defined.
        """
        if not self.object_classes:
            raise ValueError("No object classes defined")
        def make_tuple(v):
            if isinstance(v, (list, tuple)):
                assert all(isinstance(i, int) for i in v), "All values must be integers"
                return tuple(v)
            elif isinstance(v, int):
                return (v,)

            else:

                return (int(v),)
        return {make_tuple(v["raw"]): v["mapped"] for v in self.object_classes.values()}

    @property
    def class_names(self) -> list[str]:
        """Return class names ordered by their mapped value.

        Returns
        -------
        list[str]
            List of class names ordered by mapped index.

        Raises
        ------
        ValueError
            If no object classes are defined.
        """
        if not self.object_classes:
            raise ValueError("No object classes defined")
        sortedmap = sorted(self.object_classes.items(), key=lambda x: x[1]["mapped"])
        return [n for n, _ in sortedmap]  # type: ignore[misc]

    @property
    def pv_raw_values(self) -> tuple[int, ...] | None:
        """Raw class_label value(s) that map to ``pv_class``.

        Returns
        -------
        tuple[int, ...] | None
            Tuple of raw class label values that map to ``pv_class``, or
            ``None`` if ``pv_class`` or ``object_classes`` is unset.

        Raises
        ------
        ValueError
            If ``pv_class`` is set but no raw values map to it.
        """
        if self.pv_class is None or not self.object_classes:
            return None
        for raw_tuple, mapped in self.class_map.items():
            if mapped == self.pv_class:
                return raw_tuple
        raise ValueError(
            f"No raw values in class_map map to pv_class={self.pv_class}"
        )

    @property
    def null_raw_value(self) -> int | None:
        """First raw value that maps to the null class, or ``None``.

        Returns
        -------
        int | None
            First raw value mapped to the null class, useful as a sentinel
            for pad-slot ``class_label`` so :meth:`process_labels` maps it
            cleanly to :attr:`null_index`. Returns ``None`` if
            ``object_classes`` is unset.
        """
        if not self.object_classes:
            return None
        null_raw = self.object_classes["null"].get("raw", -1)
        if isinstance(null_raw, (list, tuple)):
            return int(null_raw[0])
        return int(null_raw)


@dataclass
class MaskformerConfig:
    """Configuration for the Maskformer model.

    Attributes
    ----------
    object : MaskformerObjectConfig
        Configuration for the reconstructed objects.
    constituent : MaskformerObjectConfig
        Configuration for the object constituents. The ``class_label`` and
        ``object_classes`` attributes are not used here.
    """

    object: MaskformerObjectConfig
    constituent: MaskformerObjectConfig

    def __post_init__(self) -> None:
        """Convert dicts to MaskformerObjectConfig if necessary."""
        if isinstance(self.object, dict):
            self.object = MaskformerObjectConfig(**self.object)
        if isinstance(self.constituent, dict):
            self.constituent = MaskformerObjectConfig(**self.constituent)


@dataclass
class LabellerConfig:
    """Configuration for the on-the-fly labeller.

    Attributes
    ----------
    use_labeller : bool
        Whether to use the labeller on-the-fly.
    class_names : list[str] | None
        Names of the output classes after relabelling.
    require_labels : bool
        Whether all jets are required to be relabelled (``True``) or if some may
        remain unlabelled (``False``).

    Raises
    ------
    ValueError
        If the taget is not specified
    """

    use_labeller: bool
    class_names: list[str] | None = None
    require_labels: bool = True

    def __post_init__(self) -> None:
        if not self.use_labeller:
            warnings.warn(
                "Use Labeller set to False. "
                "If you've set up other labeller config parameters they will be ignored.",
                stacklevel=2,
            )
            self.class_names = []  # always a list
            self.require_labels = False
            return

        # From here, use_labeller is True
        if not self.class_names:
            raise ValueError("Specify target classes for relabelling")

        # Now we know class_names is a non-empty list[str]
        warnings.warn(
            f"Use Labeller set to True. Relabelling to target classes {self.class_names}",
            stacklevel=2,
        )
