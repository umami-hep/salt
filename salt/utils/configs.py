import warnings
from dataclasses import dataclass


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
    """

    name: str
    id_label: str
    class_label: str | None = None
    object_classes: dict[str | None, dict[str, int]] | None = None

    def __post_init__(self) -> None:
        """Ensure that the object_classes dictionary is valid."""
        if self.object_classes:
            # When reading in 'null', jsonargparse may cast it to None, so cast back
            assert None in self.object_classes, "Null class must be present"
            self.object_classes["null"] = self.object_classes.pop(None)

            assert (
                self.object_classes["null"]["mapped"] == len(self.object_classes) - 1
            ), "Null class must be last"
            assert set(self.class_map.values()) == set(range(len(self.object_classes)))

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
        return {v["raw"]: v["mapped"] for v in self.object_classes.values()}

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
