from dataclasses import dataclass


@dataclass
class MaskformerObjectConfig:
    """Sub-config for the Maskformer config, containing information about the object we
    wish to reconstruct.

    Attributes
    ----------
        name (str): Name of the container containing the object
        id_label (str): Name of the variable in the global object to use as an ID
        class_label (str): Name of the variable in the global object to use as a class label
        object_classes: A dictionary of the form:
            class_name_1 (str):
                raw (int): Value that corresponds to value in 'class_label'
                mapped (int): Value to map to (such that all mapped are in {range(num_classes)})
            class_name_2:
                ...
            null:
                mapped: len(object_classes) - 1
    """

    name: str
    id_label: str
    class_label: str | None = None
    object_classes: dict[str, dict[str, int]] | None = None

    def __post_init__(self):
        if self.object_classes:
            # When reading in 'null' jsonparse seems to cast this to 'None', so here we
            # cast it back
            assert None in self.object_classes, "Null class must be present"
            self.object_classes["null"] = self.object_classes.pop(None)

            assert (
                self.object_classes["null"]["mapped"] == len(self.object_classes) - 1
            ), "Null class must be last"
            assert set(self.class_map.values()) == set(range(len(self.object_classes)))

    @property
    def num_classes(self) -> int:
        if not self.object_classes:
            raise ValueError("No object classes defined")
        return len(self.object_classes)

    @property
    def num_not_null_classes(self) -> int:
        if not self.object_classes:
            raise ValueError("No object classes defined")
        return len(self.object_classes) - 1

    @property
    def null_index(self) -> int:
        return self.num_not_null_classes

    @property
    def class_map(self) -> dict[int, int]:
        """Returns a dictionary mapping raw class labels to mapped class labels."""
        if not self.object_classes:
            raise ValueError("No object classes defined")
        return {v["raw"]: v["mapped"] for v in self.object_classes.values()}

    @property
    def class_names(self) -> list[str]:
        """Returns a list of each class name, ordered by the mapped value."""
        if not self.object_classes:
            raise ValueError("No object classes defined")
        sortedmap = sorted(self.object_classes.items(), key=lambda x: x[1]["mapped"])
        return [n for n, _ in sortedmap]


@dataclass
class MaskformerConfig:
    """Configuration for Maskformer model.

    Attributes
    ----------
        objects (MaskformerObjectConfig): Configuration for the objects to be reconstructed
        constituent (MaskformerObjectConfig): Configuration for the constituents of the objects
            The 'class_label' and 'object_classes' attributes are not used here.


    """

    object: MaskformerObjectConfig
    constituent: MaskformerObjectConfig
