import warnings
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
        """Returns a dictionary mapping raw class labels to mapped class labels.

        Raises
        ------
        ValueError
            If object classes are not defined
        """
        if not self.object_classes:
            raise ValueError("No object classes defined")
        return {v["raw"]: v["mapped"] for v in self.object_classes.values()}

    @property
    def class_names(self) -> list[str]:
        """Returns a list of each class name, ordered by the mapped value.

        Raises
        ------
        ValueError
            If object classes are not defined
        """
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

    def __post_init__(self):
        if isinstance(self.object, dict):
            self.object = MaskformerObjectConfig(**self.object)
        if isinstance(self.constituent, dict):
            self.constituent = MaskformerObjectConfig(**self.constituent)


@dataclass
class LabellerConfig:
    """Sub-config for the Labeller config.

    Attributes
    ----------
    use_labeller: boolean
        Confirms if user wants to use the labeller on-the-fly
    class_names: list
        Names of the new output classes
    require_labels: boolean
        Confirms if all jets are required to be relabelled,
        or if it is fine for some to remain unlabelled
    """

    use_labeller: bool
    require_labels: bool
    class_names: list[str]

    def __init__(self, use_labeller, class_names=None, require_labels=True):
        self.use_labeller = use_labeller
        self.class_names = class_names
        self.require_labels = require_labels
        if not self.use_labeller:
            warnings.warn(
                "Use Labeller set to False. "
                "If you've setup other labeller config parameters those will be ignored",
                stacklevel=2,
            )
            self.require_labels = False
            self.class_names = []
        if self.use_labeller and not self.class_names:
            raise ValueError("Specify target classes for relabelling")
        if self.use_labeller and self.class_names:
            self.class_names = class_names
            warnings.warn(
                "Use Labeller set to True." f"Relabelling to target classes {class_names}",
                stacklevel=2,
            )
            if self.require_labels:
                self.require_labels = require_labels
