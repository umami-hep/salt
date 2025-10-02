from collections.abc import Mapping

import numpy as np


def get_structured_input_dict(
    input_dict: Mapping[str, np.ndarray],
    variable_map: Mapping[str, list[str]],
    global_object: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Convert flat input arrays into a structured dictionary by variable name.

    Parameters
    ----------
    input_dict : Mapping[str, np.ndarray]
        Mapping from object type (e.g. ``"jets"`` or ``"tracks"``) to a 2D array
        of features. Each row corresponds to an instance, each column to a feature.
    variable_map : Mapping[str, list[str]]
        Mapping from object type to the list of variable names that correspond to
        the columns of ``input_dict``.
    global_object : str
        Name of the global object to process. Currently only this object type is
        supported (non-global objects may be added in the future).

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        A nested dictionary where:
        - First-level keys are object types (currently only ``global_object``).
        - Second-level keys are variable names from ``variable_map``.
        - Values are 1D arrays (columns of ``input_dict[global_object]``) corresponding
          to each variable.

    Notes
    -----
    Currently only the global object is processed. In the future this could be
    extended to handle non-global objects as well.
    """
    structured_input_dict: dict[str, dict[str, np.ndarray]] = {}
    for obj_type in [global_object]:  # TODO @wlai: extend to non-global objects
        structured_input_dict[obj_type] = {}
        for i, var in enumerate(variable_map[obj_type]):
            structured_input_dict[obj_type][var] = input_dict[obj_type][:, i]
    return structured_input_dict
