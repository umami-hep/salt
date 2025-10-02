from __future__ import annotations

from typing import TypedDict

import numpy as np


class NoiseSpec(TypedDict):
    """Specification for applying Gaussian noise to one field.

    Attributes
    ----------
    variable : str
        Name of the field within the structured array that should receive noise
        (e.g., ``"pt"``, ``"eta"``).
    mean : float
        Mean of the Gaussian noise distribution.
    std : float
        Standard deviation of the Gaussian noise distribution.
    """

    variable: str
    mean: float
    std: float


class GaussianNoise:
    """Gaussian noise generator with per-input/variable specifications.

    Parameters
    ----------
    noise_params : dict[str, list[NoiseSpec]] | None, optional
        Mapping from an input type (e.g. ``"tracks"``, ``"jets"``) to a list of
        :class:`NoiseSpec` entries. If ``None``, no noise is applied.
    """

    def __init__(self, noise_params: dict[str, list[NoiseSpec]] | None = None) -> None:
        self.noise_params: dict[str, list[NoiseSpec]] = noise_params or {}

    def add_noise(self, data: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Add multiplicative Gaussian noise to an array.

        Parameters
        ----------
        data : np.ndarray
            Array to which noise will be applied. This array is **not** modified in-place;
            a noisy copy is returned.
        mean : float
            Mean of the Gaussian noise distribution.
        std : float
            Standard deviation of the Gaussian noise distribution.

        Returns
        -------
        np.ndarray
            A new array with multiplicative noise applied: ``data * N(mean, std)``.
        """
        rng = np.random.default_rng()
        noise: np.ndarray = rng.normal(mean, std, size=data.shape)
        return data * noise

    def __call__(self, data: np.ndarray, input_type: str) -> np.ndarray:
        """Apply Gaussian noise to all variables for the given input type.

        Parameters
        ----------
        data : np.ndarray
            Structured/record array containing the input variables for ``input_type``.
            Fields are accessed as ``data[variable]``.
        input_type : str
            Name of the input group (e.g. ``"jets"`` or ``"tracks"``) whose variables
            should receive noise according to the stored specifications.

        Returns
        -------
        np.ndarray
            The modified data array with noise applied to the specified fields.
        """
        for spec in self.noise_params.get(input_type, []):
            var = spec["variable"]
            mean = spec["mean"]
            std = spec["std"]

            # Apply noise to this field and write back
            noisy = self.add_noise(data[var].astype(np.float32, copy=False), mean, std)
            data[var] = noisy
        return data
