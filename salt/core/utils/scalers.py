from typing import Any

import torch


class RegressionTargetScaler:
    """Functional-based scaler for regression targets.

    This class applies configurable transformations (log, exp, linear) to
    regression targets and provides both forward (scale) and inverse
    transformations. Scaling parameters are specified via a dictionary
    mapping each target name to a configuration.

    Parameters
    ----------
    scales : dict[str, dict[str, Any]]
        Dictionary defining the scaling operations. Each key is a target
        name, and each value is a dictionary with keys:

        - ``op`` (str): Operation type ("log", "exp", or "linear").
        - ``x_scale`` (float, optional): Multiplier for input (default 1).
        - ``x_off`` (float, optional): Offset for input (default 0).
        - ``op_scale`` (float, optional): Multiplier after op (default 1).
        - ``op_off`` (float, optional): Offset after op (default 0).

    Example
    -------
    >>> scales = {
    ...     "pt": {"op": "log", "x_scale": 5},
    ...     "Lxy": {"op": "log", "x_scale": 5},
    ...     "deta": {"op": "linear", "x_scale": 1, "x_off": 1, "op_scale": 10},
    ... }
    >>> scaler = RegressionTargetScaler(scales)
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> scaler.scale("pt", x)
    tensor([...])
    """

    def __init__(self, scales: dict[str, dict[str, Any]]) -> None:
        self.scales = scales

    @staticmethod
    def log_scale(
        x: torch.Tensor,
        x_scale: float = 1,
        x_off: float = 0,
        op_scale: float = 1,
        op_off: float = 0,
    ) -> torch.Tensor:
        """Apply logarithmic scaling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to transform.
        x_scale : float, optional
            Scaling factor applied before log (default 1).
        x_off : float, optional
            Offset applied before log (default 0).
        op_scale : float, optional
            Scaling factor applied after log (default 1).
        op_off : float, optional
            Offset applied after log (default 0).

        Returns
        -------
        torch.Tensor
            Log-scaled tensor.
        """
        return torch.log(x * x_scale + x_off) * op_scale + op_off

    @staticmethod
    def log_inverse(
        x: torch.Tensor,
        x_scale: float = 1,
        x_off: float = 0,
        op_scale: float = 1,
        op_off: float = 0,
    ) -> torch.Tensor:
        """Invert logarithmic scaling.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to invert from log scaling.
        x_scale : float, optional
            Scaling factor used before log (default 1).
        x_off : float, optional
            Offset used before log (default 0).
        op_scale : float, optional
            Scaling factor used after log (default 1).
        op_off : float, optional
            Offset used after log (default 0).

        Returns
        -------
        torch.Tensor
            Tensor mapped back to original scale.
        """
        return (torch.exp((x - op_off) / op_scale) - x_off) / x_scale

    @staticmethod
    def exp_scale(
        x: torch.Tensor,
        x_scale: float = 1,
        x_off: float = 0,
        op_scale: float = 1,
        op_off: float = 0,
    ) -> torch.Tensor:
        """Apply exponential scaling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to transform.
        x_scale : float, optional
            Scaling factor applied before exp (default 1).
        x_off : float, optional
            Offset applied before exp (default 0).
        op_scale : float, optional
            Scaling factor applied after exp (default 1).
        op_off : float, optional
            Offset applied after exp (default 0).

        Returns
        -------
        torch.Tensor
            Exponentially-scaled tensor.
        """
        return torch.exp(x * x_scale + x_off) * op_scale + op_off

    @staticmethod
    def exp_inverse(
        x: torch.Tensor,
        x_scale: float = 1,
        x_off: float = 0,
        op_scale: float = 1,
        op_off: float = 0,
    ) -> torch.Tensor:
        """Invert exponential scaling.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to invert from exponential scaling.
        x_scale : float, optional
            Scaling factor used before exp (default 1).
        x_off : float, optional
            Offset used before exp (default 0).
        op_scale : float, optional
            Scaling factor used after exp (default 1).
        op_off : float, optional
            Offset used after exp (default 0).

        Returns
        -------
        torch.Tensor
            Tensor mapped back to original scale.
        """
        return (torch.log((x - op_off) / op_scale) - x_off) / x_scale

    @staticmethod
    def linear_scale(
        x: torch.Tensor,
        x_scale: float = 1,
        x_off: float = 0,
        op_scale: float = 1,
        op_off: float = 0,
    ) -> torch.Tensor:
        """Apply linear scaling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to transform.
        x_scale : float, optional
            Scaling factor applied before linear op (default 1).
        x_off : float, optional
            Offset applied before linear op (default 0).
        op_scale : float, optional
            Scaling factor applied after linear op (default 1).
        op_off : float, optional
            Offset applied after linear op (default 0).

        Returns
        -------
        torch.Tensor
            Linearly-scaled tensor.
        """
        return (x * x_scale + x_off) * op_scale + op_off

    @staticmethod
    def linear_inverse(
        x: torch.Tensor,
        x_scale: float = 1,
        x_off: float = 0,
        op_scale: float = 1,
        op_off: float = 0,
    ) -> torch.Tensor:
        """Invert linear scaling.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to invert from linear scaling.
        x_scale : float, optional
            Scaling factor used before linear op (default 1).
        x_off : float, optional
            Offset used before linear op (default 0).
        op_scale : float, optional
            Scaling factor used after linear op (default 1).
        op_off : float, optional
            Offset used after linear op (default 0).

        Returns
        -------
        torch.Tensor
            Tensor mapped back to original scale.
        """
        return ((x - op_off) / op_scale - x_off) / x_scale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def scale(self, target: str, values: torch.Tensor) -> torch.Tensor:
        """Scale values for a given target according to its config.

        Parameters
        ----------
        target : str
            Target name corresponding to a key in ``self.scales``.
        values : torch.Tensor
            Tensor of values to scale.

        Returns
        -------
        torch.Tensor
            Scaled tensor.

        Raises
        ------
        ValueError
            If the operation is not recognized.
        """
        params = self.scales[target].copy()
        op = params.pop("op")
        if op == "log":
            return self.log_scale(values, **params)
        if op == "exp":
            return self.exp_scale(values, **params)
        if op == "linear":
            return self.linear_scale(values, **params)
        raise ValueError(f"Unknown operation: {op}")

    def inverse(self, target: str, values: torch.Tensor) -> torch.Tensor:
        """Apply the inverse scaling transformation for a given target.

        Parameters
        ----------
        target : str
            Target name corresponding to a key in ``self.scales``.
        values : torch.Tensor
            Tensor of scaled values to invert.

        Returns
        -------
        torch.Tensor
            Tensor mapped back to original space.

        Raises
        ------
        ValueError
            If the operation is not recognized.
        """
        params = self.scales[target].copy()
        op = params.pop("op")
        if op == "log":
            return self.log_inverse(values, **params)
        if op == "exp":
            return self.exp_inverse(values, **params)
        if op == "linear":
            return self.linear_inverse(values, **params)
        raise ValueError(f"Unknown operation: {op}")
