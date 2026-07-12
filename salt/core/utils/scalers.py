from typing import Any

import torch


class RegressionTargetScaler:
    """Functional-based scaler for regression targets.

    Applies configurable transformations (log, exp, linear) to regression
    targets, with forward (`scale`) and inverse (`inverse`) transforms.
    ``scales`` maps each target name to a config dict: ``op`` ("log"/"exp"/
    "linear"), plus optional ``x_scale``/``x_off``/``op_scale``/``op_off``
    (each default to the identity value).
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
        """Log scaling: ``log(x * x_scale + x_off) * op_scale + op_off``."""
        return torch.log(x * x_scale + x_off) * op_scale + op_off

    @staticmethod
    def log_inverse(
        x: torch.Tensor,
        x_scale: float = 1,
        x_off: float = 0,
        op_scale: float = 1,
        op_off: float = 0,
    ) -> torch.Tensor:
        """Invert `log_scale`."""
        return (torch.exp((x - op_off) / op_scale) - x_off) / x_scale

    @staticmethod
    def exp_scale(
        x: torch.Tensor,
        x_scale: float = 1,
        x_off: float = 0,
        op_scale: float = 1,
        op_off: float = 0,
    ) -> torch.Tensor:
        """Exponential scaling: ``exp(x * x_scale + x_off) * op_scale + op_off``."""
        return torch.exp(x * x_scale + x_off) * op_scale + op_off

    @staticmethod
    def exp_inverse(
        x: torch.Tensor,
        x_scale: float = 1,
        x_off: float = 0,
        op_scale: float = 1,
        op_off: float = 0,
    ) -> torch.Tensor:
        """Invert `exp_scale`."""
        return (torch.log((x - op_off) / op_scale) - x_off) / x_scale

    @staticmethod
    def linear_scale(
        x: torch.Tensor,
        x_scale: float = 1,
        x_off: float = 0,
        op_scale: float = 1,
        op_off: float = 0,
    ) -> torch.Tensor:
        """Linear scaling: ``(x * x_scale + x_off) * op_scale + op_off``."""
        return (x * x_scale + x_off) * op_scale + op_off

    @staticmethod
    def linear_inverse(
        x: torch.Tensor,
        x_scale: float = 1,
        x_off: float = 0,
        op_scale: float = 1,
        op_off: float = 0,
    ) -> torch.Tensor:
        """Invert `linear_scale`."""
        return ((x - op_off) / op_scale - x_off) / x_scale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def scale(self, target: str, values: torch.Tensor) -> torch.Tensor:
        """Scale values for `target` according to its config in ``self.scales``.

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
        """Apply the inverse scaling transformation for `target`.

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
