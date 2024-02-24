import torch


class RegressionTargetScaler:
    def __init__(self, scales):
        """Generates a functional based scaler for regression targets.
        scales should be a dict of the following form:
        scales = {
            'pt': {
                'op' : 'log',
                'x_scale' : 5,
                'x_off' : 0,
                'op_scale' : 0,
                'op_off' : 0
            },
            'Lxy': {
                'op' : 'log',
                'x_scale' : 5,
                'x_off' : 0,
                'op_scale' : 0,
                'op_off' : 0
            },
            'deta' : {
                'op' : 'linear',
                'x_scale' : 1,
                'x_off' : 1,
                'op_scale' : 10,
                'op_off' : 0
            }
        }
        This would result in scaling functions of the form:
            func = lambda x: (op(x_scale * x + x_off) * op_scale + op_off)
            scale(pt) = lambda x: (log(5 * x) * 0 + 0)
            scale(Lxy) = lambda x: (log(5 * x) * 0 + 0)
            scale(deta) = lambda x: (linear(1 * x + 1) * 10 + 0)
        Only the operation is required, the rest is optional, with defaults
        of 1 for scales and 0 for offsets.
        """
        self.scales = scales

    @staticmethod
    def log_scale(x: torch.Tensor, x_scale=1, x_off=0, op_scale=1, op_off=0) -> torch.Tensor:
        return torch.log(x * x_scale + x_off) * op_scale + op_off

    @staticmethod
    def log_inverse(x: torch.Tensor, x_scale=1, x_off=0, op_scale=1, op_off=0) -> torch.Tensor:
        return (torch.exp((x - op_off) / op_scale) - x_off) / x_scale

    @staticmethod
    def exp_scale(x: torch.Tensor, x_scale=1, x_off=0, op_scale=1, op_off=0) -> torch.Tensor:
        return torch.exp(x * x_scale + x_off) * op_scale + op_off

    @staticmethod
    def exp_inverse(x: torch.Tensor, x_scale=1, x_off=0, op_scale=1, op_off=0) -> torch.Tensor:
        return (torch.log((x - op_off) / op_scale) - x_off) / x_scale

    @staticmethod
    def linear_scale(x: torch.Tensor, x_scale=1, x_off=0, op_scale=1, op_off=0) -> torch.Tensor:
        return (x * x_scale + x_off) * op_scale + op_off

    @staticmethod
    def linear_inverse(x: torch.Tensor, x_scale=1, x_off=0, op_scale=1, op_off=0) -> torch.Tensor:
        return ((x - op_off) / op_scale - x_off) / x_scale

    def scale(self, target: str, values: torch.Tensor) -> torch.Tensor:
        params = self.scales[target].copy()
        op = params.pop("op")
        if op == "log":
            return self.log_scale(values, **params)
        if op == "exp":
            return self.exp_scale(values, **params)
        if op == "linear":
            return self.linear_scale(values, **params)

        raise ValueError("Unknown operation: {}".format(params["op"]))

    def inverse(self, target: str, values: torch.Tensor) -> torch.Tensor:
        params = self.scales[target].copy()
        op = params.pop("op")
        if op == "log":
            return self.log_inverse(values, **params)
        if op == "exp":
            return self.exp_inverse(values, **params)
        if op == "linear":
            return self.linear_inverse(values, **params)

        raise ValueError("Unknown operation: {}".format(params["op"]))
