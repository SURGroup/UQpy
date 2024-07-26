import torch
import torch.nn.functional as F
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import BayesianLayer
from UQpy.utilities.ValidationTypes import PositiveInteger
from typing import Union


class BayesianFourier2d(BayesianLayer):
    def __init__(
        self,
        width: PositiveInteger,
        modes: tuple[PositiveInteger, PositiveInteger],
        bias: bool = True,
        priors: dict = None,
        sampling: bool = True,
        device: Union[torch.device, str] = None,
    ):
        """

        :param width:
        :param modes:
        :param bias:
        :param priors:
        :param sampling:
        :param device:
        """
        kernel_size = (1, 1)
        parameter_shapes = {
            "weight_spectral_1": (self.width, self.width, *modes),
            "weight_spectral_2": (self.width, self.width, *modes),
            "weight_conv": (self.width, self.width, *kernel_size),
            "bias_conv": (width, width) if bias else None,
        }
        super().__init__(
            parameter_shapes,
            priors,
            sampling,
            device,
            dtype=(torch.cfloat, torch.cfloat, torch.float, torch.float),
        )
        self.width = width
        self.modes = modes
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        w1, w2, weight_conv, bias_conv = self.get_bayesian_weights()
        return func.spectral_conv2d(x, (w1, w2), self.width, self.modes) + F.conv2d(
            x, weight_conv, bias_conv
        )

    def extra_repr(self):
        return (
            f"width={self.width}, "
            f"modes={self.modes}, "
            f"priors={self.priors}, "
            f"sampling={self.sampling}"
        )
