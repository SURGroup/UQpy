import torch
import torch.nn.functional as F
from UQpy.scientific_machine_learning.baseclass import DropoutActivationFunction


class Dropout(DropoutActivationFunction):
    """Randomly zero out elements."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: Input tensor
        :return: Output tensor
        """
        return F.dropout(x, self.p, self.dropping, self.inplace)


class Dropout1d(DropoutActivationFunction):
    """Randomly zero out entire 1D feature maps."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: Input tensor
        :return: Output tensor
        """
        return F.dropout1d(x, self.p, self.dropping, self.inplace)


class Dropout2d(DropoutActivationFunction):
    """Randomly zero out entire 2D feature maps."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: Input tensor
        :return: Output tensor
        """
        return F.dropout2d(x, self.p, self.dropping, self.inplace)


class Dropout3d(DropoutActivationFunction):
    """Randomly zero out entire 3D feature maps."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: Input tensor
        :return: Output tensor
        """
        return F.dropout3d(x, self.p, self.dropping, self.inplace)
