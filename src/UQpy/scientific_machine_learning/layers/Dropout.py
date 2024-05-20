import torch
import torch.nn as nn
from beartype import beartype
from beartype.vale import Is
from typing import Annotated
from UQpy.scientific_machine_learning.baseclass.Layer import Layer


#@beartype
class Dropout(Layer):

    def __init__(
        self,
        drop_rate: Annotated[float, Is[lambda p: 0 <= p <= 1]] = 0.5,
        drop_type: int = 0,
        drop: bool = True,
        **kwargs
    ):
        """Initialize a dropout layer to randomly zero components of a tensor

        :param drop_rate: Probability of a dropout occurring
        :param drop_type: If ``0`` drop elements, ``1`` drops vectors, ``2`` drops channels, ``3`` drops 3D
        :param drop: IF True apply drop out
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.drop_rate = drop_rate
        self.drop_type = drop_type
        self.drop = drop
        if self.drop_type == 0:
            self.dropout_function = nn.Dropout(self.drop_rate)
        elif self.drop_type == 1:
            self.dropout_function = nn.Dropout(self.drop_rate)
        elif self.drop_type == 2:
            self.dropout_function = nn.Dropout(self.drop_rate)
        elif self.drop_type == 3:
            self.dropout_function = nn.Dropout3d(self.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computational call

        Note: Function behavior is determined by the dimensions of ``x`` as N x C x H x W

        :param x:
        :return:
        """
        return self.dropout_function(x) if self.drop else x

    def drop(self, mode: bool = True):
        """Set sampling mode.

        :param mode: If ``True``, layer parameters are dropped.
        """
        self.drop = mode

    def extra_repr(self) -> str:
        return f"drop_rate={self.drop_rate}, drop_type={self.drop_type}"
