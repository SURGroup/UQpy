import torch
import torch.nn as nn
import torch.nn.functional as F
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
        dropping: bool = True,
        **kwargs
    ):
        """Initialize a dropout layer to randomly zero components of a tensor

        :param drop_rate: Probability of a dropout occurring
        :param drop_type: If ``0`` drop elements, ``1`` drops vectors, ``2`` drops arrays, ``3`` drops channels
        :param dropping: If ``True`` randomly set elements to zero. If ``False`` behaves as identify function.
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.drop_rate = drop_rate
        self.drop_type = drop_type
        self.dropping = dropping
        # if self.drop_type == 0:
        #     self.dropout_function = F.dropout(p=self.drop_rate)  # nn.Dropout(self.drop_rate)
        # elif self.drop_type == 1:
        #     self.dropout_function = nn.Dropout1d(self.drop_rate)
        # elif self.drop_type == 2:
        #     self.dropout_function = nn.Dropout2d(self.drop_rate)
        # elif self.drop_type == 3:
        #     self.dropout_function = nn.Dropout3d(self.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computational call

        :param x: Input tensor
        :return: Output tensor
        """
        if not self.dropping:
            return x

        if self.drop_type == 0:
            return F.dropout(x, p=self.drop_rate)
        elif self.drop_type == 1:
            return F.dropout1d(x, p=self.drop_rate)
        elif self.drop_type == 2:
            return F.dropout2d(x, p=self.drop_rate)
        elif self.drop_type == 3:
            return F.dropout3d(x, p=self.drop_rate)
        # return self.dropout_function(x) if self.dropping else x

    def drop(self, mode: bool = True):
        """Set dropping mode.

        :param mode: If ``True``, layer parameters are dropped.
        """
        self.dropping = mode

    def extra_repr(self) -> str:
        return f"drop_rate={self.drop_rate}, drop_type={self.drop_type}, dropping={self.dropping}"
