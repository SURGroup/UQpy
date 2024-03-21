import torch
import torch.nn as nn
from beartype import beartype
from beartype.vale import Is
from typing import Annotated
from UQpy.scientific_machine_learning.baseclass.Layer import Layer


@beartype
class Dropout(Layer):

    def __init__(
        self,
        drop_rate: Annotated[float, Is[lambda p: 0 <= p <= 1]] = 0.5,
        drop_type: int = 0,
        **kwargs
    ):
        """

        :param drop_rate:
        :param drop_type:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.drop_rate = drop_rate
        self.drop_type = drop_type
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

        Note: Function behavior is determind by the dimensions of ``x`` as N x C x H x W

        :param x:
        :return:
        """
        return self.dropout_function(x)
