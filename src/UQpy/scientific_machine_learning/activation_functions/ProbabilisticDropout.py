import torch
from beartype import beartype
from beartype.vale import Is
from typing import Annotated
from UQpy.scientific_machine_learning.baseclass.ActivationFunction import ActivationFunction


# @beartype
class ProbabilisticDropout(ActivationFunction):
    def __init__(self,
                 p: Annotated[float, Is[lambda p: 0 <= p <= 1]] = 0.5,
                 active: bool = False):
        """Dropout function to randomly assign tensor elements to zero with probability :math:`p`

        :param p: Probability of *dropping* a tensor element. Default ``p=0.5``.
        :param active: Flag to toggle layer on or off. If ``False``, this is the identity function.
        """
        super().__init__()
        self.p = p
        self.active = active

    def forward(self, x):
        if self.active:
            mask = torch.rand_like(x, dtype=torch.float) < self.p
            x[mask] = 0
            return x
        else:
            return x

    def extra_repr(self):
        return f'p={self.p}, active={self.active}'
