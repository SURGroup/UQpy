from typing import Union

import scipy.stats as stats
from beartype import beartype

from UQpy.distributions.baseclass import DistributionContinuous1D


class Exponential(DistributionContinuous1D):

    @beartype
    def __init__(
        self, loc: Union[None, float, int] = 0.0, scale: Union[None, float, int] = 1.0
    ):
        """

        :param loc: location parameter
        :param scale: scale parameter
        """
        super().__init__(loc=loc, scale=scale, ordered_parameters=("loc", "scale"))
        self._construct_from_scipy(scipy_name=stats.expon)

    def __repr__(self):
        s = []
        if self.parameters["loc"] != 0.0:
            s.append("loc={loc}")
        if self.parameters["scale"] != 1.0:
            s.append("scale={scale}")
        s = ", ".join(s)
        return "Exponential(" + s.format(**self.parameters) + ")"
