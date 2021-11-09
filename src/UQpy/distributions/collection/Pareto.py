from typing import Union

import scipy.stats as stats
from beartype import beartype

from UQpy.distributions.baseclass import DistributionContinuous1D


class Pareto(DistributionContinuous1D):
    @beartype
    def __init__(
        self,
        b: Union[None, float, int],
        loc: Union[None, float, int] = 0.0,
        scale: Union[None, float, int] = 1.0,
    ):
        """

        :param b: shape parameter
        :param loc: location parameter
        :param scale: scale parameter
        """
        super().__init__(
            b=b, loc=loc, scale=scale, ordered_parameters=("b", "loc", "scale")
        )
        self._construct_from_scipy(scipy_name=stats.pareto)
