from typing import Union

import scipy.stats as stats
from beartype import beartype

from UQpy.distributions.baseclass import DistributionDiscrete1D


class Poisson(DistributionDiscrete1D):

    @beartype
    def __init__(self, mu: Union[None, float, int], loc: Union[None, float, int] = 0.0):
        """

        :param mu: shape parameter
        :param loc: location parameter
        """
        super().__init__(mu=mu, loc=loc, ordered_parameters=("mu", "loc"))
        self._construct_from_scipy(scipy_name=stats.poisson)
