from typing import Union

import scipy.stats as stats
from beartype import beartype

from UQpy.distributions.baseclass import DistributionContinuous1D


class InverseGauss(DistributionContinuous1D):

    @beartype
    def __init__(
        self,
        mu: Union[None, float, int],
        loc: Union[None, float, int] = 0.0,
        scale: Union[None, float, int] = 1.0,
    ):
        """

        :param mu: shape parameter :math:`\mu`
        :param loc: location parameter
        :param scale: scale parameter
        """
        super().__init__(
            mu=mu, loc=loc, scale=scale, ordered_parameters=("mu", "loc", "scale")
        )
        self._construct_from_scipy(scipy_name=stats.invgauss)
