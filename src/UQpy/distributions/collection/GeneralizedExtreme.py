from typing import Union

import scipy.stats as stats
from beartype import beartype

from UQpy.distributions.baseclass import DistributionContinuous1D


class GeneralizedExtreme(DistributionContinuous1D):

    @beartype
    def __init__(
        self,
        c: Union[None, float, int],
        loc: Union[None, float, int] = 0.0,
        scale: Union[None, float, int] = 1.0,
    ):
        """

        :param c: shape parameter
        :param loc: location parameter
        :param scale: scale parameter
        """
        super().__init__(
            c=c, loc=loc, scale=scale, ordered_parameters=("c", "loc", "scale")
        )
        self._construct_from_scipy(scipy_name=stats.genextreme)
