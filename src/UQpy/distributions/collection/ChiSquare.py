from typing import Union

import scipy.stats as stats
from beartype import beartype

from UQpy.distributions.baseclass import DistributionContinuous1D


class ChiSquare(DistributionContinuous1D):
    @beartype
    def __init__(
        self,
        df: Union[None, float, int],
        loc: Union[None, float, int] = 0.0,
        scale: Union[None, float, int] = 1.0,
    ):
        """

        :param df: shape parameter (degrees of freedom) (given by k in the equation)
        :param loc: location parameter
        :param scale: scale parameter
        """
        super().__init__(
            df=df, loc=loc, scale=scale, ordered_parameters=("df", "loc", "scale")
        )
        self._construct_from_scipy(scipy_name=stats.chi2)
