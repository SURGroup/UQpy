from typing import Union

import scipy.stats as stats
from beartype import beartype

from UQpy.distributions.baseclass import DistributionDiscrete1D


class Binomial(DistributionDiscrete1D):
    @beartype
    def __init__(
        self,
        n: Union[None, int],
        p: Union[None, float, int],
        loc: Union[None, float, int] = 0.0,
    ):
        """

        :param int n: number of trials, integer >=0
        :param float p: success probability for each trial, real number in [0, 1]
        :param float loc: location parameter
        """
        super().__init__(n=n, p=p, loc=loc, ordered_parameters=("n", "p", "loc"))
        self._construct_from_scipy(scipy_name=stats.binom)
