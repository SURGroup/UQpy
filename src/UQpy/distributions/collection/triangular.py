import scipy.stats as stats

from UQpy.distributions.baseclass.DistributionContinuous1D import DistributionContinuous1D


class Triangular(DistributionContinuous1D):
    def __init__(self, c, loc=0, scale=1):
        """

        :param c: mode
        :param loc: lower bound
        :param scale: range
        """
        super().__init__(
            c=c, loc=loc, scale=scale, ordered_parameters=("c", "loc", "scale")
        )
        self._construct_from_scipy(scipy_name=stats.triang)
