import scipy.stats as stats
from UQpy.distributions.baseclass.Distribution1D import Distribution1D
from abc import ABC


class DistributionDiscrete1D(Distribution1D, ABC):
    def __init__(self, **kwargs):
        """
        Parent class for univariate discrete distributions.
        """
        super().__init__(**kwargs)

    def _construct_from_scipy(self, scipy_name=stats.rv_discrete):
        self.pmf = lambda x: scipy_name.pmf(k=self.check_x_dimension(x), **self.parameters)
        self.log_pmf = lambda x: scipy_name.logpmf(k=self.check_x_dimension(x), **self.parameters)
        self._retrieve_1d_data_from_scipy(scipy_name, is_continuous=False)
