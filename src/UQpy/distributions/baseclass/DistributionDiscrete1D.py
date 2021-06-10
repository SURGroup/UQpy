import numpy as np
import scipy.stats as stats
from UQpy.distributions.baseclass.Distribution1D import Distribution1D


class DistributionDiscrete1D(Distribution1D):
    """
    Parent class for univariate discrete distributions.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _construct_from_scipy(self, scipy_name=stats.rv_discrete):
        self.pmf = lambda x: scipy_name.pmf(k=self._check_x_dimension(x), **self.parameters)
        self.log_pmf = lambda x: scipy_name.logpmf(k=self._check_x_dimension(x), **self.parameters)
        self._retrieve_1d_data_from_scipy(scipy_name)
