import numpy as np
import scipy.stats as stats
from UQpy.Distributions.baseclass.Distribution import Distribution

########################################################################################################################
#        Univariate Discrete Distributions
########################################################################################################################

class DistributionDiscrete1D(Distribution):
    """
    Parent class for univariate discrete distributions.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _check_x_dimension(x):
        """
        Check the dimension of input x - must be an ndarray of shape (npoints,) or (npoints, 1)
        """
        x = np.atleast_1d(x)
        if len(x.shape) > 2 or (len(x.shape) == 2 and x.shape[1] != 1):
            raise ValueError('Wrong dimension in x.')
        return x.reshape((-1, ))

    def _construct_from_scipy(self, scipy_name=stats.rv_discrete):
        self.cdf = lambda x: scipy_name.cdf(x=self._check_x_dimension(x), **self.params)
        self.pmf = lambda x: scipy_name.pmf(x=self._check_x_dimension(x), **self.params)
        self.log_pmf = lambda x: scipy_name.logpmf(x=self._check_x_dimension(x), **self.params)
        self.icdf = lambda x: scipy_name.ppf(q=self._check_x_dimension(x), **self.params)
        self.moments = lambda moments2return='mvsk': scipy_name.stats(moments=moments2return, **self.params)
        self.rvs = lambda nsamples=1, random_state=None: scipy_name.rvs(
            size=nsamples, random_state=random_state, **self.params).reshape((nsamples, 1))