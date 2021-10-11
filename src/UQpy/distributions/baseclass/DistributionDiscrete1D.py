import scipy.stats as stats
from UQpy.distributions.baseclass.Distribution1D import Distribution1D
from abc import ABC


class DistributionDiscrete1D(Distribution1D, ABC):
    """
    Parent class for univariate discrete distributions.

    **pmf** *(x)*
        Evaluate the probability mass function of a discrete distribution.

        **Input:**

        * **x** (`ndarray`):
            Point(s) at which to evaluate the `pmf`, must be of shape `(npoints, dimension)`.

        **Output/Returns:**

        * (`ndarray`):
            Evaluated pmf values, `ndarray` of shape `(npoints,)`.

    **log_pmf** *(x)*
        Evaluate the logarithm of the probability mass function of a discrete distribution.

        **Input:**

        * **x** (`ndarray`):
            Point(s) at which to evaluate the `log_pmf`, must be of shape `(npoints, dimension)`.

        **Output/Returns:**

        * (`ndarray`):
            Evaluated log-pmf values, `ndarray` of shape `(npoints,)`.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _construct_from_scipy(self, scipy_name=stats.rv_discrete):
        self.pmf = lambda x: scipy_name.pmf(k=self.check_x_dimension(x), **self.parameters)
        self.log_pmf = lambda x: scipy_name.logpmf(k=self.check_x_dimension(x), **self.parameters)
        self._retrieve_1d_data_from_scipy(scipy_name, is_continuous=False)
