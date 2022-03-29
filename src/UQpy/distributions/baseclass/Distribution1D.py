import numpy as np
import scipy.stats as stats
from UQpy.distributions.baseclass.Distribution import Distribution
from abc import ABC


class Distribution1D(Distribution, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def check_x_dimension(x):
        """
        Check the dimension of input x - must be an ndarray of shape (npoints,) or (npoints, 1)
        """
        array = np.atleast_1d(x)
        if len(array.shape) > 2 or (len(array.shape) == 2 and array.shape[1] != 1):
            raise ValueError("Wrong dimension in x.")
        return array.reshape((-1,))

    def _retrieve_1d_data_from_scipy(self, scipy_name=stats.rv_continuous, is_continuous=True):
        if is_continuous:
            self.cdf = lambda x: scipy_name.cdf(x=self.check_x_dimension(x), **self.parameters)
        else:
            self.cdf = lambda x: scipy_name.cdf(k=self.check_x_dimension(x), **self.parameters)
        self.icdf = lambda x: scipy_name.ppf(q=self.check_x_dimension(x), **self.parameters)
        self.moments = lambda moments2return="mvsk": scipy_name.stats(moments=moments2return, **self.parameters)
        self.rvs = lambda nsamples=1, random_state=None: scipy_name.rvs(
            size=nsamples, random_state=random_state, **self.parameters).reshape((nsamples, 1))
