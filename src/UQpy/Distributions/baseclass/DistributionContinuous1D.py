import numpy as np
import scipy.stats as stats
from UQpy.Distributions.baseclass.Distribution import Distribution


class DistributionContinuous1D(Distribution):
    """
    Parent class for univariate continuous probability distributions.


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
        return x.reshape((-1,))

    def _construct_from_scipy(self, scipy_name=stats.rv_continuous):
        self.cdf = lambda x: scipy_name.cdf(x=self._check_x_dimension(x), **self.params)
        self.pdf = lambda x: scipy_name.pdf(x=self._check_x_dimension(x), **self.params)
        self.log_pdf = lambda x: scipy_name.logpdf(x=self._check_x_dimension(x), **self.params)
        self.icdf = lambda x: scipy_name.ppf(q=self._check_x_dimension(x), **self.params)
        self.moments = lambda moments2return='mvsk': scipy_name.stats(moments=moments2return, **self.params)
        self.rvs = lambda nsamples=1, random_state=None: scipy_name.rvs(
            size=nsamples, random_state=random_state, **self.params).reshape((nsamples, 1))

        def tmp_fit(dist, data):
            data = self._check_x_dimension(data)
            fixed_params = {}
            for key, value in dist.params.items():
                if value is not None:
                    fixed_params['f' + key] = value
            params_fitted = scipy_name.fit(data=data, **fixed_params)
            return dict(zip(dist.order_params, params_fitted))
        self.fit = lambda data: tmp_fit(self, data)