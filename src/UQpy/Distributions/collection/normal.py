import numpy as np
import scipy.stats as stats
from UQpy.Distributions.baseclass import DistributionContinuous1D

class Normal(DistributionContinuous1D):
    """
    Normal distribution having probability density function

    .. math:: f(x) = \dfrac{\exp(-x^2/2)}{\sqrt{2\pi}}

    In this standard form `(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
    this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

    **Inputs:**

    * **loc** (`float`):
        mean
    * **scale** (`float`):
        standard deviation

    The following methods are available for ``Normal``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, loc=0., scale=1.):
        super().__init__(loc=loc, scale=scale, order_params=('loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.norm)

    def fit(self, x):
        x = self._check_x_dimension(x)
        mle_loc, mle_scale = self.params['loc'], self.params['scale']
        if mle_loc is None:
            mle_loc = np.mean(x)
        if mle_scale is None:
            mle_scale = np.sqrt(np.mean((x - mle_loc) ** 2))
        return {'loc': mle_loc, 'scale': mle_scale}


