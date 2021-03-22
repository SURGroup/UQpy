import scipy.stats as stats
from UQpy.Distributions.baseclass import DistributionContinuous1D

class Lognormal(DistributionContinuous1D):
    """
    Lognormal distribution having probability density function

    .. math:: f(x|s) = \dfrac{1}{sx\sqrt{2\pi}}\exp(-\dfrac{\log^2(x)}{2s^2})

    for :math:`x>0, s>0`.

    A common parametrization for a lognormal random variable Y is in terms of the mean, mu, and standard deviation,
    sigma, of the gaussian random variable X such that exp(X) = Y. This parametrization corresponds to setting
    s = sigma and scale = exp(mu).

    **Inputs:**

    * **s** (`float`):
        shape parameter
    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``Lognormal``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, s, loc=0., scale=1.):
        super().__init__(s=s, loc=loc, scale=scale, order_params=('s', 'loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.lognorm)

