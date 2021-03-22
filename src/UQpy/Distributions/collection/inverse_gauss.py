import scipy.stats as stats
from UQpy.Distributions.baseclass import DistributionContinuous1D

class InvGauss(DistributionContinuous1D):
    """
    Inverse Gaussian distribution having probability density function

    .. math:: f(x|\mu) = \dfrac{1}{2\pi x^3}\exp{(-\dfrac{(x\\mu)^2}{2x\mu^2})}

    for :math:`x>0`. ``cdf`` method returns `NaN` for :math:`\mu<0.0028`.

    **Inputs:**

    * **mu** (`float`):
        shape parameter, :math:`\mu`
    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    In this standard form `(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
    this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

    The following methods are available for ``InvGauss``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, mu, loc=0., scale=1.):
        super().__init__(mu=mu, loc=loc, scale=scale, order_params=('mu', 'loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.invgauss)

