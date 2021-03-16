import scipy.stats as stats
from UQpy.Distributions.baseclass import DistributionDiscrete1D

class Poisson(DistributionDiscrete1D):
    """
    Poisson distribution having probability mass function:

    .. math:: f(x) = \exp{(-\mu)}\dfrac{\mu^k}{k!}

    for :math:`x\ge 0`.

    In this standard form `(loc=0)`. Use `loc` to shift the distribution. Specifically, this is equivalent to computing
    :math:`f(y)` where :math:`y=x-loc`.

    **Inputs:**

    * **mu** (`float`):
        shape parameter
    * **loc** (`float`):
        location parameter

    The following methods are available for ``Poisson``:

    * ``cdf``, ``pmf``, ``log_pmf``, ``icdf``, ``rvs``, ``moments``.
    """
    def __init__(self, mu, loc=0.):
        super().__init__(mu=mu, loc=loc, order_params=('mu', 'loc'))
        self._construct_from_scipy(scipy_name=stats.poisson)




