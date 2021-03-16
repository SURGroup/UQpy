import scipy.stats as stats
from UQpy.Distributions.baseclass import DistributionContinuous1D

class Gamma(DistributionContinuous1D):
    """
    Gamma distribution having probability density function:

    .. math:: f(x|a) = \dfrac{x^{a-1}\exp(-x)}{\Gamma(a)}

    for :math:`x\ge 0`, :math:`a>0`. Here :math:`\Gamma(a)` refers to the Gamma function.

    In this standard form `(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
    this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

    **Inputs:**

    * **a** (`float`):
        shape parameter
    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``Gamma``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, a, loc=0., scale=1.):
        super().__init__(a=a, loc=loc, scale=scale, order_params=('a', 'loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.gamma)
