import scipy.stats as stats
from UQpy.Distributions.baseclass import DistributionContinuous1D

class Laplace(DistributionContinuous1D):
    """
    Laplace distribution having probability density function

    .. math:: f(x) = \dfrac{1}{2}\exp{-|x|}

    In this standard form `(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
    this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

    **Inputs:**

    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``Laplace``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, loc=0, scale=1):
        super().__init__(loc=loc, scale=scale, order_params=('loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.laplace)

