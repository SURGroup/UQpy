import scipy.stats as stats
from UQpy.Distributions.baseclass import DistributionContinuous1D

class ChiSquare(DistributionContinuous1D):
    """
    Chi-square distribution having probability density:

    .. math:: f(x|k) = \dfrac{1}{2^{k/2}\Gamma(k/2)}x^{k/2-1}\exp{(-x/2)}

    for :math:`x\ge 0`, :math:`k>0`. Here :math:`\Gamma(\cdot)` refers to the Gamma function.

    In this standard form `(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
    this is equivalent to computing :math:`f(y|k)` where :math:`y=(x-loc)/scale`.

    **Inputs:**

    * **df** (`float`):
        shape parameter (degrees of freedom) (given by `k` in the equation above)
    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``ChiSquare``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, df, loc=0., scale=1):
        super().__init__(df=df, loc=loc, scale=scale, order_params=('df', 'loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.chi2)