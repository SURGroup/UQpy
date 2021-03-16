import scipy.stats as stats
from UQpy.Distributions.baseclass import DistributionContinuous1D

class Uniform(DistributionContinuous1D):
    """
    Uniform distribution having probability density function

    .. math:: f(x|a, b) = \dfrac{1}{b-a}

    where :math:`a=loc` and :math:`b=loc+scale`

    **Inputs:**

    * **loc** (`float`):
        lower bound
    * **scale** (`float`):
        range

    The following methods are available for ``Uniform``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, loc=0., scale=1.):
        super().__init__(loc=loc, scale=scale, order_params=('loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.uniform)



