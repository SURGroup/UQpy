import scipy.stats as stats
from beartype import beartype

from UQpy.distributions.baseclass import DistributionContinuous1D


class Cauchy(DistributionContinuous1D):
    """
    Cauchy distribution having probability density function

    .. math:: f(x) = \dfrac{1}{\pi(1+x^2)}

    In this standard form `(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
    this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

    **Inputs:**

    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``Cauchy``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``
    """
    @beartype
    def __init__(self, location: float = 0., scale: float = 1.):
        super().__init__(loc=location, scale=scale, ordered_parameters=('location', 'scale'))
        self._construct_from_scipy(scipy_name=stats.cauchy)
