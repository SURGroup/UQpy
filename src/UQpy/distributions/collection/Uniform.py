import scipy.stats as stats
from beartype import beartype

from UQpy.distributions.baseclass import DistributionContinuous1D


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
    @beartype
    def __init__(self, location: float = 0., scale: float = 1.):
        super().__init__(loc=location, scale=scale, ordered_parameters=('location', 'scale'))
        self._construct_from_scipy(scipy_name=stats.uniform)
