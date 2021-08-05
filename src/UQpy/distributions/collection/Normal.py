import numpy as np
import scipy.stats as stats
from UQpy.distributions.baseclass import DistributionContinuous1D


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
        super().__init__(loc=loc, scale=scale, ordered_parameters=('loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.norm)
