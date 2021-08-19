from typing import Union

import scipy.stats as stats
from beartype import beartype

from UQpy.distributions.baseclass import DistributionContinuous1D


class Exponential(DistributionContinuous1D):
    """
    Exponential distribution having probability density function:

    .. math:: f(x) = \exp(-x)

    In this standard form `(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
    this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

    A common parameterization for Exponential is in terms of the rate parameter :math:`\lambda`, which corresponds to
    using :math:`scale = 1 / \lambda`.

    **Inputs:**

    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``Exponential``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    @beartype
    def __init__(self, loc: Union[None, float] = 0., scale: Union[None, float] = 1.):
        super().__init__(loc=loc, scale=scale, ordered_parameters=('loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.expon)
