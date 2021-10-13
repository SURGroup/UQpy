from typing import Union

import scipy.stats as stats
from beartype import beartype

from UQpy.distributions.baseclass import DistributionContinuous1D


class GeneralizedExtreme(DistributionContinuous1D):
    """
    Generalized Extreme Value distribution having probability density function:

    .. math:: `f(x|c) = \exp(-(1-cx)^{1/c})(1-cx)^{1/c-1}`

    for :math:`x\le 1/c, c>0`.

    For `c=0`

    .. math:: f(x) = \exp(\exp(-x))\exp(-x)

    In this standard form `(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
    this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

    **Inputs:**

    * **c** (`float`):
        shape parameter
    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``GenExtreme``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """

    @beartype
    def __init__(
        self,
        c: Union[None, float, int],
        loc: Union[None, float, int] = 0.0,
        scale: Union[None, float, int] = 1.0,
    ):
        super().__init__(
            c=c, loc=loc, scale=scale, ordered_parameters=("c", "loc", "scale")
        )
        self._construct_from_scipy(scipy_name=stats.genextreme)
