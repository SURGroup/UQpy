from typing import Union

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
    def __init__(
        self, loc: Union[None, float, int] = 0.0, scale: Union[None, float, int] = 1.0
    ):
        super().__init__(loc=loc, scale=scale, ordered_parameters=("loc", "scale"))
        self._construct_from_scipy(scipy_name=stats.uniform)
