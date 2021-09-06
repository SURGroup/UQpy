from typing import Union

import scipy.stats as stats
from beartype import beartype

from UQpy.distributions.baseclass import DistributionContinuous1D


class TruncatedNormal(DistributionContinuous1D):
    """
    Truncated normal distribution

    The standard form of this distribution (i.e, loc=0., scale=1) is a standard normal truncated to the range [a, b].
    Note that a and b are defined over the domain of the standard normal.

    **Inputs:**

    * **a** (`float`):
        shape parameter
    * **b** (`float`):
        shape parameter
    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``TruncNorm``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    @beartype
    def __init__(self, a: Union[None, float, int], b: Union[None, float, int],
                 loc: Union[None, float, int] = 0., scale: Union[None, float, int] = 1.):
        super().__init__(a=a, b=b, loc=loc, scale=scale, ordered_parameters=('a', 'b', 'loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.truncnorm)
