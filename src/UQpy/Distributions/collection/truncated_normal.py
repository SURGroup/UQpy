import scipy.stats as stats
from UQpy.Distributions.baseclass import DistributionContinuous1D


class TruncNorm(DistributionContinuous1D):
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
    def __init__(self, a, b, loc=0, scale=1.):
        super().__init__(a=a, b=b, loc=loc, scale=scale, order_params=('a', 'b', 'loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.truncnorm)


