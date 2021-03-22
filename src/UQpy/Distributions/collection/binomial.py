import scipy.stats as stats
from UQpy.Distributions.baseclass import DistributionDiscrete1D

class Binomial(DistributionDiscrete1D):
    """
    Binomial distribution having probability mass function:

    .. math:: f(x) = {number_of_dimensions \choose x} p^x(1-p)^{number_of_dimensions-x}

    for :math:`x\inumber_of_dimensions\{0, 1, 2, ..., number_of_dimensions\}`.

    In this standard form `(loc=0)`. Use `loc` to shift the distribution. Specifically, this is equivalent to computing
    :math:`f(y)` where :math:`y=x-loc`.

    **Inputs:**

    * **number_of_dimensions** (`int`):
        number of trials, integer >= 0
    * **p** (`float`):
        success probability for each trial, real number in [0, 1]
    * **loc** (`float`):
        location parameter

    The following methods are available for ``Binomial``:

    * ``cdf``, ``pmf``, ``log_pmf``, ``icdf``, ``rvs, moments``.
    """
    def __init__(self, n, p, loc=0.):
        super().__init__(n=n, p=p, loc=loc, order_params=('number_of_dimensions', 'p', 'loc'))
        self._construct_from_scipy(scipy_name=stats.binom)



