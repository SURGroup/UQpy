import numpy as np
import scipy.stats as stats
from UQpy.Distributions.baseclass import DistributionND


class Multinomial(DistributionND):
    """
    Multinomial distribution having probability mass function

    .. math:: f(x) = \dfrac{n!}{x_1!\dots x_k!}p_1^{x_1}\dots p_k^{x_k}

    for :math:`x=\{x_1,\dots,x_k\}` where each :math:`x_i` is a non-negative integer and :math:`\sum_i x_i = n`.

    **Inputs:**

    * **n** (`int`):
        number of trials
    * **p** (`array_like`):
        probability of a trial falling into each category; should sum to 1

    The following methods are available for ``Multinomial``:

    * ``pmf``, ``log_pmf``, ``rvs``, ``moments``.
    """
    def __init__(self, n, p):
        super().__init__(n=n, p=p)

    def pmf(self, x):
        pdf_val = stats.multinomial.pmf(x=x, **self.params)
        return np.atleast_1d(pdf_val)

    def log_pmf(self, x):
        logpdf_val = stats.multinomial.logpmf(x=x, **self.params)
        return np.atleast_1d(logpdf_val)

    def rvs(self, nsamples=1, random_state=None):
        if not (isinstance(nsamples, int) and nsamples >= 1):
            raise ValueError('Input nsamples must be an integer > 0.')
        return stats.multinomial.rvs(
            size=nsamples, random_state=random_state, **self.params).reshape((nsamples, -1))

    def moments(self, moments2return='mv'):
        n, p = self.get_params()['n'], np.array(self.get_params()['p'])
        d = len(p)
        if moments2return == 'm':
            mean = n * np.array(p)
            return mean
        elif moments2return == 'v':
            cov = - n * np.tile(p[np.newaxis, :], [d, 1]) * np.tile(p[:, np.newaxis], [1, d])
            np.fill_diagonal(cov, n * p * (1. - p))
            return cov
        elif moments2return == 'mv':
            cov = - n * np.tile(p[np.newaxis, :], [d, 1]) * np.tile(p[:, np.newaxis], [1, d])
            np.fill_diagonal(cov, n * p * (1. - p))
            mean = n * np.array(p)
            return mean, cov
        else:
            raise ValueError('UQpy: moments2return must be "m", "v" or "mv".')

