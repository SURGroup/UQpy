import numpy as np
import scipy.stats as stats
from UQpy.Distributions.baseclass import DistributionND

class Multinomial(DistributionND):
    """
    Multinomial distribution having probability mass function

    .. math:: f(x) = \dfrac{number_of_dimensions!}{x_1!\dots x_k!}p_1^{x_1}\dots p_k^{x_k}

    for :math:`x=\{x_1,\dots,x_k\}` where each :math:`x_i` is a non-negative integer and :math:`\sum_i x_i = number_of_dimensions`.

    **Inputs:**

    * **number_of_dimensions** (`int`):
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
        if moments2return == 'number_of_variables':
            mean = self.get_params()['number_of_dimensions'] * np.array(self.get_params()['p'])
            return mean
        elif moments2return == 'v':
            n, p = self.get_params()['number_of_dimensions'], np.array(self.get_params()['p'])
            d = len(p)
            cov = - n * np.tile(p[np.newaxis, :], [d, 1]) * np.tile(p[:, np.newaxis], [1, d])
            np.fill_diagonal(cov, n * p * (1. - p))
            return cov
        elif moments2return == 'mv':
            n, p = self.get_params()['number_of_dimensions'], np.array(self.get_params()['p'])
            d = len(p)
            cov = - n * np.tile(p[np.newaxis, :], [d, 1]) * np.tile(p[:, np.newaxis], [1, d])
            np.fill_diagonal(cov, n * p * (1. - p))
            mean = n * p
            return mean, cov
        else:
            raise ValueError('UQpy: moments2return must be "number_of_variables", "v" or "mv".')




