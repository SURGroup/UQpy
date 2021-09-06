from typing import Union

import numpy as np
import scipy.stats as stats
from beartype import beartype
from beartype.vale import IsAttr, IsEqual

from UQpy.distributions.baseclass import DistributionND


class MultivariateNormal(DistributionND):
    """
    Multivariate normal distribution having probability density function

    .. math:: f(x) = \dfrac{1}{\sqrt{(2\pi)^k\det\Sigma}}\exp{-\dfrac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}

    where :math:`\mu` is the mean vector, :math:`\Sigma` is the covariance matrix, and :math:`k` is the dimension of
    `x`.

    **Inputs:**

    * **mean** (`ndarray`):
        mean vector, `ndarray` of shape `(dimension, )`
    * **cov** (`float` or `ndarray`):
        covariance, `float` or `ndarray` of shape `(dimension, )` or `(dimension, dimension)`. Default is 1.

    The following methods are available for ``MVNormal``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``rvs``, ``fit``, ``moments``.
    """
    @beartype
    def __init__(self, mean: Union[None, np.ndarray, list], cov: Union[None, int, float, np.ndarray] = 1.):
        if mean is not None and cov is not None:
            if isinstance(cov, (int, float)):
                pass
            else:
                if not (len(np.array(cov).shape) in [1, 2] and
                        all(sh == len(mean) for sh in np.array(cov).shape)):
                    raise ValueError('Input covariance must be a float or ndarray of appropriate dimensions.')
        super().__init__(mean=mean, cov=cov, ordered_parameters=['mean', 'cov'])

    def cdf(self, x):
        cdf_val = stats.multivariate_normal.cdf(x=x, **self.parameters)
        return np.atleast_1d(cdf_val)

    def pdf(self, x):
        pdf_val = stats.multivariate_normal.pdf(x=x, **self.parameters)
        return np.atleast_1d(pdf_val)

    def log_pdf(self, x):
        logpdf_val = stats.multivariate_normal.logpdf(x=x, **self.parameters)
        return np.atleast_1d(logpdf_val)

    def rvs(self, nsamples=1, random_state=None):
        if not (isinstance(nsamples, int) and nsamples >= 1):
            raise ValueError('Input nsamples must be an integer > 0.')
        return stats.multivariate_normal.rvs(
            size=nsamples, random_state=random_state, **self.parameters).reshape((nsamples, -1))

    def fit(self, data):
        data = self.check_x_dimension(data)
        mle_mu, mle_cov = self.parameters['mean'], self.parameters['cov']
        if mle_mu is None:
            mle_mu = np.mean(data, axis=0)
        if mle_cov is None:
            mle_cov = np.cov(data, rowvar=False)
        return {'mean': mle_mu, 'cov': mle_cov}

    def moments(self, moments2return='mv'):
        if moments2return == 'm':
            return self.get_parameters()['mean']
        elif moments2return == 'v':
            return self.get_parameters()['cov']
        elif moments2return == 'mv':
            return self.get_parameters()['mean'], self.get_parameters()['cov']
        else:
            raise ValueError('UQpy: moments2return must be "m", "v" or "mv".')
