from typing import Union

from beartype import beartype

from UQpy.distributions.baseclass import Copula
from numpy import log, exp


class Gumbel(Copula):
    """
    Gumbel copula having cumulative distribution function

    .. math:: F(u_1, u_2) = \exp(-(-\log(u_1))^{\Theta} + (-\log(u_2))^{\Theta})^{1/{\Theta}}

    where :math:`u_1 = F_1(x_1), u_2 = F_2(x_2)` are uniformly distributed on the interval `[0, 1]`.

    **Input:**

    * **theta** (`float`):
        Parameter of the Gumbel copula, real number in :math:`[1, +\infty)`.

    This copula possesses the following methods:

    * ``evaluate_cdf``, ``evaluate_pdf`` and ``check_copula``

    (``check_copula`` checks that `marginals` consist of solely 2 continuous univariate distributions).

    **evaluate_pdf** *(unif)*
        Compute the copula pdf :math:`c(u_1, u_2, ..., u_d)` for a `d`-variate uniform distribution.

        For a generic multivariate distribution with marginals pdfs :math:`f_1, ..., f_d` and marginals cdfs
        :math:`F_1, ..., F_d`, the joint pdf is computed as:

        :math:`f(x_1, ..., x_d) = c(u_1, u_2, ..., u_d) f_1(x_1) ... f_d(x_d)`

        where :math:`u_i = F_i(x_i)` is uniformly distributed. This computation is performed in the ``JointCopula.pdf``
        method.

        **Input:**

        * **unif** (`ndarray`):
            Points (uniformly distributed) at which to evaluate the copula pdf, must be of shape `(npoints, dimension)`.

        **Output/Returns:**

        * (`tuple`):
            Values of the copula pdf term, ndarray of shape `(npoints, )`.

    **evaluate_cdf** *(unif)*
        Compute the copula cdf :math:`C(u_1, u_2, ..., u_d)` for a `d`-variate uniform distribution.

        For a generic multivariate distribution with marginal cdfs :math:`F_1, ..., F_d` the joint cdf is computed as:

        :math:`F(x_1, ..., x_d) = C(u_1, u_2, ..., u_d)`

        where :math:`u_i = F_i(x_i)` is uniformly distributed. This computation is performed in the ``JointCopula.cdf``
        method.

        **Input:**

        * **unif** (`ndarray`):
            Points (uniformly distributed) at which to evaluate the copula cdf, must be of shape `(npoints, dimension)`.

        **Output/Returns:**

        * (`tuple`):
            Values of the cdf, `ndarray` of shape `(npoints, )`.
    """

    @beartype
    def __init__(self, theta: Union[None, float]):
        super().__init__(theta=theta)

    def evaluate_cdf(self, unit_uniform_samples):
        if unit_uniform_samples.shape[1] > 2:
            raise ValueError('Maximum dimension for the Gumbel Copula is 2')

        theta, u, v = self.extract_data(unit_uniform_samples)

        cdf_val = exp(-((-log(u)) ** theta +
                        (-log(v)) ** theta) ** (1 / theta))

        return cdf_val

    def evaluate_pdf(self, unit_uniform_samples):
        theta, u, v = self.extract_data(unit_uniform_samples)
        c = exp(-((-log(u)) ** theta +
                  (-log(v)) ** theta) ** (1 / theta))

        pdf_val = c * 1 / u * 1 / v * \
                  ((-log(u)) ** theta + (-log(v)) ** theta) ** (-2 + 2 / theta) \
                  * (log(u) * log(v)) ** (theta - 1) * \
                  (1 + (theta - 1) * ((-log(u)) ** theta +
                                           (-log(v)) ** theta) ** (-1 / theta))
        return pdf_val

    def extract_data(self, unit_uniform_samples):
        u = unit_uniform_samples[:, 0]
        v = unit_uniform_samples[:, 1]
        theta = self.parameters['theta']
        return theta, u, v
