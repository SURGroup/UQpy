from typing import Union

import numpy
import numpy as np
from beartype import beartype

from UQpy.utilities.ValidationTypes import Numpy2DFloatArray
from UQpy.distributions.baseclass import Copula
from numpy import log, exp


class Gumbel(Copula):
    @beartype
    def __init__(self, theta: Union[None, float]):
        """

        :param theta: Parameter of the Gumbel copula, real number in :math:`[1, +\infty)`.
        """
        super().__init__(theta=theta)

    def evaluate_cdf(self, unit_uniform_samples: Numpy2DFloatArray) -> np.ndarray:
        """
        Compute the copula cdf :math:`C(u_1, u_2, ..., u_d)` for a `d`-variate uniform distribution.

        For a generic multivariate distribution with marginal cdfs :math:`F_1, ..., F_d` the joint cdf is computed as:

        :math:`F(x_1, ..., x_d) = C(u_1, u_2, ..., u_d)`

        where :math:`u_i = F_i(x_i)` is uniformly distributed. This computation is performed in the
        :meth:`.JointCopula.cdf` method.

        :param unit_uniform_samples: Points (uniformly distributed) at which to evaluate the copula cdf, must be of
         shape :code:`(npoints, dimension)`.
        :return: Values of the cdf.
        """
        if unit_uniform_samples.shape[1] > 2:
            raise ValueError("Maximum dimension for the Gumbel Copula is 2")

        theta, u, v = self.extract_data(unit_uniform_samples)

        cdf_val = exp(-(((-log(u)) ** theta + (-log(v)) ** theta) ** (1 / theta)))

        return cdf_val

    def evaluate_pdf(self, unit_uniform_samples: Numpy2DFloatArray) -> numpy.ndarray:
        """
        Compute the copula pdf :math:`c(u_1, u_2, ..., u_d)` for a `d`-variate uniform distribution.

        For a generic multivariate distribution with marginals pdfs :math:`f_1, ..., f_d` and marginals cdfs
        :math:`F_1, ..., F_d`, the joint pdf is computed as:

        :math:`f(x_1, ..., x_d) = c(u_1, u_2, ..., u_d) f_1(x_1) ... f_d(x_d)`

        where :math:`u_i = F_i(x_i)` is uniformly distributed. This computation is performed in the
        :meth:`.JointCopula.pdf` method.

        :param unit_uniform_samples: Points (uniformly distributed) at which to evaluate the copula pdf, must be of
         shape :code:`(npoints, dimension)`.

        :return: Values of the copula pdf term.
        """
        theta, u, v = self.extract_data(unit_uniform_samples)
        c = exp(-(((-log(u)) ** theta + (-log(v)) ** theta) ** (1 / theta)))

        pdf_val = (c * 1 / u * 1 / v
                   * ((-log(u)) ** theta + (-log(v)) ** theta) ** (-2 + 2 / theta)
                   * (log(u) * log(v)) ** (theta - 1)
                   * (1 + (theta - 1) * ((-log(u)) ** theta + (-log(v)) ** theta) ** (-1 / theta)))
        return pdf_val

    def extract_data(self, unit_uniform_samples):
        u = unit_uniform_samples[:, 0]
        v = unit_uniform_samples[:, 1]
        theta = self.parameters["theta"]
        return theta, u, v
