import numpy
from beartype import beartype

from UQpy.utilities.ValidationTypes import Numpy2DFloatArray
from UQpy.distributions.baseclass import Copula
import numpy as np


class Clayton(Copula):

    @beartype
    def __init__(self, theta: float):
        """

        :param theta: Parameter of the Clayton copula, real number in :math:`[1, +\infty)`
        """
        super().__init__(theta=theta)

    def evaluate_cdf(self, unit_uniform_samples: Numpy2DFloatArray) -> numpy.ndarray:
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
        theta, u, v = self.extract_data(unit_uniform_samples)
        cdf_val = (np.maximum(u ** (-theta) + v ** (-theta) - 1.0, 0.0)) ** (-1.0 / theta)
        return cdf_val

    def extract_data(self, unit_uniform_samples: Numpy2DFloatArray):
        u = unit_uniform_samples[:, 0]
        v = unit_uniform_samples[:, 1]
        theta = self.parameters["theta"]
        return theta, u, v
