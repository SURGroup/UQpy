from UQpy.distributions.baseclass import Copula
from UQpy.utilities.validation.Validations import *
import numpy as np


class Clayton(Copula):
    """
    Clayton copula having cumulative distribution function

    .. math:: F(u_1, u_2) = \max(u_1^{-\Theta} + u_2^{-\Theta} - 1, 0)^{-1/{\Theta}}

    where :math:`u_1 = F_1(x_1), u_2 = F_2(x_2)` are uniformly distributed on the interval `[0, 1]`.

    **Input:**

    * **theta** (`float`):
        Parameter of the copula, real number in [-1, +oo)\{0}.

    This copula possesses the following methods:

    * ``evaluate_cdf`` and ``check_copula``

    (``check_copula`` checks that `marginals` consist of solely 2 continuous univariate distributions).
    """
    @check_copula_theta()
    def __init__(self, theta):
        super().__init__(theta=theta)

    @check_sample_dimensions()
    def evaluate_cdf(self, unit_uniform_samples):
        theta, u, v = self.extract_data(unit_uniform_samples)
        cdf_val = (np.maximum(u ** (-theta) +
                              v ** (-theta) - 1., 0.)) ** (-1. / theta)
        return cdf_val

    def extract_data(self, unit_uniform_samples):
        u = unit_uniform_samples[:, 0]
        v = unit_uniform_samples[:, 1]
        theta = self.parameters['theta']
        return theta, u, v
