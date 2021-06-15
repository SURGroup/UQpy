from UQpy.distributions.baseclass import Copula
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
    def __init__(self, theta):
        super().__init__(theta=theta)

    def evaluate_cdf(self, first_uniform, second_uniform):
        theta = self.parameters['theta']
        cdf_val = (np.maximum(first_uniform ** (-theta) + second_uniform ** (-theta) - 1., 0.)) ** (-1. / theta)
        return cdf_val
