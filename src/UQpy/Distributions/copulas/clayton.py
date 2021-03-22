from UQpy.Distributions.baseclass import Copula
from UQpy.Distributions.baseclass import DistributionContinuous1D, DistributionND, DistributionDiscrete1D
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
        # Check the input copula_params
        if theta is not None and ((not isinstance(theta, (float, int))) or (theta < -1 or theta == 0.)):
            raise ValueError('Input theta should be a float in [-1, +oo)\{0}.')
        super().__init__(theta=theta)

    def evaluate_cdf(self, unif):
        if unif.shape[1] > 2:
            raise ValueError('Maximum dimension for the Clayton Copula is 2.')
        if self.params['theta'] == 1:
            return np.prod(unif, axis=1)

        u = unif[:, 0]
        v = unif[:, 1]
        theta = self.params['theta']
        cdf_val = (np.maximum(u ** (-theta) + v ** (-theta) - 1., 0.)) ** (-1. / theta)
        return cdf_val

    def check_marginals(self, marginals):
        if len(marginals) != 2:
            raise ValueError('Maximum dimension for the Clayton Copula is 2.')
        if not all(isinstance(m, DistributionContinuous1D) for m in marginals):
            raise ValueError('Marginals should be 1d continuous distributions.')




