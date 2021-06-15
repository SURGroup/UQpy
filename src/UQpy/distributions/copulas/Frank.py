import numpy as np
from UQpy.distributions.baseclass import Copula


class Frank(Copula):
    """
    Frank copula having cumulative distribution function

    :math:`F(u_1, u_2) = -\dfrac{1}{\Theta} \log(1+\dfrac{(\exp(-\Theta u_1)-1)(\exp(-\Theta u_2)-1)}{\exp(-\Theta)-1})`

    where :math:`u_1 = F_1(x_1), u_2 = F_2(x_2)` are uniformly distributed on the interval `[0, 1]`.

    **Input:**

    * **theta** (`float`):
        Parameter of the copula, real number in R\{0}.

    This copula possesses the following methods:

    * ``evaluate_cdf`` and ``check_copula``

    (``check_copula`` checks that `marginals` consist of solely 2 continuous univariate distributions).
    """
    def __init__(self, theta):
        super().__init__(theta=theta)

    def evaluate_cdf(self, first_uniform, second_uniform):
        theta = self.parameters['theta']
        tmp_ratio = (np.exp(-theta * first_uniform) - 1.) * \
                    (np.exp(-theta * second_uniform) - 1.) / (np.exp(-theta) - 1.)
        cdf_val = -1. / theta * np.log(1. + tmp_ratio)
        return cdf_val
