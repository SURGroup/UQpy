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
    """

    def __init__(self, theta):
        super().__init__(theta=theta)

    def evaluate_cdf(self, first_uniform, second_uniform):
        theta = self.parameters['theta']
        cdf_val = exp(-((-log(first_uniform)) ** theta + (-log(second_uniform)) ** theta) ** (1 / theta))

        return cdf_val

    def evaluate_pdf(self, first_uniform, second_uniform):
        theta = self.parameters['theta']
        c = exp(-((-log(first_uniform)) ** theta + (-log(second_uniform)) ** theta) ** (1 / theta))

        pdf_val = c * 1 / first_uniform * 1 / second_uniform * \
                  ((-log(first_uniform)) ** theta + (-log(second_uniform)) ** theta) ** (-2 + 2 / theta) \
                  * (log(first_uniform) * log(second_uniform)) ** (theta - 1) * \
                  (1 + (theta - 1) * ((-log(first_uniform)) ** theta + (-log(second_uniform)) ** theta) ** (-1 / theta))
        return pdf_val


