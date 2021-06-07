from UQpy.distributions.baseclass import Copula
from UQpy.distributions.baseclass import DistributionContinuous1D
from numpy import prod, log, ones, exp


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
        # Check the input copula_params
        if theta is not None and ((not isinstance(theta, (float, int))) or (theta < 1)):
            raise ValueError('Input theta should be a float in [1, +oo).')
        super().__init__(theta=theta)

    def evaluate_cdf(self, uniform_distributions):
        if uniform_distributions.shape[1] > 2:
            raise ValueError('Maximum dimension for the Gumbel Copula is 2.')

        first_uniform = uniform_distributions[:, 0]
        second_uniform = uniform_distributions[:, 1]
        theta = self.params['theta']
        cdf_val = exp(-((-log(first_uniform)) ** theta + (-log(second_uniform)) ** theta) ** (1 / theta))

        return cdf_val

    def evaluate_pdf(self, uniform_distributions):
        if uniform_distributions.shape[1] > 2:
            raise ValueError('Maximum dimension for the Gumbel Copula is 2.')

        first_uniform = uniform_distributions[:, 0]
        second_uniform = uniform_distributions[:, 1]
        theta = self.params['theta']
        c = exp(-((-log(first_uniform)) ** theta + (-log(second_uniform)) ** theta) ** (1 / theta))

        pdf_val = c * 1 / first_uniform * 1 / second_uniform * \
                  ((-log(first_uniform)) ** theta + (-log(second_uniform)) ** theta) ** (-2 + 2 / theta) \
                  * (log(first_uniform) * log(second_uniform)) ** (theta - 1) * \
                  (1 + (theta - 1) * ((-log(first_uniform)) ** theta + (-log(second_uniform)) ** theta) ** (-1 / theta))
        return pdf_val

    @staticmethod
    def check_marginals(marginals):
        """
        Check that marginals contains 2 continuous univariate distributions.
        """
        if len(marginals) != 2:
            raise ValueError('Maximum dimension for the Gumbel Copula is 2.')
        if not all(isinstance(m, DistributionContinuous1D) for m in marginals):
            raise ValueError('Marginals should be 1d continuous distributions.')
