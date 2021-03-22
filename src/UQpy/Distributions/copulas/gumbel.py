from UQpy.Distributions.baseclass import Copula
from UQpy.Distributions.baseclass import DistributionContinuous1D
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

    def evaluate_cdf(self, unif):
        if unif.shape[1] > 2:
            raise ValueError('Maximum dimension for the Gumbel Copula is 2.')
        if self.params['theta'] == 1:
            return prod(unif, axis=1)

        u = unif[:, 0]
        v = unif[:, 1]
        theta = self.params['theta']
        cdf_val = exp(-((-log(u)) ** theta + (-log(v)) ** theta) ** (1 / theta))

        return cdf_val

    def evaluate_pdf(self, unif):
        if unif.shape[1] > 2:
            raise ValueError('Maximum dimension for the Gumbel Copula is 2.')
        if self.params['theta'] == 1:
            return ones(unif.shape[0])

        u = unif[:, 0]
        v = unif[:, 1]
        theta = self.params['theta']
        c = exp(-((-log(u)) ** theta + (-log(v)) ** theta) ** (1 / theta))

        pdf_val = c * 1 / u * 1 / v * ((-log(u)) ** theta + (-log(v)) ** theta) ** (-2 + 2 / theta) \
                  * (log(u) * log(v)) ** (theta - 1) * \
                  (1 + (theta - 1) * ((-log(u)) ** theta + (-log(v)) ** theta) ** (-1 / theta))
        return pdf_val

    def check_marginals(self, marginals):
        """
        Check that marginals contains 2 continuous univariate distributions.
        """
        if len(marginals) != 2:
            raise ValueError('Maximum dimension for the Gumbel Copula is 2.')
        if not all(isinstance(m, DistributionContinuous1D) for m in marginals):
            raise ValueError('Marginals should be 1d continuous distributions.')




