import numpy as np
from beartype import beartype

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
    def __init__(self, theta: float):
        super().__init__(theta=theta)

    def evaluate_cdf(self, unit_uniform_samples):
        theta, u, v = self.extract_data(unit_uniform_samples)
        tmp_ratio = (np.exp(-theta * u) - 1.) * \
                    (np.exp(-theta * v) - 1.) / (np.exp(-theta) - 1.)
        cdf_val = -1. / theta * np.log(1. + tmp_ratio)
        return cdf_val

    def extract_data(self, unit_uniform_samples):
        u = unit_uniform_samples[:, 0]
        v = unit_uniform_samples[:, 1]
        theta = self.parameters['theta']
        return theta, u, v
