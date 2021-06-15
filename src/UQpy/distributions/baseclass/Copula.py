from UQpy.distributions.baseclass import DistributionContinuous1D
from abc import ABC


class Copula(ABC):

    """
    Define a copula for a multivariate distribution whose dependence structure is defined with a copula.

    This class is used in support of the ``JointCopula`` distribution class.

    **Attributes:**

    * **params** (`dict`):
        Parameters of the copula.

    * **order_params** (`list`):
        List of parameter names

    **Methods:**

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

    **evaluate_pdf** *(unif)*
        Compute the copula pdf :math:`c(u_1, u_2, ..., u_d)` for a `d`-variate uniform distribution.

        For a generic multivariate distribution with marginals pdfs :math:`f_1, ..., f_d` and marginals cdfs
        :math:`F_1, ..., F_d`, the joint pdf is computed as:

        :math:`f(x_1, ..., x_d) = c(u_1, u_2, ..., u_d) f_1(x_1) ... f_d(x_d)`

        where :math:`u_i = F_i(x_i)` is uniformly distributed. This computation is performed in the ``JointCopula.pdf``
        method.

        **Input:**

        * **unif** (`ndarray`):
            Points (uniformly distributed) at which to evaluate the copula pdf, must be of shape `(npoints, dimension)`.

        **Output/Returns:**

        * (`tuple`):
            Values of the copula pdf term, ndarray of shape `(npoints, )`.

    **check_marginals** *(marginals)*
        Perform some checks on the marginals, raise errors if necessary.

        As an example, Archimedian copula are only defined for bi-variate continuous distributions, thus this method
        checks that marginals is of length 2 and continuous, and raise an error if that is not the case.

        **Input:**

        * **unif** (ndarray):
            Points (uniformly distributed) at which to evaluate the copula pdf, must be of shape
            ``(npoints, dimension)``.

        **Output/Returns:**

        No outputs, this code raises errors if necessary.
    """
    def __init__(self, ordered_parameters=None, **kwargs):
        theta = kwargs['theta']
        if theta is not None and ((not isinstance(theta, (float, int))) or (theta < -1 or theta == 0.)):
            raise ValueError('Input theta should be a float in [-1, +oo).')
        self.parameters = kwargs
        self.ordered_parameters = ordered_parameters if not None else tuple(kwargs.keys())
        if len(self.ordered_parameters) != len(self.parameters):
            raise ValueError('Inconsistent dimensions between order_params tuple and params dictionary.')

    def get_parameters(self):
        return self.parameters

    def update_parameters(self, **kwargs):
        for key in kwargs.keys():
            if key not in self.parameters.keys():
                raise ValueError('Wrong parameter name.')
            self.parameters[key] = kwargs[key]

    @staticmethod
    def check_marginals(marginals):
        """
        Check that marginals contains 2 continuous univariate distributions.
        """
        if len(marginals) != 2:
            raise ValueError('Maximum dimension for the Copula is 2.')
        if not all(isinstance(m, DistributionContinuous1D) for m in marginals):
            raise ValueError('Marginals should be 1d continuous distributions.')
