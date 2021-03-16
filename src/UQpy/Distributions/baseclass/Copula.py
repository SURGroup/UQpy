########################################################################################################################
#        Copulas
########################################################################################################################

class Copula:
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

    """
    def __init__(self, order_params=None, **kwargs):
        self.params = kwargs
        self.order_params = order_params
        if self.order_params is None:
            self.order_params = tuple(kwargs.keys())
        if len(self.order_params) != len(self.params):
            raise ValueError('Inconsistent dimensions between order_params tuple and params dictionary.')

    def check_marginals(self, marginals):
        """
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
        pass

    def get_params(self):
        return self.params

    def update_params(self, **kwargs):
        for key in kwargs.keys():
            if key not in self.params.keys():
                raise ValueError('Wrong parameter name.')
            self.params[key] = kwargs[key]
