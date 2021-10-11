from UQpy.distributions.baseclass.DistributionContinuous1D import DistributionContinuous1D
from abc import ABC


class Copula(ABC):
    """
    Define a copula for a multivariate distribution whose dependence structure is defined with a copula.

    This class is used in support of the ``JointCopula`` distribution class.

    **Attributes:**

    * **kwargs** (`dict`):
        Parameters of the copula.

    * **ordered_parameters** (`list`):
        List of parameter names

    **Methods:**

    **check_marginals** *(marginals)*
        Perform some checks on the marginals, raise errors if necessary.

        As an example, Archimedian copula are only defined for bi-variate continuous distributions, thus this method
        checks that marginals is of length 2 and continuous, and raise an error if that is not the case.

        **Input:**

        * **marginals** (list[DistributionContinuous1D]):
            List of 1D continuous distributions.

        **Output/Returns:**

        No outputs, this code raises errors if necessary.

    **get_parameters** *()*

        **Output/Returns:**

        A list containing the parameter names.

    **update_parameters** *(**kwargs)*
        Given a dictionary with keys the names and values the new parameter values,
        the method updates the current values.

        **Input:**

        * **kwargs**:
            Dictionary containing the updated parameter values.

        **Output/Returns:**

        A list containing the parameter names.
    """
    def __init__(self, ordered_parameters: dict = None, **kwargs):
        self.parameters = kwargs
        self.ordered_parameters = ordered_parameters
        if self.ordered_parameters is None:
            self.ordered_parameters = tuple(kwargs.keys())
        if len(self.ordered_parameters) != len(self.parameters):
            raise ValueError('Inconsistent dimensions between ordered_parameters tuple and parameters dictionary.')

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
