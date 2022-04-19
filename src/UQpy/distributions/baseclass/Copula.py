from UQpy.distributions.baseclass.DistributionContinuous1D import (
    DistributionContinuous1D,
)
from abc import ABC
from typing import Union

class Copula(ABC):

    def __init__(self, ordered_parameters: dict = None, **kwargs: dict):
        """
        Define a copula for a multivariate distribution whose dependence structure is defined with a copula.
        This class is used in support of the :class:`.JointCopula` class.

        :param ordered_parameters: List of parameter names
        :param kwargs: Parameters of the copula.
        """
        self.parameters: dict = kwargs
        """Parameters of the copula."""
        self.ordered_parameters: dict = ordered_parameters
        """List of parameter names"""
        if self.ordered_parameters is None:
            self.ordered_parameters = tuple(kwargs.keys())
        if len(self.ordered_parameters) != len(self.parameters):
            raise ValueError(
                "Inconsistent dimensions between ordered_parameters tuple and parameters dictionary."
            )

    def get_parameters(self) -> dict:
        """
        :return: A dictionary containing the parameter names.
        """
        return self.parameters

    def update_parameters(self, **kwargs: dict):
        """
        Given a dictionary with keys the names and values the new parameter values,
        the method updates the current values.

        :param kwargs: Dictionary containing the updated parameter values.
        """
        for key in kwargs.keys():
            if key not in self.parameters.keys():
                raise ValueError("Wrong parameter name.")
            self.parameters[key] = kwargs[key]

    @staticmethod
    def check_marginals(marginals: Union[list, DistributionContinuous1D]):
        """
        Perform some checks on the marginals, raise errors if necessary.

        As an example, Archimedian copulas are only defined for bi-variate continuous distributions, thus this method
        checks that marginals is of length 2 and continuous, and raise an error if that is not the case.

        :param marginals: List of 1D continuous distributions.
        """
        if len(marginals) != 2:
            raise ValueError("Maximum dimension for the Copula is 2.")
        if not all(isinstance(m, DistributionContinuous1D) for m in marginals):
            raise ValueError("Marginals should be 1d continuous distributions.")
