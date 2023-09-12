from typing import Union
import numpy as np
import scipy.stats as stats
from beartype import beartype
from scipy.special import erf, erfinv
from UQpy.distributions.baseclass import DistributionContinuous1D
from UQpy.utilities.ValidationTypes import NumericArrayLike


class Normal(DistributionContinuous1D):

    @beartype
    def __init__(
        self, loc: Union[None, float, int] = 0.0, scale: Union[None, float, int] = 1.0
    ):
        """

        :param loc: location parameter
        :param scale: scale parameter
        """
        super().__init__(loc=loc, scale=scale, ordered_parameters=("loc", "scale"))
        self._construct_from_scipy(scipy_name=stats.norm)
        self.pdf = self.__probability_density_function
        self.cdf = self.__cumulative_distribution_function

    @beartype
    def __probability_density_function(self, x: NumericArrayLike):
        """Probability density function for normal distribution

        :param x:
        :return:
        """
        x_array = np.atleast_1d(x)
        mean = self.parameters['loc']
        standard_deviation = self.parameters['scale']
        normalizing_constant = 1 / (standard_deviation * np.sqrt(2 * np.pi))
        pdf = normalizing_constant * np.exp(-0.5 * ((x_array - mean) / standard_deviation)**2)
        if isinstance(x, int) or isinstance(x, float):
            return pdf[0]
        return pdf

    @beartype
    def __cumulative_distribution_function(self, x: NumericArrayLike):
        """Cumulative distribution function for the normal distribution defined with the error function

        :param x:
        :return:
        """
        x_array = np.atleast_1d(x)
        mean = self.parameters['loc']
        standard_deviation = self.parameters['scale']
        erf_input = (x_array - mean) / (standard_deviation * np.sqrt(2.0))
        cdf = (1.0 + erf(erf_input)) / 2.0
        if isinstance(x, int) or isinstance(x, float):
            return cdf[0]
        return cdf
