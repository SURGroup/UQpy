from typing import Union
import numpy as np
import scipy.stats as stats
from beartype import beartype
from scipy.special import erf, erfinv
from UQpy.distributions.baseclass import DistributionContinuous1D
from UQpy.utilities.ValidationTypes import NumericArrayLike


@beartype
class Normal(DistributionContinuous1D):
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
        self.icdf = self.__inverse_cumulative_distribution_function

    def __probability_density_function(
        self, x: NumericArrayLike
    ) -> Union[float, np.ndarray]:
        """Probability density function for normal distribution

        :param x: Points to evaluate pdf at
        :return: PDF of ``x`` as defined by :math:`f_x(x)`
        """
        x_array = np.atleast_1d(x)
        mean = self.parameters["loc"]
        standard_deviation = self.parameters["scale"]
        normalizing_constant = 1 / (standard_deviation * np.sqrt(2 * np.pi))
        pdf = normalizing_constant * np.exp(
            -0.5 * ((x_array - mean) / standard_deviation) ** 2
        )
        if isinstance(x, int) or isinstance(x, float):
            return pdf[0]
        return pdf

    def __cumulative_distribution_function(
        self, x: NumericArrayLike
    ) -> Union[float, np.ndarray]:
        """Cumulative distribution function for the normal distribution defined with the error function

        :param x: Points to evaluate cdf at
        :return: CDF of ``x`` as defined by :math:`F_X(x)`
        """
        x_array = np.atleast_1d(x)
        mean = self.parameters["loc"]
        standard_deviation = self.parameters["scale"]
        erf_input = (x_array - mean) / (standard_deviation * np.sqrt(2.0))
        cdf = (1.0 + erf(erf_input)) / 2.0
        if isinstance(x, int) or isinstance(x, float):
            return cdf[0]
        return cdf

    def __inverse_cumulative_distribution_function(
        self, y: NumericArrayLike
    ) -> Union[float, np.ndarray]:
        """Compute the inverse CDF for the normal distribution with the inverse error function

        :param y:
        :return:
        """
        y_array = np.atleast_1d(y)
        mean = self.parameters["loc"]
        standard_deviation = self.parameters["scale"]
        normalized_icdf = erfinv((2 * y_array) - 1)
        icdf = (normalized_icdf * standard_deviation * np.sqrt(2.0)) + mean
        if isinstance(y, int) or isinstance(y, float):
            return icdf[0]
        return icdf
