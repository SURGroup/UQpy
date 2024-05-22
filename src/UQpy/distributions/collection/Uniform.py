import numpy as np
import scipy.stats as stats
from beartype import beartype
from typing import Union
from UQpy.distributions.baseclass import DistributionContinuous1D
from UQpy.utilities.ValidationTypes import NumericArrayLike
from line_profiler_pycharm import profile

@beartype
class Uniform(DistributionContinuous1D):
    def __init__(
        self, loc: Union[None, float, int] = 0.0, scale: Union[None, float, int] = 1.0
    ):
        """

        :param loc: lower bound
        :param scale: range
        """
        super().__init__(loc=loc, scale=scale, ordered_parameters=("loc", "scale"))
        self._construct_from_scipy(scipy_name=stats.uniform)

        self.pdf = self.__probability_density_function
        self.cdf = self.__cumulative_distribution_function
        self.icdf = self.__inverse_cumulative_distribution_function

    @profile
    def __probability_density_function(
        self, x: NumericArrayLike
    ) -> Union[float, np.ndarray]:
        """Probability Density Function for the uniform distribution

        :param x: Points at which to evaluate the probability density function
        :return: pdf at all points in x
        """
        x_array = np.atleast_1d(x)
        loc = self.parameters["loc"]
        scale = self.parameters["scale"]
        mask = (loc <= x_array) & (x_array <= loc + scale)
        pdf = np.zeros_like(x_array)
        pdf[mask] = 1 / scale
        if isinstance(x, int) or isinstance(x, float):
            return pdf[0]
        return pdf

    @profile
    def __cumulative_distribution_function(
        self, x: NumericArrayLike
    ) -> Union[float, np.ndarray]:
        """Cumulative Distribution Function for the Uniform Distribution

        :param x: Points at which to evaluate the cumulative distribution function
        :return:  cdf of ``x`` as defined by :math:`F_X(x)`
        """
        x_array = np.atleast_1d(x)
        loc = self.parameters["loc"]
        scale = self.parameters["scale"]
        cdf = np.zeros_like(x_array)
        middle_mask = (loc < x_array) & (x_array < loc + scale)
        upper_mask = loc + scale <= x_array
        cdf[middle_mask] = (x_array[middle_mask] - loc) / scale
        cdf[upper_mask] = 1
        if isinstance(x, int) or isinstance(x, float):
            return cdf[0]
        return cdf

    @profile
    def __inverse_cumulative_distribution_function(
        self, x: NumericArrayLike
    ) -> Union[float, np.ndarray]:
        """Inverse cumulative distribution function for uniform distribution

        :param x: Point at which to evaluate the inverse cumulative distribution function
        :return: inverse cdf of ``x`` as defined by :math:`F^{-1}_X(x)`
        """
        x_array = np.atleast_1d(x)
        loc = self.parameters["loc"]
        scale = self.parameters["scale"]
        icdf = np.full(x_array.shape, np.nan)
        mask = (0 <= x_array) & (x_array <= 1)
        icdf[mask] = loc + (x_array[mask] * scale)
        if isinstance(x, int) or isinstance(x, float):
            return icdf[0]
        return icdf
