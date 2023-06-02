from typing import Union

import numpy as np
import scipy.stats as stats
from beartype import beartype

from UQpy.distributions.baseclass import DistributionContinuous1D
from UQpy.utilities.ValidationTypes import NumericArrayLike


class Uniform(DistributionContinuous1D):
    @beartype
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

    @beartype
    def __probability_density_function(self, x: NumericArrayLike) -> np.ndarray:
        """Probability Density Function for the uniform distribution

        :param x:
        :return: pdf at all points in x
        """
        x = np.atleast_1d(x)
        loc = self.parameters['loc']
        scale = self.parameters['scale']
        mask = (loc <= x) & (x <= loc + scale)
        pdf = np.zeros_like(x)
        pdf[mask] = 1 / scale
        return pdf

    @beartype
    def __cumulative_distribution_function(self, x: NumericArrayLike) -> np.ndarray:
        """Cumulative Distribution Function for the Uniform Distribution

        :param x:
        """
        x = np.atleast_1d(x)
        loc = self.parameters['loc']
        scale = self.parameters['scale']
        cdf = np.zeros_like(x)
        middle_mask = (loc < x) & (x < loc + scale)
        upper_mask = loc + scale <= x
        cdf[middle_mask] = (x[middle_mask] - loc) / scale
        cdf[upper_mask] = 1
        return cdf

    def __inverse_cumulative_distribution_function(self, y: NumericArrayLike) -> np.ndarray:
        """Inverse cumulative distribution function for uniform distribution

        :param y:
        :return:
        """
        y = np.atleast_1d(y)
        loc = self.parameters['loc']
        scale = self.parameters['scale']
        icdf = np.full(y.shape, np.nan)
        mask = (0 <= y) & (y <= 1)
        icdf[mask] = loc + (y[mask] * scale)
        return icdf
