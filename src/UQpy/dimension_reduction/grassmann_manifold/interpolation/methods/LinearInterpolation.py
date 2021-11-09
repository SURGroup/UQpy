import numpy as np
from scipy.interpolate import LinearNDInterpolator

from UQpy.utilities.ValidationTypes import NumpyFloatArray
from UQpy.dimension_reduction.grassmann_manifold.interpolation.baseclass.InterpolationMethod import (
    InterpolationMethod,
)


class LinearInterpolation(InterpolationMethod):
    """
    A class to perform piecewise linear interpolation in high dimensions.
    """
    def interpolate(self,
                    coordinates: NumpyFloatArray,
                    samples: NumpyFloatArray,
                    point: NumpyFloatArray) -> NumpyFloatArray:
        """
        Method to perform the interpolation.

        :param coordinates: Nodes of the interpolant
        :param samples: Data to interpolate
        :param point: Point to interpolation

        """
        nargs = len(samples)
        shape_ref = np.shape(samples[0])
        for i in range(1, nargs):
            if np.shape(samples[i]) != shape_ref:
                raise TypeError("UQpy: Input matrices have different shape.")

        interp = LinearNDInterpolator(coordinates, samples)
        interp_point = interp(point)
        return interp_point
