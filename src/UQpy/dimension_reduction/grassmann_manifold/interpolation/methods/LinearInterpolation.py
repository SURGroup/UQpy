import numpy as np
from scipy.interpolate import LinearNDInterpolator

from UQpy.utilities.ValidationTypes import NumpyFloatArray
from UQpy.dimension_reduction.grassmann_manifold.interpolation.baseclass.InterpolationMethod import (
    InterpolationMethod,
)


class LinearInterpolation(InterpolationMethod):
    def __init__(self):
        """
        A class to perform piecewise linear interpolation in high dimensions.
        """
        self.coordinates=None
        self.samples = None

    def interpolate(self, point: NumpyFloatArray) -> NumpyFloatArray:
        """
        Method to perform the interpolation.

        :param coordinates: Nodes of the interpolant
        :param samples: Data to interpolate
        :param point: Point to interpolation

        """
        nargs = len(self.samples)
        shape_ref = np.shape(self.samples[0])
        for i in range(1, nargs):
            if np.shape(self.samples[i]) != shape_ref:
                raise TypeError("UQpy: Input matrices have different shape.")

        interp = LinearNDInterpolator(self.coordinates, self.samples)
        interp_point = interp(point)
        return interp_point

    def fit(self, coordinates: NumpyFloatArray, manifold_data, samples: list[NumpyFloatArray]):
        self.coordinates = coordinates
        self.samples = samples