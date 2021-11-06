import numpy as np
from scipy.interpolate import LinearNDInterpolator

from UQpy.utilities.ValidationTypes import NumpyFloatArray
from UQpy.dimension_reduction.grassmann_manifold.interpolation.baseclass.InterpolationMethod import (
    InterpolationMethod,
)


class LinearInterpolation(InterpolationMethod):
    def interpolate(self,
                    coordinates: NumpyFloatArray,
                    samples: NumpyFloatArray,
                    point: NumpyFloatArray):
        nargs = len(samples)
        shape_ref = np.shape(samples[0])
        for i in range(1, nargs):
            if np.shape(samples[i]) != shape_ref:
                raise TypeError("UQpy: Input matrices have different shape.")

        interp = LinearNDInterpolator(coordinates, samples)
        interp_point = interp(point)
        return interp_point
