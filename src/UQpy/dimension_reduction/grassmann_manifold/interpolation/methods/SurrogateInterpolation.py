import copy

from UQpy.dimension_reduction.grassmann_manifold import GrassmannPoint
from UQpy.dimension_reduction.grassmann_manifold.interpolation.baseclass.InterpolationMethod import InterpolationMethod
from UQpy.surrogates.baseclass import Surrogate
import numpy as np
from UQpy.utilities.ValidationTypes import NumpyFloatArray


class SurrogateInterpolation(InterpolationMethod):
    """
    A class to perform interpolation using a UQpy surrogate model.
    """

    def __init__(self, surrogate: Surrogate):
        """

        :param surrogate: The interpolant to be used. It must be an object of type Surrogate.
        """
        self.surrogates = [[]]
        self.surrogate = surrogate
        self.samples = None

    def interpolate(self, point: NumpyFloatArray) -> NumpyFloatArray:
        """
        Method to perform the interpolation.

        :param point: Point to interpolation
        """
        nargs = len(self.samples)

        shape_ref = np.shape(self.samples[0])
        for i in range(1, nargs):
            if np.shape(self.samples[i]) != shape_ref:
                raise TypeError("UQpy: Input matrices have different shape.")

        shape_ref = np.shape(self.samples[0])
        interp_point = np.zeros(shape_ref)
        n_rows = self.samples[0].shape[0]
        n_cols = self.samples[0].shape[1]

        for j in range(n_rows):
            for k in range(n_cols):
                y = self.surrogate.predict(point, return_std=False)
                interp_point[j, k] = y

        return interp_point

    def fit(self, coordinates: NumpyFloatArray, manifold_data, samples: list[NumpyFloatArray]):
        self.samples = samples
        nargs = len(samples)
        n_rows = samples[0].shape[0]
        n_cols = samples[0].shape[1]
        for j in range(n_rows):
            for k in range(n_cols):
                val_data = [[samples[i][j, k]] for i in range(nargs)]
                # if all the elements of val_data are the same.
                val_data = np.array(val_data)
                self.surrogates[j][k] = copy.copy(self.surrogate)
                self.surrogates[j][k].fit(coordinates, val_data)
