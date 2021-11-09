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
        self.surrogate = surrogate

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

        shape_ref = np.shape(samples[0])
        interp_point = np.zeros(shape_ref)
        n_rows = samples[0].shape[0]
        n_cols = samples[0].shape[1]

        for j in range(n_rows):
            for k in range(n_cols):
                val_data = []
                for i in range(nargs):
                    val_data.append([samples[i][j, k]])

                # if all the elements of val_data are the same.
                if val_data.count(val_data[0]) == len(val_data):
                    val = np.array(val_data)
                    y = val[0]
                else:
                    val_data = np.array(val_data)
                    self.surrogate.fit(coordinates, val_data)
                    y = self.surrogate.predict(point, return_std=False)

                interp_point[j, k] = y

        return interp_point
