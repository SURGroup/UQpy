
from sklearn.gaussian_process import GaussianProcessRegressor

from UQpy.surrogates.kriging.Kriging import Kriging
from UQpy.dimension_reduction.grassmann_manifold.interpolation.LinearInterpolation import (
    LinearInterpolation,
)
import numpy as np

from UQpy.surrogates.baseclass import Surrogate


class Interpolation:
    def __init__(self, interpolation_method):
        self.interpolation_method = interpolation_method

    def interpolate_sample(self, coordinates, samples, point, element_wise=True):
        if isinstance(samples, list):
            samples = np.array(samples)

        # Test if the sample is stored as a list
        if isinstance(point, list):
            point = np.array(point)

        # Test if the nodes are stored as a list
        if isinstance(coordinates, list):
            coordinates = np.array(coordinates)

        nargs = len(samples)

        if self.interpolation_method is None:
            raise TypeError("UQpy: `interp_object` cannot be NoneType")
        else:
            if self.interpolation_method is LinearInterpolation:
                element_wise = False

            if isinstance(self.interpolation_method, Surrogate):
                element_wise = True

        shape_ref = np.shape(samples[0])
        for i in range(1, nargs):
            if np.shape(samples[i]) != shape_ref:
                raise TypeError("UQpy: Input matrices have different shape.")

        if element_wise:
            shape_ref = np.shape(samples[0])
            interp_point = np.zeros(shape_ref)
            nrows = samples[0].shape[0]
            ncols = samples[0].shape[1]

            val_data = []
            dim = np.shape(coordinates)[1]

            for j in range(nrows):
                for k in range(ncols):
                    val_data = []
                    for i in range(nargs):
                        val_data.append([samples[i][j, k]])

                    # if all the elements of val_data are the same.
                    if val_data.count(val_data[0]) == len(val_data):
                        val = np.array(val_data)
                        y = val[0]
                    else:
                        val_data = np.array(val_data)
                        if isinstance(
                            self.interpolation_method,
                            (Kriging, GaussianProcessRegressor),
                        ):
                            self.interpolation_method.fit(coordinates, val_data)
                            y = self.interpolation_method.predict(
                                point, return_std=False
                            )
                        else:
                            y = self.interpolation_method.interpolate(
                                coordinates, samples, point
                            )

                    interp_point[j, k] = y

        else:
            if isinstance(self.interpolation_method, Surrogate):
                raise TypeError(
                    "UQpy: Kriging only can be used in the elementwise interpolation."
                )
            else:
                interp_point = self.interpolation_method.interpolate(
                    coordinates, samples, point
                )

        return interp_point
