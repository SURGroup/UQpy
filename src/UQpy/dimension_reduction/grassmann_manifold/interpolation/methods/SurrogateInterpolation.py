from UQpy import InterpolationMethod
from UQpy.surrogates.baseclass import Surrogate
import numpy as np


class SurrogateInterpolation(InterpolationMethod):

    def __init__(self, surrogate: Surrogate):
        self.surrogate = surrogate

    def interpolate(self, coordinates, samples, point):
        nargs = len(samples)

        shape_ref = np.shape(samples[0])
        for i in range(1, nargs):
            if np.shape(samples[i]) != shape_ref:
                raise TypeError("UQpy: Input matrices have different shape.")

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
                    self.surrogate.fit(coordinates, val_data)
                    y = self.surrogate.predict(point, return_std=False)

                interp_point[j, k] = y

        return interp_point
