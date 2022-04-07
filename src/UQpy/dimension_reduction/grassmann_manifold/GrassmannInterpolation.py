import copy
from typing import Union

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from UQpy.dimension_reduction.grassmann_manifold import GrassmannOperations
from UQpy.surrogates.baseclass import Surrogate
from UQpy.utilities.GrassmannPoint import GrassmannPoint
from UQpy.utilities.ValidationTypes import NumpyFloatArray
from UQpy.utilities.distances import GrassmannianDistance


class GrassmannInterpolation:

    def __init__(self, interpolation_method: Union[Surrogate, callable, None],
                 manifold_data: list[GrassmannPoint],
                 coordinates: Union[np.ndarray, list[NumpyFloatArray]],
                 distance: GrassmannianDistance,
                 optimization_method: str = "GradientDescent"):
        """
        A class to perform interpolation of points on the Grassmann manifold.

        :param interpolation_method: Type of interpolation to perform. This may be specified as a :class:`Surrogate`
            object or a callable function. If :any:`None`, then multi-linear interpolation is performed.
        :param manifold_data: Data points on the Grassmann manifold.
        :param optimization_method: Optimization method for calculating the Karcher mean. See
            :py:meth:`.GrassmannOperations.karcher_mean`.
        :param coordinates: Nodes of the interpolant.
        :param distance:  Distance measure.
        """
        self.interpolation_method = interpolation_method

        self.mean = GrassmannOperations.karcher_mean(grassmann_points=manifold_data,
                                                     optimization_method=optimization_method,
                                                     distance=distance)

        self.tangent_points = GrassmannOperations.log_map(grassmann_points=manifold_data,
                                                          reference_point=self.mean)

        if self.interpolation_method is Surrogate:
            self.surrogates = [[]]
            nargs = len(self.tangent_points)
            n_rows = self.tangent_points[0].shape[0]
            n_cols = self.tangent_points[0].shape[1]
            for j in range(n_rows):
                for k in range(n_cols):
                    val_data = [[self.tangent_points[i][j, k]] for i in range(nargs)]
                    # if all the elements of val_data are the same.
                    val_data = np.array(val_data)
                    self.surrogates[j][k] = copy.copy(self.interpolation_method)
                    self.surrogates[j][k].fit(coordinates, val_data)

        else:
            self.coordinates = coordinates

    def interpolate_manifold(self, point: np.ndarray):
        """
        :param point:  Point at which to interpolate.
        :return: Interpolated point on the Grassmann manifold.
        """

        shape_ref = np.shape(self.tangent_points[0])
        interp_point = np.zeros(shape_ref)
        n_rows = self.tangent_points[0].shape[0]
        n_cols = self.tangent_points[0].shape[1]

        nargs = len(self.tangent_points)
        shape_ref = np.shape(self.tangent_points[0])
        for i in range(1, nargs):
            if np.shape(self.tangent_points[i]) != shape_ref:
                raise TypeError("UQpy: Input matrices have different shape.")

        if self.interpolation_method is Surrogate:
            for j in range(n_rows):
                for k in range(n_cols):
                    y = self.surrogates[j][k].predict(point, return_std=False)
                    interp_point[j, k] = y
        elif self.interpolation_method is callable:
            interp_point = self.interpolation_method(self.coordinates, self.tangent_points, point)
        else:
            interp = LinearNDInterpolator(self.coordinates, self.tangent_points)
            interp_point = interp(point)

        return GrassmannOperations.exp_map(tangent_points=[interp_point.squeeze()], reference_point=self.mean)[0]
