import itertools

from UQpy.utilities.ValidationTypes import Numpy2DFloatArray
from UQpy.dimension_reduction.grassman.manifold_projections.baseclass.ManifoldProjection import ManifoldProjection
from UQpy.dimension_reduction.kernels.baseclass.Kernel import Kernel
import numpy as np
import scipy.spatial.distance as sd


class OrthoMatrixPoints(ManifoldProjection):

    def __init__(self, input_points: list[Numpy2DFloatArray],
                 p_planes_dimensions: int):

        points_number = max(np.shape(input_points[0]))
        for i in range(len(input_points)):
            if points_number != max(np.shape(input_points[i])):
                raise TypeError('UQpy: The shape of the input matrices must be the same.')

        # Check the embedding dimension and its consistency.
        p_dim = []
        for i in range(len(input_points)):
            p_dim.append(min(np.shape(np.array(input_points[i]))))

        if p_dim.count(p_dim[0]) != len(p_dim):
            raise TypeError('UQpy: The input points do not belong to the same manifold.')
        else:
            p0 = p_dim[0]
            if p0 != p_planes_dimensions:
                raise ValueError('UQpy: The input points do not belong to the manifold G(n,p).')

        self.data = input_points
        self.p_planes_dimensions = p0

    def evaluate_matrix(self, kernel: Kernel):
        return self.__estimate_kernel(self.data, p_dim=self.p_planes_dimensions, kernel=kernel)

    def interpolate(self, karcher_mean, interpolator, coordinates, point, element_wise=True):
        pass

    def __estimate_kernel(self, points, p_dim, kernel):

        # Check points for type and shape consistency.
        # -----------------------------------------------------------
        if not isinstance(points, list) and not isinstance(points, np.ndarray):
            raise TypeError('UQpy: `points` must be either list or numpy.ndarray.')

        nargs = len(points)

        if nargs < 2:
            raise ValueError('UQpy: At least two matrices must be provided.')
        # ------------------------------------------------------------

        # Define the pairs of points to compute the entries of the kernel matrix.
        indices = range(nargs)
        pairs = list(itertools.combinations(indices, 2))

        # Estimate off-diagonal entries of the kernel matrix.
        kernel_list = []
        for id_pair in range(np.shape(pairs)[0]):
            ii = pairs[id_pair][0]  # Point i
            jj = pairs[id_pair][1]  # Point j

            x0 = np.asarray(points[ii])[:, :p_dim]
            x1 = np.asarray(points[jj])[:, :p_dim]

            ker = kernel.apply_method(x0, x1)
            kernel_list.append(ker)

        # Diagonal entries of the kernel matrix.
        kernel_diag = []
        for id_elem in range(nargs):
            xd = np.asarray(points[id_elem])
            xd = xd[:, :p_dim]

            kerd = kernel.apply_method(xd, xd)
            kernel_diag.append(kerd)

        # Add the diagonals and off-diagonal entries of the Kernel matrix.
        kernel_matrix = sd.squareform(np.array(kernel_list)) + np.diag(kernel_diag)

        # Return the kernel matrix.
        return kernel_matrix