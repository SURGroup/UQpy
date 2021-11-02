import itertools
from UQpy.dimension_reduction.grassmann_manifold.manifold_projections.baseclass.ManifoldProjection import (
    ManifoldProjection,
)
from UQpy.dimension_reduction.kernels.baseclass.Kernel import Kernel
from UQpy.utilities.ValidationTypes import Numpy2DFloatArray
import numpy as np
import sys
import scipy.spatial.distance as sd


class QrProjection(ManifoldProjection):
    def __init__(self, input_points: list[Numpy2DFloatArray], p_planes_dimensions: int):
        self.data = input_points

        points_number = len(input_points)

        n_left = []
        n_right = []
        for i in range(points_number):
            n_left.append(max(np.shape(input_points[i])))
            n_right.append(min(np.shape(input_points[i])))

        bool_left = n_left.count(n_left[0]) != len(n_left)
        bool_right = n_right.count(n_right[0]) != len(n_right)

        if bool_left and bool_right:
            raise TypeError("UQpy: The shape of the input matrices must be the same.")
        else:
            n_psi = n_left[0]
            n_phi = n_right[0]

        ranks = []
        for i in range(points_number):
            ranks.append(np.linalg.matrix_rank(input_points[i]))

        if p_planes_dimensions == 0:
            p_planes_dimensions = int(min(ranks))
        elif p_planes_dimensions == sys.maxsize:
            p_planes_dimensions = int(max(ranks))
        else:
            for i in range(points_number):
                if min(np.shape(input_points[i])) < p_planes_dimensions:
                    raise ValueError(
                        "UQpy: The dimension of the input data is not consistent with `p` of G(n,p)."
                    )  # write something that makes sense

        ranks = np.ones(points_number) * [int(p_planes_dimensions)]
        ranks = ranks.tolist()

        ranks = list(map(int, ranks))

        q = []
        r = []
        for i in range(points_number):
            orthonormal_matrix, upper_triangular = np.linalg.qr(input_points[i])
            q.append(orthonormal_matrix)
            r.append(upper_triangular)
        self.q = q
        self.r = r
        self.p_planes_dimensions = p_planes_dimensions
        self.ranks = ranks
        self.points_number = points_number
        self.max_rank = int(np.max(ranks))

    def evaluate_matrix(self, operator: Kernel):
        return self.__apply_operator(
            self.q, p_dim=self.p_planes_dimensions, operator=operator
        )

    def __apply_operator(self, points, p_dim, operator):

        # Check points for type and shape consistency.
        # -----------------------------------------------------------
        if not isinstance(points, list) and not isinstance(points, np.ndarray):
            raise TypeError("UQpy: `points` must be either list or numpy.ndarray.")

        nargs = len(points)

        if nargs < 2:
            raise ValueError("UQpy: At least two matrices must be provided.")
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

            ker = operator.apply_method(x0, x1)
            kernel_list.append(ker)

        # Diagonal entries of the kernel matrix.
        kernel_diag = []
        for id_elem in range(nargs):
            xd = np.asarray(points[id_elem])
            xd = xd[:, :p_dim]

            kerd = operator.apply_method(xd, xd)
            kernel_diag.append(kerd)

        # Add the diagonals and off-diagonal entries of the Kernel matrix.
        kernel_matrix = sd.squareform(np.array(kernel_list)) + np.diag(kernel_diag)

        # Return the kernel matrix.
        return kernel_matrix

    def reconstruct_solution(
        self, karcher_mean, interpolation, coordinates, point, element_wise=True
    ):
        pass
