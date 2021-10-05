import itertools
import sys
import scipy.spatial.distance as sd

from UQpy.dimension_reduction.SVD import SVD
from UQpy.dimension_reduction.grassman.manifold_projections.KernelComposition import KernelComposition, \
    CompositionAction
from UQpy.dimension_reduction.grassman.manifold_projections.baseclass.ManifoldProjection import ManifoldProjection
from UQpy.dimension_reduction.grassman.methods.ExpMap import exp_map
from UQpy.dimension_reduction.grassman.methods.LogMap import log_map
from UQpy.dimension_reduction.kernels.grassmanian.baseclass.Kernel import Kernel
from UQpy.utilities.ValidationTypes import Numpy2DFloatArray
from UQpy.utilities.Utilities import *


class SvdProjection(ManifoldProjection):

    def __init__(self, input_points: list[Numpy2DFloatArray],
                 p_planes_dimensions: int,
                 kernel_composition: KernelComposition = KernelComposition.LEFT):
        self.kernel_composition = kernel_composition
        self.data = input_points

        points_number = len(input_points)

        n_left = []
        n_right = []
        for i in range(points_number):
            n_left.append(max(np.shape(input_points[i])))
            n_right.append(min(np.shape(input_points[i])))

        bool_left = (n_left.count(n_left[0]) != len(n_left))
        bool_right = (n_right.count(n_right[0]) != len(n_right))

        if bool_left and bool_right:
            raise TypeError('UQpy: The shape of the input matrices must be the same.')
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
                        'UQpy: The dimension of the input data is not consistent with `p` of G(n,p).')  # write something that makes sense

        ranks = np.ones(points_number) * [int(p_planes_dimensions)]
        ranks = ranks.tolist()

        ranks = list(map(int, ranks))

        psi = []  # initialize the left singular eigenvectors as a list.
        sigma = []  # initialize the singular values as a list.
        phi = []  # initialize the right singular eigenvectors as a list.
        for i in range(points_number):
            u, s, v = SVD.factorize(input_points[i], int(ranks[i]))
            psi.append(u)
            sigma.append(np.diag(s))
            phi.append(v)

        self.input_points = input_points
        self.psi = psi
        self.sigma = sigma
        self.phi = phi

        self.n_psi = n_psi
        self.n_phi = n_phi
        self.p_planes_dimensions = p_planes_dimensions
        self.ranks = ranks
        self.points_number = points_number
        self.max_rank = int(np.max(ranks))

    def reconstruct_solution(self, karcher_mean, interpolation, coordinates, point, element_wise=True):
        # Find the Karcher mean.
        ref_psi = karcher_mean.compute_mean(self.psi)
        ref_phi = karcher_mean.compute_mean(self.phi)

        # Reshape the vector containing the singular values as a diagonal matrix.
        sigma_m = []
        for i in range(len(self.sigma)):
            sigma_m.append(np.diag(self.sigma[i]))

        # Project the points on the manifold to the tangent space created over the Karcher mean.
        gamma_psi = log_map(points_grassmann=self.psi, reference_point=ref_psi)
        gamma_phi = log_map(points_grassmann=self.phi, reference_point=ref_phi)

        # Perform the interpolation in the tangent space.
        interp_psi = interpolation \
            .interpolate_sample(coordinates=coordinates, samples=gamma_psi,
                                point=point, element_wise=element_wise)
        interp_phi = interpolation \
            .interpolate_sample(coordinates=coordinates, samples=gamma_phi,
                                point=point, element_wise=element_wise)
        interp_sigma = interpolation \
            .interpolate_sample(coordinates=coordinates, samples=sigma_m,
                                point=point, element_wise=element_wise)

        # Map the interpolated point back to the manifold.
        psi_tilde = exp_map(points_tangent=[interp_psi], reference_point=ref_psi)
        phi_tilde = exp_map(points_tangent=[interp_phi], reference_point=ref_phi)

        # Estimate the interpolated solution.
        psi_tilde = np.array(psi_tilde[0])
        phi_tilde = np.array(phi_tilde[0])
        interpolated = np.dot(np.dot(psi_tilde, interp_sigma), phi_tilde.T)

        return interpolated

    def evaluate_matrix(self, operator: Kernel):
        kernel_psi = self.__apply_operator(self.psi, p_dim=self.p_planes_dimensions, operator=operator)
        kernel_phi = self.__apply_operator(self.phi, p_dim=self.p_planes_dimensions, operator=operator)
        return CompositionAction[self.kernel_composition.name](kernel_psi, kernel_phi)

    def __apply_operator(self, points, p_dim, operator):

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
