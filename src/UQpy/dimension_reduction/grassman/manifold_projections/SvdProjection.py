 import itertools
import sys
from typing import List
import scipy.spatial.distance as sd
from UQpy.dimension_reduction.grassman.Grassman import Grassmann
from UQpy.dimension_reduction.grassman.interpolations import LinearInterpolation
from UQpy.dimension_reduction.grassman.manifold_projections.KernelComposition import KernelComposition, \
    CompositionAction
from UQpy.dimension_reduction.grassman.manifold_projections.baseclass.ManifoldProjection import ManifoldProjection
from UQpy.dimension_reduction.kernels.grassmanian.baseclass.Kernel import Kernel
from UQpy.utilities.ValidationTypes import Numpy2DFloatArray
import numpy as np
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
            u, s, v = svd(input_points[i], int(ranks[i]))
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

    def interpolate(self, karcher_mean, interpolator, coordinates, point, element_wise=True):
        # Find the Karcher mean.
        ref_psi = karcher_mean.compute_mean(self.psi)
        ref_phi = karcher_mean.compute_mean(self.phi)

        # Reshape the vector containing the singular values as a diagonal matrix.
        sigma_m = []
        for i in range(len(self.sigma)):
            sigma_m.append(np.diag(self.sigma[i]))

        # Project the points on the manifold to the tangent space created over the Karcher mean.
        gamma_psi = Grassmann.log_map(points_grassmann=self.psi, reference_point=ref_psi)
        gamma_phi = Grassmann.log_map(points_grassmann=self.phi, reference_point=ref_phi)

        # Perform the interpolation in the tangent space.
        interp_psi = SvdProjection \
            ._interpolate_sample(interpolator=interpolator, coordinates=coordinates, samples=gamma_psi,
                                 point=point, element_wise=element_wise)
        interp_phi = SvdProjection \
            ._interpolate_sample(interpolator=interpolator, coordinates=coordinates, samples=gamma_phi,
                                 point=point, element_wise=element_wise)
        interp_sigma = SvdProjection \
            ._interpolate_sample(interpolator=interpolator, coordinates=coordinates, samples=sigma_m,
                                 point=point, element_wise=element_wise)

        # Map the interpolated point back to the manifold.
        psi_tilde = Grassmann.exp_map(points_tangent=[interp_psi], reference_point=ref_psi)
        phi_tilde = Grassmann.exp_map(points_tangent=[interp_phi], reference_point=ref_phi)

        # Estimate the interpolated solution.
        psi_tilde = np.array(psi_tilde[0])
        phi_tilde = np.array(phi_tilde[0])
        interpolated = np.dot(np.dot(psi_tilde, interp_sigma), phi_tilde.T)

        return interpolated

    @staticmethod
    def _interpolate_sample(interpolator, coordinates, samples, point, element_wise=True):
        if isinstance(samples, list):
            samples = np.array(samples)

        # Test if the sample is stored as a list
        if isinstance(point, list):
            point = np.array(point)

        # Test if the nodes are stored as a list
        if isinstance(coordinates, list):
            coordinates = np.array(coordinates)

        nargs = len(samples)

        if interpolator is None:
            raise TypeError('UQpy: `interp_object` cannot be NoneType')
        else:
            if interpolator is LinearInterpolation:
                element_wise = False

            if isinstance(interpolator, Surrogate):
                element_wise = True

        shape_ref = np.shape(samples[0])
        for i in range(1, nargs):
            if np.shape(samples[i]) != shape_ref:
                raise TypeError('UQpy: Input matrices have different shape.')

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
                        # self.skl_str = "<class 'sklearn.gaussian_process.gpr.GaussianProcessRegressor'>"
                        if isinstance(interpolator, Surrogate):
                            interpolator.fit(coordinates, val_data)
                            y = interpolator.predict(point, return_std=False)
                        else:
                            y = interpolator.interpolate(coordinates, samples, point)

                    interp_point[j, k] = y

        else:
            if isinstance(interpolator, Surrogate):
                raise TypeError('UQpy: Kriging only can be used in the elementwise interpolation.')
            else:
                interp_point = interpolator.interpolate(coordinates, samples, point)

        return interp_point

    def evaluate_kernel_matrix(self, kernel: Kernel):
        kernel_psi = self.__estimate_kernel(self.psi, p_dim=self.p_planes_dimensions, kernel=kernel)
        kernel_phi = self.__estimate_kernel(self.phi, p_dim=self.p_planes_dimensions, kernel=kernel)
        return CompositionAction[self.kernel_composition.name](kernel_psi, kernel_phi)

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
