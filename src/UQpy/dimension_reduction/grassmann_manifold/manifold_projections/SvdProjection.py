import sys
from UQpy.dimension_reduction.grassmann_manifold.manifold_projections.KernelComposition import (
    KernelComposition,
    CompositionAction,
)
from UQpy.dimension_reduction.grassmann_manifold.manifold_projections.baseclass.ManifoldProjection import (
    ManifoldProjection,
)
from UQpy.dimension_reduction.kernels.baseclass.Kernel import Kernel
from UQpy.utilities.ValidationTypes import Numpy2DFloatArray
from UQpy.utilities.Utilities import *
from UQpy.dimension_reduction.grassmann_manifold.Grassmann import Grassmann


class SvdProjection(ManifoldProjection):
    def __init__(
        self,
        data: list[Numpy2DFloatArray],
        p: int,
        kernel_composition: KernelComposition = KernelComposition.LEFT,
    ):
        """
        :param data: Raw data given as a list of 2-d arrays.
        :param p: Number of independent p-planes of each Grassmann point.
        :param kernel_composition: Composition of the kernel.
        """
        self.kernel_composition = kernel_composition
        self.data = data

        points_number = len(data)

        n_left = []
        n_right = []
        for i in range(points_number):
            n_left.append(max(np.shape(data[i])))
            n_right.append(min(np.shape(data[i])))

        bool_left = n_left.count(n_left[0]) != len(n_left)
        bool_right = n_right.count(n_right[0]) != len(n_right)

        if bool_left and bool_right:
            raise TypeError("UQpy: The shape of the input matrices must be the same.")
        else:
            n_psi = n_left[0]
            n_phi = n_right[0]

        ranks = []
        for i in range(points_number):
            ranks.append(np.linalg.matrix_rank(data[i]))

        if p == 0:
            p_planes_dimensions = int(min(ranks))
        elif p == sys.maxsize:
            p_planes_dimensions = int(max(ranks))
        else:
            for i in range(points_number):
                if min(np.shape(data[i])) < p:
                    raise ValueError(
                        "UQpy: The dimension of the input data is not consistent with `p` of G(n,p)."
                    )  # write something that makes sense

        ranks = np.ones(points_number) * [int(p)]
        ranks = ranks.tolist()

        ranks = list(map(int, ranks))

        psi = []  # initialize the left singular eigenvectors as a list.
        sigma = []  # initialize the singular values as a list.
        phi = []  # initialize the right singular eigenvectors as a list.
        for i in range(points_number):
            u, s, v = svd(data[i], int(ranks[i]))
            psi.append(u)
            sigma.append(np.diag(s))
            phi.append(v)

        self.input_points = data
        self.psi = psi
        self.sigma = sigma
        self.phi = phi

        self.n_psi = n_psi
        self.n_phi = n_phi
        self.p_planes_dimensions = p
        self.ranks = ranks
        self.points_number = points_number
        self.max_rank = int(np.max(ranks))

    def reconstruct_solution(
        self,
        interpolation,
        coordinates,
        point,
        p_planes_dimensions,
        optimization_method,
        distance,
        element_wise=True,
    ):
        # Find the Karcher mean.

        ref_psi = Grassmann.karcher_mean(
            manifold_points=self.psi,
            optimization_method=optimization_method,
            distance=distance,
        )

        ref_phi = Grassmann.karcher_mean(
            manifold_points=self.phi,
            optimization_method=optimization_method,
            distance=distance,
        )

        # Reshape the vector containing the singular values as a diagonal matrix.
        sigma_m = []
        for i in range(len(self.sigma)):
            sigma_m.append(np.diag(self.sigma[i]))

        # Project the points on the manifold to the tangent space created over the Karcher mean.
        gamma_psi = Grassmann.log_map(
            manifold_points=self.psi, reference_point=ref_psi
        )
        gamma_phi = Grassmann.log_map(
            manifold_points=self.phi, reference_point=ref_phi
        )

        # Perform the interpolation in the tangent space.
        interp_psi = interpolation.interpolate_sample(
            coordinates=coordinates,
            samples=gamma_psi,
            point=point,
            element_wise=element_wise,
        )
        interp_phi = interpolation.interpolate_sample(
            coordinates=coordinates,
            samples=gamma_phi,
            point=point,
            element_wise=element_wise,
        )
        interp_sigma = interpolation.interpolate_sample(
            coordinates=coordinates,
            samples=sigma_m,
            point=point,
            element_wise=element_wise,
        )

        # Map the interpolated point back to the manifold.
        psi_tilde = Grassmann.exp_map(
            tangent_points=[interp_psi], reference_point=ref_psi
        )
        phi_tilde = Grassmann.exp_map(
            tangent_points=[interp_phi], reference_point=ref_phi
        )

        # Estimate the interpolated solution.
        psi_tilde = np.array(psi_tilde[0])
        phi_tilde = np.array(phi_tilde[0])
        interpolated = np.dot(np.dot(psi_tilde, interp_sigma), phi_tilde.T)

        return interpolated

    def evaluate_matrix(self, kernel: Kernel):
        kernel_psi = kernel.kernel_operator(self.psi, p=self.p_planes_dimensions)
        kernel_phi = kernel.kernel_operator(self.phi, p=self.p_planes_dimensions)
        return CompositionAction[self.kernel_composition.name](kernel_psi, kernel_phi)
