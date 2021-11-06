import sys

from beartype import beartype

from UQpy.dimension_reduction.grassmann_manifold.GrassmannPoint import GrassmannPoint
from UQpy.dimension_reduction.grassmann_manifold.projection.KernelComposition import (
    KernelComposition,
    CompositionAction,
)
from UQpy.dimension_reduction.grassmann_manifold.projection.baseclass.ManifoldProjection import (
    ManifoldProjection,
)
from UQpy.dimension_reduction.kernels.baseclass.Kernel import Kernel
from UQpy.utilities.ValidationTypes import Numpy2DFloatArray
from UQpy.utilities.Utilities import *


class SvdProjection(ManifoldProjection):
    @beartype
    def __init__(
        self,
        data: list[Numpy2DFloatArray],
        p: int,
        kernel_composition: KernelComposition = KernelComposition.LEFT,
    ):
        """

        :param data: Raw data given as a list of matrices.
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
            p = int(min(ranks))
        elif p == sys.maxsize:
            p = int(max(ranks))
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
            psi.append(GrassmannPoint(u))
            sigma.append(np.diag(s))
            phi.append(GrassmannPoint(v))

        self.input_points = data
        self.psi = psi
        self.sigma = sigma
        self.phi = phi

        self.n_psi = n_psi
        self.n_phi = n_phi
        self.p = p
        self.ranks = ranks
        self.points_number = points_number
        self.max_rank = int(np.max(ranks))

    @beartype
    def evaluate_matrix(self, kernel: Kernel):
        kernel_psi = kernel.kernel_operator(self.psi, p=self.p)
        kernel_phi = kernel.kernel_operator(self.phi, p=self.p)
        return CompositionAction[self.kernel_composition.name](kernel_psi, kernel_phi)
