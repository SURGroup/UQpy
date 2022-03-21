from typing import Union

from beartype import beartype

from UQpy.utilities.GrassmannPoint import GrassmannPoint
from UQpy.dimension_reduction.grassmann_manifold.projections.baseclass.ManifoldProjection import ManifoldProjection
from UQpy.utilities.ValidationTypes import Numpy2DFloatArray
from UQpy.utilities.Utilities import *


class SvdProjection(ManifoldProjection):
    @beartype
    def __init__(
            self,
            data: list[Numpy2DFloatArray],
            p: Union[int, str],
            tol: float = None,
    ):
        """

        :param data: Raw data given as a list of matrices.
        :param p: Number of independent p-planes of each Grassmann point.
        :param tol: Tolerance on the SVD decomposition
        """
        self.data = data
        self.tolerance = tol

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
        n_psi = n_left[0]
        n_phi = n_right[0]

        ranks = [np.linalg.matrix_rank(data[i], tol=self.tolerance) for i in range(points_number)]

        if isinstance(p, str) and p == "min":
            p = int(min(ranks))
        elif isinstance(p, str) and p == "max":
            p = int(max(ranks))
        elif isinstance(p, str):
            raise ValueError("The input parameter p must me either 'min', 'max' or a integer.")
        else:
            for i in range(points_number):
                if min(np.shape(data[i])) < p:
                    raise ValueError("UQpy: The dimension of the input data is not consistent with `p` of G(n,p).")
                    # write something that makes sense

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
