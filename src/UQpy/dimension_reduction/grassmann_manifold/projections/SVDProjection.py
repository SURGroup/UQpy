from typing import Union

from beartype import beartype

from UQpy.utilities.GrassmannPoint import GrassmannPoint
from UQpy.dimension_reduction.grassmann_manifold.projections.baseclass.GrassmannProjection import GrassmannProjection
from UQpy.utilities.ValidationTypes import Numpy2DFloatArray
from UQpy.utilities.Utilities import *


class SVDProjection(GrassmannProjection):
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
            Options:

            :any:`int`: Integer specifying the number of p-planes

            :any:`str`:
             `"max"`: Set p equal to the maximum rank of all provided data matrices

             `"min"`: Set p equal to the minimum rank of all provided data matrices
        :param tol: Tolerance on the SVD
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
        n_u = n_left[0]
        n_v = n_right[0]

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

        phi = []  # initialize the left singular eigenvectors as a list.
        sigma = []  # initialize the singular values as a list.
        psi = []  # initialize the right singular eigenvectors as a list.
        for i in range(points_number):
            u, s, v = svd(data[i], int(ranks[i]))
            phi.append(GrassmannPoint(u))
            sigma.append(np.diag(s))
            psi.append(GrassmannPoint(v))

        self.input_points = data
        self.u: list[GrassmannPoint] = phi
        """Left singular vectors from the SVD of each sample in `data` representing a point on the Grassmann 
        manifold. """
        self.sigma: np.ndarray = sigma
        """Singular values from the SVD of each sample in `data`."""
        self.v: list[GrassmannPoint] = psi
        """Right singular vectors from the SVD of each sample in `data` representing a point on the Grassmann 
        manifold."""

        self.n_u = n_u
        self.n_v = n_v
        self.p = p
        self.ranks = ranks
        self.points_number = points_number
        self.max_rank = int(np.max(ranks))
