import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.spatial.distance as sd
import scipy
from UQpy.utilities.Utilities import *
from UQpy.utilities.Utilities import _nn_coord
from beartype import beartype
from typing import Annotated, Union
from beartype.vale import Is
from UQpy.utilities import Numpy2DFloatArray
from UQpy.dimension_reduction.kernels.GaussianKernel import GaussianKernel


class DiffusionMaps:

    AlphaType = Annotated[Union[float, int], Is[lambda number: 0 <= number <= 1]]
    IntegerLargerThanUnityType = Annotated[int, Is[lambda number: number >= 1]]

    @beartype
    def __init__(
        self,
        alpha: AlphaType = 0.5,
        eigenvectors_number: IntegerLargerThanUnityType = 2,
        is_sparse: bool = False,
        neighbors_number: IntegerLargerThanUnityType = 1,
        kernel_matrix=None,
        parsimonious: bool = False,
        random_state: Union[None, int] = None,
        t: int = 1
    ):
        self.alpha = alpha
        self.eigenvectors_number = eigenvectors_number
        self.is_sparse = is_sparse
        self.neighbors_number = neighbors_number
        self.kernel_matrix = kernel_matrix
        self.parsimonious = parsimonious
        self.random_state = random_state,
        self.t = t

        self.transition_matrix = None
        self.diffusion_coordinates = None
        self.eigenvectors = None
        self.eigenvalues = None

        if kernel_matrix is not None:
            self.kernel_matrix = kernel_matrix

    @classmethod
    def create_from_data(
        cls,
        data: Numpy2DFloatArray,
        alpha: AlphaType = 0.5,
        eigenvectors_number: IntegerLargerThanUnityType = 2,
        is_sparse: bool = False,
        neighbors_number: IntegerLargerThanUnityType = 1,
        optimize_parameters: bool = False,
        parsimonious: bool = False,
        t: int = 1,
        cut_off: float = None,
        k_nn: int = 10,
        n_partition: Union[None, int] = None,
        distance_matrix: Union[None, Numpy2DFloatArray] = None,
        random_state: Union[None, int] = None,
        tol: float = 1e-8,
        kernel=GaussianKernel(),
    ):
        kernel_matrix = kernel.kernel_operator(points=data)
        if optimize_parameters:
            epsilon, cut_off = DiffusionMaps.__estimate_epsilon(data, cut_off=cut_off, tol=tol,
                                                                k_nn=k_nn, n_partition=n_partition,
                                                                distance_matrix=distance_matrix,
                                                                random_state=random_state)
            kernel.epsilon = epsilon

        return cls(
            alpha=alpha,
            eigenvectors_number=eigenvectors_number,
            is_sparse=is_sparse,
            neighbors_number=neighbors_number,
            kernel_matrix=kernel_matrix,
            parsimonious=parsimonious,
            random_state=random_state,
            t=t
        )

    def mapping(self):

        alpha = self.alpha
        eigenvectors_number = self.eigenvectors_number
        sparse = self.is_sparse
        k_neighbors = self.neighbors_number

        n = np.shape(self.kernel_matrix)[0]
        if sparse:
            self.kernel_matrix = self.__sparse_kernel(self.kernel_matrix, k_neighbors)

        # Compute the diagonal matrix D(i,i) = sum(Kernel(i,j)^alpha,j) and its inverse.
        d, d_inv = self.__diagonal_matrix(self.kernel_matrix, alpha)

        # Compute L^alpha = D^(-alpha)*L*D^(-alpha).
        l_star = self.__normalize_kernel_matrix(self.kernel_matrix, d_inv)

        d_star, d_star_inv = self.__diagonal_matrix(l_star, 1.0)
        if sparse:
            d_star_invd = sps.spdiags(
                d_star_inv, 0, d_star_inv.shape[0], d_star_inv.shape[0]
            )
        else:
            d_star_invd = np.diag(d_star_inv)

        transition_matrix = d_star_invd.dot(l_star)

        # Find the eigenvalues and eigenvectors of Ps.
        if sparse:
            eigenvalues, eigenvectors = spsl.eigs(
                transition_matrix, k=(eigenvectors_number + 1), which="LR"
            )
        else:
            eigenvalues, eigenvectors = np.linalg.eig(transition_matrix)

        ix = np.argsort(np.abs(eigenvalues))
        ix = ix[::-1]
        s = np.real(eigenvalues[ix])
        u = np.real(eigenvectors[:, ix])

        if self.parsimonious:
            index, residuals = self.__parsimonious(s, u)
            coord = index + 1
            print(coord)
            eigenvalues = s[coord]
            eigenvectors = u[:, coord]
        else:
            # Truncated eigenvalues and eigenvectors
            eigenvalues = s[:eigenvectors_number]
            eigenvectors = u[:, :eigenvectors_number]

        # Compute the diffusion coordinates
        diffusion_coordinates = np.zeros([n, eigenvectors_number])
        for i in range(eigenvectors_number):
            diffusion_coordinates[:, i] = (eigenvalues[i] ** self.t) * eigenvectors[:, i]

        self.transition_matrix = transition_matrix
        self.diffusion_coordinates = diffusion_coordinates
        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues

        return diffusion_coordinates, eigenvalues, eigenvectors

    # Private method
    @staticmethod
    def __sparse_kernel(kernel_matrix, neighbors_number):
        rows = np.shape(kernel_matrix)[0]
        for i in range(rows):
            row_data = kernel_matrix[i, :]
            index = _nn_coord(row_data, neighbors_number)
            kernel_matrix[i, index] = 0
            if sum(kernel_matrix[i, :]) <= 0:
                raise ValueError(
                    "UQpy: Consider increasing `neighbors_number` to have a connected graph."
                )

        sparse_kernel_matrix = sps.csc_matrix(kernel_matrix)

        return sparse_kernel_matrix

    # Private method
    @staticmethod
    def __diagonal_matrix(kernel_matrix, alpha):

        diagonal_matrix = np.array(kernel_matrix.sum(axis=1)).flatten()
        inverse_diagonal_matrix = np.power(diagonal_matrix, -alpha)

        return diagonal_matrix, inverse_diagonal_matrix

    def __normalize_kernel_matrix(self, kernel_matrix, inverse_diagonal_matrix):

        rows = inverse_diagonal_matrix.shape[0]
        d_alpha = (
            sps.spdiags(inverse_diagonal_matrix, 0, rows, rows)
            if self.is_sparse
            else np.diag(inverse_diagonal_matrix)
        )

        normalized_kernel = d_alpha.dot(kernel_matrix.dot(d_alpha))

        return normalized_kernel

    def __parsimonious(self, eigenvalues, eigenvectors):

        residuals = np.zeros(self.eigenvectors_number)
        residuals[0] = np.nan
        # residual 1 for the first eigenvector.
        residuals[1] = 1.0

        # Get the residuals of each eigenvector.
        for i in range(2, self.eigenvectors_number):
            residuals[i] = self.__get_residual(f_mat=eigenvectors[:, 1:i], f=eigenvectors[:, i])

        # Get the index of the eigenvalues associated with each residual.
        index = np.argsort(residuals)[::-1][:len(eigenvalues)]
        return index, residuals

    @staticmethod
    def __get_residual(f_mat, f):
        n_samples = np.shape(f_mat)[0]
        distance_matrix = sd.squareform(sd.pdist(f_mat))
        m = 3
        epsilon = (np.median(distance_matrix.flatten()) / m) ** 2
        kernel_matrix = np.exp(-np.square(distance_matrix) / epsilon)
        coefficients = np.zeros((n_samples, n_samples))

        vec_1 = np.ones((n_samples, 1))

        for i in range(n_samples):
            # Weighted least squares:
            matx = np.hstack([vec_1, f_mat - f_mat[i, :]])
            matx_k = matx.T * kernel_matrix[i, :]
            w_data = matx_k.dot(matx)
            u, _, _, _ = np.linalg.lstsq(w_data, matx_k, rcond=1e-6)

            coefficients[i, :] = u[0, :]

        estimated_f = coefficients.dot(f)

        # normalized leave-one-out cross-validation error.
        residual = np.sqrt(np.sum(np.square((f - estimated_f))) / np.sum(np.square(f)))

        return residual

    @staticmethod
    def __estimate_cutoff(data, k_nn: int = 10, n_partition: Union[None, int] = None,
                          distance_matrix: Union[None, Numpy2DFloatArray] = None,
                          random_state: Union[None, int] = None) -> float:
        n_points = data.shape[0]
        if n_points < 10:
            d = scipy.spatial.distance.pdist(data)
            return np.max(d)

        if distance_matrix is None:
            if n_partition is not None:
                random_indices = np.random.default_rng(random_state).permutation(n_points)
                distance_matrix = sd.cdist(data[random_indices[:n_partition]], data,  metric='euclidean')
                k = np.min([k_nn, distance_matrix.shape[1]])
                k_smallest_values = np.partition(distance_matrix, k - 1, axis=1)[:, k - 1]
            else:
                distance_matrix = sd.squareform(sd.pdist(data, metric='euclidean'))
                k = np.min([k_nn, distance_matrix.shape[1]])
                k_smallest_values = np.partition(distance_matrix, k - 1, axis=1)[:, k - 1]
        else:
            k_smallest_values = np.partition(distance_matrix, k_nn - 1, axis=1)[:, k_nn - 1]
        est_cutoff = np.max(k_smallest_values)
        return float(est_cutoff)

    @staticmethod
    def __estimate_epsilon(data, tol=1e-8, cut_off: float = None, **estimate_cutoff_params) -> float:

        if cut_off is None:
            cut_off = DiffusionMaps.__estimate_cutoff(data,  **estimate_cutoff_params)

        # tol >= exp(-cut_off**2 / epsilon)
        eps0 = cut_off ** 2 / (-np.log(tol))
        return float(eps0), cut_off
