import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.spatial.distance as sd
import scipy
from UQpy.utilities.Utilities import *
from UQpy.utilities.Utilities import _nn_coord
from beartype import beartype
from typing import Annotated, Union
from beartype.vale import Is
from UQpy.utilities.ValidationTypes import Numpy2DFloatArray, NumpyFloatArray
from UQpy.dimension_reduction.kernels.GaussianKernel import GaussianKernel


class DiffusionMaps:

    AlphaType = Annotated[Union[float, int], Is[lambda number: 0 <= number <= 1]]
    IntegerLargerThanUnityType = Annotated[int, Is[lambda number: number >= 1]]

    @beartype
    def __init__(
        self,
        kernel_matrix: Numpy2DFloatArray,
        alpha: AlphaType = 0.5,
        eigenvectors_number: IntegerLargerThanUnityType = 2,
        is_sparse: bool = False,
        neighbors_number: IntegerLargerThanUnityType = 1,
        random_state: Union[None, int] = None,
        t: int = 0
    ):
        """

        :param alpha: Corresponds to different diffusion operators. It should be between zero and one.
        :param eigenvectors_number: Number of eigenvectors to keep.
        :param is_sparse: Work with sparse matrices. Increase the computational performance.
        :param neighbors_number: If :code:`distance_matrix is True` defines the number of nearest neighbors.
        :param kernel_matrix: kernel matrix defining the similarity between the points.
        :param random_state: sets :code:`np.random.default_rng(random_state)`.
        :param t: Time exponent.
        """
        self.alpha = alpha
        self.eigenvectors_number = eigenvectors_number
        self.is_sparse = is_sparse
        self.neighbors_number = neighbors_number
        self.kernel_matrix = kernel_matrix
        self.random_state = random_state,
        self.t = t

        self.transition_matrix = None
        self.diffusion_coordinates = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.cut_off = None

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
        t: int = 1,
        cut_off: float = None,
        k_nn: int = 10,
        n_partition: Union[None, int] = None,
        distance_matrix: Union[None, Numpy2DFloatArray] = None,
        random_state: Union[None, int] = None,
        tol: float = 1e-8,
        kernel=GaussianKernel(),
    ):

        """

        :param data: Cloud of data points.
        :param alpha: Corresponds to different diffusion operators. It should be between zero and one.
        :param eigenvectors_number: Number of eigenvectors to keep.
        :param is_sparse: Work with sparse matrices. Increase the computational performance.
        :param neighbors_number: If :code:`distance_matrix is True` defines the number of nearest neighbors.
        :param optimize_parameters: Estimate the kernel scale from the data.
        :param t: Time exponent.
        :param cut_off: Cut-off for a Gaussian kernel, below which the kernel values are considered zero.
        :param k_nn: k-th nearest neighbor distance to estimate the cut-off distance.
        :param n_partition: Maximum subsample used for the estimation. Ignored if :code:`distance_matrix is not None`.
        :param distance_matrix:  Pre-computed distance matrix.
        :param random_state: sets :code:`np.random.default_rng(random_state)`.
        :param tol: Tolerance where the cut_off should be made.
        :param kernel: kernel matrix defining the similarity between the points.

        """

        if optimize_parameters:
            epsilon, cut_off = DiffusionMaps.estimate_epsilon(data, cut_off=cut_off, tol=tol,
                                                              k_nn=k_nn, n_partition=n_partition,
                                                              distance_matrix=distance_matrix,
                                                              random_state=random_state)
            kernel.epsilon = epsilon

        kernel_matrix = kernel.kernel_operator(points=data)

        return cls(
            alpha=alpha,
            eigenvectors_number=eigenvectors_number,
            is_sparse=is_sparse,
            neighbors_number=neighbors_number,
            kernel_matrix=kernel_matrix,
            random_state=random_state,
            t=t
        )

    def fit(self) -> tuple[NumpyFloatArray, NumpyFloatArray, NumpyFloatArray]:
        """
        Perform diffusion map embedding.

        :returns: dmaps_embedding, eigenvalues, eigenvectors

        """

        if self.is_sparse:
            self.kernel_matrix = self.__sparse_kernel(self.kernel_matrix, self.neighbors_number)

        # Compute the diagonal matrix D(i,i) = sum(Kernel(i,j)^alpha,j) and its inverse.
        d, d_inv = self.__diagonal_matrix(self.kernel_matrix, self.alpha)

        # Compute L^alpha = D^(-alpha)*L*D^(-alpha).
        l_star = self.__normalize_kernel_matrix(self.kernel_matrix, d_inv)

        d_star, d_star_inv = self.__diagonal_matrix(l_star, 1.0)
        if self.is_sparse:
            d_star_inv_diag = sps.spdiags(
                d_star_inv, 0, d_star_inv.shape[0], d_star_inv.shape[0]
            )
        else:
            d_star_inv_diag = np.diag(d_star_inv)

        transition_matrix = d_star_inv_diag.dot(l_star)

        # Find the eigenvalues and eigenvectors of Ps.
        if self.is_sparse:
            eigenvalues, eigenvectors = spsl.eigs(
                transition_matrix, k=(self.eigenvectors_number + 1), which="LR")
        else:
            eigenvalues, eigenvectors = np.linalg.eig(transition_matrix)

        ix = np.argsort(np.abs(eigenvalues))
        ix = ix[::-1]
        s = np.real(eigenvalues[ix])
        u = np.real(eigenvectors[:, ix])

        eigenvalues = s[:self.eigenvectors_number]
        eigenvectors = u[:, :self.eigenvectors_number]

        # Compute the diffusion coordinates
        eig_values_time = np.power(eigenvalues, self.t)
        dmaps_embedding = eigenvectors @ np.diag(eig_values_time)

        self.transition_matrix = transition_matrix

        return dmaps_embedding, eigenvalues, eigenvectors

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

        m = inverse_diagonal_matrix.shape[0]
        if self.is_sparse:
            d_alpha = sps.spdiags(inverse_diagonal_matrix, 0, m, m)
        else:
            d_alpha = np.diag(inverse_diagonal_matrix)

        normalized_kernel = d_alpha.dot(kernel_matrix.dot(d_alpha))

        return normalized_kernel

    @staticmethod
    def parsimonious(eigenvectors: Numpy2DFloatArray, dim: int) -> tuple[list, NumpyFloatArray]:
        """
        Selection of independent vectors for parsimonious data manifold embedding, based on
        local regression.  The eigenvectors with the largest residuals are considered for the
        embedding. The scale of the kernel used for the local linear regression is:

        .. code::

            scale = median(distances) / 3

        :param eigenvectors: Eigenvectors of the diffusion maps embedding.
        :param dim: Number of eigenvectors to select with largest residuals.
        :returns: indices, residuals

        References
        ----------

        :cite:`dsilva_parsimonious_2018`

        """

        residuals = np.zeros(eigenvectors.shape[1])
        residuals[0] = np.nan
        # residual 1 for the first eigenvector.
        residuals[1] = 1.0

        # Get the residuals of each eigenvector.
        for i in range(2, eigenvectors.shape[1]):
            residuals[i] = DiffusionMaps.__get_residual(f_mat=eigenvectors[:, 1:i], f=eigenvectors[:, i])

        # Get the index of the eigenvalues associated with each residual.
        indices = np.argsort(residuals)[::-1][1:dim+1]
        return indices, residuals

    @staticmethod
    def __get_residual(f_mat, f):
        n_samples = np.shape(f_mat)[0]
        distance_matrix = sd.squareform(sd.pdist(f_mat))
        m = 3
        epsilon = (np.median(np.square(distance_matrix.flatten()))/m)
        kernel_matrix = np.exp(-1 * np.square(distance_matrix) / epsilon)
        coefficients = np.zeros((n_samples, n_samples))

        vec_1 = np.ones((n_samples, 1))

        for i in range(n_samples):
            # Weighted least squares:
            mat_x = np.hstack([vec_1, f_mat - f_mat[i, :]])
            mat_x_k = mat_x.T * kernel_matrix[i, :]
            u, _, _, _ = np.linalg.lstsq((mat_x_k @ mat_x), mat_x_k, rcond=1e-6)
            coefficients[i, :] = u[0, :]

        estimated_f = coefficients @ f

        # normalized leave-one-out cross-validation error.
        residual = np.sqrt(np.sum(np.square((f - estimated_f))) / np.sum(np.square(f)))
        return residual

    @staticmethod
    def estimate_cut_off(data, k_nn: int = 20, n_partition: Union[None, int] = None,
                         distance_matrix: Union[None, Numpy2DFloatArray] = None,
                         random_state: Union[None, int] = None) -> float:
        """
        Estimates the cut-off for a Gaussian kernel, given a tolerance below which the kernel values are
        considered zero.

        :param data: Cloud of data points.
        :param k_nn: k-th nearest neighbor distance to estimate the cut-off distance.
        :param n_partition: maximum subsample used for the estimation. Ignored if :code:`distance_matrix is not None`.
        :param distance_matrix:  Pre-computed distance matrix.
        :param random_state: sets :code:`np.random.default_rng(random_state)`.
        :return:

        """

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
    def estimate_epsilon(data, tol=1e-8, cut_off: float = None, **estimate_cutoff_params) -> float:
        """
        Estimates the scale paramter for a Gaussian kernel, given a tolerance below which the kernel values are
        considered zero.

        :param data: Cloud of data points.
        :param tol: Tolerance where the cut_off should be made.
        :param cut_off: User-defined cut-off.
        :param estimate_cutoff_params: Parameters to handle to method :py:meth:`estimate_cutoff` if ``cut_off is None``.
        :return:

        See Also
        --------

        :py:class:`UQpy.dimension_reduction.kernels.GaussianKernel`

        """

        if cut_off is None:
            cut_off = DiffusionMaps.estimate_cut_off(data,  **estimate_cutoff_params)

        # tol >= exp(-cut_off**2 / epsilon)
        eps0 = cut_off ** 2 / (-np.log(tol))
        return float(eps0), cut_off
