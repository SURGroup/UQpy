import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import matplotlib.pyplot as plt
import scipy.spatial.distance as sd
from matplotlib.cm import ScalarMappable
from typing import Optional
import scipy
from UQpy.utilities.Utilities import *
from UQpy.utilities.Utilities import _nn_coord
from beartype import beartype
from typing import Annotated, Union
from beartype.vale import Is
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
    ):

        self.alpha = alpha
        self.eigenvectors_number = eigenvectors_number
        self.is_sparse = is_sparse
        self.neighbors_number = neighbors_number
        self.kernel_matrix = kernel_matrix

        self.transition_matrix = None
        self.diffusion_coordinates = None
        self.eigenvectors = None
        self.eigenvalues = None

        if kernel_matrix is not None:
            self.kernel_matrix = kernel_matrix

    @classmethod
    def create_from_data(
        cls,
        data,
        alpha: AlphaType = 0.5,
        eigenvectors_number: IntegerLargerThanUnityType = 2,
        is_sparse: bool = False,
        neighbors_number: IntegerLargerThanUnityType = 1,
        kernel=GaussianKernel(),
    ):
        kernel_matrix = kernel.kernel_operator(points=data)
        return cls(
            alpha=alpha,
            eigenvectors_number=eigenvectors_number,
            is_sparse=is_sparse,
            neighbors_number=neighbors_number,
            kernel_matrix=kernel_matrix,
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

        # Truncated eigenvalues and eigenvectors
        eigenvalues = s[:eigenvectors_number]
        eigenvectors = u[:, :eigenvectors_number]

        # Compute the diffusion coordinates
        diffusion_coordinates = np.zeros([n, eigenvectors_number])
        for i in range(eigenvectors_number):
            diffusion_coordinates[:, i] = eigenvalues[i] * eigenvectors[:, i]

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

    def parsimonious(self, num_eigenvectors: int, visualization=False):

        if num_eigenvectors is None:
            num_eigenvectors = self.eigenvectors_number
        elif num_eigenvectors > self.eigenvectors_number:
            raise ValueError('UQpy: num_eigenvectors cannot be larger than n_evecs.')

        eig_vec = np.asarray(self.eigenvectors)
        eig_vec = eig_vec[:, 0:num_eigenvectors]

        residuals = np.zeros(num_eigenvectors)
        residuals[0] = np.nan
        # residual 1 for the first eigenvector.
        residuals[1] = 1.0

        # Get the residuals of each eigenvector.
        for i in range(2, num_eigenvectors):
            residuals[i] = self._get_residual(f_mat=eig_vec[:, 1:i], f=eig_vec[:, i])

        # Get the index of the eigenvalues associated with each residual.
        index = np.argsort(residuals)[::-1][:len(self.eigenvalues)]

        # Plot the graphic
        if visualization:
            data_x = np.arange(1, len(residuals)).tolist()
            data_height = self.eigenvalues[1:num_eigenvectors]
            data_color = residuals[1:]

            data_color = [x / max(data_color) for x in data_color]

            fig, ax = plt.subplots(figsize=(15, 4))

            my_color_map = plt.cm.get_cmap('Purples')
            colors = my_color_map(data_color)
            _ = ax.bar(data_x, data_height, color=colors)

            sm = ScalarMappable(cmap=my_color_map, norm=plt.Normalize(0, max(data_color)))
            sm.set_array([])

            cbar = plt.colorbar(sm)
            cbar.set_label('Residual', rotation=270, labelpad=25)

            plt.xticks(data_x)
            plt.ylabel("Eigenvalue(k)")
            plt.xlabel("k")

            plt.show()

        return index, residuals

    @staticmethod
    def _get_residual(f_mat, f):

        n_samples = np.shape(f_mat)[0]
        distance_matrix = sd.squareform(sd.pdist(f_mat))
        # m=3 is suggested on Nadler et al. 2008.
        m = 3

        # Compute an appropriate value for epsilon.
        # epsilon = np.median(abs(np.square(distance_matrix.flatten())))/m
        epsilon = (np.median(distance_matrix.flatten()) / m) ** 2

        # Gaussian kernel. It is implemented here because of the factor m and the
        # shape of the argument of the exponential is the one suggested on
        # Nadler et al. 2008.
        kernel_matrix = np.exp(-np.square(distance_matrix) / epsilon)

        # Matrix to store the coefficients from the linear system.
        coefficients = np.zeros((n_samples, n_samples))

        vec_1 = np.ones((n_samples, 1))

        for i in range(n_samples):
            # Weighted least squares:
            matx = np.hstack([vec_1, f_mat - f_mat[i, :]])

            # matx.T*Kernel
            matx_k = matx.T * kernel_matrix[i, :]

            # matx.T*Kernel*matx
            w_data = matx_k.dot(matx)
            u, _, _, _ = np.linalg.lstsq(w_data, matx_k, rcond=1e-6)

            coefficients[i, :] = u[0, :]

        estimated_f = coefficients.dot(f)

        # normalized leave-one-out cross-validation error.
        residual = np.sqrt(np.sum(np.square((f - estimated_f))) / np.sum(np.square(f)))

        return residual

    @staticmethod
    def estimate_cutoff(
        data,
        n_subsample: int = 1000,
        k: int = 10,
        random_state: Optional[int] = None,
        distance_matrix=None,
    ) -> float:
        """Estimates a good choice of cut-off for a Gaussian radial basis kernel, given a
        certain tolerance below which the kernel values are considered zero.

        Parameters
        ----------
        pcm
            point cloud to compute pair-wise kernel matrix with

        n_subsample
            Maximum subsample used for the estimation. Ignored if :code:`distance_matrix is not
            None`.

        k
            Compute the `k`-th nearest neighbor distance to estimate the
            cut-off distance.

        random_state
            sets :code:`np.random.default_rng(random_state)`

        distance_matrix
            pre-computed distance matrix instead of using the internal `cdist` method

        See Also
        --------

        :py:class:`datafold.pcfold.kernels.GaussianKernel`

        """

        if k <= 1 and not isinstance(k, int):
            raise ValueError("Parameter 'k' must be an integer greater than 1.")
        else:
            k = int(k)

        n_points = data.shape[0]
        n_subsample = np.min([n_points, n_subsample])

        if n_points < 10:
            d = scipy.spatial.distance.pdist(data)
            return np.max(d)

        if distance_matrix is None:

            distance_matrix = sd.squareform(sd.pdist(data))
            k = np.min([k, distance_matrix.shape[1]])
            k_smallest_values = _kth_nearest_neighbor_dist(distance_matrix.T, k)
        else:
            k_smallest_values = _kth_nearest_neighbor_dist(distance_matrix, k)

        est_cutoff = np.max(k_smallest_values)
        return float(est_cutoff)

    @staticmethod
    def estimate_scale(
        data, tol=1e-8, cut_off: Optional[float] = None, **estimate_cutoff_params
    ) -> float:
        """Estimates the Gaussian kernel scale (epsilon) for a Gaussian kernel, given a
        certain tolerance below which the kernel values are considered zero.

        Parameters
        ----------
        pcm
            Point cloud to estimate the kernel scale with.

        tol
            Tolerance where the cut_off should be made.

        cut_off
            The `tol` parameter is ignored and the cut-off is used directly

        **estimate_cutoff_params
            Parameters to handle to method :py:meth:`estimate_cutoff` if ``cut_off is None``.
        """

        if cut_off is None:
            cut_off = DiffusionMaps.estimate_cutoff(data, **estimate_cutoff_params)

        # this formula is derived by solving for epsilon in
        # tol >= exp(-cut_off**2 / epsilon)
        eps0 = cut_off ** 2 / (-np.log(tol))
        return float(eps0)


def _kth_nearest_neighbor_dist(
    distance_matrix: np.ndarray, k
) -> np.ndarray:
    """Compute the distance to the `k`-th nearest neighbor.

    Parameters
    ----------
    distance_matrix
        Matrix of shape `(n_samples_Y, n_samples_X)` to partition to find the distance of
        the `k`-th nearest neighbor.

    k
        The distance of the `k`-th nearest neighbor is returned. The value must be a
        positive integer.

    Returns
    -------
    numpy.ndarray
        distance values
    """

    if not isinstance(k, int):
        raise ValueError(f"parameter 'k={k}' must be a positive integer")
    else:
        # make sure we deal with Python built-in
        k = int(k)

    if not (0 <= k <= distance_matrix.shape[1]):
        raise ValueError(
            "'k' must be an integer between 1 and "
            f"distance_matrix.shape[1]={distance_matrix.shape[1]}"
        )

    dist_knn = np.partition(distance_matrix, k - 1, axis=1)[:, k - 1]

    return dist_knn
