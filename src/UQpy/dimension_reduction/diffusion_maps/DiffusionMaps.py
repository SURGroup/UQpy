import itertools

import scipy.sparse as sps
import scipy as sp
from scipy.sparse.linalg import eigsh, eigs
import scipy.spatial.distance as sd
import matplotlib.pyplot as plt
import scipy
from itertools import combinations
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
        t: int = 1
    ):
        """

        :param alpha: Corresponds to different diffusion operators. It should be between 0 and 1.
        :param eigenvectors_number: Number of eigenvectors to keep.
        :param is_sparse: Work with sparse matrices. Increase the computational performance.
        :param neighbors_number: Defines the number of nearest neighbors.
        :param kernel_matrix: kernel matrix defining the similarity between the points.
        :param random_state: Random seed used to initialize the pseudo-random number generator. If an :any:`int` is
         provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise, the
         object itself can be passed directly.
        :param t: Time exponent.
        """
        self.alpha = alpha
        self.eigenvectors_number = eigenvectors_number
        self.is_sparse = is_sparse
        self.neighbors_number = neighbors_number
        self.kernel_matrix = kernel_matrix
        self.random_state = random_state,
        self.t = t

        self.transition_matrix: NumpyFloatArray = None
        """Transition kernel of a Markov chain on the data."""
        self.diffusion_coordinates: NumpyFloatArray = None
        """Diffusion coordinates generated after the mapping of the Diffusion maps"""
        self.eigenvectors: NumpyFloatArray = None
        """Eigenvectors of the transition kernel of a Markov chain on the data."""
        self.eigenvalues: NumpyFloatArray = None
        """Eigenvalues of the transition kernel of a Markov chain on the data."""
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
        Alternative way of generating a :class:`.DiffusionMaps` object in case of raw data.

        :param data: Cloud of data points.
        :param alpha: Corresponds to different diffusion operators. It should be between 0 and 1.
        :param eigenvectors_number: Number of eigenvectors to keep.
        :param is_sparse: Work with sparse matrices. Increase the computational performance.
        :param neighbors_number: Defines the number of nearest neighbors.
        :param optimize_parameters: Estimate the kernel scale from the data.
        :param t: Time exponent.
        :param cut_off: Cut-off for a Gaussian kernel, below which the kernel values are considered zero.
        :param k_nn: k-th nearest neighbor distance to estimate the cut-off distance.
        :param n_partition: Maximum subsample used for the estimation. Ignored if *distance_matrix* is not None.
        :param distance_matrix:  Pre-computed distance matrix.
        :param random_state: Random seed used to initialize the pseudo-random number generator. If an :any:`int` is
         provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise, the
         object itself can be passed directly.
        :param tol: Tolerance where the *cut_off* should be made.
        :param kernel: Kernel matrix defining the similarity between the points.


        See Also
        --------

        :py:class:`.GaussianKernel`

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

        :returns: diffusion_coordinates, eigenvalues, eigenvectors

        """

        if self.is_sparse:
            self.kernel_matrix = self.__sparse_kernel(self.kernel_matrix, self.neighbors_number)

        alpha = self.alpha
        # Compute the diagonal matrix D(i,i) = sum(Kernel(i,j)^alpha,j) and its inverse.
        # d, d_inv = self._d_matrix(self.kernel_matrix, self.alpha)
        d = np.array(self.kernel_matrix.sum(axis=1)).flatten()
        d_inv = np.power(d, -alpha)

        # Compute L^alpha = D^(-alpha)*L*D^(-alpha).
        m = d_inv.shape[0]
        if self.is_sparse:
            d_alpha = sps.spdiags(d_inv, 0, m, m)
        else:
            d_alpha = np.diag(d_inv)

        l_star = d_alpha.dot(self.kernel_matrix.dot(d_alpha))

        # d_star, d_star_inv = self._d_matrix(l_star, 1.0)
        d_star = np.array(l_star.sum(axis=1)).flatten()
        d_star_inv = np.power(d_star, -1)

        if self.is_sparse:
            d_star_inv_d = sps.spdiags(d_star_inv, 0, d_star_inv.shape[0], d_star_inv.shape[0])
        else:
            d_star_inv_d = np.diag(d_star_inv)

        # Compute the transition matrix.
        transition_matrix = d_star_inv_d.dot(l_star)

        if self.is_sparse:
            is_symmetric = sp.sparse.linalg.norm(transition_matrix - transition_matrix.T, sp.inf) < 1e-08
        else:
            is_symmetric = np.allclose(transition_matrix, transition_matrix.T, rtol=1e-5,  atol=1e-08)

        # Find the eigenvalues and eigenvectors of Ps.
        eigenvalues, eigenvectors = DiffusionMaps.eig_solver(transition_matrix, is_symmetric,
                                                             (self.eigenvectors_number + 1))

        ix = np.argsort(np.abs(eigenvalues))
        ix = ix[::-1]
        s = np.real(eigenvalues[ix])
        u = np.real(eigenvectors[:, ix])

        eigenvalues = s[:self.eigenvectors_number]
        eigenvectors = u[:, :self.eigenvectors_number]

        # Compute the diffusion coordinates
        eig_values_time = np.power(eigenvalues, self.t)
        diffusion_coordinates = eigenvectors @ np.diag(eig_values_time)

        self.transition_matrix = transition_matrix

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

    @staticmethod
    def diffusion_distance(diffusion_coordinates: Numpy2DFloatArray) -> Numpy2DFloatArray:
        distance_matrix = np.zeros((diffusion_coordinates.shape[0], diffusion_coordinates.shape[0]))
        pairs = list(itertools.combinations(diffusion_coordinates.shape[0], 2))
        for id_pair in range(np.shape(pairs)[0]):
            i = pairs[id_pair][0]
            j = pairs[id_pair][1]

            xi = diffusion_coordinates[i, :]
            xj = diffusion_coordinates[j, :]

            distance_matrix[i, j] = np.linalg.norm(xi - xj)
            distance_matrix[j, i] = distance_matrix[i, j]

        return distance_matrix

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
        :param n_partition: maximum subsample used for the estimation. Ignored if *distance_matrix* is not None.
        :param distance_matrix:  Pre-computed distance matrix.
        :param random_state: Random seed used to initialize the pseudo-random number generator. If an :any:`int` is
         provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise, the
         object itself can be passed directly.
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
    def estimate_epsilon(data, tol=1e-8, cut_off: float = None, **estimate_cutoff_params) -> tuple[float, float]:
        """
        Estimates the scale parameter for a Gaussian kernel, given a tolerance below which the kernel values are
        considered zero.

        ``scale = cut_off ** 2 / -log(tol)``

        :param data: Cloud of data points.
        :param tol: Tolerance where the *cut_off* should be made.
        :param cut_off: User-defined cut-off.
        :param estimate_cutoff_params: Parameters to handle to method :py:meth:`estimate_cutoff`
         if *cut_off* is None.
        """

        if cut_off is None:
            cut_off = DiffusionMaps.estimate_cut_off(data,  **estimate_cutoff_params)

        scale = cut_off ** 2 / (-np.log(tol))
        return scale, cut_off

    @staticmethod
    def eig_solver(kernel_matrix: Numpy2DFloatArray, is_symmetric: bool, n_eigenvectors: int) -> \
            tuple[NumpyFloatArray, Numpy2DFloatArray]:

        n_samples, n_features = kernel_matrix.shape

        if n_eigenvectors == n_features:
            if is_symmetric:
                solver = sp.linalg.eigh
            else:
                solver = sp.linalg.eig

            solver_kwargs = {"check_finite": False}

        else:
            if is_symmetric:
                solver = eigsh
            else:
                solver = eigs

            solver_kwargs = {
                "sigma": None,
                "k": n_eigenvectors,
                "which": "LM",
                "v0": np.ones(n_samples),
                "tol": 1e-14,
            }

        eigenvalues, eigenvectors = solver(kernel_matrix, **solver_kwargs)

        eigenvectors /= np.linalg.norm(eigenvectors, axis=0)[np.newaxis, :]

        return eigenvalues, eigenvectors

    @staticmethod
    def _plot_eigen_pairs(eigenvectors: Numpy2DFloatArray,
                          trivial: bool = False, pair_indices: list = None, **kwargs):
        """
        Plot scatter plot of n-th eigenvector on x-axis and remaining eigenvectors on
        y-axis.

        :param eigenvectors: Eigenvectors of the kernel matrix of shape `(n_samples, n_eigenvectors)`.
        :param  trivial: When trivial constant eigenvectors are ignored is set to `False`.
        :param pair_indices: Indices of the pair of eigenvectors to plot.
        :param kwargs:
            color: visualize the points.
            figure_size: Size of the figure to be passed as keyword argument to `matplotlib.pyplot.figure()`.
            font_size: Size of the font to be passed as keyword argument to `matplotlib.pyplot.figure()`.
        """
        figure_size = kwargs.get('figure_size', None)
        font_size = kwargs.get('font_size', None)
        color = kwargs.get('color', None)

        plt.figure()

        if figure_size is None and font_size is None:
            plt.rcParams["figure.figsize"] = (10, 10)
            plt.rcParams.update({'font.size': 18})
        elif figure_size is not None and font_size is None:
            plt.rcParams["figure.figsize"] = kwargs['figure_size']
            plt.rcParams.update({'font.size': 18})
        elif figure_size is None and font_size is not None:
            plt.rcParams["figure.figsize"] = (10, 10)
            plt.rcParams.update({'font.size': kwargs['font_size']})
        else:
            plt.rcParams["figure.figsize"] = kwargs['figure_size']
            plt.rcParams.update({'font.size': kwargs['font_size']})

        if color is None:
            color = 'b'

        n_eigenvectors = eigenvectors.shape[1]

        if not trivial:
            start = 1
        else:
            start = 0

        num_pairs = 0
        for _ in combinations(range(start, n_eigenvectors), 2):
            num_pairs = num_pairs + 1

        if num_pairs % 2 == 0:
            n_rows = int(np.ceil(num_pairs / 2))
        else:
            n_rows = int(np.ceil(num_pairs / 2)) + 1

        if pair_indices is None:
            _, _ = plt.subplots(
                nrows=n_rows, ncols=2, sharex=True, sharey=True,
            )

            count = 1
            for arg in combinations(range(start, n_eigenvectors), 2):
                i = arg[0]
                j = arg[1]
                plt.subplot(n_rows, 2, count)
                plt.scatter(eigenvectors[:, i], eigenvectors[:, j], c=color, cmap=plt.cm.Spectral)
                plt.title(
                    r"$\Psi_{{{}}}$ vs. $\Psi_{{{}}}$".format(i, j))

                count = count + 1
        else:
            _, _ = plt.subplots(
                nrows=1, ncols=1, sharex=True, sharey=True)
            plt.scatter(eigenvectors[:, pair_indices[0]], eigenvectors[:, pair_indices[1]], c=color,
                        cmap=plt.cm.Spectral)
            plt.title(
                r"$\Psi_{{{}}}$ vs. $\Psi_{{{}}}$".format(pair_indices[0], pair_indices[1]))

