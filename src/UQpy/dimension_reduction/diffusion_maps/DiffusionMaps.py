import itertools

import scipy.sparse as sps
import scipy as sp
from scipy.sparse.linalg import eigsh, eigs
import scipy.spatial.distance as sd
import matplotlib.pyplot as plt
import scipy
import numpy as np
from itertools import combinations

from UQpy.utilities import GaussianKernel, GrassmannPoint
from UQpy.utilities.Utilities import *
from UQpy.utilities.Utilities import _nn_coord
from beartype import beartype
from typing import Annotated, Union
from beartype.vale import Is
from UQpy.utilities.ValidationTypes import Numpy2DFloatArray, NumpyFloatArray
from UQpy.utilities.kernels.baseclass.Kernel import Kernel


class DiffusionMaps:
    AlphaType = Annotated[Union[float, int], Is[lambda number: 0 <= number <= 1]]
    IntegerLargerThanUnityType = Annotated[int, Is[lambda number: number >= 1]]

    @beartype
    def __init__(
            self,
            kernel_matrix: Numpy2DFloatArray = None,
            data: Union[Numpy2DFloatArray, list[GrassmannPoint]] = None,
            kernel: Kernel = None,
            alpha: AlphaType = 0.5,
            n_eigenvectors: IntegerLargerThanUnityType = 2,
            is_sparse: bool = False,
            n_neighbors: IntegerLargerThanUnityType = 1,
            random_state: RandomStateType = None,
            t: int = 1
    ):
        """

        :param kernel_matrix: Kernel matrix defining the similarity between the data points. Either `kernel_matrix` or
            both `data` and `kernel` parameters must be provided. In the former case, `kernel_matrix` is precomputed
            using a :class:`Kernel` class. In the second case the `kernel_matrix` is internally and used for the
            evaluation of the :class:`.DiffusionMaps`. In case all three of the aforementioned parameters are provided,
            then :class:`.DiffusionMaps` will be fitted only using the `kernel_matrix`
        :param data: Set of data points. Either `kernel_matrix` or both `data` and `kernel` parameters must be
            provided.
        :param kernel: Kernel object used to compute the kernel matrix defining similarity between the data points.
            Either `kernel_matrix` or both `data` and `kernel` parameters must be provided.
        :param alpha: A scalar that corresponds to different diffusion operators. `alpha` should be between zero and
            one.
        :param n_eigenvectors: Number of eigenvectors to retain.
        :param is_sparse:  Work with sparse matrices to improve computational performance.
        :param n_neighbors: If :code:`is_sparse is True`, defines the number of nearest neighbors to use when making
            matrices sparse.
        :param random_state: Random seed used to initialize the pseudo-random number generator. If an :any:`int` is
         provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise, the
         object itself can be passed directly.
        :param t: Time exponent.
        """
        self.parsimonious_residuals = None
        """Residuals calculated from the Parsimonious Representation. This attribute will only be populated if the 
        :py:meth:`parsimonious`  method is invoked."""
        self.parsimonious_indices = None
        """Indices of the most important eigenvectors. This attribute will only be populated if the 
        :py:meth:`parsimonious`  method is invoked."""

        if kernel_matrix is not None:
            self.kernel_matrix = kernel_matrix
        elif data is not None and kernel is not None:
            kernel.calculate_kernel_matrix(x=data, s=data)
            self.kernel_matrix = kernel.kernel_matrix
        else:
            raise ValueError("Either `kernel_matrix` or both `data` and `kernel` must be provided")

        self.alpha = alpha
        self.eigenvectors_number = n_eigenvectors
        self.is_sparse = is_sparse
        self.neighbors_number = n_neighbors

        self.random_state = random_state,
        self.t = t

        self.transition_matrix: np.ndarray = None
        '''
        Markov Transition Probability Matrix.
        '''
        self.diffusion_coordinates: np.ndarray = None
        '''
        Coordinates of the data in the diffusion space.
        '''
        self.eigenvectors: np.ndarray = None
        '''
        Eigenvectors of the transition probability matrix.
        '''
        self.eigenvalues: np.ndarray = None
        '''
        Eigenvalues of the transition probability matrix.
        '''
        self.cut_off = None

        self._fit()

    def _fit(self):
        """
        Perform diffusion map embedding.
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
        d_alpha = sps.spdiags(d_inv, 0, m, m) if self.is_sparse else np.diag(d_inv)
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
            is_symmetric = np.allclose(transition_matrix, transition_matrix.T, rtol=1e-5, atol=1e-08)

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

        self.diffusion_coordinates = diffusion_coordinates
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

    # Private method
    @staticmethod
    def __sparse_kernel(kernel_matrix, neighbors_number):
        rows = np.shape(kernel_matrix)[0]
        for i in range(rows):
            row_data = kernel_matrix[i, :]
            index = _nn_coord(row_data, neighbors_number)
            kernel_matrix[i, index] = 0
            if sum(kernel_matrix[i, :]) <= 0:
                raise ValueError("UQpy: Consider increasing `n_neighbors` to have a connected graph.")

        return sps.csc_matrix(kernel_matrix)

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

    def parsimonious(self, dim: int):
        """
        Selection of independent vectors for parsimonious data manifold embedding, based on
        local regression.  The eigenvectors with the largest residuals are considered for the
        embedding. The scale of the kernel used for the local linear regression is:

        .. code::

            scale = median(distances) / 3

        :param dim: Number of eigenvectors to select with largest residuals.
        """

        residuals = np.zeros(self.eigenvectors.shape[1])
        residuals[0] = np.nan
        # residual 1 for the first eigenvector.
        residuals[1] = 1.0

        # Get the residuals of each eigenvector.
        for i in range(2, self.eigenvectors.shape[1]):
            residuals[i] = DiffusionMaps.__get_residual(f_mat=self.eigenvectors[:, 1:i], f=self.eigenvectors[:, i])

        # Get the index of the eigenvalues associated with each residual.
        indices = np.argsort(residuals)[::-1][1:dim + 1]
        self.parsimonious_indices = indices
        self.parsimonious_residuals = residuals

    @staticmethod
    def __get_residual(f_mat, f):
        n_samples = np.shape(f_mat)[0]
        distance_matrix = sd.squareform(sd.pdist(f_mat))
        m = 3
        epsilon = (np.median(np.square(distance_matrix.flatten())) / m)
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
    def eig_solver(kernel_matrix: Numpy2DFloatArray, is_symmetric: bool, n_eigenvectors: int) -> tuple[
        NumpyFloatArray, Numpy2DFloatArray]:

        n_samples, n_features = kernel_matrix.shape

        if n_eigenvectors == n_features:
            solver = sp.linalg.eigh if is_symmetric else sp.linalg.eig
            solver_kwargs = {"check_finite": False}

        else:
            solver = eigsh if is_symmetric else eigs
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
                          trivial: bool = False, pair_indices: list = None, **kwargs):  # pragma: no cover
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
        elif figure_size is None:
            plt.rcParams["figure.figsize"] = (10, 10)
            plt.rcParams.update({'font.size': kwargs['font_size']})
        else:
            plt.rcParams["figure.figsize"] = kwargs['figure_size']
            plt.rcParams.update({'font.size': kwargs['font_size']})

        if color is None:
            color = 'b'

        n_eigenvectors = eigenvectors.shape[1]

        start = 1 if not trivial else 0
        num_pairs = sum(1 for _ in combinations(range(start, n_eigenvectors), 2))
        if num_pairs % 2 == 0:
            n_rows = int(np.ceil(num_pairs / 2))
        else:
            n_rows = int(np.ceil(num_pairs / 2)) + 1

        if pair_indices is None:
            _, _ = plt.subplots(
                nrows=n_rows, ncols=2, sharex=True, sharey=True,
            )

            for count, arg in enumerate(combinations(range(start, n_eigenvectors), 2), start=1):
                i = arg[0]
                j = arg[1]
                plt.subplot(n_rows, 2, count)
                plt.scatter(eigenvectors[:, i], eigenvectors[:, j], c=color, cmap=plt.cm.Spectral)
                plt.title(
                    r"$\Psi_{{{}}}$ vs. $\Psi_{{{}}}$".format(i, j))

        else:
            _, _ = plt.subplots(
                nrows=1, ncols=1, sharex=True, sharey=True)
            plt.scatter(eigenvectors[:, pair_indices[0]], eigenvectors[:, pair_indices[1]], c=color,
                        cmap=plt.cm.Spectral)
            plt.title(
                r"$\Psi_{{{}}}$ vs. $\Psi_{{{}}}$".format(pair_indices[0], pair_indices[1]))
