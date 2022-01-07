from UQpy.dimension_reduction.diffusion_maps.DiffusionMaps import DiffusionMaps
from UQpy.utilities import *
from UQpy.dimension_reduction.kernels import GaussianKernel


class GeometricHarmonics:
    """
    Geometric Harmonics for domain extension.
    """

    def __init__(self, eigenvectors_number: int = 5, kernel_object=GaussianKernel()):
        """

        :param eigenvectors_number: The number of eigenvectors used in the decomposition of the kernel matrix.
        :param kernel_object: Kernel used for the construction of the geometric harmonics.

        See Also
        --------

        :py:class:`.GaussianKernel`

        """

        self.kernel_object = kernel_object
        self.n_eigen_pairs = eigenvectors_number
        self.basis = None
        self.x = None
        self.y = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.kwargs_kernel = None

    def fit(self, x: Numpy2DFloatArray, y: Numpy2DFloatArray, **kwargs):

        """
        Fit model with training data.

        :param x: Training points of shape :code:`(n_samples, n_features)`.
        :param y: Target function values of shape :code:`(n_samples, n_targets)`
        :param kwargs: Scale parameter of the kernel. if :code:`epsilon is None` then the parameters for the
        estimation of the kernel's scale must be provided.

        See Also
        --------

        :py:meth:`.DiffusionMaps.estimate_epsilon`

        """

        self.x = x
        self.y = y
        self.kwargs_kernel = kwargs

        if kwargs['epsilon'] is None:

            epsilon, _ = DiffusionMaps.estimate_epsilon(x, cut_off=kwargs['cut_off'], tol=kwargs['tol'],
                                                        k_nn=kwargs['k_nn'], n_partition=kwargs['n_partition'],
                                                        distance_matrix=kwargs['distance_matrix'],
                                                        random_state=kwargs['random_state'])

        else:
            epsilon = kwargs['epsilon']

        self.kernel_object.epsilon = epsilon
        kernel_matrix = self.kernel_object.kernel_operator(points=x)

        eigenvalues, eigenvectors = DiffusionMaps.eig_solver(kernel_matrix, self.n_eigen_pairs)

        idv = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idv]
        eigenvectors = eigenvectors[:, idv]

        # Basis construction.
        basis = eigenvectors.dot(np.diag(np.reciprocal(eigenvalues))).dot(eigenvectors.T) @ y
        self.basis = basis
        self.x = x
        self.y = y
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

    def predict(self, x: Numpy2DFloatArray) -> tuple[Numpy2DFloatArray, NumpyFloatArray]:

        """
        Evaluate model for out-of-sample points.

        :param x:  Points of shape :code:`(n_samples, n_features)`
        :return: The interpolated function values of shape :code:`(n_samples, n_targets)`
        """

        if self.kwargs_kernel['epsilon'] is None:

            epsilon, _ = DiffusionMaps.estimate_epsilon(x, cut_off=self.kwargs_kernel['cut_off'],
                                                        tol=self.kwargs_kernel['tol'],
                                                        k_nn=self.kwargs_kernel['k_nn'],
                                                        n_partition=self.kwargs_kernel['n_partition'],
                                                        distance_matrix=self.kwargs_kernel['distance_matrix'],
                                                        random_state=self.kwargs_kernel['random_state'])
        else:
            epsilon = self.kwargs_kernel['epsilon']

        self.kernel_object.epsilon = epsilon
        kernel_matrix = self.kernel_object.kernel_operator(points=x)
        y = kernel_matrix.T @ self.basis
        score = self.score(x, y, self.kwargs_kernel['score'])
        return y, score

    @staticmethod
    def score(y: Numpy2DFloatArray, y_predicted: Numpy2DFloatArray, kind: str = 'abs') -> float:

        """
        Score interpolation model with negative mean squared error metric.

        :param y: The exact function values of shape :code:`(n_test, n_targets)`
        :param y_predicted: The interpolated function values of shape :code:`(n_test, n_targets)`
        :param kind: Error metric to be used. If :code:`kind is None` then the absolute mean error will be calculated.
        :return: The mean computed error between the exact values and the predictions.
        """
        n_samples = y.shape[0]
        error = []
        for i in range(n_samples):
            # Compute the error between the exact value and the prediction.
            if kind == 'abs':
                error_value = np.linalg.norm(y_predicted[i, :] - y[i, :])
            elif kind == 'l1':
                error_value = np.linalg.norm(y_predicted[i, :] - y[i, :], 1)
            elif kind == 'max':
                error_value = np.max(abs(y_predicted[i, :] - y[i, :]))
            elif kind == 'rel':
                error_value = np.linalg.norm(y_predicted[i, :] - y[i, :]) / np.linalg.norm(y[i, :])
            else:
                raise NotImplementedError('UQpy: Not implement kind of error.')

            error.append(error_value)

        return np.mean(error)
