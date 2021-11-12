import numpy as np
from UQpy.utilities import *
from UQpy.dimension_reduction.kernels import GaussianKernel


class GeometricHarmonics:
    """
    Geometric Harmonics for domain extension. The class ``GeometricHarmonics`` is used in the domain extension of
    functions defined only on few observations.

    **Input:**
    * **n_evecs** (`int`)
        The number of eigenvectors used in the eigendecomposition of the kernel matrix.
    * **kernel_method** (`callable`)
        Kernel method used in the construction of the geometric harmonics.
    **Attributes:**
    * **n_evecs** (`int`)
        The number of eigenvectors used in the eigendecomposition of the kernel matrix.
    * **X** (`list`)
        Independent variables.
    * **y** (`list`)
        Function values.
    * **basis** (`list`)
        Basis used in the domain extension.
    **Methods:**
    """

    def __init__(self, n_eigen_pairs: int = 5, kernel_object=GaussianKernel()):
        """

        :param n_eigen_pairs: The number of eigenvectors used in the decomposition of the kernel matrix.
        :param kernel_object: Kernel used for the construction of the geometric harmonics.

        See Also
        --------

        :py:class:`UQpy.dimension_reduction.kernels.GaussianKernel`

        """

        self.kernel_object = kernel_object
        self.n_eigen_pairs = n_eigen_pairs
        self.basis = None
        self.x = None
        self.y = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.kwargs_kernel = None

    def fit(self, x: Numpy2DFloatArray, y: Numpy2DFloatArray, **kwargs):

        """
        Fit model with training data.
        In this method, `X` is a list of data points, `y` are the function values. `epsilon` can be
        provided, otherwise it is computed from the median of the pairwise distances of X.

        :param x: Training points of shape `(n_samples, n_features)`.
        :param y: Target function values of shape `(n_samples, n_targets)`
        :param kwargs:
        """

        self.x = x
        self.y = y
        self.kwargs_kernel = kwargs

        if epsilon is None:
            epsilon, _ = DiffusionMaps.estimate_epsilon(x, cut_off=cut_off, tol=tol,
                                                        k_nn=k_nn, n_partition=n_partition,
                                                        distance_matrix=distance_matrix,
                                                        random_state=random_state)
        kernel.epsilon = epsilon
        kernel_matrix = self.kernel_object.kernel_operator(points=x)

        eigenvalues, eigenvectors = eigsolver(kernel_matrix, self.n_eigen_pairs)

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

    def predict(self, x) -> Numpy2DFloatArray:

        """
        Evaluate model for out-of-sample points.

        :param x:  Points of shape `(n_samples, n_features)`
        :return: The interpolated function values of shape `(n_samples, n_targets)`
        """



        # Compute the partial kernel matrix with respect to Xtest.
        kernel_matrix = self.kernel_object.kernel_operator(points=x)


        # Prediction using the partial kernel matrix and the basis.
        y = kernel_matrix.T @ self.basis


        return y

    def score(self, kind='abs'):

        """
        Score for the approximation using the values in `X`.
        """

        # Initial checks.
        if self.X is None or self.y is None:
            raise TypeError('UQpy: please, train the model.')

        yexact = self.y
        nx = len(self.X)
        xt = self.X
        error = []
        for i in range(nx):
            ypred = self.predict([xt[i]])[0]

            # Compute the error between the exact value and the prediction.
            if kind == 'abs':
                error_value = np.linalg.norm(ypred - yexact[i])
            elif kind == 'rel':
                error_value = np.linalg.norm(ypred - yexact[i]) / np.linalg.norm(yexact[i])
            elif kind == 'max':
                error_value = np.max(abs(ypred - yexact[i]))
            elif kind == 'l1':
                error_value = np.linalg.norm(ypred - yexact[i], 1)
            else:
                raise NotImplementedError('UQpy: not implement kind of error.')

            error.append(error_value)

        mean_error = np.mean(error)

        return mean_error