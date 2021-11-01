import numpy as np
from UQpy.Utilities import *
from UQpy.dimension_reduction.kernels.euclidean import GaussianKernel


class GeometricHarmonics:
    """
    Geometric Harmonics for domain extension.
    The class ``GeometricHarmonics`` is used in the domain extension of functions defined only on few observations.
    ``GeometricHarmonics`` is a Subclass of Similarity.
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

    def __init__(self, n_evecs=None, kernel_object=Gaussian()):

        self.kernel_object = kernel_object
        self.n_evecs = n_evecs
        self.basis = None
        self.X = None
        self.y = None
        self.evals = None
        self.evecs = None
        self.kwargs_kernel = None

    def fit(self, X, y, **kwargs):

        """
        Train the model using `fit`.
        In this method, `X` is a list of data points, `y` are the function values. `epsilon` can be
        provided, otherwise it is computed from the median of the pairwise distances of X.
        **Input:**
        * **X** (`list`)
            Input data (independent variables).
        * **y** (`list`)
            Function values.
        * **epsilon** (`float`)
            Parameter of the Gaussian kernel.
        **Output/Returns:**
        """

        if X is not None:

            self.X = X
            self.y = y
            #if not isinstance(X, list):
            #    raise TypeError('UQpy: `X` must be a list.')

            #if not isinstance(y, list):
            #    raise TypeError('UQpy: `y` must be a list.')

            # Get the Gaussian kernel matrix.
            self.kwargs_kernel = kwargs
            self.kernel_object.fit(X=X, **kwargs)
            kernel_matrix = self.kernel_object.kernel_matrix

        else:
            raise TypeError('UQpy: `X` cannot be NoneType.')

        # if n_evecs is NoneType use all the aigenvectors.
        if self.n_evecs is None:
            self.n_evecs = len(X)

        # Eigendecomposition.
        eivals, eivec = eigsolver(kernel_matrix, self.n_evecs)

        idv = np.argsort(eivals)[::-1]
        evals = eivals[idv]
        evecs = eivec[:, idv]

        # Basis construction.
        basis = evecs.dot(np.diag(np.reciprocal(evals))).dot(evecs.T) @ y
        self.basis = basis
        self.X = X
        self.y = y
        self.evals = evals
        self.evecs = evecs

    def predict(self, X):

        """
        Predict the function value for `Xtest`.
        In this method, `Xtest` is a list of data points.
        **Input:**
        * **X** (`list`)
            Input test data (independent variables).
        **Output/Returns:**
        * **ypred** (`list`)
            Predicted function values.
        """

        # Initial checks.
        if self.X is None or self.y is None:
            raise TypeError('UQpy: please, train the model.')

        if X is None:
            raise TypeError('UQpy: Not valid type for X, it should be either a list or a Grassmann object.')

        else:
            # Compute the partial kernel matrix with respect to Xtest.
            self.kernel_object.fit(X=self.X, y=X, **self.kwargs_kernel)
            kernel_matrix = self.kernel_object.kernel_matrix

            # Compute the kernel matrix of X with respect to Xtest using the Gaussian kernel.
            # self.compute_similarity(X=Xtest, y=self.X, epsilon=self.epsilon)
            # kernel_matrix = self.kernel_matrix

        # Prediction using the partial kernel matrix and the basis.
        y = kernel_matrix.T @ self.basis
        #y = (self.basis.T @ kernel_matrix).T

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