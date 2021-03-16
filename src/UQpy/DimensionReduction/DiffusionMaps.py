import copy
import itertools

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from UQpy.Utilities import *
import functools
from UQpy.DimensionReduction.Grassmann import Grassmann

import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.spatial.distance as sd
from scipy.interpolate import LinearNDInterpolator
from UQpy.Utilities import _nn_coord
from UQpy.Surrogates.Kriging import Kriging

########################################################################################################################
########################################################################################################################
#                                            Diffusion Maps                                                            #
########################################################################################################################
########################################################################################################################

class DiffusionMaps:
    """
    Perform the diffusion maps on the input data to reveal its lower dimensional embedded geometry.

    In this class, the diffusion maps create a connection between the spectral properties of the diffusion process and
    the intrinsic geometry of the data resulting in a multiscale representation of it. In this regard, an affinity
    matrix containing the degree of similarity of the data points is either estimated based on the euclidean distance,
    using a Gaussian kernel, or it is computed using any other Kernel definition passed to the main
    method (e.g., defining a kernel on the Grassmann manifold).

    **Input:**

    * **alpha** (`float`)
        Assumes a value between 0 and 1 and corresponding to different diffusion operators. In this regard, one can use
        this parameter to take into consideration the distribution of the data points on the diffusion process.
        It happens because the distribution of the data is not necessarily dependent on the geometry of the manifold.
        Therefore, if alpha` is equal to 1, the Laplace-Beltrami operator is approximated and the geometry of the
        manifold is recovered without taking the distribution of the points into consideration. On the other hand, when
        `alpha` is equal to 0.5 the Fokker-Plank operator is approximated and the distribution of points is taken into
        consideration. Further, when `alpha` is equal to zero the Laplace normalization is recovered.

    * **n_evecs** (`int`)
        The number of eigenvectors and eigenvalues used in the representation of the diffusion coordinates.

    * **sparse** (`bool`)
        Is a boolean variable to activate the `sparse` mode of the method.

    * **k_neighbors** (`int`)
        Used when `sparse` is True to select the k samples close to a given sample in the construction
        of an sparse graph defining the affinity of the input data. For instance, if `k_neighbors` is equal to 10, only
        the closest ten points of a given point are connect to a given point in the graph. As a consequence, the
        obtained affinity matrix is sparse which reduces the computational effort of the eigendecomposition of the
        transition kernel of the Markov chain.
        
    * **kernel_object** (`function`)
        An object of a callable object used to compute the kernel matrix. Three different options are provided:

        - Using the ``DiffusionMaps`` method ``gaussian_kernel`` as
          DiffusionMaps(kernel_object=DiffusionMaps.gaussian_kernel);
        - Using an user defined function as DiffusionMaps(kernel_object=user_kernel);
        - Passing a ``Grassmann`` class object DiffusionMaps(kernel_object=Grassmann_Object). In this case, the user has
          to select ``kernel_grassmann`` in order to define which kernel matrix will be used because when the the
          ``Grassmann`` class is used in a dataset a kernel matrix can be constructed with both the left and right
          singular eigenvectors.

    * **kernel_grassmann** (`str`)
        It assumes the values 'left' and 'right' for the left and right singular eigenvectors used to compute the kernel
        matrix, respectively. Moreover, if 'sum' is selected, it means that the kernel matrix is composed by the sum of
        the kernel matrices estimated using the left and right singular eigenvectors. On the other hand, if 'prod' is used
        instead, it means that the kernel matrix is composed by the product of the matrices estimated using the left and
        right singular eigenvectors.
    
    **Attributes:** 
    
    * **kernel_matrix** (`ndarray`)
        Kernel matrix.
    
    * **transition_matrix** (`ndarray`)
        Transition kernel of a Markov chain on the data.
        
    * **dcoords** (`ndarray`)
        Diffusion coordinates
    
    * **evecs** (`ndarray`)
        Eigenvectors of the transition kernel of a Markov chanin on the data.
    
    * **evals** (`ndarray`)
        Eigenvalues of the transition kernel of a Markov chanin on the data.

    **Methods:**

    """

    def __init__(self, alpha=0.5, n_evecs=2, sparse=False, k_neighbors=1, kernel_object=None, kernel_grassmann=None):

        self.alpha = alpha
        self.n_evecs = n_evecs
        self.sparse = sparse
        self.k_neighbors = k_neighbors
        self.kernel_object = kernel_object
        self.kernel_grassmann = kernel_grassmann

        # from UQpy.DimensionReduction import Grassmann
        # from DimensionReduction import Grassmann

        if kernel_object is not None:
            if callable(kernel_object) or isinstance(kernel_object, Grassmann):
                self.kernel_object = kernel_object
            else:
                raise TypeError('UQpy: Either a callable kernel or a Grassmann class object must be provided.')

        if alpha < 0 or alpha > 1:
            raise ValueError('UQpy: `alpha` must be a value between 0 and 1.')

        if isinstance(n_evecs, int):
            if n_evecs < 1:
                raise ValueError('UQpy: `n_evecs` must be larger than or equal to one.')
        else:
            raise TypeError('UQpy: `n_evecs` must be integer.')

        if not isinstance(sparse, bool):
            raise TypeError('UQpy: `sparse` must be a boolean variable.')
        elif sparse is True:
            if isinstance(k_neighbors, int):
                if k_neighbors < 1:
                    raise ValueError('UQpy: `k_neighbors` must be larger than or equal to one.')
            else:
                raise TypeError('UQpy: `k_neighbors` must be integer.')

    def mapping(self, data=None, epsilon=None):

        """
        Perform diffusion maps to reveal the embedded geometry of datasets.

        In this method, the users have the option to work with input data defined by subspaces obtained via projection
        of input data points on the Grassmann manifold, or directly with the input data points. For example,
        considering that a ``Grassmann`` object is provided using the following command:

        one can instantiate the DiffusionMaps class and run the diffusion maps as follows:

        On the other hand, if the user wish to pass a dataset (samples) to compute the diffusion coordinates using the Gaussian
        kernel, one can use the following commands:

        In the latest case, if `epsilon` is not provided it is estimated based on the median of the square of the
        euclidian distances between data points.

        **Input:**

        * **data** (`list`)
            Data points in the ambient space.
        
        * **epsilon** (`floar`)
            Parameter of the Gaussian kernel.

        **Output/Returns:**

        * **dcoords** (`ndarray`)
            Diffusion coordinates.

        * **evals** (`ndarray`)
            eigenvalues.

        * **evecs** (`ndarray`)
            eigenvectors.

        """

        alpha = self.alpha
        n_evecs = self.n_evecs
        sparse = self.sparse
        k_neighbors = self.k_neighbors

        if data is None and not isinstance(self.kernel_object, Grassmann):
            raise TypeError('UQpy: Data cannot be NoneType.')

        if isinstance(self.kernel_object, Grassmann):

            if self.kernel_grassmann is None:
                raise ValueError('UQpy: kernel_grassmann is not provided.')

            if self.kernel_grassmann == 'left':
                kernel_matrix = self.kernel_object.kernel(self.kernel_object.psi)
            elif self.kernel_grassmann == 'right':
                kernel_matrix = self.kernel_object.kernel(self.kernel_object.phi)
            elif self.kernel_grassmann == 'sum':
                kernel_psi, kernel_phi = self.kernel_object.kernel()
                kernel_matrix = kernel_psi + kernel_phi
            elif self.kernel_grassmann == 'prod':
                kernel_psi, kernel_phi = self.kernel_object.kernel()
                kernel_matrix = kernel_psi * kernel_phi
            else:
                raise ValueError('UQpy: the provided kernel_grassmann is not valid.')

        elif self.kernel_object == DiffusionMaps.gaussian_kernel:
            kernel_matrix = self.kernel_object(self, data=data, epsilon=epsilon)
        elif callable(self.kernel_object) and self.kernel_object != DiffusionMaps.gaussian_kernel:
            kernel_matrix = self.kernel_object(data=data)
        else:
            raise TypeError('UQpy: Not valid type for kernel_object')

        n = np.shape(kernel_matrix)[0]
        if sparse:
            kernel_matrix = self.__sparse_kernel(kernel_matrix, k_neighbors)

        # Compute the diagonal matrix D(i,i) = sum(Kernel(i,j)^alpha,j) and its inverse.
        d, d_inv = self.__d_matrix(kernel_matrix, alpha)

        # Compute L^alpha = D^(-alpha)*L*D^(-alpha).
        l_star = self.__l_alpha_normalize(kernel_matrix, d_inv)

        d_star, d_star_inv = self.__d_matrix(l_star, 1.0)
        if sparse:
            d_star_invd = sps.spdiags(d_star_inv, 0, d_star_inv.shape[0], d_star_inv.shape[0])
        else:
            d_star_invd = np.diag(d_star_inv)

        transition_matrix = d_star_invd.dot(l_star)

        # Find the eigenvalues and eigenvectors of Ps.
        if sparse:
            evals, evecs = spsl.eigs(transition_matrix, k=(n_evecs + 1), which='LR')
        else:
            evals, evecs = np.linalg.eig(transition_matrix)

        ix = np.argsort(np.abs(evals))
        ix = ix[::-1]
        s = np.real(evals[ix])
        u = np.real(evecs[:, ix])

        # Truncated eigenvalues and eigenvectors
        evals = s[:n_evecs]
        evecs = u[:, :n_evecs]

        # Compute the diffusion coordinates
        dcoords = np.zeros([n, n_evecs])
        for i in range(n_evecs):
            dcoords[:, i] = evals[i] * evecs[:, i]

        self.kernel_matrix = kernel_matrix
        self.transition_matrix = transition_matrix
        self.dcoords = dcoords
        self.evecs = evecs
        self.evals = evals

        return dcoords, evals, evecs

    def gaussian_kernel(self, data, epsilon=None):

        """
        Compute the Gaussian Kernel matrix.

        Estimate the affinity matrix using the Gaussian kernel. If no `epsilon` is provided the method estimates a
        suitable value taking the median of the square value of the pairwise euclidean distances of the points in the
        input dataset.

        **Input:**

        * **data** (`list`)
            Input data.

        * **epsilon** (`float`)
            Parameter of the Gaussian kernel.

        **Output/Returns:**

        * **Kernel matrix** (`ndarray`)
            Kernel matrix.

        """

        sparse = self.sparse
        k_neighbors = self.k_neighbors

        # Compute the pairwise distances.
        if len(np.shape(data)) == 2:
            # Set of 1-D arrays
            distance_pairs = sd.pdist(data, 'euclidean')
        elif len(np.shape(data)) == 3:
            # Set of 2-D arrays
            # Check arguments: verify the consistency of input arguments.
            nargs = len(data)
            indices = range(nargs)
            pairs = list(itertools.combinations(indices, 2))

            distance_pairs = []
            for id_pair in range(np.shape(pairs)[0]):
                ii = pairs[id_pair][0]  # Point i
                jj = pairs[id_pair][1]  # Point j

                x0 = data[ii]
                x1 = data[jj]

                distance = np.linalg.norm(x0 - x1, 'fro')

                distance_pairs.append(distance)
        else:
            raise TypeError('UQpy: The size of the input data is not consistent with this method.')

        if epsilon is None:
            # Compute a suitable episilon when it is not provided by the user.
            # Compute epsilon as the median of the square of the euclidean distances
            epsilon = np.median(np.array(distance_pairs) ** 2)

        kernel_matrix = np.exp(-sd.squareform(distance_pairs) ** 2 / (4 * epsilon))

        return kernel_matrix

    # Private method
    @staticmethod
    def __sparse_kernel(kernel_matrix, k_neighbors):

        """
        Private method: Construct a sparse kernel.

        Given the number the k nearest neighbors and a kernel matrix, return a sparse kernel matrix.

        **Input:**

        * **kernel_matrix** (`list` or `ndarray`)
            Kernel matrix.
            
        * **alpha** (`float`)
            Assumes a value between 0 and 1 and corresponding to different diffusion operators.
            
        **Output/Returns:**

        * **D** (`list`)
            Matrix D.

        * **D_inv** (`list`)
            Inverse of matrix D.

        """

        nrows = np.shape(kernel_matrix)[0]
        for i in range(nrows):
            vec = kernel_matrix[i, :]
            idx = _nn_coord(vec, k_neighbors)
            kernel_matrix[i, idx] = 0
            if sum(kernel_matrix[i, :]) <= 0:
                raise ValueError('UQpy: Consider increasing `k_neighbors` to have a connected graph.')

        sparse_kernel_matrix = sps.csc_matrix(kernel_matrix)

        return sparse_kernel_matrix

    # Private method
    @staticmethod
    def __d_matrix(kernel_matrix, alpha):

        """
        Private method: Compute the diagonal matrix D and its inverse.

        In the normalization process we have to estimate matrix D(i,i) = sum(Kernel(i,j)^alpha,j) and its inverse.

        **Input:**

        * **kernel_matrix** (`list` or `ndarray`)
            Kernel matrix.
            
        * **alpha** (`float`)
            Assumes a value between 0 and 1 and corresponding to different diffusion operators.
            
        **Output/Returns:**

        * **d** (`list`)
            Matrix D.

        * **d_inv** (`list`)
            Inverse of matrix D.

        """

        d = np.array(kernel_matrix.sum(axis=1)).flatten()
        d_inv = np.power(d, -alpha)

        return d, d_inv

    # Private method
    def __l_alpha_normalize(self, kernel_mat, d_inv):

        """
        Private method: Compute and normalize the kernel matrix with the matrix D.

        In the normalization process we have to estimate matrix D(i,i) = sum(Kernel(i,j)^alpha,j) and its inverse.
        We now use this information to normalize the kernel matrix.

        **Input:**

        * **kernel_mat** (`list` or `ndarray`)
            Kernel matrix.

        * **d_inv** (`list` or `ndarray`)
            Inverse of matrix D.

        **Output/Returns:**

        * **normalized_kernel** (`list` or `ndarray`)
            Normalized kernel.

        """

        sparse = self.sparse
        m = d_inv.shape[0]
        if sparse:
            d_alpha = sps.spdiags(d_inv, 0, m, m)
        else:
            d_alpha = np.diag(d_inv)

        normalized_kernel = d_alpha.dot(kernel_mat.dot(d_alpha))

        return normalized_kernel
