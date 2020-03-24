# UQpy is distributed under the MIT license.
#
# Copyright (C) 2018  -- Michael D. Shields
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""This module contains functionality for all the surrogate methods supported in UQpy."""

from UQpy.Surrogates import Krig
from Utilities import *
import scipy as sp
import numpy as np
import itertools
from scipy.interpolate import LinearNDInterpolator
# from fbpca import pca
from os import path
import math

import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.spatial.distance as sd


########################################################################################################################
########################################################################################################################
#                                            Dimension Reduction                                                       #
########################################################################################################################
########################################################################################################################


class Grassmann:
    """

        Description:

            DimensionReduction is a class containing the methods to compute distances on the Grassmann manifold,
            as well as to project points onto the Grassmann manifold and the tangent space. Moreover, it also implements
            the interpolation of large matrices on a tangent space taking advantage of their lower dimensional structure
            when projected on the Grassmann manifold.

            References:
            Jiayao Zhang, Guangxu Zhu, Robert W. Heath Jr., and Kaibin Huang, "Grassmannian Learning: Embedding Geometry
            Awareness in Shallow and Deep Learning", arXiv:1808.02229 [cs, eess, math, stat], Aug. 2018.

            D.G. Giovanis, M.D. Shields, "Uncertainty quantification for complex systems with very high dimensional
            response using Grassmann manifold variations",
            Journal of Computational Physics, Volume 364, Pages 393-415, 2018.

        Input:

            :param distance_object: It specifies the name of a function or class implementing the distance metric.
            :type distance_object: str

            :param distance_script: The filename (with extension) of a Python script implementing dist_object
                                (only for user defined metrics).
            :type distance_script: str

            :param kernel_object: It specifies the name of a function or class implementing the Grassmann kernel.
            :type distance_object: str

            :param kernel_script: The filename (with extension) of a Python script implementing kernel_object
                                (only for user defined metrics).
            :type distance_script: str

            :param interp_object: It specifies the name of the function or class implementing the interpolator.
            :type interp_object: str

            :param interp_script: The filename (with extension) of the Python script implementing of interp_object
                                  (only for user defined interpolator).
            :type interp_script: str

        """

    # Authors: Ketson R. M. dos Santos, Dimitris G. Giovanis
    # Updated: 03/24/20 by Ketson R. M. dos Santos

    def __init__(self, distance_object=None, distance_script=None, kernel_object=None, kernel_script=None, interp_object=None,
                 interp_script=None):

        # Distance.
        self.distance_script = distance_script
        self.distance_object = distance_object
        if distance_script is not None:
            self.user_distance_check = path.exists(distance_script)
        else:
            self.user_distance_check = False

        if self.user_distance_check:
            try:
                self.module_dist = __import__(self.distance_script[:-3])
            except ImportError:
                raise ImportError('There is no module implementing a metric on the Grassmann manifold.')

        # Kernels.
        self.kernel_script = kernel_script
        self.kernel_object = kernel_object
        if kernel_script is not None:
            self.user_kernel_check = path.exists(kernel_script)
        else:
            self.user_kernel_check = False

        if self.user_kernel_check:
            try:
                self.module_kernel = __import__(self.kernel_script[:-3])
            except ImportError:
                raise ImportError('There is no module implementing a Grassmann kernel.')

        # Interpolation.
        self.interp_script = interp_script
        self.interp_object = interp_object
        if interp_script is not None:
            self.user_interp_check = path.exists(interp_script)
        else:
            self.user_interp_check = False

        if self.user_interp_check:
            try:
                self.module_interp = __import__(self.interp_script[:-3])
            except ImportError:
                raise ImportError('There is no module implementing the distance metric.')

    # Calculate the distance on the manifold
    def distance(self, *argv, **kwargs):
        
        """
        distance: Estimate the distance of points on the Grassmann manifold.

        :param argv: list or numpy ndarray containing all the matrices (at least 2) corresponding to the points
              on the Grassmann manifold
        :param kwargs: contains the keyword for the user defined rank. If a list or numpy ndarray containing the
                       rank of each matrix is not provided, the code will compute them using numpy.linalg.matrix_rank. 
        """

        nargs = len(argv[0])
        psi = argv[0]

        if 'rank' in kwargs.keys():
            ranks = kwargs['rank']
        else:
            ranks = None

        # If rank is not provide compute internally.
        if ranks is None:
            ranks = []
            for i in range(nargs):
                ranks.append(np.linalg.matrix_rank(psi[i]))

        if type(ranks) != list and type(ranks) != np.ndarray:
            raise TypeError('rank MUST be either a list or ndarray.')

        # Define the pairs of points to compute the Grassmann distance.
        indices = range(nargs)
        pairs = list(itertools.combinations(indices, 2))

        if self.user_distance_check:
            exec('from ' + self.distance_script[:-3] + ' import ' + self.distance_object)
            distance_fun = eval(self.distance_object)
        else:
            distance_fun = eval("Grassmann." + self.distance_object)

        distance_list = []
        for id_pair in range(np.shape(pairs)[0]):
            ii = pairs[id_pair][0]  # Point i
            jj = pairs[id_pair][1]  # Point j

            rank0 = int(ranks[ii])
            rank1 = int(ranks[jj])

            x0 = np.asarray(psi[ii])[:, :rank0]
            x1 = np.asarray(psi[jj])[:, :rank1]

            dist = distance_fun(x0, x1)

            distance_list.append(dist)

        return distance_list

    # ==================================================================================================================
    # Built-in metrics are implemented in this section. Any new built-in metric must be implemented
    # here with the decorator @staticmethod.

    # Grassmann distance.
    @staticmethod
    def grassmann_distance(x0, x1):
       
        l = min(np.shape(x0))
        k = min(np.shape(x1))
        rank = min(l, k)

        r = np.dot(x0.T, x1)
        (ui, si, vi) = pca(r, rank, True)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(si)
        distance = np.sqrt(abs(k - l) * np.pi ** 2 / 4 + np.sum(theta ** 2))

        return distance

    # Chordal distance.
    @staticmethod
    def chordal_distance(x0, x1):

        l = min(np.shape(x0))
        k = min(np.shape(x1))
        rank = min(l, k)

        r_star = np.dot(x0.T, x1)
        (ui, si, vi) = pca(r_star, rank, True)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(si)
        theta = (np.sin(theta)) ** 2
        distance = np.sqrt(abs(k - l) + np.sum(theta))

        return distance

    # Procrustes distance.
    @staticmethod
    def procrustes_distance(x0, x1):

        l = min(np.shape(x0))
        k = min(np.shape(x1))
        rank = min(l, k)

        r = np.dot(x0.T, x1)
        (ui, si, vi) = pca(r, rank, True)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(si)
        theta = np.sin(theta / 2) ** 2
        distance = np.sqrt(abs(k - l) + 2 * np.sum(theta))

        return distance

    # Projections distance.
    @staticmethod
    def projection_distance(x0, x1):

        # Check rank and swap.
        c = np.zeros([x0.shape[0], x0.shape[1]])
        if x0.shape[1] < x1.shape[1]:
            x0 = x1
            x1 = c

        # Compute the projection.
        r = np.dot(x0.T, x1)
        x1 = x1 - np.dot(x0, r)

        distance = np.arcsin(min(1, sp.linalg.norm(x1)))

        return distance

    # ==================================================================================================================
    def kernel(self, *argv, **kwargs):

        """
        kernel: it implements different kernels defined on the Grassmann manifold.
        :param argv: list or numpy ndarray containing all the matrices (at least 2) corresponding to the points
              on the Grassmann manifold.
        :param kwargs: contains the keyword for the user defined rank. If a list or numpy ndarray containing the
                       rank of each matrix is not provided, the code will compute them using numpy.linalg.matrix_rank.      
        """

        nargs = len(argv[0])
        psi = argv[0]

        if 'rank' in kwargs.keys():
            ranks = kwargs['rank']
        else:
            ranks = None

        # If rank is not provide compute internally.
        if ranks is None:
            ranks = []
            for i in range(nargs):
                ranks.append(np.linalg.matrix_rank(psi[i]))

        if type(ranks) != list and type(ranks) != np.ndarray:
            raise TypeError('rank MUST be either a list or ndarray.')

        # Define the pairs of points to compute the Grassmann distance.
        indices = range(nargs)
        pairs = list(itertools.combinations(indices, 2))

        if self.user_kernel_check:
            exec('from ' + self.kernel_script[:-3] + ' import ' + self.kernel_object)
            kernel_fun = eval(self.kernel_object)
        else:
            kernel_fun = eval("Grassmann." + self.kernel_object)

        kernel_list = []
        for id_pair in range(np.shape(pairs)[0]):
            ii = pairs[id_pair][0]  # Point i
            jj = pairs[id_pair][1]  # Point j

            x0 = psi[ii]
            x1 = psi[jj]

            rank0 = ranks[ii]
            rank1 = ranks[jj]
            x0 = x0[:, :rank0]  # Truncating the matrices.
            x1 = x1[:, :rank1]

            ker = kernel_fun(x0, x1)
            kernel_list.append(ker)

        kernel_diag = []
        for id_elem in range(nargs):
            xd = psi[id_elem]
            rankd = xd.shape[1]
            xd = xd[:, :rankd]

            kerd = kernel_fun(xd, xd)
            kernel_diag.append(kerd)

        kernel_matrix = sd.squareform(kernel_list) + np.diag(kernel_diag)

        return kernel_matrix

    # ==================================================================================================================
    # Built-in kernels are implemented in this section. Any new built-in kernel must be implemented
    # here with the decorator @staticmethod.

    # Projection kernel.
    @staticmethod
    def projection_kernel(x0, x1):

        r = np.dot(x0.T, x1)
        n = np.linalg.norm(r, 'fro')
        ker = n * n
        return ker

    # Binet-Cauchy kernel.
    @staticmethod
    def binet_cauchy_kernel(x0, x1):

        r = np.dot(x0.T, x1)
        det = np.linalg.det(r)
        ker = det * det
        return ker

    # ==================================================================================================================
    @staticmethod
    def project_points(*argv, **kwargs):

        """
        project_points: project the input matrices onto the Grassmann manifold via singular value decomposition (SVD)
        :param argv: list or numpy nadarray containing the matrices to be projected onto the Grassmann manifold.
        :param kwargs: contains the keyword for the user defined rank. If a list or numpy ndarray containing the
                       rank of each matrix is not provided, the code will compute them using numpy.linalg.matrix_rank.
        """

        # Check the input arguments.
        matrix, nargs = check_arguments(argv, min_num_matrix=1, ortho=False)

        ranks = None
        cont = 0
        for key, value in kwargs.items():
            if key == "rank":
                ranks = value
                break

            cont += 1

        # If user defined rank is not provided.
        if ranks is None:
            ranks = []
            for i in range(nargs):
                # Estimate the rank of each solution.
                ranks.append(np.linalg.matrix_rank(matrix[i]))

        if type(ranks) != list and type(ranks) != np.ndarray:
            raise TypeError('rank MUST be either a list or ndarray.')

        max_rank = int(np.max(ranks))  # rank = max(r1, r2, ..., rn)
        psi = []  # initialize the left singular vectors as a list.
        sigma = []  # initialize the singular values as a list.
        phi = []  # initialize the right singular vectors as a list.

        # For each point perform svd with max_rank columns.
        for i in range(nargs):
            u, s, v = svd(matrix[i], max_rank)
            psi.append(u)
            sigma.append(np.diag(s))
            phi.append(v.T)

        return psi, sigma, phi, max_rank, ranks

    @staticmethod
    def log_mapping(*argv, ref=None):
        """
        log_mapping: projecting points on the Grassmann manifold onto the tangent space.

        :param argv: list or numpy ndarray containing all the points on the Grassmann manifold.
        :param ref: list or numpy ndarray containing the reference point on the Grassmann manifold
                    where the tangent space is approximated.
        """
        # Check the input arguments.
        u, nr = check_arguments(argv, min_num_matrix=1, ortho=False)

        # The input matrices are truncated by the user.
        rank = []
        for i in range(nr):
            rank.append(min(np.shape(u[i])))

        if type(ref) != list:
            ref_list = ref.tolist()
        else:
            ref_list = ref
            ref = np.array(ref)

        refT = ref.T
        m0 = np.dot(ref, refT)

        _gamma = list()
        for i in range(nr):
            if u[i].tolist() == ref_list:
                _gamma.append(np.zeros(np.shape(ref)))
            else:

                # M = ((I - psi0*psi0')*psi1)*inv(psi0'*psi1)
                minv = np.linalg.inv(np.dot(refT, u[i]))
                m = np.dot(u[i] - np.dot(m0, u[i]), minv)
                ui, si, vi = np.linalg.svd(m, full_matrices=False)  # svd(m, max_rank)
                _gamma.append(np.dot(np.dot(ui, np.diag(np.arctan(si))), vi))

        return _gamma

    @staticmethod
    def exp_mapping(*argv, ref=None):

        """
        exp_mapping: perform the exponential mapping from the tangent space
                     to the Grassmann manifold.
        :param ref: list or ndarray containing the reference point on the Grassmann manifold
                    where the tangent space is approximated.

        """
        # Check input arguments.
        u, nr = check_arguments(argv, min_num_matrix=1, ortho=False)

        rank = []
        for i in range(nr):
            rank.append(min(np.shape(u[i])))

        x = list()
        for j in range(nr):

            ui, si, vi = np.linalg.svd(u[j], full_matrices=False)

            # Exponential mapping.
            x0 = np.dot(np.dot(np.dot(ref, vi.T), np.diag(np.cos(si))) + np.dot(ui, np.diag(np.sin(si))), vi)

            # Test orthogonality.
            xtest = np.dot(x0.T, x0)

            if not np.allclose(xtest, np.identity(np.shape(xtest)[0])):
                x0, unused = np.linalg.qr(x0)  # re-orthonormalizing.

            x.append(x0)

        return x

    # @staticmethod
    def karcher_mean(self, *argv, acc=False, tol=1e-3, maxiter=1000):

        """
        karcher_mean: estimate the Karcher mean.
        :param acc: boolean variable for the use of the Nesterov approach to accelerate the rate of convergence.
        :param tol: tolerance.
        :param maxiter: integer with the maximum number of iterations.
        """
        # np.random.seed( 234 ) if a fixed seed is desired.
        matrix, n_mat = check_arguments(argv, min_num_matrix=2, ortho=False)

        alpha = 0.5  # todo: this can be adaptive (e.g., Armijo rule).
        rnk = []
        for i in range(n_mat):
            rnk.append(min(np.shape(matrix[i])))

        # Maximum rank
        max_rank = np.max(rnk)

        fmean = []
        for i in range(n_mat):
            fmean.append(self.frechet_mean(matrix[i], matrix))

        index_0 = fmean.index(min(fmean))
        mean_element = list()
        mean_element = matrix[index_0].tolist()

        avg_gamma = np.zeros([np.shape(matrix[0])[0], np.shape(matrix[0])[1]])

        itera = 0

        l = 0
        avg = []
        _gamma = []
        if acc:
            _gamma = Grassmann.log_mapping(matrix, ref=np.asarray(mean_element))
            avg_gamma.fill(0)
            for i in range(n_mat):
                avg_gamma += _gamma[i] / n_mat
            avg.append(avg_gamma)

        # Main loop
        while itera <= maxiter:

            _gamma = Grassmann.log_mapping(matrix, ref=np.asarray(mean_element))
            avg_gamma.fill(0)

            for i in range(n_mat):
                avg_gamma += _gamma[i] / n_mat

            test_0 = np.linalg.norm(avg_gamma, 'fro')
            if test_0 < tol and itera == 0:
                break

            # Nesterov: Accelerated Gradient Descent
            if acc:
                avg.append(avg_gamma)
                l0 = l
                l1 = 0.5 * (1 + np.sqrt(1 + 4 * l * l))
                ls = (l0 - 1) / l1
                step = (1 - ls) * avg[itera + 1] + ls * avg[itera]
                L = l1
            else:
                step = alpha * avg_gamma

            x = Grassmann.exp_mapping(step, ref=np.asarray(mean_element))

            test_1 = np.linalg.norm(x[0] - mean_element, 'fro')

            if test_1 < tol:
                break

            mean_element = []
            mean_element = x[0]

            itera += 1

        return mean_element

    # @staticmethod
    def karcher_mean_sgd(self, *argv, tol=1e-3, maxiter=1000):

        """
        karcher_mean_sgd: estimate the Karcher mean using the stochastic gradient descent.
        :param acc: boolean variable for the use of the Nesterov approach to accelerate the rate of convergence.
        :param tol: real value for the tolerance.
        :param maxiter: integer with the maximum number of iterations.
        """
        matrix, n_mat = check_arguments(argv, min_num_matrix=2, ortho=False)

        rnk = []
        for i in range(n_mat):
            rnk.append(min(np.shape(matrix[i])))

        max_rank = np.max(rnk)

        fmean = []
        for i in range(n_mat):
            fmean.append(self.frechet_mean(matrix[i], matrix))

        index_0 = fmean.index(min(fmean))

        mean_element = matrix[index_0].tolist()
        itera = 0
        _gamma = []
        k = 1
        while itera < maxiter:

            indices = np.arange(n_mat)
            np.random.shuffle(indices)

            melem = mean_element
            for i in range(len(indices)):
                alpha = 0.5 / k
                idx = indices[i]
                _gamma = Grassmann.log_mapping(matrix[idx], ref=np.asarray(mean_element))

                step = 2 * alpha * _gamma[0]

                X = Grassmann.exp_mapping(step, ref=np.asarray(mean_element))

                _gamma = []
                mean_element = X[0]

                k += 1

            test_1 = np.linalg.norm(mean_element - melem, 'fro')
            if test_1 < tol:
                break

            itera += 1

        return mean_element

    # @staticmethod
    def frechet_mean(self, k_mean, *argv):

        """
        frechet_mean: estimate the Frechet mean.
        :param k_mean: list of numpy ndarray point of interested (on a manifold).
        """
        matrix, n_mat = check_arguments(argv, min_num_matrix=2, ortho=False)

        accum = 0
        for i in range(n_mat):
            d = self.distance([k_mean, matrix[i]])
            accum += d[0] * d[0]

        fmean = accum / n_mat
        return fmean

    def interpolate_sample(self, *argv, sample=None, nodes=None, reg_model=None, corr_model=None, n_opt=1):

        """
        interpolate_sample: interpolate U, Sigma, and V.
        :param sample: list or numpy ndarray with the coordinates of the point being interpolated.
        :param nodes: list or numpy ndarray with the coordinates of the nodes of the element.
        :param argv: list or numpy ndarray containing the solution matrix assigned to each node.
        :param reg_model: str with the used regression method (linear_interp or kriging_interp).
        :param corr_model: Correlation model contains the correlation function, which uses sample distance
                    to define similarity between samples.
                    Options: Exponential, Gaussian, Linear, Spherical, Cubic, Spline.
        :param corr_model_params: Initial values corresponding to hyperparameters/scale parameters.
        :param n_opt: int with the number of times optimization problem is to be solved with different starting point.
               Default: 1
        """

        matrix, nargs = check_arguments(argv, min_num_matrix=3, ortho=False)

        if self.user_interp_check:
            exec('from ' + self.interp_script[:-3] + ' import ' + self.interp_object)
            interp_fun = eval(self.interp_object)
        else:
            interp_fun = eval("Grassmann." + self.interp_object)

        shape_ref = np.shape(matrix[0])
        for i in range(1, nargs):
            if np.shape(matrix[i]) != shape_ref:
                raise TypeError('Input matrices have different shape.')

        # Test if the sample is stored as a list
        if type(sample) == list:
            sample = np.array(sample)

        # Test if the nodes are stored as a list
        if type(nodes) == list:
            nodes = np.array(nodes)

        interp_matrix = interp_fun(nodes, matrix, sample, reg_model, corr_model, n_opt)

        return interp_matrix

    # ==================================================================================================================
    # The pre-defined interpolators are implemented in this section. Any new pre-defined interpolator must be
    # implemented here with the decorator @staticmethod.

    # Linear interpolation
    @staticmethod
    def linear_interp(*argv):

        nodes = argv[0]
        matrix = argv[1]
        sample = argv[2]
        myInterpolator = LinearNDInterpolator(nodes, matrix)
        interp_matrix = myInterpolator(sample)
        interp_matrix = interp_matrix[0]

        return interp_matrix

    # Kringing interpolation
    @staticmethod
    def kriging_interp(*argv):

        if len(argv) < 6:
            raise ValueError('Six arguments are expected.')

        nodes = argv[0]
        matrix = argv[1]
        sample = argv[2]

        reg_model = argv[3]
        corr_model = argv[4]
        n_opt = argv[5]

        if reg_model is None:
            raise ValueError('reg_model is missing.')

        if corr_model is None:
            raise ValueError('corr_model is missing.')

        if n_opt is None:
            raise ValueError('n_opt is missing.')

        if not isinstance(n_opt, int):
            raise ValueError('n_opt must be integer.')
        else:
            if n_opt < 1:
                raise ValueError('n_opt must be larger than or equal to 1.')

        shape_ref = np.shape(matrix[0])
        interp_matrix = np.zeros(shape_ref)
        nargs = len(matrix)
        nrows = matrix[0].shape[0]
        ncols = matrix[0].shape[1]

        val_matrix = []
        # to_interp = np.zeros(nargs)
        for j in range(nrows):
            for k in range(ncols):
                for i in range(nargs):
                    # to_interp[i] = [matrix[i][j,k]]
                    val_matrix.append([matrix[i][j, k]])

                if val_matrix.count(val_matrix[0]) == len(val_matrix):
                    val = np.array(val_matrix)
                    y = val[0]
                else:
                    val = np.array(val_matrix)
                    K = Krig(samples=nodes, values=val, reg_model=reg_model, corr_model=corr_model, n_opt=n_opt)
                    y, mse = K.interpolate(sample, dy=True)

                val_matrix.clear()

                interp_matrix[j, k] = y

        return interp_matrix

# ========================= DIFFUSION MAPS ===========================================
class DiffusionMaps:
    """

    Description:

        DiffusionMaps is a class containing the methods to perform the diffusion maps based on the input dataset.

        References:
        Ronald R. Coifman, Stéphane Lafon, "Diffusion maps", Applied and Computational Harmonic Analysis, Volume 21,
        Issue 1, Pages 5-30, 2006.

        Nadler, B., Lafon, S., Coifman, R., and Kevrekidis, I., "Diffusion maps, spectral clustering
        and eigenfunctions of Fokker-Planck operators", In Y. Weiss, B. Scholkopf, and J. Platt (Eds.), Advances in Neural
        Information Processing Systems, 18, pages 955 – 962, 2006, Cambridge, MA: MIT Press

    Input:

        :param alpha: Assumes a value between 0 and 1 (default values is 0.5) and corresponding to different diffusion
                      operators.
        :type alpha: float

        :param n_evecs: the number of eigenvectors and eigenvalues used in the representation of the diffusion coordinates.
        :type n_evecs: int

        :param sparse: Is a boolean variable defining the sparsity of the graph generated when data is provided to estimate the
                       diffusion coordinates using the standard approach.
        :type sparse: bool

        :param k_neighbors: Used when sparse is True to define the number of samples close to a given sample
                            used in the construction of the affinity matrix.
        :type k_neighbors: int

    """

    # Authors: Ketson R. M. dos Santos, Dimitris G. Giovanis
    # Updated: 03/04/20 by Ketson R. M. dos Santos

    def __init__(self, alpha=0.5, n_evecs=2, sparse=False, k_neighbors=1):
        self.alpha = alpha
        self.n_evecs = n_evecs
        self.sparse = sparse
        self.k_neighbors = k_neighbors
        
        if alpha < 0 or alpha > 1:
                    raise ValueError('alpha should be a value between 0 and 1.')
        
        if isinstance(n_evecs, int):
            if n_evecs<1:
                raise ValueError('n_evecs should be larger than or equal to one.')
        else:
            raise TypeError('n_evecs should be integer.')

        if not isinstance(sparse, bool):
            raise TypeError('sparse should be a boolean variable.')
        elif sparse is True:    
            if isinstance(k_neighbors, int):
                if k_neighbors<1:
                    raise ValueError('k_neighbors should be larger than or equal to one.')
            else:
                raise TypeError('k_neighbors should be integer.')
            
    #@staticmethod
    #def mapping(self, data=None, epsilon=None, kernel_mat=None):
    def mapping(self, **kwargs):

        """
        mapping: compute the diffusion cordinates when either the kernel_matrix of the data are provided.
        :param kwargs: input arguments used in the diffusion maps technique.
        """
        
        if 'data' in kwargs.keys():
            data = kwargs['data']
        else:
            data = None
            
        if 'epsilon' in kwargs.keys():
            epsilon = kwargs['epsilon']
        else:
            epsilon = None
            
        if 'kernel_mat' in kwargs.keys():
            kernel_mat = kwargs['kernel_mat']
        else:
            kernel_mat = None
        
        alpha = self.alpha
        n_evecs = self.n_evecs
        sparse = self.sparse
        k_neighbors = self.k_neighbors

        if data is None and kernel_mat is None:
            raise ValueError('data and kernel_mat both are None.')

        if kernel_mat is not None:
            if np.shape(kernel_mat)[0] != np.shape(kernel_mat)[1]:
                raise ValueError('kernel_mat is not a square matrix.')
            else:
                if n_evecs > max(np.shape(kernel_mat)):
                    raise ValueError('n_evecs is larger than the size of kernel_mat.')

        if data is None:
            N = np.shape(kernel_mat)[0]
        else:
            N = np.shape(data)[0]

        # Construct the Kernel matrix if no Kernel matrix is provided.
        #k_matrix, epsilon = DiffusionMaps.create_kernel_matrix(self, data, epsilon, kernel_mat=kernel_mat)
        if kernel_mat is None:
            k_matrix = DiffusionMaps.create_kernel_matrix(self, data, epsilon)
        else:
            k_matrix = kernel_mat

        # Compute the diagonal matrix D(i,i) = sum(Kernel(i,j)^alpha,j) and its inverse.
        D, D_inv = DiffusionMaps.D_matrix(k_matrix, alpha)

        # Compute L^alpha = D^(-alpha)*L*D^(-alpha).
        Lstar = DiffusionMaps.l_alpha_normalize(self, k_matrix, D_inv)

        Dstar, Dstar_inv = DiffusionMaps.D_matrix(Lstar, 1.0)
        if sparse:
            Dstar_invd = sps.spdiags(Dstar_inv, 0, Dstar_inv.shape[0], Dstar_inv.shape[0])
        else:
            Dstar_invd = np.diag(Dstar_inv)
            
        Ps = Dstar_invd.dot(Lstar)

        # Find the eigenvalues and eigenvectors of Ps.
        if sparse:
            evals, evecs = spsl.eigs(Ps, k=(n_evecs + 1), which='LR')
        else:
            evals, evecs = np.linalg.eig(Ps)

        ix = np.argsort(np.abs(evals))
        ix = ix[::-1]
        s = np.real(evals[ix])
        u = np.real(evecs[:, ix])

        evals = s[:n_evecs]
        evecs = u[:, :n_evecs]

        dcoords = np.zeros([N, n_evecs])
        for i in range(n_evecs):
            dcoords[:, i] = evals[i]* evecs[:, i]

        return dcoords, evals, evecs

    #@staticmethod
    def create_kernel_matrix(self, data, epsilon=1):

        """
        create_kernel_matrix: Compute the kernel matrix.
        :param data: input data
        :param epsilon: real value containing the epsilon used in the exponential kernel (if no kernel matrix is provided).
        sparse: boolean variable to define if a sparse representation of the graph is of interest (it can improve the computational performance).
        """
        
        sparse = self.sparse
        k_neighbors = self.k_neighbors

        # Compute the pairwise distances.
        if len(np.shape(data)) == 2:
            dist_pairs = sd.pdist(data, 'euclidean')
        elif len(np.shape(data)) == 3:

            # Check arguments: verify the consistency of input arguments.
            datam, nargs = check_arguments(data, min_num_matrix=2, ortho=False)

            indices = range(nargs)
            pairs = list(itertools.combinations(indices, 2))

            dist_pairs = []
            for id_pair in range(np.shape(pairs)[0]):
                ii = pairs[id_pair][0]  # Point i
                jj = pairs[id_pair][1]  # Point j

                x0 = datam[ii]
                x1 = datam[jj]

                dist = np.linalg.norm(x0 - x1, 'fro')

                dist_pairs.append(dist)
        else:
            raise TypeError('The size of the input data is not adequate.')

        # Find a suitable episilon if it is not provided by the user.
        if epsilon is None:
            epsilon = DiffusionMaps.find_epsilon(dist_pairs)

        kernel_mat = np.exp(-sd.squareform(dist_pairs) ** 2 / (4 * epsilon))

        # If the user prefer to use a sparse graph.
        if sparse:

            nrows = np.shape(kernel_mat)[0]

            for i in range(nrows):
                vec = kernel_mat[i, :]
                idx = nn_coord(vec, k_neighbors)
                kernel_mat[i, idx] = 0

            kernel_mat = sps.csc_matrix(kernel_mat)

        return kernel_mat

    @staticmethod
    def find_epsilon(dist_pairs):

        """
        find_epsilon: Find a suitable episilon based on the median of the square value of the pairwise distances.
        :param dist_pairs: list of numpy ndarray containing the pairwise distances.
        """

        dist_pairs_sq = np.array(dist_pairs) ** 2
        epsilon = np.median(dist_pairs_sq)

        return epsilon

    @staticmethod
    def D_matrix(kernel_mat, alpha):

        """
        D_matrix: Compute the diagonal matrix D(i,i) = sum(Kernel(i,j)^alpha,j) and its inverse.
        :param kernel_mat: list of numpy ndarray containing the kernel matrix.
        :param alpha:real constant defining the diffusion operator - alpha = 0 (Graph Laplacian normalization), 0.5 (Fokker-Plank), 1 (Laplace-Beltrami).
        """

        m = np.shape(kernel_mat)[0]
        kmat = kernel_mat
        D = np.array(kmat.sum(axis=1)).flatten()

        D_inv = np.power(D, -alpha)
        return D, D_inv

    #@staticmethod
    def l_alpha_normalize(self, kernel_mat, D_inv):

        """
        lr_normalize
        :param kernel_mat: list of numpy ndarray containing the kernel matrix.
        :param D_inv: inverse of D(i,i) = sum(Kernel(i,j)^alpha,j)
        """

        sparse = self.sparse
        m = D_inv.shape[0]

        if sparse:
            Dalpha = sps.spdiags(D_inv, 0, m, m)
        else:
            Dalpha = np.diag(D_inv)

        Ps = Dalpha.dot(kernel_mat.dot(Dalpha))

        return Ps


