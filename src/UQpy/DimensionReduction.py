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
from fbpca import pca
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
            as well as to project points onto the Grassmann manifold and the tangent space. It also perform the
            interpolation of points on the Grassmann manifold projecting then onto the tangent space and to interpolate
            them employing standard methods.

            References:
            Jiayao Zhang, Guangxu Zhu, Robert W. Heath Jr., and Kaibin Huang, "Grassmannian Learning: Embedding Geometry
            Awareness in Shallow and Deep Learning", arXiv:1808.02229 [cs, eess, math, stat], Aug. 2018.

            D.G. Giovanis, M.D. Shields, "Uncertainty quantification for complex systems with very high dimensional
            response using Grassmann manifold variations",
            Journal of Computational Physics, Volume 364, Pages 393-415, 2018.

        Input:

            :param dist_object: It specifies the name of the function or class with the distance metric.
            :type dist_object: str

            :param dist_script: The filename (with extension) of the Python script containing the implementation
                                of dist_object (only for user defined metrics).
            :type dist_script: str

            :param interp_object: It specifies the name of the function or class with the employed interpolator.
            :type interp_object: str

            :param interp_script: The filename (with extension) of the Python script containing the implementation
                                  of interp_object (only for user defined interpolator).
            :type interp_script: str

        """

    # Authors: Ketson R. M. dos Santos
    # Updated: 10/30/19 by Ketson R. M. dos Santos

    def __init__(self, dist_object=None, dist_script=None, kernel_object=None, kernel_script=None, interp_object=None,
                 interp_script=None):

        # Distance
        self.dist_script = dist_script
        self.dist_object = dist_object
        if dist_script is not None:
            self.user_dist_check = path.exists(dist_script)
        else:
            self.user_dist_check = False

        if self.user_dist_check:
            try:
                # import user_distance
                self.module_dist = __import__(self.dist_script[:-3])
            except ImportError:
                raise ImportError('There is no module implementing a  metric on the Grassmann manifold.')

        # Kernels
        self.kernel_script = kernel_script
        self.kernel_object = kernel_object
        if kernel_script is not None:
            self.user_kernel_check = path.exists(kernel_script)
        else:
            self.user_kernel_check = False

        if self.user_kernel_check:
            try:
                # import user_distance
                self.module_kernel = __import__(self.kernel_script[:-3])
            except ImportError:
                raise ImportError('There is no module implementing a kernel.')

        # Interpolation
        self.interp_script = interp_script
        self.interp_object = interp_object
        if interp_script is not None:
            self.user_interp_check = path.exists(interp_script)
            # self.user_dist_check = path.exists("user_distance.py")
        else:
            self.user_interp_check = False

        if self.user_interp_check:
            try:
                # import user_distance
                # self.user_distance = user_distance.user_dist
                self.module_interp = __import__(self.interp_script[:-3])
                # self.user_distance = self.module_dist.user_dist
            except ImportError:
                raise ImportError('There is no module implementing the distance metric.')
        # pass

    def distance(self, *argv, **kwargs):

        """
        GrassmannDistance: Estimate the distance of points in the Grassmann manifold
                           using different metrics.
        argv: list or numpy ndarray containing all the matrices (at least 2) corresponding to the posints
              in the Grassmann manifold
        """

        # vari is a list containing each point, and nargs is the number of points
        # Check arguments: verify the consistency of input arguments
        nargs = len(argv[0])
        psi = argv[0]
        #psi, nargs = check_arguments(argv, min_num_matrix=2, ortho=False)
        
        ranks = None
        cont = 0
        for key, value in kwargs.items():
            if key == "rank":
                ranks = value
                break

            cont += 1

        # If rank is not provide compute internally 
        if ranks is None:
            ranks = []
            for i in range(nargs):
                ranks.append(np.linalg.matrix_rank(psi[i]))
       
        if type(ranks) != list and type(ranks) != np.ndarray:
            raise TypeError('rank MUST be either a list or ndarray.')


        # Define the pairs of points to compute the Grassmann distance
        indices = range(nargs)
        pairs = list(itertools.combinations(indices, 2))

        if self.user_dist_check:
            exec('from ' + self.dist_script[:-3] + ' import ' + self.dist_object)
            dist_fun = eval(self.dist_object)
        else:
            dist_fun = eval("Grassmann." + self.dist_object)

        distances_list = []
        for id_pair in range(np.shape(pairs)[0]):
            ii = pairs[id_pair][0]  # Point i
            jj = pairs[id_pair][1]  # Point j

            #rank = min(x0.shape[1], x1.shape[1])
            #x0 = x0[:, :rank]  # Truncate matrices with rank
            #x1 = x1[:, :rank]
            #rank0 = min(x0.shape[1], x0.shape[1])
            #rank1 = min(x1.shape[1], x1.shape[1])
            rank0 = int(ranks[ii])
            rank1 = int(ranks[jj])
            
            x0 = np.asarray(psi[ii])[:, :rank0]
            x1 = np.asarray(psi[jj])[:, :rank1]
            
            dist = dist_fun(x0, x1)

            distances_list.append(dist)

        return distances_list

    # ==================================================================================================================
    # The pre-defined metrics are implemented in this section. Any new pre-defined metric must be implemented
    # here with the decorator @staticmethod.
    @staticmethod
    def grassmann_dist(x0, x1):

        l = min(np.shape(x0))
        k = min(np.shape(x1))
        rank = min(l, k)

        r = np.dot(x0.T, x1)
        (ui, si, vi) = pca(r, rank, True)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(si)
        dist = np.sqrt(abs(k - l) * np.pi ** 2 / 4 + np.sum(theta ** 2))

        return dist

    @staticmethod
    def chordal_dist(x0, x1):

        l = min(np.shape(x0))
        k = min(np.shape(x1))
        rank = min(l, k)

        r_star = np.dot(x0.T, x1)
        (ui, si, vi) = pca(r_star, rank, True)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(si)
        theta = (np.sin(theta)) ** 2
        dist = np.sqrt(abs(k - l) + np.sum(theta))

        return dist

    @staticmethod
    def procrustes_dist(x0, x1):

        l = min(np.shape(x0))
        k = min(np.shape(x1))
        rank = min(l, k)

        r = np.dot(x0.T, x1)
        (ui, si, vi) = pca(r, rank, True)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(si)
        theta = np.sin(theta / 2) ** 2
        dist = np.sqrt(abs(k - l) + 2 * np.sum(theta))

        return dist

    @staticmethod
    def projection_dist(x0, x1):

        # Check rank and swap
        c = np.zeros([x0.shape[0], x0.shape[1]])
        if x0.shape[1] < x1.shape[1]:
            x0 = x1
            x1 = c

        # Compute the projection according to[1].
        r = np.dot(x0.T, x1)
        x1 = x1 - np.dot(x0, r)

        # Make sure it's magnitude is less than 1.
        dist = np.arcsin(min(1, scipy.linalg.norm(x1)))

        return dist

    # ==================================================================================================================
    # The pre-defined kernels are implemented in this section. Any new pre-defined kernel must be implemented
    # here with the decorator @staticmethod.

    def kernel(self, *argv, **kwargs):

        """
        GrassmannDistance: Estimate the distance of points in the Grassmann manifold
                           using different metrics.
        argv: list or numpy ndarray containing all the matrices (at least 2) corresponding to the posints
              in the Grassmann manifold
        """
        # vari is a list containing each point, and nargs is the number of points
        # Check arguments: verify the consistency of input arguments
        nargs = len(argv[0])
        psi = argv[0]
        #psi, nargs = check_arguments(argv, min_num_matrix=2, ortho=False)
        
        ranks = None
        cont = 0
        for key, value in kwargs.items():
            if key == "rank":
                ranks = value
                break

            cont += 1

        # If rank is not provide compute internally 
        if ranks is None:
            ranks = []
            for i in range(nargs):
                ranks.append(np.linalg.matrix_rank(psi[i]))
       
        if type(ranks) != list and type(ranks) != np.ndarray:
            raise TypeError('rank MUST be either a list or ndarray.')

        # Define the pairs of points to compute the Grassmann distance
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

            #rank = min(x0.shape[1], x1.shape[1])
            #rank0 = min(x0.shape[1], x0.shape[1])
            #rank1 = min(x1.shape[1], x1.shape[1])
            rank0 = ranks[ii]
            rank1 = ranks[jj]
            x0 = x0[:, :rank0]  # Truncating matrices
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
            
            
        kernel_matrix = sd.squareform(kernel_list)+np.diag(kernel_diag)
        
        return kernel_matrix

    @staticmethod
    def projection_kernel(x0, x1):

        r = np.dot(x0.T, x1)
        n = np.linalg.norm(r, 'fro')
        ker = n * n
        return ker

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
        project_points: project the input matrices on the Grassmann manifold via
                        singular value decomposition (SVD)
        argv: list or numpy nadarray containing the matrices to be projected onto the Grassmann manifold.
        kwargs: contains the keyword for the rank. If a list or numpy ndarray containing the rank of each matrix is not
                provided, the code compute them using numpy.linalg.matrix_rank.
        """

        # matrix is a list containing each point, and nargs is the number of points
        # Check arguments: verify the consistency of input arguments
        matrix, nargs = check_arguments(argv, min_num_matrix=1, ortho=False)

        ranks = None
        cont = 0
        for key, value in kwargs.items():
            if key == "rank":
                ranks = value
                break

            cont += 1

        # If rank is not provide compute internally 
        if ranks is None:
            ranks = []
            for i in range(nargs):
                ranks.append(np.linalg.matrix_rank(matrix[i]))
       
        if type(ranks) != list and type(ranks) != np.ndarray:
            raise TypeError('rank MUST be either a list or ndarray.')

        max_rank = int(np.max(ranks))  # rank = max(r1, r2, ..., rn)
        psi = []  # initialize the left singular vectors as a list
        sigma = []  # initialize the singular values as a list
        phi = []  # initialize the right singular vectors as a list

        # For each point perform svd with max_rank columns
        for i in range(nargs):
            u, s, v = svd(matrix[i], max_rank) #Old version
            #(u, s, v) = pca(matrix[i], max_rank, True)
            #(u, s, v) = pca(matrix[i], ranks[i], True)
            psi.append(u)
            sigma.append(np.diag(s))
            phi.append(v.T)

        return psi, sigma, phi, max_rank, ranks

    @staticmethod
    def log_mapping(*argv, ref=None):
        """
        log_mapping: projecting points on the Grassmann manifold onto the tangent space.
        
        argv: list or numpy ndarray containing all the points in the Grassmann manifold
        ref: list or numpy ndarray containing the reference point in the Grassmann manifold
             where the tangent space is approximated.
        """
        # Check the input arguments
        u, nr = check_arguments(argv, min_num_matrix=1, ortho=False)

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
                # m1 = np.dot(u[i], np.dot(refT, u[i]))
                # m = m1 - np.dot(m0, m1)
                minv = np.linalg.inv(np.dot(refT,u[i]))
                m = np.dot(u[i] - np.dot(m0, u[i]),minv)
                ui, si, vi = np.linalg.svd(m, full_matrices=False)              #svd(m, max_rank)
                #(ui, si, vi) = pca(m, rank[i], True)
                _gamma.append(np.dot(np.dot(ui, np.diag(np.arctan(si))), vi))

        return _gamma

    @staticmethod
    def exp_mapping(*argv, ref=None):

        """
        exp_mapping: perform the exponential mapping from the tangent space
                     to the Grassmann manifold.
        ref: list or ndarray containing the reference point in the Grassmann manifold 
             where the tangent space is approximated.
        
        """
        # sample, nodes, matrix
        u, nr = check_arguments(argv, min_num_matrix=1, ortho=False)

        rank = []
        for i in range(nr):
            rank.append(min(np.shape(u[i])))

        x = list()
        for j in range(nr):

            ui, si, vi = np.linalg.svd(u[j], full_matrices=False)
            #(ui, si, vi) = pca(u[j], rank[j], True)

            # Exponential mapping
            x0 = np.dot(np.dot(np.dot(ref, vi.T), np.diag(np.cos(si))) + np.dot(ui, np.diag(np.sin(si))), vi)

            # Test orthogonality
            xtest = np.dot(x0.T, x0)

            if not np.allclose(xtest, np.identity(np.shape(xtest)[0])):
                x0, unused = np.linalg.qr(x0)  # re-orthonormalizing

            x.append(x0)

        return x

    # @staticmethod
    def karcher_mean(self, *argv, acc=False, tol=1e-3, maxiter=1000):

        # np.random.seed( 234 )
        matrix, n_mat = check_arguments(argv, min_num_matrix=2, ortho=False)

        alpha = 0.5  # todo: this can be adaptive (e.g., Armijo rule).
        rnk = []
        for i in range(n_mat):
            rnk.append(min(np.shape(matrix[i])))

        max_rank = np.max(rnk)
        # max_rank = max(rank)

        fmean = []
        for i in range(n_mat):
            fmean.append(self.frechet_mean(matrix[i], matrix))

        index_0 = fmean.index(min(fmean))
        # index_0 = np.random.randint(0, n_mat)
        mean_element = list()
        mean_element = matrix[index_0].tolist()
        # nrow = np.shape(mean_element)[0]

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

        print(itera)
        return mean_element, _gamma

    # @staticmethod
    def karcher_mean_sgd(self, *argv, tol=1e-3, maxiter=1000):

        matrix, n_mat = check_arguments(argv, min_num_matrix=2, ortho=False)

        rnk = []
        for i in range(n_mat):
            rnk.append(min(np.shape(matrix[i])))

        max_rank = np.max(rnk)

        fmean = []
        for i in range(n_mat):
            fmean.append(self.frechet_mean(matrix[i], matrix))

        index_0 = fmean.index(min(fmean))
        # index_0 = np.random.randint(0, n_mat)

        mean_element = matrix[index_0].tolist()
        itera = 0
        _gamma = []
        k = 1
        while itera < maxiter:

            # indices = np.random.choice(n_mat, 7)
            indices = np.arange(n_mat)
            np.random.shuffle(indices)

            melem = mean_element
            for i in range(len(indices)):
                alpha = 0.5 / k
                # idx = np.random.randint(0, n_mat)
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

        print(itera)
        return mean_element, _gamma

    # @staticmethod
    def frechet_mean(self, k_mean, *argv):

        matrix, n_mat = check_arguments(argv, min_num_matrix=2, ortho=False)

        accum = 0
        for i in range(n_mat):
            # D = Grassmann.distance(k_mean, matrix[i], metric=metric)
            d = self.distance([k_mean, matrix[i]])
            accum += d[0] * d[0]

        fmean = accum / n_mat
        return fmean

    def interpolate_sample(self, *argv, sample=None, nodes=None, reg_model=None, corr_model=None, n_opt=1):

        """
        interpolate_sample: interpolate U, Sigma, and V using the volumes of sub-simplexes.
        sample: list or numpy ndarray with the coordinates of the point being interpolated.
        nodes: list or numpy ndarray with the coordinates of the nodes of the element.
        argv: list or numpy ndarray containing the solution matrix assigned to each node.
        corr_model: Correlation model contains the correlation function, which uses sample distance
                    to define similarity between samples.
                    Options: Exponential, Gaussian, Linear, Spherical, Cubic, Spline.
        corr_model_params: Initial values corresponding to hyperparameters/scale parameters.
        n_opt: Number of times optimization problem is to be solved with different starting point.
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

        # Test if the sample lies within the element
        #isinside = inelement(sample, nodes)
        #if isinside:

        interp_matrix = interp_fun(nodes, matrix, sample, reg_model, corr_model, n_opt)

        #else:
        #    raise TypeError('The sample MUST be within the element.')

        return interp_matrix

    # ==================================================================================================================
    # The pre-defined interpolators are implemented in this section. Any new pre-defined interpolator must be
    # implemented here with the decorator @staticmethod.

    @staticmethod
    def linear_interp(*argv):

        nodes = argv[0]
        matrix = argv[1]
        sample = argv[2]
        myInterpolator = LinearNDInterpolator(nodes, matrix)
        interp_matrix = myInterpolator(sample)
        interp_matrix = interp_matrix[0]

        return interp_matrix

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

    # ==================================================================================================================

    def exp_mapping_interp(self, sample, nodes, ref, *argv):

        """
        exp_mapping_interp: perform the exponential mapping of the interpolated point
                            from the tangent space to the Grassmann manifold.
        sample: list or ndarray with the coordinates of the point being interpolated.
        nodes: list or ndarray with the coordinates of the nodes of the element.
        ref: list or ndarray containing the reference point in the Grassmann manifold
             where the tangent space is approximated.
        argv: list or ndarray containing the solution matrix assigned to each node.

        """

        matrix, n_mat = check_arguments(argv, min_num_matrix=3, ortho=False)

        # Test if the sample is stored as a list
        if type(sample) == list:
            sample = np.asarray(sample)

        # Test if the nodes are stored as a list
        if type(nodes) == list:
            nodes = np.asarray(nodes)

        # Test if the sample lies within the element
        isinside = inelement(sample, nodes)
        if isinside:

            # interp_matrix = Grassmann.interpolate_points(point, nodes, matrix)
            interp_matrix = Grassmann(interp_object=self.interp_object,
                                      interp_script=self.interp_script).interpolate_sample(matrix, sample=sample,
                                                                                           nodes=nodes)
            # interp_matrix = Grassmann.interpolate_sample(matrix, interpolator=interpolator, sample=sample, nodes=nodes)
            # ui, si, vi = np.linalg.svd(interp_matrix, full_matrices=False)
            rank = min(np.shape(interp_matrix))
            (ui, si, vi) = pca(interp_matrix, rank, True)

            # Exponential mapping
            x = np.dot(np.dot(np.dot(ref, vi.T), np.diag(np.cos(si))) + np.dot(ui, np.diag(np.sin(si))), vi)

            # Test orthogonality
            xtest = np.dot(x.T, x)
            if not np.allclose(xtest, np.identity(np.shape(xtest)[0])):
                x, unused = np.linalg.qr(x)  # re-orthonormalizing

        else:
            raise TypeError('The sample MUST be within the element.')

        return x


# ========================= DIFFUSION MAPS ===========================================
class DiffusionMaps:

    def __init__(self):
        pass

    @staticmethod
    def mapping(data=None, alpha=0.5, n_evecs=2, epsilon=None, kernel_mat=None, sparse=False, k_neighbors=1):
        
        if data is None and kernel_mat is None:
            raise TypeError('data and kernel_mat both are None.')
         
        if kernel_mat is not None:
            if np.shape(kernel_mat)[0] != np.shape(kernel_mat)[1]:
                raise TypeError('kernel_mat is not a square matrix.')
            else: 
                if n_evecs > max(np.shape(kernel_mat)): 
                    raise TypeError('n_evecs is larger than the size of kernel_mat.')
                
            
        if data is None:
            N=np.shape(kernel_mat)[0]
        else:
            N = np.shape(data)[0]

        k_matrix, epsilon = DiffusionMaps.create_kernel_matrix(data, epsilon, sparse=sparse, kernel_mat=kernel_mat, k_neighbors=k_neighbors)

        b, b_inv = DiffusionMaps.b_matrix(k_matrix, alpha)

        Ps = DiffusionMaps.lr_normalize(k_matrix, b_inv, sparse)
        
        #Ps0 = sps.csr_matrix(Ps)
        #u, s, v = spsl.svds(Ps0, k=n_evecs + 1)
        #s=np.flip(s)
        #u=np.fliplr(u)

        #u, s, v = np.linalg.svd(Ps, full_matrices=False)
        
        if sparse:
            evals, evecs = spsl.eigs(Ps, k=(n_evecs+1), which='LR')
        else:
            evals, evecs = np.linalg.eig(Ps)
            
        #Ps0 = sps.csr_matrix(Ps)
        #evals, evecs = spsl.eigs(Ps0, k=(n_evecs+1), which='LR')

        ix = np.argsort(np.abs(evals))
        ix = ix[::-1]
        s = np.real(evals[ix])
        u = np.real(evecs[:,ix])

        evals = s[:n_evecs]
        Pfia = u[:, :n_evecs]
        b_inv_ = np.diag(b_inv)
        evecs = np.dot(b_inv_, Pfia)
        # A = np.dot(np.dot(psa.T, b), psa)
        dcoords = np.zeros([N, n_evecs])
        for i in range(n_evecs):
            dcoords[:, i] = evals[i] * evecs[:, i]

        return dcoords, evals, evecs
    
    @staticmethod
    def create_kernel_matrix(data, epsilon, sparse=False, kernel_mat=None, k_neighbors=1):
 
        if kernel_mat is None and data is not None:
  
            if len(np.shape(data)) == 2:
                dist_pairs = sd.pdist(data, 'euclidean')
            elif len(np.shape(data)) == 3:

                # Check arguments: verify the consistency of input arguments
                datam, nargs = check_arguments(data, min_num_matrix=2, ortho=False)

                indices = range(nargs)
                pairs = list(itertools.combinations(indices, 2))

                dist_pairs = []
                for id_pair in range(np.shape(pairs)[0]):
                    ii = pairs[id_pair][0]  # Point i
                    jj = pairs[id_pair][1]  # Point j

                    x0 = datam[ii]
                    x1 = datam[jj]

                    dist = np.linalg.norm(x0-x1,'fro')

                    dist_pairs.append(dist)
            else:
                raise TypeError('The size of the input data is not adequate.')
            
            if epsilon is None:
                epsilon = DiffusionMaps.find_epsilon(dist_pairs)
            
            #print(np.std(dist_pairs), np.mean(dist_pairs))
            kernel_mat = np.exp(-sd.squareform(dist_pairs) ** 2 / (4 * epsilon))
            
        if sparse:

            nrows = np.shape(kernel_mat)[0]

            for i in range(nrows):
                vec = kernel_mat[i,:]
                idx = nn_coord(vec, k_neighbors)
                kernel_mat[i,idx]=0

            kernel_mat = sps.csc_matrix(kernel_mat)
        
        #plt.figure()
        #plt.matshow(kernel_mat)
        #plt.show()

        return kernel_mat, epsilon
    
    @staticmethod
    def find_epsilon(dist_pairs):
        
        dist_pairs_sq = dist_pairs**2
        epsilon = np.median(dist_pairs_sq)
        return epsilon

    @staticmethod
    def b_matrix(kernel_matrix, alpha):
        m = np.shape(kernel_matrix)[0]
        kmat = kernel_matrix  # todo .toarray()
        b = np.array(kmat.sum(axis=1)).flatten()

        b_inv = np.power(b, -alpha)
        return b, b_inv

    @staticmethod
    def lr_normalize(kernel_matrix, b_inv, sparse=False):
        m = b_inv.shape[0]
        # Dalpha = sps.spdiags(b_inv, 0, m, m)
        # kernel_matrix = kernel_matrix * Dalpha
        # Ps = Dalpha * kernel_matrix
        
        if sparse:
            Dalpha = sps.spdiags(b_inv, 0, m, m)
        else:
            Dalpha = np.diag(b_inv)
            
        Ps = Dalpha.dot(kernel_matrix.dot(Dalpha))
        """
        row_sum = kernel_matrix.sum(axis=1).transpose()
        n = row_sum.shape[1]
        Dalpha = sps.spdiags(np.power(row_sum, -1), 0, n, n)
        P = Dalpha * kernel_matrix
        """
        return Ps
