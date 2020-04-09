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

"""
This module contains the classes and methods to perform the point wise and multi point data-based dimensionality
reduction via projection onto the Grassmann manifold and Diffusion Maps, respectively. Further, interpolation in the
tangent space centered at a given point on the Grassmann manifold can be performed.

* Grassmann: This class contains methods to perform the projection of matrices onto the Grassmann manifold where their
  dimensionality are reduced and where standard interpolation can be performed on a tangent space.
* DiffusionMaps: In this class the diffusion maps create a connection between the spectral properties of the diffusion
  process and the intrinsic geometry of the data resulting in a multiscale representation of the data.
"""

from UQpy.Surrogates import Krig
from UQpy.Utilities import *
import scipy as sp
import numpy as np
import itertools
from scipy.interpolate import LinearNDInterpolator
from os import path
import math

import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.spatial.distance as sd


########################################################################################################################
########################################################################################################################
#                                            Grassmann Manifold                                                        #
########################################################################################################################
########################################################################################################################


class Grassmann:
    """
    Project matrices onto the Grassmann manifold and create a tangent space where standard interpolation is performed.

    This class contains methods to perform the projection of matrices onto the Grassmann manifold via singular value
    decomposition(SVD) where their dimensionality are reduced. Further, the mapping from the Grassmannian to a tangent
    space centered at a given reference point (exponential mapping) as well as the mapping from the tangent space to the
    manifold (logarithmic mapping). Moreover, a interpolation can be performed in the tangent space. Further, additional
    quantities such as the Karcher mean, the distance on the manifold, and the kernel defined on the Grassmann manifold
    can be obtained.

    **References:**

    1. Jiayao Zhang, Guangxu Zhu, Robert W. Heath Jr., and Kaibin Huang, "Grassmannian Learning: Embedding Geometry
       Awareness in Shallow and Deep Learning", arXiv:1808.02229 [cs, eess, math, stat], Aug. 2018.

    2. D.G. Giovanis, M.D. Shields, "Uncertainty quantification for complex systems with very high dimensional response
       using Grassmann manifold variations", Journal of Computational Physics, Volume 364, Pages 393-415, 2018.

    **Input:**

    :param distance_object: It specifies the name of a function or class implementing the distance on the manifold.

                            Default: None
    :type distance_object: str

    :param distance_script: The filename (with extension) of a Python script implementing dist_object
                            (only for user defined metrics).

                            Default: None
    :type distance_script: str

    :param kernel_object: It specifies the name of a function or class implementing the Grassmann kernel.
                          Default: None
    :type kernel_object: str

    :param kernel_script: The filename (with extension) of a Python script implementing kernel_object
                          (only for user defined metrics).

                          Default: None
    :type distance_script: str

    :param interp_object: It specifies the name of the function or class implementing the interpolator.

                          Default: None
    :type interp_object: str

    :param interp_script: The filename (with extension) of the Python script implementing of interp_object
                          (only for user defined interpolator).

                          Default: None
    :type interp_script: str

    **Authors:**

    Ketson R. M. dos Santos, Dimitris G. Giovanis

    Last modified: 03/26/20 by Ketson R. M. dos Santos
    """

    def __init__(self, distance_object=None, distance_script=None, kernel_object=None, kernel_script=None,
                 interp_object=None,
                 interp_script=None, karcher_object=None, karcher_script=None):

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
                raise ImportError('There is no module implementing a distance.')

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
                raise ImportError('There is no module implementing the interpolation.')

        # Karcher mean.
        self.karcher_script = karcher_script
        self.karcher_object = karcher_object
        if karcher_script is not None:
            self.user_karcher_check = path.exists(karcher_script)
        else:
            self.user_karcher_check = False

        if self.user_karcher_check:
            try:
                self.module_karcher = __import__(self.karcher_script[:-3])
            except ImportError:
                raise ImportError('There is no module implementing an optimizer to find the Karcher mean.')

    # Calculate the distance on the manifold
    def distance(self, *argv, **kwargs):

        """
        Estimate the distance between points on the Grassmann manifold.

        This method computes the pairwise distance of points projected on the Grassmann manifold. The input arguments
        are passed through a list of arguments (argv) containing a list of lists or a list of numpy arrays. Further,
        the user has the option either to pass the rank of each list or numpy array or to let the method compute them.
        When the user call this method a list containing the pairwise distances is returned as an output argument where
        the distances are stored as [{0,0},{0,1},{0,2},...,{1,0},{1,1},{1,2}], where {a,b} corresponds to the distance
        between the points 'a' and 'b'. Further, users are asked to provide the distance definition when the class
        Grassmann is instatiated. The current built-in options are the Grassmann, chordal, procrustes, and projection
        distances, but the users have also the option to implement their own distance definition.

        **Input:**

        :param argv: Matrices (at least 2) corresponding to the points on the Grassmann manifold.
        :type  argv: list of arguments

        :param kwargs: Contains the keyword for the user defined rank. If a list or numpy ndarray containing the rank of
               each matrix is not provided, the code will compute them using numpy.linalg.matrix_rank.
        :type kwargs: dictionary of arguments

        **Output/Returns:**

        :param distance_list: Pairwise distance.
        :type distance_list: list
        """

        nargs = len(argv[0])
        psi = argv[0]

        if 'rank' in kwargs.keys():
            ranks = kwargs['rank']
        else:
            ranks = None

        # Initial tests
        #-----------------------------------------------------------
        if ranks is None:
            ranks = []
            for i in range(nargs):
                ranks.append(np.linalg.matrix_rank(psi[i]))
        elif type(ranks) != list and type(ranks) != np.ndarray:
            raise TypeError('rank MUST be either a list or ndarray.')
            
        if nargs < 2:
            raise ValueError('Two matrices or more MUST be provided.')
        elif len(ranks) != nargs:
            raise ValueError('The number of elements in rank and in the input data MUST be the same.')
        #------------------------------------------------------------
            
        # Define the pairs of points to compute the Grassmann distance.
        indices = range(nargs)
        pairs = list(itertools.combinations(indices, 2))

        if self.user_distance_check:
            if self.distance_script is None:
                raise TypeError('distance_script cannot be None')

            exec('from ' + self.distance_script[:-3] + ' import ' + self.distance_object)
            distance_fun = eval(self.distance_object)
        else:
            if self.distance_object is None:
                raise TypeError('distance_object cannot be None')

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

    @staticmethod
    def grassmann_distance(x0, x1):

        """
        Estimate the Grassmann distance.

        One of the distances defined on a manifold is the Grassmann distance implemented herein. As the user gives the
        distance definition when the class Grassmann is instantiated the method 'distance' uses this information to call
        the respective distance definition.

        **Input:**

        :param x0: Point on the Grassmann manifold.
        :type  x0: list or numpy array

        :param x1: Point on the Grassmann manifold.
        :type  x1: list or numpy array

        **Output/Returns:**

        :param distance: Grassmann distance between x0 and x1.
        :type distance: float
        """

        l = min(np.shape(x0))
        k = min(np.shape(x1))
        rank = min(l, k)

        r = np.dot(x0.T, x1)
        (ui, si, vi) = svd(r, rank)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(si)
        distance = np.sqrt(abs(k - l) * np.pi ** 2 / 4 + np.sum(theta ** 2))

        return distance

    @staticmethod
    def chordal_distance(x0, x1):

        """
        Estimate the chordal distance.

        One of the distances defined on a manifold is the chordal distance implemented herein. As the user gives the
        distance definition when the class Grassmann is instantiated the method 'distance' uses this information to call
        the respective distance definition.

        **Input:**

        :param x0: Point on the Grassmann manifold.
        :type  x0: list or numpy array

        :param x1: Point on the Grassmann manifold.
        :type  x1: list or numpy array

        **Output/Returns:**

        :param distance: Chordal distance between x0 and x1.
        :type distance: float
        """

        l = min(np.shape(x0))
        k = min(np.shape(x1))
        rank = min(l, k)

        r_star = np.dot(x0.T, x1)
        (ui, si, vi) = svd(r_star, rank)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(si)
        theta = (np.sin(theta)) ** 2
        distance = np.sqrt(abs(k - l) + np.sum(theta))

        return distance

    @staticmethod
    def procrustes_distance(x0, x1):

        """
        Estimate the Procrustes distance.

        One of the distances defined on a manifold is the Procrustes distance implemented herein. As the user gives the
        distance definition when the class Grassmann is instantiated the method 'distance' uses this information to call
        the respective distance definition.

        **Input:**

        :param x0: Point on the Grassmann manifold.
        :type  x0: list or numpy array

        :param x1: Point on the Grassmann manifold.
        :type  x1: list or numpy array

        **Output/Returns:**

        :param distance: Procrustes distance between x0 and x1.
        :type distance: float
        """

        l = min(np.shape(x0))
        k = min(np.shape(x1))
        rank = min(l, k)

        r = np.dot(x0.T, x1)
        (ui, si, vi) = svd(r, rank)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(si)
        theta = np.sin(theta / 2) ** 2
        distance = np.sqrt(abs(k - l) + 2 * np.sum(theta))

        return distance

    @staticmethod
    def projection_distance(x0, x1):

        """
        Estimate the projection distance.

        One of the distances defined on a manifold is the projection distance implemented herein. As the user gives the
        distance definition when the class Grassmann is instantiated the method 'distance' uses this information to call
        the respective distance definition.

        **Input:**

        :param x0: Point on the Grassmann manifold.
        :type  x0: list or numpy array

        :param x1: Point on the Grassmann manifold.
        :type  x1: list or numpy array

        **Output/Returns:**

        :param distance: Projection distance between x0 and x1.
        :type distance: float
        """

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
        Compute a kernel matrix on the Grassmann manifold.

        This method computes the kernel matrix of points projected on the Grassmann manifold. The input arguments
        are passed through a list of arguments (argv) containing a list of lists or a list of numpy arrays. Further,
        the user has the option either to pass the rank of each list or numpy array or to let the method compute them.
        When the user call this method a Numpy array containing the kernel matrix is returned as an output argument.
        Further, users are asked to provide the kernel definition when the class Grassmann is instatiated. The current
        built-in options are the projection kernel and the Binet-Cauchy kernel, but the users have also the option to
        implement their own kernel definition.

        **Input:**

        :param argv: Matrices (at least 2) corresponding to the points on the Grassmann manifold.
        :type  argv: list of arguments

        :param kwargs: Contains the keyword for the user defined rank. If a list or numpy ndarray containing the rank of
               each matrix is not provided, the code will compute them using numpy.linalg.matrix_rank.
        :type kwargs: dictionary of arguments

        **Output/Returns:**

        :param kernel_matrix: Kernel matrix.
        :type kernel_matrix: Numpy array
        """

        nargs = len(argv[0])
        psi = argv[0]

        if 'rank' in kwargs.keys():
            ranks = kwargs['rank']
        else:
            ranks = None

        # Initial tests
        #-----------------------------------------------------------
        if ranks is None:
            ranks = []
            for i in range(nargs):
                ranks.append(np.linalg.matrix_rank(psi[i]))
        elif type(ranks) != list and type(ranks) != np.ndarray:
            raise TypeError('rank MUST be either a list or ndarray.')
            
        if nargs < 2:
            raise ValueError('Two matrices or more MUST be provided.')
        elif len(ranks) != nargs:
            raise ValueError('The number of elements in rank and in the input data MUST be the same.')
        #------------------------------------------------------------

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

    @staticmethod
    def projection_kernel(x0, x1):

        """
        Estimate the value of the projection kernel between x0 and x1.

        One of the kernels defined on a manifold is the projection kernel implemented herein. As the user gives the
        kernel definition when the class Grassmann is instantiated the method 'kernel' uses this information to call
        the respective kernel definition.

        **Input:**

        :param x0: Point on the Grassmann manifold.
        :type  x0: list or numpy array

        :param x1: Point on the Grassmann manifold.
        :type x1: list or numpy array

        **Output/Returns:**

        :param ker: Kernel value for x0 and x1.
        :type ker: float
        """

        r = np.dot(x0.T, x1)
        n = np.linalg.norm(r, 'fro')
        ker = n * n
        return ker

    @staticmethod
    def binet_cauchy_kernel(x0, x1):

        """
        Estimate the value of the Binet-Cauchy kernel between x0 and x1.

        One of the kernels defined on a manifold is the Binet-Cauchy kernel implemented herein. As the user gives the
        kernel definition when the class Grassmann is instantiated the method 'kernel' uses this information to call
        the respective kernel definition.

        **Input:**

        :param x0: Point on the Grassmann manifold.
        :type  x0: list or numpy array

        :param x1: Point on the Grassmann manifold.
        :type x1: list or numpy array

        **Output/Returns:**

        :param ker: Kernel value for x0 and x1.
        :type ker: float
        """

        r = np.dot(x0.T, x1)
        det = np.linalg.det(r)
        ker = det * det
        return ker

    # ==================================================================================================================
    @staticmethod
    def project_points(*argv, **kwargs):

        """
        Project the input matrices onto the Grassmann manifold via singular value decomposition (SVD).

        This method project matrices onto the Grassmann manifold via singular value decomposition. The input arguments
        are passed through a list of arguments (argv) containing a list of lists or a list of numpy arrays. Further,
        the user has the option either to pass the rank of each list or numpy array or to let the method compute them.
        When the user call this method the left and right -singular eigenvectors, as well as the singular values are
        returned together with the individual ranks and the maximum rank. The size of the output matrices are given by
        the maximum rank observed.

        **Input:**

        :param argv: Matrices (at least 2) corresponding to the points on the Grassmann manifold.
        :type  argv: list of arguments

        :param kwargs: Contains the keyword for the user defined rank. If a list or numpy ndarray containing the rank of
               each matrix is not provided, the code will compute them using numpy.linalg.matrix_rank.
        :type kwargs: dictionary of arguments

        **Output/Returns:**

        :param psi: Left-singular eigenvectors.
        :type psi: list

        :param sigma: Singular eigenvalues.
        :type sigma: list

        :param phi: Right-singular eigenvectors.
        :type phi: list

        :param max_rank: Maximum rank.
        :type max_rank: float

        :param ranks: Ranks of the input matrices.
        :type ranks: list
        """

        matrix, nargs = check_arguments(argv, min_num_matrix=1, ortho=False)

        if 'rank' in kwargs.keys():
            ranks = kwargs['rank']
        else:
            ranks = None   

        # Initial tests
        #-----------------------------------------------------------
        if ranks is None:
            ranks = []
            for i in range(nargs):
                ranks.append(np.linalg.matrix_rank(matrix[i]))
        elif type(ranks) != list and type(ranks) != np.ndarray:
            raise TypeError('rank MUST be either a list or ndarray.')
            
        if len(ranks) != nargs:
            raise ValueError('The number of elements in rank and in the input data MUST be the same.')
        #------------------------------------------------------------

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
    def log_mapping(*argv, ref=None, rank_ref=None):

        """
        Map points on the Grassmann manifold onto tangent space.

        It maps the points on the Grassmann manifold, passed to the method using *argv, onto the tangent space
        constructed on ref (a reference point on the Grassmann manifold). It is mandatory that the user pass a reference
        point to the method. Moreover, the user is asked to to provide the dimension of the manifold where the tangent 
        space is constructed, but if this value is not provided the method compute the rank of the reference point.

        **Input:**

        :param argv: Matrices (at least 2) corresponding to the points on the Grassmann manifold.
        :type  argv: list of arguments

        :param ref: A point on the Grassmann manifold used as reference to construct the tangent space.
        :type ref: list or numpy array
        
        :param rank_ref: Rank of the reference point.
        :type rank_ref: int

        **Output/Returns:**

        :param _gamma: Point on the tangent space.
        :type _gamma: list
        """
        
        u, nr = check_arguments(argv, min_num_matrix=1, ortho=False)

        # Initial tests
        #-----------------------------------------------------------
        if rank_ref is None:
            rank_ref = np.linalg.matrix_rank(ref)
        elif type(rank_ref) != int:
            raise TypeError('rank of reference MUST be integer.')   
        #------------------------------------------------------------
        
        # Check the input arguments.
        #u, nr = check_arguments(argv, min_num_matrix=1, ortho=False)

        ref = ref[:,0:rank_ref]
        if type(ref) != list:
            ref_list = ref.tolist()
        else:
            ref_list = ref
            ref = np.array(ref)

        refT = ref.T
        m0 = np.dot(ref, refT)

        _gamma = []
        for i in range(nr):
            utrunc=u[i][:,0:rank_ref]
            if utrunc.tolist() == ref_list:
                _gamma.append(np.zeros(np.shape(ref)))
            else:

                # M = ((I - psi0*psi0')*psi1)*inv(psi0'*psi1)
                minv = np.linalg.inv(np.dot(refT, utrunc))
                m = np.dot(utrunc - np.dot(m0, utrunc), minv)
                ui, si, vi = np.linalg.svd(m, full_matrices=False)  # svd(m, max_rank)
                _gamma.append(np.dot(np.dot(ui, np.diag(np.arctan(si))), vi))

        return _gamma

    @staticmethod
    def exp_mapping(*argv, ref=None, rank_ref=None):

        """
        Map points on the tangent space onto the Grassmann manifold.

        It maps the points on the tangent space, passed to the method using *argv, onto the Grassmann manifold
        constructed on ref (a reference point on the Grassmann manifold). It is mandatory that the user pass a reference
        point to the method. Moreover, the user is asked to to provide the dimension of the manifold where the tangent 
        space is constructed, but if this value is not provided the method compute the rank of the reference point.

        **Input:**

        :param argv: Matrices (at least 2) corresponding to the points on the Grassmann manifold.
        :type  argv: list of arguments

        :param ref: A point on the Grassmann manifold used as reference to construct the tangent space.
        :type ref: list or numpy array
        
        :param rank_ref: Rank of the reference point.
        :type rank_ref: int

        **Output/Returns:**

        :param _gamma: Point on the tangent space.
        :type _gamma: list
        """
        """
        exp_mapping: perform the exponential mapping from the tangent space
                     to the Grassmann manifold.
        :param ref: list or ndarray containing the reference point on the Grassmann manifold
                    where the tangent space is approximated.

        """
        # Check input arguments.
        u, nr = check_arguments(argv, min_num_matrix=1, ortho=False)

        # Initial tests
        #-----------------------------------------------------------
        if rank_ref is None:
            rank_ref = np.linalg.matrix_rank(ref)
        elif type(rank_ref) != int:
            raise TypeError('rank of reference MUST be integer.')    
        #------------------------------------------------------------

        ref = ref[:,0:rank_ref]
        x = []
        for i in range(nr):
            
            utrunc=u[i][:,0:rank_ref]
            ui, si, vi = np.linalg.svd(utrunc, full_matrices=False)

            # Exponential mapping.
            x0 = np.dot(np.dot(np.dot(ref, vi.T), np.diag(np.cos(si))) + np.dot(ui, np.diag(np.sin(si))), vi)

            # Test orthogonality.
            xtest = np.dot(x0.T, x0)

            if not np.allclose(xtest, np.identity(np.shape(xtest)[0])):
                x0, unused = np.linalg.qr(x0)  # re-orthonormalizing.

            x.append(x0)

        return x

    def karcher_mean(self, *argv, **kwargs):

        """
        Compute the Karcher mean.

        This method computes the Karcher mean given a set of points on the Grassmann manifold. The Karcher mean is
        estimated by the minimization of the Frechet variance, where the Frechet variance corresponds to the summation of the
        square distances, defined on a manifold, to a given point.
        
        **Input:**

        :param argv: Matrices (at least 2) corresponding to the points on the Grassmann manifold.
        :type  argv: list of arguments
        
        :param kwargs: Contains the keywords for the used in the optimizers to find the Karcher mean.
        :type kwargs: dictionary of arguments

        **Output/Returns:**

        :param kr_mean: Karcher mean.
        :type kr_mean: list
        """

        matrix, nargs = check_arguments(argv, min_num_matrix=2, ortho=False)

        if self.user_karcher_check:
            exec('from ' + self.karcher_script[:-3] + ' import ' + self.karcher_object)
            karcher_fun = eval(self.karcher_object)
        else:
            karcher_fun = eval("Grassmann." + self.karcher_object)

        shape_ref = np.shape(matrix[0])
        for i in range(1, nargs):
            if np.shape(matrix[i]) != shape_ref:
                raise TypeError('Input matrices have different shape.')

        kr_mean = karcher_fun(self, matrix, kwargs)

        return kr_mean

    def gradient_descent(self, data_points, kwargs):

        """
        Compute the Karcher mean using the gradient descent method.

        This method computes the Karcher mean given a set of points on the Grassmann manifold. In this regard, the
        gradient descent method is implemented herein also considering the acceleration scheme due to Nesterov. Further,
        this method is called by the method 'karcher_mean'.

        **Input:**

        :param data_points: Points on the Grassmann manifold.
        :type  data_points: list

        :param kwargs: Contains the keywords for the used in the optimizers to find the Karcher mean.
        :type kwargs: dictionary of arguments

        **Output/Returns:**

        :param mean_element: Karcher mean.
        :type mean_element: list
        """

        if 'acc' in kwargs.keys():
            acc = kwargs['acc']
        else:
            acc = False

        if 'tol' in kwargs.keys():
            tol = kwargs['tol']
        else:
            tol = 1e-3

        if 'maxiter' in kwargs.keys():
            maxiter = kwargs['maxiter']
        else:
            maxiter = 1000

        n_mat = len(data_points)

        alpha = 0.5  # todo: this can be adaptive (e.g., Armijo rule).
        rnk = []
        for i in range(n_mat):
            rnk.append(min(np.shape(data_points[i])))

        max_rank = max(rnk)
        
        fmean = []
        for i in range(n_mat):
            fmean.append(self.frechet_variance(data_points[i], data_points))

        index_0 = fmean.index(min(fmean))
        mean_element = data_points[index_0].tolist()

        avg_gamma = np.zeros([np.shape(data_points[0])[0], np.shape(data_points[0])[1]])

        itera = 0

        l = 0
        avg = []
        _gamma = []
        if acc:
            _gamma = Grassmann.log_mapping(data_points, ref=np.asarray(mean_element), rank_ref=max_rank)
            avg_gamma.fill(0)
            for i in range(n_mat):
                avg_gamma += _gamma[i] / n_mat
            avg.append(avg_gamma)

        # Main loop
        while itera <= maxiter:

            _gamma = Grassmann.log_mapping(data_points, ref=np.asarray(mean_element), rank_ref=max_rank)
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

            x = Grassmann.exp_mapping(step, ref=np.asarray(mean_element), rank_ref=max_rank)

            test_1 = np.linalg.norm(x[0] - mean_element, 'fro')

            if test_1 < tol:
                break

            mean_element = []
            mean_element = x[0]

            itera += 1

        return mean_element

    def stochastic_gradient_descent(self, data_points, kwargs):

        """
        Compute the Karcher mean using the stochastic gradient descent method.

        This method computes the Karcher mean given a set of points on the Grassmann manifold. In this regard, the
        stochastic gradient descent method is implemented herein. Further, this method is called by the method
        'karcher_mean'.

        **Input:**

        :param data_points: Points on the Grassmann manifold.
        :type  data_points: list

        :param kwargs: Contains the keywords for the used in the optimizers to find the Karcher mean.
        :type kwargs: dictionary of arguments

        **Output/Returns:**

        :param mean_element: Karcher mean.
        :type mean_element: list
        """

        if 'tol' in kwargs.keys():
            tol = kwargs['tol']
        else:
            tol = 1e-3

        if 'maxiter' in kwargs.keys():
            maxiter = kwargs['maxiter']
        else:
            maxiter = 1000

        n_mat = len(data_points)

        rnk = []
        for i in range(n_mat):
            rnk.append(min(np.shape(data_points[i])))
            
        max_rank = max(rnk)
        
        fmean = []
        for i in range(n_mat):
            fmean.append(self.frechet_variance(data_points[i], data_points), rank_ref=max_rank)

        index_0 = fmean.index(min(fmean))

        mean_element = data_points[index_0].tolist()
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
                _gamma = Grassmann.log_mapping(data_points[idx], ref=np.asarray(mean_element), rank_ref=max_rank)

                step = 2 * alpha * _gamma[0]

                X = Grassmann.exp_mapping(step, ref=np.asarray(mean_element), rank_ref=max_rank)

                _gamma = []
                mean_element = X[0]

                k += 1

            test_1 = np.linalg.norm(mean_element - melem, 'fro')
            if test_1 < tol:
                break

            itera += 1

        return mean_element

    def frechet_variance(self, point, *argv):

        """
        Compute Frechet variance.

        The Frechet variance corresponds to the summation of the square distances, defined on a manifold, to a given
        point also on the manifold. Thie method is employed in the minimization scheme used to find the Karcher mean.

        **Input:**

        :param point: Point on the Grassmann manifold where the Frechet variance is computed.
        :type  point: list or numpy array

        :param argv: Matrices (at least 2) corresponding to the points on the Grassmann manifold.
        :type  argv: list of arguments

        **Output/Returns:**

        :param fmean: Frechet variance.
        :type fmean: list
        """

        matrix, n_mat = check_arguments(argv, min_num_matrix=2, ortho=False)

        accum = 0
        for i in range(n_mat):
            d = self.distance([point, matrix[i]])
            accum += d[0] * d[0]

        frechet_var = accum / n_mat
        return frechet_var

    def interpolate_sample(self, *argv, coordinates=None, point=None, **kwargs):

        """
        Interpolate a point.

        Once the points on the Grassmann manifold are projected onto the tangent space standard interpolation can be
        performed. In this regard, the user should provide the data points, the coordinates of each input data point,
        and the point to be interpolated. Furthermore, additional parameters, depending on the selected interpolation
        method, can be provided via kwargs.

        **Input:**

        :param coordinates: Coordinates of the input data points.
        :type  coordinates: numpy array

        :param data_points: Matrices corresponding to the points on the Grassmann manifold.
        :type  data_points: numpy array

        :param point: Coordinates of the point to be interpolated.
        :type  point: numpy array

        :param kwargs: Contains the keywords for the parameters used in different interpolation methods.
        :type kwargs: dictionary of arguments

        **Output/Returns:**

        :param interp_point: Interpolated point.
        :type interp_point: numpy array
        """

        data_points, nargs = check_arguments(argv, min_num_matrix=3, ortho=False)

        if self.user_interp_check:
            exec('from ' + self.interp_script[:-3] + ' import ' + self.interp_object)
            interp_fun = eval(self.interp_object)
        else:
            interp_fun = eval("Grassmann." + self.interp_object)

        shape_ref = np.shape(data_points[0])
        for i in range(1, nargs):
            if np.shape(data_points[i]) != shape_ref:
                raise TypeError('Input matrices have different shape.')

        # Test if the sample is stored as a list
        if type(point) == list:
            point = np.array(point)

        # Test if the nodes are stored as a list
        if type(coordinates) == list:
            coordinates = np.array(coordinates)

        interp_point = interp_fun(coordinates, data_points, point, kwargs)

        return interp_point

    # ==================================================================================================================
    # The pre-defined interpolators are implemented in this section. Any new pre-defined interpolator must be
    # implemented here with the decorator @staticmethod.

    @staticmethod
    def linear_interp(coordinates, data_points, point, kwargs):

        """
        Interpolate a point using the linear interpolation.

        For the linear interpolation the user are asked to provide the data points, the coordinates of the data points,
        and the coordinate of the point to be interpolated.

        **Input:**

        :param coordinates: Coordinates of the input data points.
        :type  coordinates: numpy array

        :param data_points: Matrices corresponding to the points on the Grassmann manifold.
        :type  data_points: numpy array

        :param point: Coordinates of the point to be interpolated.
        :type  point: numpy array

        :param kwargs: Contains the keywords for the parameters used in different interpolation methods.
        :type kwargs: dictionary of arguments

        **Output/Returns:**

        :param interp_point: Interpolated point.
        :type interp_point: numpy array
        """

        myInterpolator = LinearNDInterpolator(coordinates, data_points)
        interp_point = myInterpolator(point)
        interp_point = interp_point[0]

        return interp_point

    @staticmethod
    def kriging_interp(coordinates, data_points, point, kwargs):

        """
        Interpolate a point using the Kriging interpolation.

        For the Kriging interpolation the user are asked to provide the data points, the coordinates of the data points,
        and the coordinate of the point to be interpolated. Further, the necessary parameters to perform the Kriging
        interpolation are the regression model 'reg_model', the correlation model 'corr_model', and the number of times
        optimization problem is to be solved with different starting point'n_opt'. In fact, additional parameters can be
        considered by the user if necessary.

        **Input:**

        :param coordinates: Coordinates of the input data points.
        :type  coordinates: numpy array

        :param data_points: Matrices corresponding to the points on the Grassmann manifold.
        :type  data_points: numpy array

        :param point: Coordinates of the point to be interpolated.
        :type  point: numpy array

        :param kwargs: Contains the keywords for the parameters used in different interpolation methods.
        :type kwargs: dictionary of arguments

        **Output/Returns:**

        :param interp_point: Interpolated point.
        :type interp_point: numpy array
        """

        if 'reg_model' in kwargs.keys():
            reg_model = kwargs['reg_model']
        else:
            reg_model = None

        if 'corr_model' in kwargs.keys():
            corr_model = kwargs['corr_model']
        else:
            corr_model = None

        if 'n_opt' in kwargs.keys():
            n_opt = kwargs['n_opt']
        else:
            n_opt = None

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

        shape_ref = np.shape(data_points[0])
        interp_point = np.zeros(shape_ref)
        nargs = len(data_points)
        nrows = data_points[0].shape[0]
        ncols = data_points[0].shape[1]

        val_data = []

        for j in range(nrows):
            for k in range(ncols):
                for i in range(nargs):
                    val_data.append([data_points[i][j, k]])

                if val_data.count(val_data[0]) == len(val_data):
                    val = np.array(val_data)
                    y = val[0]
                else:
                    val = np.array(val_data)
                    K = Krig(samples=coordinates, values=val, reg_model=reg_model, corr_model=corr_model, n_opt=n_opt)
                    y, mse = K.interpolate(point, dy=True)

                val_data.clear()

                interp_point[j, k] = y

        return interp_point


########################################################################################################################
########################################################################################################################
#                                            Diffusion Maps                                                            #
########################################################################################################################
########################################################################################################################

class DiffusionMaps:
    """
    Perform the diffusion maps on the input data to reveal the lower dimensional embedded data geometry.

    In this class the diffusion maps create a connection between the spectral properties of the diffusion process and
    the intrinsic geometry of the data resulting in a multiscale representation of the data. In this regard an affinity
    matrix containing the degree of similarity of the data points is either estimated based on the euclidean distance
    using a Gaussian kernel, or the affinity matrix is computed using other Kernel definition and passed to the main
    method (e.g., defining a kernel on the Grassmann manifold).

    **References:**

    1. Ronald R. Coifman, Stéphane Lafon, "Diffusion maps", Applied and Computational Harmonic Analysis, Volume 21,
       Issue 1, Pages 5-30, 2006.

    2. Nadler, B., Lafon, S., Coifman, R., and Kevrekidis, I., "Diffusion maps, spectral clustering and eigenfunctions
       of Fokker-Planck operators", In Y. Weiss, B. Scholkopf, and J. Platt (Eds.), Advances in Neural Information
       Processing Systems, 18, pages 955 – 962, 2006, Cambridge, MA: MIT Press.

    **Input:**

    :param alpha: Assumes a value between 0 and 1 and corresponding to different diffusion operators.

                  Default: 0.5
    :type alpha: float

    :param n_evecs: the number of eigenvectors and eigenvalues used in the representation of the diffusion coordinates.

                    Default: 2
    :type n_evecs: int

    :param sparse: Is a boolean variable defining the sparsity of the graph generated when data is provided to estimate
                   the diffusion coordinates using the standard approach.

                   Default: False
    :type sparse: bool

    :param k_neighbors: Used when sparse is True to define the number of samples close to a given sample
                        used in the construction of the affinity matrix.

                        Default: 1
    :type k_neighbors: int

    **Authors:**

    Ketson R. M. dos Santos, Dimitris G. Giovanis

    Last modified: 03/26/20 by Ketson R. M. dos Santos
    """

    def __init__(self, alpha=0.5, n_evecs=2, sparse=False, k_neighbors=1):
        self.alpha = alpha
        self.n_evecs = n_evecs
        self.sparse = sparse
        self.k_neighbors = k_neighbors

        if alpha < 0 or alpha > 1:
            raise ValueError('alpha should be a value between 0 and 1.')

        if isinstance(n_evecs, int):
            if n_evecs < 1:
                raise ValueError('n_evecs should be larger than or equal to one.')
        else:
            raise TypeError('n_evecs should be integer.')

        if not isinstance(sparse, bool):
            raise TypeError('sparse should be a boolean variable.')
        elif sparse is True:
            if isinstance(k_neighbors, int):
                if k_neighbors < 1:
                    raise ValueError('k_neighbors should be larger than or equal to one.')
            else:
                raise TypeError('k_neighbors should be integer.')

    def mapping(self, **kwargs):

        """
        Perform diffusion maps to reveal the embedded geometry of datasets.

        In nonlinear dimensionality reduction Diffusion Maps corresponds to a technique used to reveal the intrinsic
        structure of datasets based on diffusion processes. In particular, the eigenfunctions function of Markov
        matrices defining a random walk on the data are used to obtain coordinate system, represented by the diffusion
        coordinates, revealing the geometric description of the data. It is worth to mention that the diffusion
        coordinates are defined on a Euclidean space where usual metrics define the distances between pairs of data
        points. Moreover, the diffusion maps create a connection between the spectral properties of the diffusion
        process and the intrinsic geometry of the data resulting in a multiscale representation of the data. In this
        method, the users either provide the input data or they provide the affinity matrix. If the input data is
        provided the standard diffusion maps is performed where the parameter 'epsilon' of the Gaussian kernel is either
        provided by the user or estimated based on the median of the square of the euclidian distances between data
        points. On the other hand, the user can compute the affinity matrix externally (e.g., using the Grassmann
        kernel) in order to pass it to the method.

        **Input:**

        :param kwargs: Contains the following keywords: data (input data), epsilon (parameter of the Gaussian kernel),
                       and kernel_matrix (when the kernel matrix is provided).
        :type kwargs: dictionary of arguments

        **Output/Returns:**

        :param dcoords: Diffusion coordinates.
        :type dcoords: numpy array

        :param evals: eigenvalues.
        :type evals: numpy array

        :param evecs: eigenvectors.
        :type evecs: numpy array
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
        # k_matrix, epsilon = DiffusionMaps.create_kernel_matrix(self, data, epsilon, kernel_mat=kernel_mat)
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
            dcoords[:, i] = evals[i] * evecs[:, i]

        return dcoords, evals, evecs

    # @staticmethod
    def create_kernel_matrix(self, data, epsilon=None):

        """
        Compute the Kernel matrix for the standard diffusion maps technique.

        If a dataset is provided and no kernel matrix is computed externally one can use this method to estimate the
        affinity matrix using the Gaussian kernel. In this regard, if no 'epsilon' is provided the method estimates a
        suitable value taking the median of the square value of the pairwise euclidean distances of the points in the
        input dataset.

        **Input:**

        :param data: Contains the input data.
        :type data: list of numpy array

        :param epsilon: Parameter of the Gaussian kernel.

                        Default: None
        :type epsilon: float

        **Output/Returns:**

        :param kernel_mat: Kernel matrix.
        :type kernel_mat: numpy array
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
        Find epsilon based on the pairwise distances.

        If the user does not provide the value of 'epsilon' use the pairwise distances to estimate it based on the
        median of their values squared.

        **Input:**

        :param dist_pairs: Pairwise distance.
        :type dist_pairs: list of numpy array

        **Output/Returns:**

        :param epsilon: Gaussian kernel parameter.
        :type epsilon: float
        """

        dist_pairs_sq = np.array(dist_pairs) ** 2
        epsilon = np.median(dist_pairs_sq)

        return epsilon

    @staticmethod
    def D_matrix(kernel_mat, alpha):

        """
        Compute the diagonal matrix D and its inverse.

        In the normalization process we have to estimate matrix D(i,i) = sum(Kernel(i,j)^alpha,j) and its inverse.

        **Input:**

        :param kernel_mat: Kernel matrix.
        :type kernel_mat: list of numpy array

        **Output/Returns:**

        :param D: Matrix D.
        :type D: list of numpy array

        :param D_inv: Inverse of matrix D.
        :type D_inv: list of numpy array
        """

        kmat = kernel_mat
        D = np.array(kmat.sum(axis=1)).flatten()

        D_inv = np.power(D, -alpha)
        return D, D_inv

    # @staticmethod
    def l_alpha_normalize(self, kernel_mat, D_inv):

        """
        Compute and normalize the kernel matrix with the matrix D.

        In the normalization process we have to estimate matrix D(i,i) = sum(Kernel(i,j)^alpha,j) and its inverse.
        We now use this information to normalize the kernel matrix.

        **Input:**

        :param kernel_mat: Kernel matrix.
        :type kernel_mat: list of numpy array

        :param D_inv: Inverse of matrix D.
        :type D_inv: list of numpy array

        **Output/Returns:**

        :param norm_ker: Normalized kernel.
        :type norm_ker: list of numpy array
        """

        sparse = self.sparse
        m = D_inv.shape[0]

        if sparse:
            Dalpha = sps.spdiags(D_inv, 0, m, m)
        else:
            Dalpha = np.diag(D_inv)

        norm_ker = Dalpha.dot(kernel_mat.dot(Dalpha))

        return norm_ker
