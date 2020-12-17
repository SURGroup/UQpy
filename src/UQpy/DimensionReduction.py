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
``DimensionReduction`` is the module for ``UQpy`` to perform the dimensionality reduction of high-dimensional data.

This module contains the classes and methods to perform pointwise and multi point data-based dimensionality reduction
via projection onto Grassmann manifolds and Diffusion Maps, respectively. Further, interpolation in the tangent space
centered at a given point on the Grassmann manifold can be performed, as well as the computation of kernels defined on
the Grassmann manifold to be employed in techniques using subspaces.

The module currently contains the following classes:

* ``Grassmann``: Class for for analysis of samples on the Grassmann manifold.

* ``DiffusionMaps``: Class for multi point data-based dimensionality reduction.

* ``POD``: Class for dimension reduction of solution snapshots.

"""

import copy
import itertools

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from UQpy.Utilities import svd
import functools

import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.spatial.distance as sd
from scipy.interpolate import LinearNDInterpolator

from UQpy.Surrogates import Kriging
from UQpy.Utilities import *
from UQpy.Utilities import _nn_coord


########################################################################################################################
########################################################################################################################
#                                            Grassmann Manifold                                                        #
########################################################################################################################
########################################################################################################################

class Grassmann:
    """
    Mathematical analysis on the Grassmann manifold.

    The ``Grassmann`` class contains methods of data analysis on the Grassmann manifold, which is a special case of flag
    manifold. The projection of matrices onto the Grassmann manifold is performed via singular value decomposition(SVD),
    where their dimensionality are reduced. Further, the mapping from the Grassmann manifold to a tangent space
    constructed at a given reference point (logarithmic mapping), as well as the mapping from the tangent space to the
    manifold (exponential mapping) are implemented as methods. Moreover, an interpolation can be performed on the
    tangent space taking advantage of the implemented logarithmic and exponential maps. Additional quantities such
    as the Karcher mean, distance, and kernel, all defined on the Grassmann manifold, can be obtained using
    implemented methods. When the class ``Grassmann`` is instantiated some attributes are set using the method
    ``manifold``, where the dimension of the manifold as well as the samples are input arguments.
    
    **Input:**

    * **distance_method** (`callable`)
        Defines the distance metric on the manifold. The user can pass a callable object defining the distance metric
        using two different ways. First, the user can pass one of the methods implemented in the class ``Grassmann``, they are:

        - `grassmann_distance`;
        - `chordal_distance`;
        - `procrustes_distance`;
        - `projection_distance`
        - `binet_cauchy_distance`.

        In this regard,
        the class ``Grassmann`` is instantiated and the attributes are set using the method ``manifold``. Thus,
        an object containing the attribute `distance_method` is passed as `Grassmann.distance_method`. Second, the user
        can pass either a method of a class or a function. For example, if the user wish to
        use `grassmann_distance` to compute the distance, one can use the following command:

        On the other hand, if the user implemented a function
        (e.g., `user_distance`) to compute the distance, `distance_method` must assume the following value
        `distance_method = user_distance`, which must be pre-loaded using import. In this regard, the
        function input must contain the first (x0) and second (x1) matrices as arguments (e.g, user_distance(x0,x1))

    * **kernel_method** (`callable`)
        Object of the kernel function defined on the Grassmann manifold. The user can pass a object using two different
        ways. First, the user can pass one of the methods implemented in the class ``Grassmann``, they are:

        - `projection_kernel`;
        - `binet_cauchy_kernel`.

        In this regard, the object is passed as `Grassmann.kernel_method`.
        Second, the user can pass callable objects either as a method of a class or as a function. 
        For example, if the user wish to use `projection_kernel` to estimate the kernel matrix, one can use the
        following command:

        On the other hand, if the user implemented
        a function (e.g., `user_kernel`) to compute the kernel matrix, `kernel_method` must assume the following value
        `kernel_object = user_kernel`, which must be pre-loaded using import. In this regard, the function
        input must contain the first (x0) and second (x1) matrices as arguments (e.g, user_kernel(x0,x1))
    
    * **interp_object** (`object`)
        Interpolator to be used in the Tangent space. The user can pass an object defining the interpolator
        via four different ways.

        - Using the ``Grassmann`` method ``linear_interp`` as Grassmann(interp_object=Grassmann.linear_interp).
        - Using an object of ``UQpy.Kriging`` as Grassmann(interp_object=Kriging_Object)
        - Using an object of ``sklearn.gaussian_process`` as Grassmann(interp_object=Sklearn_Object)
        - Using an user defined object (e.g., user_interp). In this case, the function must contain the following
          arguments: `coordinates`, `samples`, and `point`.
    
    * **karcher_method** (`callable`)
        Optimization method used in the estimation of the Karcher mean. The user can pass a callable object via
        two different ways. First, the user can pass one of the methods implemented in the class ``Grassmann``,
        they are:

        - ``gradient_descent``;
        - ``stochastic_gradient_descent``.

        Second, the user can pass callable objects either as a method of a class or as a function. It is worth
        mentioning that the method ``gradient_descent`` also allows the accelerated descent method due to Nesterov.
    
    **Attributes:**

    * **p** (`int` or `str`)
        Dimension of the p-planes defining the Grassmann manifold G(n,p).
        
    * **ranks** (`list`)
        Dimension of the embedding dimension for the manifolds G(n,p) of each sample.
        
    * **samples** (`list` of `list` or `ndarray`)
        Input samples defined as a `list` of matrices.
        
    * **nargs** (`int`)
        Number of matrices in `samples`.
        
    * **max_rank** (`int`)
        Maximum value of `ranks`.
        
    * **psi** (`list`)
        Left singular eigenvectors from the singular value decomposition of each sample in `samples`
        representing a point on the Grassmann manifold.
    
    * **sigma** (`list`)
        Singular values from the singular value decomposition of each sample in `samples`.
    
    * **phi** (`list`)
        Right singular eigenvector from the singular value decomposition of each sample in `samples`
        representing a point on the Grassmann manifold.

    **Methods:**

    """

    def __init__(self, distance_method=None, kernel_method=None, interp_object=None, karcher_method=None):

        # Distance.
        if distance_method is not None:
            if callable(distance_method):
                self.distance_object = distance_method
            else:
                raise TypeError('UQpy: A callable distance object must be provided.')

        # Kernels.
        if kernel_method is not None:
            if callable(kernel_method):
                self.kernel_object = kernel_method
            else:
                raise TypeError('UQpy: A callable kernel object must be provided.')
        
        # Interpolation.
        skl_str = "<class 'sklearn.gaussian_process._gpr.GaussianProcessRegressor'>"
        self.skl = str(type(interp_object))==skl_str
        if interp_object is not None:
            if callable(interp_object) or isinstance(interp_object, Kriging) or self.skl:
                self.interp_object = interp_object
            else:
                raise TypeError('UQpy: A callable interpolation object must be provided.')

        # Karcher mean.
        if karcher_method is not None:
            if distance_method is None:
                raise ValueError('UQpy: A callable distance object must be provided too.')

            if callable(karcher_method):
                self.karcher_object = karcher_method
            else:
                raise TypeError('UQpy: A callable Karcher mean object must be provided.')

        self.samples = []
        self.psi = []
        self.sigma = []
        self.phi = []
        self.n_psi = []
        self.n_phi = []
        self.p = None
        self.ranks = []
        self.nargs = 0
        self.max_rank = None

    def manifold(self, p=None, samples=None, append_samples=False):

        """
        Set the grassmann manifold and project the samples on it via singular value decomposition.

        This method project samples onto the Grassmann manifold via singular value decomposition. The input arguments
        are passed through the argument `samples` containing a list of lists or a list of ndarrays. Moreover, this
        method serves to set the manifold and to verify the consistency of the input data. This method can be called
        using the following command:

        In this case, append_samples is a boolean variable used to define if new sample will replace the previous ones
        or will just get appended (`append_samples=True`).

        **Input:**
        
        * **p** (`int` or `str` or `NoneType`)
            Dimension of the p-planes defining the Grassmann manifold G(n,p). This parameter can assume an integer value 
            larger than 0 or the strings `max`, when `p` assumes the maximum rank of the input matrices, or `min` when
            it assumes the minimum one. If `p` is not provided `ranks` will store the ranks of each input
            matrix and each sample will lie on a distinct manifold.
            
        * **samples** (`list`)
            Input samples defined as a `list` of matrices. In this regard, `samples` is a 
            collection of matrices stored as a `list`. Moreover, The shape of the input matrices stored 
            in samples are verified and compared for consistency with `p`.
            
        * **append_samples** (`bool`)
            The attributes are replaced when manifold is called if `append_samples` is False, otherwise the lists are 
            appended.

        """

        # If manifold called for the first time 
        # force append_samples to be False.
        if not self.samples:
            append_samples = False

        # samples must be a list.
        # Test samples for type consistency.
        if not isinstance(samples, list) and not isinstance(samples, np.ndarray):
            raise TypeError('UQpy: `samples` must be either a list or numpy.ndarray.')
        elif isinstance(samples, np.ndarray):
            samples = samples.tolist()

        # If append_samples is true, store the new samples
        # in a new list.
        if append_samples:
            samples_new = copy.copy(samples)
            nargs_new = len(samples_new)

            for i in range(nargs_new):
                self.samples.append(samples_new[i])

            samples = self.samples
            if p is None:
                p = self.p

        # samples must be converted into a ndarray due to the necessary computation over slices.
        # Check the number of matrices stored in samples.
        nargs = len(samples)

        # At least one argument must be provided, otherwise show an error message.
        if nargs < 1:
            raise ValueError('UQpy: At least one input matrix must be provided.')

        n_left = []
        n_right = []
        for i in range(nargs):
            n_left.append(max(np.shape(samples[i])))
            n_right.append(min(np.shape(samples[i])))

        bool_left = (n_left.count(n_left[0]) != len(n_left))
        bool_right = (n_right.count(n_right[0]) != len(n_right))

        if bool_left and bool_right:
            raise TypeError('UQpy: The shape of the input matrices must be the same.')
        else:
            n_psi = n_left[0]
            n_phi = n_right[0]

        if isinstance(p, str):

            # If append_sample just compute the rank of the new samples.
            if append_samples:
                ranks = self.ranks
                ranks_new = []
                for i in range(nargs_new):
                    rnk = int(np.linalg.matrix_rank(samples_new[i]))
                    ranks.append(rnk)
                    ranks_new.append(rnk)
            else:
                ranks = []
                for i in range(nargs):
                    ranks.append(np.linalg.matrix_rank(samples[i]))

            if p is "max":
                # Get the maximum rank of the input matrices
                p = int(max(ranks))
            elif p is "min":
                # Get the minimum rank of the input matrices
                p = int(min(ranks))
            else:
                raise ValueError('UQpy: The only allowable input strings are `min` and `max`.')

            ranks = np.ones(nargs) * [int(p)]
            ranks = ranks.tolist()

            if append_samples:
                ranks_new = np.ones(nargs_new) * [int(p)]
                ranks_new = ranks_new.tolist()
        else:
            if p is None:
                if append_samples:
                    ranks = self.ranks
                    ranks_new = []
                    for i in range(nargs_new):
                        rnk = int(np.linalg.matrix_rank(samples_new[i]))
                        ranks.append(rnk)
                        ranks_new.append(rnk)
                else:
                    ranks = []
                    for i in range(nargs):
                        ranks.append(np.linalg.matrix_rank(samples[i]))
            else:
                if not isinstance(p, int):
                    raise TypeError('UQpy: `p` must be integer.')

                if p < 1:
                    raise ValueError('UQpy: `p` must be an integer larger than or equal to one.')

                for i in range(nargs):
                    if min(np.shape(samples[i])) < p:
                        raise ValueError('UQpy: The dimension of the input data is not consistent with `p` of G(n,p).')

                ranks = np.ones(nargs) * [int(p)]
                ranks = ranks.tolist()

                if append_samples:
                    ranks_new = np.ones(nargs_new) * [int(p)]
                    ranks_new = ranks_new.tolist()

        ranks = list(map(int, ranks))

        # For each point perform svd.
        if append_samples:
            for i in range(nargs_new):
                u, s, v = svd(samples_new[i], int(ranks_new[i]))
                self.psi.append(u)
                self.sigma.append(np.diag(s))
                self.phi.append(v)
        else:
            psi = []  # initialize the left singular eigenvectors as a list.
            sigma = []  # initialize the singular values as a list.
            phi = []  # initialize the right singular eigenvectors as a list.
            for i in range(nargs):
                u, s, v = svd(samples[i], int(ranks[i]))
                psi.append(u)
                sigma.append(np.diag(s))
                phi.append(v)

            self.samples = samples
            self.psi = psi
            self.sigma = sigma
            self.phi = phi

        self.n_psi = n_psi
        self.n_phi = n_phi
        self.p = p
        self.ranks = ranks
        self.nargs = nargs
        self.max_rank = int(np.max(ranks))

    def distance(self, points_grassmann=None):

        """
        Estimate the distance between points on the Grassmann manifold.

        This method computes the pairwise distance of points projected on the Grassmann manifold. The input arguments
        are passed through a `list` of `list` or a `list` of `ndarray`. When the user call this method a list containing
        the pairwise distances is returned as an output argument where the distances are stored as 
        [{0,1},{0,2},...,{1,0},{1,1},{1,2},...], where {a,b} corresponds to the distance between the points 'a' and 
        'b'. Further, users are asked to provide the distance definition when the class `Grassmann` is instatiated. 
        The current built-in options are the `grassmann_distance`, `chordal_distance`, `procrustes_distance`, 
        `projection_distance`, and `binet_cauchy_distance`, but the users have also the option to implement their own 
        distance definition. In this case, the user must be aware that the matrices in `points_grassmann` must represent 
        points on the Grassmann manifold. For example, given the points on the Grassmann manifold one can compute the 
        pairwise distances in the following way:

        **Input:**

        * **points_grassmann** (`list` or `NoneType`) 
            Matrices (at least 2) corresponding to the points on the Grassmann manifold. If `points_grassmann` is not
            provided it means that the samples in `manifold` are employed, and the pairwise distances of the points on
            the manifold defined by the left and right singular eigenvectors are computed.

        **Output/Returns:**

        * **points_distance** (`list`)
            Pairwise distances if `points_grassmann` is provided.
                
        * **points_distance_psi** (`list`)
            Pairwise distance of points on the manifold defined by the left singular eigenvectors if `points_grassmann` 
            is not provided.
            
        * **points_distance_phi** (`list`)
            Pairwise distance of points on the manifold defined by the right singular eigenvectors if `points_grassmann` 
            is not provided.

        """

        # Show an error message if no distance_object is identified.
        if self.distance_object is None:
            raise TypeError('UQpy: `distance_object` cannot be NoneType')
        else:
            distance_fun = self.distance_object

        # If points_grassmann is not provided compute the distances on the manifold defined
        # by the left (psi) and right (phi) singular eigenvectors. In this case, use the information
        # set by the method manifold.
        if points_grassmann is None:

            # Compute the pairwise distances.
            points_distance_psi = self.__estimate_distance(self.psi, self.ranks, distance_fun)
            points_distance_phi = self.__estimate_distance(self.phi, self.ranks, distance_fun)

            # Return the parwise distances for the left and right singular eigenvectors.
            return points_distance_psi, points_distance_phi
        else:
            if isinstance(points_grassmann, np.ndarray):
                points_grassmann = points_grassmann.tolist()

            n_size = max(np.shape(points_grassmann[0]))
            for i in range(len(points_grassmann)):
                if n_size != max(np.shape(points_grassmann[i])):
                    raise TypeError('UQpy: The shape of the input matrices must be the same.')

            # if points_grasssmann is provided, use the shape of the input matrices to define 
            # the dimension of the p-planes defining the manifold of each individual input matrix.
            p_dim = []
            for i in range(len(points_grassmann)):
                p_dim.append(min(np.shape(np.array(points_grassmann[i]))))

            # Compute the pairwise distances.
            points_distance = self.__estimate_distance(points_grassmann, p_dim, distance_fun)

            # Return the pairwise distances.
            return points_distance

    @staticmethod
    def __estimate_distance(points, p_dim, distance_fun):

        """
        Private method: Estimate the distance between points on the Grassmann manifold.

        This is an auxiliary method to compute the pairwise distance of points on the Grassmann manifold. 
        The input arguments are passed through a list . Further, the user has the option to pass the dimension 
        of the embedding space.

        **Input:**

        * **points** (`list` or `ndarray`)
            Matrices (at least 2) corresponding to the points on the Grassmann manifold.
        
        * **p_dim** (`list`)
            Embedding dimension.

        **Output/Returns:**

        * **distance_list** (`list`)
            Pairwise distance.

        """

        # Check points for type and shape consistency.
        # -----------------------------------------------------------
        if not isinstance(points, list) and not isinstance(points, np.ndarray):
            raise TypeError('UQpy: The input matrices must be either list or numpy.ndarray.')

        nargs = len(points)

        if nargs < 2:
            raise ValueError('UQpy: At least two matrices must be provided.')

        # ------------------------------------------------------------

        # Define the pairs of points to compute the Grassmann distance.
        indices = range(nargs)
        pairs = list(itertools.combinations(indices, 2))

        # Compute the pairwise distances.
        distance_list = []
        for id_pair in range(np.shape(pairs)[0]):
            ii = pairs[id_pair][0]  # Point i
            jj = pairs[id_pair][1]  # Point j

            p0 = int(p_dim[ii])
            p1 = int(p_dim[jj])

            x0 = np.asarray(points[ii])[:, :p0]
            x1 = np.asarray(points[jj])[:, :p1]

            # Call the functions where the distance metric is implemented.
            distance_value = distance_fun(x0, x1)

            distance_list.append(distance_value)

        return distance_list

    # ==================================================================================================================
    # Built-in metrics are implemented in this section. Any new built-in metric must be implemented
    # here with the decorator @staticmethod.  

    @staticmethod
    def grassmann_distance(x0, x1):

        """
        Estimate the Grassmann distance.

        One of the distances defined on the Grassmann manifold is the Grassmann distance.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        * **x1** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        **Output/Returns:**

        * **distance** (`float`)
            Grassmann distance between x0 and x1.

        """

        if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            x0 = np.array(x0)

        if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            x1 = np.array(x1)

        l = min(np.shape(x0))
        k = min(np.shape(x1))
        rank = min(l, k)

        r = np.dot(x0.T, x1)
        (ui, si, vi) = svd(r, rank)

        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(np.diag(si))
        distance = np.sqrt(abs(k - l) * np.pi ** 2 / 4 + np.sum(theta ** 2))

        return distance

    @staticmethod
    def chordal_distance(x0, x1):

        """
        Estimate the chordal distance.

        One of the distances defined on the Grassmann manifold is the chordal distance.

        **Input:**

        * **x0** (`list` or `ndarray`) 
            Point on the Grassmann manifold.

        * **x1** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        **Output/Returns:**

        * **distance** (`float`)
            Chordal distance between x0 and x1.

        """

        if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            x0 = np.array(x0)

        if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            x1 = np.array(x1)

        l = min(np.shape(x0))
        k = min(np.shape(x1))
        rank = min(l, k)

        r_star = np.dot(x0.T, x1)
        (ui, si, vi) = svd(r_star, rank)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(np.diag(si))
        sin_sq = (np.sin(theta/2)) ** 2
        distance = 2*np.sqrt(0.5*abs(k - l) + np.sum(sin_sq))

        return distance

    @staticmethod
    def procrustes_distance(x0, x1):

        """
        Estimate the Procrustes distance.

        One of the distances defined on the Grassmann manifold is the Procrustes distance.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        * **x1** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        **Output/Returns:**

        * **distance** (`float`)
            Procrustes distance between x0 and x1.

        """

        if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            x0 = np.array(x0)

        if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            x1 = np.array(x1)

        l = min(np.shape(x0))
        k = min(np.shape(x1))
        rank = min(l, k)

        r = np.dot(x0.T, x1)
        (ui, si, vi) = svd(r, rank)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(np.diag(si))
        sin_t = np.sin(theta/2)
        distance = 2*np.sqrt(abs(k - l)*np.sqrt(2)/2 + np.sum(sin_t))

        return distance

    @staticmethod
    def projection_distance(x0, x1):

        """
        Estimate the projection distance.

        One of the distances defined on the Grassmann manifold is the projection distance.

        **Input:**

        * **x0** (`list` or `ndarray`) 
            Point on the Grassmann manifold.

        * **x1** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        **Output/Returns:**

        * **distance** (`float`)
            Projection distance between x0 and x1.

        """

        if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            x0 = np.array(x0)

        if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            x1 = np.array(x1)

        l = min(np.shape(x0))
        k = min(np.shape(x1))
        rank = min(l, k)

        r = np.dot(x0.T, x1)
        (ui, si, vi) = svd(r, rank)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(np.diag(si))
        sin_t = np.sin(theta) ** 2
        distance = np.sqrt(abs(k - l) + np.sum(sin_t))

        return distance
    
    @staticmethod
    def binet_cauchy_distance(x0, x1):

        """
        Estimate the Binet-Cauchy distance.

        One of the distances defined on the Grassmann manifold is the projection distance.

        **Input:**

        * **x0** (`list` or `ndarray`) 
            Point on the Grassmann manifold.

        * **x1** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        **Output/Returns:**

        * **distance** (`float`)
            Projection distance between x0 and x1.

        """

        if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            x0 = np.array(x0)

        if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            x1 = np.array(x1)

        l = min(np.shape(x0))
        k = min(np.shape(x1))
        rank = min(l, k)

        r = np.dot(x0.T, x1)
        (ui, si, vi) = svd(r, rank)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(np.diag(si))
        sin_t = np.cos(theta) ** 2
        distance = np.sqrt(1 - np.prod(sin_t))

        return distance

    # ==================================================================================================================
    def kernel(self, points_grassmann=None):

        """
        Compute a kernel matrix on the Grassmann manifold.

        This method computes the kernel matrix of points on the Grassmann manifold. The input arguments are passed 
        through a list. When the user call this method a ndarray containing the kernel matrix is returned as an
        output argument. Further, users are asked to provide the kernel method when the class Grassmann is instatiated.
        The current built-in options are the `projection_kernel` and the `binet_cauchy_kernel`, but the users have
        the option to implement their own kernel definition. However, the user must be aware that the points used
        to compute the kernel must be the points on the same manifold; therefore, it is important to ensure that the 
        dimension of all the input matrices are the same. Therefore, the following commands can be used:

        **Input:**

        * **points_grassmann** (`list` or `NoneType`) 
            Matrices (at least 2) corresponding to the points on the Grassmann manifold. If `points_grassmann` is not
            provided it means that the samples set by `manifold` is employed and kernel matrices are computed for the 
            points on the manifold defined by the left and right singular eigenvectors are computed.

        **Output/Returns:**

        * **kernel_matrix** (`list`)
            Kernel matrix if `points_grassmann` is provided.
                
        * **kernel_matrix_psi** (`list`)
            Kernel matrix on the manifold defined by the left singular eigenvectors if `points_grassmann` 
            is not provided.
            
        * **kernel_matrix_phi** (`list`)
            Kernel matrix on the manifold defined by the right singular eigenvectors if `points_grassmann` 
            is not provided.

        """

        # If points_grassmann is None get the information set using the method manifold.
        if points_grassmann is None:
            ranks = self.ranks

            # Check if all the input points belong to the same manifold.
            if ranks.count(ranks[0]) != len(ranks):
                raise TypeError('UQpy: the input points do not belog to the same manifold.')
            else:
                p_dim = ranks[0]

            # Compute the kernel matrix    
            kernel_psi = self.__estimate_kernel(self.psi, p_dim=p_dim)
            kernel_phi = self.__estimate_kernel(self.phi, p_dim=p_dim)

            # Return both kernel matrices.
            return kernel_psi, kernel_phi
        else:
            if isinstance(points_grassmann, np.ndarray):
                points_grassmann = points_grassmann.tolist()

            n_size = max(np.shape(points_grassmann[0]))
            for i in range(len(points_grassmann)):
                if n_size != max(np.shape(points_grassmann[i])):
                    raise TypeError('UQpy: The shape of the input matrices must be the same.')

            # Check the embedding dimension and its consistency.
            p_dim = []
            for i in range(len(points_grassmann)):
                p_dim.append(min(np.shape(np.array(points_grassmann[i]))))

            if p_dim.count(p_dim[0]) != len(p_dim):
                raise TypeError('UQpy: The input points do not belog to the same manifold.')
            else:
                p0 = p_dim[0]
                if p0 != self.p:
                    raise ValueError('UQpy: The input points do not belog to the manifold G(n,p).')

                    # Compute the kernel matrix.
            kernel_matrix = self.__estimate_kernel(np.array(points_grassmann), p0)

            # Return the kernel matrix.
            return kernel_matrix

    def __estimate_kernel(self, points, p_dim):

        """
        Private method: Compute a kernel matrix on the Grassmann manifold.

        This an auxiliary and private method to compute the kernel matrix of points on the Grassmann manifold.
        When using this method a ndarray containing the kernel matrix is returned as an output argument.

        **Input:**

        * **points** (`list` or `ndarray`)
            Matrices (at least 2) corresponding to the points on the Grassmann manifold.
        
        * **p_dim** (`int`)
            Embedding dimension.

        **Output/Returns:**

        * **kernel_matrix** (`list`)
            Kernel matrix.

        """

        # Check points for type and shape consistency.
        # -----------------------------------------------------------
        if not isinstance(points, list) and not isinstance(points, np.ndarray):
            raise TypeError('UQpy: `points` must be either list or numpy.ndarray.')

        nargs = len(points)

        if nargs < 2:
            raise ValueError('UQpy: At least two matrices must be provided.')
        # ------------------------------------------------------------

        # Define the pairs of points to compute the entries of the kernel matrix.
        indices = range(nargs)
        pairs = list(itertools.combinations(indices, 2))

        # Show an error message if no kernel_object is found.    
        if self.kernel_object is None:
            raise TypeError('UQpy: `kernel_object` cannot be NoneType')
        else:
            kernel_fun = self.kernel_object

        # Estimate off-diagonal entries of the kernel matrix.
        kernel_list = []
        for id_pair in range(np.shape(pairs)[0]):
            ii = pairs[id_pair][0]  # Point i
            jj = pairs[id_pair][1]  # Point j

            x0 = np.asarray(points[ii])[:, :p_dim]
            x1 = np.asarray(points[jj])[:, :p_dim]

            ker = kernel_fun(x0, x1)
            kernel_list.append(ker)

        # Diagonal entries of the kernel matrix.
        kernel_diag = []
        for id_elem in range(nargs):
            xd = np.asarray(points[id_elem])
            xd = xd[:, :p_dim]

            kerd = kernel_fun(xd, xd)
            kernel_diag.append(kerd)

        # Add the diagonals and off-diagonal entries of the Kernel matrix.
        kernel_matrix = sd.squareform(np.array(kernel_list)) + np.diag(kernel_diag)

        # Return the kernel matrix.
        return kernel_matrix

    # ==================================================================================================================
    @staticmethod
    def projection_kernel(x0, x1):

        """
        Estimate the value of the projection kernel between x0 and x1.

        One of the kernels defined on a manifold is the projection kernel.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        * **x1** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        **Output/Returns:**

        * **distance** (`float`)
            Kernel value for x0 and x1.

        """

        if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            x0 = np.array(x0)

        if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            x1 = np.array(x1)

        r = np.dot(x0.T, x1)
        n = np.linalg.norm(r, 'fro')
        ker = n * n
        return ker

    @staticmethod
    def binet_cauchy_kernel(x0, x1):

        """
        Estimate the value of the Binet-Cauchy kernel between x0 and x1.

        One of the kernels defined on a manifold is the Binet-Cauchy kernel.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        * **x1** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        **Output/Returns:**

        * **distance** (`float`)
            Kernel value for x0 and x1.

        """

        if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            x0 = np.array(x0)

        if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            x1 = np.array(x1)

        r = np.dot(x0.T, x1)
        det = np.linalg.det(r)
        ker = det * det
        return ker

    # ==================================================================================================================
    @staticmethod
    def log_map(points_grassmann=None, ref=None):

        """
        Mapping points from the Grassmann manifold on the tangent space.

        It maps the points on the Grassmann manifold, passed to the method using the input argument `points_grassmann`,
        onto the tangent space constructed on ref (a reference point on the Grassmann manifold).
        It is mandatory that the user pass a reference point to the method. Further, the reference point and the points
        in `points_grassmann` must belong to the same manifold.

        **Input:**

        * **points_grassmann** (`list`)
            Matrices (at least 2) corresponding to the points on the Grassmann manifold.

        * **ref** (`list` or `ndarray`)
            A point on the Grassmann manifold used as reference to construct the tangent space.

        **Output/Returns:**

        * **points_tan**: (`list`)
            Point on the tangent space.

        """

        # Show an error message if points_grassmann is not provided.
        if points_grassmann is None:
            raise TypeError('UQpy: No input data is provided.')

        # Show an error message if ref is not provided.
        if ref is None:
            raise TypeError('UQpy: No reference point is provided.')

        # Check points_grassmann for type consistency.
        if not isinstance(points_grassmann, list) and not isinstance(points_grassmann, np.ndarray):
            raise TypeError('UQpy: `points_grassmann` must be either a list or numpy.ndarray.')

        # Get the number of matrices in the set.
        nargs = len(points_grassmann)

        shape_0 = np.shape(points_grassmann[0])
        shape_ref = np.shape(ref)
        p_dim = []
        for i in range(nargs):
            shape = np.shape(points_grassmann[i])
            p_dim.append(min(np.shape(np.array(points_grassmann[i]))))
            if shape != shape_0:
                raise Exception('The input points are in different manifold.')

            if shape != shape_ref:
                raise Exception('The ref and points_grassmann are in different manifolds.')

        p0 = p_dim[0]

        # Check reference for type consistency.
        ref = np.asarray(ref)
        if not isinstance(ref, list):
            ref_list = ref.tolist()
        else:
            ref_list = ref
            ref = np.array(ref)

        # Multiply ref by its transpose.
        refT = ref.T
        m0 = np.dot(ref, refT)

        # Loop over all the input matrices.
        points_tan = []
        for i in range(nargs):
            utrunc = points_grassmann[i][:, 0:p0]

            # If the reference point is one of the given points
            # set the entries to zero.
            if utrunc.tolist() == ref_list:
                points_tan.append(np.zeros(np.shape(ref)))
            else:
                # compute: M = ((I - psi0*psi0')*psi1)*inv(psi0'*psi1)
                minv = np.linalg.inv(np.dot(refT, utrunc))
                m = np.dot(utrunc - np.dot(m0, utrunc), minv)
                ui, si, vi = np.linalg.svd(m, full_matrices=False)  # svd(m, max_rank)
                points_tan.append(np.dot(np.dot(ui, np.diag(np.arctan(si))), vi))

        # Return the points on the tangent space
        return points_tan

    @staticmethod
    def exp_map(points_tangent=None, ref=None):

        """
        Map points on the tangent space onto the Grassmann manifold.

        It maps the points on the tangent space, passed to the method using points_tangent, onto the Grassmann manifold. 
        It is mandatory that the user pass a reference point where the tangent space was created.

        **Input:**

        * **points_tangent** (`list`)
            Matrices (at least 2) corresponding to the points on the Grassmann manifold.

        * **ref** (`list` or `ndarray`)
            A point on the Grassmann manifold used as reference to construct the tangent space.
       
        **Output/Returns:**

        * **points_manifold**: (`list`)
            Point on the tangent space.

        """

        # Show an error message if points_tangent is not provided.
        if points_tangent is None:
            raise TypeError('UQpy: No input data is provided.')

        # Show an error message if ref is not provided.
        if ref is None:
            raise TypeError('UQpy: No reference point is provided.')

        # Test points_tangent for type consistency.
        if not isinstance(points_tangent, list) and not isinstance(points_tangent, np.ndarray):
            raise TypeError('UQpy: `points_tangent` must be either list or numpy.ndarray.')

        # Number of input matrices.
        nargs = len(points_tangent)

        shape_0 = np.shape(points_tangent[0])
        shape_ref = np.shape(ref)
        p_dim = []
        for i in range(nargs):
            shape = np.shape(points_tangent[i])
            p_dim.append(min(np.shape(np.array(points_tangent[i]))))
            if shape != shape_0:
                raise Exception('The input points are in different manifold.')

            if shape != shape_ref:
                raise Exception('The ref and points_grassmann are in different manifolds.')

        p0 = p_dim[0]

        # -----------------------------------------------------------

        ref = np.array(ref)
        # ref = ref[:,:p0]

        # Map the each point back to the manifold.
        points_manifold = []
        for i in range(nargs):
            utrunc = points_tangent[i][:, :p0]
            ui, si, vi = np.linalg.svd(utrunc, full_matrices=False)

            # Exponential mapping.
            x0 = np.dot(np.dot(np.dot(ref, vi.T), np.diag(np.cos(si))) + np.dot(ui, np.diag(np.sin(si))), vi)

            # Test orthogonality.
            xtest = np.dot(x0.T, x0)

            if not np.allclose(xtest, np.identity(np.shape(xtest)[0])):
                x0, unused = np.linalg.qr(x0)  # re-orthonormalizing.

            points_manifold.append(x0)

        return points_manifold

    def karcher_mean(self, points_grassmann=None, **kwargs):

        """
        Compute the Karcher mean.

        This method computes the Karcher mean given a set of points on the Grassmann manifold. The Karcher mean is
        estimated by the minimization of the Frechet variance, where the Frechet variance corresponds to the sum of the
        square distances, defined on the Grassmann manifold, to a given point. The command to compute the Karcher mean
        given a seto of points on the Grassmann manifold is.

        In this case two values are returned corresponding to the ones related to the manifolds defined by the left and
        right singular eigenvectors.

        **Input:**

        * **points_grassmann** (`list` or `ndarray`)
            Matrices (at least 2) corresponding to the points on the Grassmann manifold.
        
        * **kwargs** (`dictionary`)
            Contains the keywords for the used in the optimizers to find the Karcher mean. If ``gradient_descent`` is
            employed the keywords are `acc`, a boolean variable for the accelerated method; `tol`, tolerance with
            default value equal to 1e-3; and `maxiter`, maximum number of iterations with default value equal to 1000.
            If `stochastic_gradient_descent` is employed instead, `acc` is not used.

        **Output/Returns:**

        * **kr_mean** (`list`)
            Karcher mean.
            
        * **kr_mean_psi** (`list`)
            Karcher mean for left singular eigenvectors if `points_grassmann` is not provided.
            
        * **kr_mean_phi** (`list`)
            Karcher mean for right singular eigenvectors if `points_grassmann` is not provided.

        """

        # Show an error message if karcher_object is not provided.
        if self.karcher_object is None:
            raise TypeError('UQpy: `karcher_object` cannot be NoneType')
        else:
            karcher_fun = self.karcher_object

        if self.distance_object is None:
            raise TypeError('UQpy: `distance_object` cannot be NoneType')
        else:
            distance_fun = self.distance_object

        # Compute the Karcher mean for psi and phi if points_grassmann is not provided.
        if points_grassmann is None:
            kr_mean_psi = karcher_fun(self.psi, distance_fun, kwargs)
            kr_mean_phi = karcher_fun(self.phi, distance_fun, kwargs)

            # Return both mean values.  
            return kr_mean_psi, kr_mean_phi
        else:

            # Test the input data for type consistency.
            if not isinstance(points_grassmann, list) and not isinstance(points_grassmann, np.ndarray):
                raise TypeError('UQpy: `points_grassmann` must be either list or numpy.ndarray.')

            # Compute and test the number of input matrices necessary to compute the Karcher mean.
            nargs = len(points_grassmann)
            if nargs < 2:
                raise ValueError('UQpy: At least two matrices must be provided.')

            # Test the dimensionality of the input data.
            p = []
            for i in range(len(points_grassmann)):
                p.append(min(np.shape(np.array(points_grassmann[i]))))

            if p.count(p[0]) != len(p):
                raise TypeError('UQpy: The input points do not belog to the same manifold.')
            else:
                p0 = p[0]
                if p0 != self.p:
                    raise ValueError('UQpy: The input points do not belog to the manifold G(n,p).')

            kr_mean = karcher_fun(points_grassmann, distance_fun, kwargs)

            return kr_mean

    @staticmethod
    def gradient_descent(data_points, distance_fun, kwargs):

        """
        Compute the Karcher mean using the gradient descent method.

        This method computes the Karcher mean given a set of points on the Grassmann manifold. In this regard, the
        ``gradient_descent`` method is implemented herein also considering the acceleration scheme due to Nesterov.
        Further, this method is called by the method ``karcher_mean``.

        **Input:**

        * **data_points** (`list`)
            Points on the Grassmann manifold.
        
        * **distance_fun** (`callable`)
            Distance function.

        * **kwargs** (`dictionary`)
            Contains the keywords for the used in the optimizers to find the Karcher mean.

        **Output/Returns:**

        * **mean_element** (`list`)
            Karcher mean.

        """

        # acc is a boolean varible to activate the Nesterov acceleration scheme.
        if 'acc' in kwargs.keys():
            acc = kwargs['acc']
        else:
            acc = False

        # Error tolerance
        if 'tol' in kwargs.keys():
            tol = kwargs['tol']
        else:
            tol = 1e-3

        # Maximum number of iterations.
        if 'maxiter' in kwargs.keys():
            maxiter = kwargs['maxiter']
        else:
            maxiter = 1000

        # Number of points.
        n_mat = len(data_points)

        # =========================================
        alpha = 0.5
        rnk = []
        for i in range(n_mat):
            rnk.append(min(np.shape(data_points[i])))

        max_rank = max(rnk)
        fmean = []
        for i in range(n_mat):
            fmean.append(Grassmann.frechet_variance(data_points[i], data_points, distance_fun))

        index_0 = fmean.index(min(fmean))
        mean_element = data_points[index_0].tolist()

        avg_gamma = np.zeros([np.shape(data_points[0])[0], np.shape(data_points[0])[1]])

        itera = 0

        l = 0
        avg = []
        _gamma = []
        if acc:
            _gamma = Grassmann.log_map(points_grassmann=data_points, ref=np.asarray(mean_element))

            avg_gamma.fill(0)
            for i in range(n_mat):
                avg_gamma += _gamma[i] / n_mat
            avg.append(avg_gamma)

        # Main loop
        while itera <= maxiter:
            _gamma = Grassmann.log_map(points_grassmann=data_points, ref=np.asarray(mean_element))
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

            x = Grassmann.exp_map(points_tangent=[step], ref=np.asarray(mean_element))

            test_1 = np.linalg.norm(x[0] - mean_element, 'fro')

            if test_1 < tol:
                break

            mean_element = []
            mean_element = x[0]

            itera += 1

        # return the Karcher mean.
        return mean_element

    @staticmethod
    def stochastic_gradient_descent(data_points, distance_fun, kwargs):

        """
        Compute the Karcher mean using the stochastic gradient descent method.

        This method computes the Karcher mean given a set of points on the Grassmann manifold. In this regard, the
        ``stochastic_gradient_descent`` method is implemented herein. Further, this method is called by the method
        ``karcher_mean``.

        **Input:**

        * **data_points** (`list`)
            Points on the Grassmann manifold.
            
        * **distance_fun** (`callable`)
            Distance function.

        * **kwargs** (`dictionary`)
            Contains the keywords for the used in the optimizers to find the Karcher mean.

        **Output/Returns:**

        * **mean_element** (`list`)
            Karcher mean.

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
            fmean.append(Grassmann.frechet_variance(data_points[i], data_points, distance_fun))

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
                _gamma = Grassmann.log_map(points_grassmann=[data_points[idx]], ref=np.asarray(mean_element))

                step = 2 * alpha * _gamma[0]

                X = Grassmann.exp_map(points_tangent=[step], ref=np.asarray(mean_element))

                _gamma = []
                mean_element = X[0]

                k += 1

            test_1 = np.linalg.norm(mean_element - melem, 'fro')
            if test_1 < tol:
                break

            itera += 1

        return mean_element

    # Private method
    @staticmethod
    def frechet_variance(point_grassmann, points_grassmann, distance_fun):

        """
        Compute the Frechet variance.

        The Frechet variance corresponds to the summation of the square distances, on the manifold, to a given
        point also on the manifold. This method is employed in the minimization scheme used to find the Karcher mean.

        **Input:**

        * **point_grassmann** (`list` or `ndarray`)
            Point on the Grassmann manifold where the Frechet variance is computed.
            
        * **points_grassmann** (`list` or `ndarray`)
            Points on the Grassmann manifold.  
            
        * **distance_fun** (`callable`)
            Distance function.      

        **Output/Returns:**

        * **frechet_var** (`list`)
            Frechet variance.

        """
        p_dim = []
        for i in range(len(points_grassmann)):
            p_dim.append(min(np.shape(np.array(points_grassmann[i]))))

        nargs = len(points_grassmann)

        if nargs < 2:
            raise ValueError('UQpy: At least two input matrices must be provided.')

        accum = 0
        for i in range(nargs):
            distances = Grassmann.__estimate_distance([point_grassmann, points_grassmann[i]], p_dim, distance_fun)
            accum += distances[0] ** 2

        frechet_var = accum / nargs
        return frechet_var

    def interpolate(self, coordinates, point, element_wise=True):

        """
        Interpolate a point on the Grassmann manifold given the samples in the ambient space (sample space).

        Interpolate a `point` on the Grassmann manifold given the `coordinates`, support points, and the `samples`.
        Further, the user must select the option `element_wise` to perform the interpolation in the entries of the input
        matrices, if `point` and `samples` are matrices. The samples related to `coordinates` are set using
        `manifold`. For example, the following command is used to perform the interpolation.

        On the other hand, if a scikit learn gaussian_process object is provided, one can use the following commands:

        **Input:**

        * **coordinates** (`list` or `ndarray`)
            Coordinate of the support samples.
            
        * **point** (`list` or `ndarray`)
            Point to be interpolated.      

        * **element_wise** (`bool`)
            Element wise interpolation. 
            
        **Output/Returns:**

        * **interpolated** (`list`)
            Interpolated point.

        """

        # Find the Karcher mean.
        ref_psi, ref_phi = self.karcher_mean()

        # Reshape the vector containing the singular values as a diagonal matrix.
        sigma_m = []
        for i in range(len(self.sigma)):
            sigma_m.append(np.diag(self.sigma[i]))

        # Project the points on the manifold to the tangent space created over the Karcher mean.
        gamma_psi = self.log_map(points_grassmann=self.psi, ref=ref_psi)
        gamma_phi = self.log_map(points_grassmann=self.phi, ref=ref_phi)

        # Perform the interpolation in the tangent space.
        interp_psi = self.interpolate_sample(coordinates=coordinates, samples=gamma_psi, point=point,
                                             element_wise=element_wise)
        interp_phi = self.interpolate_sample(coordinates=coordinates, samples=gamma_phi, point=point,
                                             element_wise=element_wise)
        interp_sigma = self.interpolate_sample(coordinates=coordinates, samples=sigma_m, point=point,
                                               element_wise=element_wise)

        # Map the interpolated point back to the manifold.
        psi_tilde = self.exp_map(points_tangent=[interp_psi], ref=ref_psi)
        phi_tilde = self.exp_map(points_tangent=[interp_phi], ref=ref_phi)

        # Estimate the interpolated solution.
        psi_tilde = np.array(psi_tilde[0])
        phi_tilde = np.array(phi_tilde[0])
        interpolated = np.dot(np.dot(psi_tilde, interp_sigma), phi_tilde.T)

        return interpolated

    def interpolate_sample(self, coordinates, samples, point, element_wise=True):

        """
        Interpolate a point on the tangent space.

        Once the points on the Grassmann manifold are projected onto the tangent space standard interpolation can be
        performed. In this regard, the user should provide the data points, the coordinates of each input data point,
        and the point to be interpolated. Furthermore, additional parameters, depending on the selected interpolation
        method, can be provided via kwargs. In comparison to ``interpolate``, here the samples prodived are points on
        the TANGENT SPACE.

        **Input:**

        * **coordinates** (`list` or `ndarray`)
            Coordinates of the input data points.

        * **samples** (`list` or `ndarray`)
            Matrices corresponding to the points on the tangent space.

        * **point** (`list` or `ndarray`)
            Coordinates of the point to be interpolated.

        * **element_wise** (`bool`)
            Boolean variable for the element wise intepolation of a matrix.

        **Output/Returns:**

        * **interp_point** (`ndarray`)
            Interpolated point on the tangent space.

        """

        if isinstance(samples, list):
            samples = np.array(samples)

        # Test if the sample is stored as a list
        if isinstance(point, list):
            point = np.array(point)

        # Test if the nodes are stored as a list
        if isinstance(coordinates, list):
            coordinates = np.array(coordinates)

        nargs = len(samples)

        if self.interp_object is None:
            raise TypeError('UQpy: `interp_object` cannot be NoneType')
        else:
            if self.interp_object is Grassmann.linear_interp:
                element_wise = False

            if isinstance(self.interp_object, Kriging):
                #K = self.interp_object
                element_wise = True
            else:
                interp_fun = self.interp_object

        shape_ref = np.shape(samples[0])
        for i in range(1, nargs):
            if np.shape(samples[i]) != shape_ref:
                raise TypeError('UQpy: Input matrices have different shape.')

        if element_wise:

            shape_ref = np.shape(samples[0])
            interp_point = np.zeros(shape_ref)
            nrows = samples[0].shape[0]
            ncols = samples[0].shape[1]

            val_data = []
            dim = np.shape(coordinates)[1]

            for j in range(nrows):
                for k in range(ncols):
                    val_data = []
                    for i in range(nargs):
                        val_data.append([samples[i][j, k]])

                    # if all the elements of val_data are the same.
                    if val_data.count(val_data[0]) == len(val_data):
                        val = np.array(val_data)
                        y = val[0]
                    else:
                        val_data = np.array(val_data)
                        self.skl_str = "<class 'sklearn.gaussian_process.gpr.GaussianProcessRegressor'>"
                        if isinstance(self.interp_object, Kriging) or self.skl:
                            self.interp_object.fit(coordinates, val_data)
                            y = self.interp_object.predict(point, return_std=False)
                        else:
                            y = interp_fun(coordinates, samples, point)

                    interp_point[j, k] = y

        else:
            if isinstance(self.interp_object, Kriging):
                raise TypeError('UQpy: Kriging only can be used in the elementwise interpolation.')
            else:
                interp_point = interp_fun(coordinates, samples, point)

        return interp_point

    # ==================================================================================================================
    # The pre-defined interpolators are implemented in this section. Any new pre-defined interpolator must be
    # implemented here with the decorator @staticmethod.

    @staticmethod
    def linear_interp(coordinates, samples, point):

        """
        Interpolate a point using the linear interpolation.

        For the linear interpolation the user are asked to provide the data points, the coordinates of the data points,
        and the coordinate of the point to be interpolated.

        **Input:**

        * **coordinates** (`ndarray`)
            Coordinates of the input data points.

        * **samples** (`ndarray`)
            Matrices corresponding to the points on the Grassmann manifold.

        * **point** (`ndarray`)
            Coordinates of the point to be interpolated.

        **Output/Returns:**

        * **interp_point** (`ndarray`)
            Interpolated point.

        """

        if not isinstance(coordinates, list) and not isinstance(coordinates, np.ndarray):
            raise TypeError('UQpy: `coordinates` must be either list or ndarray.')
        else:
            coordinates = np.array(coordinates)

        if not isinstance(samples, list) and not isinstance(samples, np.ndarray):
            raise TypeError('UQpy: `samples` must be either list or ndarray.')
        else:
            samples = np.array(samples)

        if not isinstance(point, list) and not isinstance(point, np.ndarray):
            raise TypeError('UQpy: `point` must be either list or ndarray.')
        else:
            point = np.array(point)

        myInterpolator = LinearNDInterpolator(coordinates, samples)
        interp_point = myInterpolator(point)
        interp_point = interp_point[0]

        return interp_point


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


########################################################################################################################
########################################################################################################################
#                                                     POD                                                              #
########################################################################################################################
########################################################################################################################


class POD:
    """
    Performs Direct and Snapshot Proper Orthogonal Decomposition (POD) as well as Higher-order Singular Value
    Decomposition (HOSVD) for dimension reduction of datasets.

    **Input:**

    * **input_sol** (`ndarray`) or (`list`):
        Second order tensor or list containing the solution snapshots. Third dimension or length of list corresponds
        to the number of snapshots.

    * **verbose** (`Boolean`):
        A boolean declaring whether to write text to the terminal.

    **Methods:**
    """

    def __init__(self, input_sol, verbose=True, **kwargs):

        self.input_sol = input_sol
        self.verbose = verbose
        self.kwargs = kwargs

    @staticmethod
    def unfold(data):
        """
        Method for unfolding second order tensors.

        **Input:**

        * **data** (`ndarray`) or (`list`):
            Input second order tensor to be unfolded.

        **Output/Returns:**

        * **M0, M1, M2** (`ndarrays`):
            Returns the 2-dimensional unfolded matrices.
        """

        if type(data) == list:
            x, y, z = data[0].shape[0], data[0].shape[1], len(data)
            data_ = np.zeros((x, y, z))
            for i in range(z):
                data_[:, :, i] = data[i]
            del data
            data = np.copy(data_)

        d0, d1, d2 = [0, 2, 1], [1, 2, 0], [2, 0, 1]
        z0, z1, z2 = np.transpose(data, d0), np.transpose(data, d1), np.transpose(data, d2)

        m0 = z0.reshape(data.shape[0], data.shape[2] * data.shape[1])
        m1 = z1.reshape(data.shape[1], data.shape[2] * data.shape[0])
        m2 = z2.reshape(data.shape[2], data.shape[0] * data.shape[1])

        return m0, m1, m2


class DirectPOD(POD):
    """
    Direct POD child class generates a set of spatial modes and time coefficients to approximate the solution.

    **Input:**

    * **input_sol** (`ndarray`) or (`list`):
        Second order tensor or list containing the solution snapshots. Third dimension or length of list corresponds
        to the number of snapshots.

    * **modes** (`int`):
        Number of POD modes used to approximate the input solution. Must be less than or equal
        to the number of grid points.

    * **reconstr_perc** (`float`):
        Dataset reconstruction percentage.

    **Methods:**
    """

    def __init__(self, input_sol, modes=10**10, reconstr_perc=10**10, verbose=False):

        super().__init__(input_sol, verbose)
        self.verbose = verbose
        self.modes = modes
        self.reconstr_perc = reconstr_perc

    def run(self):

        """
        Executes the Direct POD method in the ''Direct'' class.

        **Output/Returns:**

        * **reconstructed_solutions** (`ndarray`):
            Second order tensor containing the reconstructed solution snapshots in their initial spatial and
            temporal dimensions.

        * **reduced_solutions** (`ndarray`):
            An array containing the solution snapshots reduced in the spatial dimension.

        """

        if type(self.input_sol) == list:

            x, y, z = self.input_sol[0].shape[0], self.input_sol[0].shape[1], len(self.input_sol)
            u = np.zeros((z, x * y))

            for i in range(z):
                u[i, :] = self.input_sol[i].ravel()

        else:
            x, y, z = self.input_sol.shape[0], self.input_sol.shape[1], self.input_sol.shape[2]
            u = np.zeros((z, x * y))

            for i in range(z):
                u[i, :] = self.input_sol[:, :, i].ravel()

        c = np.dot(u.T, u) / (z - 1)
        eigval, phi = np.linalg.eig(c)
        phi = phi.real
        eigval_ = eigval.real
        a = np.dot(u, phi)

        if self.modes <= 0:
            print('Warning: Invalid input, the number of modes must be positive.')
            return [], []

        elif self.reconstr_perc <= 0:
            print('Warning: Invalid input, the reconstruction percentage is defined in the range (0,100].')
            return [], []

        elif self.modes != 10**10 and self.reconstr_perc != 10**10:
            print('Warning: Either a number of modes or a reconstruction percentage must be chosen, not both.')
            return [], []

        elif type(self.modes) != int:
            print('Warning: The number of modes must be an integer.')
            return [], []

        else:

            perc = []
            for i in range(x * y):
                perc.append((eigval_[:i + 1].sum() / eigval_.sum()) * 100)

            percentage = min(perc, key=lambda x: abs(x - self.reconstr_perc))

            if self.modes == 10**10:

                self.modes = perc.index(percentage) + 1

            else:

                if self.modes > x * y:
                    print("Warning: A number of modes greater than the number of dimensions was given.")
                    print("Number of dimensions is {}".format(x * y))

            reconstructed_solutions_ = np.dot(a[:, :self.modes], phi[:, :self.modes].T)
            reduced_solutions = np.dot(u, phi[:, :self.modes])

            reconstructed_solutions = np.zeros((x, y, z))
            for i in range(z):
                reconstructed_solutions[0:x, 0:y, i] = reconstructed_solutions_[i, :].reshape((x, y))

            if self.verbose:
                print("UQpy: Successful execution of Direct POD!")
                if z < x * y and x * y > 1000:
                    print("Snapshot POD is recommended.")

            if self.verbose:
                print('Dataset reconstruction: {:.3%}'.format(perc[self.modes - 1] / 100))

            return reconstructed_solutions, reduced_solutions


class SnapshotPOD(POD):
    """
    Snapshot POD child class generates a set of temporal modes and spatial coefficients to approximate the solution.
    (Faster that direct POD)

    **Input:**

    * **input_sol** (`ndarray`) or (`list`):
        Second order tensor or list containing the solution snapshots. Third dimension or length of list corresponds
        to the number of snapshots.

    * **modes** (`int`):
        Number of POD modes used to approximate the input solution. Must be less than or equal
        to the number of grid points.

    * **reconstr_perc** (`float`):
        Dataset reconstruction percentage.

    **Methods:**
   """

    def __init__(self, input_sol, modes=10**10, reconstr_perc=10**10, verbose=False):

        super().__init__(input_sol, verbose)
        self.verbose = verbose
        self.modes = modes
        self.reconstr_perc = reconstr_perc

    def run(self):
        """
        Executes the Snapshot POD method in the ''Snapshot'' class.

        **Output/Returns:**

        * **reconstructed_solutions** (`ndarray`):
            Second order tensor containing the reconstructed solution snapshots in their initial spatial and
            temporal dimensions.

        * **reduced_solutions** (`ndarray`):
            An array containing the solution snapshots reduced in the temporal dimension.

        """
        if type(self.input_sol) == list:

            x, y, z = self.input_sol[0].shape[0], self.input_sol[0].shape[1], len(self.input_sol)
            u = np.zeros((z, x * y))

            for i in range(z):
                u[i, :] = self.input_sol[i].ravel()

        else:
            x, y, z = self.input_sol.shape[0], self.input_sol.shape[1], self.input_sol.shape[2]
            u = np.zeros((z, x * y))

            for i in range(z):
                u[i, :] = self.input_sol[:, :, i].ravel()

        c_s = np.dot(u, u.T) / (z - 1)

        eigval, a_s = np.linalg.eig(c_s)
        a_s = a_s.real
        eigval_ = eigval.real

        if self.modes <= 0:
            print('Warning: Invalid input, the number of modes must be positive.')
            return [], []

        elif self.reconstr_perc <= 0:
            print('Warning: Invalid input, the reconstruction percentage is defined in the range (0,100].')
            return [], []

        elif self.modes != 10**10 and self.reconstr_perc != 10**10:
            print('Warning: Either a number of modes or a reconstruction percentage must be chosen, not both.')
            return [], []

        elif type(self.modes) != int:
            print('Warning: The number of modes must be an integer.')
            return [], []

        else:

            perc = []
            for i in range(z):
                perc.append((eigval_[:i + 1].sum() / eigval_.sum()) * 100)

            percentage = min(perc, key=lambda x: abs(x - self.reconstr_perc))

            if self.modes == 10**10:

                self.modes = perc.index(percentage) + 1

            else:

                if self.modes > z:
                    print("Warning: A number of modes greater than the number of dimensions was given.")
                    print("Number of dimensions is {}".format(z))

            phi_s = np.dot(u.T, a_s)
            reconstructed_solutions_ = np.dot(a_s[:, :self.modes], phi_s[:, :self.modes].T)
            reduced_solutions_ = (np.dot(u.T, a_s[:, :self.modes])).T

            reconstructed_solutions = np.zeros((x, y, z))
            reduced_solutions = np.zeros((x, y, self.modes))

            for i in range(z):
                reconstructed_solutions[0:x, 0:y, i] = reconstructed_solutions_[i, :].reshape((x, y))

            for i in range(self.modes):
                reduced_solutions[0:x, 0:y, i] = reduced_solutions_[i, :].reshape((x, y))

            if self.verbose:
                print("UQpy: Successful execution of Snapshot POD!")

            if self.verbose:
                print('Dataset reconstruction: {:.3%}'.format(perc[self.modes - 1] / 100))

            return reconstructed_solutions, reduced_solutions


class HOSVD(POD):
    """
    HOSVD child class is used for higher-order singular value decomposition on the input solutions tensor.

    **Inputs:**

    * **input_sol** (`ndarray`) or (`list`):
        Second order tensor or list containing the solution snapshots. Third dimension or length of list corresponds
        to the number of snapshots.

    * **modes** (`int`):
        Number of modes to keep for dataset reconstruction.

    * **reconstr_perc** (`float`):
        Dataset reconstruction percentage

    **Methods:**
    """

    def __init__(self, input_sol, modes=10**10, reconstr_perc=10**10, verbose=False):

        super().__init__(input_sol, verbose)
        self.verbose = verbose
        self.modes = modes
        self.reconstr_perc = reconstr_perc

    def run(self, get_error=False):
        """
        Executes the HOSVD method in the ''HOSVD'' class.

        **Input:**

        * **get_error** (`Boolean`):
            A boolean declaring whether to return the reconstruction error.

        **Output/Returns:**

        * **reconstructed_solutions** (`ndarray`):
            Second order tensor containing the reconstructed solution snapshots in their initial spatial and
            temporal dimensions.

        * **reduced_solutions** (`ndarray`):
            An array containing the reduced solutions snapshots. The array's dimensions depends on the dimensions
            of input second order tensor and not on the input number of modes or reconstruction percentage.

        """

        if type(self.input_sol) == list:
            x, y, z = self.input_sol[0].shape[0], self.input_sol[0].shape[1], len(self.input_sol)
        else:
            x, y, z = self.input_sol.shape[0], self.input_sol.shape[1], self.input_sol.shape[2]

        a1, a2, a3 = POD.unfold(self.input_sol)

        u1, sig_1, v1 = np.linalg.svd(a1, full_matrices=True)
        u2, sig_2, v2 = np.linalg.svd(a2, full_matrices=True)
        u3, sig_3, v3 = np.linalg.svd(a3, full_matrices=True)

        sig_3_ = np.diag(sig_3)
        hold = np.dot(np.linalg.inv(u3), a3)
        kron_ = np.kron(u1, u2)

        s3 = np.array(np.dot(hold, np.linalg.inv(kron_.T)))

        if self.modes <= 0:
            print('Warning: Invalid input, the number of modes must be positive.')
            return [], []

        elif self.reconstr_perc <= 0:
            print('Warning: Invalid input, the reconstruction percentage is defined in the range (0,100].')
            return [], []

        elif self.modes != 10**10 and self.reconstr_perc != 10**10:
            print('Warning: Either a number of modes or a reconstruction percentage must be chosen, not both.')
            return [], []

        elif type(self.modes) != int:
            print('Warning: The number of modes must be an integer.')
            return [], []

        else:

            if self.modes == 10**10:
                error_ = []
                for i in range(0, x):
                    error_.append(np.sqrt(((sig_3[i + 1:]) ** 2).sum()) / np.sqrt((sig_3 ** 2).sum()))
                    if i == x:
                        error_.append(0)
                error = [i * 100 for i in error_]
                error.reverse()
                perc = error.copy()
                percentage = min(perc, key=lambda x: abs(x-self.reconstr_perc))
                self.modes = perc.index(percentage) + 1

            else:
                if self.modes > x:
                    print("Warning: A number of modes greater than the number of temporal dimensions was given.")
                    print("Number of temporal dimensions is {}.".format(x))

            reduced_solutions = np.dot(u3, sig_3_)
            u3hat = np.dot(u3[:, :self.modes], sig_3_[:self.modes, :self.modes])
            s3hat = np.dot(np.linalg.inv(sig_3_[:self.modes, :self.modes]), s3[:self.modes, :])

            b = np.kron(u1, u2)
            c = np.dot(s3hat, b.T)
            d = np.dot(u3hat[:, :], c)

            reconstructed_solutions = np.zeros((x, y, z))
            for i in range(z):
                reconstructed_solutions[0:x, 0:y, i] = d[i, :].reshape((x, y))

            if self.verbose:
                print("UQpy: Successful execution of HOSVD!")

            if self.modes == 10**10:
                if self.verbose:
                    print('Dataset reconstruction: {0:.3%}'.format(percentage / 100))

            else:
                if get_error:
                    error_rec = np.sqrt(((sig_3[self.modes:]) ** 2).sum()) / np.sqrt((sig_3 ** 2).sum())
                    print("Reduced-order reconstruction error: {0:.3%}".format(error_rec))

            return reconstructed_solutions, reduced_solutions
