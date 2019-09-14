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

"""This module contains functionality for all the sampling methods supported in UQpy."""

import copy
from scipy.spatial.distance import pdist
import random
from UQpy.Distributions import *
from UQpy.Utilities import *
from os import sys
from inspect import signature
from functools import partial


########################################################################################################################
########################################################################################################################
#                                         Monte Carlo Simulation
########################################################################################################################


class MCS:
    """
        Description:

            Perform Monte Carlo sampling (MCS) of independent random variables from a user-specified probability
            distribution using inverse transform method.

        Input:
            :param dist_name: A string or string list containing the names of the distributions of the random variables.
            Distribution names must match those in the Distributions module.
            If the distribution does not match one from the Distributions module, the user must provide a custom
            distribution file with name dist_name.py. See documentation for the Distributions module. The length of the
            list must equal the dimension of the random vector.
            :type dist_name: string or string list

            :param dist_params: Parameters of the distribution.
            Parameters for each random variable are defined as ndarrays.
            Each item in the list, dist_params[i], specifies the parameters for the corresponding distribution,
            dist_name[i]. Relevant parameters for each distribution can be found in the documentation for the
            Distributions module.
            :type dist_params: ndarray or list

            :param nsamples: Number of samples to generate.
            No Default Value: nsamples must be prescribed.
            :type nsamples: int

            :param var_names: names of variables
            :type var_names: list of strings

            :param verbose: A boolean declaring whether to write text to the terminal.
            :type verbose: bool

        Output:
            :return: MCS.samples: Set of generated samples
            :rtype: MCS.samples: ndarray of dimension (nsamples, ndim)

    """

    # Authors: Dimitris G.Giovanis
    # Last Modified: 11/12/2018 by Audrey Olivier

    def __init__(self, dist_name=None, dist_params=None, nsamples=None, var_names=None, verbose=False):

        if nsamples is None:
            raise ValueError('UQpy error: nsamples must be defined.')
        # No need to do other checks as they will be done within Distributions.py
        self.dist_name = dist_name
        self.dist_params = dist_params
        self.nsamples = nsamples
        self.var_names = var_names
        if verbose:
            print('UQpy: Running Monte Carlo Sampling...')
        self.samples = Distribution(dist_name=self.dist_name).rvs(params=self.dist_params, nsamples=nsamples)
        self.samplesU01 = np.zeros_like(self.samples)
        for i in range(self.samples.shape[1]):
            self.samplesU01[:,i] = Distribution(dist_name=self.dist_name[i]).cdf(x=self.samples[:,i],
                                                                            params=self.dist_params[i])

        if verbose:
            print('UQpy: Monte Carlo Sampling Complete.')

        # Shape the array as (1,n) if nsamples=1, and (n,1) if nsamples=n
        if len(self.samples.shape) == 1:
            if self.nsamples == 1:
                self.samples = self.samples.reshape((1, -1))
            else:
                self.samples = self.samples.reshape((-1, 1))


########################################################################################################################
########################################################################################################################
#                                         Latin hypercube sampling  (LHS)
########################################################################################################################

class LHS:
    """
        Description:

            A class that creates a Latin Hypercube Design for experiments. Samples on hypercube [0, 1]^n  and on the
            parameter space are generated.

        Input:
            :param dimension: A scalar value defining the dimension of the random variables.
                              If dimension is not provided then dimension is equal to the length of the dist_name.
            :type dimension: int

            :param dist_name: A list containing the names of the distributions of the random variables.
                              Distribution names must match those in the Distributions module.
                              If the distribution does not match one from the Distributions module, the user must
                              provide custom_dist.py.
                              The length of the string must be 1 (if all distributions are the same) or equal to
                              dimension.
            :type dist_name: string list

            :param dist_params: Parameters of the distribution.
                                Parameters for each random variable are defined as ndarrays.
                                Each item in the list, dist_params[i], specifies the parameters for the corresponding
                                distribution, dist[i].
            :type dist_params: list

            param: distribution: An object list containing the distributions of the random variables.
                                 Each item in the list is an object of the Distribution class (see Distributions.py).
                                 The list has length equal to dimension.
            :type distribution: list

            :param lhs_criterion: The criterion for generating sample points
                                  Options:
                                        1. 'random' - completely random \n
                                        2. 'centered' - points only at the centre \n
                                        3. 'maximin' - maximising the minimum distance between points \n
                                        4. 'correlate' - minimizing the correlation between the points \n
                                  Default: 'random'
            :type lhs_criterion: str

            :param lhs_metric: The distance metric to use. Supported metrics are:
                               'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice',
                               'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
                               'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
                               'sqeuclidean', 'yule'.
                                Default: 'euclidean'.
            :type lhs_metric: str

            :param lhs_iter: The number of iteration to run. Required only for maximin, correlate and criterion.
                             Default: 100
            :type lhs_iter: int

            :param nsamples: Number of samples to generate.
                             No Default Value: nsamples must be prescribed.
            :type nsamples: int

        Output:
            :return: LHS.samples: Set of LHS samples
            :rtype: LHS.samples: ndarray

            :return: LHS.samplesU01: Set of uniform LHS samples on [0, 1]^dimension.
            :rtype: LHS.samplesU01: ndarray.

    """

    # Created by: Lohit Vandanapu
    # Last modified: 6/20/2018 by Dimitris G. Giovanis

    def __init__(self, dist_name=None, dist_params=None, lhs_criterion='random', lhs_metric='euclidean',
                 lhs_iter=100, var_names = None, nsamples=None, verbose=False):

        self.nsamples = nsamples
        self.dist_name = dist_name
        self.dist_params = dist_params
        self.dimension = len(self.dist_name)
        self.lhs_criterion = lhs_criterion
        self.lhs_metric = lhs_metric
        self.lhs_iter = lhs_iter
        self.init_lhs()
        self.var_names = var_names

        self.distribution = [None] * self.dimension
        for i in range(self.dimension):
            self.distribution[i] = Distribution(dist_name=self.dist_name[i])

        self.samplesU01, self.samples = self.run_lhs()

    def run_lhs(self):

        cut = np.linspace(0, 1, self.nsamples + 1)
        a = cut[:self.nsamples]
        b = cut[1:self.nsamples + 1]

        samples = self._samples(a, b)

        samples_u_to_x = np.zeros_like(samples)
        for j in range(samples.shape[1]):
            i_cdf = self.distribution[j].icdf
            samples_u_to_x[:, j] = i_cdf(samples[:, j], self.dist_params[j])

        print('Successful execution of LHS design..')
        return samples, samples_u_to_x

    def _samples(self, a, b):

        if self.lhs_criterion == 'random':
            return self._random(a, b)
        elif self.lhs_criterion == 'centered':
            return self._centered(a, b)
        elif self.lhs_criterion == 'maximin':
            return self._max_min(a, b)
        elif self.lhs_criterion == 'correlate':
            return self._correlate(a, b)

    def _random(self, a, b):
        u = np.random.rand(self.nsamples, self.dimension)
        samples = np.zeros_like(u)

        for i in range(self.dimension):
            samples[:, i] = u[:, i] * (b - a) + a

        for j in range(self.dimension):
            order = np.random.permutation(self.nsamples)
            samples[:, j] = samples[order, j]

        return samples

    def _centered(self, a, b):

        samples = np.zeros([self.nsamples, self.dimension])
        centers = (a + b) / 2

        for i in range(self.dimension):
            samples[:, i] = np.random.permutation(centers)

        return samples

    def _max_min(self, a, b):

        max_min_dist = 0
        samples = self._random(a, b)
        for _ in range(self.lhs_iter):
            samples_try = self._random(a, b)
            d = pdist(samples_try, metric=self.lhs_metric)
            if max_min_dist < np.min(d):
                max_min_dist = np.min(d)
                samples = copy.deepcopy(samples_try)

        print('Achieved max_min distance of ', max_min_dist)

        return samples

    def _correlate(self, a, b):

        min_corr = np.inf
        samples = self._random(a, b)
        for _ in range(self.lhs_iter):
            samples_try = self._random(a, b)
            r = np.corrcoef(np.transpose(samples_try))
            np.fill_diagonal(r, 1)
            r1 = r[r != 1]
            if np.max(np.abs(r1)) < min_corr:
                min_corr = np.max(np.abs(r1))
                samples = copy.deepcopy(samples_try)

        print('Achieved minimum correlation of ', min_corr)

        return samples

    ################################################################################################################
    # Latin hypercube checks.
    # Necessary parameters:  1. Probability distribution, 2. Probability distribution parameters
    # Optional: number of samples (default 100), criterion, metric, iterations

    def init_lhs(self):

        # Ensure that the number of samples is defined
        if self.nsamples is None:
            raise NotImplementedError("Exit code: Number of samples not defined.")

        # Check the dimension
        if self.dimension is None:
            self.dimension = len(self.dist_name)

        # Ensure that distribution parameters are assigned
        if self.dist_params is None:
            raise NotImplementedError("Exit code: Distribution parameters not defined.")

        # Check dist_params
        if type(self.dist_params).__name__ != 'list':
            self.dist_params = [self.dist_params]
        if len(self.dist_params) == 1 and self.dimension != 1:
            self.dist_params = self.dist_params * self.dimension
        elif len(self.dist_params) != self.dimension:
            raise NotImplementedError("Length of dist_params list should be 1 or equal to dimension.")

        # Check for dimensional consistency
        if len(self.dist_name) != len(self.dist_params):
            raise NotImplementedError("Exit code: Incompatible dimensions.")

        if self.lhs_criterion is None:
            self.lhs_criterion = 'random'
        else:
            if self.lhs_criterion not in ['random', 'centered', 'maximin', 'correlate']:
                raise NotImplementedError("Exit code: Supported lhs criteria: 'random', 'centered', 'maximin', "
                                          "'correlate'.")

        if self.lhs_metric is None:
            self.lhs_metric = 'euclidean'
        else:
            if self.lhs_metric not in ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine',
                                       'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
                                       'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                                       'sokalmichener', 'sokalsneath', 'sqeuclidean']:
                raise NotImplementedError("Exit code: Supported lhs distances: 'braycurtis', 'canberra', 'chebyshev', "
                                          "'cityblock',"
                                          " 'correlation', 'cosine','dice', 'euclidean', 'hamming', 'jaccard', "
                                          "'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',"
                                          "'russellrao', 'seuclidean','sokalmichener', 'sokalsneath', 'sqeuclidean'.")

        if self.lhs_iter is None or self.lhs_iter == 0:
            self.lhs_iter = 1000
        elif self.lhs_iter is not None:
            self.lhs_iter = int(self.lhs_iter)


########################################################################################################################
########################################################################################################################
#                                         Stratified Sampling  (STS)
########################################################################################################################
class STS:
    """
        Description:

            Generate samples from an assigned probability density function using Stratified Sampling.

            References:
            M.D. Shields, K. Teferra, A. Hapij, and R.P. Daddazio, "Refined Stratified Sampling for efficient Monte
            Carlo based uncertainty quantification," Reliability Engineering and System Safety,vol.142, pp.310-325,2015.

        Input:
            :param dimension: A scalar value defining the dimension of target density function.
                              Default: Length of sts_design.
            :type dimension: int

            :param dist_name: A list containing the names of the distributions of the random variables.
                              Distribution names must match those in the Distributions module.
                              If the distribution does not match one from the Distributions module, the user must
                              provide custom_dist.py.
                              The length of the string must be 1 (if all distributions are the same) or equal to
                              dimension.
            :type dist_name: string list

            :param dist_params: Parameters of the distribution
                                Parameters for each random variable are defined as ndarrays.
                                Each item in the list, dist_params[i], specifies the parameters for the corresponding
                                distribution, dist[i].
            :type dist_params: list

            param: distribution: An object list containing the distributions of the random variables.
                                 Each item in the list is an object of the Distribution class (see Distributions.py).
                                 The list has length equal to dimension.
            :type distribution: list

            :param sts_design: Specifies the number of strata in each dimension
            :type sts_design: int list

            :param input_file: File path to input file specifying stratum origins and stratum widths.
                               Default: None.
            :type input_file: string

        Output:
            :return: STS.samples: Set of stratified samples.
            :rtype: STS.samples: ndarray

            :return: STS.samplesU01: Set of uniform stratified samples on [0, 1]^dimension
            :rtype: STS.samplesU01: ndarray

            :return: STS.strata: Instance of the class SampleMethods.Strata
            :rtype: STS.strata: ndarray

    """

    # Authors: Michael Shields
    # Last modified: 6/7/2018 by Dimitris Giovanis & Michael Shields

    def __init__(self, dimension=None, dist_name=None, dist_params=None, sts_design=None, input_file=None,
                 sts_criterion="random", stype='Rectangular', nsamples=None, n_iters=20):

        if dimension is None:
            self.dimension = len(dist_name)
        else:
            self.dimension = dimension
        self.stype = stype
        self.sts_design = sts_design
        self.input_file = input_file
        self.dist_name = dist_name
        self.dist_params = dist_params
        self.strata = None
        self.sts_criterion = sts_criterion
        self.nsamples = nsamples

        if self.stype == 'Voronoi':
            self.n_iters = n_iters


        self.distribution = [None] * self.dimension
        for i in range(self.dimension):
            self.distribution[i] = Distribution(self.dist_name[i])

        self.init_sts()
        if self.stype == 'Voronoi':
            self.run_sts()
        elif self.stype == 'Rectangular':
            self.samplesU01, self.samples = self.run_sts()

    def run_sts(self):

        if self.stype == 'Rectangular':
            samples = np.empty([self.strata.origins.shape[0], self.strata.origins.shape[1]], dtype=np.float32)
            samples_u_to_x = np.empty([self.strata.origins.shape[0], self.strata.origins.shape[1]], dtype=np.float32)
            for j in range(0, self.strata.origins.shape[1]):
                i_cdf = self.distribution[j].icdf
                if self.sts_criterion == "random":
                    for i in range(0, self.strata.origins.shape[0]):
                        samples[i, j] = np.random.uniform(self.strata.origins[i, j], self.strata.origins[i, j]
                                                          + self.strata.widths[i, j])
                elif self.sts_criterion == "centered":
                    for i in range(0, self.strata.origins.shape[0]):
                        samples[i, j] = self.strata.origins[i, j] + self.strata.widths[i, j] / 2.

                samples_u_to_x[:, j] = i_cdf(samples[:, j], self.dist_params[j])

            print('UQpy: Successful execution of STS design..')
            return samples, samples_u_to_x

        elif self.stype == 'Voronoi':
            from scipy.spatial import Voronoi
            from UQpy.Utilities import compute_Voronoi_centroid_volume, voronoi_unit_hypercube

            samples_init = np.random.rand(self.nsamples, self.dimension)

            for i in range(self.n_iters):
                x = self.in_hypercube(samples_init)

                self.strata = voronoi_unit_hypercube(samples_init)

                self.strata.centroids = []
                self.strata.weights = []
                for region in self.strata.bounded_regions:
                    vertices = self.strata.vertices[region + [region[0]], :]
                    centroid, volume = compute_Voronoi_centroid_volume(vertices)
                    self.strata.centroids.append(centroid[0, :])
                    self.strata.weights.append(volume)

                samples_init = np.asarray(self.strata.centroids)

            self.samplesU01 = self.strata.bounded_points

            self.samples = np.zeros(np.shape(self.samplesU01))
            for i in range(self.dimension):
                self.samples[:, i] = self.distribution[i].icdf(self.samplesU01[:, i], self.dist_params[i]).T

    def in_hypercube(self, samples):
        str_temp = 'np.logical_and('

        in_cube = True * self.nsamples
        for i in range(self.dimension):
            in_cube = np.logical_and(in_cube, np.logical_and(0 <= samples[:, i], samples[:, i] <= 1))

        return(in_cube)




    # def voronoi_centroid_volume(self, vertices):
    #
    #     from scipy.spatial import Delaunay, ConvexHull
    #
    #     T = Delaunay(vertices)
    #
    #     w = np.zeros((T.nsimplex, 1))
    #     cent = np.zeros((T.nsimplex, self.dimension))
    #     for i in range(T.nsimplex):
    #         ch = ConvexHull(T.points[T.simplices[i]])
    #         w[i] = ch.volume
    #         cent[i, :] = np.mean(T.points[T.simplices[i]], axis=0)
    #     V = np.sum(w)
    #     C = np.matmul(np.divide(w, V).T, cent)
    #
    #     return C, V

    def init_sts(self):

        # Check for dimensional consistency
        if self.dimension is None and self.sts_design is not None:
            self.dimension = len(self.sts_design)
        elif self.sts_design is not None:
            if self.dimension != len(self.sts_design):
                raise NotImplementedError("Exit code: Incompatible dimensions.")
        elif self.sts_design is None and self.dimension is None:
            raise NotImplementedError("Exit code: Dimension must be specified.")

        # Check dist_name
        if type(self.dist_name).__name__ != 'list':
            self.dist_name = [self.dist_name]
        if len(self.dist_name) == 1 and self.dimension != 1:
            self.dist_name = self.dist_name * self.dimension
        elif len(self.dist_name) != self.dimension:
            raise NotImplementedError("Length of i_cdf should be 1 or equal to dimension.")

        # Check dist_params
        if type(self.dist_params).__name__ != 'list':
            self.dist_params = [self.dist_params]
        if len(self.dist_params) == 1 and self.dimension != 1:
            self.dist_params = self.dist_params * self.dimension
        elif len(self.dist_params) != self.dimension:
            raise NotImplementedError("Length of dist_params list should be 1 or equal to dimension.")

        # Ensure that distribution parameters are assigned
        if self.dist_params is None:
            raise NotImplementedError("Exit code: Distribution parameters not defined.")

        if self.stype == 'Rectangular':
            if self.sts_design is None:
                if self.input_file is None:
                    raise NotImplementedError("Exit code: Stratum design is not defined.")
                else:
                    self.strata = Strata(input_file=self.input_file)
            else:
                if len(self.sts_design) != self.dimension:
                    raise NotImplementedError("Exit code: Incompatible dimensions in 'sts_design'.")
                else:
                    self.strata = Strata(n_strata=self.sts_design)

        # Check sampling criterion
        if self.sts_criterion not in ['random', 'centered']:
            raise NotImplementedError("Exit code: Supported sts criteria: 'random', 'centered'")

########################################################################################################################
########################################################################################################################
#                                         Class Strata
########################################################################################################################


class Strata:
    """
        Description:

            Define a rectilinear stratification of the n-dimensional unit hypercube [0, 1]^dimension with N strata.

        Input:
            :param n_strata: A list of dimension n defining the number of strata in each of the n dimensions
                            Creates an equal stratification with strata widths equal to 1/n_strata
                            The total number of strata, N, is the product of the terms of n_strata
                            Example -
                            n_strata = [2, 3, 2] creates a 3d stratification with:
                            2 strata in dimension 0 with stratum widths 1/2
                            3 strata in dimension 1 with stratum widths 1/3
                            2 strata in dimension 2 with stratum widths 1/2
            :type n_strata int list

            :param input_file: File path to input file specifying stratum origins and stratum widths.
                               Default: None
            :type input_file: string

        Output:
            :return origins: An array of dimension N x n specifying the origins of all strata
                            The origins of the strata are the coordinates of the stratum orthotope nearest the global
                            origin.
                            Example - A 2D stratification with 2 strata in each dimension
                            origins = [[0, 0]
                                      [0, 0.5]
                                      [0.5, 0]
                                      [0.5, 0.5]]
            :rtype origins: array

            :return widths: An array of dimension N x n specifying the widths of all strata in each dimension
                           Example - A 2D stratification with 2 strata in each dimension
                           widths = [[0.5, 0.5]
                                     [0.5, 0.5]
                                     [0.5, 0.5]
                                     [0.5, 0.5]]
            :rtype widths: ndarray

            :return weights: An array of dimension 1 x N containing sample weights.
                            Sample weights are equal to the product of the strata widths (i.e. they are equal to the
                            size of the strata in the [0, 1]^n space.
            :rtype weights: ndarray

    """

    def __init__(self, n_strata=None, input_file=None, origins=None, widths=None):

        self.input_file = input_file
        self.n_strata = n_strata
        self.origins = origins
        self.widths = widths

        # Read a stratified design from an input file.
        if self.n_strata is None:
            if self.input_file is None:
                if self.widths is None or self.origins is None:
                    sys.exit('Error: The strata are not fully defined. Must provide [n_strata], '
                             'input file, or [origins] and [widths].')

            else:
                # Read the strata from the specified input file
                # See documentation for input file formatting
                array_tmp = np.loadtxt(input_file)
                self.origins = array_tmp[:, 0:array_tmp.shape[1] // 2]
                self.widths = array_tmp[:, array_tmp.shape[1] // 2:]

                # Check to see that the strata are space-filling
                space_fill = np.sum(np.prod(self.widths, 1))
                if 1 - space_fill > 1e-5:
                    sys.exit('Error: The stratum design is not space-filling.')
                if 1 - space_fill < -1e-5:
                    sys.exit('Error: The stratum design is over-filling.')

        # Define a rectilinear stratification by specifying the number of strata in each dimension via nstrata
        else:
            self.origins = np.divide(self.fullfact(self.n_strata), self.n_strata)
            self.widths = np.divide(np.ones(self.origins.shape), self.n_strata)

        self.weights = np.prod(self.widths, axis=1)

    @staticmethod
    def fullfact(levels):

        """
            Description:

                Create a full-factorial design

                Note: This function has been modified from pyDOE, released under BSD License (3-Clause)
                Copyright (C) 2012 - 2013 - Michael Baudin
                Copyright (C) 2012 - Maria Christopoulou
                Copyright (C) 2010 - 2011 - INRIA - Michael Baudin
                Copyright (C) 2009 - Yann Collette
                Copyright (C) 2009 - CEA - Jean-Marc Martinez
                Original source code can be found at:
                https://pythonhosted.org/pyDOE/#
                or
                https://pypi.org/project/pyDOE/
                or
                https://github.com/tisimst/pyDOE/

            Input:
                :param levels: A list of integers that indicate the number of levels of each input design factor.
                :type levels: list

            Output:
                :return ff: Full-factorial design matrix.
                :rtype ff: ndarray

        """

        # Number of factors
        n_factors = len(levels)
        # Number of combinations
        n_comb = np.prod(levels)
        ff = np.zeros((n_comb, n_factors))

        level_repeat = 1
        range_repeat = np.prod(levels)
        for i in range(n_factors):
            range_repeat //= levels[i]
            lvl = []
            for j in range(levels[i]):
                lvl += [j] * level_repeat
            rng = lvl * range_repeat
            level_repeat *= levels[i]
            ff[:, i] = rng

        return ff


########################################################################################################################
########################################################################################################################
#                                         Refined Stratified Sampling (RSS)
########################################################################################################################


class RSS:
    """

        Description:

            Generate new samples using adaptive sampling methods, i.e. Refined Stratified Sampling and Gradient
            Enhanced Refined Stratified Sampling.

            References:
            Michael D. Shields, Kirubel Teferra, Adam Hapij and Raymond P. Daddazio, "Refined Stratified Sampling for
                efficient Monte Carlo based uncertainty quantification", Reliability Engineering & System Safety,
                ISSN: 0951-8320, Vol: 142, Page: 310-325, 2015.

            M. D. Shields, "Adaptive Monte Carlo analysis for strongly nonlinear stochastic systems",
                Reliability Engineering & System Safety, ISSN: 0951-8320, Vol: 175, Page: 207-224, 2018.
        Input:
            :param x: A class object, it should be generated using STS or RSS class.
            :type x: class

            :param model: Python model which is used to evaluate the function value
            :type model: str

            :param meta: A string specifying the method used to estimate the gradient.
                         Options: Delaunay, Kriging
            :type meta: str

            :param cell: A string specifying the stratification of sample domain.
                         Options: Rectangular and Voronoi
            :type cell: str

            :param nsamples: Final size of the samples.
            :type nsamples: int

            :param max_train_size: Minimum size of training data around new sample used to update surrogate.
                                   Default: nsamples
            :type max_train_size: int

            :param step_size: Step size to calculate the gradient using central difference. Only required if Delaunay is
                              used as surrogate approximation.
            :type step_size: float

            :param reg_model: Regression model used to estimate gradient by using kriging surrogate. Only required
                               if kriging is used as surrogate approximation.
            :type reg_model: str

            :param corr_model: Correlation model used to estimate gradient by using kriging surrogate. Only required
                               if kriging is used as surrogate approximation.
            :type corr_model: str

            :param corr_model_params: Correlation model parameters used to estimate hyperparamters for kriging
                                      surrogate.
            :type corr_model_params: ndarray

            :param n_opt: Number of times optimization problem is to be solved with different starting point.
                          Default: 1
            :type n_opt: int

        Output:
            :return: RSS.samples: Final/expanded samples.
            :rtype: RSS.samples: ndarray

            :return: RSS.values: Function value evaluated at the expanded samples.
            :rtype: RSS.values: ndarray

    """

    # Authors: Mohit S. Chauhan
    # Last modified: 12/03/2018 by Mohit S. Chauhan

    def __init__(self, sample_object=None, run_model_object=None, meta='Kriging', cell='Rectangular', nsamples=None,
                 max_train_size=None, step_size=0.005, corr_model='Gaussian', reg_model='Quadratic',
                 corr_model_params=None, n_opt=10, option=None, qoi_name=None, verbose=False, local=False,
                 visualize=False, **kwargs):

        from UQpy.RunModel import RunModel

        # Initialize attributes that are common to all approaches
        self.sample_object = sample_object
        self.run_model_object = run_model_object
        self.verbose = verbose
        self.option = option
        self.dimension = np.shape(self.sample_object.samples)[1]
        self.cell = cell
        self.nsamples = nsamples
        self.visualize = visualize

        # Run Initial Error Checks
        self.init_rss()

        if self.option == 'Gradient':
            self.local = local
            self.max_train_size = max_train_size
            self.meta = meta
            self.corr_model = corr_model
            self.corr_model_params = corr_model_params
            self.reg_model = reg_model
            self.n_opt = n_opt
            self.qoi_name = qoi_name
            self.step_size = step_size
            if not kwargs:
                self.run_gerss()
            else:
                self.run_gerss(kwargs)
        else:
            self.run_rss()


    ###################################################
    # Run Gradient-Enhanced Refined Stratified Sampling
    ###################################################
    def run_gerss(self, *args):

        # Check if the initial sample design already has model evalutions with it.
        # If it does not, run the initial calculations.
        if self.run_model_object is None:
            raise NotImplementedError('UQpy Error: Gradient Enhanced RSS requires a predefined RunModel object.')
        elif not self.run_model_object.samples:
            if self.verbose:
                print('UQpy: GE-RSS - Running the initial sample set.')
            if not args:
                self.run_model_object.run(samples=self.sample_object.samples)
            else:
                self.run_model_object.run(args[0], samples=self.sample_object.samples)

        self.nexist = len(self.run_model_object.samples)

        if self.nsamples <= self.nexist:
            raise NotImplementedError('UQpy Error: The number of requested samples must be larger than the existing '
                                      'sample set.')

        # --------------------------
        # RECTANGULAR STRATIFICATION
        # --------------------------

        if self.cell == 'Rectangular':

            if self.verbose:
                print('UQpy: Performing GE-RSS with rectangular stratification...')

            # Initialize the training points for the surrogate model
            self.training_points = self.sample_object.samplesU01

            # Initialize the vector of gradients at each training point
            dydx = np.zeros((self.nsamples, np.size(self.training_points[1])))

            # Primary loop for adding samples and performing refinement.
            for i in range(self.nexist, self.nsamples):

                # TODO: Add visualize option to plot the points and their strata in 2D.

                # If the quantity of interest is a dictionary, convert it to a list
                qoi = [None] * len(self.run_model_object.qoi_list)
                if type(self.run_model_object.qoi_list[0]) is dict:
                    for j in range(len(self.run_model_object.qoi_list)):
                        qoi[j] = self.run_model_object.qoi_list[j][self.qoi_name]
                else:
                    qoi = self.run_model_object.qoi_list

                # ---------------------------------------------------
                # Compute the gradients at the existing sample points
                # ---------------------------------------------------

                # Use the entire sample set to train the surrogate model (more expensive option)
                if self.max_train_size is None or len(self.training_points) <= self.max_train_size or i == self.nexist:
                    dydx[:i], self.corr_model_params = self.estimate_gradient(np.atleast_2d(self.training_points),
                                                                              np.atleast_2d(np.array(qoi)),
                                                                              self.corr_model_params, self.reg_model,
                                                                              self.corr_model,
                                                                              self.sample_object.strata.origins + \
                                                                              0.5 * self.sample_object.strata.widths,
                                                                              self.n_opt)

                # Use only max_train_size points to train the surrogate model (more economical option)
                else:
                    # Find the nearest neighbors to the most recently added point
                    from sklearn.neighbors import NearestNeighbors
                    knn = NearestNeighbors(n_neighbors=self.max_train_size)
                    knn.fit(np.atleast_2d(self.training_points))
                    neighbors = knn.kneighbors(np.atleast_2d(self.training_points[-1]), return_distance=False)

                    # Recompute the gradient only at the nearest neighbor points.
                    dydx[neighbors], self.corr_model_params = \
                        self.estimate_gradient(np.squeeze(self.training_points[neighbors]),
                                               np.atleast_2d(np.array(qoi)[neighbors]),
                                               self.corr_model_params, self.reg_model,
                                               self.corr_model,
                                               np.squeeze(self.sample_object.strata.origins[neighbors] + \
                                                          0.5 * self.sample_object.strata.widths[neighbors]),
                                               self.n_opt)

                # Define the gradient vector for application of the Delta Method
                dydx1 = dydx[:i]

                # ------------------------------
                # Determine the stratum to break
                # ------------------------------

                # Estimate the variance within each stratum by assuming a uniform distribution over the stratum.
                # All input variables are independent
                var = (1 / 12) * self.sample_object.strata.widths ** 2

                # Estimate the variance over the stratum by Delta Method
                s = np.zeros([i, 1])
                for j in range(i):
                    s[j, 0] = np.sum(dydx1[j, :] * var[j, :] * dydx1[j, :] * \
                                     (self.sample_object.strata.weights[j] ** 2))

                # TODO: Break n strata with the highest gradients and sample in each. Allow this to use multiple cores.

                # Break the stratum with the maximum variance
                bin2break = np.argmax(s)

                # Cut the stratum in the direction of maximum gradient
                cut_dir_temp = self.sample_object.strata.widths[bin2break, :]
                t = np.argwhere(cut_dir_temp == np.amax(cut_dir_temp))
                dir2break = t[np.argmax(abs(dydx1[bin2break, t]))]

                # Divide the stratum bin2break in the direction dir2break
                self.sample_object.strata.widths[bin2break, dir2break] = \
                    self.sample_object.strata.widths[bin2break, dir2break] / 2
                self.sample_object.strata.widths = np.vstack([self.sample_object.strata.widths,
                                                              self.sample_object.strata.widths[bin2break, :]])
                self.sample_object.strata.origins = np.vstack([self.sample_object.strata.origins,
                                                               self.sample_object.strata.origins[bin2break, :]])
                if self.sample_object.samplesU01[bin2break, dir2break] < \
                        self.sample_object.strata.origins[-1, dir2break] + \
                        self.sample_object.strata.widths[bin2break, dir2break]:
                    self.sample_object.strata.origins[-1, dir2break] = \
                        self.sample_object.strata.origins[-1, dir2break] + \
                        self.sample_object.strata.widths[bin2break, dir2break]
                else:
                    self.sample_object.strata.origins[bin2break, dir2break] = \
                        self.sample_object.strata.origins[bin2break, dir2break] + \
                        self.sample_object.strata.widths[bin2break, dir2break]

                self.sample_object.strata.weights[bin2break] = self.sample_object.strata.weights[bin2break] / 2
                self.sample_object.strata.weights = np.append(self.sample_object.strata.weights,
                                                              self.sample_object.strata.weights[bin2break])

                # Add a uniform random sample inside the new stratum
                new_point = np.random.uniform(self.sample_object.strata.origins[i, :],
                                        self.sample_object.strata.origins[i, :] + \
                                        self.sample_object.strata.widths[i, :])

                # Adding new sample to training points, samplesU01 and samples attributes
                self.training_points = np.vstack([self.training_points, new_point])
                self.sample_object.samplesU01 = np.vstack([self.sample_object.samplesU01, new_point])
                for j in range(0, self.dimension):
                    icdf = self.sample_object.distribution[j].icdf
                    new_point[j] = icdf(new_point[j], self.sample_object.dist_params[j])
                self.sample_object.samples = np.vstack([self.sample_object.samples, new_point])

                # Run the model at the new sample point
                self.run_model_object.ntasks = 1
                if not args:
                    self.run_model_object.run(samples=np.atleast_2d(new_point))
                else:
                    self.run_model_object.run(args[0], samples=np.atleast_2d(new_point))

                if self.verbose:
                    print("Iteration:", i)


        # ----------------------
        # VORONOI STRATIFICATION
        # ----------------------

        elif self.cell == 'Voronoi':

            from UQpy.Utilities import compute_Delaunay_centroid_volume, voronoi_unit_hypercube
            from scipy.spatial import Delaunay
            import math
            import itertools

            self.training_points = self.sample_object.samplesU01

            # Extract the boundary vertices and use them in the Delaunay triangulation / mesh generation
            self.mesh_vertices = self.training_points
            self.points_to_samplesU01 = np.arange(0,self.training_points.shape[0])
            for i in range(np.shape(self.sample_object.strata.vertices)[0]):
                if any(np.logical_and(self.sample_object.strata.vertices[i, :] >= -1e-10,
                                       self.sample_object.strata.vertices[i, :] <= 1e-10)) or \
                    any(np.logical_and(self.sample_object.strata.vertices[i, :] >= 1-1e-10,
                                       self.sample_object.strata.vertices[i, :] <= 1+1e-10)):
                    self.mesh_vertices = np.vstack([self.mesh_vertices, self.sample_object.strata.vertices[i,:]])
                    self.points_to_samplesU01 = np.hstack([self.points_to_samplesU01, np.array([np.nan])])

            # Define the simplex mesh to be used for gradient estimation and sampling
            self.mesh = Delaunay(self.mesh_vertices, incremental=True)

            # Primary loop for adding samples and performing refinement.
            for i in range(self.nexist, self.nsamples):

                # Compute the centroids and the volumes of each simplex cell in the mesh
                self.mesh.centroids = np.zeros([self.mesh.nsimplex, self.dimension])
                self.mesh.volumes = np.zeros([self.mesh.nsimplex, 1])
                for j in range(self.mesh.nsimplex):
                    self.mesh.centroids[j, :], self.mesh.volumes[j] = \
                        compute_Delaunay_centroid_volume(self.mesh.points[self.mesh.vertices[j]])

                # If the quantity of interest is a dictionary, convert it to a list
                qoi = [None] * len(self.run_model_object.qoi_list)
                if type(self.run_model_object.qoi_list[0]) is dict:
                    for j in range(len(self.run_model_object.qoi_list)):
                        qoi[j] = self.run_model_object.qoi_list[j][self.qoi_name]
                else:
                    qoi = self.run_model_object.qoi_list

                # ---------------------------------------------------
                # Compute the gradients at the existing sample points
                # ---------------------------------------------------

                # Use the entire sample set to train the surrogate model (more expensive option)
                if self.max_train_size is None or len(self.training_points) <= self.max_train_size or \
                        i == self.nexist:
                    dydx, self.corr_model_params = self.estimate_gradient(np.atleast_2d(self.training_points),
                                                                              np.atleast_2d(np.array(qoi)),
                                                                              self.corr_model_params,
                                                                              self.reg_model,
                                                                              self.corr_model,
                                                                              self.mesh.centroids,
                                                                              self.n_opt)

                # Use only max_train_size points to train the surrogate model (more economical option)
                else:

                    # Build a mapping from the new vertex indices to the old vertex indices.
                    self.mesh.new_vertices = []
                    self.mesh.new_indices = []
                    self.mesh.new_to_old = np.zeros([self.mesh.vertices.shape[0], ]) * np.nan
                    j = 0
                    k = 0
                    while j < self.mesh.vertices.shape[0] and k < self.mesh.old_vertices.shape[0]:

                        if np.all(self.mesh.vertices[j, :] == self.mesh.old_vertices[k, :]):
                            self.mesh.new_to_old[j] = int(k)
                            j += 1
                            k = 0
                        else:
                            k += 1
                            if k == self.mesh.old_vertices.shape[0]:
                                self.mesh.new_vertices.append(self.mesh.vertices[j])
                                self.mesh.new_indices.append(j)
                                j += 1
                                k = 0

                    # Find the nearest neighbors to the most recently added point
                    from sklearn.neighbors import NearestNeighbors
                    knn = NearestNeighbors(n_neighbors=self.max_train_size)
                    knn.fit(np.atleast_2d(self.sample_object.samplesU01))
                    neighbors = knn.kneighbors(np.atleast_2d(self.sample_object.samplesU01[-1]),
                                               return_distance=False)

                    # For every simplex, check if at least dimension-1 vertices are in the neighbor set.
                    # Only update the gradient in simplices that meet this criterion.
                    update_list = []
                    for j in range(self.mesh.vertices.shape[0]):
                        self.vertices_in_U01 = self.points_to_samplesU01[self.mesh.vertices[j]]
                        self.vertices_in_U01[np.isnan(self.vertices_in_U01)] = 10 ** 18
                        v_set = set(self.vertices_in_U01)
                        v_list = list(self.vertices_in_U01)
                        if len(v_set) != len(v_list):
                            continue
                        else:
                            if all(np.isin(self.vertices_in_U01, np.hstack([neighbors, np.atleast_2d(10**18)]))):
                                update_list.append(j)

                    update_array = np.asarray(update_list)

                    # Initialize the gradient vector
                    dydx = np.zeros((self.mesh.new_to_old.shape[0], self.dimension))

                    # For those simplices that will not be updated, use the previous gradient
                    for j in range(dydx.shape[0]):
                        if np.isnan(self.mesh.new_to_old[j]):
                            continue
                        else:
                            dydx[j, :] = dydx_old[int(self.mesh.new_to_old[j]), :]

                    # For those simplices that will be updated, compute the new gradient
                    dydx[update_array, :], self.corr_model_params = \
                        self.estimate_gradient(np.squeeze(self.sample_object.samplesU01[neighbors]),
                                               np.atleast_2d(np.array(qoi)[neighbors]),
                                               self.corr_model_params, self.reg_model,
                                               self.corr_model,
                                               self.mesh.centroids[update_array],
                                               self.n_opt)

                # ----------------------------------------------------
                # Determine the simplex to break and draw a new sample
                # ----------------------------------------------------

                # Estimate the variance over each simplex by Delta Method. Moments of the simplices are computed using
                # Eq. (19) from the following reference:
                # Good, I.J. and Gaskins, R.A. (1971). The Centroid Method of Numerical Integration. Numerische
                #       Mathematik. 16: 343--359.
                var = np.zeros((self.mesh.nsimplex, self.dimension))
                s = np.zeros((self.mesh.nsimplex))
                for j in range(self.mesh.nsimplex):
                    for k in range(self.dimension):
                        std = np.std(self.mesh.points[self.mesh.vertices[j]][:, k])
                        var[j, k] = (self.mesh.volumes[j] * math.factorial(self.dimension) /
                                     math.factorial(self.dimension + 2)) * (self.dimension * std ** 2)
                    s[j] = np.sum(dydx[j, :] * var[j, :] * dydx[j, :] * (self.mesh.volumes ** 2))
                dydx_old = dydx

                # TODO: Update this to sample n points with the highest variance. Allow it to run on multiple cores.

                # Identify the stratum with the maximum variance
                bin2add = np.argmax(s)

                # Create a sub-simplex within the simplex with maximum variance.
                tmp_verts = self.mesh.points[self.mesh.simplices[bin2add, :]]
                col_one = np.array(list(itertools.combinations(np.arange(self.dimension + 1), self.dimension)))
                self.mesh.subsimplex = np.zeros_like(tmp_verts)  # node: an array containing mid-point of edges
                for m in range(self.dimension + 1):
                    self.mesh.subsimplex[m, :] = np.sum(tmp_verts[col_one[m] - 1, :], 0) / self.dimension

                # Using the Simplex class to generate a new sample in the subsimplex
                new_point = Simplex(nodes=self.mesh.subsimplex, nsamples=1).samples

                # If the visualize option is selected, plot the simplex evolution
                if self.visualize:
                    if self.dimension != 2:
                        raise NotImplementedError(
                            'UQpy Error: Voronoi visualization is only permitted for 2D problems.')

                    # Check if the directory exists for placing the generated images.
                    if not os.path.exists('./figures_voronoi'):
                        os.makedirs('./figures_voronoi')

                    # Plot the points, simplices, and their centroids with the centroids colored by variance
                    if i == self.nexist:
                        fig = plt.figure(1)
                        ax = plt.axes()
                        ax.plot(self.training_points[:, 0], self.training_points[:, 1], 'xk')
                        ax.scatter(self.mesh.centroids[:, 0], self.mesh.centroids[:, 1], c=s, vmin=0, vmax=0.00001)
                        ax.triplot(self.mesh.points[:, 0], self.mesh.points[:, 1], self.mesh.simplices, color='blue')
                        plt.savefig('./figures_voronoi/p1_' + str(i) + '.png')
                        plt.close(fig)

                    else:
                        fig = plt.figure(1)
                        ax = plt.axes()
                        ax.plot(self.training_points[:, 0], self.training_points[:, 1], 'xk')
                        ax.scatter(self.mesh.centroids[:, 0], self.mesh.centroids[:, 1], c=s, vmin=0, vmax=0.00001)
                        ax.triplot(self.mesh.points[:, 0], self.mesh.points[:, 1], self.mesh.simplices, color='blue')
                        if i > self.max_train_size:
                            ax.triplot(self.mesh.points[:, 0], self.mesh.points[:, 1],
                                       self.mesh.simplices[update_array], color='purple')
                        plt.savefig('./figures_voronoi/p1_' + str(i) + '.png')
                        plt.close(fig)

                        # Plot the points, simplices, and their centroids along with the subsimplex to be sampled.
                        fig = plt.figure(1)
                        ax = plt.axes()
                        ax.plot(self.training_points[:, 0], self.training_points[:, 1], 'xk')
                        ax.scatter(self.mesh.centroids[:, 0], self.mesh.centroids[:, 1], c=s, vmin=0, vmax=0.00001)
                        ax.triplot(self.mesh.points[:, 0], self.mesh.points[:, 1], self.mesh.simplices, color='blue')
                        if i > self.max_train_size:
                            ax.triplot(self.mesh.points[:, 0], self.mesh.points[:, 1],
                                       self.mesh.simplices[update_array], color='purple')
                        t1 = plt.Polygon(np.vstack([self.mesh.subsimplex[:, 0], self.mesh.subsimplex[:, 1]]).T,
                                         color='red')
                        plt.gca().add_patch(t1)
                        plt.savefig('./figures_voronoi/p2_' + str(i) + '.png')
                        plt.close(fig)

                        # Plot the points, simplices, their centroids, the subsimplex to be sampled, and the new sample
                        # point.
                        fig = plt.figure(1)
                        ax = plt.axes()
                        ax.plot(self.training_points[:, 0], self.training_points[:, 1], 'x')
                        ax.scatter(self.mesh.centroids[:, 0], self.mesh.centroids[:, 1], c=s, vmin=0, vmax=0.00001)
                        ax.triplot(self.mesh.points[:, 0], self.mesh.points[:, 1], self.mesh.simplices, color='blue')
                        if i > self.max_train_size:
                            ax.triplot(self.mesh.points[:, 0], self.mesh.points[:, 1],
                                       self.mesh.simplices[update_array], color='purple')
                        t1 = plt.Polygon(np.vstack([self.mesh.subsimplex[:, 0], self.mesh.subsimplex[:, 1]]).T,
                                         color='red')
                        plt.gca().add_patch(t1)
                        ax.plot(new_point[:, 0], new_point[:,1], 'xk')
                        plt.savefig('./figures_voronoi/p3_' + str(i) + '.png')
                        plt.close(fig)


                # Update the matrices to have recognize the new point
                self.points_to_samplesU01 = np.hstack([self.points_to_samplesU01, np.array([i])])
                self.mesh.old_vertices = self.mesh.vertices

                # Update the Delaunay triangulation mesh to include the new point.
                self.mesh.add_points(new_point)

                # Update the sample arrays to include the new point
                self.sample_object.samplesU01 = np.vstack([self.sample_object.samplesU01, new_point])
                self.training_points = np.vstack([self.training_points, new_point])

                # Identify the new point in the parameter space and update the sample array to include the new point.
                for j in range(self.dimension):
                    new_point[0, j] = self.sample_object.distribution[j].icdf(new_point[0, j],
                                                                        self.sample_object.dist_params[j])
                self.sample_object.samples = np.vstack([self.sample_object.samples, new_point])

                # Run the mode at the new point.
                if not args:
                    self.run_model_object.run(samples=np.atleast_2d(new_point))
                else:
                    self.run_model_object.run(args[0], samples=np.atleast_2d(new_point))

                # Compute the strata weights.
                self.sample_object.strata = voronoi_unit_hypercube(self.sample_object.samplesU01)

                self.sample_object.strata.centroids = []
                self.sample_object.strata.weights = []
                for region in self.sample_object.strata.bounded_regions:
                    vertices = self.sample_object.strata.vertices[region + [region[0]], :]
                    centroid, volume = compute_Voronoi_centroid_volume(vertices)
                    self.sample_object.strata.centroids.append(centroid[0, :])
                    self.sample_object.strata.weights.append(volume)

                if self.visualize:
                    if self.dimension != 2:
                        raise NotImplementedError(
                            'UQpy Error: Voronoi visualization is only permitted for 2D problems.')
                    from scipy.spatial import voronoi_plot_2d
                    fig = voronoi_plot_2d(self.sample_object.strata)
                    axes = plt.gca()
                    axes.set_xlim([0, 1])
                    axes.set_ylim([0, 1])
                    plt.savefig('./figures_voronoi/p4_' + str(i) + '.png')
                    plt.close(fig)

                if self.verbose:
                    print("Iteration:", i)

        #################################
        # Run Refined Stratified Sampling
        #################################

        # TODO: Rebuild run_rss(). Some code bits to start are below.

        # for i in range(self.nexist, self.nsamples):
        #     if self.cell == 'Rectangular':
        #         # Determine the stratum to break
        #         if self.option == 'Gradient':
        #             # Estimate the variance within each stratum by assuming a uniform distribution over the stratum.
        #             # All input variables are independent
        #             var = (1 / 12) * self.strata.widths ** 2
        #             # Estimate the variance over the stratum by Delta Method
        #             s = np.zeros([i, 1])
        #             for j in range(i):
        #                 s[j, 0] = np.sum(dydx1[j, :] * var[j, :] * dydx1[j, :] * (self.strata.weights[j] ** 2))
        #             bin2break = np.argmax(s)
        #         else:
        #             w = np.argwhere(self.strata.weights == np.amax(self.strata.weights))
        #             bin2break = w[np.random.randint(len(w))]
        #
        #         # Determine the largest dimension of the stratum and define this as the cut direction
        #         if self.option == 'Refined':
        #             # Cut the stratum in a random direction
        #             cut_dir_temp = self.strata.widths[bin2break, :]
        #             t = np.argwhere(cut_dir_temp[0] == np.amax(cut_dir_temp[0]))
        #             dir2break = t[np.random.randint(len(t))]
        #         else:
        #             # Cut the stratum in the direction of maximum gradient
        #             cut_dir_temp = self.strata.widths[bin2break, :]
        #             t = np.argwhere(cut_dir_temp == np.amax(cut_dir_temp))
        #             dir2break = t[np.argmax(abs(dydx1[bin2break, t]))]
        #
        #         # Divide the stratum bin2break in the direction dir2break
        #         self.strata.widths[bin2break, dir2break] = self.strata.widths[bin2break, dir2break] / 2
        #         self.strata.widths = np.vstack([self.strata.widths, self.strata.widths[bin2break, :]])
        #
        #         self.strata.origins = np.vstack([self.strata.origins, self.strata.origins[bin2break, :]])
        #         if self.samplesU01[bin2break, dir2break] < self.strata.origins[-1, dir2break] + \
        #                 self.strata.widths[bin2break, dir2break]:
        #             self.strata.origins[-1, dir2break] = self.strata.origins[-1, dir2break] + self.strata.widths[
        #                 bin2break, dir2break]
        #         else:
        #             self.strata.origins[bin2break, dir2break] = self.strata.origins[bin2break, dir2break] + \
        #                                                         self.strata.widths[bin2break, dir2break]
        #
        #         self.strata.weights[bin2break] = self.strata.weights[bin2break] / 2
        #         self.strata.weights = np.append(self.strata.weights, self.strata.weights[bin2break])
        #
        #         # Add an uniform random sample inside new stratum
        #         new = np.random.uniform(self.strata.origins[i, :], self.strata.origins[i, :] + self.strata.widths[i, :])
        #         # Adding new sample to points, samplesU01 and samples attributes
        #         self.points = np.vstack([self.points, new])
        #         self.samplesU01 = np.vstack([self.samplesU01, new])
        #         for j in range(0, dimension):
        #             icdf = self.distribution[j].icdf
        #             new[j] = icdf(new[j], self.dist_params[j])
        #         self.samples = np.vstack([self.samples, new])
        #
        #     elif self.cell == 'Voronoi':
        #         simplex = getattr(tri, 'simplices')
        #         # Estimate the variance over the stratum by Delta Method
        #         weights = np.zeros(((np.size(simplex, 0)), 1))
        #         var = np.zeros((np.size(simplex, 0), dimension))
        #         s = np.zeros(((np.size(simplex, 0)), 1))
        #         for j in range((np.size(simplex, 0))):
        #             # Define Simplex
        #             sim = self.points[simplex[j, :]]
        #             # Estimate the volume of simplex
        #             v1 = np.concatenate((np.ones([np.size(sim, 0), 1]), sim), 1)
        #             weights[j] = (1 / math.factorial(np.size(simplex[j, :]) - 1)) * np.linalg.det(v1)
        #             if self.option == 'Gradient':
        #                 for k in range(dimension):
        #                     # Estimate standard deviation of points
        #                     from statistics import stdev
        #                     std = stdev(sim[:, k].tolist())
        #                     var[j, k] = (weights[j] * math.factorial(dimension) / math.factorial(dimension + 2)) * (
        #                             dimension * std ** 2)
        #                 s[j, 0] = np.sum(dydx1[j, :] * var[j, :] * dydx1[j, :] * (weights[j] ** 2))
        #
        #         if self.option == 'Refined':
        #             w = np.argwhere(weights[:, 0] == np.amax(weights[:, 0]))
        #             bin2add = w[0, np.random.randint(len(w))]
        #         else:
        #             bin2add = np.argmax(s)
        #
        #         # Creating sub-simplex
        #         tmp = self.points[simplex[bin2add, :]]
        #         col_one = np.array(list(itertools.combinations(np.arange(dimension + 1), dimension)))
        #         node = np.zeros_like(tmp)  # node: an array containing mid-point of edges
        #         for m in range(dimension + 1):
        #             node[m, :] = np.sum(tmp[col_one[m] - 1, :], 0) / dimension
        #
        #         # Using Simplex class to generate new sample
        #         new = Simplex(nodes=node, nsamples=1).samples
        #         # Adding new sample to points, samplesU01 and samples attributes
        #         self.points = np.vstack([self.points, new])
        #         self.samplesU01 = np.vstack([self.samplesU01, new])
        #         for j in range(0, dimension):
        #             icdf = self.distribution[j].icdf
        #             new[0, j] = icdf(new[0, j], self.dist_params[j])
        #         self.samples = np.vstack([self.samples, new])
        #         # Creating Delaunay triangulation from the new points
        #         tri = Delaunay(self.points)
        #     else:
        #         raise NotImplementedError("Exit code: Does not identify 'cell'.")
        #
        #     if self.option == 'Gradient':
        #
        #         with suppress_stdout():  # disable printing output comments
        #             y_new = RunModel(np.atleast_2d(self.samples[i, :]), model_script=self.model_script,
        #                              model_object_name=self.model_object_name).qoi_list
        #         values = np.vstack([values, y_new])
        #
        #         if np.size(self.samples, 0) < self.max_train_size:
        #             # Global surrogate updating: Update the surrogate model using all the points
        #             if self.cell == 'Rectangular':
        #                 in_train = np.arange(self.points.shape[0])
        #                 in_update = np.arange(i)
        #             else:
        #                 simplex = getattr(tri, 'simplices')
        #                 in_train = np.arange(self.points.shape[0])
        #                 in_update = np.arange(simplex.shape[0])
        #         else:
        #             # Local surrogate updating: Update the surrogate model using max_train_size
        #             if self.cell == 'Rectangular':
        #                 if self.meta == 'Delaunay':
        #                     in_train = local(self.samplesU01[i, :], self.points, self.max_train_size,
        #                                      np.amax(self.strata.widths))
        #                 else:
        #                     in_train = local(self.samplesU01[i, :], self.samplesU01, self.max_train_size,
        #                                      np.amax(self.strata.widths))
        #                 in_update = local(self.samplesU01[i, :], self.strata.origins + .5 * self.strata.widths,
        #                                   self.max_train_size / 2, np.amax(self.strata.widths))
        #             else:
        #                 simplex = getattr(tri, 'simplices')
        #                 # in_train: Indices of samples used to update surrogate approximation
        #                 in_train = local(self.samplesU01[i, :], self.samplesU01, self.max_train_size,
        #                                  np.amax(np.sqrt(self.strata.weights)))
        #                 # in_update: Indices of centroid of simplex, where gradient is updated
        #                 in_update = local(self.samplesU01[i, :], np.mean(tri.points[simplex], 1),
        #                                   self.max_train_size / 2, np.amax(np.sqrt(self.strata.weights)))
        #
        #         # Update the surrogate model & the store the updated gradients
        #         if self.cell == 'Rectangular':
        #             dydx1 = np.vstack([dydx1, np.zeros(dimension)])
        #             dydx1[in_update, :], self.corr_model_params = surrogate(self.points[in_train, :],
        #                                                                     values[in_train, :],
        #                                                                     self.corr_model_params, self.reg_model,
        #                                                                     self.corr_model,
        #                                                                     self.strata.origins[in_update, :] +
        #                                                                     .5 * self.strata.widths[in_update, :], 1)
        #         else:
        #             simplex = getattr(tri, 'simplices')
        #             dydx1 = np.vstack([dydx1, np.zeros([simplex.shape[0] - dydx1.shape[0], dimension])])
        #             dydx1[in_update, :], self.corr_model_params = surrogate(self.points[in_train, :],
        #                                                                     values[in_train, :],
        #                                                                     self.corr_model_params,
        #                                                                     self.reg_model, self.corr_model,
        #                                                                     np.mean(tri.points[
        #                                                                                 simplex[in_update, :]],
        #                                                                             1), 1)
        # print('Done!')
        # if self.option == 'Gradient':
        #     if self.cell == 'Rectangular':
        #         if self.meta != 'Delaunay':
        #             return values
        #     else:
        #         return values[2 ** dimension:, :]


    # Support functions for RSS and GE-RSS

    # Code for estimating gradients with a metamodel (surrogate)
    # TODO: We may want to consider moving this to Utilities.
    def estimate_gradient(self, x, y, corr_m_p, reg_m, corr_m, xt, n):
        # meta = 'Delaunay' is not currently functional.
        if self.meta == 'Delaunay':

            # TODO: Here we need to add a reflection of the sample points over each face of the hypercube and build the
            # linear interpolator from the reflected points.

            from scipy.interpolate import LinearNDInterpolator

            tck = LinearNDInterpolator(x, y.T, fill_value=0)
            gr = self.cent_diff(tck, xt, self.step_size)
        elif self.meta == 'Kriging':
            from UQpy.Surrogates import Krig
            with suppress_stdout():  # disable printing output comments
                tck = Krig(samples=x, values=y.T, reg_model=reg_m, corr_model=corr_m, corr_model_params=corr_m_p,
                           n_opt=n)
            corr_m_p = tck.corr_model_params
            gr = self.cent_diff(tck.interpolate, xt, self.step_size)

        elif self.meta == 'Kriging_Sklearn':
            gp = GaussianProcessRegressor(kernel=corr_m, n_restarts_optimizer=0)
            gp.fit(x, y)
            gr = self.cent_diff(gp.predict, xt, self.step_size)
        else:
            raise NotImplementedError("UQpy Error: 'meta' must be specified in order to calculate gradients.")
        return gr, corr_m_p

    # Implementation of the central difference method for calculating gradients.
    # TODO: This should probably be moved to Utilities.
    def cent_diff(self, f, x, h):
        dydx = np.zeros((np.size(x, 0), np.size(x, 1)))
        for dirr in range(np.size(x, 1)):
            temp = np.zeros((np.size(x, 0), np.size(x, 1)))
            temp[:, dirr] = np.ones(np.size(x, 0))
            low = x - h / 2 * temp
            hi = x + h / 2 * temp
            dydx[:, dirr] = ((f.__call__(hi) - f.__call__(low)) / h)[:].reshape((len(hi),))
        return dydx

    # This code is not currently used.
    # def local(pt, x, mts, max_dim):
    #     # Identify the indices of 'mts' number of points in array 'x', which are closest to point 'pt'.
    #     ff = 0.2
    #     train = []
    #     while len(train) < mts:
    #         a = x > matlib.repmat(pt - ff * max_dim, x.shape[0], 1)
    #         b = x < matlib.repmat(pt + ff * max_dim, x.shape[0], 1)
    #         x_ind = a & b
    #         train = []
    #         for k_ in range(x.shape[0]):
    #             if np.array_equal(x_ind[k_, :], np.ones(x.shape[1])):
    #                 train.append(k_)
    #         ff = ff + 0.1
    #     return train

    # Initialization and preliminary error checks.
    def init_rss(self):
        if type(self.sample_object).__name__ not in ['STS', 'RSS']:
            raise NotImplementedError("UQpy Error: sample_object must be an object of the STS or RSS class.")
        if self.run_model_object is not None:
            self.option = 'Gradient'
            if type(self.run_model_object).__name__ not in ['RunModel']:
                raise NotImplementedError("UQpy Error: run_model_object must be an object of the RunModel class.")



########################################################################################################################
########################################################################################################################
#                                        Generating random samples inside a Simplex
########################################################################################################################
class Simplex:
    """
        Description:

            Generate random samples inside a simplex using uniform probability distribution.

            References:
            W. N. Edelinga, R. P. Dwightb, P. Cinnellaa, "Simplex-stochastic collocation method with improved
                calability",Journal of Computational Physics, 310:301–328 2016.
        Input:
            :param nodes: The vertices of the simplex
            :type nodes: ndarray

            :param nsamples: The number of samples to be generated inside the simplex
            :type nsamples: int
        Output:
            :return samples: New generated samples
            :rtype samples: ndarray
    """

    # Authors: Dimitris G.Giovanis
    # Last modified: 11/28/2018 by Mohit S. Chauhan

    def __init__(self, nodes=None, nsamples=1):
        self.nodes = np.atleast_2d(nodes)
        self.nsamples = nsamples
        self.init_sis()
        self.samples = self.run_sis()

    def run_sis(self):
        dimension = self.nodes.shape[1]
        if dimension > 1:
            sample = np.zeros([self.nsamples, dimension])
            for i in range(self.nsamples):
                r = np.zeros([dimension])
                ad = np.zeros(shape=(dimension, len(self.nodes)))
                for j in range(dimension):
                    b_ = list()
                    for k in range(1, len(self.nodes)):
                        ai = self.nodes[k, j] - self.nodes[k - 1, j]
                        b_.append(ai)
                    ad[j] = np.hstack((self.nodes[0, j], b_))
                    r[j] = np.random.uniform(0.0, 1.0, 1) ** (1 / (dimension - j))
                d = np.cumprod(r)
                r_ = np.hstack((1, d))
                sample[i, :] = np.dot(ad, r_)
        else:
            a = min(self.nodes)
            b = max(self.nodes)
            sample = a + (b - a) * np.random.rand(dimension, self.nsamples).reshape(self.nsamples, dimension)
        return sample

    def init_sis(self):
        if self.nsamples <= 0 or type(self.nsamples).__name__ != 'int':
            raise NotImplementedError("Exit code: Number of samples to be generated 'nsamples' should be a positive "
                                      "integer.")

        if self.nodes.shape[0] != self.nodes.shape[1] + 1:
            raise NotImplementedError("Size of simplex (nodes) is not consistent.")


########################################################################################################################
########################################################################################################################
#                                  Adaptive Kriging-Monte Carlo Simulation (AK-MCS)
########################################################################################################################


class AKMCS:
    """

        Description:

            Generate new samples using different active learning method and properties of kriging surrogate along with
            MCS.

            References:
        Input:
            :param model: Python model which is used to evaluate the function value
            :type model: str

            :param dist_name: A list containing the names of the distributions of the random variables.
                              Distribution names must match those in the Distributions module.
                              If the distribution does not match one from the Distributions module, the user must
                              provide custom_dist.py.
                              The length of the string must be 1 (if all distributions are the same) or equal to
                              dimension.
            :type dist_name: string list

            :param dist_params: Parameters of the distribution
                                Parameters for each random variable are defined as ndarrays.
                                Each item in the list, dist_params[i], specifies the parameters for the corresponding
                                distribution, dist[i].
            :type dist_params: list

            :param nsamples: Number of samples to generate.
            No Default Value: nsamples must be prescribed.
            :type nsamples: int

            :param doe: Design of Experiment.
            :type doe: ndarray

            :param population: Monte Carlo Population, new samples are selected from this set of points.
            :type doe: ndarray

            :param n_doe: Number of samples to be selected as design point from Population. It is only required if
                          design points are not define.
            :type n_doe: int

            :param lf: Learning function used as selection criteria to identify the new samples.
                       Options: U, Weighted-U, EFF, EIF and EGIF
            :type n_doe: str

            :param n_add: Number of samples to be selected per iteration.
            :type n_add: int

            :param min_cov: Minimum Covariance used as the stopping criteria of AKMCS method in case of relaibilty
                            analysis.
            :type min_cov: int

            :param n_stop: Final number of samples to be selected as design point from Population.
            :type n_stop: int

            :param max_p: Maximum possible value of probabilty density function of samples. Only required with
                          'Weighted-U' learning function.
            :type max_p: float

            :param reg_model: Regression model used to estimate gradient by using kriging surrogate. Only required
                               if kriging is used as surrogate approximation.
            :type reg_model: str

            :param corr_model: Correlation model used to estimate gradient by using kriging surrogate. Only required
                               if kriging is used as surrogate approximation.
            :type corr_model: str

            :param corr_model_params: Correlation model parameters used to estimate hyperparamters for kriging
                                      surrogate.
            :type corr_model_params: ndarray

            :param n_opt: Number of times optimization problem is to be solved with different starting point.
                          Default: 1
            :type n_opt: int

        Output:
            :return: AKMCS.DoE: Final/expanded samples.
            :rtype: AKMCS.DoE: ndarray

            :return: AKMCS.values:
            :rtype: AKMCS.values: ndarray

            :return: AKMCS.pr: Prediction function for the final surrogate model.
            :rtype: AKMCS.pf: function

            :return: AKMCS.pf: Probability of failure. Available as an output only after Reliability Analysis.
            :rtype: AKMCS.pf: int

            :return: AKMCS.cov_pf: Covariance of probability of failure.  Available as an output only after Reliability
                                   Analysis.
            :rtype: AKMCS.pf: int
    """

    # Authors: Mohit S. Chauhan
    # Last modified: 08/04/2019 by Mohit S. Chauhan

    def __init__(self, run_model_object=None, sample_object=None, nlearn=10000, nstart=None, population=None,
                 dist_name=None, dist_params=None, nsamples=None, corr_model='Gaussian', reg_model='Linear',
                 corr_model_params=None, n_add=1, qoi_name=None, n_opt=10, lf=None, min_cov=None, max_p=None,
                 verbose = False, kriging='UQpy', visualize=False, **kwargs):

        # TODO: Modify Kriging so it can be passed in as an object. This will change some of the code below.

        # Initialize the internal variables of the class.
        self.run_model_object = run_model_object
        self.sample_object = sample_object
        self.nlearn = nlearn
        self.nstart = nstart
        self.verbose = verbose
        self.qoi_name = qoi_name
        self.n_opt = n_opt

        # TODO: Make self.lf a function itself. That way, it can be passed in or it can be assigned to one of the
        #  internal functions based on the keyword lf.
        self.lf = lf
        self.min_cov = min_cov
        self.max_p = max_p
        self.dist_name = dist_name
        self.dist_params = dist_params
        self.nsamples = nsamples
        self.corr_model = corr_model
        self.corr_model_params = corr_model_params
        self.reg_model = reg_model
        self.n_opt = n_opt

        # TODO: Allow the code to add more than one point in a give iteration.
        self.n_add = n_add
        self.population = population
        self.kriging = kriging
        self.visualize = visualize

        # Initialize and run preliminary error checks.
        self.init_akmcs()

        # Run AKMCS
        self.run_akmcs(kwargs)


    def run_akmcs(self, *args):
        from UQpy.RunModel import RunModel
        if self.kriging == 'UQpy':
            from UQpy.Surrogates import Krig
        else:
            from sklearn.gaussian_process import GaussianProcessRegressor

        # Check if the initial sample design already exists and has model evalutions with it.
        # If it does not, run the initial calculations.
        if self.run_model_object is None:
            raise NotImplementedError('UQpy: AKMCS requires a predefined RunModel object.')
        elif self.sample_object.samples is not None:
            self.dimension = np.shape(self.sample_object.samples)[1]
            self.training_points = self.sample_object.samplesU01
            if self.verbose:
                print('UQpy: AKMCS - Running the initial sample set.')
            if not args:
                self.run_model_object.run(samples=self.sample_object.samples)
            else:
                self.run_model_object.run(args[0], samples=self.sample_object.samples)
        else:
            if self.verbose:
                print('UQpy: AKMCS - Generating the initial sample set using Latin hypercube sampling.')
            self.sample_object = LHS(dist_name=self.dist_name, dist_params=self.dist_params, nsamples=nstart)
            self.training_points = self.sample_object.samplesU01
            self.dimension = np.shape(self.sample_object.samples)[1]
            if self.verbose:
                print('UQpy: AKMCS - Running the initial sample set.')
            if not args:
                self.run_model_object.run(samples=self.sample_object.samples)
            else:
                self.run_model_object.run(args[0], samples=self.sample_object.samples)


        if self.verbose:
            print('UQpy: Performing AK-MCS design...')

        # Initialize the population of samples at which to evaluate the learning function and from which to draw in the
        # sampling.
        if self.population is None:
            self.population = MCS(dist_name=self.sample_object.dist_name, dist_params=self.sample_object.dist_params,
                                  nsamples=self.nlearn)

        # If the quantity of interest is a dictionary, convert it to a list
        qoi = [None] * len(self.run_model_object.qoi_list)
        if type(self.run_model_object.qoi_list[0]) is dict:
            for j in range(len(self.run_model_object.qoi_list)):
                qoi[j] = self.run_model_object.qoi_list[j][self.qoi_name]
        else:
            qoi = self.run_model_object.qoi_list

        # Train the initial Kriging model.
        if self.kriging == 'UQpy':
            with suppress_stdout():  # disable printing output comments
                k = Krig(samples=np.atleast_2d(self.training_points), values=np.atleast_2d(np.array(qoi)).T,
                         corr_model=self.corr_model, reg_model=self.reg_model,
                         corr_model_params=self.corr_model_params, n_opt=self.n_opt)
            self.corr_model_params = k.corr_model_params
            self.krig_model = k.interpolate
        else:
            gp = GaussianProcessRegressor(kernel=self.corr_model_params, n_restarts_optimizer=0)
            gp.fit(self.DoE, values)
            self.krig_model = gp.predict


        # ---------------------------------------------
        # Primary loop for learning and adding samples.
        # ---------------------------------------------

        for i in range(self.training_points.shape[0], self.nsamples):

            # Find all of the points in the population that have not already been integrated into the training set
            rest_pop = np.array([x for x in self.population.samplesU01.tolist() if x not in self.training_points.tolist()])

            # Apply the learning function to identify the new point to run the model.
            # TODO: Rewrite this section so that each learning function is called directly as a function through self.lf
            # TODO: Add the other learning fuctions.
            if self.lf == 'EIGF':
                new_ind = self.eigf(self.krig_model, rest_pop, qoi, i)
                new_point = np.atleast_2d(rest_pop[new_ind])

            # Add the new points to the training set and to the sample set.
            self.training_points = np.vstack([self.training_points, new_point])
            self.sample_object.samplesU01 = np.vstack([self.sample_object.samplesU01, new_point])
            for j in range(self.dimension):
                new_point[0, j] = self.sample_object.distribution[j].icdf(new_point[0, j],
                                                                          self.sample_object.dist_params[j])
            self.sample_object.samples = np.vstack([self.sample_object.samples, new_point])

            # Run the model at the new points
            if not args:
                self.run_model_object.run(samples=np.atleast_2d(new_point))
            else:
                self.run_model_object.run(args[0], samples=np.atleast_2d(new_point))

            # If the quantity of interest is a dictionary, convert it to a list
            qoi = [None] * len(self.run_model_object.qoi_list)
            if type(self.run_model_object.qoi_list[0]) is dict:
                for j in range(len(self.run_model_object.qoi_list)):
                    qoi[j] = self.run_model_object.qoi_list[j][self.qoi_name]
            else:
                qoi = self.run_model_object.qoi_list

            # Retrain the Kriging surrogate model
            if self.kriging == 'UQpy':
                with suppress_stdout():
                    # disable printing output comments
                    k = Krig(samples=np.atleast_2d(self.training_points), values=np.atleast_2d(np.array(qoi)).T,
                             corr_model=self.corr_model, reg_model=self.reg_model,
                             corr_model_params=self.corr_model_params, n_opt=self.n_opt)
                self.corr_model_params = k.corr_model_params
                self.krig_model = k.interpolate
            else:
                gp = GaussianProcessRegressor(kernel=self.corr_model_params, n_restarts_optimizer=0)
                gp.fit(self.DoE, values)
                self.krig_model = gp.predict

            if self.verbose:
                print("Iteration:", i)

        if self.verbose:
            print('UQpy: AKMCS complete')

    # ------------------
    # LEARNING FUNCTIONS
    # ------------------

    def eigf(self, surr, pop, qoi, i):
        # Expected Improvement for Global Fit (EIGF)
        # Refrence: J.N Fuhg, "Adaptive surrogate models for parametric studies", Master's Thesis
        # Link: https://arxiv.org/pdf/1905.05345.pdf
        if self.kriging == 'UQpy':
            g, sig = surr(pop, dy=True)
            sig = np.sqrt(sig)
        else:
            g, sig = surr(pop, return_std=True)
            sig = sig.reshape(sig.size, 1)
        sig[sig == 0.] = 0.00001

        if self.visualize:
            if self.dimension != 2:
                raise NotImplementedError('UQpy: AKMCS - Visualize option is only available when dimension = 2.')

            # Check if the directory exists for placing the generated images.
            if not os.path.exists('./figures_akmcs'):
                os.makedirs('./figures_akmcs')

            # Plot the Kriging model at the population points.
            fig = plt.figure()
            ax = plt.axes()
            im = ax.scatter(list(pop[:, 0]), list(pop[:, 1]), c=list(np.squeeze(g)), s=10, vmin=0, vmax=6,
                            cmap=plt.cm.jet)
            ax.plot(self.training_points[:, 0], self.training_points[:, 1], 'xk')
            fig.colorbar(im, ax=ax)
            plt.savefig('./figures_akmcs/p1_' + str(i) + '.png')
            plt.close(fig)

            # Plot the standard deviation of the Kriging model at the population points.
            fig = plt.figure()
            ax = plt.axes()
            im = ax.scatter(list(pop[:, 0]), list(pop[:, 1]), c=list(np.squeeze(sig)), s=10, vmin=0, vmax=0.6,
                            cmap=plt.cm.jet)
            fig.colorbar(im, ax=ax)
            plt.savefig('./figures_akmcs/p2_' + str(i) + '.png')
            plt.close(fig)

        # Evaluation of the learning function
        # First, find the nearest neighbor in the training set for each point in the population.
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(np.atleast_2d(self.training_points))
        neighbors = knn.kneighbors(np.atleast_2d(pop), return_distance=False)

        qoi_array = np.array([qoi[x] for x in np.squeeze(neighbors)])

        # Compute the learning function at every point in the population.
        u = np.square(np.squeeze(g) - qoi_array) + np.square(np.squeeze(sig))

        if self.visualize:
            # Plot the first term of the EIGF learning function at each of the points in the population.
            fig = plt.figure()
            ax = plt.axes()
            im = ax.scatter(list(pop[:, 0]), list(pop[:, 1]), c=list(np.square(np.squeeze(g) - qoi_array)), s=10,
                            vmin=0, vmax=2, cmap=plt.cm.jet)
            fig.colorbar(im, ax=ax)
            plt.savefig('./figures_akmcs/p3_' + str(i) + '.png')
            plt.close(fig)

        rows = np.argmax(u)

        if self.visualize:
            # Plot the learning function at each of the points in the population.
            fig = plt.figure()
            ax = plt.axes()
            im = ax.scatter(list(pop[:, 0]), list(pop[:, 1]), c=list(np.squeeze(u)), s=10, vmin=0, vmax=2,
                            cmap=plt.cm.jet)
            plt.plot(pop[rows, 0], pop[rows, 1], 'ok')
            fig.colorbar(im, ax=ax)
            plt.savefig('./figures_akmcs/p4_' + str(i) + '.png')
            plt.close(fig)

        return rows

    # This learning function has not yet been tested.
    def u(self, surr, pop):
        # U-function
        # References: B. Echard, N. Gayton and M. Lemaire, "AK-MCS: An active learning reliability method combining
        # Kriging and Monte Carlo Simulation", Structural Safety, Pages 145-154, 2011.
        if self.kriging == 'UQpy':
            g, sig = surr(pop, dy=True)
            sig = np.sqrt(sig)
        else:
            g, sig = surr(pop, return_std=True)
            sig = sig.reshape(sig.size, 1)
        # sig[sig == 0.] = 0.00001

        u = abs(g) / sig
        rows = u[:, 0].argsort()[:self.n_add]

        indicator = False
        if min(u[:, 0]) >= 2:
            indicator = True

        # print(g[rows])
        return pop[rows, :], indicator, g

    # This learning function has not yet been tested.
    def weighted_u(self, surr, pop, a, p_, mp):
        # Probability Weighted U-function
        # References: V.S. Sundar and M.S. Shields, "RELIABILITY ANALYSIS USING ADAPTIVE KRIGING SURROGATES WITH
        # MULTIMODEL INFERENCE".
        if self.kriging == 'UQpy':
            g, sig = surr(pop, dy=True)
            sig = np.sqrt(sig)
        else:
            g, sig = surr(pop, return_std=True)
            sig = sig.reshape(sig.size, 1)
        sig[sig == 0.] = 0.00001

        u = abs(g) / sig
        p1 = p_.pdf(pop, params=self.dist_params).reshape(u.size, 1)
        u_ = u * ((self.max_p - p1) / self.max_p)
        rows = u_[:, 0].argsort()[:self.n_add]

        indicator = False
        if min(u[:, 0]) >= 2:
            indicator = True

        # print(g[rows])
        return pop[rows, :], indicator, g

    # This learning function has not yet been tested.
    def eff(self, surr, pop):
        # Expected Feasibilty Function (EFF)
        # References: B.J. Bichon, M.S. Eldred, L.P.Swiler, S. Mahadevan, J.M. McFarland, "Efficient Global Reliability
        # Analysis for Nonlinear Implicit Performance Functions", AIAA JOURNAL, Volume 46, 2008.
        if self.kriging == 'UQpy':
            g, sig = surr(pop, dy=True)
            sig = np.sqrt(sig)
        else:
            g, sig = surr(pop, return_std=True)
            sig = sig.reshape(sig.size, 1)
        sig[sig == 0.] = 0.00001
        # Reliability threshold: a_ = 0
        # EGRA method: epshilon = 2*sigma(x)
        a_, ep = 0, 2 * sig
        t1 = (a_ - g) / sig
        t2 = (a_ - ep - g) / sig
        t3 = (a_ + ep - g) / sig
        eff = (g - a_) * (2 * sp.norm.cdf(t1) - sp.norm.cdf(t2) - sp.norm.cdf(t3))
        eff += -sig * (2 * sp.norm.pdf(t1) - sp.norm.pdf(t2) - sp.norm.pdf(t3))
        eff += ep*(sp.norm.cdf(t3) - sp.norm.cdf(t2))
        rows = eff[:, 0].argsort()[:self.n_add]
        indicator = False
        if max(eff) <= 0.001:
            indicator = True

        return pop[rows, :], indicator, g

    # This learning function has not yet been tested.
    def eif(self, surr, pop, fm):
        # Expected Improvement Function (EIF)
        # References: D.R. Jones, M. Schonlau, W.J. Welch, "Efficient Global Optimization of Expensive Black-Box
        # Functions", Journal of Global Optimization, Pages 455–492, 1998.
        if self.kriging == 'UQpy':
            g, sig = surr(pop, dy=True)
            sig = np.sqrt(sig)
        else:
            g, sig = surr(pop, return_std=True)
            sig = sig.reshape(sig.size, 1)
        sig[sig == 0.] = 0.00001
        u = (fm - g) * sp.norm.cdf((fm - g) / sig) + sig * sp.norm.pdf((fm - g) / sig)
        rows = u[:, 0].argsort()[(np.size(g) - self.n_add):]

        return rows


    # Initial check for errors
    def init_akmcs(self):

        if type(self.lf).__name__ == 'function':
            self.lf = self.lf
        elif self.lf not in ['EFF', 'U', 'Weighted-U', 'EIF', 'EIGF']:
            raise NotImplementedError("UQpy Error: The provided learning function is not recognized.")

        if type(self.corr_model).__name__ != 'str':
            self.kriging = 'Sklearn'




########################################################################################################################
########################################################################################################################
#                                         Class Markov Chain Monte Carlo
########################################################################################################################


class MCMC:
    """
        Description:
            Generate samples from arbitrary user-specified probability density function using Markov Chain Monte Carlo.
            This class generates samples using Metropolis-Hastings(MH), Modified Metropolis-Hastings,
            or Affine Invariant Ensemble Sampler with stretch moves.
            References:
            S.-K. Au and J. L. Beck,“Estimation of small failure probabilities in high dimensions by subset simulation,”
                Probabilistic Eng. Mech., vol. 16, no. 4, pp. 263–277, Oct. 2001.
            J. Goodman and J. Weare, “Ensemble samplers with affine invariance,” Commun. Appl. Math. Comput. Sci.,vol.5,
                no. 1, pp. 65–80, 2010.
        Input:
            :param dimension: A scalar value defining the dimension of target density function.
                              Default: 1
            :type dimension: int
            :param pdf_proposal_type: Type of proposal density function for MCMC. Only used with algorithm ='MH' or'MMH'
                            Options:
                                    'Normal' : Normal proposal density.
                                    'Uniform' : Uniform proposal density.
                            Default: 'Uniform'
                            If dimension > 1 and algorithm = 'MMH', this may be input as a list to assign different
                            proposal densities to each dimension. Example pdf_proposal_name = ['Normal','Uniform'].
                            If dimension > 1, algorithm = 'MMH' and this is input as a string, the proposal densities
                            for all dimensions are set equal to the assigned proposal type.
            :type pdf_proposal_type: str or str list
            :param pdf_proposal_scale: Scale of the proposal distribution
                            If algorithm == 'MH' or 'MMH'
                                For pdf_proposal_type = 'Uniform'
                                    Proposal is Uniform in [x-pdf_proposal_scale/2, x+pdf_proposal_scale/2].
                                For pdf_proposal_type = 'Normal'
                                    Proposal is Normal with standard deviation equal to pdf_proposal_scale.
                            If algorithm == 'Stretch'
                                pdf_proposal_scale sets the scale of the stretch density.
                                    g(z) = 1/sqrt(z) for z in [1/pdf_proposal_scale, pdf_proposal_scale].
                            Default value: dimension x 1 list of ones.
            :type pdf_proposal_scale: float or float list
                            If dimension > 1, this may be defined as float or float list.
                                If input as float, pdf_proposal_scale is assigned to all dimensions.
                                If input as float list, each element is assigned to the corresponding dimension.
            :param pdf_target: Target density function from which to draw random samples
                            The target joint probability density must be a function, or list of functions, or a string.
                            If type == 'str'
                                The assigned string must refer to a custom pdf defined in the file custom_pdf.py in the
                                 working directory.
                            If type == function
                                The function must be defined in the python script calling MCMC.
                            If dimension > 1 and pdf_target_type='marginal_pdf', the input to pdf_target is a list of
                            size [dimensions x 1] where each item of the list defines a marginal pdf.
                            Default: Multivariate normal distribution having zero mean and unit standard deviation.
            :type pdf_target: function, function list, or str
            :param pdf_target_params: Parameters of the target pdf.
            :type pdf_target_params: list
            :param algorithm:  Algorithm used to generate random samples.
                            Options:
                                'MH': Metropolis Hastings Algorithm
                                'MMH': Component-wise Modified Metropolis Hastings Algorithm
                                'Stretch': Affine Invariant Ensemble MCMC with stretch moves
                            Default: 'MMH'
            :type algorithm: str
            :param jump: Number of samples between accepted states of the Markov chain.
                                Default value: 1 (Accepts every state)
            :type: jump: int
            :param nsamples: Number of samples to generate
                                No Default Value: nsamples must be prescribed
            :type nsamples: int
            :param seed: Seed of the Markov chain(s)
                            For 'MH' and 'MMH', this is a single point, defined as a numpy array of dimension
                             (1 x dimension).
                            For 'Stretch', this is a numpy array of dimension N x dimension, where N is the ensemble
                            size.
                            Default:
                                For 'MH' and 'MMH': zeros(1 x dimension)
                                For 'Stretch': No default, this must be specified.
            :type seed: float or numpy array
            :param nburn: Length of burn-in. Number of samples at the beginning of the chain to discard.
                            This option is only used for the 'MMH' and 'MH' algorithms.
                            Default: nburn = 0
            :type nburn: int
        Output:
            :return: MCMC.samples: Set of MCMC samples following the target distribution
            :rtype: MCMC.samples: ndarray

            :return: MCMC.accept_ratio: Acceptance ratio of the MCMC samples
            :rtype: MCMC.accept_ratio: float

    """

    # Authors: Michael D. Shields, Mohit Chauhan, Dimitris G. Giovanis
    # Updated: 04/08/2019 by Audrey Olivier

    def __init__(self, dimension=None, pdf_proposal_type=None, pdf_proposal_scale=None,
                 pdf_target=None, log_pdf_target=None, pdf_target_params=None, pdf_target_copula=None,
                 pdf_target_copula_params=None, pdf_target_type='joint_pdf',
                 algorithm='MH', jump=1, nsamples=None, seed=None, nburn=0,
                 verbose=False):

        self.pdf_proposal_type = pdf_proposal_type
        self.pdf_proposal_scale = pdf_proposal_scale
        self.pdf_target = pdf_target
        self.log_pdf_target = log_pdf_target
        self.pdf_target_params = pdf_target_params
        self.pdf_target_copula = pdf_target_copula
        self.pdf_target_copula_params = pdf_target_copula_params
        self.algorithm = algorithm
        self.jump = jump
        self.nsamples = nsamples
        self.dimension = dimension
        self.seed = seed
        self.nburn = nburn
        self.pdf_target_type = pdf_target_type
        self.init_mcmc()
        self.verbose = verbose
        if self.algorithm is 'Stretch':
            self.ensemble_size = len(self.seed)
        self.samples, self.accept_ratio = self.run_mcmc()

    def run_mcmc(self):
        n_accepts = 0

        # Defining an array to store the generated samples
        samples = np.zeros([self.nsamples * self.jump + self.nburn, self.dimension])

        ################################################################################################################
        # Classical Metropolis-Hastings Algorithm with symmetric proposal density
        if self.algorithm == 'MH':
            samples[0, :] = self.seed.reshape((-1,))
            log_pdf_ = self.log_pdf_target
            log_p_current = log_pdf_(samples[0, :])

            # Loop over the samples
            for i in range(self.nsamples * self.jump - 1 + self.nburn):
                if self.pdf_proposal_type[0] == 'Normal':
                    cholesky_cov = np.diag(self.pdf_proposal_scale)
                    z_normal = np.random.normal(size=(self.dimension, ))
                    candidate = samples[i, :] + np.matmul(cholesky_cov, z_normal)
                    log_p_candidate = log_pdf_(candidate)
                    log_p_accept = log_p_candidate - log_p_current
                    accept = np.log(np.random.random()) < log_p_accept

                    if accept:
                        samples[i + 1, :] = candidate
                        log_p_current = log_p_candidate
                        n_accepts += 1
                    else:
                        samples[i + 1, :] = samples[i, :]

                elif self.pdf_proposal_type[0] == 'Uniform':
                    low = -np.array(self.pdf_proposal_scale) / 2
                    high = np.array(self.pdf_proposal_scale) / 2
                    candidate = samples[i, :] + np.random.uniform(low=low, high=high, size=(self.dimension, ))
                    log_p_candidate = log_pdf_(candidate)
                    log_p_accept = log_p_candidate - log_p_current
                    accept = np.log(np.random.random()) < log_p_accept

                    if accept:
                        samples[i + 1, :] = candidate
                        log_p_current = log_p_candidate
                        n_accepts += 1
                    else:
                        samples[i + 1, :] = samples[i, :]
            accept_ratio = n_accepts/(self.nsamples * self.jump - 1 + self.nburn)

        ################################################################################################################
        # Modified Metropolis-Hastings Algorithm with symmetric proposal density
        elif self.algorithm == 'MMH':

            samples[0, :] = self.seed.reshape((-1,))

            if self.pdf_target_type == 'marginal_pdf':
                list_log_p_current = []
                for j in range(self.dimension):
                    log_pdf_ = self.log_pdf_target[j]
                    list_log_p_current.append(log_pdf_(samples[0, j]))
                for i in range(self.nsamples * self.jump - 1 + self.nburn):
                    for j in range(self.dimension):

                        log_pdf_ = self.log_pdf_target[j]

                        if self.pdf_proposal_type[j] == 'Normal':
                            candidate = np.random.normal(samples[i, j], self.pdf_proposal_scale[j], size=1)
                            log_p_candidate = log_pdf_(candidate)
                            log_p_current = list_log_p_current[j]
                            log_p_accept = log_p_candidate - log_p_current

                            accept = np.log(np.random.random()) < log_p_accept

                            if accept:
                                samples[i + 1, j] = candidate
                                list_log_p_current[j] = log_p_candidate
                                n_accepts += 1 / self.dimension
                            else:
                                samples[i + 1, j] = samples[i, j]

                        elif self.pdf_proposal_type[j] == 'Uniform':
                            candidate = np.random.uniform(low=samples[i, j] - self.pdf_proposal_scale[j] / 2,
                                                          high=samples[i, j] + self.pdf_proposal_scale[j] / 2, size=1)
                            log_p_candidate = log_pdf_(candidate)
                            log_p_current = list_log_p_current[j]
                            log_p_accept = log_p_candidate - log_p_current

                            accept = np.log(np.random.random()) < log_p_accept

                            if accept:
                                samples[i + 1, j] = candidate
                                list_log_p_current[j] = log_p_candidate
                                n_accepts += 1 / self.dimension
                            else:
                                samples[i + 1, j] = samples[i, j]
            else:
                log_pdf_ = self.log_pdf_target

                for i in range(self.nsamples * self.jump - 1 + self.nburn):
                    candidate = np.copy(samples[i, :])
                    current = np.copy(samples[i, :])
                    log_p_current = log_pdf_(samples[i, :])
                    for j in range(self.dimension):
                        if self.pdf_proposal_type[j] == 'Normal':
                            candidate[j] = np.random.normal(samples[i, j], self.pdf_proposal_scale[j])

                        elif self.pdf_proposal_type[j] == 'Uniform':
                            candidate[j] = np.random.uniform(low=samples[i, j] - self.pdf_proposal_scale[j] / 2,
                                                             high=samples[i, j] + self.pdf_proposal_scale[j] / 2,
                                                             size=1)

                        log_p_candidate = log_pdf_(candidate)
                        log_p_accept = log_p_candidate - log_p_current

                        accept = np.log(np.random.random()) < log_p_accept

                        if accept:
                            current[j] = candidate[j]
                            log_p_current = log_p_candidate
                            n_accepts += 1
                        else:
                            candidate[j] = current[j]

                    samples[i + 1, :] = current
            accept_ratio = n_accepts / (self.nsamples * self.jump - 1 + self.nburn)

        ################################################################################################################
        # Affine Invariant Ensemble Sampler with stretch moves

        elif self.algorithm == 'Stretch':

            samples[0:self.ensemble_size, :] = self.seed
            log_pdf_ = self.log_pdf_target
            # list_log_p_current = [log_pdf_(samples[i, :], self.pdf_target_params) for i in range(self.ensemble_size)]

            for i in range(self.ensemble_size - 1, self.nsamples * self.jump - 1):
                complementary_ensemble = samples[i - self.ensemble_size + 2:i + 1, :]
                s0 = random.choice(complementary_ensemble)
                s = (1 + (self.pdf_proposal_scale[0] - 1) * random.random()) ** 2 / self.pdf_proposal_scale[0]
                candidate = s0 + s * (samples[i - self.ensemble_size + 1, :] - s0)

                log_p_candidate = log_pdf_(candidate)
                log_p_current = log_pdf_(samples[i - self.ensemble_size + 1, :])
                # log_p_current = list_log_p_current[i - self.ensemble_size + 1]
                log_p_accept = np.log(s ** (self.dimension - 1)) + log_p_candidate - log_p_current

                accept = np.log(np.random.random()) < log_p_accept

                if accept:
                    samples[i + 1, :] = candidate.reshape((-1, ))
                    # list_log_p_current.append(log_p_candidate)
                    n_accepts += 1
                else:
                    samples[i + 1, :] = samples[i - self.ensemble_size + 1, :]
                    # list_log_p_current.append(list_log_p_current[i - self.ensemble_size + 1])
            accept_ratio = n_accepts / (self.nsamples * self.jump - self.ensemble_size)


        ################################################################################################################
        # Return the samples

        if self.algorithm is 'MMH' or self.algorithm is 'MH':
            if self.verbose:
                print('Successful execution of the MCMC design')
            return samples[self.nburn:self.nsamples * self.jump + self.nburn:self.jump], accept_ratio
        else:
            output = np.zeros((self.nsamples, self.dimension))
            j = 0
            for i in range(self.jump * self.ensemble_size - self.ensemble_size, samples.shape[0],
                           self.jump * self.ensemble_size):
                output[j:j + self.ensemble_size, :] = samples[i:i + self.ensemble_size, :]
                j = j + self.ensemble_size
            return output, accept_ratio

    ####################################################################################################################
    # Check to ensure consistency of the user input and assign defaults
    def init_mcmc(self):

        # Check dimension
        if self.dimension is None:
            self.dimension = 1

        # Check nsamples
        if self.nsamples is None:
            raise NotImplementedError('Exit code: Number of samples not defined.')

        # Check nburn
        if self.nburn is None:
            self.nburn = 0

        # Check jump
        if self.jump is None:
            self.jump = 1
        if self.jump == 0:
            raise ValueError("Exit code: Value of jump must be greater than 0")

        # Check seed
        if self.algorithm is not 'Stretch':
            if self.seed is None:
                self.seed = np.zeros(self.dimension)
            self.seed = np.array(self.seed)
            if (len(self.seed.shape) == 1) and (self.seed.shape[0] != self.dimension):
                raise NotImplementedError("Exit code: Incompatible dimensions in 'seed'.")
            self.seed = self.seed.reshape((1, -1))
        else:
            if self.seed is None or len(self.seed.shape) != 2:
                raise NotImplementedError("For Stretch algorithm, a seed must be given as a ndarray")
            if self.seed.shape[1] != self.dimension:
                raise NotImplementedError("Exit code: Incompatible dimensions in 'seed'.")
            if self.seed.shape[0] < 3:
                raise NotImplementedError("Exit code: Ensemble size must be > 2.")

        # Check algorithm
        if self.algorithm is None:
            self.algorithm = 'MH'
        if self.algorithm not in ['MH', 'MMH', 'Stretch']:
            raise NotImplementedError('Exit code: Unrecognized MCMC algorithm. Supported algorithms: '
                                      'Metropolis-Hastings (MH), '
                                      'Modified Metropolis-Hastings (MMH), '
                                      'Affine Invariant Ensemble with Stretch Moves (Stretch).')

        # Check pdf_proposal_type
        if self.pdf_proposal_type is None:
            self.pdf_proposal_type = 'Normal'
        # If pdf_proposal_type is entered as a string, make it a list
        if isinstance(self.pdf_proposal_type, str):
            self.pdf_proposal_type = [self.pdf_proposal_type]
        for i in self.pdf_proposal_type:
            if i not in ['Uniform', 'Normal']:
                raise ValueError('Exit code: Unrecognized type for proposal distribution. Supported distributions: '
                                 'Uniform, '
                                 'Normal.')
        if self.algorithm is 'MH' and len(self.pdf_proposal_type) != 1:
            raise ValueError('Exit code: MH algorithm can only take one proposal distribution.')
        elif len(self.pdf_proposal_type) != self.dimension:
            if len(self.pdf_proposal_type) == 1:
                self.pdf_proposal_type = self.pdf_proposal_type * self.dimension
            else:
                raise NotImplementedError("Exit code: Incompatible dimensions in 'pdf_proposal_type'.")

        # Check pdf_proposal_scale
        if self.pdf_proposal_scale is None:
            if self.algorithm == 'Stretch':
                self.pdf_proposal_scale = 2
            else:
                self.pdf_proposal_scale = 1
        if not isinstance(self.pdf_proposal_scale, list):
            self.pdf_proposal_scale = [self.pdf_proposal_scale]
        if len(self.pdf_proposal_scale) != self.dimension:
            if len(self.pdf_proposal_scale) == 1:
                self.pdf_proposal_scale = self.pdf_proposal_scale * self.dimension
            else:
                raise NotImplementedError("Exit code: Incompatible dimensions in 'pdf_proposal_scale'.")

        # Check log_pdf_target and pdf_target
        if self.log_pdf_target is None and self.pdf_target is None:
            raise ValueError('UQpy error: a target function must be provided, in log_pdf_target of pdf_target')
        if isinstance(self.log_pdf_target, list) and len(self.log_pdf_target) != self.dimension:
            raise ValueError('UQpy error: inconsistent dimensions.')
        if isinstance(self.pdf_target, list) and len(self.pdf_target) != self.dimension:
            raise ValueError('UQpy error: inconsistent dimensions.')

        # Check pdf_target_type
        if self.pdf_target_type not in ['joint_pdf', 'marginal_pdf']:
            raise ValueError('pdf_target_type should be "joint_pdf", "marginal_pdf"')

        # Check MMH
        if self.algorithm is 'MMH':
            if (self.pdf_target_type == 'marginal_pdf') and (self.pdf_target_copula is not None):
                raise ValueError('UQpy error: MMH with pdf_target_type="marginal" cannot be used when the'
                                 'target pdf has a copula, use pdf_target_type="joint" instead')

        # If pdf_target or log_pdf_target are given as lists, they should be of the right dimension
        if isinstance(self.log_pdf_target, list):
            if len(self.log_pdf_target) != self.dimension:
                raise ValueError('log_pdf_target given as a list should have length equal to dimension')
            if (self.pdf_target_params is not None) and (len(self.log_pdf_target) != len(self.pdf_target_params)):
                raise ValueError('pdf_target_params should be given as a list of length equal to log_pdf_target')
        if isinstance(self.pdf_target, list):
            if len(self.pdf_target) != self.dimension:
                raise ValueError('pdf_target given as a list should have length equal to dimension')
            if (self.pdf_target_params is not None) and (len(self.pdf_target) != len(self.pdf_target_params)):
                raise ValueError('pdf_target_params should be given as a list of length equal to pdf_target')


        # Define a helper function
        def compute_log_pdf(x, pdf_func, params=None, copula_params=None):
            kwargs_ = {}
            if params is not None:
                kwargs_['params'] = params
            if copula_params is not None:
                kwargs_['copula_params'] = copula_params
            pdf_value = max(pdf_func(x, **kwargs_), 10 ** (-320))
            return np.log(pdf_value)

        # Either pdf_target or log_pdf_target must be defined
        if (self.pdf_target is None) and (self.log_pdf_target is None):
            raise ValueError('The target distribution must be defined, using inputs'
                             ' log_pdf_target or pdf_target.')
        # For MMH with pdf_target_type == 'marginals', pdf_target or its log should be lists
        if (self.algorithm == 'MMH') and (self.pdf_target_type == 'marginal_pdf'):
            kwargs = [{}]*self.dimension
            for j in range(self.dimension):
                if self.pdf_target_params is not None:
                    kwargs[j]['params'] = self.pdf_target_params[j]
                if self.pdf_target_copula_params is not None:
                    kwargs[j]['copula_params'] = self.pdf_target_copula_params[j]

            if self.log_pdf_target is not None:
                if not isinstance(self.log_pdf_target, list):
                    raise ValueError('For MMH algo with pdf_target_type="marginal_pdf", '
                                     'log_pdf_target should be a list')
                if isinstance(self.log_pdf_target[0], str):
                    p_js = [Distribution(dist_name=pdf_target_j) for pdf_target_j in self.pdf_target]
                    try:
                        [p_j.log_pdf(x=self.seed[0, j], **kwargs[j]) for (j, p_j) in enumerate(p_js)]
                        self.log_pdf_target = [partial(p_j.log_pdf, **kwargs[j]) for (j, p_j) in enumerate(p_js)]
                    except AttributeError:
                        raise AttributeError('log_pdf_target given as a list of strings must point to Distributions '
                                             'with an existing log_pdf method.')
                elif callable(self.log_pdf_target[0]):
                    self.log_pdf_target = [partial(pdf_target_j, **kwargs[j]) for (j, pdf_target_j)
                                           in enumerate(self.log_pdf_target)]
                else:
                    raise ValueError('log_pdf_target must be a list of strings or a list of callables')
            else:
                if not isinstance(self.pdf_target, list):
                    raise ValueError('For MMH algo with pdf_target_type="marginal_pdf", '
                                     'pdf_target should be a list')
                if isinstance(self.pdf_target[0], str):
                    p_js = [Distribution(dist_name=pdf_target_j) for pdf_target_j in self.pdf_target]
                    try:
                        [p_j.pdf(x=self.seed[0, j], **kwargs[j]) for (j, p_j) in enumerate(p_js)]
                        self.log_pdf_target = [partial(compute_log_pdf, pdf_func=p_j.pdf, **kwargs[j])
                                               for (j, p_j) in enumerate(p_js)]
                    except AttributeError:
                        raise AttributeError('pdf_target given as a list of strings must point to Distributions '
                                             'with an existing pdf method.')
                elif callable(self.pdf_target[0]):
                    self.log_pdf_target = [partial(compute_log_pdf, pdf_func=pdf_target_j, **kwargs[j])
                                           for (j, pdf_target_j) in enumerate(self.pdf_target)]
                else:
                    raise ValueError('pdf_target must be a list of strings or a list of callables')
        else:
            kwargs = {}
            if self.pdf_target_params is not None:
                kwargs['params'] = self.pdf_target_params
            if self.pdf_target_copula_params is not None:
                kwargs['copula_params'] = self.pdf_target_copula_params

            if self.log_pdf_target is not None:
                if isinstance(self.log_pdf_target, str) or (isinstance(self.log_pdf_target, list)
                                                            and isinstance(self.log_pdf_target[0], str)):
                    p = Distribution(dist_name=self.log_pdf_target, copula=self.pdf_target_copula)
                    try:
                        p.log_pdf(x=self.seed[0, :], **kwargs)
                        self.log_pdf_target = partial(p.log_pdf, **kwargs)
                    except AttributeError:
                        raise AttributeError('log_pdf_target given as a string must point to a Distribution '
                                             'with an existing log_pdf method.')
                elif callable(self.log_pdf_target):
                    self.log_pdf_target = partial(self.log_pdf_target, **kwargs)
                else:
                    raise ValueError('For MH and Stretch, log_pdf_target must be a callable function, '
                                     'a str or list of str')
            else:
                if isinstance(self.pdf_target, str) or (isinstance(self.pdf_target, list)
                                                        and isinstance(self.pdf_target[0], str)):
                    p = Distribution(dist_name=self.pdf_target, copula=self.pdf_target_copula)
                    try:
                        p.pdf(x=self.seed[0, :], **kwargs)
                        self.log_pdf_target = partial(compute_log_pdf, pdf_func=p.pdf, **kwargs)
                    except AttributeError:
                        raise AttributeError('pdf_target given as a string must point to a Distribution '
                                             'with an existing pdf method.')
                elif callable(self.pdf_target):
                    self.log_pdf_target = partial(compute_log_pdf, pdf_func=self.pdf_target, **kwargs)
                else:
                    raise ValueError('For MH and Stretch, pdf_target must be a callable function, '
                                     'a str or list of str')


########################################################################################################################
########################################################################################################################
#                                         Importance Sampling
########################################################################################################################

class IS:
    """
        Description:

            Perform Importance Sampling (IS) of independent random variables given a target and a
            proposal distribution.

        Input:

            :param pdf_proposal: A list containing the names of the proposal distribution for each random variable.
                                 Distribution names must match those in the Distributions module.
                                 If the distribution does not match one from the Distributions module, the user
                                 must provide custom_dist.py. The length of the string must be 1 (if all
                                 distributions are the same) or equal to dimension.
            :type pdf_proposal: string list

            :param pdf_proposal_params: Parameters of the proposal distribution.
                                        Parameters for each random variable are defined as ndarrays.
                                        Each item in the list, pdf_proposal_params[i], specifies the parameters for the
                                        corresponding proposal distribution, pdf_proposal[i].
            :type pdf_proposal_params: list

            :param pdf_target: A list containing the names of the target distribution for each random variable.
                                 Distribution names must match those in the Distributions module.
                                 If the distribution does not match one from the Distributions module, the user
                                 must provide custom_dist.py. The length of the string must be 1 (if all
                                 distributions are the same) or equal to dimension.
            :type pdf_target: string list

            :param pdf_target_params: Parameters of the target distribution.
                                        Parameters for each random variable are defined as ndarrays.
                                        Each item in the list, pdf_target_params[i], specifies the parameters for the
                                        corresponding target distribution, pdf_target[i].
            :type pdf_target_params: list

            :param nsamples: Number of samples to generate.
                             No Default Value: nsamples must be prescribed.
            :type nsamples: int

        Output:
            :return: IS.samples: Set of generated samples
            :rtype: IS.samples: ndarray

            :return: IS.weights: Importance weights of samples
            :rtype: IS.weights: ndarray
    """

    # Authors: Dimitris G.Giovanis
    # Last Modified: 04/08/2019 by Audrey Olivier

    def __init__(self, nsamples=None,
                 pdf_proposal=None, pdf_proposal_params=None,
                 pdf_target=None, log_pdf_target=None, pdf_target_params=None,
                 pdf_target_copula=None, pdf_target_copula_params=None
                 ):

        self.nsamples = nsamples
        self.pdf_proposal = pdf_proposal
        self.pdf_proposal_params = pdf_proposal_params
        self.pdf_target = pdf_target
        self.log_pdf_target = log_pdf_target
        self.pdf_target_params = pdf_target_params
        self.pdf_target_copula = pdf_target_copula
        self.pdf_target_copula_params = pdf_target_copula_params

        self.init_is()

        # Step 1: sample from proposal
        self.samples = self.sampling_step()
        # Step 2: weight samples
        self.unnormalized_log_weights, self.weights = self.weighting_step()

    def sampling_step(self):

        proposal_pdf_ = Distribution(dist_name=self.pdf_proposal)
        samples = proposal_pdf_.rvs(params=self.pdf_proposal_params, nsamples=self.nsamples)
        return samples

    def weighting_step(self):

        x = self.samples
        # evaluate qs (log_pdf_proposal)
        proposal_pdf_ = Distribution(dist_name=self.pdf_proposal)
        try:
            log_qs = proposal_pdf_.log_pdf(x, params=self.pdf_proposal_params)
        except AttributeError:
            log_qs = np.log(proposal_pdf_.pdf(x, params=self.pdf_proposal_params))
        # evaluate ps (log_pdf_target)
        log_ps = self.log_pdf_target(x)

        log_weights = log_ps-log_qs
        # this rescale is used to avoid having NaN of Inf when taking the exp
        weights = np.exp(log_weights - max(log_weights))
        sum_w = np.sum(weights, axis=0)
        return log_weights, weights/sum_w

    def resample(self, method='multinomial', size=None):
        from .Utilities import resample
        return resample(self.samples, self.weights, method=method, size=size)

    ################################################################################################################
    # Initialize Importance Sampling.

    def init_is(self):

        # Check nsamples
        if self.nsamples is None:
            raise NotImplementedError('Exit code: Number of samples is not defined.')

        # helper function
        def compute_log_pdf(x, pdf_func, params, copula_params):
            kwargs_ = {}
            if params is not None:
                kwargs_['params'] = params
            if copula_params is not None:
                kwargs_['copula_params'] = copula_params
            tmp = pdf_func(x, **kwargs_)
            pdf_value = np.fmax(tmp, 10 ** (-320)*np.ones_like(tmp))
            return np.log(pdf_value)
        # Check log_pdf_target, pdf_target
        if (self.pdf_target is None) and (self.log_pdf_target is None):
            raise ValueError('UQpy error: a target pdf must be defined (pdf_target or log_pdf_target).')
        # The code first checks if log_pdf_target is defined, if yes, no need to check pdf_target
        x_test = Distribution(dist_name=self.pdf_proposal).rvs(params=self.pdf_proposal_params, nsamples=1)
        kwargs = {}
        if self.pdf_target_params is not None:
            kwargs['params'] = self.pdf_target_params
        if self.pdf_target_copula_params is not None:
            kwargs['copula_params'] = self.pdf_target_copula_params
        if self.log_pdf_target is not None:
            # log_pdf_target can be defined as a callable or string.
            if isinstance(self.log_pdf_target, str) or (isinstance(self.log_pdf_target, list) and
                                                        isinstance(self.log_pdf_target[0], str)):
                p = Distribution(dist_name=self.log_pdf_target, copula=self.pdf_target_copula)
                try:
                    p.log_pdf(x=x_test, **kwargs)
                    self.log_pdf_target = partial(p.log_pdf, **kwargs)
                except AttributeError:
                    raise AttributeError('log_pdf_target given as a string must point to a Distribution '
                                         'with an existing log_pdf method.')
            elif callable(self.log_pdf_target):
                self.log_pdf_target = partial(self.log_pdf_target, **kwargs)
            else:
                raise ValueError('log_pdf_target should be a callable or a string/list of strings.')
        else:
            # pdf_target can be a str of list of strings, then compute the log_pdf
            if isinstance(self.pdf_target, str) or (isinstance(self.pdf_target, list) and
                                                    isinstance(self.pdf_target[0], str)):
                p = Distribution(dist_name=self.pdf_target, copula=self.pdf_target_copula)
                try:
                    p.pdf(x=x_test, **kwargs)
                    self.log_pdf_target = partial(compute_log_pdf, pdf_func=p.pdf, **kwargs)
                except AttributeError:
                    raise AttributeError('pdf_target given as a string must point to a Distribution '
                                         'with an existing pdf method.')
            # otherwise it may be a function that computes the pdf, then just take the logarithm
            elif callable(self.pdf_target):
                self.log_pdf_target = partial(compute_log_pdf, pdf_func=self.pdf_target, **kwargs)
            else:
                raise ValueError('pdf_target should be a callable or a string/list of strings.')

        # Check pdf_proposal_name
        if self.pdf_proposal is None:
            raise ValueError('Exit code: A proposal distribution is required.')
        # can be given as a name or a list of names, transform it to a distribution class
        if not isinstance(self.pdf_proposal, str) and not (isinstance(self.pdf_proposal, list)
           and isinstance(self.pdf_proposal[0], str)):
            raise ValueError('UQpy error: proposal pdf must be given as a str or a list of str')
