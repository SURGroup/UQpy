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
from functools import partial
import warnings


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

            :return: MCS.samplesU01: If the Distribution object has a .cdf method, MCS also returns the samples in the
            Uniform(0,1) hypercube.
            :rtype: MCS.samplesU01: ndarray of dimension(nsamples, ndim)

    """

    # Authors: Dimitris G.Giovanis
    # Last Modified: 11/25/2019 by Michael D. Shields

    def __init__(self, dist_name=None, dist_params=None, nsamples=None, var_names=None, verbose=False):

        # No need to do other checks as they will be done within Distributions.py
        self.dist_name = dist_name
        self.dist_params = dist_params
        self.var_names = var_names
        self.verbose = verbose
        self.nsamples = nsamples
        if self.verbose:
            print('UQpy: MCS object created.')

        self.samples = None
        self.samplesU01 = None

        if nsamples is not None:
            self.sample(nsamples)

    def sample(self, nsamples):
        self.nsamples = nsamples
        if nsamples is None:
            raise ValueError('UQpy error: nsamples must be defined.')
        if not isinstance(nsamples, int):
            raise ValueError('UQpy error: nsamples must be integer valued.')

        if self.verbose:
            print('UQpy: Running Monte Carlo Sampling...')

        samples_new = Distribution(dist_name=self.dist_name).rvs(params=self.dist_params, nsamples=nsamples)

        # Shape the arrays as (1,n) if nsamples=1, and (n,1) if nsamples=n
        if len(samples_new.shape) == 1:
            if self.nsamples == 1:
                samples_new = samples_new.reshape((1, -1))
            else:
                samples_new = samples_new.reshape((-1, 1))

        # If self.samples already has existing samples,
        # append the new samples to the existing attribute.
        if self.samples is None:
            self.samples = samples_new
        else:
            self.samples = np.concatenate([self.samples, samples_new], axis=0)

        att = (hasattr(Distribution(dist_name=self.dist_name[i]), 'cdf') for i in range(samples_new.shape[1]))
        if all(att):
            samples_u01_new = np.zeros_like(samples_new)
            for i in range(samples_new.shape[1]):
                samples_u01_new[:, i] = Distribution(dist_name=self.dist_name[i]).cdf(
                    x=np.atleast_2d(samples_new[:, i]).T, params=self.dist_params[i])
            if len(samples_u01_new.shape) == 1:
                if self.nsamples == 1:
                    samples_u01_new = samples_u01_new.reshape((1, -1))
                else:
                    samples_u01_new = samples_u01_new.reshape((-1, 1))

            # If self.samplesU01 already has existing samplesU01,
            # append the new samples to the existing attribute.
            if self.samplesU01 is None:
                self.samplesU01 = samples_u01_new
            else:
                self.samplesU01 = np.concatenate([self.samplesU01, samples_u01_new], axis=0)

        if self.verbose:
            print('UQpy: Monte Carlo Sampling Complete.')

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
                 lhs_iter=100, var_names=None, nsamples=None, verbose=False):

        self.nsamples = nsamples
        self.dist_name = dist_name
        self.dist_params = dist_params
        self.dimension = len(self.dist_name)
        self.lhs_criterion = lhs_criterion
        self.lhs_metric = lhs_metric
        self.lhs_iter = lhs_iter
        self.init_lhs()
        self.var_names = var_names
        self.verbose = verbose

        self.distribution = [None] * self.dimension
        for i in range(self.dimension):
            self.distribution[i] = Distribution(dist_name=self.dist_name[i])

        self.samplesU01, self.samples = self.run_lhs()

    def run_lhs(self):

        if self.verbose:
            print('UQpy: Running Latin Hypercube Sampling...')

        cut = np.linspace(0, 1, self.nsamples + 1)
        a = cut[:self.nsamples]
        b = cut[1:self.nsamples + 1]

        samples = self._samples(a, b)

        samples_u_to_x = np.zeros_like(samples)
        for j in range(samples.shape[1]):
            i_cdf = self.distribution[j].icdf
            samples_u_to_x[:, j] = i_cdf(samples[:, j], self.dist_params[j])

        if self.verbose:
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

        if self.verbose:
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

        if self.verbose:
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

        self.init_sts()
        self.distribution = [None] * self.dimension
        for i in range(self.dimension):
            self.distribution[i] = Distribution(self.dist_name[i])

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

                samples_u_to_x[:, j] = i_cdf(np.atleast_2d(samples[:, j]).T, self.dist_params[j])

            print('UQpy: Successful execution of STS design..')
            return samples, samples_u_to_x

        elif self.stype == 'Voronoi':
            from UQpy.Utilities import compute_Voronoi_centroid_volume, voronoi_unit_hypercube

            samples_init = np.random.rand(self.nsamples, self.dimension)

            for i in range(self.n_iters):
                # x = self.in_hypercube(samples_init)
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
                self.samples[:, i] = self.distribution[i].icdf(np.atleast_2d(self.samplesU01[:, i]).T,
                                                               self.dist_params[i]).T

    def in_hypercube(self, samples):

        in_cube = True * self.nsamples
        for i in range(self.dimension):
            in_cube = np.logical_and(in_cube, np.logical_and(0 <= samples[:, i], samples[:, i] <= 1))

        return in_cube

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
            :param run_model_object: A RunModel object, which is used to evaluate the function value
            :type run_model_object: class

            :param sample_object: A SampleMethods class object, which contains information about existing samples
            :type sample_object: class

            :param krig_object: A kriging class object, only  required if meta is 'Kriging'.
            :type krig_object: class

            :param local: Indicator to update surrogate locally.
            :type local: boolean

            :param max_train_size: Minimum size of training data around new sample used to update surrogate.
                                   Default: nsamples
            :type max_train_size: int

            :param step_size: Step size to calculate the gradient using central difference. Only required if Delaunay is
                              used as surrogate approximation.
            :type step_size: float

            :param n_add: Number of samples generated in each iteration
            :type n_add: int

            :param verbose: A boolean declaring whether to write text to the terminal.
            :type verbose: bool

        Output:
            :return: RSS.sample_object.samples: Final/expanded samples.
            :rtype: RSS.sample_object.samples: ndarray

    """

    # Authors: Mohit S. Chauhan
    # Last modified: 01/07/2020 by Mohit S. Chauhan

    def __init__(self, sample_object=None, run_model_object=None, krig_object=None, local=False, max_train_size=None,
                 step_size=0.005, qoi_name=None, n_add=1, verbose=False):

        # Initialize attributes that are common to all approaches
        self.sample_object = sample_object
        self.run_model_object = run_model_object
        self.verbose = verbose
        self.nsamples = 0

        self.cell = self.sample_object.stype
        self.dimension = np.shape(self.sample_object.samples)[1]
        self.nexist = 0
        self.n_add = n_add

        if self.cell == 'Voronoi':
            self.mesh = []
            self.mesh_vertices, self.vertices_in_U01 = [], []
            self.points_to_samplesU01, self.training_points = [], []

        # Run Initial Error Checks
        self.init_rss()

        if run_model_object is not None:
            self.local = local
            self.max_train_size = max_train_size
            self.krig_object = krig_object
            self.qoi_name = qoi_name
            self.step_size = step_size
            if self.verbose:
                print('UQpy: GE-RSS - Running the initial sample set.')
            self.run_model_object.run(samples=self.sample_object.samples)
            if self.verbose:
                print('UQpy: GE-RSS - A RSS class object has been initiated.')
        else:
            if self.verbose:
                print('UQpy: RSS - A RSS class object has been initiated.')

    def sample(self, nsamples=0, n_add=None):
        """
                Inputs:
                    :param nsamples: Final size of the samples.
                    :type nsamples: int

                    :param n_add: Number of samples to generate with each iteration.
                    :type n_add: int
        """
        self.nsamples = nsamples
        self.nexist = self.sample_object.samples.shape[0]
        if n_add is not None:
            self.n_add = n_add
        if self.nsamples <= self.nexist:
            raise NotImplementedError('UQpy Error: The number of requested samples must be larger than the existing '
                                      'sample set.')
        if self.run_model_object is not None:
            self.run_gerss()
        else:
            self.run_rss()

    ###################################################
    # Run Gradient-Enhanced Refined Stratified Sampling
    ###################################################
    def run_gerss(self):
        # --------------------------
        # RECTANGULAR STRATIFICATION
        # --------------------------
        if self.cell == 'Rectangular':

            if self.verbose:
                print('UQpy: Performing GE-RSS with rectangular stratification...')

            # Initialize the training points for the surrogate model
            self.training_points = self.sample_object.samplesU01

            # Initialize the vector of gradients at each training point
            dy_dx = np.zeros((self.nsamples, np.size(self.training_points[1])))

            # Primary loop for adding samples and performing refinement.
            for i in range(self.nexist, self.nsamples, self.n_add):
                p = min(self.n_add, self.nsamples - i)  # Number of points to add in this iteration

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
                    dy_dx[:i] = self.estimate_gradient(np.atleast_2d(self.training_points),
                                                       np.atleast_2d(np.array(qoi)),
                                                       self.sample_object.strata.origins +
                                                       0.5 * self.sample_object.strata.widths)

                # Use only max_train_size points to train the surrogate model (more economical option)
                else:
                    # Find the nearest neighbors to the most recently added point
                    from sklearn.neighbors import NearestNeighbors
                    knn = NearestNeighbors(n_neighbors=self.max_train_size)
                    knn.fit(np.atleast_2d(self.training_points))
                    neighbors = knn.kneighbors(np.atleast_2d(self.training_points[-1]), return_distance=False)

                    # Recompute the gradient only at the nearest neighbor points.
                    dy_dx[neighbors] = self.estimate_gradient(np.squeeze(self.training_points[neighbors]),
                                                              np.array(qoi)[neighbors][0],
                                                              np.squeeze(self.sample_object.strata.origins[neighbors] +
                                                                         0.5 * self.sample_object.strata.widths[
                                                                             neighbors]))

                # Define the gradient vector for application of the Delta Method
                dy_dx1 = dy_dx[:i]

                # ------------------------------
                # Determine the stratum to break
                # ------------------------------

                # Estimate the variance within each stratum by assuming a uniform distribution over the stratum.
                # All input variables are independent
                var = (1 / 12) * self.sample_object.strata.widths ** 2

                # Estimate the variance over the stratum by Delta Method
                s = np.zeros([i])
                for j in range(i):
                    s[j] = np.sum(dy_dx1[j, :] * var[j, :] * dy_dx1[j, :]) * self.sample_object.strata.weights[j] ** 2

                # Break the 'p' stratum with the maximum weight
                bin2break, p_left = np.array([]), p
                while np.where(s == s.max())[0].shape[0] < p_left:
                    t = np.where(s == s.max())[0]
                    bin2break = np.hstack([bin2break, t])
                    s[t] = 0
                    p_left -= t.shape[0]
                bin2break = np.hstack(
                    [bin2break, np.random.choice(np.where(s == s.max())[0], p_left, replace=False)])
                bin2break = list(map(int, bin2break))

                new_point = np.zeros([p, self.dimension])
                for j in range(p):
                    # Cut the stratum in the direction of maximum gradient
                    cut_dir_temp = self.sample_object.strata.widths[bin2break[j], :]
                    t = np.argwhere(cut_dir_temp == np.amax(cut_dir_temp))
                    dir2break = t[np.argmax(abs(dy_dx1[bin2break[j], t]))]

                    # Divide the stratum bin2break in the direction dir2break
                    self.sample_object.strata.widths[bin2break[j], dir2break] = \
                        self.sample_object.strata.widths[bin2break[j], dir2break] / 2
                    self.sample_object.strata.widths = np.vstack([self.sample_object.strata.widths,
                                                                  self.sample_object.strata.widths[bin2break[j], :]])
                    self.sample_object.strata.origins = np.vstack([self.sample_object.strata.origins,
                                                                   self.sample_object.strata.origins[bin2break[j], :]])
                    if self.sample_object.samplesU01[bin2break[j], dir2break] < \
                            self.sample_object.strata.origins[-1, dir2break] + \
                            self.sample_object.strata.widths[bin2break[j], dir2break]:
                        self.sample_object.strata.origins[-1, dir2break] = \
                            self.sample_object.strata.origins[-1, dir2break] + \
                            self.sample_object.strata.widths[bin2break[j], dir2break]
                    else:
                        self.sample_object.strata.origins[bin2break[j], dir2break] = \
                            self.sample_object.strata.origins[bin2break[j], dir2break] + \
                            self.sample_object.strata.widths[bin2break[j], dir2break]

                    self.sample_object.strata.weights[bin2break[j]] = self.sample_object.strata.weights[bin2break[j]]/2
                    self.sample_object.strata.weights = np.append(self.sample_object.strata.weights,
                                                                  self.sample_object.strata.weights[bin2break[j]])

                    # Add a uniform random sample inside the new stratum
                    new_point[j, :] = np.random.uniform(self.sample_object.strata.origins[i+j, :],
                                                        self.sample_object.strata.origins[i+j, :] +
                                                        self.sample_object.strata.widths[i+j, :])

                # Adding new sample to training points, samplesU01 and samples attributes
                self.training_points = np.vstack([self.training_points, new_point])
                self.sample_object.samplesU01 = np.vstack([self.sample_object.samplesU01, new_point])
                for j in range(0, self.dimension):
                    i_cdf = self.sample_object.distribution[j].icdf
                    new_point[:, j] = i_cdf(np.atleast_2d(new_point[:, j]).T, self.sample_object.dist_params[j])
                self.sample_object.samples = np.vstack([self.sample_object.samples, new_point])

                # Run the model at the new sample point
                self.run_model_object.run(samples=np.atleast_2d(new_point))

                if self.verbose:
                    print("Iteration:", i)

        # ----------------------
        # VORONOI STRATIFICATION
        # ----------------------
        elif self.cell == 'Voronoi':

            from UQpy.Utilities import compute_Delaunay_centroid_volume, voronoi_unit_hypercube
            from scipy.spatial.qhull import Delaunay
            import math
            import itertools

            self.training_points = self.sample_object.samplesU01

            # Extract the boundary vertices and use them in the Delaunay triangulation / mesh generation
            self.mesh_vertices = self.training_points
            self.points_to_samplesU01 = np.arange(0, self.training_points.shape[0])
            for i in range(np.shape(self.sample_object.strata.vertices)[0]):
                if any(np.logical_and(self.sample_object.strata.vertices[i, :] >= -1e-10,
                                      self.sample_object.strata.vertices[i, :] <= 1e-10)) or \
                    any(np.logical_and(self.sample_object.strata.vertices[i, :] >= 1-1e-10,
                                       self.sample_object.strata.vertices[i, :] <= 1+1e-10)):
                    self.mesh_vertices = np.vstack([self.mesh_vertices, self.sample_object.strata.vertices[i, :]])
                    self.points_to_samplesU01 = np.hstack([np.array([-1]), self.points_to_samplesU01])

            # Define the simplex mesh to be used for gradient estimation and sampling
            self.mesh = Delaunay(self.mesh_vertices, furthest_site=False, incremental=True, qhull_options=None)

            # Defining attributes of Delaunay, so that pycharm can check that it exists
            self.mesh.nsimplex: int = self.mesh.nsimplex
            self.mesh.vertices: np.ndarray = self.mesh.vertices
            self.mesh.simplices: np.ndarray = self.mesh.simplices
            self.mesh.add_points: classmethod = self.mesh.add_points
            points = getattr(self.mesh, 'points')
            dy_dx_old = 0

            # Primary loop for adding samples and performing refinement.
            for i in range(self.nexist, self.nsamples, self.n_add):
                p = min(self.n_add, self.nsamples - i)  # Number of points to add in this iteration

                # Compute the centroids and the volumes of each simplex cell in the mesh
                self.mesh.centroids = np.zeros([self.mesh.nsimplex, self.dimension])
                self.mesh.volumes = np.zeros([self.mesh.nsimplex, 1])
                for j in range(self.mesh.nsimplex):
                    self.mesh.centroids[j, :], self.mesh.volumes[j] = \
                        compute_Delaunay_centroid_volume(points[self.mesh.vertices[j]])

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
                    dy_dx = self.estimate_gradient(np.atleast_2d(self.training_points), np.atleast_2d(np.array(qoi)),
                                                   self.mesh.centroids)

                # Use only max_train_size points to train the surrogate model (more economical option)
                else:

                    # Build a mapping from the new vertex indices to the old vertex indices.
                    self.mesh.new_vertices, self.mesh.new_indices = [], []
                    self.mesh.new_to_old = np.zeros([self.mesh.vertices.shape[0], ]) * np.nan
                    j, k = 0, 0
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
                    dy_dx = np.zeros((self.mesh.new_to_old.shape[0], self.dimension))

                    # For those simplices that will not be updated, use the previous gradient
                    for j in range(dy_dx.shape[0]):
                        if np.isnan(self.mesh.new_to_old[j]):
                            continue
                        else:
                            dy_dx[j, :] = dy_dx_old[int(self.mesh.new_to_old[j]), :]

                    # For those simplices that will be updated, compute the new gradient
                    dy_dx[update_array, :] = self.estimate_gradient(
                        np.squeeze(self.sample_object.samplesU01[neighbors]),
                        np.atleast_2d(np.array(qoi)[neighbors]),
                        self.mesh.centroids[update_array])

                # ----------------------------------------------------
                # Determine the simplex to break and draw a new sample
                # ----------------------------------------------------

                # Estimate the variance over each simplex by Delta Method. Moments of the simplices are computed using
                # Eq. (19) from the following reference:
                # Good, I.J. and Gaskins, R.A. (1971). The Centroid Method of Numerical Integration. Numerische
                #       Mathematik. 16: 343--359.
                var = np.zeros((self.mesh.nsimplex, self.dimension))
                s = np.zeros(self.mesh.nsimplex)
                for j in range(self.mesh.nsimplex):
                    for k in range(self.dimension):
                        std = np.std(points[self.mesh.vertices[j]][:, k])
                        var[j, k] = (self.mesh.volumes[j] * math.factorial(self.dimension) /
                                     math.factorial(self.dimension + 2)) * (self.dimension * std ** 2)
                    s[j] = np.sum(dy_dx[j, :] * var[j, :] * dy_dx[j, :]) * (self.mesh.volumes[j] ** 2)
                dy_dx_old = dy_dx

                # Identify the stratum with the maximum weight
                bin2add, p_left = np.array([]), p
                while np.where(s == s.max())[0].shape[0] < p_left:
                    t = np.where(s == s.max())[0]
                    bin2add = np.hstack([bin2add, t])
                    s[t] = 0
                    p_left -= t.shape[0]
                bin2add = np.hstack([bin2add, np.random.choice(np.where(s == s.max())[0], p_left, replace=False)])

                # Create 'p' sub-simplex within the simplex with maximum variance
                new_point = np.zeros([p, self.dimension])
                for j in range(p):
                    # Create a sub-simplex within the simplex with maximum variance.
                    tmp_vertices = points[self.mesh.simplices[int(bin2add[j]), :]]
                    col_one = np.array(list(itertools.combinations(np.arange(self.dimension + 1), self.dimension)))
                    self.mesh.sub_simplex = np.zeros_like(tmp_vertices)  # node: an array containing mid-point of edges
                    for m in range(self.dimension + 1):
                        self.mesh.sub_simplex[m, :] = np.sum(tmp_vertices[col_one[m] - 1, :], 0) / self.dimension

                    # Using the Simplex class to generate a new sample in the sub-simplex
                    new_point[j, :] = Simplex(nodes=self.mesh.sub_simplex, nsamples=1).samples

                # Update the matrices to have recognize the new point
                self.points_to_samplesU01 = np.hstack([self.points_to_samplesU01, np.arange(i, i+p)])
                self.mesh.old_vertices = self.mesh.vertices

                # Update the Delaunay triangulation mesh to include the new point.
                self.mesh.add_points(new_point)
                points = getattr(self.mesh, 'points')

                # Update the sample arrays to include the new point
                self.sample_object.samplesU01 = np.vstack([self.sample_object.samplesU01, new_point])
                self.training_points = np.vstack([self.training_points, new_point])
                self.mesh_vertices = np.vstack([self.mesh_vertices, new_point])

                # Identify the new point in the parameter space and update the sample array to include the new point.
                for j in range(self.dimension):
                    new_point[:, j] = self.sample_object.distribution[j].icdf(np.atleast_2d(new_point[:, j]).T,
                                                                              self.sample_object.dist_params[j])
                self.sample_object.samples = np.vstack([self.sample_object.samples, new_point])

                # Run the mode at the new point.
                self.run_model_object.run(samples=new_point)

                # Compute the strata weights.
                self.sample_object.strata = voronoi_unit_hypercube(self.sample_object.samplesU01)

                self.sample_object.strata.centroids = []
                self.sample_object.strata.weights = []
                for region in self.sample_object.strata.bounded_regions:
                    vertices = self.sample_object.strata.vertices[region + [region[0]], :]
                    centroid, volume = compute_Voronoi_centroid_volume(vertices)
                    self.sample_object.strata.centroids.append(centroid[0, :])
                    self.sample_object.strata.weights.append(volume)

                if self.verbose:
                    print("Iteration:", i)

    #################################
    # Run Refined Stratified Sampling
    #################################
    def run_rss(self):
        # --------------------------
        # RECTANGULAR STRATIFICATION
        # --------------------------
        if self.cell == 'Rectangular':

            if self.verbose:
                print('UQpy: Performing RSS with rectangular stratification...')

            # Initialize the training points for the surrogate model
            self.training_points = self.sample_object.samplesU01

            # Primary loop for adding samples and performing refinement.
            for i in range(self.nexist, self.nsamples, self.n_add):
                p = min(self.n_add, self.nsamples - i)  # Number of points to add in this iteration
                # ------------------------------
                # Determine the stratum to break
                # ------------------------------
                # Estimate the weight corresponding to each stratum
                s = np.zeros(i)
                for j in range(i):
                    s[j] = self.sample_object.strata.weights[j] ** 2

                # Break the 'p' stratum with the maximum weight
                bin2break, p_left = np.array([]), p
                while np.where(s == s.max())[0].shape[0] < p_left:
                    t = np.where(s == s.max())[0]
                    bin2break = np.hstack([bin2break, t])
                    s[t] = 0
                    p_left -= t.shape[0]
                bin2break = np.hstack([bin2break, np.random.choice(np.where(s == s.max())[0], p_left, replace=False)])
                bin2break = list(map(int, bin2break))

                new_point = np.zeros([p, self.dimension])
                for j in range(p):
                    # Cut the stratum in the direction of maximum length
                    cut_dir_temp = self.sample_object.strata.widths[bin2break[j], :]
                    dir2break = np.random.choice(np.argwhere(cut_dir_temp == np.amax(cut_dir_temp))[0])

                    # Divide the stratum bin2break in the direction dir2break
                    self.sample_object.strata.widths[bin2break[j], dir2break] = \
                        self.sample_object.strata.widths[bin2break[j], dir2break] / 2
                    self.sample_object.strata.widths = np.vstack([self.sample_object.strata.widths,
                                                                  self.sample_object.strata.widths[bin2break[j], :]])
                    self.sample_object.strata.origins = np.vstack([self.sample_object.strata.origins,
                                                                   self.sample_object.strata.origins[bin2break[j], :]])
                    if self.sample_object.samplesU01[bin2break[j], dir2break] < \
                            self.sample_object.strata.origins[-1, dir2break] + \
                            self.sample_object.strata.widths[bin2break[j], dir2break]:
                        self.sample_object.strata.origins[-1, dir2break] = \
                            self.sample_object.strata.origins[-1, dir2break] + \
                            self.sample_object.strata.widths[bin2break[j], dir2break]
                    else:
                        self.sample_object.strata.origins[bin2break[j], dir2break] = \
                            self.sample_object.strata.origins[bin2break[j], dir2break] + \
                            self.sample_object.strata.widths[bin2break[j], dir2break]

                    self.sample_object.strata.weights[bin2break[j]] = self.sample_object.strata.weights[bin2break[j]]/2
                    self.sample_object.strata.weights = np.append(self.sample_object.strata.weights,
                                                                  self.sample_object.strata.weights[bin2break[j]])

                    # Add a uniform random sample inside the new stratum
                    new_point[j, :] = np.random.uniform(self.sample_object.strata.origins[i+j, :],
                                                        self.sample_object.strata.origins[i+j, :] +
                                                        self.sample_object.strata.widths[i+j, :])

                # Adding new sample to training points, samplesU01 and samples attributes
                self.training_points = np.vstack([self.training_points, new_point])
                self.sample_object.samplesU01 = np.vstack([self.sample_object.samplesU01, new_point])
                for k in range(self.dimension):
                    i_cdf = self.sample_object.distribution[k].icdf
                    new_point[:, k] = i_cdf(np.atleast_2d(new_point[:, k]).T, self.sample_object.dist_params[k])
                self.sample_object.samples = np.vstack([self.sample_object.samples, new_point])

                if self.verbose:
                    print("Iteration:", i)

        # ----------------------
        # VORONOI STRATIFICATION
        # ----------------------
        elif self.cell == 'Voronoi':

            from UQpy.Utilities import compute_Delaunay_centroid_volume, voronoi_unit_hypercube
            from scipy.spatial.qhull import Delaunay
            import math
            import itertools

            self.training_points = self.sample_object.samplesU01

            # Extract the boundary vertices and use them in the Delaunay triangulation / mesh generation
            self.mesh_vertices = self.training_points
            self.points_to_samplesU01 = np.arange(0, self.training_points.shape[0])
            for i in range(np.shape(self.sample_object.strata.vertices)[0]):
                if any(np.logical_and(self.sample_object.strata.vertices[i, :] >= -1e-10,
                                      self.sample_object.strata.vertices[i, :] <= 1e-10)) or \
                        any(np.logical_and(self.sample_object.strata.vertices[i, :] >= 1 - 1e-10,
                                           self.sample_object.strata.vertices[i, :] <= 1 + 1e-10)):
                    self.mesh_vertices = np.vstack(
                        [self.mesh_vertices, self.sample_object.strata.vertices[i, :]])
                    self.points_to_samplesU01 = np.hstack([np.array([-1]), self.points_to_samplesU01, ])

            # Define the simplex mesh to be used for sampling
            self.mesh = Delaunay(self.mesh_vertices, furthest_site=False, incremental=True, qhull_options=None)

            # Defining attributes of Delaunay, so that pycharm can check that it exists
            self.mesh.nsimplex: int = self.mesh.nsimplex
            self.mesh.vertices: np.ndarray = self.mesh.vertices
            self.mesh.simplices: np.ndarray = self.mesh.simplices
            self.mesh.add_points: classmethod = self.mesh.add_points
            points = getattr(self.mesh, 'points')

            # Primary loop for adding samples and performing refinement.
            for i in range(self.nexist, self.nsamples, self.n_add):
                p = min(self.n_add, self.nsamples - i)  # Number of points to add in this iteration

                # Compute the centroids and the volumes of each simplex cell in the mesh
                self.mesh.centroids = np.zeros([self.mesh.nsimplex, self.dimension])
                self.mesh.volumes = np.zeros([self.mesh.nsimplex, 1])
                for j in range(self.mesh.nsimplex):
                    self.mesh.centroids[j, :], self.mesh.volumes[j] = \
                        compute_Delaunay_centroid_volume(points[self.mesh.vertices[j]])

                # ----------------------------------------------------
                # Determine the simplex to break and draw a new sample
                # ----------------------------------------------------
                s = np.zeros(self.mesh.nsimplex)
                for j in range(self.mesh.nsimplex):
                    s[j] = self.mesh.volumes[j] ** 2

                # Identify the stratum with the maximum weight
                bin2add, p_left = np.array([]), p
                while np.where(s == s.max())[0].shape[0] < p_left:
                    t = np.where(s == s.max())[0]
                    bin2add = np.hstack([bin2add, t])
                    s[t] = 0
                    p_left -= t.shape[0]
                bin2add = np.hstack([bin2add, np.random.choice(np.where(s == s.max())[0], p_left, replace=False)])

                # Create 'p' sub-simplex within the simplex with maximum weight
                new_point = np.zeros([p, self.dimension])
                for j in range(p):
                    tmp_vertices = points[self.mesh.simplices[int(bin2add[j]), :]]
                    col_one = np.array(list(itertools.combinations(np.arange(self.dimension + 1), self.dimension)))
                    self.mesh.sub_simplex = np.zeros_like(
                        tmp_vertices)  # node: an array containing mid-point of edges
                    for m in range(self.dimension + 1):
                        self.mesh.sub_simplex[m, :] = np.sum(tmp_vertices[col_one[m] - 1, :], 0) / self.dimension

                    # Using the Simplex class to generate a new sample in the sub-simplex
                    new_point[j, :] = Simplex(nodes=self.mesh.sub_simplex, nsamples=1).samples

                # Update the matrices to have recognize the new point
                self.points_to_samplesU01 = np.hstack([self.points_to_samplesU01, np.arange(i, i+p)])
                self.mesh.old_vertices = self.mesh.vertices

                # Update the Delaunay triangulation mesh to include the new point.
                self.mesh.add_points(new_point)
                points = getattr(self.mesh, 'points')

                # Update the sample arrays to include the new point
                self.sample_object.samplesU01 = np.vstack([self.sample_object.samplesU01, new_point])
                self.training_points = np.vstack([self.training_points, new_point])
                self.mesh_vertices = np.vstack([self.mesh_vertices, new_point])

                # Identify the new point in the parameter space and update the sample array to include the new point.
                for j in range(self.dimension):
                    new_point[:, j] = self.sample_object.distribution[j].icdf(np.atleast_2d(new_point[:, j]).T,
                                                                              self.sample_object.dist_params[j])
                self.sample_object.samples = np.vstack([self.sample_object.samples, new_point])

                # Compute the strata weights.
                self.sample_object.strata = voronoi_unit_hypercube(self.sample_object.samplesU01)

                self.sample_object.strata.centroids = []
                self.sample_object.strata.weights = []
                for region in self.sample_object.strata.bounded_regions:
                    vertices = self.sample_object.strata.vertices[region + [region[0]], :]
                    centroid, volume = compute_Voronoi_centroid_volume(vertices)
                    self.sample_object.strata.centroids.append(centroid[0, :])
                    self.sample_object.strata.weights.append(volume)

                if self.verbose:
                    print("Iteration:", i)

    # Support functions for RSS and GE-RSS

    # Code for estimating gradients with a metamodel (surrogate)
    # TODO: We may want to consider moving this to Utilities.
    def estimate_gradient(self, x, y, xt):
        from UQpy.Reliability import TaylorSeries
        if type(self.krig_object).__name__ == 'Krig':
            self.krig_object.fit(samples=x, values=y)
            tck = self.krig_object
        elif type(self.krig_object).__name__ == 'GaussianProcessRegressor':
            self.krig_object.fit(x, y)
            tck = self.krig_object.predict
        else:
            from scipy.interpolate import LinearNDInterpolator

            # TODO: Here we need to add a reflection of the sample points over each face of the hypercube and build the
            #       linear interpolator from the reflected points.
            tck = LinearNDInterpolator(x, y, fill_value=0).__call__

        gr = TaylorSeries.gradient(samples=xt, model=tck, dimension=self.dimension, order='first',
                                   df_step=self.step_size, scale=False)
        return gr

    # Initialization and preliminary error checks.
    def init_rss(self):
        if type(self.sample_object).__name__ not in ['STS', 'RSS']:
            raise NotImplementedError("UQpy Error: sample_object must be an object of the STS or RSS class.")

        if self.run_model_object is not None:
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
            :param run_model_object: A RunModel object, which is used to evaluate the function value
            :type run_model_object: class

            :param samples: A 2d-array of samples
            :type samples: ndarray

            :param krig_object: A kriging class object
            :type krig_object: class

            :param population: Sample which are used as learning set by AKMCS class.
            :type population: ndarray

            :param nlearn: Number of sample generated using MCS, which are used as learning set by AKMCS. Only required
                           if population is not defined.
            :type nlearn: int

            :param nstart: Number of initial samples generated using LHS. Only required if sample_object is not defined.
            :type nstart: int

            :param dist_name: A list containing the names of the distributions of the random variables. This is only
                              required if sample_object is not defined.
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

            :param lf: Learning function used as selection criteria to identify the new samples.
                       Options: U, Weighted-U, EFF, EIF and EGIF
            :type lf: str/function

            :param n_add: Number of samples to be selected per iteration.
            :type n_add: int

            :param min_cov: Minimum Covariance used as the stopping criteria of AKMCS method in case of reliability
                            analysis.
            :type min_cov: float

            :param max_p: Maximum possible value of probability density function of samples. Only required with
                          'Weighted-U' learning function.
            :type max_p: float

            :param save_pf: Indicator to estimate probability of failure after each iteration. Only required if
                            user-defined learning function is used.
            :type save_pf: boolean

            :param verbose: A boolean declaring whether to write text to the terminal.
            :type verbose: bool

        Output:
            :return: AKMCS.sample_object.samples: Final/expanded samples.
            :rtype: AKMCS..sample_object.samples: ndarray

            :return: AKMCS.krig_model: Prediction function for the final surrogate model.
            :rtype: AKMCS.krig_model: function

            :return: AKMCS.pf: Probability of failure after every iteration of AKMCS. Available as an output only for
                               Reliability Analysis.
            :rtype: AKMCS.pf: float list

            :return: AKMCS.cov_pf: Covariance of probability of failure after every iteration of AKMCS. Available as an
                                   output only for Reliability Analysis.
            :rtype: AKMCS.pf: float list
    """

    # Authors: Mohit S. Chauhan
    # Last modified: 01/07/2020 by Mohit S. Chauhan

    def __init__(self, run_model_object=None, samples=None, krig_object=None, nlearn=10000, nstart=None,
                 population=None, dist_name=None, dist_params=None, qoi_name=None, lf='U', n_add=1,
                 min_cov=0.05, max_p=None, verbose=False, kriging='UQpy', save_pf=None):

        # Initialize the internal variables of the class.
        self.run_model_object = run_model_object
        self.samples = np.array(samples)
        self.krig_object = krig_object
        self.nlearn = nlearn
        self.nstart = nstart
        self.verbose = verbose
        self.qoi_name = qoi_name

        self.lf = lf
        self.min_cov = min_cov
        self.max_p = max_p
        self.dist_name = dist_name
        self.dist_params = dist_params
        self.nsamples = []

        self.moments = None
        self.distribution = None
        self.training_points = None
        self.n_add = n_add
        self.indicator = False
        self.pf = []
        self.cov_pf = []
        self.population = population
        self.kriging = kriging
        self.save_pf = save_pf
        self.dimension = 0
        self.qoi = None
        self.krig_model = None

        # Initialize and run preliminary error checks.
        self.init_akmcs()

        # Run AKMCS
        self.run_akmcs()

    def run_akmcs(self):

        # Check if the initial sample design already exists and has model evaluations with it.
        # If it does not, run the initial calculations.
        if self.samples is None:
            if self.verbose:
                print('UQpy: AKMCS - Generating the initial sample set using Latin hypercube sampling.')
            self.samples = LHS(dist_name=self.dist_name, dist_params=self.dist_params, nsamples=self.nstart).samples

        if self.verbose:
            print('UQpy: AKMCS - Running the initial sample set using RunModel.')

        self.run_model_object.run(samples=self.samples)

    def sample(self, samples=None, n_add=1, append_samples=True, nsamples=0, lf=None):
        """
        Description:

        Inputs:
            :param samples: A 2d-array of samples
            :type samples: ndarray

            :param n_add: Number of samples to be selected per iteration.
            :type n_add: int

            :param append_samples: If 'True', new samples are append to existing samples in sample_object. Otherwise,
                                   existing samples are discarded.
            :type append_samples: boolean

            :param nsamples: Number of samples to generate. No Default Value: nsamples must be prescribed.
            :type nsamples: int

            :param lf: Learning function used as selection criteria to identify the new samples. Only required, if
                       samples are generated using multiple criterion
                       Options: U, Weighted-U, EFF, EIF and EGIF
            :type lf: str/function

        """

        if self.kriging != 'UQpy':
            from sklearn.gaussian_process import GaussianProcessRegressor

        self.nsamples = nsamples
        if n_add is not None:
            self.n_add = n_add
        if lf is not None:
            self.lf = lf
            self.learning()

        if samples is not None:
            # New samples are appended to existing samples, if append_samples is TRUE
            if append_samples:
                self.samples = np.vstack([self.samples, samples])
            else:
                self.samples = samples
                self.run_model_object.qoi_list = []

            if self.verbose:
                print('UQpy: AKMCS - Running the provided sample set using RunModel.')

            self.run_model_object.run(samples=samples, append_samples=append_samples)

        if self.verbose:
            print('UQpy: Performing AK-MCS design...')

        # Initialize the population of samples at which to evaluate the learning function and from which to draw in the
        # sampling.
        if self.population is None:
            self.population = MCS(dist_name=self.dist_name, dist_params=self.dist_params,
                                  nsamples=self.nlearn)

        # If the quantity of interest is a dictionary, convert it to a list
        self.qoi = [None] * len(self.run_model_object.qoi_list)
        if type(self.run_model_object.qoi_list[0]) is dict:
            for j in range(len(self.run_model_object.qoi_list)):
                self.qoi[j] = self.run_model_object.qoi_list[j][self.qoi_name]
        else:
            self.qoi = self.run_model_object.qoi_list

        # Train the initial Kriging model.
        if self.kriging == 'UQpy':
            with suppress_stdout():  # disable printing output comments
                self.krig_object.fit(samples=self.samples, values=np.atleast_2d(np.array(self.qoi)))
            self.krig_model = self.krig_object.interpolate
        else:
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor(kernel=self.krig_object, n_restarts_optimizer=0)
            gp.fit(self.training_points, self.qoi)
            self.krig_model = gp.predict

        # ---------------------------------------------
        # Primary loop for learning and adding samples.
        # ---------------------------------------------

        for i in range(self.samples.shape[0], self.nsamples):
            # Find all of the points in the population that have not already been integrated into the training set
            rest_pop = np.array([x for x in self.population.samples.tolist() if x not in self.samples.tolist()])

            # Apply the learning function to identify the new point to run the model.

            new_ind = self.lf(self.krig_model, rest_pop)
            new_point = np.atleast_2d(rest_pop[new_ind])

            # Add the new points to the training set and to the sample set.
            self.samples = np.vstack([self.samples, new_point])

            # Run the model at the new points
            self.run_model_object.run(samples=np.atleast_2d(new_point))

            # If the quantity of interest is a dictionary, convert it to a list
            self.qoi = [None] * len(self.run_model_object.qoi_list)
            if type(self.run_model_object.qoi_list[0]) is dict:
                for j in range(len(self.run_model_object.qoi_list)):
                    self.qoi[j] = self.run_model_object.qoi_list[j][self.qoi_name]
            else:
                self.qoi = self.run_model_object.qoi_list

            # Retrain the Kriging surrogate model
            if self.kriging == 'UQpy':
                with suppress_stdout():
                    # disable printing output comments
                    self.krig_object.fit(samples=self.samples, values=np.atleast_2d(np.array(self.qoi)))
                self.krig_model = self.krig_object.interpolate
            else:
                from sklearn.gaussian_process import GaussianProcessRegressor
                gp = GaussianProcessRegressor(kernel=self.krig_object, n_restarts_optimizer=0)
                gp.fit(self.training_points, self.qoi)
                self.krig_model = gp.predict

            if self.verbose:
                print("Iteration:", i)

            if self.save_pf:
                if self.kriging == 'UQpy':
                    g = self.krig_model(rest_pop)
                else:
                    g = self.krig_model(rest_pop, return_std=False)

                n_ = g.shape[0] + len(self.qoi)
                pf = (sum(g < 0) + sum(np.array(self.qoi) < 0)) / n_
                self.pf.append(pf)
                self.cov_pf.append(np.sqrt((1 - pf) / (pf * n_)))

        if self.verbose:
            print('UQpy: AKMCS complete')

    # ------------------
    # LEARNING FUNCTIONS
    # ------------------
    def eigf(self, surr, pop):
        # Expected Improvement for Global Fit (EIGF)
        # Reference: J.N Fuhg, "Adaptive surrogate models for parametric studies", Master's Thesis
        # Link: https://arxiv.org/pdf/1905.05345.pdf
        if self.kriging == 'UQpy':
            g, sig = surr(pop, dy=True)
            sig = np.sqrt(sig)
        else:
            g, sig = surr(pop, return_std=True)
            sig = sig.reshape(sig.size, 1)
        sig[sig == 0.] = 0.00001

        # Evaluation of the learning function
        # First, find the nearest neighbor in the training set for each point in the population.
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(np.atleast_2d(self.training_points))
        neighbors = knn.kneighbors(np.atleast_2d(pop), return_distance=False)

        # noinspection PyTypeChecker
        qoi_array = np.array([self.qoi[x] for x in np.squeeze(neighbors)])

        # Compute the learning function at every point in the population.
        u = np.square(np.squeeze(g) - qoi_array) + np.square(np.squeeze(sig))

        rows = np.argmax(u)
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
        sig[sig == 0.] = 0.00001

        u = abs(g) / sig
        rows = u[:, 0].argsort()[:self.n_add]

        if min(u[:, 0]) >= 2:
            self.indicator = True

        return rows

    # This learning function has not yet been tested.
    def weighted_u(self, surr, pop):
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
        p1, p2 = np.ones([pop.shape[0], pop.shape[1]]), np.ones([pop.shape[0], pop.shape[1]])
        for j in range(self.dimension):
            p2[:, j] = self.population.distribution[j].icdf(np.atleast_2d(pop[:, j]).T, self.dist_params[j])
            p1[:, j] = self.population.distribution[j].pdf(np.atleast_2d(p2[:, j]).T, self.dist_params[j])

        p1 = p1.prod(1).reshape(u.size, 1)
        u_ = u * ((self.max_p - p1) / self.max_p)
        # u_ = u * p1/max(p1)
        rows = u_[:, 0].argsort()[:self.n_add]

        if min(u[:, 0]) >= 2:
            self.indicator = True

        return rows

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
            g = g.reshape(g.size, 1)
            sig = sig.reshape(sig.size, 1)
        sig[sig == 0.] = 0.00001
        # Reliability threshold: a_ = 0
        # EGRA method: epshilon = 2*sigma(x)
        a_, ep = 0, 2 * sig
        t1 = (a_ - g) / sig
        t2 = (a_ - ep - g) / sig
        t3 = (a_ + ep - g) / sig
        eff = (g - a_) * (2 * stats.norm.cdf(t1) - stats.norm.cdf(t2) - stats.norm.cdf(t3))
        eff += -sig * (2 * stats.norm.pdf(t1) - stats.norm.pdf(t2) - stats.norm.pdf(t3))
        eff += ep * (stats.norm.cdf(t3) - stats.norm.cdf(t2))
        rows = eff[:, 0].argsort()[-self.n_add:]

        if max(eff[:, 0]) <= 0.001:
            self.indicator = True

        n_ = g.shape[0] + len(self.qoi)
        pf = (np.sum(g < 0) + sum(iin < 0 for iin in self.qoi)) / n_
        self.pf.append(pf)
        self.cov_pf.append(np.sqrt((1 - pf) / (pf * n_)))

        return rows

    # This learning function has not yet been tested.
    def eif(self, surr, pop):
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
        fm = min(self.qoi)
        u = (fm - g) * stats.norm.cdf((fm - g) / sig) + sig * stats.norm.pdf((fm - g) / sig)
        rows = u[:, 0].argsort()[(np.size(g) - self.n_add):]

        return rows

    def learning(self):
        if type(self.lf).__name__ == 'function':
            self.lf = self.lf
        elif self.lf not in ['EFF', 'U', 'Weighted-U', 'EIF', 'EIGF']:
            raise NotImplementedError("UQpy Error: The provided learning function is not recognized.")
        elif self.lf == 'EIGF':
            self.lf = self.eigf
        elif self.lf == 'EIF':
            self.lf = self.eif
        elif self.lf == 'U':
            self.lf = self.u
        elif self.lf == 'Weighted-U':
            self.lf = self.weighted_u
        else:
            self.lf = self.eff

    # Initial check for errors
    def init_akmcs(self):
        if self.run_model_object is None:
            raise NotImplementedError('UQpy: AKMCS requires a predefined RunModel object.')

        if self.samples is not None:
            self.dimension = np.shape(self.samples)[1]
        else:
            self.dimension = np.shape(self.dist_name)[0]

        if self.save_pf is None:
            if self.lf not in ['EFF', 'U', 'Weighted-U']:
                self.save_pf = False
            else:
                self.save_pf = True

        self.learning()

########################################################################################################################
########################################################################################################################
#                                         Class Markov Chain Monte Carlo
########################################################################################################################


class MCMC_old:
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
        n_accepts, accept_ratio = 0, 0

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


class MCMC:
    """
        Description:
            Generate samples from arbitrary user-specified probability density function using Markov Chain Monte Carlo.
            Supported algorithms at this time are:
            - Metropolis-Hastings(MH),
            - Modified Metropolis-Hastings (MMH),
            - Affine Invariant Ensemble Sampler with stretch moves (Stretch),
            - DEMC,
            - Delayed Rejection Adaptive Metropolis (DRAM)
            References:
            S.-K. Au and J. L. Beck,“Estimation of small failure probabilities in high dimensions by subset simulation,”
                Probabilistic Eng. Mech., vol. 16, no. 4, pp. 263–277, Oct. 2001.
            J. Goodman and J. Weare, “Ensemble samplers with affine invariance,” Commun. Appl. Math. Comput. Sci.,vol.5,
                no. 1, pp. 65–80, 2010.
            R.C. Smith, "Uncertainty Quantification - Theory, Implementation and Applications", CS&E, 2014
        Input:
            :param dimension: A scalar value defining the dimension of target density function. Default: 1
            :type dimension: int

            :param pdf_target: Target density function from which to draw random samples
            :type pdf_target: (list of) callables

            :param log_pdf_target: Alternative way to define the target pdf, see above.
            :type log_pdf_target: (list of) callables

            :param args_target: Parameters of the target pdf copula (used when calling log_pdf method).
            :type args_target: tuple

            :param algorithm:  Algorithm used to generate random samples.
                            Options:
                                'MH': Metropolis Hastings Algorithm
                                'MMH': Component-wise Modified Metropolis Hastings Algorithm
                                'Stretch': Affine Invariant Ensemble MCMC with stretch moves
                                'DEMC': Affine Invariant Ensemble MCMC with stretch moves
                                'DRAM': Delayed Rejection Adaptive Metropolis
                            Default: 'MH'
            :type algorithm: str

            :param nsamples: Number of samples to generate
            :type nsamples: int

            :param nsamples_per_chain: Number of samples to generate
            :type nsamples_per_chain: int

            :param jump: Number of samples between accepted states of the Markov chain.
                                Default value: 1 (Accepts every state)
            :type: jump: int

            :param nburn: Length of burn-in. Number of samples at the beginning of the chain to discard.
                            This option is only used for the 'MMH' and 'MH' algorithms.
                            Default: nburn = 0
            :type nburn: int

            :param seed: Seed of the Markov chain(s)
                            Default: zeros(1 x dimension) - will raise an error for some algorithms for which the seed
                            must be specified
            :type seed: numpy array of dimension (nchains, dimension)

            :param **algorithm_inputs: Inputs that are algorithm specific - see user manual for a detailed list
            :type **algorithm_inputs: dictionary

            :param save_log_pdf: boolean that indicates whether to save log_pdf_values along with the samples
            :type save_log_pdf: bool, default False

            :param concat_chains_: boolean that indicates whether to concatenate the chains after a run,
                    if True: self.samples will be of size (nchains * nsamples, dimension)
                    if False: self.samples will be of size (nsamples, nchains, dimension)
            :type concat_chains_: bool, default True
        Output:
            :return: MCMC.samples: Set of MCMC samples following the target distribution
            :rtype: MCMC.samples: ndarray

            :return: MCMC.log_pdf_values: Values of
            :rtype: MCMC.log_pdf_values: ndarray

            :return: MCMC.acceptance_rate: Acceptance ratio of the MCMC samples
            :rtype: MCMC.acceptance_rate: float

    """

    # Authors: Audrey Olivier, Michael D. Shields, Mohit Chauhan, Dimitris G. Giovanis
    # Updated: 04/08/2019 by Audrey Olivier

    def __init__(self, dimension=1, pdf_target=None, log_pdf_target=None, args_target=None,
                 algorithm='MH', seed=None, nsamples=None, nsamples_per_chain=None, nburn=0, jump=1,
                 save_log_pdf=False, verbose=False, concat_chains_=True, **algorithm_inputs):

        if not (isinstance(dimension, int) and dimension >= 1):
            raise TypeError('dimension should be an integer >= 1')
        if not (isinstance(nburn, int) and nburn >= 0):
            raise TypeError('nburn should be an integer >= 0')
        if not (isinstance(jump, int) and jump >= 1):
            raise TypeError('jump should be an integer >= 1')
        self.dimension, self.nburn, self.jump = dimension, nburn, jump
        self.seed = self.preprocess_seed(seed, dim=self.dimension)    # check type and assign default [0., ... 0.]
        self.nchains = self.seed.shape[0]

        ##### ADDED MDS 1/21/20
        self.log_pdf_target = log_pdf_target
        self.pdf_target = pdf_target
        self.args_target = args_target

        # Check target pdf
        self.evaluate_log_target, self.evaluate_log_target_marginals = self.preprocess_target(
            pdf=pdf_target, log_pdf=log_pdf_target, args=args_target)
        self.save_log_pdf = save_log_pdf
        self.concat_chains_ = concat_chains_
        self.verbose = verbose
        self.algorithm = algorithm
        self.algorithm_inputs = algorithm_inputs

        # Do algorithm dependent initialization
        if algorithm.lower() == 'mh':
            self.init_mh()
        elif algorithm.lower() == 'mmh':
            self.init_mmh()
        elif algorithm.lower() == 'stretch':
            self.init_stretch()
        elif algorithm.lower() == 'dram':
            self.init_dram()
        elif algorithm.lower() == 'dream':
            self.init_dream()
        else:
            raise NotImplementedError('MCMC algorithms currently supported in UQpy are: MH, MMH, Stretch, DEMC, DRAM.')

        # Initialize a few more variables
        self.samples = None
        self.log_pdf_values = None
        self.acceptance_rate = [0.] * self.nchains

        if self.verbose:
            print('Initialization of mcmc algorithm ' + self.algorithm + ' completed.')

        # If nsamples is provided, run the algorithm
        if (nsamples is not None) or (nsamples_per_chain is not None):
            self.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

    def run(self, nsamples=None, nsamples_per_chain=None):
        """ Run MCMC algorithm. If run was called before, new samples are appended to self.samples, otherwise
        self.samples is created from scratch. """

        # Compute nsamples from nsamples_per_chain or vice-versa
        nsamples, nsamples_per_chain = self.preprocess_nsamples(nchains=self.nchains, nsamples=nsamples,
                                                                nsamples_per_chain=nsamples_per_chain)
        # Initialize the runs: allocate space for the new samples and log pdf values
        nsims, current_state = self.initialize_samples(nsamples_per_chain=nsamples_per_chain)

        if self.verbose:
            print('Running MCMC...')
        # Run nsims iterations of the MCMC algorithm, starting at current_state
        if self.algorithm.lower() == 'mh':
            self.run_mh(nsims, current_state)
        elif self.algorithm.lower() == 'mmh':
            self.run_mmh(nsims, current_state)
        elif self.algorithm.lower() == 'stretch':
            self.run_stretch(nsims, current_state)
        elif self.algorithm.lower() == 'dram':
            self.run_dram(nsims, current_state)
        elif self.algorithm.lower() == 'dream':
            self.run_dream(nsims, current_state)
        else:
            warnings.warn('This algorithm is not (yet!) supported.')
        if self.verbose:
            print('MCMC run successfully !')

        # Concatenate chains maybe
        if self.concat_chains_:
            self.concatenate_chains()

    ####################################################################################################################
    # Functions for MH algorithm: init_mh and run_mh
    def init_mh(self):
        """ Check MH algorithm inputs """

        # MH algorithm inputs: proposal and proposal_params
        names = ['proposal', 'proposal_params', 'proposal_is_symmetric']

        # print Warning if certain inputs are not supposed to be here
        for key in self.algorithm_inputs.keys():
            if key not in names:
                print('!!! Warning !!! Input '+key+' not used in MH algorithm - used inputs are ' + ', '.join(names))

        # Assign a default: gaussian with zero mean and unit variance in all directions
        if 'proposal' not in self.algorithm_inputs.keys():
            self.algorithm_inputs['proposal'] = Distribution(dist_name=['normal'] * self.dimension,
                                                             params=[[0., 1.]] * self.dimension)
            self.algorithm_inputs['proposal_is_symmetric'] = True

        # If the proposal is provided, check it (Distribution object, has rvs and log pdf or pdf methods, update params)
        else:
            proposal = self.algorithm_inputs['proposal']
            proposal_params = None
            if 'proposal_params' in self.algorithm_inputs.keys():
                proposal_params = self.algorithm_inputs['proposal_params']
            proposal = self.check_methods_proposal(proposal, proposal_params)
            self.algorithm_inputs['proposal'] = proposal
            #del self.algorithm_inputs['proposal_params']

        # check the symmetry of proposal, assign False as default
        if 'proposal_is_symmetric' not in self.algorithm_inputs.keys():
            self.algorithm_inputs['proposal_is_symmetric'] = False

    def run_mh(self, nsims, current_state):
        """ Run ns_per_chain * jump + nburn iterations  """
        current_log_pdf = self.evaluate_log_target(current_state)

        # Loop over the samples
        for iter_nb in range(nsims):

            # Sample candidate
            candidate = current_state + self.algorithm_inputs['proposal'].rvs(nsamples=self.nchains)

            # Compute log_pdf_target of candidate sample
            log_p_candidate = self.evaluate_log_target(candidate)

            # Compute acceptance ratio
            if self.algorithm_inputs['proposal_is_symmetric']:    # proposal is symmetric
                log_ratios = log_p_candidate - current_log_pdf
            else:    # If the proposal is non-symmetric, one needs to account for it in computing acceptance ratio
                log_proposal_ratio = self.algorithm_inputs['proposal'].log_pdf(candidate - current_state) - \
                                     self.algorithm_inputs['proposal'].log_pdf(current_state - candidate)
                log_ratios = log_p_candidate - current_log_pdf - log_proposal_ratio

            # Compare candidate with current sample and decide or not to keep the candidate (loop over nc chains)
            accept_vec = np.zeros((self.nchains, ))    # this vector will be used to compute accept_ratio of each chain
            for nc, (cand, log_p_cand, r_) in enumerate(zip(candidate, log_p_candidate, log_ratios)):
                accept = np.log(np.random.random()) < r_
                if accept:
                    current_state[nc, :] = cand
                    current_log_pdf[nc] = log_p_cand
                    accept_vec[nc] = 1.

            # Save the current state if needed, update acceptance rate
            self.update_samples(current_state, current_log_pdf)
            # Update the acceptance rate
            self.update_acceptance_rate(accept_vec)
            # update the total number of iterations
            self.total_iterations += 1

    ####################################################################################################################
    # Functions for MMH algorithm: init_mmh and iterations_mmh
    def init_mmh(self):
        """ Perform some checks and initialize the MMH algorithm """

        # Algorithms inputs are pdf_target_type, proposal_type and proposal_scale.
        used_inputs = ['proposal', 'proposal_params', 'proposal_is_symmetric']
        for key in self.algorithm_inputs.keys():
            if key not in used_inputs:
                warnings.warn('Input ' + key + ' not used in MMH algorithm - used inputs are: '+', '.join(used_inputs))

        # If proposal is not provided: set it as a list of standard gaussians
        if 'proposal' not in self.algorithm_inputs.keys():
            self.algorithm_inputs['proposal'] = [Distribution('normal', params=[0., 1.])] * self.dimension
            self.algorithm_inputs['proposal_is_symmetric'] = [True] * self.dimension

        # Proposal is provided, check it
        else:
            proposal = self.algorithm_inputs['proposal']
            if not isinstance(proposal, list):  # only one Distribution is provided, check it and transform it to a list
                proposal_params = None
                if 'proposal_params' in self.algorithm_inputs.keys():
                    proposal_params = self.algorithm_inputs['proposal_params']
                proposal = self.check_methods_proposal(proposal, proposal_params)
                self.algorithm_inputs['proposal'] = [proposal] * self.dimension
            else:    # a list of proposals is provided
                if len(proposal) != self.dimension:
                    raise ValueError('proposal given as a list should be of length dimension')
                proposal_params = [None] * self.dimension
                if 'proposal_params' in self.algorithm_inputs.keys():
                    proposal_params = self.algorithm_inputs['proposal_params']
                    if not (isinstance(proposal_params, list) and len(proposal_params) == self.dimension):
                        raise TypeError('MMH: proposal_params should be a list of same length as proposal')
                marginal_proposals = [self.check_methods_proposal(p, p_params)
                                      for (p, p_params) in zip(proposal, proposal_params)]
                self.algorithm_inputs['proposal'] = marginal_proposals

        # check the symmetry of proposal, assign False as default
        if 'proposal_is_symmetric' not in self.algorithm_inputs.keys():
            self.algorithm_inputs['proposal_is_symmetric'] = [False] * self.dimension
        else:
            b = self.algorithm_inputs['proposal_is_symmetric']
            if isinstance(b, bool):
                self.algorithm_inputs['proposal_is_symmetric'] = [b] * self.dimension
            elif isinstance(b, list) and all(isinstance(b_, bool) for b_ in b):
                pass
            else:
                raise TypeError('MMH: proposal_is_symmetric should be a (list of) boolean(s)')

    def run_mmh(self, nsims, current_state):
        # Loop over the samples

        # The target pdf is provided via its marginals
        if self.evaluate_log_target_marginals is not None:
            # Evaluate the current log_pdf
            current_log_p_marginals = [self.evaluate_log_target_marginals[j](current_state[:, j, np.newaxis])
                                       for j in range(self.dimension)]
            for iter_nb in range(nsims):
                # Sample candidate (independently in each dimension)
                accept_vec = np.zeros((self.nchains, ))
                for j in range(self.dimension):
                    candidate_j = current_state[:, j, np.newaxis] + self.algorithm_inputs['proposal'][j].rvs(
                        nsamples=self.nchains)

                    # Compute log_pdf_target of candidate sample
                    log_p_candidate_j = self.evaluate_log_target_marginals[j](candidate_j)

                    # Compute acceptance ratio
                    if self.algorithm_inputs['proposal_is_symmetric'][j]:  # proposal is symmetric
                        log_ratios = log_p_candidate_j - current_log_p_marginals[j]
                    else:  # If the proposal is non-symmetric, one needs to account for it in computing acceptance ratio
                        log_prop_j = self.algorithm_inputs['proposal'][j].log_pdf
                        log_proposal_ratio = log_prop_j(candidate_j - current_state[:, j, np.newaxis]) - \
                                             log_prop_j(current_state[:, j, np.newaxis] - candidate_j)
                        log_ratios = log_p_candidate_j - current_log_p_marginals[j] - log_proposal_ratio

                    # Compare candidate with current sample and decide or not to keep the candidate
                    for nc, (cand, log_p_cand, r_) in enumerate(zip(candidate_j, log_p_candidate_j, log_ratios)):
                        accept = np.log(np.random.random()) < r_
                        if accept:
                            current_state[nc, j] = cand
                            current_log_p_marginals[j][nc] = log_p_cand
                            accept_vec[nc] += 1. / self.dimension

                # Save the current state if needed, update acceptance rate
                self.update_samples(current_state, np.sum(np.array(current_log_p_marginals), axis=0))
                # Update the acceptance rate
                self.update_acceptance_rate(accept_vec)
                # update the total number of iterations
                self.total_iterations += 1

        # The target pdf is provided as a joint pdf
        else:
            current_log_pdf = self.evaluate_log_target(current_state)
            for iter_nb in range(nsims):

                accept_vec = np.zeros((self.nchains,))
                candidate = np.copy(current_state)
                for j in range(self.dimension):
                    candidate_j = current_state[:, j, np.newaxis] + self.algorithm_inputs['proposal'][j].rvs(
                        nsamples=self.nchains)
                    candidate[:, j] = candidate_j[:, 0]

                    # Compute log_pdf_target of candidate sample
                    log_p_candidate = self.evaluate_log_target(candidate)

                    # Compare candidate with current sample and decide or not to keep the candidate
                    if self.algorithm_inputs['proposal_is_symmetric'][j]:  # proposal is symmetric
                        log_ratios = log_p_candidate - current_log_pdf
                    else:  # If the proposal is non-symmetric, one needs to account for it in computing acceptance ratio
                        log_prop_j = self.algorithm_inputs['proposal'][j].log_pdf
                        log_proposal_ratio = log_prop_j(candidate_j - current_state[:, j, np.newaxis]) - \
                                             log_prop_j(current_state[:, j, np.newaxis] - candidate_j)
                        log_ratios = log_p_candidate - current_log_pdf - log_proposal_ratio
                    for nc, (cand, log_p_cand, r_) in enumerate(zip(candidate_j, log_p_candidate, log_ratios)):
                        accept = np.log(np.random.random()) < r_
                        if accept:
                            current_state[nc, j] = cand
                            current_log_pdf[nc] = log_p_cand
                            accept_vec[nc] += 1. / self.dimension
                        else:
                            candidate[:, j] = current_state[:, j]

                # Save the current state if needed, update acceptance rate
                self.update_samples(current_state, current_log_pdf)
                # Update the acceptance rate
                self.update_acceptance_rate(accept_vec)
                # update the total number of iterations
                self.total_iterations += 1
        return None

    ####################################################################################################################
    # Functions for Stretch algorithm: init_stretch and iterations_stretch
    def init_stretch(self):
        """ Perform some checks and initialize the Stretch algorithm """

        # Check nchains = ensemble size for the Stretch algorithm
        if self.nchains < 2:
            raise ValueError('For the Stretch algorithm, a seed must be provided with at least two samples.')

        # Check Stretch algorithm inputs: proposal_type and proposal_scale
        for key in self.algorithm_inputs.keys():
            if key not in ['scale']:  # remove inputs that are not being used
                print('!!! Warning !!! Input ' + key + ' not used in Stretch algorithm - used input is scale')
        if 'scale' not in self.algorithm_inputs.keys():
            self.algorithm_inputs['scale'] = 2.
        if not isinstance(self.algorithm_inputs['scale'], (float, int)):
            raise ValueError('For Stretch, algorithm input "scale" should be a float.')

    def run_stretch(self, nsims, current_state):
        # Evaluate the current log_pdf and initialize acceptance ratio
        current_log_pdf = self.evaluate_log_target(current_state)

        # Start the loop over nsamples - this code uses the parallel version of the stretch algorithm
        all_inds = np.arange(self.nchains)
        inds = all_inds % 2
        for iter_nb in range(nsims):

            accept_vec = np.zeros((self.nchains, ))
            # Separate the full ensemble into two sets, use one as a complementary ensemble to the other and vice-versa
            for split in range(2):
                S1 = (inds == split)

                # Get current and complementary sets
                sets = [current_state[inds == j, :] for j in range(2)]
                s, c = sets[split], sets[1 - split]  # current and complementary sets respectively
                Ns, Nc = len(s), len(c)

                # Sample new state for S1 based on S0 and vice versa
                zz = ((self.algorithm_inputs['scale'] - 1.) * np.random.rand(Ns, 1) + 1) ** 2. / \
                     self.algorithm_inputs['scale']  # sample Z
                factors = (self.dimension - 1.) * np.log(zz)  # compute log(Z ** (d - 1))
                rint = np.random.choice(Nc, size=(Ns,), replace=True)  # sample X_{j} from complementary set
                candidates = c[rint, :] - (c[rint, :] - s) * np.tile(zz, [1, self.dimension])  # new candidates

                # Compute new likelihood, can be done in parallel :)
                logp_candidates = self.evaluate_log_target(candidates)

                # Compute acceptance rate
                for j, f, lpc, candidate in zip(all_inds[S1], factors, logp_candidates, candidates):
                    accept = np.log(np.random.rand()) < f + lpc - current_log_pdf[j]
                    if accept:
                        current_state[j] = candidate
                        current_log_pdf[j] = lpc
                        accept_vec[j] += 1.

            # Save the current state if needed, update acceptance rate
            self.update_samples(current_state, current_log_pdf)
            # Update the acceptance rate
            self.update_acceptance_rate(accept_vec)
            # update the total number of iterations
            self.total_iterations += 1
        return None

    ####################################################################################################################
    # Functions from DRAM algorithm
    def init_dram(self):
        """ Perform some checks and initialize the DRAM algorithm """

        # The inputs to this algorithm are the initial_cov, k0, sp and gamma_2
        used_ins = ['initial_cov', 'k0', 'sp', 'gamma_2', 'save_cov']
        for key in self.algorithm_inputs.keys():
            if key not in used_ins:
                print('!!! Warning !!! Input ' + key + ' not used in DE-MC algorithm - used inputs are ' +
                      ', '.join(used_ins))
        # Check the initial covariance
        if 'initial_cov' not in self.algorithm_inputs:
            self.algorithm_inputs['initial_cov'] = np.eye(self.dimension)
        if not(isinstance(self.algorithm_inputs['initial_cov'], np.ndarray)
               and self.algorithm_inputs['initial_cov'].shape == (self.dimension, self.dimension)):
            raise TypeError('DRAM: initial_cov should be a 2D ndarray of shape (dimension, dimension)')

        # Check the other parameters
        keys = ['k0', 'sp', 'gamma_2', 'save_cov']
        defaults = [100, 2.38 ** 2 / self.dimension, 1. / 5., False]
        types = [int, (float, int), (float, int), bool]
        for (key, default_val, type_) in zip(keys, defaults, types):
            if key not in self.algorithm_inputs.keys():
                self.algorithm_inputs[key] = default_val
            elif not isinstance(self.algorithm_inputs[key], type_):
                raise TypeError('Wrong type for DRAM algo parameter ' + key)
        if self.algorithm_inputs['save_cov']:
            self.adaptive_covariance = [self.algorithm_inputs['initial_cov']]

    def run_dram(self, nsims, current_state):
        # Evaluate the current log_pdf and initialize acceptance ratio
        current_log_pdf = self.evaluate_log_target(current_state)

        # Initialize scale parameter
        sample_mean = np.zeros((self.dimension, ))
        sample_covariance = np.zeros((self.dimension, self.dimension))
        current_covariance = self.algorithm_inputs['initial_cov']
        mvp, mvp_DR = Distribution('mvnormal'), Distribution('mvnormal')

        # Loop over the samples
        for iter_nb in range(nsims):
            # compute the scale parameter

            # Sample candidate
            mvp.update_params(params=[np.zeros((self.dimension, )), current_covariance])
            candidate = current_state + mvp.rvs(nsamples=self.nchains)

            # Compute log_pdf_target of candidate sample
            log_p_candidate = self.evaluate_log_target(candidate)

            # Compare candidate with current sample and decide or not to keep the candidate (loop over nc chains)
            accept_vec = np.zeros((self.nchains, ))
            inds_DR = []   # indices of chains that will undergo delayed rejection
            for nc, (cand, log_p_cand, log_p_curr) in enumerate(zip(candidate, log_p_candidate, current_log_pdf)):
                accept = np.log(np.random.random()) < log_p_cand - log_p_curr
                if accept:
                    current_state[nc, :] = cand
                    current_log_pdf[nc] = log_p_cand
                    accept_vec[nc] += 1.
                else:    # enter delayed rejection
                    inds_DR.append(nc)    # these indices will enter the delayed rejection part

            if len(inds_DR) > 0:   # performed delayed rejection for some samples
                current_states_DR = np.array([current_state[nc, :] for nc in range(self.nchains) if nc in inds_DR])
                candidates_DR = np.array([candidate[nc, :] for nc in range(self.nchains) if nc in inds_DR])

                # Sample other candidates closer to the current one
                params_DR = [np.zeros((self.dimension, )),
                             self.algorithm_inputs['gamma_2'] ** 2 * current_covariance]
                mvp_DR.update_params(params=params_DR)
                candidate2 = current_states_DR + mvp_DR.rvs(nsamples=len(inds_DR))
                # Evaluate their log_target
                log_p_candidate2 = self.evaluate_log_target(candidate2)
                log_prop_cand_cand2 = mvp.log_pdf(candidates_DR - candidate2)
                log_prop_cand_curr = mvp.log_pdf(candidates_DR - current_states_DR)
                # Accept or reject
                for (nc, cand2, log_p_cand2, J1, J2) in zip(inds_DR, candidate2, log_p_candidate2, log_prop_cand_cand2,
                                                            log_prop_cand_curr):
                    alpha_cand_cand2 = min(1., np.exp(log_p_candidate[nc] - log_p_cand2))
                    alpha_cand_curr = min(1., np.exp(log_p_candidate[nc] - current_log_pdf[nc]))
                    log_alpha2 = log_p_cand2 - current_log_pdf[nc] + J1 - J2 + \
                                 np.log(max(1. - alpha_cand_cand2, 10 ** (-320))) \
                                 - np.log(max(1. - alpha_cand_curr, 10 ** (-320)))
                    accept = np.log(np.random.random()) < min(0., log_alpha2)
                    if accept:
                        current_state[nc, :] = cand2
                        current_log_pdf[nc] = log_p_cand2
                        accept_vec[nc] += 1.

            # Adaptive part: update the covariance
            for nc in range(self.nchains):
                # update covariance
                sample_mean, sample_covariance = recursive_update_mean_covariance(
                    n=self.total_iterations + 1, new_sample=current_state[nc, :], previous_mean=sample_mean,
                    previous_covariance=sample_covariance)
                if (self.total_iterations + 1) % self.algorithm_inputs['k0'] == 0:
                    current_covariance = self.algorithm_inputs['sp'] * sample_covariance + \
                                         1e-6 * np.eye(self.dimension)
                    if self.algorithm_inputs['save_cov']:
                        self.adaptive_covariance.append(current_covariance)

            # Save the current state if needed, update acceptance rate
            self.update_samples(current_state, current_log_pdf)
            # Update the acceptance rate
            self.update_acceptance_rate(accept_vec)
            # update the total number of iterations
            self.total_iterations += 1

    ####################################################################################################################
    # Functions for DREAM algorithm
    def init_dream(self):
        """ Perform some checks and initialize the DREAM algorithm """

        # Check nb of chains
        if self.nchains < 2:
            raise ValueError('For the DREAM algorithm, a seed must be provided with at least two samples.')

        # The algorithm inputs are: jump rate gamma - default is 3, c and c_star are parameters involved in the
        # differential evolution part of the algorithm - c_star should be small compared to the width of the target,
        # n_CR is the number of crossover probabilities - default 3, and p_g: prob(gamma=1) - default is 0.2
        # adapt_CR = (iter_max, rate) governs the adapation of crossover probabilities (default: no adaptation)
        # check_chains = (iter_max, rate) governs the discrading of outlier chains (default: no check on outlier chains)
        names = ['delta', 'c', 'c_star', 'n_CR', 'p_g', 'adapt_CR', 'check_chains']
        defaults = [3, 0.1, 1e-6, 3, 0.2, (-1, 1), (-1, 1)]
        types = [int, (float, int), (float, int), int, float, tuple, tuple]
        for key in self.algorithm_inputs.keys():
            if key not in names:
                print('!!! Warning !!! Input ' + key + ' not used in DREAM algorithm - used inputs are ' +
                      ', '.join(names))
        for key, default_value, typ in zip(names, defaults, types):
            if key not in self.algorithm_inputs.keys():
                self.algorithm_inputs[key] = default_value
            if not isinstance(self.algorithm_inputs[key], typ):
                raise TypeError('Wrong type for input ' + key)
        if self.algorithm_inputs['n_CR'] > self.dimension:
            self.algorithm_inputs['n_CR'] = self.dimension
        for key in ['adapt_CR', 'check_chains']:
            if len(self.algorithm_inputs[key])!=2 or (not all(isinstance(i, (int, float))
                                                              for i in self.algorithm_inputs[key])):
                raise TypeError('Inputs adapt_CR and check_chains should be tuples of 2 integers.')


    def run_dream(self, nsims, current_state):
        # Initialize some variables
        delta, c, c_star, n_CR, p_g = self.algorithm_inputs['delta'], self.algorithm_inputs['c'], \
                                      self.algorithm_inputs['c_star'], self.algorithm_inputs['n_CR'], \
                                      self.algorithm_inputs['p_g']
        adapt_CR = self.algorithm_inputs['adapt_CR']
        check_chains = self.algorithm_inputs['check_chains']
        J, n_id = np.zeros((n_CR,)), np.zeros((n_CR,))
        R = np.array([np.setdiff1d(np.arange(self.nchains), j) for j in range(self.nchains)])
        CR = np.arange(1, n_CR + 1) / n_CR
        pCR = np.ones((n_CR,)) / n_CR

        # Evaluate the current log_pdf and initialize acceptance ratio
        current_log_pdf = self.evaluate_log_target(current_state)

        # dynamic part: evolution of chains
        for iter_nb in range(nsims):

            draw = np.argsort(np.random.rand(self.nchains - 1, self.nchains), axis=0)
            dX = np.zeros_like(current_state)
            lmda = np.random.uniform(low=-c, high=c, size=(self.nchains,))
            std_x_tmp = np.std(current_state, axis=0)

            D = np.random.choice(delta, size=(self.nchains,), replace=True)
            as_ = [R[j, draw[slice(D[j]), j]] for j in range(self.nchains)]
            bs_ = [R[j, draw[slice(D[j], 2 * D[j], 1), j]] for j in range(self.nchains)]
            id = np.random.choice(n_CR, size=(self.nchains, ), replace=True, p=pCR)
            z = np.random.rand(self.nchains, self.dimension)
            A = [np.where(z_j < CR[id_j])[0] for (z_j, id_j) in zip(z, id)]  # subset A of selected dimensions
            d_star = np.array([len(A_j) for A_j in A])
            for j in range(self.nchains):
                if d_star[j] == 0:
                    A[j] = np.array([np.argmin(z[j])])
                    d_star[j] = 1
            gamma_d = 2.38 / np.sqrt(2 * (D + 1) * d_star)
            g = [np.random.choice([gamma_d[j], 1], size=1, replace=True, p=[1 - p_g, p_g]) for j in range(self.nchains)]
            for j in range(self.nchains):
                for i in A[j]:
                    dX[j, i] = c_star * np.random.randn() + \
                               (1 + lmda[j]) * g[j] * np.sum(current_state[as_[j], i] - current_state[bs_[j], i])
            candidates = current_state + dX

            # Evaluate log likelihood of candidates
            logp_candidates = self.evaluate_log_target(candidates)

            # Accept or reject
            accept_vec = np.zeros((self.nchains, ))
            for nc, (lpc, candidate, log_p_curr) in enumerate(zip(logp_candidates, candidates, current_log_pdf)):
                accept = np.log(np.random.random()) < lpc - log_p_curr
                if accept:
                    current_state[nc, :] = candidate
                    current_log_pdf[nc] = lpc
                    accept_vec[nc] = 1.
                else:
                    dX[nc, :] = 0
                J[id[nc]] = J[id[nc]] + np.sum((dX[nc, :] / std_x_tmp) ** 2)
                n_id[id[nc]] += 1

            # Save the current state if needed, update acceptance rate
            self.update_samples(current_state, current_log_pdf)
            # Update the acceptance rate
            self.update_acceptance_rate(accept_vec)
            # update the total number of iterations
            self.total_iterations += 1

            # update selection cross prob
            if self.total_iterations < adapt_CR[0] and self.total_iterations % adapt_CR[1] == 0:
                pCR = J / n_id
                pCR /= sum(pCR)
            # check outlier chains (only if you have saved at least 100 values already)
            if (self.current_sample_index * self.nchains >= 100) and \
                    (self.total_iterations < check_chains[0]) and (self.total_iterations % check_chains[1] == 0):
                self.check_outlier_chains(replace_with_best=True)
        return None

    def check_outlier_chains(self, replace_with_best=False):
        if not self.save_log_pdf:
            return ValueError('attribute save_log_pdf must be True in order to check outlier chains')
        start_ = self.current_sample_index // 2
        avgs_logpdf = np.mean(self.log_pdf_values[start_:], axis=0)
        best_ = np.argmax(avgs_logpdf)
        avg_sorted = np.sort(avgs_logpdf)
        ind1, ind3 = 1 + round(0.25 * self.nchains), 1 + round(0.75 * self.nchains)
        q1, q3 = avg_sorted[ind1], avg_sorted[ind3]
        qr = q3 - q1

        outlier_num = 0
        for j in range(self.nchains):
            if avgs_logpdf[j] < q1 - 2.0 * qr:
                outlier_num += 1
                if replace_with_best:
                    self.samples[start_:, j, :] = self.samples[start_:, best_, :]
                    self.log_pdf_values[start_:, j] = self.log_pdf_values[start_:, best_]
                else:
                    print('Chain {} is an outlier chain'.format(j))
        if self.verbose and outlier_num > 0:
            print('Detected {} outlier chains'.format(outlier_num))

    ####################################################################################################################
    # Helper functions that can be used by all algorithms
    # Methods update_samples, update_accept_ratio and sample_candidate_from_proposal can be called in the run stage.
    # Methods preprocess_target, preprocess_proposal, check_seed and check_integers can be called in the init stage.

    def concatenate_chains(self):
        # Concatenate chains so that samples go from (nsamples, nchains, dimension) to (nsamples * nchains, dimension)
        self.samples = self.samples.reshape((-1, self.dimension), order='C')
        if self.save_log_pdf:
            self.log_pdf_values = self.log_pdf_values.reshape((-1, ), order='C')
        return None

    def unconcatenate_chains(self):
        # Inverse of concatenate_chains method
        self.samples = self.samples.reshape((-1, self.nchains, self.dimension), order='C')
        if self.save_log_pdf:
            self.log_pdf_values = self.log_pdf_values.reshape((-1, self.nchains), order='C')
        return None

    def initialize_samples(self, nsamples_per_chain):
        """ Allocate space for samples and log likelihood values, initialize sample_index, acceptance ratio
        If some samples already exist, allocate space to append new samples to the old ones """
        if self.samples is None:    # very first call of run, set current_state as the seed and initialize self.samples
            self.samples = np.zeros((nsamples_per_chain, self.nchains, self.dimension))
            if self.save_log_pdf:
                self.log_pdf_values = np.zeros((nsamples_per_chain, self.nchains))
            current_state = np.zeros_like(self.seed)
            np.copyto(current_state, self.seed)
            if self.nburn == 0:    # save the seed
                self.samples[0, :, :] = current_state
                if self.save_log_pdf:
                    self.log_pdf_values[0, :] = self.evaluate_log_target(current_state)
                self.current_sample_index = 1
                self.total_iterations = 1  # total nb of iterations, grows if you call run several times
                nsims = self.jump * nsamples_per_chain - 1
            else:
                self.current_sample_index = 0
                self.total_iterations = 0  # total nb of iterations, grows if you call run several times
                nsims = self.nburn + self.jump * nsamples_per_chain

        else:    # fetch previous samples to start the new run, current state is last saved sample
            if len(self.samples.shape) == 2:   # the chains were previously concatenated
                self.unconcatenate_chains()
            current_state = self.samples[-1]
            self.samples = np.concatenate(
                [self.samples, np.zeros((nsamples_per_chain, self.nchains, self.dimension))], axis=0)
            if self.save_log_pdf:
                self.log_pdf_values = np.concatenate(
                    [self.log_pdf_values, np.zeros((nsamples_per_chain, self.nchains))], axis=0)
            nsims = self.jump * nsamples_per_chain
        return nsims, current_state

    def update_samples(self, current_state, current_log_pdf):
        # Update the chain, only if burn-in is over and the sample is not being jumped over
        if self.total_iterations >= self.nburn and (self.total_iterations-self.nburn) % self.jump == 0:
            self.samples[self.current_sample_index, :, :] = current_state
            if self.save_log_pdf:
                self.log_pdf_values[self.current_sample_index, :] = current_log_pdf
            self.current_sample_index += 1

    def update_acceptance_rate(self, new_accept=None):
        # Use an iterative function to update the acceptance rate
        self.acceptance_rate = [na / (self.total_iterations+1) + self.total_iterations / (self.total_iterations+1) * a
                                for (na, a) in zip(new_accept, self.acceptance_rate)]

    @staticmethod
    def preprocess_target(log_pdf, pdf, args):
        """ This function transforms the log_pdf, pdf, args inputs into a function that evaluates log_pdf_target(x)
        for a given x. """
        # log_pdf is provided
        if log_pdf is not None:
            if callable(log_pdf):
                if args is None:
                    args = ()
                evaluate_log_pdf = (lambda x: log_pdf(x, *args))
                evaluate_log_pdf_marginals = None
            elif isinstance(log_pdf, list) and (all(callable(p) for p in log_pdf)):

                if args is None:
                    args = [()] * len(log_pdf)
                if not (isinstance(args, list) and len(args) == len(log_pdf)):
                    raise ValueError('When log_pdf_target is a list, args should be a list (of tuples) of same length.')
                evaluate_log_pdf_marginals = list(map(lambda i: lambda x: log_pdf[i](x, *args[i]), range(len(log_pdf))))
                #evaluate_log_pdf_marginals = [partial(log_pdf_, *args_) for (log_pdf_, args_) in zip(log_pdf, args)]
                evaluate_log_pdf = (lambda x: np.sum(
                    [log_pdf[i](x[:, i, np.newaxis], *args[i]) for i in range(len(log_pdf))]))
            else:
                raise TypeError('log_pdf_target must be a callable or list of callables')
        # pdf is provided
        elif pdf is not None:
            if callable(pdf):
                if args is None:
                    args = ()
                evaluate_log_pdf = (lambda x: np.log(np.maximum(pdf(x, *args), 10 ** (-320) * np.ones((x.shape[0],)))))
                evaluate_log_pdf_marginals = None
            elif isinstance(pdf, (list, tuple)) and (all(callable(p) for p in pdf)):
                if args is None:
                    args = [()] * len(pdf)
                if not (isinstance(args, (list, tuple)) and len(args) == len(pdf)):
                    raise ValueError('When pdf_target is given as a list, args should also be a list of same length.')
                evaluate_log_pdf_marginals = list(
                    map(lambda i: lambda x: np.log(np.maximum(pdf[i](x, *args[i]),
                                                              10 ** (-320) * np.ones((x.shape[0],)))),
                        range(len(pdf))
                        ))
                evaluate_log_pdf = (lambda x: np.sum(
                    [np.log(np.maximum(pdf[i](x[:, i, np.newaxis], *args[i]), 10**(-320)*np.ones((x.shape[0],))))
                     for i in range(len(log_pdf))]))
                #evaluate_log_pdf = None
            else:
                raise TypeError('pdf_target must be a callable or list of callables')
        else:
            raise ValueError('log_pdf_target or pdf_target should be provided.')
        return evaluate_log_pdf, evaluate_log_pdf_marginals

    @staticmethod
    def preprocess_nsamples(nchains, nsamples=None, nsamples_per_chain=None):
        """ Compute nsamples_per_chain from nsamples and vice-versa """
        if ((nsamples is not None) and (nsamples_per_chain is not None)) or (
                nsamples is None and nsamples_per_chain is None):
            raise ValueError('Either nsamples or nsamples_per_chain must be provided (not both)')
        if nsamples is not None:
            if not (isinstance(nsamples, int) and nsamples >= 0):
                raise TypeError('nsamples must be an integer >= 0.')
            nsamples_per_chain = nsamples // nchains
        else:
            if not (isinstance(nsamples_per_chain, int) and nsamples_per_chain >= 0):
                raise TypeError('nsamples_per_chain must be an integer >= 0.')
            nsamples = nsamples_per_chain * nchains
        return nsamples, nsamples_per_chain

    @staticmethod
    def preprocess_seed(seed, dim):
        """ Check the dimension of seed, assign [0., 0., ..., 0.] if not provided. """
        if seed is None:
            seed = np.zeros((1, dim))
        else:
            try:
                seed = np.array(seed, dtype=float).reshape((-1, dim))
            except:
                raise TypeError('Input seed should be a nd array of dimensions (?, dimension).')
        return seed

    @staticmethod
    def check_methods_proposal(proposal, proposal_params=None):
        """ Check that the given proposal distribution has 1) a rvs method and 2) a log pdf or pdf method
        Used in the MH and MMH initializations"""
        if not isinstance(proposal, Distribution):
            raise TypeError('proposal should be a Distribution object')
        if proposal_params is not None:
            proposal.update_params(params=proposal_params)
        if not hasattr(proposal, 'rvs'):
            raise AttributeError('The proposal should have an rvs method')
        if not hasattr(proposal, 'log_pdf'):
            if not hasattr(proposal, 'pdf'):
                raise AttributeError('The proposal should have a log_pdf or pdf method')
            proposal.log_pdf = lambda x: np.log(np.maximum(proposal.pdf(x), 10 ** (-320) * np.ones((x.shape[0],))))
        return proposal


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

            :param proposal: proposal to sample from: this Distribution object must have an rvs method and a log_pdf (
                             or pdf) methods
            :type proposal: Distribution object

            :param proposal_params: parameters of the proposal distribution
            :type proposal_params: list

            :param log_pdf_target: callable that evaluates the target log pdf
            :type log_pdf_target: callable

            :param pdf_target: callable that evaluates the target pdf (log_pdf_target is preferred though)
            :type pdf_target: callable

            :param args_target: arguments of the target log_pdf (pdf) callable - i.e., log pdf target at x is evaluated
                                as log_pdf_target(x, *args)
            :type args_target: tuple

            :param nsamples: Number of samples to generate.
            :type nsamples: int

        Output:
            :return: IS.samples: Set of generated samples
            :rtype: IS.samples: ndarray (nsamples, dim)

            :return: IS.weights: Importance weights of samples (weighted so that they sum up to 1)
            :rtype: IS.weights: ndarray (nsamples, )

            :return: IS.unnormalized_log_weights: unnormalized log weights of samples
            :rtype: IS.unnormalized_log_weights: ndarray (nsamples, )
    """

    # Authors: Audrey Olivier, Dimitris G.Giovanis
    # Last Modified: 10/2019 by Audrey Olivier

    def __init__(self, nsamples=None, pdf_target=None, log_pdf_target=None, args_target=None,
                 proposal=None, proposal_params=None, verbose=False):

        self.verbose = verbose
        # Initialize proposal: it should have an rvs and log pdf or pdf method
        if not isinstance(proposal, Distribution):
            raise TypeError('The proposal should be of type Distribution.')
        if not hasattr(proposal, 'rvs'):
            raise AttributeError('The proposal should have an rvs method')
        if not hasattr(proposal, 'log_pdf'):
            if not hasattr(proposal, 'pdf'):
                raise AttributeError('The proposal should have a log_pdf or pdf method')
            proposal.log_pdf = lambda x: np.log(np.maximum(proposal.pdf(x), 10 ** (-320) * np.ones((x.shape[0],))))
        self.proposal = proposal
        self.proposal.update_params(params=proposal_params)

        # Initialize target
        self.evaluate_log_target = self.preprocess_target(log_pdf=log_pdf_target, pdf=pdf_target, args=args_target)

        # Initialize the samples and weights
        self.samples = None
        self.unnormalized_log_weights = None
        self.weights = None

        # Run IS if nsamples is provided
        if nsamples is not None and nsamples != 0:
            self.run(nsamples)

    def run(self, nsamples):
        """ Perform IS """

        if self.verbose:
            print('Running Importance Sampling')
        # Sample from proposal
        new_samples = self.proposal.rvs(nsamples=nsamples)
        # Compute un-scaled weights of new samples
        new_log_weights = self.evaluate_log_target(x=new_samples) - self.proposal.log_pdf(x=new_samples)

        # Save samples and weights (append to existing if necessary)
        if self.samples is None:
            self.samples = new_samples
            self.unnormalized_log_weights = new_log_weights
        else:
            self.samples = np.concatenate([self.samples, new_samples], axis=0)
            self.unnormalized_log_weights = np.concatenate([self.unnormalized_log_weights, new_log_weights], axis=0)

        # Take the exponential and normalize the weights
        weights = np.exp(self.unnormalized_log_weights - max(self.unnormalized_log_weights))
        # note: scaling with max avoids having NaN of Inf when taking the exp
        sum_w = np.sum(weights, axis=0)
        self.weights = weights / sum_w
        if self.verbose:
            print('Importance Sampling performed successfully')

    def resample(self, method='multinomial', size=None):
        """ Resample: create a set of un-weighted samples from a set of weighted samples """
        from .Utilities import resample
        return resample(self.samples, self.weights, method=method, size=size)

    @staticmethod
    def preprocess_target(log_pdf, pdf, args):
        """ This function transforms the log_pdf, pdf, args inputs into a function that evaluates log_pdf_target(x)
        for a given x. """
        # log_pdf is provided
        if log_pdf is not None:
            if callable(log_pdf):
                if args is None:
                    args = ()
                evaluate_log_pdf = (lambda x: log_pdf(x, *args))
            else:
                raise TypeError('log_pdf_target must be a callable')
        # pdf is provided
        elif pdf is not None:
            if callable(pdf):
                if args is None:
                    args = ()
                evaluate_log_pdf = (lambda x: np.log(np.maximum(pdf(x, *args), 10 ** (-320) * np.ones((x.shape[0],)))))
            else:
                raise TypeError('pdf_target must be a callable')
        else:
            raise ValueError('log_pdf_target or pdf_target should be provided.')
        return evaluate_log_pdf
