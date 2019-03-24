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

    def __init__(self, dimension=None, dist_name=None, dist_params=None, lhs_criterion='random', lhs_metric='euclidean',
                 lhs_iter=100, nsamples=None):

        self.dimension = dimension
        self.nsamples = nsamples
        self.dist_name = dist_name
        self.dist_params = dist_params
        self.lhs_criterion = lhs_criterion
        self.lhs_metric = lhs_metric
        self.lhs_iter = lhs_iter
        self.init_lhs()

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
            samples_u_to_x[:, j] = i_cdf(samples[:, j], self.distribution[j].params)

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
                 sts_criterion="random"):

        self.dimension = dimension
        self.sts_design = sts_design
        self.input_file = input_file
        self.dist_name = dist_name
        self.dist_params = dist_params
        self.strata = None
        self.sts_criterion = sts_criterion
        self.init_sts()

        self.distribution = [None] * self.dimension
        for i in range(self.dimension):
            self.distribution[i] = Distribution(self.dist_name[i])
        self.samplesU01, self.samples = self.run_sts()
        del self.dist_name

    def run_sts(self):
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

            :param min_train_size: Minimum size of training data around new sample used to update surrogate.
                                   Default: nsamples
            :type min_train_size: int

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

    def __init__(self, x=None, model=None, meta='Delaunay', cell='Rectangular', nsamples=None,
                 min_train_size=None, step_size=0.005, corr_model='Gaussian', reg_model='Quadratic',
                 corr_model_params=None, n_opt=None):

        self.x = x
        self.model = model
        self.option = 'Refined'
        self.meta = meta
        self.cell = cell
        self.nsamples = nsamples
        self.points = 0
        self.min_train_size = min_train_size
        self.step_size = step_size
        self.corr_model = corr_model
        self.corr_model_params = corr_model_params
        self.reg_model = reg_model
        self.n_opt = n_opt
        self.init_rss()
        self.samples = x.samples
        self.samplesU01 = x.samplesU01
        self.distribution = x.distribution
        self.dist_params = x.dist_params
        self.strata = x.strata
        self.values = self.run_rss()

    def run_rss(self):
        from UQpy.RunModel import RunModel
        from UQpy.Surrogates import Krig
        from sklearn.gaussian_process import GaussianProcessRegressor
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import Delaunay
        import numpy.matlib as matlib
        import itertools
        import math

        print('UQpy: Performing RSS design...')

        def cent_diff(f, x, h):
            dydx = np.zeros((np.size(x, 0), np.size(x, 1)))
            for dirr in range(np.size(x, 1)):
                temp = np.zeros((np.size(x, 0), np.size(x, 1)))
                temp[:, dirr] = np.ones(np.size(x, 0))
                low = x - h / 2 * temp
                hi = x + h / 2 * temp
                dydx[:, dirr] = ((f.__call__(hi) - f.__call__(low)) / h)[:, 0]
            return dydx

        def surrogate(x, y, corr_m_p, reg_m, corr_m, xt, n):
            if self.meta == 'Delaunay':
                tck = LinearNDInterpolator(x, y, fill_value=0)
                gr = cent_diff(tck, xt, self.step_size)
            elif self.meta == 'Kriging':
                with suppress_stdout():  # disable printing output comments
                    tck = Krig(samples=x, values=y, reg_model=reg_m, corr_model=corr_m, corr_model_params=corr_m_p,
                               n_opt=n)
                corr_m_p = tck.corr_model_params
                gr = cent_diff(tck.interpolate, xt, self.step_size)
                # gr = tck.jacobian(xt)
            elif self.meta == 'Kriging_Sklearn':
                gp = GaussianProcessRegressor(kernel=corr_m, n_restarts_optimizer=0)
                gp.fit(x, y)
                gr = cent_diff(gp.predict, xt, self.step_size)
            else:
                raise NotImplementedError("Exit code: Does not identify 'meta'.")
            return gr, corr_m_p

        def local(pt, x, mts, max_dim):
            # Identify the indices of 'mts' number of points in array 'x', which are closest to point 'pt'.
            ff = 0.2
            train = []
            while len(train) < mts:
                a = x > matlib.repmat(pt - ff * max_dim, x.shape[0], 1)
                b = x < matlib.repmat(pt + ff * max_dim, x.shape[0], 1)
                x_ind = a & b
                train = []
                for k_ in range(x.shape[0]):
                    if np.array_equal(x_ind[k_, :], np.ones(x.shape[1])):
                        train.append(k_)
                ff = ff + 0.1
            return train

        values, dydx1, tri = 0, 0, 0
        dimension = self.samples.shape[1]

        if self.cell == 'Voronoi':
            lst = np.array(list(itertools.product([0, 1], repeat=dimension)))
            self.points = np.vstack([lst, self.samplesU01])
            tri = Delaunay(self.points)
        else:
            if self.meta == 'Delaunay':
                lst = np.array(list(itertools.product([0, 1], repeat=dimension)))
                self.points = np.vstack([lst, self.samplesU01])
            else:
                self.points = self.samplesU01

        if self.option == 'Gradient':
            with suppress_stdout():  # disable printing output comments
                values = np.array(RunModel(self.points, model_script=self.model).qoi_list)
            if self.cell == 'Rectangular':
                dydx1, self.corr_model_params = surrogate(self.points, values, self.corr_model_params,
                                                          self.reg_model, self.corr_model,
                                                          self.strata.origins + 0.5 * self.strata.widths, self.n_opt)
            else:
                simplex = getattr(tri, 'simplices')
                dydx1, self.corr_model_params = surrogate(self.points, values, self.corr_model_params, self.reg_model,
                                                          self.corr_model, np.mean(tri.points[simplex], 1),
                                                          self.n_opt)

        initial_s = np.size(self.samplesU01, 0)
        for i in range(initial_s, self.nsamples):
            if self.cell == 'Rectangular':
                # Determine the stratum to break
                if self.option == 'Gradient':
                    # Estimate the variance within each stratum by assuming a uniform distribution over the stratum.
                    # All input variables are independent
                    var = (1 / 12) * self.strata.widths ** 2
                    # Estimate the variance over the stratum by Delta Method
                    s = np.zeros([i, 1])
                    for j in range(i):
                        s[j, 0] = np.sum(dydx1[j, :] * var[j, :] * dydx1[j, :] * (self.strata.weights[j] ** 2))
                    bin2break = np.argmax(s)
                else:
                    w = np.argwhere(self.strata.weights == np.amax(self.strata.weights))
                    bin2break = w[np.random.randint(len(w))]

                # Determine the largest dimension of the stratum and define this as the cut direction
                if self.option == 'Refined':
                    # Cut the stratum in a random direction
                    cut_dir_temp = self.strata.widths[bin2break, :]
                    t = np.argwhere(cut_dir_temp[0] == np.amax(cut_dir_temp[0]))
                    dir2break = t[np.random.randint(len(t))]
                else:
                    # Cut the stratum in the direction of maximum gradient
                    cut_dir_temp = self.strata.widths[bin2break, :]
                    t = np.argwhere(cut_dir_temp == np.amax(cut_dir_temp))
                    dir2break = t[np.argmax(abs(dydx1[bin2break, t]))]

                # Divide the stratum bin2break in the direction dir2break
                self.strata.widths[bin2break, dir2break] = self.strata.widths[bin2break, dir2break] / 2
                self.strata.widths = np.vstack([self.strata.widths, self.strata.widths[bin2break, :]])

                self.strata.origins = np.vstack([self.strata.origins, self.strata.origins[bin2break, :]])
                if self.samplesU01[bin2break, dir2break] < self.strata.origins[-1, dir2break] + \
                        self.strata.widths[bin2break, dir2break]:
                    self.strata.origins[-1, dir2break] = self.strata.origins[-1, dir2break] + self.strata.widths[
                        bin2break, dir2break]
                else:
                    self.strata.origins[bin2break, dir2break] = self.strata.origins[bin2break, dir2break] + \
                                                                self.strata.widths[bin2break, dir2break]

                self.strata.weights[bin2break] = self.strata.weights[bin2break] / 2
                self.strata.weights = np.append(self.strata.weights, self.strata.weights[bin2break])

                # Add an uniform random sample inside new stratum
                new = np.random.uniform(self.strata.origins[i, :], self.strata.origins[i, :] + self.strata.widths[i, :])
                # Adding new sample to points, samplesU01 and samples attributes
                self.points = np.vstack([self.points, new])
                self.samplesU01 = np.vstack([self.samplesU01, new])
                for j in range(0, dimension):
                    icdf = self.distribution[j].icdf
                    new[j] = icdf(new[j], self.dist_params[j])
                self.samples = np.vstack([self.samples, new])

            elif self.cell == 'Voronoi':
                simplex = getattr(tri, 'simplices')
                # Estimate the variance over the stratum by Delta Method
                weights = np.zeros(((np.size(simplex, 0)), 1))
                var = np.zeros((np.size(simplex, 0), dimension))
                s = np.zeros(((np.size(simplex, 0)), 1))
                for j in range((np.size(simplex, 0))):
                    # Define Simplex
                    sim = self.points[simplex[j, :]]
                    # Estimate the volume of simplex
                    v1 = np.concatenate((np.ones([np.size(sim, 0), 1]), sim), 1)
                    weights[j] = (1 / math.factorial(np.size(simplex[j, :]) - 1)) * np.linalg.det(v1)
                    if self.option == 'Gradient':
                        for k in range(dimension):
                            # Estimate standard deviation of points
                            from statistics import stdev
                            std = stdev(sim[:, k].tolist())
                            var[j, k] = (weights[j] * math.factorial(dimension) / math.factorial(dimension + 2)) * (
                                    dimension * std ** 2)
                        s[j, 0] = np.sum(dydx1[j, :] * var[j, :] * dydx1[j, :] * (weights[j] ** 2))

                if self.option == 'Refined':
                    w = np.argwhere(weights[:, 0] == np.amax(weights[:, 0]))
                    bin2add = w[0, np.random.randint(len(w))]
                else:
                    bin2add = np.argmax(s)

                # Creating sub-simplex
                tmp = self.points[simplex[bin2add, :]]
                col_one = np.array(list(itertools.combinations(np.arange(dimension + 1), dimension)))
                node = np.zeros_like(tmp)    # node: an array containing mid-point of edges
                for m in range(dimension + 1):
                    node[m, :] = np.sum(tmp[col_one[m] - 1, :], 0) / dimension

                # Using Simplex class to generate new sample
                new = Simplex(nodes=node, nsamples=1).samples
                # Adding new sample to points, samplesU01 and samples attributes
                self.points = np.vstack([self.points, new])
                self.samplesU01 = np.vstack([self.samplesU01, new])
                for j in range(0, dimension):
                    icdf = self.distribution[j].icdf
                    new[0, j] = icdf(new[0, j], self.dist_params[j])
                self.samples = np.vstack([self.samples, new])
                # Creating Delaunay triangulation from the new points
                tri = Delaunay(self.points)
            else:
                raise NotImplementedError("Exit code: Does not identify 'cell'.")

            if self.option == 'Gradient':

                with suppress_stdout():  # disable printing output comments
                    y_new = RunModel(np.atleast_2d(self.samples[i, :]), model_script=self.model).qoi_list
                values = np.vstack([values, y_new])

                if np.size(self.samples, 0) < self.min_train_size:
                    # Global surrogate updating: Update the surrogate model using all the points
                    if self.cell == 'Rectangular':
                        in_train = np.arange(self.points.shape[0])
                        in_update = np.arange(i)
                    else:
                        simplex = getattr(tri, 'simplices')
                        in_train = np.arange(self.points.shape[0])
                        in_update = np.arange(simplex.shape[0])
                else:
                    # Local surrogate updating: Update the surrogate model using min_train_size
                    if self.cell == 'Rectangular':
                        if self.meta == 'Delaunay':
                            in_train = local(self.samplesU01[i, :], self.points, self.min_train_size,
                                             np.amax(self.strata.widths))
                        else:
                            in_train = local(self.samplesU01[i, :], self.samplesU01, self.min_train_size,
                                             np.amax(self.strata.widths))
                        in_update = local(self.samplesU01[i, :], self.strata.origins + .5 * self.strata.widths,
                                          self.min_train_size / 2, np.amax(self.strata.widths))
                    else:
                        simplex = getattr(tri, 'simplices')
                        # in_train: Indices of samples used to update surrogate approximation
                        in_train = local(self.samplesU01[i, :], self.samplesU01, self.min_train_size,
                                         np.amax(np.sqrt(self.strata.weights)))
                        # in_update: Indices of centroid of simplex, where gradient is updated
                        in_update = local(self.samplesU01[i, :], np.mean(tri.points[simplex], 1),
                                          self.min_train_size / 2, np.amax(np.sqrt(self.strata.weights)))

                # Update the surrogate model & the store the updated gradients
                if self.cell == 'Rectangular':
                    dydx1 = np.vstack([dydx1, np.zeros(dimension)])
                    dydx1[in_update, :], self.corr_model_params = surrogate(self.points[in_train, :],
                                                                            values[in_train, :],
                                                                            self.corr_model_params, self.reg_model,
                                                                            self.corr_model,
                                                                            self.strata.origins[in_update, :] +
                                                                            .5 * self.strata.widths[in_update, :], 1)
                else:
                    simplex = getattr(tri, 'simplices')
                    dydx1 = np.vstack([dydx1, np.zeros([simplex.shape[0] - dydx1.shape[0], dimension])])
                    dydx1[in_update, :], self.corr_model_params = surrogate(self.points[in_train, :],
                                                                            values[in_train, :],
                                                                            self.corr_model_params,
                                                                            self.reg_model, self.corr_model,
                                                                            np.mean(tri.points[
                                                                                        simplex[in_update, :]],
                                                                                    1), 1)
        print('Done!')
        if self.option == 'Gradient':
            if self.cell == 'Rectangular':
                if self.meta != 'Delaunay':
                    return values
            else:
                return values[2 ** dimension:, :]

    def init_rss(self):
        if type(self.x).__name__ not in ['STS', 'RSS']:
            raise NotImplementedError("Exit code: x should be a class object from STS or RSS class.")

        if self.model is not None:
            self.option = 'Gradient'

        if self.meta is None:
            self.meta = 'Delaunay'
        elif self.meta not in ['Delaunay', 'Kriging', 'Kriging_Sklearn']:
            raise NotImplementedError("Exit code: Input 'meta' is not specified correctly.")

        if self.cell is None:
            self.cell = 'Rectangular'
        elif self.cell not in ['Rectangular', 'Voronoi']:
            raise NotImplementedError("Exit code: Input 'cell' is not specified correctly.")

        if type(self.nsamples).__name__ != 'int':
            raise NotImplementedError("Exit code: nsamples should be integer.")
        if self.nsamples <= self.x.samples.shape[0]:
            raise NotImplementedError("Exit code: Already have desired number of samples.")

        if self.min_train_size is None:
            self.min_train_size = self.nsamples


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
    # Updated: 4/26/18 by Michael D. Shields

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

        # Define a kwargs for the parameter arguments
        kwargs = {}
        if self.pdf_target_params is not None:
            kwargs['params'] = self.pdf_target_params
        if self.pdf_target_copula_params is not None:
            kwargs['copula_params'] = self.pdf_target_copula_params

        ################################################################################################################
        # Classical Metropolis-Hastings Algorithm with symmetric proposal density
        if self.algorithm == 'MH':
            samples[0, :] = self.seed
            log_pdf_ = partial(self.log_pdf_target, **kwargs)
            log_p_current = log_pdf_(samples[0, :])

            # Loop over the samples
            for i in range(self.nsamples * self.jump - 1 + self.nburn):
                if self.pdf_proposal_type[0] == 'Normal':
                    cholesky_cov = np.diag(self.pdf_proposal_scale)
                    z_normal = np.random.normal(size=(self.dimension, 1))
                    candidate = samples[i, :] + np.matmul(cholesky_cov, z_normal).reshape((self.dimension,))
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
                    candidate = samples[i, :] + np.random.uniform(low=low, high=high, size=self.dimension)
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

            samples[0, :] = self.seed[0:]

            if self.pdf_target_type == 'marginal_pdf':
                list_log_p_current = [self.pdf_target[j](samples[0, j], **kwargs)
                                      for j in range(self.dimension)]
                for i in range(self.nsamples * self.jump - 1 + self.nburn):
                    for j in range(self.dimension):

                        log_pdf_ = partial(self.log_pdf_target[j], **kwargs)

                        if self.pdf_proposal_type[j] == 'Normal':
                            candidate = np.random.normal(samples[i, j], self.pdf_proposal_scale[j], size=1)
                            log_p_candidate = log_pdf_(candidate)
                            log_p_current = list_log_p_current[j]
                            log_p_accept = log_p_candidate - log_p_current

                            accept = np.log(np.random.random()) < log_p_accept

                            if accept:
                                samples[i + 1, j] = candidate
                                list_log_p_current[j] = log_p_candidate
                                n_accepts += 1
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
                                n_accepts += 1
                            else:
                                samples[i + 1, j] = samples[i, j]
            else:
                log_pdf_ = partial(self.log_pdf_target, **kwargs)

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
            log_pdf_ = partial(self.log_pdf_target, **kwargs)
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
                    samples[i + 1, :] = candidate
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
        if self.seed is None:
            self.seed = np.zeros(self.dimension)
        if self.algorithm is not 'Stretch':
            if self.seed.__len__() != self.dimension:
                raise NotImplementedError("Exit code: Incompatible dimensions in 'seed'.")
        else:
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

        # Check MMH
        if self.algorithm is 'MMH':
            if (self.pdf_target_type == 'marginal_pdf') and (self.pdf_target_copula is not None):
                raise ValueError('UQpy error: MMH with pdf_target_type="marginal" cannot be used when the'
                                 'target pdf has a copula, use pdf_target_type="joint" instead')

        # Define a helper function
        def compute_log_pdf(x, pdf_func, params=None, copula_params=None):
            kwargs = {}
            if params is not None:
                kwargs['params'] = params
            if copula_params is not None:
                kwargs['copula_params'] = copula_params
            pdf_value = max(pdf_func(x, **kwargs), 10 ** (-320))
            return np.log(pdf_value)
        # Either pdf_target or log_pdf_target must be defined
        if (self.pdf_target is None) and (self.log_pdf_target is None):
            raise ValueError('The target distribution must be defined, using inputs'
                             ' log_pdf_target or pdf_target.')
        # For MMH with pdf_target_type == 'marginals', pdf_target or its log should be lists
        if (self.algorithm == 'MMH') and (self.pdf_target_type == 'marginal'):
            if self.log_pdf_target is not None:
                if not isinstance(self.log_pdf_target, list):
                    raise ValueError('For MMH algo with pdf_target_type="marginal", '
                                     'log_pdf_target should be a list')
                if isinstance(self.log_pdf_target[0], str):
                    p_js = [Distribution(dist_name=pdf_target_j) for pdf_target_j in self.pdf_target]
                    try:
                        [p_j.log_pdf(x=None, params=None) for p_j in p_js]
                    except TypeError:
                        self.log_pdf_target = [p_j.log_pdf for p_j in p_js]
                    except AttributeError:
                        raise AttributeError('log_pdf_target given as a list of strings must point to Distributions '
                                             'with an existing log_pdf method.')
                elif callable(self.log_pdf_target[0]):
                    pass
                else:
                    raise ValueError('log_pdf_target must be a list of strings or a list of callables')
            else:
                if not isinstance(self.pdf_target, list):
                    raise ValueError('For MMH algo with pdf_target_type="marginal", '
                                     'pdf_target should be a list')
                if isinstance(self.pdf_target[0], str):
                    p_js = [Distribution(dist_name=pdf_target_j) for pdf_target_j in self.pdf_target]
                    try:
                        [p_j.pdf(x=None, params=None) for p_j in p_js]
                    except TypeError:
                        self.log_pdf_target = [partial(compute_log_pdf, pdf_func=p_j.pdf) for p_j in p_js]
                    except AttributeError:
                        raise AttributeError('pdf_target given as a list of strings must point to Distributions '
                                             'with an existing pdf method.')
                elif callable(self.pdf_target[0]):
                    self.log_pdf_target = [partial(compute_log_pdf, pdf_func=pdf_target_j)
                                           for pdf_target_j in self.pdf_target]
                else:
                    raise ValueError('pdf_target must be a list of strings or a list of callables')
        else:
            if self.log_pdf_target is not None:
                if isinstance(self.log_pdf_target, str) or (isinstance(self.log_pdf_target, list)
                                                            and isinstance(self.log_pdf_target[0], str)):
                    p = Distribution(dist_name=self.log_pdf_target, copula=self.pdf_target_copula)
                    try:
                        p.log_pdf(x=None, params=None, copula_params=None)
                    except TypeError:
                        self.log_pdf_target = p.log_pdf
                    except AttributeError:
                        raise AttributeError('log_pdf_target given as a string must point to a Distribution '
                                             'with an existing log_pdf method.')
                elif callable(self.log_pdf_target):
                    pass
                else:
                    raise ValueError('For MH and Stretch, log_pdf_target must be a callable function, '
                                     'a str or list of str')
            else:
                if isinstance(self.pdf_target, str) or (isinstance(self.pdf_target, list)
                                                        and isinstance(self.pdf_target[0], str)):
                    p = Distribution(dist_name=self.pdf_target, copula=self.pdf_target_copula)
                    try:
                        p.pdf(x=None, params=None, copula_params=None)
                    except TypeError:
                        self.log_pdf_target = partial(compute_log_pdf, pdf_func=p.pdf)
                    except AttributeError:
                        raise AttributeError('pdf_target given as a string must point to a Distribution '
                                             'with an existing pdf method.')
                elif callable(self.pdf_target):
                    self.log_pdf_target = partial(compute_log_pdf, pdf_func=self.pdf_target)
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
    # Last Modified: 11/02/18 by Audrey Olivier

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
        kwargs = {}
        if self.pdf_target_params is not None:
            kwargs['params'] = self.pdf_target_params
        if self.pdf_target_copula_params is not None:
            kwargs['copula_params'] = self.pdf_target_copula_params
        log_ps = self.log_pdf_target(x, **kwargs)

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
        def compute_log_pdf(x, params, copula_params, pdf_func):
            kwargs = {}
            if params is not None:
                kwargs['params'] = params
            if copula_params is not None:
                kwargs['copula_params'] = copula_params
            tmp = pdf_func(x, **kwargs)
            pdf_value = np.fmax(tmp, 10 ** (-320)*np.ones_like(tmp))
            return np.log(pdf_value)
        # Check log_pdf_target, pdf_target
        if (self.pdf_target is None) and (self.log_pdf_target is None):
            raise ValueError('UQpy error: a target pdf must be defined (pdf_target or log_pdf_target).')
        # The code first checks if log_pdf_target is defined, if yes, no need to check pdf_target
        if self.log_pdf_target is not None:
            # log_pdf_target can be defined as a callable or string.
            if isinstance(self.log_pdf_target, str) or (isinstance(self.log_pdf_target, list) and
                                                        isinstance(self.log_pdf_target[0], str)):
                p = Distribution(dist_name=self.log_pdf_target, copula=self.pdf_target_copula)
                try:
                    p.log_pdf(x=None, params=None, copula_params=None)
                except TypeError:
                    self.log_pdf_target = p.log_pdf
                except AttributeError:
                    raise AttributeError('log_pdf_target given as a string must point to a Distribution '
                                         'with an existing log_pdf method.')
            elif callable(self.log_pdf_target):
                pass
            else:
                raise ValueError('log_pdf_target should be a callable or a string/list of strings.')
        else:
            # pdf_target can be a str of list of strings, then compute the log_pdf
            if isinstance(self.pdf_target, str) or (isinstance(self.pdf_target, list) and
                                                    isinstance(self.pdf_target[0], str)):
                p = Distribution(dist_name=self.pdf_target, copula=self.pdf_target_copula)
                try:
                    p.pdf(x=None, params=None, copula_params=None)
                except TypeError:
                    self.log_pdf_target = partial(compute_log_pdf, pdf_func=p.pdf)
                except AttributeError:
                    raise AttributeError('pdf_target given as a string must point to a Distribution '
                                         'with an existing pdf method.')
            # otherwise it may be a function that computes the pdf, then just take the logarithm
            elif callable(self.pdf_target):
                self.log_pdf_target = partial(compute_log_pdf, pdf_func=self.pdf_target)
            else:
                raise ValueError('pdf_target should be a callable or a string/list of strings.')

        # Check pdf_proposal_name
        if self.pdf_proposal is None:
            raise ValueError('Exit code: A proposal distribution is required.')
        # can be given as a name or a list of names, transform it to a distribution class
        if not isinstance(self.pdf_proposal, str) and not (isinstance(self.pdf_proposal, list)
           and isinstance(self.pdf_proposal[0], str)):
            raise ValueError('UQpy error: proposal pdf must be given as a str or a list of str')
