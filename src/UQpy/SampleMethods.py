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
import numpy as np
from scipy.spatial.distance import pdist
import scipy.stats as sp
import random
from UQpy.Distributions import *

########################################################################################################################
########################################################################################################################
#                                         Monte Carlo simulation
########################################################################################################################


class MCS:
    """
    A class used to perform Monte Carlo sampling (MCS) of independent random variables from a user-specified probability
    distribution using inverse transform method.

    :param dimension: A scalar value defining the dimension of the random variables
                        Default: len(icdf)
    :type dimension: int

    :param icdf: Inverse cumulative distribution for each random variable.
                The inverse cdf may be defined as a function, a string, a list of functions, a list of strings, or a
                    list of functions and strings
                Each item in the list specifies the distribution of the corresponding random variable.
                If icdf[i] is a string, the cdf is defined in Distributions.py or custom_dist.py
                If icdf[i] is a function, the user must define this function in the script and pass it
    :type icdf: function/string list

    :param icdf_params: Parameters of the inverse cdf (icdf)
                Parameters for each random variable are defined as ndarrays
                Each item in the list, icdf_params[i], specifies the parameters for the corresponding inverse cdf,
                    icdf[i]
    :type icdf_params: list

    :param nsamples: Number of samples to generate
                      No Default Value: nsamples must be prescribed
    :type nsamples: int

    Output:
    :return: MCS.samples: Set of generated samples
    :rtype: MCS.samples: ndarray

    :return: MCS.samplesU01: Set of uniform samples on [0, 1]^dimension
    :rtype: MCS.samplesU01: ndarray

    """

    # Authors: Dimitris G.Giovanis
    # Last Modified: 6/7/18 by Dimitris G. Giovanis & Michael Shields

    def __init__(self, dimension=None, icdf=None, icdf_params=None, nsamples=None):

        self.dimension = dimension
        self.nsamples = nsamples
        self.icdf = icdf
        self.icdf_params = icdf_params
        self.init_mcs()
        self.samplesU01, self.samples = self.run_mcs()

    def run_mcs(self):
        print('UQpy: Performing MCS design...')
        samples = np.random.rand(self.nsamples, self.dimension)
        samples_u_to_x = np.zeros_like(samples)
        for j in range(samples.shape[1]):
            icdf = self.icdf[j]
            for i in range(samples.shape[0]):
                samples_u_to_x[i, j] = icdf(samples[i, j], self.icdf_params[j])
        print('Done!')
        return samples, samples_u_to_x

    ################################################################################################################
    # Initialize Monte Carlo simulation.
    # Necessary parameters:  1. Probability distribution, 2. Probability distribution parameters 3. Number of samples
    # Optional: dimension, names of random variables

    def init_mcs(self):

        # Ensure that the number of samples is defined
        if self.nsamples is None:
            raise NotImplementedError("Exit code: Number of samples not defined.")

        # Ensure that a distribution is assigned.
        if self.icdf is None:
            raise NotImplementedError("Exit code: Distributions not defined.")

        # Check inverse cdf (i_cdf)
        if type(self.icdf).__name__ != 'list':
            self.icdf = [self.icdf]
        if len(self.icdf) == 1 and self.dimension != 1:
            self.icdf = self.icdf * self.dimension
        elif len(self.icdf) != self.dimension:
            raise NotImplementedError("Length of i_cdf should be 1 or equal to dimension")

        # Assign inverse cdf (i_cdf) function for each dimension
        for i in range(len(self.icdf)):
            if type(self.icdf[i]).__name__ == 'function':
                self.icdf[i] = self.icdf[i]
            elif type(self.icdf[i]).__name__ == 'str':
                self.icdf[i] = inv_cdf(self.icdf[i])
            else:
                raise NotImplementedError("Distribution type should be either 'function' or 'list'")

        # Check the dimension
        if self.dimension is None:
            self.dimension = len(self.icdf)

        # Ensure that distribution parameters are assigned
        if self.icdf_params is None:
            raise NotImplementedError("Exit code: Distribution parameters not defined.")

        # Check i_cdf_params
        if type(self.icdf_params).__name__ != 'list':
            self.icdf_params = [self.icdf_params]
        if len(self.icdf_params) == 1 and self.dimension != 1:
            self.icdf_params = self.icdf_params * self.dimension
        elif len(self.icdf_params) != self.dimension:
            raise NotImplementedError("Length of i_cdf_params should be 1 or equal to dimension")

        # Check for dimensional consistency
        if len(self.icdf) != len(self.icdf_params):
            raise NotImplementedError("Exit code: Incompatible dimensions.")


########################################################################################################################
########################################################################################################################
#                                         Latin hypercube sampling  (LHS)
########################################################################################################################


class LHS:
    """Generate samples based on the Latin Hypercube Design.

    A class that creates a Latin Hypercube Design for experiments. Firstly, samples on hypercube [0, 1]^n are generated
    and then translated to the parameter space.

    Input:

    :param dimension: A scalar value defining the dimension of the random variables
                        Default: len(i_cdf)
    :type dimension: int

    :param icdf: Inverse cumulative distribution for each random variable.
                The inverse cdf may be defined as a function, a string, a list of functions, a list of strings, or a
                    list of functions and strings
                Each item in the list specifies the distribution of the corresponding random variable.
                If icdf[i] is a string, the cdf is defined in Distributions.py or custom_dist.py
                If icdf[i] is a function, the user must define this function in the script and pass it
    :type icdf: function/string list

    :param icdf_params: Parameters of the inverse cdf (icdf)
                Parameters for each random variable are defined as arrays
                Each item in the list, icdf_params[i], specifies the parameters for the corresponding inverse cdf,
                    icdf[i]
    :type icdf_params: list

    :param lhs_criterion: The criterion for generating sample points
                            Options:
                                1. 'random' - completely random \n
                                2. 'centered' - points only at the centre \n
                                3. 'maximin' - maximising the minimum distance between points \n
                                4. 'correlate' - minimizing the correlation between the points \n
                            Default: 'random'
    :type lhs_criterion: str

    :param lhs_metric: The distance metric to use. Supported metrics are
                        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice',
                        'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
                        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
                        'yule'.
                        Default: 'euclidean'
    :type lhs_metric: str

    :param lhs_iter: The number of iteration to run. Required only for maximin, correlate and criterion
                        Default: 100
    :type lhs_iter: int

    :param nsamples: Number of samples to generate
                        No Default Value: nsamples must be prescribed
    :type nsamples: int

    Output
    :return: LHS.samples: Set of LHS samples
    :rtype: LHS.samples: ndarray

    :return: LHS.samplesU01: Set of uniform LHS samples on [0, 1]^dimension
    :rtype: LHS.samplesU01: ndarray

    """
    # Created by: Lohit Vandanapu
    # Last modified: 6/7/2018 by Dimitris Giovanis & Michael Shields

    def __init__(self, dimension=1, icdf=None, icdf_params=None, lhs_criterion='random', lhs_metric='euclidean',
                 lhs_iter=100, nsamples=None):

        self.dimension = dimension
        self.nsamples = nsamples
        self.icdf = icdf
        self.icdf_params = icdf_params
        self.lhs_criterion = lhs_criterion
        self.lhs_metric = lhs_metric
        self.lhs_iter = lhs_iter
        self.init_lhs()
        self.samplesU01, self.samples = self.run_lhs()

    def run_lhs(self):

        print('UQpy: Performing LHS design...')
        cut = np.linspace(0, 1, self.nsamples + 1)
        a = cut[:self.nsamples]
        b = cut[1:self.nsamples + 1]

        samples = self._samples(a, b)

        samples_u_to_x = np.zeros_like(samples)
        for j in range(samples.shape[1]):
            i_cdf = self.icdf[j]
            for i in range(samples.shape[0]):
                samples_u_to_x[i, j] = i_cdf(samples[i, j], self.icdf_params[j])

        print('Done')
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

        # Check inverse cdf (i_cdf)
        if type(self.icdf).__name__ != 'list':
            self.icdf = [self.icdf]
        if len(self.icdf) == 1 and self.dimension != 1:
            self.icdf = self.icdf * self.dimension
        elif len(self.icdf) != self.dimension:
            raise NotImplementedError("Length of i_cdf should be 1 or equal to dimension")

        # Assign i_cdf function for each dimension
        for i in range(len(self.icdf)):
            if type(self.icdf[i]).__name__ == 'function':
                self.icdf[i] = self.icdf[i]
            elif type(self.icdf[i]).__name__ == 'str':
                self.icdf[i] = inv_cdf(self.icdf[i])
            else:
                raise NotImplementedError("Distribution type should be either 'function' or 'list'")

        # Check the dimension
        if self.dimension is None:
            self.dimension = len(self.icdf)

        # Ensure that distribution parameters are assigned
        if self.icdf_params is None:
            raise NotImplementedError("Exit code: Distribution parameters not defined.")

        # Check i_cdf_params
        if type(self.icdf_params).__name__ != 'list':
            self.icdf_params = [self.icdf_params]
        if len(self.icdf_params) == 1 and self.dimension != 1:
            self.icdf_params = self.icdf_params * self.dimension
        elif len(self.icdf_params) != self.dimension:
            raise NotImplementedError("Length of i_cdf_params should be 1 or equal to dimension")

        # Check for dimensional consistency
        if len(self.icdf) != len(self.icdf_params):
            raise NotImplementedError("Exit code: Incompatible dimensions.")

        if self.lhs_criterion is None:
            self.lhs_criterion = 'random'
        else:
            if self.lhs_criterion not in ['random', 'centered', 'maximin', 'correlate']:
                raise NotImplementedError("Exit code: Supported lhs criteria: 'random', 'centered', 'maximin', "
                                          "'correlate'")

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
                                          "'russellrao', 'seuclidean','sokalmichener', 'sokalsneath', 'sqeuclidean'")

        if self.lhs_iter is None or self.lhs_iter == 0:
            self.lhs_iter = 1000
        elif self.lhs_iter is not None:
            self.lhs_iter = int(self.lhs_iter)


########################################################################################################################
########################################################################################################################
#                                         Stratified Sampling  (STS)
########################################################################################################################

class STS:

    """Generate samples from an assigned probability density function using Stratified Sampling.

    References:
    M.D. Shields, K. Teferra, A. Hapij, and R.P. Daddazio, "Refined Stratified Sampling for efficient Monte Carlo based
        uncertainty quantification," Reliability Engineering and System Safety, vol. 142, pp. 310-325, 2015.

    Input:
    :param dimension:  A scalar value defining the dimension of target density function.
                    Default: Length of sts_design                
    :type dimension: int
    
    :param icdf: Inverse cumulative distribution for each random variable.
                The inverse cdf may be defined as a function, a string, a list of functions, a list of strings, or a
                    list of functions and strings
                Each item in the list specifies the distribution of the corresponding random variable.
                If icdf[i] is a string, the cdf is defined in Distributions.py or custom_dist.py
                If icdf[i] is a function, the user must define this function in the script and pass it
    :type icdf: function/string list

    :param icdf_params: Parameters of the inverse cdf (icdf)
                Parameters for each random variable are defined as arrays
                Each item in the list, icdf_params[i], specifies the parameters for the corresponding inverse cdf,
                    i_cdf[i]
    :type icdf_params: list

    :param sts_design: Specifies the number of strata in each dimension
    :type sts_design: int list

    :param input_file: File path to input file specifying stratum origins and stratum widths
                    Default: None
    :type input_file: string
    
    Output:
    :return: STS.samples: Set of stratified samples
    :rtype: STS.samples: ndarray

    :return: STS.samplesU01: Set of uniform stratified samples on [0, 1]^dimension
    :rtype: STS.samplesU01: ndarray

    :return: STS.strata: Instance of the class SampleMethods.Strata
    :rtype: STS.strata: ndarray
    
    """

    # Authors: Michael Shields
    # Last modified: 6/7/2018 by Dimitris Giovanis & Michael Shields

    def __init__(self, dimension=None, icdf=None, icdf_params=None, sts_design=None, input_file=None):

        self.dimension = dimension
        self.icdf = icdf
        self.icdf_params = icdf_params
        self.sts_design = sts_design
        self.input_file = input_file
        self.strata = None
        self.init_sts()
        self.samplesU01, self.samples = self.run_sts()

    def run_sts(self):
        print('UQpy: Performing STS design...')
        samples = np.empty([self.strata.origins.shape[0], self.strata.origins.shape[1]], dtype=np.float32)
        samples_u_to_x = np.empty([self.strata.origins.shape[0], self.strata.origins.shape[1]], dtype=np.float32)
        for j in range(0, self.strata.origins.shape[1]):
            icdf = self.icdf[j]
            for i in range(0, self.strata.origins.shape[0]):
                samples[i, j] = np.random.uniform(self.strata.origins[i, j], self.strata.origins[i, j]
                                                  + self.strata.widths[i, j])
                samples_u_to_x[i, j] = icdf(samples[i, j], self.icdf_params[j])
        print('Done!')
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

        # Check i_cdf
        if type(self.icdf).__name__ != 'list':
            self.icdf = [self.icdf]
        if len(self.icdf) == 1 and self.dimension != 1:
            self.icdf = self.icdf * self.dimension
        elif len(self.icdf) != self.dimension:
            raise NotImplementedError("Length of i_cdf should be 1 or equal to dimension")

        # Assign i_cdf function for each dimension
        for i in range(len(self.icdf)):
            if type(self.icdf[i]).__name__ == 'function':
                self.icdf[i] = self.icdf[i]
            elif type(self.icdf[i]).__name__ == 'str':
                self.icdf[i] = inv_cdf(self.icdf[i])
            else:
                raise NotImplementedError("Distribution type should be either 'function' or 'list'")

        # Check the dimension
        if self.dimension is None:
            self.dimension = len(self.icdf)

        # Ensure that distribution parameters are assigned
        if self.icdf_params is None:
            raise NotImplementedError("Exit code: Distribution parameters not defined.")

        # Check i_cdf_params
        if type(self.icdf_params).__name__ != 'list':
            self.icdf_params = [self.icdf_params]
        if len(self.icdf_params) == 1 and self.dimension != 1:
            self.icdf_params = self.icdf_params * self.dimension
        elif len(self.icdf_params) != self.dimension:
            raise NotImplementedError("Length of i_cdf_params should be 1 or equal to dimension")

        # Check for dimensional consistency
        if len(self.icdf) != len(self.icdf_params):
            raise NotImplementedError("Exit code: Incompatible dimensions.")

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

########################################################################################################################
########################################################################################################################
#                                         Class Strata
########################################################################################################################


class Strata:
    """
    Define a rectilinear stratification of the n-dimensional unit hypercube with N strata.

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

    :param input_file: File path to input file specifying stratum origins and stratum widths
                    Default: None
    :type input_file: string

    Output:
    :return origins: An array of dimension N x n specifying the origins of all strata
                    The origins of the strata are the coordinates of the stratum orthotope nearest the global origin
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
                    Sample weights are equal to the product of the strata widths (i.e. they are equal to the size of the
                        strata in the [0, 1]^n space.
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
                             'input file, or [origins] and [widths]')

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
        :return ff: Full-factorial design matrix
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
#                                         Markov Chain Monte Carlo  (MCMC)
########################################################################################################################


class MCMC:

    """Generate samples from an arbitrary probability density function using Markov Chain Monte Carlo.

    This class generates samples from an arbitrary user-specified distribution using Metropolis-Hastings(MH),
    Modified Metropolis-Hastings, of Affine Invariant Ensemble Sampler with stretch moves.

    References:
    S.-K. Au and J. L. Beck, “Estimation of small failure probabilities in high dimensions by subset simulation,”
        Probabilistic Eng. Mech., vol. 16, no. 4, pp. 263–277, Oct. 2001.
    J. Goodman and J. Weare, “Ensemble samplers with affine invariance,” Commun. Appl. Math. Comput. Sci., vol. 5,
        no. 1, pp. 65–80, 2010.

    Input:
    :param dimension:  A scalar value defining the dimension of target density function.
                    Default: 1
    :type dimension: int

    :param pdf_proposal_type: Type of proposal density function for MCMC. Only used with algorithm = 'MH' or 'MMH'
                    Options:
                        'Normal' : Normal proposal density
                        'Uniform' : Uniform proposal density
                    Default: 'Uniform'
                    If dimension > 1 and algorithm = 'MMH', this may be input as a list to assign different proposal
                        densities to each dimension. Example pdf_proposal_type = ['Normal','Uniform'].
                    If dimension > 1, algorithm = 'MMH' and this is input as a string, the proposal densities for all
                        dimensions are set equal to the assigned proposal type.
    :type pdf_proposal_type: str or str list

    :param pdf_proposal_scale: Scale of the proposal distribution
                    If algorithm == 'MH' or 'MMH'
                        For pdf_proposal_type = 'Uniform'
                            Proposal is Uniform in [x-pdf_proposal_scale/2, x+pdf_proposal_scale/2]
                        For pdf_proposal_type = 'Normal'
                            Proposal is Normal with standard deviation equal to pdf_proposal_scale
                    If algorithm == 'Stretch'
                        pdf_proposal_scale sets the scale of the stretch density
                            g(z) = 1/sqrt(z) for z in [1/pdf_proposal_scale, pdf_proposal_scale]
                    Default value: dimension x 1 list of ones
    :type pdf_proposal_scale: float or float list
                    If dimension > 1, this may be defined as float or float list
                        If input as float, pdf_proposal_scale is assigned to all dimensions
                        If input as float list, each element is assigned to the corresponding dimension

    :param pdf_target_type: Type of target density function for acceptance/rejection in MMH. Not used for MH or Stretch.
                    Options:
                        'marginal_pdf': Check acceptance/rejection for a candidate in MMH using the marginal pdf
                                        For independent variables only
                        'joint_pdf': Check acceptance/rejection for a candidate in MMH using the joint pdf
                    Default: 'marginal_pdf'
    :type pdf_target_type: str

    :param pdf_target: Target density function from which to draw random samples
                    The target joint probability density must be a function, or list of functions, or a string.
                    If type == 'str'
                        The assigned string must refer to a custom pdf defined in the file custom_pdf.py in the working
                            directory
                    If type == function
                        The function must be defined in the python script calling MCMC
                    If dimension > 1 and pdf_target_type='marginal_pdf', the input to pdf_target is a list of size
                        [dimensions x 1] where each item of the list defines a marginal pdf.
                    Default: Multivariate normal distribution having zero mean and unit standard deviation
    :type pdf_target: function, function list, or str

    :param pdf_target_params: Parameters of the target pdf
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
                    For 'MH' and 'MMH', this is a single point, defined as a numpy array of dimension (1 x dimension)
                    For 'Stretch', this is a numpy array of dimension N x dimension, where N is the ensemble size
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
    """

    # Authors: Michael D. Shields, Mohit Chauhan, Dimitris G. Giovanis
    # Updated: 4/26/18 by Michael D. Shields

    def __init__(self, dimension=None, pdf_proposal_type=None, pdf_proposal_scale=None, pdf_target_type=None,
                 pdf_target=None, pdf_target_params=None, algorithm=None, jump=None, nsamples=None, seed=None,
                 nburn=None):

        self.pdf_proposal_type = pdf_proposal_type
        self.pdf_proposal_scale = pdf_proposal_scale
        self.pdf_target_type = pdf_target_type
        self.pdf_target = pdf_target
        self.pdf_target_params = pdf_target_params
        self.algorithm = algorithm
        self.jump = jump
        self.nsamples = nsamples
        self.dimension = dimension
        self.seed = seed
        self.nburn = nburn
        self.init_mcmc()
        if self.algorithm is 'Stretch':
            self.ensemble_size = len(self.seed)
        self.samples = self.run_mcmc()

    def run_mcmc(self):
        rejects = 0

        # Defining an array to store the generated samples
        samples = np.zeros([self.nsamples * self.jump, self.dimension])

        ################################################################################################################
        # Classical Metropolis-Hastings Algorithm with symmetric proposal density
        if self.algorithm == 'MH':

            from numpy.random import normal, multivariate_normal, uniform

            samples[0, :] = self.seed

            pdf_ = self.pdf_target[0]

            for i in range(self.nsamples * self.jump - 1 + self.nburn):
                if self.pdf_proposal_type[0] == 'Normal':
                    if self.dimension == 1:
                        candidate = normal(samples[i, :], np.array(self.pdf_proposal_scale))
                        p_proposal = pdf_(candidate, self.pdf_target_params)
                        p_current = pdf_(samples[i, :], self.pdf_target_params)
                        p_accept = p_proposal / p_current

                        accept = np.random.random() < p_accept

                        if accept:
                            samples[i + 1, :] = candidate
                        else:
                            samples[i + 1, :] = samples[i, :]
                            rejects += 1
                    else:
                        if i == 0:
                            self.pdf_proposal_scale = np.diag(np.array(self.pdf_proposal_scale))

                        candidate = multivariate_normal(samples[i, :], np.array(self.pdf_proposal_scale))
                        p_proposal = pdf_(candidate, self.pdf_target_params)
                        p_current = pdf_(samples[i, :], self.pdf_target_params)
                        p_accept = p_proposal / p_current

                        accept = np.random.random() < p_accept

                        if accept:
                            samples[i + 1, :] = candidate
                        else:
                            samples[i + 1, :] = samples[i, :]
                            rejects += 1

                elif self.pdf_proposal_type == 'Uniform':

                    candidate = uniform(low=samples[i, :] - np.array(self.pdf_proposal_scale) / 2,
                                        high=samples[i, :] + np.array(self.pdf_proposal_scale) / 2,
                                        size=self.dimension)
                    p_proposal = pdf_(candidate, self.pdf_target_params)

                    p_current = pdf_(samples[i, :], self.pdf_target_params)
                    p_accept = p_proposal / p_current

                    accept = np.random.random() < p_accept

                    if accept:
                        samples[i + 1, :] = candidate
                    else:
                        samples[i + 1, :] = samples[i, :]
                        rejects += 1

        ################################################################################################################
        # Modified Metropolis-Hastings Algorithm with symmetric proposal density
        elif self.algorithm == 'MMH':

            samples[0, :] = self.seed[0:]

            if self.pdf_target_type == 'marginal_pdf':
                for i in range(self.nsamples * self.jump - 1 + self.nburn):
                    for j in range(self.dimension):

                        pdf_ = self.pdf_target[j]

                        if self.pdf_proposal_type[j] == 'Normal':
                            candidate = np.random.normal(samples[i, j], self.pdf_proposal_scale[j])
                            p_proposal = pdf_(candidate, self.pdf_target_params)
                            p_current = pdf_(samples[i, j], self.pdf_target_params)
                            p_accept = p_proposal / p_current

                            accept = np.random.random() < p_accept

                            if accept:
                                samples[i + 1, j] = candidate
                            else:
                                samples[i + 1, j] = samples[i, j]

                        elif self.pdf_proposal_type[j] == 'Uniform':
                            candidate = np.random.uniform(low=samples[i, j] - self.pdf_proposal_scale[j] / 2,
                                                          high=samples[i, j] + self.pdf_proposal_scale[j] / 2, size=1)

                            p_proposal = pdf_(candidate, self.pdf_target_params)
                            p_current = pdf_(samples[i, j], self.pdf_target_params)
                            p_accept = p_proposal / p_current

                            accept = np.random.random() < p_accept

                            if accept:
                                samples[i + 1, j] = candidate
                            else:
                                samples[i + 1, j] = samples[i, j]

            elif self.pdf_target_type == 'joint_pdf':
                pdf_ = self.pdf_target[0]

                for i in range(self.nsamples * self.jump - 1 + self.nburn):
                    candidate = list(samples[i, :])

                    current = list(samples[i, :])
                    for j in range(self.dimension):
                        if self.pdf_proposal_type[j] == 'Normal':
                            candidate[j] = np.random.normal(samples[i, j], self.pdf_proposal_scale[j])

                        elif self.pdf_proposal_type[j] == 'Uniform':
                            candidate[j] = np.random.uniform(low=samples[i, j] - self.pdf_proposal_scale[j] / 2,
                                                             high=samples[i, j] + self.pdf_proposal_scale[j] / 2,
                                                             size=1)

                        p_proposal = pdf_(candidate, self.pdf_target_params)
                        p_current = pdf_(current, self.pdf_target_params)
                        p_accept = p_proposal / p_current

                        accept = np.random.random() < p_accept

                        if accept:
                            current[j] = candidate[j]
                        else:
                            candidate[j] = current[j]

                    samples[i + 1, :] = current

        ################################################################################################################
        # Affine Invariant Ensemble Sampler with stretch moves
        # Reference: Goodman, J. and Weare, J., (2010) "Ensemble samplers with affine invariance." Communications in
        #               applied mathematics and computational science. 5: 65-80.

        elif self.algorithm == 'Stretch':

            samples[0:self.ensemble_size, :] = self.seed

            pdf_ = self.pdf_target[0]

            for i in range(self.ensemble_size-1, self.nsamples * self.jump - 1):
                complementary_ensemble = samples[i-self.ensemble_size+2:i+1, :]
                s0 = random.choice(complementary_ensemble)
                s = (1+(self.pdf_proposal_scale[0]-1)*random.random())**2/self.pdf_proposal_scale[0]
                candidate = s0+s*(samples[i-self.ensemble_size+1, :]-s0)

                p_proposal = pdf_(candidate, self.pdf_target_params)
                p_current = pdf_(samples[i-self.ensemble_size + 1, :], self.pdf_target_params)
                p_accept = s**(self.dimension-1)*p_proposal/p_current

                accept = np.random.random() < p_accept

                if accept:
                    samples[i + 1, :] = candidate
                else:
                    samples[i + 1, :] = samples[i-self.ensemble_size + 1, :]

        ################################################################################################################
        # Return the samples

        if self.algorithm is 'MMH' or self.algorithm is 'MH':
            print('Successful execution of the MCMC design')
            return samples[self.nburn:self.nsamples * self.jump + self.nburn:self.jump]
        else:
            output = np.zeros((self.nsamples, self.dimension))
            j = 0
            for i in range(self.jump*self.ensemble_size-self.ensemble_size, samples.shape[0],
                           self.jump*self.ensemble_size):
                output[j:j+self.ensemble_size, :] = samples[i:i+self.ensemble_size, :]
                j = j+self.ensemble_size
            return output

    ####################################################################################################################
    # Check to ensure consistency of the user input and assign defaults
    def init_mcmc(self):

        if self.dimension is None:
            self.dimension = 1

        # Check nsamples
        if self.nsamples is None:
            raise NotImplementedError('Exit code: Number of samples not defined.')

        # Check seed
        if self.seed is None:
            self.seed = np.zeros(self.dimension)
        if self.algorithm is not 'Stretch':
            if self.seed.__len__() != self.dimension:
                raise NotImplementedError("Exit code: Incompatible dimensions in 'seed'.")
        else:
            if self.seed.shape[0] < 3:
                raise NotImplementedError("Exit code: Ensemble size must be > 2.")

        # Check jump
        if self.jump is None:
            self.jump = 1

        # Check pdf_proposal_type
        if self.pdf_proposal_type is None:
            self.pdf_proposal_type = 'Uniform'
        # If pdf_proposal_type is entered as a string, make it a list
        if type(self.pdf_proposal_type).__name__ == 'str':
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
        if type(self.pdf_proposal_scale).__name__ != 'list':
            self.pdf_proposal_scale = [self.pdf_proposal_scale]
        if len(self.pdf_proposal_scale) != self.dimension:
            if len(self.pdf_proposal_scale) == 1:
                self.pdf_proposal_scale = self.pdf_proposal_scale * self.dimension
            else:
                raise NotImplementedError("Exit code: Incompatible dimensions in 'pdf_proposal_scale'.")

        # Check pdf_target_type
        if self.algorithm is 'MMH' and self.pdf_target_type is None:
            self.pdf_target_type = 'marginal_pdf'
        if self.algorithm is 'Stretch':
            self.pdf_target_type = 'joint_pdf'
        if self.pdf_target_type not in ['joint_pdf', 'marginal_pdf']:
            raise ValueError('Exit code: Unrecognized type for target distribution. Supported distributions: '
                             'joint_pdf', 'marginal_pdf.')

        # Check algorithm
        if self.algorithm is None:
            self.algorithm = 'MMH'
        else:
            if self.algorithm not in ['MH', 'MMH', 'Stretch']:
                raise NotImplementedError('Exit code: Unrecognized MCMC algorithm. Supported algorithms: '
                                          'Metropolis-Hastings (MH), '
                                          'Modified Metropolis-Hastings (MMH), '
                                          'Affine Invariant Ensemble with Stretch Moves (Stretch).')

        # Check pdf_target
        if type(self.pdf_target).__name__ == 'str':
            self.pdf_target = pdf(self.pdf_target)
        if self.pdf_target is None and self.algorithm is 'MMH':
            if self.dimension == 1 or self.pdf_target_type is 'marginal_pdf':
                def target(x, dummy):
                    return sp.norm.pdf(x)
                if self.dimension == 1:
                    self.pdf_target = [target]
                else:
                    self.pdf_target = [target] * self.dimension
            else:
                def target(x, dummy):
                    return sp.multivariate_normal.pdf(x, mean=np.zeros(self.dimension), cov=np.eye(self.dimension))
                self.pdf_target = [target]
        elif self.pdf_target is None:
            if self.dimension == 1:
                def target(x, dummy):
                    return sp.norm.pdf(x)
                self.pdf_target = [target]
            else:
                def target(x, dummy):
                    return sp.multivariate_normal.pdf(x, mean=np.zeros(self.dimension), cov=np.eye(self.dimension))
                self.pdf_target = [target]
        elif type(self.pdf_target).__name__ != 'list':
            self.pdf_target = [self.pdf_target]

        # Check pdf_target_params
        if self.pdf_target_params is None:
            self.pdf_target_params = []
        if type(self.pdf_target_params).__name__ != 'list':
            self.pdf_target_params = [self.pdf_target_params]

        if self.nburn is None:
            self.nburn = 0


########################################################################################################################
########################################################################################################################
#                                         ADD ANY NEW METHOD HERE
########################################################################################################################
