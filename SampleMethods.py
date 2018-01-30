"""This module contains functionality for all the sampling methods supported in UQpy."""

from library import *
import scipy.stats as stats
from modelist import *
import os
import sys
import copy
from scipy.spatial.distance import pdist
from scipy import optimize
from module_ import def_target


class SampleMethods:
    """
    A class that contains the information of the probability model.

    :param distribution: Probability distribution function (pdf) of the random variables
    :param dimension: Stochastic dimension of the problem (number of random variables)
    :param parameters: Parameters of the pdf
                        1. If pdf ~ Uniform :[lower, upper]
                        2. If pdf ~ Normal  :[mean, std]
    :param method:  Sampling method

    """
    def __init__(self, distribution=None, dimension=None, parameters=None, method=None):
        self.method = method
        self.distribution = distribution
        self.dimension = dimension
        self.parameters = parameters

########################################################################################################################
#                                         Monte Carlo simulation
########################################################################################################################
    class MCS:
        """
        A class used to perform brute force Monte Carlo sampling (MCS).

        :param nsamples: Number of samples to be generated
        :param dimension: Stochastic dimension of the problem (number of random variables)

        """
        def __init__(self, generator=None, number=None, dimension=None):
            self.generator = generator
            self.nsamples = number
            self.dim = dimension
            self.samples = self.run_mcs()

        # TODO: transform random variables according to generator.distribution
        def run_mcs(self):

            return np.random.rand(self.nsamples, self.dim)

########################################################################################################################
########################################################################################################################
#                                         Latin hypercube sampling  (LHS)
########################################################################################################################

    class LHS:
        """
        A class that creates a Latin Hypercube Design for experiments.
        
        These points are generated on the U-space(cdf) i.e. [0,1) and should be converted back to X-space(pdf) 
        i.e. (-Inf , Inf) for a normal distribution.

        :param ndim: The number of dimensions for the experimental design.
        :type ndim: int

        :param nsamples: The number of samples to be generated.
        :type nsamples: int

        :param criterion: The criterion for generating sample points \n
                        i) random - completely random \n
                        ii) centered - points only at the centre \n
                        iii) maximin - maximising the minimum distance between points \n
                        iv) correlate - minimizing the correlation between the points \n
        :type criterion: str

        :param iterations: The number of iteration to run. Only for maximin, correlate and criterion
        :type iterations: int

        :param dist_metric: The distance metric to use. Supported metrics are
                        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', \n
                        'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', \n
                        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', \n
                        'yule'.
        :type dist_metric: str

        """

        def __init__(self, generator=None, dimension=2, number=None, lhs_criterion=None, lhs_iter=None, lhs_metric=None):
            self.generator = generator
            self.nsamples = number
            self.lhs_criterion = lhs_criterion
            self.lhs_metric = lhs_metric
            self.lhs_iter = lhs_iter

            print('Running for ' + str(self.iterations) + ' iterations')

            cut = np.linspace(0, 1, self.nsamples + 1)
            self.a = cut[:self.nsamples]
            self.b = cut[1:self.nsamples + 1]

            if self.lhs_criterion == 'random':
                self.samples = self._random()
            elif self.lhs_criterion == 'centered':
                self.samples = self._centered()
            elif self.lhs_criterion == 'maximin':
                self.samples = self._maximin()
            elif self.lhs_criterion == 'correlate':
                self.samples = self._correlate()

        def _random(self):
            """
            :return: The samples points for the random LHS design

            """
            u = np.random.rand(self.nsamples, self.ndim)
            samples = np.zeros_like(u)

            for i in range(self.ndim):
                samples[:, i] = u[:, i] * (self.b - self.a) + self.a

            for j in range(self.ndim):
                order = np.random.permutation(self.nsamples)
                samples[:, j] = samples[order, j]

            return samples

        def _centered(self):
            """

            :return: The samples points for the centered LHS design

            """
            samples = np.zeros([self.nsamples, self.ndim])
            centers = (self.a + self.b) / 2

            for i in range(self.ndim):
                samples[:, i] = np.random.permutation(centers)

            return samples

        def _maximin(self):
            """

            :return: The samples points for the Minimax LHS design

            """
            maximin_dist = 0
            samples = self.random()
            for _ in range(self.iterations):
                samples_try = self.random()
                d = pdist(samples_try, metric=self.lhs_metric)
                if maximin_dist < np.min(d):
                    maximin_dist = np.min(d)
                    points = copy.deepcopy(samples_try)

            print('Achieved miximin distance of ', maximin_dist)

            return samples

        def _correlate(self):
            """

            :return: The samples points for the minimum correlated LHS design

            """
            min_corr = np.inf
            samples = self.random()
            for _ in range(self.lhs_iter):
                samples_try = self.random()
                R = np.corrcoef(np.transpose(samples_try))
                np.fill_diagonal(R, 1)
                R1 = R[R != 1]
                if np.max(np.abs(R1)) < min_corr:
                    min_corr = np.max(np.abs(R1))
                    samples = copy.deepcopy(samples_try)
            print('Achieved minimum correlation of ', min_corr)
            return samples

########################################################################################################################
########################################################################################################################
#                                         Partially Stratified Sampling (PSS)
########################################################################################################################

    class PSS:
        """
        This class generates a partially stratified sample set on U(0,1) as described in:
        Shields, M.D. and Zhang, J. "The generalization of Latin hypercube sampling" Reliability Engineering and
        System Safety. 148: 96-108

        :param pss_design: Vector defining the subdomains to be used.
                           Example: 5D problem with 2x2D + 1x1D subdomains using pss_design = [2,2,1]. \n
                           Note: The sum of the values in the pss_design vector equals the dimension of the problem.
        :param pss_stratum: Vector defining how each dimension should be stratified.
                            Example: 5D problem with 2x2D + 1x1D subdomains with 625 samples using pss_pss_stratum = [25,25,625].\n
                            Note: pss_pss_stratum(i)^pss_design(i) = number of samples (for all i)
        :return: pss_samples: Generated samples Array (nSamples x nRVs)
        :type pss_design: int
        :type pss_stratum: int

        """

        # TODO: Jiaxin - Add documentation to this subclass
        # TODO: the pss_design = [[1,4], [2,5], [3]] - then reorder the sequence of RVs
        # TODO: Add the sample check and pss_design check in the beginning
        # TODO: Create a list that contains all element info - parent structure

        def __init__(self, pss_design=None, pss_strata=None):
            """
            This class generates a partially stratified sample set on U(0,1) as described in:
            Shields, M.D. and Zhang, J. "The generalization of Latin hypercube sampling" Reliability Engineering and System Safety. 148: 96-108

            :param pss_design: Vector defining the subdomains to be used.
                               Example: 5D problem with 2x2D + 1x1D subdomains using pss_design = [2,2,1]. \n
                               Note: The sum of the values in the pss_design vector equals the dimension of the problem.
            :param pss_stratum: Vector defining how each dimension should be stratified.
                                Example: 5D problem with 2x2D + 1x1D subdomains with 625 samples using pss_pss_stratum = [25,25,625].\n
                                Note: pss_pss_stratum(i)^pss_design(i) = number of samples (for all i)
            :return: pss_samples: Generated samples Array (nSamples x nRVs)
            :type pss_design: int
            :type pss_stratum: int

            Created by: Jiaxin Zhang
            Last modified: 12/03/2017
            """

            # Check that the PSS design is valid
            if len(pss_design) != len(pss_strata):
                print('Input vectors "pss_design" and "pss_strata" must be the same length')

            # sample check
            sample_check = np.zeros((len(pss_strata), len(pss_design)))
            for i in range(len(pss_strata)):
                for j in range(len(pss_design)):
                    sample_check[i, j] = pss_strata[i] ** pss_design[j]

            if np.max(sample_check) != np.min(sample_check):
                print('All dimensions must have the same number of samples/strata. '
                      'Check to ensure that all values of pss_strata.^pss_design are equal.')

            n_dim = np.sum(pss_design)
            n_samples = pss_strata[0] ** pss_design[0]
            self.samples = np.zeros((n_samples, n_dim))

            col = 0
            for i in range(len(pss_design)):
                n_stratum = pss_strata[i] * np.ones(pss_design[i], dtype=np.int)

                ss = Strata(nstrata=n_stratum)
                ss = SampleMethods.STS(strata=ss)

                index = list(range(col, col + pss_design[i]))
                self.samples[:, index] = ss.samples
                arr = np.arange(n_samples).reshape((n_samples, 1))
                self.samples[:, index] = self.samples[np.random.permutation(arr), index]
                col = col + pss_design[i]


########################################################################################################################
########################################################################################################################
#                                         Stratified Sampling  (sts)
########################################################################################################################


    # def sts(self, strata):
    #
    #     # x = strata.origins.shape[1]
    #     samples = np.empty([strata.origins.shape[0],strata.origins.shape[1]],dtype=np.float32)
    #     for i in range(0,strata.origins.shape[0]):
    #         for j in range(0,strata.origins.shape[1]):
    #             samples[i,j] = np.random.uniform(strata.origins[i,j],strata.origins[i,j]+strata.widths[i,j])
    #
    #     self.samples = samples

    class STS:
        # TODO: MDS - Add documentation to this subclass
        def __init__(self, generator=None,  strata=None):

            # x = strata.origins.shape[1]
            self.samples = np.empty([strata.origins.shape[0], strata.origins.shape[1]], dtype=np.float32)
            for i in range(0, strata.origins.shape[0]):
                for j in range(0, strata.origins.shape[1]):
                    self.samples[i, j] = np.random.uniform(strata.origins[i, j], strata.origins[i, j] + strata.widths[i, j])

            self.origins = strata.origins
            self.widths = strata.widths
            self.weights = strata.weights
            # self.elements = [self.origins, self.widths, self.weights, self.samples]

            # TODO: Create a list that contains all element info - parent structure
            # e.g. SS_samples = [STS[j] for j in range(0,nsamples)]
            # hstack -

########################################################################################################################
########################################################################################################################
#                                         Markov Chain Monte Carlo  (MCMC)
########################################################################################################################

    class MCMC:

        """This class generates samples from arbitrary algorithm using Metropolis-Hastings(MH) or
        Modified Metropolis-Hastings Algorithm.

        :param nsamples: A scalar value defining the number of random samples that needs to be
                         generate using MCMC. Default value of nsample is 1000.
        :type nsamples: int

        :param dim: A scalar value defining the dimension of target density function.
        :type dim: int

        :param x0: A scalar value defining the initial mean value of proposed density.
                   Default value: x0 is zero row vector of size dim.
                   Example: x0 = 0, Starts sampling using proposed density with mean equal to 0.
        :type x0: array

        :param MCMC_algorithm: A string defining the algorithm used to generate random samples.
                               Default value: method is 'MH'.
                               Example: MCMC_algorithm = MH : Use Metropolis-Hastings Algorithm
                               MCMC_algorithm = MMH : Use Modified Metropolis-Hastings Algorithm
                               MCMC_algorithm = GIBBS : Use Gibbs Sampling Algorithm
        :type MCMC_algorithm: str

        :param proposal: A string defining the type of proposed density function. Example:
                         proposal = Normal : Normal distribution will be used to generate new estimates
                         proposal = Uniform : Uniform distribution will be used to generate new estimates
        :type proposal: str

        :param params: An array defining the Covariance matrix of the proposed density function.
                       Multivariate Uniform distribution : An array of size 'dim'. Multivariate Normal distribution:
                       Either an array of size 'dim' or array of size 'dim x dim'.
                       Default: params is unit row vector
        :type proposal: matrix

        :param target: An function defining the target distribution of generated samples using MCMC.

        :param njump: A scalar value defining the number of samples rejected to reduce the correlation
                      between generated samples.
        :type njump: int

        :param marginal_parameters: A array containing parameters of target marginal distributions.
        :type marginals_parameters: list

        """

        def __init__(self, generator=None, number=5000, dimension=2, mcmc_seed=None, mcmc_algorithm=None,
                     pdf_proposal=None, pdf_proposal_params=None, pdf_target=None, pdf_target_paramrs=None,
                     mcmc_burnIn=None):

            """This class generates samples from arbitrary algorithm using Metropolis-Hastings(MH) or
            Modified Metropolis-Hastings Algorithm.

            :param nsamples: A scalar value defining the number of random samples that needs to be
                             generate using MCMC. Default value of nsample is 1000.
            :type nsamples: int

            :param dim: A scalar value defining the dimension of target density function.
            :type dim: int

            :param x0: A scalar value defining the initial mean value of proposed density.
                       Default value: x0 is zero row vector of size dim.
                       Example: x0 = 0, Starts sampling using proposed density with mean equal to 0.
            :type x0: array

            :param MCMC_algorithm: A string defining the algorithm used to generate random samples.
                                   Default value: method is 'MH'.
                                   Example: MCMC_algorithm = MH : Use Metropolis-Hastings Algorithm
                                   MCMC_algorithm = MMH : Use Modified Metropolis-Hastings Algorithm
                                   MCMC_algorithm = GIBBS : Use Gibbs Sampling Algorithm
            :type MCMC_algorithm: str

            :param proposal: A string defining the type of proposed density function. Example:
                             proposal = Normal : Normal distribution will be used to generate new estimates
                             proposal = Uniform : Uniform distribution will be used to generate new estimates
            :type proposal: str

            :param params: An array defining the Covariance matrix of the proposed density function.
                           Multivariate Uniform distribution : An array of size 'dim'. Multivariate Normal distribution:
                           Either an array of size 'dim' or array of size 'dim x dim'.
                           Default: params is unit row vector
            :type proposal: matrix

            :param target: An function defining the target distribution of generated samples using MCMC.

            :param njump: A scalar value defining the number of samples rejected to reduce the correlation
                          between generated samples.
            :type njump: int

            :param marginal_parameters: A array containing parameters of target marginal distributions.
            :type marginals_parameters: list

            Created by: Mohit S. Chauhan
            Last modified: 12/03/2017

            """

            # TODO: Mohit - Add error checks for target and marginal PDFs

            self.nsamples = number
            self.dim = dimension
            self.mcmc_algorithm = mcmc_algorithm
            self.pdf_proposal = pdf_proposal
            self.pdf_proposal_params = pdf_proposal_params
            self.pdf_target = pdf_target
            self.pdf_target_params = pdf_target_paramrs
            self.rejects = 0
            self.mcmc_seed = mcmc_seed
            self.mcmc_burnIn = mcmc_burnIn

            # Changing the array of param into a diagonal matrix

            if self.proposal == "Normal":
                if self.params.shape[0] or self.params.shape[1] is 1:
                    self.params = np.diag(self.params)

            # TODO: MDS - If x0 is not provided, start at the mode of the target distribution (if available)
            # if x0 is None:

            # Defining a matrix to store the generated samples
            self.samples = np.empty([self.nsamples * self.njump, self.dim])
            self.samples[0] = x0

            # Classical Metropolis-Hastings Algorithm with symmetric proposal density
            if self.method == 'MH':
                for i in range(self.nsamples * self.njump - 1):

                    # Generating new sample using proposed density
                    if self.proposal == 'Normal':
                        if self.dim == 1:
                            x1 = np.random.normal(self.samples[i], self.params)
                        else:
                            x1 = np.random.multivariate_normal(self.samples[i, :], self.params)

                    elif self.proposal == 'Uniform':
                        x1 = np.random.uniform(low=self.samples[i] - self.params / 2,
                                               high=self.samples[i] + self.params / 2, size=self.dim)

                    # Ratio of probability of new sample to previous sample
                    a = self.target(x1, self.dim) / self.target(self.samples[i, :], self.dim)

                    # Accept the generated sample, if probability of new sample is higher than previous sample
                    if a >= 1:
                        self.samples[i + 1] = x1

                    # Accept the generated sample with probability a, if a < 1
                    elif np.random.random() < a:
                        self.samples[i + 1] = x1

                    # Reject the generated sample and accept the previous sample
                    else:
                        self.samples[i + 1] = self.samples[i]
                        self.rejects += 1
                # Reject the samples using njump to reduce the correlation
                self.xi = self.samples[0:self.nsamples * self.njump:self.njump]

            # Modified Metropolis-Hastings Algorithm with symmetric proposal density
            elif self.method == 'MMH':
                for i in range(self.nsamples * self.njump - 1):

                    # Samples generated from marginal PDFs will be stored in x1
                    x1 = self.samples[i, :]
                    for j in range(self.dim):

                        # Generating new sample using proposed density
                        if self.proposal == 'Normal':
                            xm = np.random.normal(self.samples[i, j], self.params[j][j])

                        elif self.proposal == 'Uniform':
                            xm = np.random.uniform(low=self.samples[i, j] - self.params[j] / 2,
                                                   high=self.samples[i, j] + self.params[j] / 2, size=1)

                        b = self.target(xm, self.Marginal_parameters[j]) / self.target(x1[j],
                                                                                       self.Marginal_parameters[j])
                        if b >= 1:
                            x1[j] = xm

                        elif np.random.random() < b:
                            x1[j] = xm

                    self.samples[i + 1, :] = x1

                # Reject the samples using njump to reduce the correlation
                self.xi = self.samples[0:self.nsamples * self.njump:self.njump]


                # TODO: MDS - Add affine invariant ensemble MCMC

                # TODO: MDS - Add Gibbs Sampler


########################################################################################################################
########################################################################################################################
#                                         Stochastic Reduced Order Model  (SROM)                                       #
########################################################################################################################
########################################################################################################################

    class SROM:

        # TODO: Mohit - Write the documentation for the class

        def __init__(self, samples=None, nsamples=None, marginal=None, moments=None, weights_errors=None,
                     weights_function=None, properties=[1, 1, 0]):
            """

            :param samples:
            :param marginal:
            :param moments:
            :param weights_errors:
            :param weights_function: An matrix (nsamples+3 x dimension) containing weights correponding to function
            value for each dimension
            :param properties:
            """

            # TODO: Mohit - Add error checks
            # TODO: Mohit - Transform sample from [0,1] into range of marginals
            dimension = np.size(marginal)
            if np.size(weights_function) == 0:
                weights_function = (1/nsamples)*np.ones([nsamples+3, dimension])
            if np.size(moments) == 0:
                sys.exit('Moments of marginal distribution are required')
            else:
                weights_function = weights_function.tolist()
                for i in range(int(np.size(moments)/np.size(moments[0]))):
                    weights_function.append(moments[i])

            self.samples = samples
            self.marginal = marginal
            self.moments = np.array(moments)
            self.weights_errors = weights_errors
            self.weights_function = np.array(weights_function)
            self.properties = properties

            # TODO: Mohit - Define default calculation for moments
            def f(p, samples, w, mar, n, d, m, alpha):
                e1 = 0.
                e2 = 0.
                e22 = 0.
                e3 = 0.
                samples = np.matrix(samples)
                p = np.transpose(np.matrix(p))
                com = np.append(samples, p, 1)
                for j in range(d):
                    srt = com[np.argsort(com[:, j].flatten())]
                    s = srt[0, :, j]
                    a = srt[0, :, d]
                    A = np.cumsum(a)
                    marginal = def_target(mar[j])
                    for i in range(n):
                        e1 = + w[i, j] * (A[0, i] - marginal(s[0, i])) ** 2

                    e2 =+ (w[i+1, j]) * (np.sum(np.transpose(p) * samples[:, j]) - m[0,j]) ** 2
                    # e22 =+ (w[i+2, j]) * (np.sum(np.array(p) * (np.array(samples[:, j])*np.array(samples[:,j]))) - m[1,j]) ** 2
                    # TODO: Mohit - Add error corresponding to correlation.
                return alpha[0]*e1 + alpha[1]*(e2+e22) + alpha[2]*e3

            def constraint(p):
                return np.sum(p) - 1

            def constraint2(p):
                n = np.size(p)
                return np.ones(n) - p

            def constraint3(p):
                n = np.size(p)
                return p - np.zeros(n)

            cons = ({'type': 'eq', 'fun': constraint}, {'type': 'ineq', 'fun': constraint2},
                    {'type': 'ineq', 'fun': constraint3})

            P = optimize.minimize(f, np.zeros(nsamples),
                                  args=(self.samples, self.weights_function, self.marginal, nsamples, dimension, self.moments, self.weights_errors),
                                  constraints=cons, method='SLSQP')

            self.probability = P.x


########################################################################################################################
########################################################################################################################
#                                         Class Strata
########################################################################################################################

class Strata:
    """
    Define a rectilinear stratification of the n-dimensional unit hypercube with N strata.

    :param nstrata: array-like
                    An array of dimension 1 x n defining the number of strata in each of the n dimensions
                    Creates an equal stratification with strata widths equal to 1/nstrata
                    The total number of strata, N, is the product of the terms of nstrata
                    Example -
                    nstrata = [2, 3, 2] creates a 3d stratification with:
                    2 strata in dimension 0 with stratum widths 1/2
                    3 strata in dimension 1 with stratum widths 1/3
                    2 strata in dimension 2 with stratum widths 1/2

    :param input_file: string
                       File path to input file specifying stratum origins and stratum widths

    :param origins: array-like
                    An array of dimension N x n specifying the origins of all strata
                    The origins of the strata are the coordinates of the stratum orthotope nearest the global origin
                    Example - A 2D stratification with 2 strata in each dimension
                    origins = [[0, 0]
                              [0, 0.5]
                              [0.5, 0]
                              [0.5, 0.5]]

    :param widths: array-like
                   An array of dimension N x n specifying the widths of all strata in each dimension
                   Example - A 2D stratification with 2 strata in each dimension
                   widths = [[0.5, 0.5]
                             [0.5, 0.5]
                             [0.5, 0.5]
                             [0.5, 0.5]]

    """

    def __init__(self,  nstrata=None, input_file=None, origins=None, widths=None):

        """
        Class defines a rectilinear stratification of the n-dimensional unit hypercube with N strata

        :param nstrata: array-like
            An array of dimension 1 x n defining the number of strata in each of the n dimensions
            Creates an equal stratification with strata widths equal to 1/nstrata
            The total number of strata, N, is the product of the terms of nstrata
            Example -
            nstrata = [2, 3, 2] creates a 3d stratification with:
                2 strata in dimension 0 with stratum widths 1/2
                3 strata in dimension 1 with stratum widths 1/3
                2 strata in dimension 2 with stratum widths 1/2

        :param input_file: string
            File path to input file specifying stratum origins and stratum widths
            See documentation ######## for input file format

        :param origins: array-like
            An array of dimension N x n specifying the origins of all strata
            The origins of the strata are the coordinates of the stratum orthotope nearest the global origin
            Example - A 2D stratification with 2 strata in each dimension
            origins = [[0, 0]
                       [0, 0.5]
                       [0.5, 0]
                       [0.5, 0.5]]

        :param widths: array-like
            An array of dimension N x n specifying the widths of all strata in each dimension
            Example - A 2D stratification with 2 strata in each dimension
            widths = [[0.5, 0.5]
                      [0.5, 0.5]
                      [0.5, 0.5]
                      [0.5, 0.5]]

        Created by: Michael D. Shields
        Last modified: 11/4/2017
        Last modified by: Michael D. Shields

        """
        self.input_file = input_file
        self.nstrata = nstrata
        self.origins = origins
        self.widths = widths

        if self.nstrata is None:
            if self.input_file is None:
                if self.widths is None or self.origins is None:
                    sys.exit(
                        'Error: The strata are not fully defined. Must provide [nstrata], input file, or [origins] and [widths]')

            else:
                # Read the strata from the specified input file
                # See documentation for input file formatting
                array_tmp = np.loadtxt(input_file)
                self.origins = array_tmp[:, 0:array_tmp.shape[1] // 2]
                self.width = array_tmp[:, array_tmp.shape[1] // 2:]

                # Check to see that the strata are space-filling
                space_fill = np.sum(np.prod(self.width, 1))
                if 1 - space_fill > 1e-5:
                    sys.exit('Error: The stratum design is not space-filling.')
                if 1 - space_fill < -1e-5:
                    sys.exit('Error: The stratum design is over-filling.')

                    # TODO: MDS - Add a check for disjointness of strata
                    # Check to see that the strata are disjoint
                    # ncorners = 2**self.strata.shape[1]
                    # for i in range(0,len(self.strata)):
                    #     for j in range(0,ncorners):

        else:
            # Use nstrata to assign the origin and widths of a specified rectilinear stratification.
            self.origins = np.divide(self.fullfact(self.nstrata), self.nstrata)
            self.widths = np.divide(np.ones(self.origins.shape), self.nstrata)
            self.weights = np.prod(self.widths, axis=1)

    def fullfact(self, levels):

        # TODO: MDS - Acknowledge the source here.
        """
        Create a general full-factorial design

        Parameters
        ----------
        levels : array-like
            An array of integers that indicate the number of levels of each input
            design factor.

        Returns
        -------
        mat : 2d-array
            The design matrix with coded levels 0 to k-1 for a k-level factor

        Example
        -------
        ::

            >>> fullfact([2, 4, 3])
            array([[ 0.,  0.,  0.],
                   [ 1.,  0.,  0.],
                   [ 0.,  1.,  0.],
                   [ 1.,  1.,  0.],
                   [ 0.,  2.,  0.],
                   [ 1.,  2.,  0.],
                   [ 0.,  3.,  0.],
                   [ 1.,  3.,  0.],
                   [ 0.,  0.,  1.],
                   [ 1.,  0.,  1.],
                   [ 0.,  1.,  1.],
                   [ 1.,  1.,  1.],
                   [ 0.,  2.,  1.],
                   [ 1.,  2.,  1.],
                   [ 0.,  3.,  1.],
                   [ 1.,  3.,  1.],
                   [ 0.,  0.,  2.],
                   [ 1.,  0.,  2.],
                   [ 0.,  1.,  2.],
                   [ 1.,  1.,  2.],
                   [ 0.,  2.,  2.],
                   [ 1.,  2.,  2.],
                   [ 0.,  3.,  2.],
                   [ 1.,  3.,  2.]])

        """
        n = len(levels)  # number of factors
        nb_lines = np.prod(levels)  # number of trial conditions
        H = np.zeros((nb_lines, n))

        level_repeat = 1
        range_repeat = np.prod(levels)
        for i in range(n):
            range_repeat //= levels[i]
            lvl = []
            for j in range(levels[i]):
                lvl += [j] * level_repeat
            rng = lvl * range_repeat
            level_repeat *= levels[i]
            H[:, i] = rng

        return H
