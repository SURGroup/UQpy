"""This module contains functionality for all the sampling methods supported in UQpy."""
import sys
import copy
from scipy.spatial.distance import pdist
from UQpyLibraries.PDFs import *
import warnings


def init_sm(data):
    ################################################################################################################
    # Add available sampling methods Here
    valid_methods = ['mcs', 'lhs', 'mcmc', 'pss', 'sts', 'SuS']

    ################################################################################################################
    # Check if requested method is available

    if 'Method' in data:
        if data['Method'] not in valid_methods:
            raise NotImplementedError("Method - %s not available" % data['Method'])
    else:
        raise NotImplementedError("No sampling method was provided")

    ################################################################################################################
    # Monte Carlo simulation checks.
    # Necessary parameters:  1. Probability distribution, 2. Probability distribution parameters
    # Optional: number of samples (default 100)

    if data['Method'] == 'mcs':
        if 'Number of Samples' not in data:
            data['Number of Samples'] = None
        if 'Probability distribution (pdf)' not in data:
            raise NotImplementedError("Probability distribution not provided")
        if 'Probability distribution parameters' not in data:
            raise NotImplementedError("Probability distribution parameters not provided")

    ################################################################################################################
    # Latin Hypercube simulation block.
    # Necessary parameters:  1. Probability distribution, 2. Probability distribution parameters
    # Optional: 1. Criterion, 2. Metric, 3. Iterations

    if data['Method'] == 'lhs':
        if 'Number of Samples' not in data:
            data['Number of Samples'] = None
        if 'Probability distribution (pdf)' not in data:
            raise NotImplementedError("Probability distribution not provided")
        if 'Probability distribution parameters' not in data:
            raise NotImplementedError("Probability distribution parameters not provided")
        if 'LHS criterion' not in data:
            data['LHS criterion'] = None
        if 'distance metric' not in data:
            data['distance metric'] = None
        if 'iterations' not in data:
            data['iterations'] = None

    ####################################################################################################################
    # Markov Chain Monte Carlo simulation block.
    # Necessary parameters:  1. Proposal pdf, 2. Probability pdf width, 3. Target pdf, 4. Target pdf parameters
    #                        5. algorithm
    # Optional: 1. Seed, 2. Burn-in

    if data['Method'] == 'mcmc':
        if 'Names of random variables' not in data:
            raise NotImplementedError('Number of random variables cannot be defined. Specify names of random variables')
        if 'seed' not in data:
            data['seed'] = np.zeros(len(data['Names of random variables']))
        if 'skip' not in data:
            data['skip'] = None
        if 'Proposal distribution' not in data:
            data['Proposal distribution'] = None
        else:
            print(data['Proposal distribution'])
            if data['Proposal distribution'] not in ['Uniform', 'Normal']:
                raise ValueError('Invalid Proposal distribution type. Available distributions: Uniform, Normal')

        if 'Target distribution' not in data:
            data['Target distribution'] = None
        else:
            if data['Target distribution'] not in ['multivariate_pdf', 'marginal_pdf', 'normal_pdf']:
                raise ValueError('InvalidTarget distribution type. Available distributions: multivariate_pdf, '
                                 'marginal_pdf,'
                                 'normal_pdf')

        if 'Target distribution parameters' not in data:
            data['Target distribution parameters'] = None

        if 'Proposal distribution width' not in data:
            data['Proposal distribution width'] = None

        if 'MCMC algorithm' not in data:
            data['MCMC algorithm'] = None

    ################################################################################################################
    # Partially stratified sampling (PSS) block.
    # Necessary parameters:  1. pdf, 2. pdf parameters 3. pss design 3. pss strata
    # Optional:

    if data['Method'] == 'pss':
        if 'Probability distribution (pdf)' not in data:
            raise NotImplementedError("Probability distribution not provided")
        elif 'Probability distribution parameters' not in data:
            raise NotImplementedError("Probability distribution parameters not provided")
        if 'PSS design' not in data:
            raise NotImplementedError("PSS design not provided")
        if 'PSS strata' not in data:
            raise NotImplementedError("PSS strata not provided")

    ################################################################################################################
    # Stratified sampling (STS) block.
    # Necessary parameters:  1. pdf, 2. pdf parameters 3. sts design
    # Optional:

    if data['Method'] == 'sts':
        if 'Probability distribution (pdf)' not in data:
            raise NotImplementedError("Probability distribution not provided")
        elif 'Probability distribution parameters' not in data:
            raise NotImplementedError("Probability distribution parameters not provided")
        if 'STS design' not in data:
            raise NotImplementedError("STS design not provided")

    ####################################################################################################################
    # Subset Simulation simulation block.
    # Necessary MCMC parameters:  1. Proposal pdf, 2. Probability pdf width, 3. Target pdf, 4. Target pdf parameters
    #                        5. algorithm
    # Optional: 1. Seed, 2. Burn-in

    if data['Method'] == 'SuS':
        if 'Number of samples' not in data:
            data['Number of samples'] = None
        if 'Number of Samples' not in data:
            data['Number of Samples'] = 100
            warnings.warn("Number of samples not provided. Default number is 100")
        if 'MCMC algorithm' not in data:
            warnings.warn("MCMC algorithm not provided. The Metropolis-Hastings algorithm will be used")
            data['MCMC algorithm'] = 'MH'
        else:
            if data['MCMC algorithm'] not in ['MH', 'MMH']:
                warnings.warn("MCMC algorithm not available. The Metropolis-Hastings algorithm will be used")
                data['MCMC algorithm'] = 'MH'
        if 'Proposal distribution' not in data:
            raise NotImplementedError("Proposal distribution not provided")
        if 'Proposal distribution width' not in data:
            raise NotImplementedError("Proposal distribution parameters (width) not provided")
        if data['MCMC algorithm'] == 'MH':
            if 'Number of random variables' not in data:
                if 'Names of random variables ' not in data:
                    raise NotImplementedError("Dimension of the problem not specified")
                else:
                    data['Number of random variables'] = len(data['Names of random variables'])
            if 'Target distribution parameters' not in data:
                raise NotImplementedError("Target distribution parameters not provided")
        if data['MCMC algorithm'] == 'MMH':
            if 'Marginal Target distribution parameters' not in data:
                raise NotImplementedError("Marginal Target distribution parameters not provided")
            if 'Number of random variables' not in data:
                raise NotImplementedError("Dimension of the problem not specified")
        if 'Burn-in samples' not in data:
            data['Burn-in samples'] = 1
            warnings.warn("Number of samples to skip in order to avoid burn-in not provided."
                          "The default will be set equal to 1")

    ####################################################################################################################
    # Check any NEW METHOD HERE
    #
    #

    ####################################################################################################################
    # Check any NEW METHOD HERE
    #
    #


########################################################################################################################
########################################################################################################################
########################################################################################################################


def run_sm(data):
    ################################################################################################################
    # Run Monte Carlo simulation
    if data['Method'] == 'mcs':
        print("\nRunning  %k \n", data['Method'])
        rvs = MCS(pdf_type=data['Probability distribution (pdf)'],
                  pdf_params=data['Probability distribution parameters'],
                  nsamples=data['Number of Samples'])
        return rvs

    ################################################################################################################
    # Run Latin Hypercube sampling
    elif data['Method'] == 'lhs':
        print("\nRunning  %k \n", data['Method'])
        rvs = LHS(pdf_type=data['Probability distribution (pdf)'],
                  pdf_params=data['Probability distribution parameters'],
                  nsamples=data['Number of Samples'], lhs_metric=data['distance metric'],
                  lhs_iter=data['iterations'], lhs_criterion=data['LHS criterion'])
        return rvs

    ################################################################################################################
    # Run partially stratified sampling
    elif data['Method'] == 'pss':
        print("\nRunning  %k \n", data['Method'])
        rvs = PSS(pdf_type=data['Probability distribution (pdf)'],
                  pdf_params=data['Probability distribution parameters'],
                  pss_design=data['PSS design'], pss_strata=data['PSS strata'])
        return rvs

    ################################################################################################################
    # Run STS sampling

    elif data['Method'] == 'sts':
        print("\nRunning  %k \n", data['Method'])
        rvs = STS(pdf_type=data['Probability distribution (pdf)'],
                  pdf_params=data['Probability distribution parameters'], sts_design=data['STS design'])
        return rvs

    ################################################################################################################
    # Run Markov Chain Monte Carlo sampling

    elif data['Method'] == 'mcmc':
        print("\nRunning  %k \n", data['Method'])
        rvs = MCMC(dimension=len(data['Names of random variables']), pdf_target_type=data['Target distribution'],
                   algorithm=data['MCMC algorithm'], pdf_proposal_type=data['Proposal distribution'],
                   pdf_proposal_width=data['Proposal distribution width'],
                   pdf_target_params=data['Target distribution parameters'], seed=data['seed'],
                   skip=data['skip'], nsamples=data['Number of Samples'])
        return rvs

    ################################################################################################################
    # Run ANY NEW METHOD HERE

    ################################################################################################################
    # Run ANY NEW METHOD HERE


########################################################################################################################
########################################################################################################################
#                                         Monte Carlo simulation
########################################################################################################################


class MCS:
    """
    A class used to perform brute force Monte Carlo sampling (MCS).

    :param nsamples: Number of samples to be generated
    :type nsamples: int
    :param pdf_type: Type of Distributions
    :type pdf_type: list
    :param pdf_params: Distribution parameters
    :type pdf_params: list

    """

    def __init__(self, pdf_type=None, pdf_params=None, nsamples=None):

        self.nsamples = nsamples
        self.pdf_type = pdf_type
        self.pdf_params = pdf_params
        self.init_mcs()
        self.dimension = len(self.pdf_type)
        self.samplesU01, self.samples = self.run_mcs()

    def run_mcs(self):

        samples = np.random.rand(self.nsamples, self.dimension)
        samples_u_to_x = transform_pdf(samples, self.pdf_type, self.pdf_params)

        return samples, samples_u_to_x

    ################################################################################################################
    # Monte Carlo simulation checks.
    # Necessary parameters:  1. Probability distribution, 2. Probability distribution parameters
    # Optional: number of samples (default 100)

    def init_mcs(self):
        if self.nsamples is None:
            self.nsamples = 100
            warnings.warn("Number of samples not provided. Default number is 100")
        if self.pdf_type is None:
            raise NotImplementedError("Probability distribution not provided")
        else:
            for i in self.pdf_type:
                if i not in ['Uniform', 'Normal', 'Lognormal', 'Weibull', 'Beta', 'Exponential']:
                    raise NotImplementedError("Supported distributions: 'Uniform', 'Normal', 'Lognormal', 'Weibull', "
                                              "'Beta', 'Exponential' ")
        if self.pdf_params is None:
            raise NotImplementedError("Probability distribution parameters not provided")
        if len(self.pdf_type) != len(self.pdf_params):
            raise NotImplementedError("Incompatible dimensions")


########################################################################################################################
########################################################################################################################
#                                         Latin hypercube sampling  (LHS)
########################################################################################################################

class LHS:
    """
    A class that creates a Latin Hypercube Design for experiments.

    These points are generated on the U-space(cdf) i.e. [0,1) and should be converted back to X-space(pdf)
    i.e. (-Inf , Inf) for a normal distribution.

    :param pdf_type: Probability distribution of the parameters
    :type pdf_type: list

    :param pdf_params: Probability distribution parameters
    :type pdf_params: list

    :param lhs_criterion: The criterion for generating sample points
                           Options:
                                    1. random - completely random \n
                                    2. centered - points only at the centre \n
                                    3. maximin - maximising the minimum distance between points \n
                                    4. correlate - minimizing the correlation between the points \n
    :type lhs_criterion: str

    :param lhs_iter: The number of iteration to run. Only for maximin, correlate and criterion
    :type lhs_iter: int

    :param lhs_metric: The distance metric to use. Supported metrics are
                        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', \n
                        'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', \n
                        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', \n
                        'yule'.
    :type lhs_metric: str

    """

    def __init__(self, pdf_type=None, pdf_params=None, lhs_criterion=None, lhs_metric=None,
                 lhs_iter=None, nsamples=None):

        self.nsamples = nsamples
        self.pdf_type = pdf_type
        self.pdf_params = pdf_params
        self.lhs_criterion = lhs_criterion
        self.lhs_metric = lhs_metric
        self.lhs_iter = lhs_iter
        self.init_lhs()
        self.samplesU01, self.samples = self.run_lhs()

    def run_lhs(self):

        print('Running LHS for ' + str(self.lhs_iter) + ' iterations')

        cut = np.linspace(0, 1, self.nsamples + 1)
        a = cut[:self.nsamples]
        b = cut[1:self.nsamples + 1]

        if self.lhs_criterion == 'random':
            samples = self._random(a, b)
            samples_u_to_x = transform_pdf(samples, self.pdf_type, self.pdf_params)
            return samples, samples_u_to_x
        elif self.lhs_criterion == 'centered':
            samples = self._centered(a, b)
            samples_u_to_x = transform_pdf(samples, self.pdf_type, self.pdf_params)
            return samples, samples_u_to_x
        elif self.lhs_criterion == 'maximin':
            samples = self._maximin(a, b)
            samples_u_to_x = transform_pdf(samples, self.pdf_type, self.pdf_params)
            return samples, samples_u_to_x
        elif self.lhs_criterion == 'correlate':
            samples = self._correlate(a, b)
            samples_u_to_x = transform_pdf(samples, self.pdf_type, self.pdf_params)
            return samples, samples_u_to_x

    def _random(self, a, b):
        """
        :return: The samples points for the random LHS design

        """
        u = np.random.rand(self.nsamples, self.dimension)
        samples = np.zeros_like(u)

        for i in range(self.dimension):
            samples[:, i] = u[:, i] * (b - a) + a

        for j in range(self.dimension):
            order = np.random.permutation(self.nsamples)
            samples[:, j] = samples[order, j]

        return samples

    def _centered(self, a, b):
        """

        :return: The samples points for the centered LHS design

        """
        samples = np.zeros([self.nsamples, self.dimension])
        centers = (a + b) / 2

        for i in range(self.dimension):
            samples[:, i] = np.random.permutation(centers)

        return samples

    def _maximin(self, a, b):
        """
        :return: The samples points for the Minimax LHS design

        """
        maximin_dist = 0
        samples = self._random(a, b)
        for _ in range(self.lhs_iter):
            samples_try = self._random(a, b)
            d = pdist(samples_try, metric=self.lhs_metric)
            if maximin_dist < np.min(d):
                maximin_dist = np.min(d)
                samples = copy.deepcopy(samples_try)

        print('Achieved miximin distance of ', maximin_dist)

        return samples

    def _correlate(self, a, b):
        """

        :return: The samples points for the minimum correlated LHS design

        """
        min_corr = np.inf
        samples = self._random(a, b)
        for _ in range(self.lhs_iter):
            samples_try = self._random(a, b)
            R = np.corrcoef(np.transpose(samples_try))
            np.fill_diagonal(R, 1)
            R1 = R[R != 1]
            if np.max(np.abs(R1)) < min_corr:
                min_corr = np.max(np.abs(R1))
                samples = copy.deepcopy(samples_try)
        print('Achieved minimum correlation of ', min_corr)
        return samples

    ################################################################################################################
    # Latin hypercube checks.
    # Necessary parameters:  1. Probability distribution, 2. Probability distribution parameters
    # Optional: number of samples (default 100), criterion, metric, iterations

    def init_lhs(self):

        if self.nsamples is None:
            self.nsamples = 100
            warnings.warn("Number of samples not provided. Default number is 100")

        if self.pdf_type is None:
            raise NotImplementedError("Probability distribution not provided")
        else:
            for i in self.pdf_type:
                if i not in ['Uniform', 'Normal', 'Lognormal', 'Weibull', 'Beta', 'Exponential']:
                    raise NotImplementedError("Supported distributions: 'Uniform', 'Normal', 'Lognormal', 'Weibull', "
                                              "'Beta', 'Exponential' ")
        if self.pdf_params is None:
            raise NotImplementedError("Probability distribution parameters not provided")
        if len(self.pdf_type) != len(self.pdf_params):
            raise NotImplementedError("Incompatible dimensions")
        else:
            self.dimension = len(self.pdf_type)

        if self.lhs_criterion is None:
            self.lhs_criterion = 'random'
            warnings.warn("LHS criterion not defined. The default is random")
        else:
            if self.lhs_criterion not in ['random', 'centered', 'maximin', 'correlate']:
                raise NotImplementedError("Supported lhs criteria: 'random', 'centered', 'maximin', 'correlate'")

        if self.lhs_metric is None:
            self.lhs_metric = 'euclidean'
            warnings.warn("Distance metric for the LHS not defined. The default is Euclidean")
        else:
            if self.lhs_metric not in ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine',
                                       'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
                                       'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                                       'sokalmichener', 'sokalsneath', 'sqeuclidean']:
                raise NotImplementedError("Supported lhs criteria: 'braycurtis', 'canberra', 'chebyshev', 'cityblock',"
                                          " 'correlation', 'cosine','dice', 'euclidean', 'hamming', 'jaccard', "
                                          "'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',"
                                          "'russellrao', 'seuclidean','sokalmichener', 'sokalsneath', 'sqeuclidean'")

        if self.lhs_iter is None:
            self.lhs_iter = 1000
            warnings.warn("Iterations for the LHS not defined. The default number is 1000")


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
    :param pss_strata: Vector defining how each dimension should be stratified.
                        Example: 5D problem with 2x2D + 1x1D subdomains with 625 samples using
                         pss_pss_stratum = [25,25,625].\n
                        Note: pss_pss_stratum(i)^pss_design(i) = number of samples (for all i)
    :return: pss_samples: Generated samples Array (nSamples x nRVs)
    :type pss_design: list
    :type pss_strata: list

    Created by: Jiaxin Zhang
    Last modified: 24/01/2018 by D.G. Giovanis

    """

    # TODO: Jiaxin - Add documentation to this subclass
    # TODO: the pss_design = [[1,4], [2,5], [3]] - then reorder the sequence of RVs
    # TODO: Add the sample check and pss_design check in the beginning
    # TODO: Create a list that contains all element info - parent structure

    def __init__(self, pdf_type=None, pdf_params=None, pss_design=None, pss_strata=None):

        self.pdf_type = pdf_type
        self.pdf_params = pdf_params
        self.pss_design = pss_design
        self.pss_strata = pss_strata
        self.dimension = np.sum(self.pss_design)
        self.init_pss()
        self.nsamples = self.pss_strata[0] ** self.pss_design[0]
        self.samplesU01, self.samples = self.run_pss()

    def run_pss(self):
        samples = np.zeros((self.nsamples, self.dimension))
        samples_u_to_x = np.zeros((self.nsamples, self.dimension))
        col = 0
        for i in range(len(self.pss_design)):
            n_stratum = self.pss_strata[i] * np.ones(self.pss_design[i], dtype=np.int)
            sts = STS(pdf_type=self.pdf_type, pdf_params=self.pdf_params, sts_design=n_stratum, pss_=True)
            index = list(range(col, col + self.pss_design[i]))
            samples[:, index] = sts.samplesU01
            samples_u_to_x[:, index] = sts.samples
            arr = np.arange(self.nsamples).reshape((self.nsamples, 1))
            samples[:, index] = samples[np.random.permutation(arr), index]
            samples_u_to_x[:, index] = samples_u_to_x[np.random.permutation(arr), index]
            col = col + self.pss_design[i]

        return samples, samples_u_to_x

    ################################################################################################################
    # Partially Stratified sampling (PSS) checks.
    # Necessary parameters:  1. pdf, 2. pdf parameters 3. pss design 4. pss strata
    # Optional:

    def init_pss(self):

        if self.pdf_type is None:
            raise NotImplementedError("Probability distribution not provided")
        else:
            for i in self.pdf_type:
                if i not in ['Uniform', 'Normal', 'Lognormal', 'Weibull', 'Beta', 'Exponential']:
                    raise NotImplementedError("Supported distributions: 'Uniform', 'Normal', 'Lognormal', 'Weibull', "
                                              "'Beta', 'Exponential' ")
        if self.pdf_params is None:
            raise NotImplementedError("Probability distribution parameters not provided")

        if self.pss_design is None or self.pss_strata is None:
            raise NotImplementedError("PSS design or strata not provided")
        else:
            if len(self.pss_design) != len(self.pss_strata):
                raise ValueError('Input vectors "pss_design" and "pss_strata" must be the same length')

        if self.dimension != len(self.pdf_type):
            raise ValueError('Incompatible number of random variables and distributions')

        sample_check = np.zeros((len(self.pss_strata), len(self.pss_design)))
        for i in range(len(self.pss_strata)):
            for j in range(len(self.pss_design)):
                sample_check[i, j] = self.pss_strata[i] ** self.pss_design[j]

        if np.max(sample_check) != np.min(sample_check):
            raise ValueError('All dimensions must have the same number of samples/strata.')


########################################################################################################################
########################################################################################################################
#                                         Stratified Sampling  (sts)
########################################################################################################################

class STS:
    # TODO: MDS - Add documentation to this subclass

    def __init__(self, pdf_type=None, pdf_params=None, sts_design=None, pss_=None):
        """

        :param pdf_type:
        :param pdf_params:
        :param sts_design:
        :param pss_: Flag indicating whether STS is used in the framework of PSS
        Last modified: 24/01/2018 by D.G. Giovanis
        """

        self.pdf_type = pdf_type
        self.pdf_params = pdf_params
        self.sts_design = sts_design
        if pss_ is None:
            self.init_sts()
        strata = Strata(nstrata=self.sts_design)
        self.origins = strata.origins
        self.widths = strata.widths
        self.weights = strata.weights
        self.samplesU01, self.samples = self.run_sts()

    def run_sts(self):
        samples = np.empty([self.origins.shape[0], self.origins.shape[1]], dtype=np.float32)
        for i in range(0, self.origins.shape[0]):
            for j in range(0, self.origins.shape[1]):
                samples[i, j] = np.random.uniform(self.origins[i, j], self.origins[i, j] + self.widths[i, j])
        samples_u_to_x = transform_pdf(samples, self.pdf_type, self.pdf_params)
        return samples, samples_u_to_x

    def init_sts(self):

        if self.pdf_type is None:
            raise NotImplementedError("Probability distribution not provided")
        else:
            for i in self.pdf_type:
                if i not in ['Uniform', 'Normal', 'Lognormal', 'Weibull', 'Beta', 'Exponential']:
                    raise NotImplementedError("Supported distributions: 'Uniform', 'Normal', 'Lognormal', 'Weibull', "
                                              "'Beta', 'Exponential' ")
        if self.pdf_params is None:
            raise NotImplementedError("Probability distribution parameters not provided")

        if self.sts_design is None:
            raise NotImplementedError("PSS design or strata not provided")

        if len(self.sts_design) != len(self.pdf_type):
            raise ValueError('Incompatible number of random variables and distributions')

        # TODO: Create a list that contains all element info - parent structure
        # e.g. SS_samples = [STS[j] for j in range(0,nsamples)]
        # hstack


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

    def __init__(self, nstrata=None, input_file=None, origins=None, widths=None):

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
                    sys.exit('Error: The strata are not fully defined. Must provide [nstrata], '
                             'input file, or [origins] and [widths]')

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


########################################################################################################################
########################################################################################################################
#                                         Markov Chain Monte Carlo  (MCMC)
########################################################################################################################


class MCMC:

    """This class generates samples from arbitrary algorithm using Metropolis-Hastings(MH) or
    Modified Metropolis-Hastings Algorithm.


    :param dim:  A scalar value defining the dimension of target density function.
    :type dim: int

    :param pdf_proposal_type: Type of proposed density function. Example:
                     'Normal' : Normal distribution will be used to generate new estimates
                     'Uniform' : Uniform distribution will be used to generate new estimates
    :type pdf_proposal_type: str

    :param pdf_proposal_width: Width of the proposal distribution
    :type pdf_proposal_width: float

    :param pdf_target_type: Type of target density function. Example:
                     'Normal' : Normal density function used to generate samples using the MH Algorithm
                     'Multivariate Normal' : Multivariate normal density function used to generate samples using MH
                     'Marginal': Marginal target density used to generate samples using MMH
    :type pdf_proposal_type: str


    :param pdf_target_params: Properties of the target density function (mean, variance)
    :type pdf_target_params: list

    :param algorithm:  Algorithm used to generate random samples.
                       Default value: method is 'MH'.
                       Example: MCMC_algorithm = MH : Use Metropolis-Hastings Algorithm
                       MCMC_algorithm = MMH : Use Modified Metropolis-Hastings Algorithm
                       MCMC_algorithm = GIBBS : Use Gibbs Sampling Algorithm
    :type algorithm: str

    :param skip: Number of samples rejected to reduce the correlation
                     between generated samples.
    :type: skip: int

    :param nsamples: Number of samples to generate
    :type nsamples: int

    :param seed: Seed of the Markov chain
    :type seed: list

    """

    def __init__(self, dimension=None, pdf_proposal_type=None, pdf_proposal_width=None, pdf_target_type=None,
                 pdf_target_params=None, algorithm=None,   skip=None, nsamples=None, seed=None):

        # TODO: Mohit - Add error checks for target and marginal PDFs

        self.pdf_proposal_type = pdf_proposal_type
        self.pdf_proposal_width = pdf_proposal_width
        self.pdf_target_type = pdf_target_type
        self.pdf_target_params = pdf_target_params
        self.algorithm = algorithm
        self.skip = skip
        self.nsamples = nsamples
        self.dimension = dimension
        self.seed = seed
        self.init_mcmc()
        self.samples = self.run_mcmc()

    def run_mcmc(self):
        rejects = 0
        # Changing the array of param into a diagonal matrix

        # TODO: MDS - If x0 is not provided, start at the mode of the target distribution (if available)
        # if x0 is None:

        # Defining an array to store the generated samples
        samples = np.zeros([self.nsamples * self.skip, self.dimension])
        samples[0, :] = self.seed

        ################################################################################################################
        # Classical Metropolis-Hastings Algorithm with symmetric proposal density
        if self.algorithm == 'MH':

            pdf_ = pdf(self.pdf_target_type)

            for i in range(self.nsamples * self.skip - 1):
                if self.pdf_proposal_type == 'Normal':
                    if self.dimension == 1:
                        proposal = np.random.normal(samples[i, :], np.array(self.pdf_proposal_width))
                    else:
                        pdf_proposal_width = np.diag(np.array(self.pdf_proposal_width))
                        proposal = np.random.multivariate_normal(samples[i, :], np.array(pdf_proposal_width))

                elif self.pdf_proposal_type == 'Uniform':

                    proposal = np.random.uniform(low=samples[i, :] - np.array(self.pdf_proposal_width) / 2,
                                                 high=samples[i, :] + np.array(self.pdf_proposal_width) / 2,
                                                 size=self.dimension)

                p_proposal = pdf_(proposal, self.dimension)
                p_current = pdf_(samples[i, :], self.dimension)
                p_accept = p_proposal / p_current

                accept = np.random.random() < p_accept

                if accept:
                    samples[i + 1, :] = proposal
                else:
                    samples[i + 1, :] = samples[i, :]
                    rejects += 1

        ################################################################################################################
        # Modified Metropolis-Hastings Algorithm with symmetric proposal density
        elif self.algorithm == 'MMH':

            for i in range(self.nsamples * self.skip - 1):
                for j in range(self.dimension):

                    pdf_ = pdf(self.pdf_target_type)

                    if self.pdf_proposal_type == 'Normal':
                        proposal = np.random.normal(samples[i, j], self.pdf_proposal_width)

                    elif self.pdf_proposal_type == 'Uniform':

                        proposal = np.random.uniform(low=samples[i, j] - self.pdf_proposal_width / 2,
                                                     high=samples[i, j] + self.pdf_proposal_width / 2, size=1)

                    p_proposal = pdf_(proposal, self.pdf_target_params)
                    p_current = pdf_(samples[i, j], self.pdf_target_params)
                    p_accept = p_proposal / p_current

                    accept = np.random.random() < p_accept

                    if accept:
                        samples[i + 1, j] = proposal
                    else:
                        samples[i + 1, j] = samples[i, j]

        return samples[0:self.nsamples * self.skip:self.skip]

            # TODO: MDS - Add affine invariant ensemble MCMC
            # TODO: MDS - Add Gibbs Sampler

    def init_mcmc(self):

        if self.nsamples is None:
            raise NotImplementedError('Number of samples not defined.')
        if self.seed is None:
            self.seed = np.zeros(self.dimension)
        if self.skip is None:
            self.skip = 1
        if self.pdf_proposal_type is None:
            self.pdf_target_type = 'Uniform'
        if self.pdf_proposal_type not in ['Uniform', 'Normal']:
            raise ValueError('Invalid Proposal distribution type. Available distributions: Uniform, Normal')
        if self.pdf_target_type is None:
            self.pdf_target_type = 'marginal_pdf'
        if self.pdf_target_type not in ['multivariate_pdf', 'marginal_pdf', 'normal_pdf']:
            raise ValueError('InvalidTarget distribution type. Available distributions: multivariate_pdf, marginal_pdf,'
                             'normal_pdf')
        if self.pdf_target_params is None:
            warnings.warn('Target parameters not defined. Default values are  [0, 1]')
            self.pdf_target_params = [0, 1]

        if self.pdf_proposal_width is None:
            warnings.warn('Proposal width not defined. Default value is 2')
            self.pdf_proposal_width = 2

        if self.algorithm is None:
            if self.pdf_target_type is not None:
                if self.pdf_target_type in ['marginal_pdf']:
                    warnings.warn('MCMC algorithm not defined. The MMH will be used')
                    self.algorithm = 'MMH'
                elif self.pdf_target_type in ['multivariate_pdf', 'normal_pdf']:
                    warnings.warn('MCMC algorithm not defined. The MH will be used')
                    self.algorithm = 'MH'
        else:
            if self.algorithm not in ['MH', 'MMH']:
                raise NotImplementedError('Invalid MCMC algorithm. Select from: MH, MMH')

########################################################################################################################
########################################################################################################################
#                                         ADD ANY NEW METHOD HERE
########################################################################################################################
