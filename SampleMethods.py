"""Design of Experiment methods. """
from library import *
import scipy.stats as stats
from modelist import *
import os
import sys
import copy
from scipy.spatial.distance import pdist


class SampleMethods:
    def __init__(self, distribution=None, dimension=None, parameters=None, method=None,
                 input_file=None, input_dir=None, save_format=None, save_name=None, lhs_params=None,
                 lss_params=None, mcmc_params=None):
        """
        :param distribution:
        :param dimension:
        :param parameters:
        :param input_file:
        :param input_dir:
        :param save_format:
        :param save_name:

        """
        self.input_file = input_file

        if self.input_file is not None:
            self.input_dir = input_dir
            if self.input_dir is None:
                raise RuntimeError('Error: Provide path for the input file')
            else:
                os.chdir(self.input_dir)
                handle = read_param_file(self.input_file)
                self.distribution = handle['distribution']
                self.method = handle['method']
                self.dimension = handle['dim']
                self.parameters = handle['parameters']
                self.nsamples = handle['number']
                self.save_format = handle['save_format']
                self.save_name = handle['save_name']
                self.method = handle['method']
                self.checks_ = self.checks_()

        else:
            self.method = method
            self.distribution = distribution
            self.dimension = dimension
            self.parameters = parameters
            self.lhs_params = lhs_params
            self.lss_params = lss_params
            self.mcmc_params = mcmc_params
            self.save_format = save_format
            self.save_name = save_name
            self.input_dir = input_dir
            self.checks_ = self.checks_()
            # self.sts = self.sts(strata)

    def checks_(self):

        """
        :return:

        """
        if self.input_file is not None:
            self.method = read_method(self.method)

        if self.dimension is None and self.distribution is None:
            self.dimension = 1
            self.distribution = 'uniform'

        if self.distribution is None and self.dimension > 1:
            self.distribution = []
            if not is_integer(self.dimension):  # Check if the number is an integer
                print('\n**ERROR**\nThe number of samples should be an integer.')
                return
            else:
                return int(self.dimension)
            for i in range(self.dimension):
                self.distribution.append('uniform')

        elif self.dimension is None and self.distribution is not None:
            self.dimension = len(self.dist)
        else:
            if self.dimension != len(self.distribution):
                raise ValueError("number of dimensions"
                                 " must be equal to the number of provided distributions.(%d != %d)"
                                 % (len(self.distribution), self.dimension))

        if self.parameters is None:
            self.parameters = []
            for i in range(self.dimension):
                self.parameters.append([0, 1])

        else:
            self.parameters = self.parameters


    ########################################################################################################################
    #                                         Monte Carlo simulation
    ########################################################################################################################
    class MCS:
        def __init__(self, nsamples=None, dimension=None):

            """
            :param nsamples:
            """

            self.nsamples = nsamples
            self.dimension = dimension
            self.samples = self.run_mcs()

        def run_mcs(self):

            if self.nsamples is None:
                n = 100
            else:
                if not is_integer(self.nsamples):
                    print('\n**ERROR**\nThe number of samples should be an integer.')
                    return
                else:
                    n = self.nsamples

            return np.random.rand(n, self.dimension)

########################################################################################################################
########################################################################################################################
#                                         Latin hypercube sampling  (LHS)
########################################################################################################################

    class LHS:
        """
        A class that can be used to create Latin Hypercube Sampling for an experimental design. These points should
        be transformed from U space back into the X space.

        :param ndim: The number of dimensions for the experimental design.
        :type: int

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

        def __init__(self, ndim, nsamples=None,  criterion='random', iterations=100, dist_metric='euclidean'):
            try:
                self.ndim = np.int32(ndim)
            except ValueError:
                print('Invalid value for ndim. Must be an integer.')

            if nsamples is None:
                nsamples = ndim

            try:
                self.nsamples = np.int32(nsamples)
            except ValueError:
                print('Invalid value for nsamples. Must be an integer.')

            self.samples = None

            if criterion not in ['random', 'centered', 'maximin', 'correlate', 'correlate_cond']:
                sys.exit('Invalid criterion requested.')

            self.criterion = criterion

            if dist_metric not in ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine',
                                      'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching',
                                      'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
                                      'sokalsneath', 'sqeuclidean', 'yule']:
                sys.exit('Invalid distance metric requested.')

            self.dist_metric = dist_metric

            if criterion in ['maximin', 'correlate']:
                try:
                    self.iterations = np.int32(iterations)
                except ValueError:
                    print('Invalid value for iterations.')

                print('Running for ' + str(self.iterations) + ' iterations')

            cut = np.linspace(0, 1, self.nsamples + 1)
            self.a = cut[:self.nsamples]
            self.b = cut[1:self.nsamples + 1]

            if self.criterion == 'random':
                self.samples = self.random()
            elif self.criterion == 'centered':
                self.samples = self.centered()
            elif self.criterion == 'maximin':
                self.samples = self.maximin()
            elif self.criterion == 'correlate':
                self.samples = self.correlate()

        def random(self):
            """
            
            :return: The samples points for the random LHS design
            
            """
            u = np.random.rand(self.nsamples, self.ndim)
            points = np.zeros_like(u)

            for i in range(self.ndim):
                points[:, i] = u[:, i] * (self.b - self.a) + self.a

            for j in range(self.ndim):
                order = np.random.permutation(self.nsamples)
                points[:, j] = points[order, j]
            return points

        def centered(self):
            """
            
            :return: The samples points for the centered LHS design
            
            """
            points = np.zeros([self.nsamples, self.ndim])
            centers = (self.a + self.b) / 2

            for i in range(self.ndim):
                points[:, i] = np.random.permutation(centers)
            return points

        def maximin(self):
            """
            
            :return: The samples points for the Minimax LHS design 
            
            """
            maximin_dist = 0
            points = self.random()
            for _ in range(self.iterations):
                points_try = self.random()
                d = pdist(points_try, metric=self.dist_metric)
                if maximin_dist < np.min(d):
                    maximin_dist = np.min(d)
                    points = copy.deepcopy(points_try)
            print('Achieved miximin distance of ', maximin_dist)
            return points

        def correlate(self):
            """
            
            :return: The samples points for the minimum correlated LHS design
             
            """
            min_corr = np.inf
            points = self.random()
            for _ in range(self.iterations):
                points_try = self.random()
                R = np.corrcoef(np.transpose(points_try))
                np.fill_diagonal(R, 1)
                R1 = R[R != 1]
                if np.max(np.abs(R1)) < min_corr:
                    min_corr = np.max(np.abs(R1))
                    points = copy.deepcopy(points_try)
            print('Achieved minimum correlation of ', min_corr)
            return points

########################################################################################################################
########################################################################################################################
#                                         Partially Stratified Sampling (PSS)
########################################################################################################################
    class PSS:
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

        # TODO: Jiaxin - Add documentation to this subclass
        # TODO: the pss_design = [[1,4], [2,5], [3]] - then reorder the sequence of RVs
        # TODO: Add the sample check and pss_design check in the beginning
        # TODO: Create a list that contains all element info - parent structure


        def __init__(self, pss_design=None, pss_stratum=None):
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
            if len(pss_design) != len(pss_stratum):
                print('Input vectors "pss_design" and "pss_strata" must be the same length')

            # sample check
            sample_check = np.zeros((len(pss_stratum), len(pss_design)))
            for i in range(len(pss_stratum)):
                for j in range(len(pss_design)):
                    sample_check[i, j] = pss_stratum[i] ** pss_design[j]

            if np.max(sample_check) != np.min(sample_check):
                print('All dimensions must have the same number of samples/strata. '
                      'Check to ensure that all values of pss_strata.^pss_design are equal.')

            n_dim = np.sum(pss_design)
            n_samples = pss_stratum[0] ** pss_design[0]
            pss_samples = np.zeros((n_samples, n_dim))

            col = 0
            for i in range(len(pss_design)):
                n_stratum = pss_stratum[i] * np.ones(pss_design[i], dtype=np.int)

                ss = Strata(nstrata=n_stratum)
                ss = SampleMethods.STS(strata=ss)

                index = list(range(col, col + pss_design[i]))
                pss_samples[:, index] = ss.samples
                arr = np.arange(n_samples).reshape((n_samples, 1))
                pss_samples[:, index] = pss_samples[np.random.permutation(arr), index]
                col = col + pss_design[i]

            self.samples = pss_samples

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
        def __init__(self, strata=None):

            # x = strata.origins.shape[1]
            samples = np.empty([strata.origins.shape[0], strata.origins.shape[1]], dtype=np.float32)
            for i in range(0, strata.origins.shape[0]):
                for j in range(0, strata.origins.shape[1]):
                    samples[i, j] = np.random.uniform(strata.origins[i, j], strata.origins[i, j] + strata.widths[i, j])

            self.samples = samples
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

        """This class generates samples from arbitrary algorithm using Metropolis Hasting(MH) or
        Modified Metroplis Hasting Algorithm.

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
        Example: MCMC_algorithm = MH : Use Metropolis-Hasting Algorithm
        MCMC_algorithm = MMH : Use Modified Metropolis-Hasting Algorithm
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
        :type marginals_parameters: matrix

        Created by: Mohit S. Chauhan
        Last modified: 12/03/2017

        """

        def __init__(self, nsamples=5000, dim=2, x0=np.zeros(2), MCMC_algorithm='MH', proposal='Normal',
                     params=np.ones(2), target=None, njump=1, marginal_parameters=np.identity(2)):

            """This class generates samples from arbitrary algorithm using Metropolis Hasting(MH) or
            Modified Metroplis Hasting Algorithm.

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
            Example: MCMC_algorithm = MH : Use Metropolis-Hasting Algorithm
            MCMC_algorithm = MMH : Use Modified Metropolis-Hasting Algorithm
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
            :type marginals_parameters: matrix

            Created by: Mohit S. Chauhan
            Last modified: 12/03/2017

            """

            # TODO: Mohit - Add error checks for target and marginal PDFs

            try:
                dim = np.int32(dim)
            except ValueError:
                print("Dimension should be an integer value. Try again...")

            if MCMC_algorithm not in ['MH', 'MMH', 'GIBBS']:
                sys.exit("Select one of the following algorithm: ['MH', 'MMH', 'GIBBS']")

            if proposal not in ['Normal', 'Uniform']:
                sys.exit("Select one of the following proposed density: ['Normal', 'Uniform']")

            if np.size(params) != np.size(x0):
                sys.exit("Dimension of parameters of Covariance matrix and Initial value should be same")

            if not is_integer(njump):
                sys.exit("Define an integer to reject generated samples")

            self.nsamples = np.int32(nsamples)
            self.dim = dim
            self.method = MCMC_algorithm
            self.proposal = proposal
            self.params = params
            self.target = target
            self.rejects = 0
            self.njump = njump
            self.Marginal_parameters = marginal_parameters

            # Changing the array of param into a diagonal matrix
            if self.proposal == "Normal":
                if self.params.shape[0] or self.params.shape[1] is 1:
                    self.params = np.diag(self.params)

            # TODO: MDS - If x0 is not provided, start at the mode of the target distribution (if available)
            # if x0 is None:

            # Defining a matrix to store the generated samples
            self.samples = np.empty([self.nsamples * self.njump, self.dim])
            self.samples[0] = x0

            # Classical Metropolis Hastings Algorithm with symmetric proposal density
            if self.method is 'MH':
                for i in range(self.nsamples * self.njump-1):

                    # Generating new sample using proposed density
                    if self.proposal is 'Normal':
                        if self.dim is 1:
                            x1 = np.random.normal(self.samples[i], self.params)
                        else:
                            x1 = np.random.multivariate_normal(self.samples[i, :], self.params)

                    elif self.proposal is 'Uniform':
                        x1 = np.random.uniform(low=self.samples[i] - self.params / 2,
                                               high=self.samples[i] + self.params / 2, size=self.dim)

                    # Ratio of probability of new sample to previous sample
                    a = self.target(x1) / self.target(self.samples[i, :])

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

            # Modified Metropolis Hastings Algorithm with symmetric proposal density
            elif self.method is 'MMH':
                for i in range(self.nsamples * self.njump - 1):

                    # Samples generated from marginal PDFs will be stored in x1
                    x1 = self.samples[i, :]
                    for j in range(self.dim):

                        # Generating new sample using proposed density
                        if self.proposal is 'Normal':
                            xm = np.random.normal(self.samples[i, j], self.params[j][j])

                        elif self.proposal is 'Uniform':
                            xm = np.random.uniform(low=self.samples[i, j] - self.params[j] / 2,
                                                   high=self.samples[i, j] + self.params[j] / 2, size=1)

                        b = self.target(xm, self.Marginal_parameters[j]) / self.target(x1[j], self.Marginal_parameters[j])
                        if b >= 1:
                            x1[j] = xm

                        elif np.random.random() < b:
                            x1[j] = xm

                    self.samples[i+1, :] = x1

                # Reject the samples using njump to reduce the correlation
                self.samples = self.samples[0:self.nsamples * self.njump:self.njump]


            # TODO: MDS - Add affine invariant ensemble MCMC

            # TODO: MDS - Add Gibbs Sampler

########################################################################################################################
########################################################################################################################
#                                         Class Strata
########################################################################################################################

class Strata:
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
            File path to input file specifying stratum origins and stratum witdths
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
