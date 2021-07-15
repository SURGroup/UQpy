from UQpy.Distributions import *
import numpy as np
from scipy.spatial.distance import pdist
import scipy.stats as stats
import copy

########################################################################################################################
########################################################################################################################
#                                         Latin hypercube sampling  (LHS)
########################################################################################################################


class LHS:

    """
    Perform Latin hypercube sampling (MCS) of random variables.

    **Input:**

    * **dist_object** ((list of) ``Distribution`` object(s)):
        List of ``Distribution`` objects corresponding to each random variable.

        All distributions in ``LHS`` must be independent. ``LHS`` does not generate correlated random variables.
        Therefore, for multi-variate designs the `dist_object` must be a list of ``DistributionContinuous1D`` objects
        or an object of the ``JointInd`` class.

    * **nsamples** (`int`):
        Number of samples to be drawn from each distribution.

    * **criterion** (`str` or `callable`):
        The criterion for pairing the generating sample points
            Options:
                1. 'random' - completely random. \n
                2. 'centered' - points only at the centre. \n
                3. 'maximin' - maximizing the minimum distance between points. \n
                4. 'correlate' - minimizing the correlation between the points. \n
                5. `callable` - User-defined method.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **verbose** (`Boolean`):
        A boolean declaring whether to write text to the terminal.

    * ****kwargs**
        Additional arguments to be passed to the method specified by `criterion`

    **Attributes:**

    * **samples** (`ndarray`):
        The generated LHS samples.

    * **samples_U01** (`ndarray`):
        The generated LHS samples on the unit hypercube.

    **Methods**

    """

    def __init__(self, dist_object, nsamples, criterion=None, random_state=None, verbose=False,
                 **kwargs):

        # Check if a Distribution object is provided.
        from UQpy.Distributions import DistributionContinuous1D, JointInd

        if isinstance(dist_object, list):
            for i in range(len(dist_object)):
                if not isinstance(dist_object[i], DistributionContinuous1D):
                    raise TypeError('UQpy: A DistributionContinuous1D object must be provided.')
        else:
            if not isinstance(dist_object, (DistributionContinuous1D, JointInd)):
                raise TypeError('UQpy: A DistributionContinuous1D or JointInd object must be provided.')

        self.dist_object = dist_object
        self.kwargs = kwargs

        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

        if isinstance(criterion, str):
            if criterion not in ['random', 'centered', 'maximin', 'correlate']:
                raise NotImplementedError("Exit code: Supported lhs criteria: 'random', 'centered', 'maximin', "
                                          "'correlate'.")
            else:
                self.criterion = criterion
        else:
            self.criterion = criterion

        if isinstance(nsamples, int):
            self.nsamples = nsamples
        else:
            raise ValueError('UQpy: number of samples must be specified.')

        # Set printing options
        self.verbose = verbose

        if isinstance(self.dist_object, list):
            self.samples = np.zeros([self.nsamples, len(self.dist_object)])
        elif isinstance(self.dist_object, DistributionContinuous1D):
            self.samples = np.zeros([self.nsamples, 1])
        elif isinstance(self.dist_object, JointInd):
            self.samples = np.zeros([self.nsamples, len(self.dist_object.marginals)])

        self.samplesU01 = np.zeros_like(self.samples)

        self.run(self.nsamples)

    def run(self, nsamples):

        """
        Execute the random sampling in the ``LHS`` class.

        The ``run`` method is the function that performs random sampling in the ``LHS`` class. If `nsamples` is
        provided, the ``run`` method is automatically called when the ``LHS`` object is defined. The user may also call
        the ``run`` method directly to generate samples. The ``run`` method of the ``LHS`` class cannot be invoked
        multiple times for sample size extension.

        **Input:**

        * **nsamples** (`int`):
            Number of samples to be drawn from each distribution.

            If the ``run`` method is invoked multiple times, the newly generated samples will overwrite the existing
            samples.

        **Output/Returns:**

        The ``run`` method has no returns, although it creates and/or appends the `samples` and `samples_U01` attributes
        of the ``LHS`` object.

        """

        if isinstance(nsamples, int):
            self.nsamples = nsamples
        else:
            raise ValueError('UQpy: number of samples must be specified.')

        if self.verbose:
            print('UQpy: Running Latin Hypercube sampling...')

        cut = np.linspace(0, 1, self.nsamples + 1)
        a = cut[:self.nsamples]
        b = cut[1:self.nsamples + 1]

        u = np.zeros(shape=(self.samples.shape[0], self.samples.shape[1]))
        samples = np.zeros_like(u)
        for i in range(self.samples.shape[1]):
            u[:, i] = stats.uniform.rvs(size=self.samples.shape[0], random_state=self.random_state)
            samples[:, i] = u[:, i] * (b - a) + a

        if self.criterion == 'random' or self.criterion is None:
            u_lhs = self.random(samples, random_state=self.random_state)
        elif self.criterion == 'centered':
            u_lhs = self.centered(samples, random_state=self.random_state, a=a, b=b)
        elif self.criterion == 'maximin':
            u_lhs = self.max_min(samples, random_state=self.random_state, **self.kwargs)
        elif self.criterion == 'correlate':
            u_lhs = self.correlate(samples, random_state=self.random_state, **self.kwargs)
        elif callable(self.criterion):
            u_lhs = self.criterion(samples, random_state=self.random_state, **self.kwargs)
        else:
            raise ValueError('UQpy: A valid criterion is required.')

        self.samplesU01 = u_lhs

        if isinstance(self.dist_object, list):
            for j in range(len(self.dist_object)):
                if hasattr(self.dist_object[j], 'icdf'):
                    self.samples[:, j] = self.dist_object[j].icdf(u_lhs[:, j])

        elif isinstance(self.dist_object, JointInd):
            if all(hasattr(m, 'icdf') for m in self.dist_object.marginals):
                for j in range(len(self.dist_object.marginals)):
                    self.samples[:, j] = self.dist_object.marginals[j].icdf(u_lhs[:, j])

        elif isinstance(self.dist_object, DistributionContinuous1D):
            if hasattr(self.dist_object, 'icdf'):
                self.samples = self.dist_object.icdf(u_lhs)

        if self.verbose:
            print('Successful execution of LHS design.')

    @staticmethod
    def random(samples, random_state=None):
        """
        Method for generating a Latin hypercube design by sampling randomly inside each bin.

        The ``random`` method takes a set of samples drawn randomly from within the Latin hypercube bins and performs a
        random shuffling of them to pair the variables.

        **Input:**

        * **samples** (`ndarray`):
            A set of samples drawn from within each bin.

        * **random_state** (``numpy.random.RandomState`` object):
            A ``numpy.RandomState`` object that fixes the seed of the pseudo random number generation.

        **Output/Returns:**

        * **lhs_samples** (`ndarray`)
            The randomly shuffled set of LHS samples.
        """

        lhs_samples = np.zeros_like(samples)
        nsamples = len(samples)
        for j in range(samples.shape[1]):
            if random_state is not None:
                order = random_state.permutation(nsamples)
            else:
                order = np.random.permutation(nsamples)
            lhs_samples[:, j] = samples[order, j]

        return lhs_samples

    def max_min(self, samples, random_state=None, iterations=100, metric='euclidean'):
        """
        Method for generating a Latin hypercube design that aims to maximize the minimum sample distance.

        **Input:**

        * **samples** (`ndarray`):
            A set of samples drawn from within each LHS bin.

        * **random_state** (``numpy.random.RandomState`` object):
            A ``numpy.RandomState`` object that fixes the seed of the pseudo random number generation.

        * **iterations** (`int`):
            The number of iteration to run in the search for a maximin design.

        * **metric** (`str` or `callable`):
            The distance metric to use.
                Options:
                    1. `str` - Available options are those supported by ``scipy.spatial.distance``
                    2. User-defined function to compute the distance between samples. This function replaces the
                       ``scipy.spatial.distance.pdist`` method.

        **Output/Returns:**

        * **lhs_samples** (`ndarray`)
            The maximin set of LHS samples.

        """

        if isinstance(metric, str):
            if metric not in ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice',
                              'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
                              'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
                              'sqeuclidean']:
                raise NotImplementedError("UQpy Exit code: Please provide a string corresponding to a distance metric"
                                          "supported by scipy.spatial.distance or provide a method to compute a user-"
                                          "defined distance.")

        if not isinstance(iterations, int):
            raise ValueError('UQpy: number of iterations must be an integer.')

        if isinstance(metric, str):
            def d_func(x): return pdist(x, metric=metric)
        elif callable(metric):
            d_func = metric
        else:
            raise ValueError("UQpy: Please provide a valid metric.")

        i = 0
        lhs_samples = LHS.random(samples, random_state)
        d = d_func(lhs_samples)
        max_min_dist = np.min(d)
        while i < iterations:
            samples_try = LHS.random(samples, random_state)
            d = d_func(samples_try)
            if max_min_dist < np.min(d):
                max_min_dist = np.min(d)
                lhs_samples = copy.deepcopy(samples_try)
            i = i + 1

        if self.verbose:
            print('UQpy: Achieved maximum distance of ', max_min_dist)

        return lhs_samples

    def correlate(self, samples, random_state=None, iterations=100):
        """
        Method for generating a Latin hypercube design that aims to minimize spurious correlations.

        **Input:**

        * **samples** (`ndarray`):
            A set of samples drawn from within each LHS bin.

        * **random_state** (``numpy.random.RandomState`` object):
            A ``numpy.RandomState`` object that fixes the seed of the pseudo random number generation.

        * **iterations** (`int`):
            The number of iteration to run in the search for a maximin design.

        **Output/Returns:**

        * **lhs_samples** (`ndarray`)
            The minimum correlation set of LHS samples.

        """

        if not isinstance(iterations, int):
            raise ValueError('UQpy: number of iterations must be an integer.')

        i = 0
        lhs_samples = LHS.random(samples, random_state)
        r = np.corrcoef(np.transpose(lhs_samples))
        np.fill_diagonal(r, 1)
        r1 = r[r != 1]
        min_corr = np.max(np.abs(r1))
        while i < iterations:
            samples_try = LHS.random(samples, random_state)
            r = np.corrcoef(np.transpose(samples_try))
            np.fill_diagonal(r, 1)
            r1 = r[r != 1]
            if np.max(np.abs(r1)) < min_corr:
                min_corr = np.max(np.abs(r1))
                lhs_samples = copy.deepcopy(samples_try)
            i = i + 1
        if self.verbose:
            print('UQpy: Achieved minimum correlation of ', min_corr)

        return lhs_samples

    @staticmethod
    def centered(samples, random_state=None, a=None, b=None):
        """
        Method for generating a Latin hypercube design with samples centered in the bins.

        **Input:**

        * **samples** (`ndarray`):
            A set of samples drawn from within each LHS bin. In this method, the samples passed in are not used.

        * **random_state** (``numpy.random.RandomState`` object):
            A ``numpy.RandomState`` object that fixes the seed of the pseudo random number generation.

        * **a** (`ndarray`)
            An array of the bin lower-bounds.

        * **b** (`ndarray`)
            An array of the bin upper-bounds

        **Output/Returns:**

        * **lhs_samples** (`ndarray`)
            The centered set of LHS samples.
        """

        u_temp = (a + b) / 2
        lhs_samples = np.zeros([samples.shape[0], samples.shape[1]])
        for i in range(samples.shape[1]):
            if random_state is not None:
                lhs_samples[:, i] = random_state.permutation(u_temp)
            else:
                lhs_samples[:, i] = np.random.permutation(u_temp)

        return lhs_samples
