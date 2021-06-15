from UQpy.distributions import *
import numpy as np
from scipy.spatial.distance import pdist
import scipy.stats as stats
import copy


class LatinHypercubeSampling:

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

    def __init__(self, distributions, samples_number, criterion=None, random_state=None, verbose=False,
                 **kwargs):

        # Check if a Distribution object is provided.
        from UQpy.distributions import DistributionContinuous1D, JointIndependent

        if isinstance(distributions, list):
            for i in range(len(distributions)):
                if not isinstance(distributions[i], DistributionContinuous1D):
                    raise TypeError('UQpy: A DistributionContinuous1D object must be provided.')
        else:
            if not isinstance(distributions, (DistributionContinuous1D, JointIndependent)):
                raise TypeError('UQpy: A DistributionContinuous1D or JointInd object must be provided.')

        self.dist_object = distributions
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

        if isinstance(samples_number, int):
            self.samples_number = samples_number
        else:
            raise ValueError('UQpy: number of samples must be specified.')

        # Set printing options
        self.verbose = verbose

        if isinstance(self.dist_object, list):
            self.samples = np.zeros([self.samples_number, len(self.dist_object)])
        elif isinstance(self.dist_object, DistributionContinuous1D):
            self.samples = np.zeros([self.samples_number, 1])
        elif isinstance(self.dist_object, JointIndependent):
            self.samples = np.zeros([self.samples_number, len(self.dist_object.marginals)])

        self.samplesU01 = np.zeros_like(self.samples)

        if self.samples_number is not None:
            self.run(self.samples_number)

    def run(self, samples_number):

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

        if self.samples_number is None:
            self.samples_number = samples_number

        if self.verbose:
            print('UQpy: Running Latin Hypercube sampling...')

        cut = np.linspace(0, 1, self.samples_number + 1)
        a = cut[:self.samples_number]
        b = cut[1:self.samples_number + 1]

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

        elif isinstance(self.dist_object, JointIndependent):
            if all(hasattr(m, 'icdf') for m in self.dist_object.marginals):
                for j in range(len(self.dist_object.marginals)):
                    self.samples[:, j] = self.dist_object.marginals[j].icdf(u_lhs[:, j])

        elif isinstance(self.dist_object, DistributionContinuous1D):
            if hasattr(self.dist_object, 'icdf'):
                self.samples = self.dist_object.icdf(u_lhs)

        if self.verbose:
            print('Successful execution of LHS design.')



