import logging

from UQpy.distributions import *
from UQpy.sampling.latin_hypercube_criteria.baseclass.Criterion import *
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

    def __init__(self, distributions, samples_number, criterion=None):

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

        if isinstance(criterion, Criterion):
            self.criterion = criterion
        else:
            raise NotImplementedError("Exit code: Supported lhs criteria must implement Criterion.")

        if isinstance(samples_number, int):
            self.samples_number = samples_number
        else:
            raise ValueError('UQpy: number of samples must be specified.')

        # Set printing options
        self.logger = logging.getLogger(__name__)

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

        self.logger.info('UQpy: Running Latin Hypercube sampling...')

        self.criterion.create_bins(self.samples)

        u_lhs = self.criterion.generate_samples()
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

        self.logger.info('Successful execution of LHS design.')

    def __copy__(self):
        new = self.__class__(distributions=self.dist_object,
                             criterion=self.criterion)
        new.__dict__.update(self.__dict__)

        return new
