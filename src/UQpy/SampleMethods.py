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

"""This module contains functionality for all the sampling methods supported in ``UQpy``.

The module currently contains the following classes:

- ``MCS``: Class to perform Monte Carlo sampling.
- ``LHS``: Class to perform Latin hypercube sampling.
- ``MCMC``: Class to perform Markov Chain Monte Carlo sampling.
- ``IS``: Class to perform Importance sampling.
"""

import copy

from scipy.spatial.distance import pdist

from UQpy.Distributions import *
from UQpy.Utilities import *


########################################################################################################################
########################################################################################################################
#                                         Monte Carlo Simulation
########################################################################################################################


class MCS:
    """
    Perform Monte Carlo sampling (MCS) of random variables.

    **Input:**

    * **dist_object** ((list of) ``Distribution`` object(s)):
        Probability distribution of each random variable. Must be an object (or a list of objects) of the
        ``Distribution`` class.

    * **nsamples** (`int`):
        Number of samples to be drawn from each distribution.

        The ``run`` method is automatically called if `nsamples` is provided. If `nsamples` is not provided, then the
        ``MCS`` object is created but samples are not generated.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **verbose** (Boolean):
        A boolean declaring whether to write text to the terminal.


    **Attributes:**

    * **samples** (`ndarray` or `list`):
        Generated samples.

        If a list of ``DistributionContinuous1D`` objects is provided for ``dist_object``, then `samples` is an
        `ndarray` with ``samples.shape=(nsamples, len(dist_object))``.

        If a ``DistributionContinuous1D`` object is provided for ``dist_object`` then `samples` is an array with
        `samples.shape=(nsamples, 1)``.

        If a ``DistributionContinuousND`` object is provided for ``dist_object`` then `samples` is an array with
        ``samples.shape=(nsamples, ND)``.

        If a list of mixed ``DistributionContinuous1D`` and ``DistributionContinuousND`` objects is provided then
        `samples` is a list with ``len(samples)=nsamples`` and ``len(samples[i]) = len(dist_object)``.

    * **samplesU01** (`ndarray` (`list`)):
        Generated samples transformed to the unit hypercube.

        This attribute exists only if the ``transform_u01`` method is invoked by the user.


    **Methods**

    """

    def __init__(self, dist_object, nsamples=None,  random_state=None, verbose=False):

        if isinstance(dist_object, list):
            add_continuous_1d = 0
            add_continuous_nd = 0
            for i in range(len(dist_object)):
                if not isinstance(dist_object[i], Distribution):
                    raise TypeError('UQpy: A UQpy.Distribution object must be provided.')
                if isinstance(dist_object[i], DistributionContinuous1D):
                    add_continuous_1d = add_continuous_1d + 1
                elif isinstance(dist_object[i], DistributionND):
                    add_continuous_nd = add_continuous_nd + 1
            if add_continuous_1d == len(dist_object):
                self.list = False
                self.array = True
            else:
                self.list = True
                self.array = False

            self.random_state = random_state
            if isinstance(self.random_state, int):
                self.random_state = np.random.RandomState(self.random_state)
            elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
                raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

            self.dist_object = dist_object
        else:
            if not isinstance(dist_object, Distribution):
                raise TypeError('UQpy: A UQpy.Distribution object must be provided.')
            else:
                self.dist_object = dist_object
                self.list = False
                self.array = True
            self.random_state = random_state
            if isinstance(self.random_state, int):
                self.random_state = np.random.RandomState(self.random_state)
            elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
                raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

        # Instantiate the output attributes.
        self.samples = None
        self.x = None
        self.samplesU01 = None

        # Set printing options
        self.verbose = verbose
        self.nsamples = nsamples

        # Run Monte Carlo sampling
        if nsamples is not None:
            self.run(nsamples=self.nsamples, random_state=self.random_state)

    def run(self, nsamples, random_state=None):
        """
        Execute the random sampling in the ``MCS`` class.

        The ``run`` method is the function that performs random sampling in the ``MCS`` class. If `nsamples` is
        provided, the ``run`` method is automatically called when the ``MCS`` object is defined. The user may also call
        the ``run`` method directly to generate samples. The ``run`` method of the ``MCS`` class can be invoked many
        times and each time the generated samples are appended to the existing samples.

        ** Input:**

        * **nsamples** (`int`):
            Number of samples to be drawn from each distribution.

            If the ``run`` method is invoked multiple times, the newly generated samples will be appended to the
            existing samples.

        * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
            Random seed used to initialize the pseudo-random number generator. Default is None.

            If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
            object itself can be passed directly.

        **Output/Returns:**

        The ``run`` method has no returns, although it creates and/or appends the `samples` attribute of the ``MCS``
        class.

        """
        # Check if a random_state is provided.
        if random_state is None:
            random_state = self.random_state
        else:
            if isinstance(random_state, int):
                random_state = np.random.RandomState(random_state)
            elif not isinstance(random_state, (type(None), np.random.RandomState)):
                raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

        if nsamples is None:
            raise ValueError('UQpy: Number of samples must be defined.')
        if not isinstance(nsamples, int):
            raise ValueError('UQpy: nsamples should be an integer.')

        if self.verbose:
            print('UQpy: Running Monte Carlo Sampling.')

        if isinstance(self.dist_object, list):
            temp_samples = list()
            for i in range(len(self.dist_object)):
                if hasattr(self.dist_object[i], 'rvs'):
                    temp_samples.append(self.dist_object[i].rvs(nsamples=nsamples, random_state=random_state))
                else:
                    ValueError('UQpy: rvs method is missing.')
            self.x = list()
            for j in range(nsamples):
                y = list()
                for k in range(len(self.dist_object)):
                    y.append(temp_samples[k][j])
                self.x.append(np.array(y))
        else:
            if hasattr(self.dist_object, 'rvs'):
                temp_samples = self.dist_object.rvs(nsamples=nsamples, random_state=random_state)
                self.x = temp_samples

        if self.samples is None:
            if isinstance(self.dist_object, list) and self.array is True:
                self.samples = np.hstack(np.array(self.x)).T
            else:
                self.samples = np.array(self.x)
        else:
            # If self.samples already has existing samples, append the new samples to the existing attribute.
            if isinstance(self.dist_object, list) and self.array is True:
                self.samples = np.concatenate([self.samples, np.hstack(np.array(self.x)).T], axis=0)
            elif isinstance(self.dist_object, Distribution):
                self.samples = np.vstack([self.samples, self.x])
            else:
                self.samples = np.vstack([self.samples, self.x])
        self.nsamples = len(self.samples)

        if self.verbose:
            print('UQpy: Monte Carlo Sampling Complete.')

    def transform_u01(self):
        """
        Transform random samples to uniform on the unit hypercube.

        **Input:**

        The ``transform_u01`` method is an instance method that perform the transformation on an existing ``MCS``
        object. It takes no input.

        **Output/Returns:**

        The ``transform_u01`` method has no returns, although it creates and/or appends the `samplesU01` attribute of
        the ``MCS`` class.

        """

        if isinstance(self.dist_object, list) and self.array is True:
            zi = np.zeros_like(self.samples)
            for i in range(self.nsamples):
                z = self.samples[i, :]
                for j in range(len(self.dist_object)):
                    if hasattr(self.dist_object[j], 'cdf'):
                        zi[i, j] = self.dist_object[j].cdf(z[j])
                    else:
                        raise ValueError('UQpy: All Distributions must have a cdf method.')
            self.samplesU01 = zi

        elif isinstance(self.dist_object, Distribution):
            if hasattr(self.dist_object, 'cdf'):
                zi = np.zeros_like(self.samples)
                for i in range(self.nsamples):
                    z = self.samples[i, :]
                    zi[i, :] = self.dist_object.cdf(z)
                self.samplesU01 = zi
            else:
                raise ValueError('UQpy: All Distributions must have a cdf method.')

        elif isinstance(self.dist_object, list) and self.list is True:
            temp_samples_u01 = list()
            for i in range(self.nsamples):
                z = self.samples[i][:]
                y = [None] * len(self.dist_object)
                for j in range(len(self.dist_object)):
                    if hasattr(self.dist_object[j], 'cdf'):
                        zi = self.dist_object[j].cdf(z[j])
                    else:
                        raise ValueError('UQpy: All Distributions must have a cdf method.')
                    y[j] = zi
                temp_samples_u01.append(np.array(y))
            self.samplesU01 = temp_samples_u01

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

        if self.nsamples is not None:
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

            If the ``run`` method is invoked multiple times, the newly generated samples will be overwrite the
            existing samples.

        **Output/Returns:**

        The ``run`` method has no returns, although it creates and/or appends the `samples` and `samples_U01` attributes
        of the ``LHS`` object.

        """

        if self.nsamples is None:
            self.nsamples = nsamples

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

########################################################################################################################
########################################################################################################################
#                                         Stratified Sampling  (STS)
########################################################################################################################
class STS:
    """
    Generate samples from an assigned probability density function using Stratified Sampling.

    **References:**

    1. M.D. Shields, K. Teferra, A. Hapij, and R.P. Daddazio, "Refined Stratified Sampling for efficient Monte
       Carlo based uncertainty quantification," Reliability Engineering and System Safety,vol.142, pp.310-325,2015.

    **Input:**

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

    **Attributes:**

    :return: STS.samples: Set of stratified samples.
    :rtype: STS.samples: ndarray

    :return: STS.samplesU01: Set of uniform stratified samples on [0, 1]^dimension
    :rtype: STS.samplesU01: ndarray

    :return: STS.strata: Instance of the class SampleMethods.Strata
    :rtype: STS.strata: ndarray

    **Authors:**

    Authors: Michael Shields
    Last modified: 6/7/2018 by Dimitris Giovanis & Michael Shields
    """
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
        """
        Execute stratified sampling

        This is an instance method that runs stratified sampling. It is automatically called when the STS class is
        instantiated.
        """

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
        """Preliminary error checks."""

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
    Define a rectilinear stratification of the n-dimensional unit hypercube [0, 1]^dimension with N strata.

    **Input:**

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

    :param origins: An array of dimension N x n specifying the origins of all strata
                    The origins of the strata are the coordinates of the stratum orthotope nearest the global
                    origin.
                    Example - A 2D stratification with 2 strata in each dimension
                    origins = [[0, 0]
                              [0, 0.5]
                              [0.5, 0]
                              [0.5, 0.5]]
    :type origins: numpy array

    :param widths: An array of dimension N x n specifying the widths of all strata in each dimension
                   Example - A 2D stratification with 2 strata in each dimension
                   widths = [[0.5, 0.5]
                             [0.5, 0.5]
                             [0.5, 0.5]
                             [0.5, 0.5]]
    :type widths: numpy array

    **Attributes:**

    :param Strata.weights: An array of dimension 1 x N containing sample weights.
                    Sample weights are equal to the product of the strata widths (i.e. they are equal to the
                    size of the strata in the [0, 1]^n space.
    :type Strata.weights: numpy array

    **Author:**

    Michael D. Shields
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

        **Input:**

        :param levels: A list of integers that indicate the number of levels of each input design factor.
        :type levels: list

        **Output:**

        :return ff: Full-factorial design matrix.
        :rtype ff: ndarray

        **Author:**

        Michael D. Shields
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
    Generate new samples using adaptive sampling methods, i.e. Refined Stratified Sampling and Gradient
    Enhanced Refined Stratified Sampling.

    **References:**

    1. Michael D. Shields, Kirubel Teferra, Adam Hapij and Raymond P. Daddazio, "Refined Stratified Sampling for
       efficient Monte Carlo based uncertainty quantification", Reliability Engineering & System Safety,
       ISSN: 0951-8320, Vol: 142, Page: 310-325, 2015.
    2. M. D. Shields, "Adaptive Monte Carlo analysis for strongly nonlinear stochastic systems",
       Reliability Engineering & System Safety, ISSN: 0951-8320, Vol: 175, Page: 207-224, 2018.

    **Input:**

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

    **Attributes:**

    :param: RSS.sample_object.samples: Final/expanded samples.
    :type: RSS.sample_object.samples: ndarray

    **Authors:**

    Authors: Mohit S. Chauhan
    Last modified: 01/07/2020 by Mohit S. Chauhan
    """

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
        Execute refined stratified sampling.

        This is an instance method that runs refined stratified sampling. It is automatically called when the RSS class
        is instantiated.

        **Inputs:**

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
        """
        Samples are generated using Gradient Enhanced-Refined Stratified Sampling.
        """
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
        """
        Samples are generated using Refined Stratified Sampling.
        """
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

    def estimate_gradient(self, x, y, xt):
        """
        Estimating gradients with a metamodel (surrogate).

        **Inputs:**

        :param x: Samples in the training data.
        :type x: numpy array

        :param y: Function values evaluated at the samples in the training data.
        :type y: numpy array

        :param xt: Samples where gradients are computed.
        :type xt: numpy array

        **Outputs:**
        :return gr: First-order gradient evaluated at the points 'xt'.
        :rtype gr: numpy array
        """
        from UQpy.Reliability import TaylorSeries
        if type(self.krig_object).__name__ == 'Krig':
            self.krig_object.fit(samples=x, values=y)
            tck = self.krig_object
        elif type(self.krig_object).__name__ == 'GaussianProcessRegressor':
            self.krig_object.fit(x, y)
            tck = self.krig_object.predict
        else:
            from scipy.interpolate import LinearNDInterpolator
            tck = LinearNDInterpolator(x, y, fill_value=0).__call__

        gr = TaylorSeries.gradient(samples=xt, model=tck, dimension=self.dimension, order='first',
                                   df_step=self.step_size, scale=False)
        return gr

    def init_rss(self):
        """Preliminary error checks."""
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
    Generate random samples inside a simplex using uniform probability distribution.

    **References:**

    1. W. N. Edelinga, R. P. Dwightb, P. Cinnellaa, "Simplex-stochastic collocation method with improved
       calability",Journal of Computational Physics, 310:301–328 2016.

    **Input:**

    :param nodes: The vertices of the simplex
    :type nodes: ndarray

    :param nsamples: The number of samples to be generated inside the simplex
    :type nsamples: int

    **Attributes:**

    :return Simplex.samples: New random samples distributed uniformly inside the simplex.
    :rtype Simplex.samples: ndarray

    **Authors:**

    Authors: Dimitris G.Giovanis
    Last modified: 11/28/2018 by Mohit S. Chauhan
    """

    # Authors: Dimitris G.Giovanis
    # Last modified: 11/28/2018 by Mohit S. Chauhan

    def __init__(self, nodes=None, nsamples=1):
        self.nodes = np.atleast_2d(nodes)
        self.nsamples = nsamples
        self.init_sis()
        self.samples = self.run_sis()

    def run_sis(self):
        """
        Generates uniformly distributed random samples inside the simplex.

        This is an instance method that generates samples. It is automatically called when the Simplex class is
        instantiated.

        **Output:**

        :return sample: Random samples
        :rtype sample: numpy array
        """
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
        """Preliminary error checks."""
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
    Generate new samples using different active learning method and properties of kriging surrogate along with
    MCS.

    **References:**

    1. B. Echard, N. Gayton and M. Lemaire, "AK-MCS: An active learning reliability method combining Kriging and
        Monte Carlo Simulation", Structural Safety, Pages 145-154, 2011.

    **Input:**

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

    **Attributes:**

    :param: AKMCS.sample_object.samples: Final/expanded samples.
    :type: AKMCS..sample_object.samples: ndarray

    :param: AKMCS.krig_model: Prediction function for the final surrogate model.
    :type: AKMCS.krig_model: function

    :param: AKMCS.pf: Probability of failure after every iteration of AKMCS. Available as an output only for
                       Reliability Analysis.
    :type: AKMCS.pf: float list

    :param: AKMCS.cov_pf: Covariance of probability of failure after every iteration of AKMCS. Available as an
                           output only for Reliability Analysis.
    :type: AKMCS.pf: float list

    **Authors:**

    Authors: Mohit S. Chauhan
    Last modified: 01/07/2020 by Mohit S. Chauhan
    """

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
        """
        Executes Adaptive Kriging - Monte Carlo Method.

        This is an instance method that check initial sample design an evaluate model at the training points. It is
        automatically called when the STS class is instantiated.
        """

        # If the initial sample design does not exists, run the initial calculations.
        if self.samples is None:
            if self.verbose:
                print('UQpy: AKMCS - Generating the initial sample set using Latin hypercube sampling.')
            self.samples = LHS(dist_name=self.dist_name, dist_params=self.dist_params, nsamples=self.nstart).samples

        if self.verbose:
            print('UQpy: AKMCS - Running the initial sample set using RunModel.')

        # Evaluate model at the training points
        self.run_model_object.run(samples=self.samples)

    def sample(self, samples=None, n_add=1, append_samples=True, nsamples=0, lf=None):
        """
        Iterative procedure is applied to learn samples based on metamodel and learning function, and then metamodel is
        updated based on new samples.

        **Inputs:**

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
    def eigf(self, pop):
        """
        Learns new samples based on Expected Improvement for Global Fit (EIGF) as learning function

        **References:**

        1. J.N Fuhg, "Adaptive surrogate models for parametric studies", Master's Thesis
           (Link: https://arxiv.org/pdf/1905.05345.pdf)

        **Inputs:**
        :param pop: Remaining sample population (new samples are learn from this population)
        :type pop
        """
        if self.kriging == 'UQpy':
            g, sig = self.krig_model(pop, dy=True)
            sig = np.sqrt(sig)
        else:
            g, sig = self.krig_model(pop, return_std=True)
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
    def u(self, pop):
        """
        Learns new samples based on U-function as learning function.

        **References:**

        1. B. Echard, N. Gayton and M. Lemaire, "AK-MCS: An active learning reliability method combining Kriging and
        Monte Carlo Simulation", Structural Safety, Pages 145-154, 2011.

        **Inputs:**

        :param pop: Remaining sample population (new samples are learn from this population)
        :type pop
        """
        if self.kriging == 'UQpy':
            g, sig = self.krig_model(pop, dy=True)
            sig = np.sqrt(sig)
        else:
            g, sig = self.krig_model(pop, return_std=True)
            sig = sig.reshape(sig.size, 1)
        sig[sig == 0.] = 0.00001

        u = abs(g) / sig
        rows = u[:, 0].argsort()[:self.n_add]

        if min(u[:, 0]) >= 2:
            self.indicator = True

        return rows

    # This learning function has not yet been tested.
    def weighted_u(self, pop):
        """
        Learns new samples based on Probability Weighted U-function as learning function.

        **References:**

        1. V.S. Sundar and M.S. Shields, "RELIABILITY ANALYSIS USING ADAPTIVE KRIGING SURROGATES WITH MULTIMODEL
           INFERENCE".

        **Inputs:**

        :param pop: Remaining sample population (new samples are learn from this population)
        :type pop: numpy array
        """
        if self.kriging == 'UQpy':
            g, sig = self.krig_model(pop, dy=True)
            sig = np.sqrt(sig)
        else:
            g, sig = self.krig_model(pop, return_std=True)
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
    def eff(self, pop):
        """
        Learns new samples based on Expected Feasibilty Function (EFF) as learning function.

        **References:**

        1. B.J. Bichon, M.S. Eldred, L.P.Swiler, S. Mahadevan, J.M. McFarland, "Efficient Global Reliability Analysis
           for Nonlinear Implicit Performance Functions", AIAA JOURNAL, Volume 46, 2008.

        **Inputs:**

        :param pop: Remaining sample population (new samples are learn from this population)
        :type pop: numpy array
        """
        if self.kriging == 'UQpy':
            g, sig = self.krig_model(pop, dy=True)
            sig = np.sqrt(sig)
        else:
            g, sig = self.krig_model(pop, return_std=True)
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
    def eif(self, pop):
        """
        Learns new samples based on Expected Improvement Function (EIF) as learning function.

        **References:**

        1. D.R. Jones, M. Schonlau, W.J. Welch, "Efficient Global Optimization of Expensive Black-Box Functions",
           Journal of Global Optimization, Pages 455–492, 1998.

        **Inputs:**

        :param pop: Remaining sample population (new samples are learn from this population)
        :type pop: numpy array
        """

        if self.kriging == 'UQpy':
            g, sig = self.krig_model(pop, dy=True)
            sig = np.sqrt(sig)
        else:
            g, sig = self.krig_model(pop, return_std=True)
            sig = sig.reshape(sig.size, 1)
        sig[sig == 0.] = 0.00001
        fm = min(self.qoi)
        u = (fm - g) * stats.norm.cdf((fm - g) / sig) + sig * stats.norm.pdf((fm - g) / sig)
        rows = u[:, 0].argsort()[(np.size(g) - self.n_add):]

        return rows

    def learning(self):
        """
        Defines the leaning function used to generate new samples.
        """
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

    def init_akmcs(self):
        """Preliminary error checks."""
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


class MCMC:
    """
    Generate samples from arbitrary user-specified probability density function using Markov Chain Monte Carlo.

    This is the parent class for all MCMC algorithms. This parent class only provides the framework for MCMC and cannot
    be used directly for sampling. Sampling is done by calling the child class for the specific MCMC algorithm.


    **Inputs:**

    * **dimension** (`int`):
        A scalar value defining the dimension of target density function. Either `dimension` and `nchains` or `seed`
        must be provided.

    * **pdf_target** ((`list` of) callables):
        Target density function from which to draw random samples. Either `pdf_target` or `log_pdf_target` must be
        provided (the latter should be preferred for better numerical stability).

        If `pdf_target` is a callable, it refers to the joint pdf to sample from, it must take at least one input `x`,
        which are the point(s) at which to evaluate the pdf. Within MCMC the `pdf_target` is evaluated as:
        ``p(x) = pdf_target(x, *args_target)``

        where `x` is a ndarray of shape (nsamples, dimension) and `args_target` are additional positional arguments that
        are provided to MCMC via its `args_target` input.

        If `pdf_target` is a list of callables, it refers to independent marginals to sample from. The marginal in
        dimension `j` is evaluated as: ``p_j(xj) = pdf_target[j](xj, *args_target[j])`` where `x` is a ndarray of shape
        (nsamples, dimension)

    * **log_pdf_target** ((`list` of) callables):
        Logarithm of the target density function from which to draw random samples. Either `pdf_target` or
        `log_pdf_target` must be provided (the latter should be preferred for better numerical stability).

        Same comments as for input `pdf_target`.

    * **args_target** ((`list` of) `tuple`):
        Positional arguments of the pdf / log-pdf target function. See `pdf_target`

    * **seed** (`ndarray`):
        Seed of the Markov chain(s), shape ``(nchains, dimension)``. Default: zeros(`nchains` x `dimension`).

        If `seed` is not provided, both `nchains` and `dimension` must be provided.

    * **nburn** (`int`):
        Length of burn-in - i.e., number of samples at the beginning of the chain to discard (note: no thinning during
        burn-in). Default is 0, no burn-in.

    * **jump** (`int`):
        Thinning parameter, used to reduce correlation between samples. Setting `jump=n` corresponds to	skipping `n-1`
        states between accepted states of the chain. Default is 1 (no thinning).

    * **nchains** (`int`):
        The number of Markov chains to generate. Either `dimension` and `nchains` or `seed` must be provided.

    * **save_log_pdf** (`bool`):
        Boolean that indicates whether to save log-pdf values along with the samples. Default: False

    * **verbose** (`boolean`)
        Set ``verbose = True`` to print status messages to the terminal during execution.

    * **concat_chains** (`bool`):
        Boolean that indicates whether to concatenate the chains after a run, i.e., samples are stored as an `ndarray`
        of shape (nsamples * nchains, dimension) if True, (nsamples, nchains, dimension) if False. Default: True

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.


    **Attributes:**

    * **samples** (`ndarray`)
        Set of MCMC samples following the target distribution, `ndarray` of shape (`nsamples` * `nchains`, `dimension`)
        or (nsamples, nchains, dimension) (see input `concat_chains`).

    * **log_pdf_values** (`ndarray`)
        Values of the log pdf for the accepted samples, `ndarray` of shape (nchains * nsamples,) or (nsamples, nchains)

    * **nsamples** (`list`)
        Total number of samples; The `nsamples` attribute tallies the total number of generated samples. After each
        iteration, it is updated by 1. At the end of the simulation, the `nsamples` attribute equals the user-specified
        value for input `nsamples` given to the child class.

    * **nsamples_per_chain** (`list`)
        Total number of samples per chain; Similar to the attribute `nsamples`, it is updated during iterations as new
        samples are saved.

    * **niterations** (`list`)
        Total number of iterations, updated on-the-fly as the algorithm proceeds. It is related to number of samples as
        niterations=nburn+jump*nsamples_per_chain.

    * **acceptance_rate** (`list`)
        Acceptance ratio of the MCMC chains, computed separately for each chain.

    **Methods:**
    """
    # Last Modified: 10/05/20 by Audrey Olivier

    def __init__(self, dimension=None, pdf_target=None, log_pdf_target=None, args_target=None, seed=None, nburn=0,
                 jump=1, nchains=None, save_log_pdf=False, verbose=False, concat_chains=True, random_state=None):

        if not (isinstance(nburn, int) and nburn >= 0):
            raise TypeError('UQpy: nburn should be an integer >= 0')
        if not (isinstance(jump, int) and jump >= 1):
            raise TypeError('UQpy: jump should be an integer >= 1')
        self.nburn, self.jump = nburn, jump
        self.seed = self._preprocess_seed(seed=seed, dim=dimension, nchains=nchains)
        self.nchains, self.dimension = self.seed.shape

        # Check target pdf
        self.evaluate_log_target, self.evaluate_log_target_marginals = self._preprocess_target(
            pdf_=pdf_target, log_pdf_=log_pdf_target, args=args_target)
        self.save_log_pdf = save_log_pdf
        self.concat_chains = concat_chains
        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')
        self.verbose = verbose

        self.log_pdf_target = log_pdf_target
        self.pdf_target = pdf_target
        self.args_target = args_target

        # Initialize a few more variables
        self.samples = None
        self.log_pdf_values = None
        self.acceptance_rate = [0.] * self.nchains
        self.nsamples, self.nsamples_per_chain = 0, 0
        self.niterations = 0  # total nb of iterations, grows if you call run several times

    def run(self, nsamples=None, nsamples_per_chain=None):
        """
        Run the MCMC algorithm.

        This function samples from the MCMC chains and appends samples to existing ones (if any). This method leverages
        the ``run_iterations`` method that is specific to each algorithm.

        **Inputs:**

        * **nsamples** (`int`):
            Number of samples to generate.

        * **nsamples_per_chain** (`int`)
            Number of samples to generate per chain.

        Either `nsamples` or `nsamples_per_chain` must be provided (not both). Not that if `nsamples` is not a multiple
        of `nchains`, `nsamples` is set to the next largest integer that is a multiple of `nchains`.

        """
        # Initialize the runs: allocate space for the new samples and log pdf values
        final_nsamples, final_nsamples_per_chain, current_state, current_log_pdf = self._initialize_samples(
            nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

        if self.verbose:
            print('UQpy: Running MCMC...')

        # Run nsims iterations of the MCMC algorithm, starting at current_state
        while self.nsamples_per_chain < final_nsamples_per_chain:
            # update the total number of iterations
            self.niterations += 1
            # run iteration
            current_state, current_log_pdf = self.run_one_iteration(current_state, current_log_pdf)
            # Update the chain, only if burn-in is over and the sample is not being jumped over
            # also increase the current number of samples and samples_per_chain
            if self.niterations > self.nburn and (self.niterations - self.nburn) % self.jump == 0:
                self.samples[self.nsamples_per_chain, :, :] = current_state.copy()
                if self.save_log_pdf:
                    self.log_pdf_values[self.nsamples_per_chain, :] = current_log_pdf.copy()
                self.nsamples_per_chain += 1
                self.nsamples += self.nchains

        if self.verbose:
            print('UQpy: MCMC run successfully !')

        # Concatenate chains maybe
        if self.concat_chains:
            self._concatenate_chains()

    def run_one_iteration(self, current_state, current_log_pdf):
        """
        Run one iteration of the MCMC algorithm, starting at `current_state`.

        This method is over-written for each different MCMC algorithm. It must return the new state and associated
        log-pdf, which will be passed as inputs to the ``run_one_iteration`` method at the next iteration.

        **Inputs:**

        * **current_state** (`ndarray`):
            Current state of the chain(s), `ndarray` of shape ``(nchains, dimension)``.

        * **current_log_pdf** (`ndarray`):
            Log-pdf of the current state of the chain(s), `ndarray` of shape ``(nchains, )``.

        **Outputs/Returns:**

        * **new_state** (`ndarray`):
            New state of the chain(s), `ndarray` of shape ``(nchains, dimension)``.

        * **new_log_pdf** (`ndarray`):
            Log-pdf of the new state of the chain(s), `ndarray` of shape ``(nchains, )``.

        """
        return [], []

    ####################################################################################################################
    # Helper functions that can be used by all algorithms
    # Methods update_samples, update_accept_ratio and sample_candidate_from_proposal can be called in the run stage.
    # Methods preprocess_target, preprocess_proposal, check_seed and check_integers can be called in the init stage.

    def _concatenate_chains(self):
        """
        Concatenate chains.

        Utility function that reshapes (in place) attribute samples from (nsamples, nchains, dimension) to
        (nsamples * nchains, dimension), and log_pdf_values from (nsamples, nchains) to (nsamples * nchains, ).

        No input / output.

        """
        self.samples = self.samples.reshape((-1, self.dimension), order='C')
        if self.save_log_pdf:
            self.log_pdf_values = self.log_pdf_values.reshape((-1, ), order='C')
        return None

    def _unconcatenate_chains(self):
        """
        Inverse of concatenate_chains.

        Utility function that reshapes (in place) attribute samples from (nsamples * nchains, dimension) to
        (nsamples, nchains, dimension), and log_pdf_values from (nsamples * nchains) to (nsamples, nchains).

        No input / output.

        """
        self.samples = self.samples.reshape((-1, self.nchains, self.dimension), order='C')
        if self.save_log_pdf:
            self.log_pdf_values = self.log_pdf_values.reshape((-1, self.nchains), order='C')
        return None

    def _initialize_samples(self, nsamples, nsamples_per_chain):
        """
        Initialize necessary attributes and variables before running the chain forward.

        Utility function that allocates space for samples and log likelihood values, initialize sample_index,
        acceptance ratio. If some samples already exist, allocate space to append new samples to the old ones. Computes
        the number of forward iterations nsims to be run (depending on burnin and jump parameters).

        **Inputs:**

        * nchains (int): number of chains run in parallel
        * nsamples (int): number of samples to be generated
        * nsamples_per_chain (int): number of samples to be generated per chain

        **Output/Returns:**

        * nsims (int): Number of iterations to perform
        * current_state (ndarray of shape (nchains, dim)): Current state of the chain to start from.

        """
        if ((nsamples is not None) and (nsamples_per_chain is not None)) or (
                nsamples is None and nsamples_per_chain is None):
            raise ValueError('UQpy: Either nsamples or nsamples_per_chain must be provided (not both)')
        if nsamples_per_chain is not None:
            if not (isinstance(nsamples_per_chain, int) and nsamples_per_chain >= 0):
                raise TypeError('UQpy: nsamples_per_chain must be an integer >= 0.')
            nsamples = int(nsamples_per_chain * self.nchains)
        else:
            if not (isinstance(nsamples, int) and nsamples >= 0):
                raise TypeError('UQpy: nsamples must be an integer >= 0.')
            nsamples_per_chain = int(np.ceil(nsamples / self.nchains))
            nsamples = int(nsamples_per_chain * self.nchains)

        if self.samples is None:    # very first call of run, set current_state as the seed and initialize self.samples
            self.samples = np.zeros((nsamples_per_chain, self.nchains, self.dimension))
            if self.save_log_pdf:
                self.log_pdf_values = np.zeros((nsamples_per_chain, self.nchains))
            current_state = np.zeros_like(self.seed)
            np.copyto(current_state, self.seed)
            current_log_pdf = self.evaluate_log_target(current_state)
            if self.nburn == 0:    # if nburn is 0, save the seed, run one iteration less 
                self.samples[0, :, :] = current_state
                if self.save_log_pdf:
                    self.log_pdf_values[0, :] = current_log_pdf
                self.nsamples_per_chain += 1
                self.nsamples += self.nchains

        else:    # fetch previous samples to start the new run, current state is last saved sample
            if len(self.samples.shape) == 2:   # the chains were previously concatenated
                self._unconcatenate_chains()
            current_state = self.samples[-1]
            current_log_pdf = self.evaluate_log_target(current_state)
            self.samples = np.concatenate(
                [self.samples, np.zeros((nsamples_per_chain, self.nchains, self.dimension))], axis=0)
            if self.save_log_pdf:
                self.log_pdf_values = np.concatenate(
                    [self.log_pdf_values, np.zeros((nsamples_per_chain, self.nchains))], axis=0)

        return nsamples, nsamples_per_chain, current_state, current_log_pdf

    def _update_acceptance_rate(self, new_accept=None):
        """
        Update acceptance rate of the chains.

        Utility function, uses an iterative function to update the acceptance rate of all the chains separately.

        **Inputs:**

        * new_accept (list (length nchains) of bool): indicates whether the current state was accepted (for each chain
          separately).

        """
        self.acceptance_rate = [na / self.niterations + (self.niterations - 1) / self.niterations * a
                                for (na, a) in zip(new_accept, self.acceptance_rate)]

    @staticmethod
    def _preprocess_target(log_pdf_, pdf_, args):
        """
        Preprocess the target pdf inputs.

        Utility function (static method), that transforms the log_pdf, pdf, args inputs into a function that evaluates
        log_pdf_target(x) for a given x. If the target is given as a list of callables (marginal pdfs), the list of
        log margianals is also returned.

        **Inputs:**

        * log_pdf_ ((list of) callables): Log of the target density function from which to draw random samples. Either
          pdf_target or log_pdf_target must be provided.
        * pdf_ ((list of) callables): Target density function from which to draw random samples. Either pdf_target or
          log_pdf_target must be provided.
        * args (tuple): Positional arguments of the pdf target.

        **Output/Returns:**

        * evaluate_log_pdf (callable): Callable that computes the log of the target density function
        * evaluate_log_pdf_marginals (list of callables): List of callables to compute the log pdf of the marginals

        """
        # log_pdf is provided
        if log_pdf_ is not None:
            if callable(log_pdf_):
                if args is None:
                    args = ()
                evaluate_log_pdf = (lambda x: log_pdf_(x, *args))
                evaluate_log_pdf_marginals = None
            elif isinstance(log_pdf_, list) and (all(callable(p) for p in log_pdf_)):
                if args is None:
                    args = [()] * len(log_pdf_)
                if not (isinstance(args, list) and len(args) == len(log_pdf_)):
                    raise ValueError('UQpy: When log_pdf_target is a list, args should be a list (of tuples) of same '
                                     'length.')
                evaluate_log_pdf_marginals = list(
                    map(lambda i: lambda x: log_pdf_[i](x, *args[i]), range(len(log_pdf_))))
                evaluate_log_pdf = (lambda x: np.sum(
                    [log_pdf_[i](x[:, i, np.newaxis], *args[i]) for i in range(len(log_pdf_))]))
            else:
                raise TypeError('UQpy: log_pdf_target must be a callable or list of callables')
        # pdf is provided
        elif pdf_ is not None:
            if callable(pdf_):
                if args is None:
                    args = ()
                evaluate_log_pdf = (lambda x: np.log(np.maximum(pdf_(x, *args), 10 ** (-320) * np.ones((x.shape[0],)))))
                evaluate_log_pdf_marginals = None
            elif isinstance(pdf_, (list, tuple)) and (all(callable(p) for p in pdf_)):
                if args is None:
                    args = [()] * len(pdf_)
                if not (isinstance(args, (list, tuple)) and len(args) == len(pdf_)):
                    raise ValueError('UQpy: When pdf_target is given as a list, args should also be a list of same '
                                     'length.')
                evaluate_log_pdf_marginals = list(
                    map(lambda i: lambda x: np.log(np.maximum(pdf_[i](x, *args[i]),
                                                              10 ** (-320) * np.ones((x.shape[0],)))),
                        range(len(pdf_))
                        ))
                evaluate_log_pdf = (lambda x: np.sum(
                    [np.log(np.maximum(pdf_[i](x[:, i, np.newaxis], *args[i]), 10**(-320)*np.ones((x.shape[0],))))
                     for i in range(len(log_pdf_))]))
            else:
                raise TypeError('UQpy: pdf_target must be a callable or list of callables')
        else:
            raise ValueError('UQpy: log_pdf_target or pdf_target should be provided.')
        return evaluate_log_pdf, evaluate_log_pdf_marginals

    @staticmethod
    def _preprocess_seed(seed, dim, nchains):
        """
        Preprocess input seed.

        Utility function (static method), that checks the dimension of seed, assign [0., 0., ..., 0.] if not provided.

        **Inputs:**

        * seed (ndarray): seed for MCMC
        * dim (int): dimension of target density

        **Output/Returns:**

        * seed (ndarray): seed for MCMC
        * dim (int): dimension of target density

        """
        if seed is None:
            if dim is None or nchains is None:
                raise ValueError('UQpy: Either `seed` or `dimension` and `nchains` must be provided.')
            seed = np.zeros((nchains, dim))
        else:
            seed = np.atleast_1d(seed)
            if len(seed.shape) == 1:
                seed = np.reshape(seed, (1, -1))
            elif len(seed.shape) > 2:
                raise ValueError('UQpy: Input seed should be an array of shape (dimension, ) or (nchains, dimension).')
            if dim is not None and seed.shape[1] != dim:
                raise ValueError('UQpy: Wrong dimensions between seed and dimension.')
            if nchains is not None and seed.shape[0] != nchains:
                raise ValueError('UQpy: The number of chains and the seed shape are inconsistent.')
        return seed

    @staticmethod
    def _check_methods_proposal(proposal):
        """
        Check if proposal has required methods.

        Utility function (static method), that checks that the given proposal distribution has 1) a rvs method and 2) a
        log pdf or pdf method. If a pdf method exists but no log_pdf, the log_pdf methods is added to the proposal
        object. Used in the MH and MMH initializations.

        **Inputs:**

        * proposal (Distribution object): proposal distribution

        """
        if not isinstance(proposal, Distribution):
            raise TypeError('UQpy: Proposal should be a Distribution object')
        if not hasattr(proposal, 'rvs'):
            raise AttributeError('UQpy: The proposal should have an rvs method')
        if not hasattr(proposal, 'log_pdf'):
            if not hasattr(proposal, 'pdf'):
                raise AttributeError('UQpy: The proposal should have a log_pdf or pdf method')
            proposal.log_pdf = lambda x: np.log(np.maximum(proposal.pdf(x), 10 ** (-320) * np.ones((x.shape[0],))))


#################################################################################################################


class MH(MCMC):
    """
    Metropolis-Hastings algorithm

    **References**

    1. Gelman et al., "Bayesian data analysis", Chapman and Hall/CRC, 2013
    2. R.C. Smith, "Uncertainty Quantification - Theory, Implementation and Applications", CS&E, 2014


    **Algorithm-specific inputs:**

    * **proposal** (``Distribution`` object):
        Proposal distribution, must have a log_pdf/pdf and rvs method. Default: standard multivariate normal

    * **proposal_is_symmetric** (`bool`):
        Indicates whether the proposal distribution is symmetric, affects computation of acceptance probability alpha
        Default: False, set to True if default proposal is used

    **Methods:**

    """
    def __init__(self, pdf_target=None, log_pdf_target=None, args_target=None, nburn=0, jump=1, dimension=None,
                 seed=None, save_log_pdf=False, concat_chains=True, nsamples=None, nsamples_per_chain=None,
                 nchains=None, proposal=None, proposal_is_symmetric=False, verbose=False, random_state=None):

        super().__init__(pdf_target=pdf_target, log_pdf_target=log_pdf_target, args_target=args_target,
                         dimension=dimension, seed=seed, nburn=nburn, jump=jump, save_log_pdf=save_log_pdf,
                         concat_chains=concat_chains, verbose=verbose, random_state=random_state, nchains=nchains)

        # Initialize algorithm specific inputs
        self.proposal = proposal
        self.proposal_is_symmetric = proposal_is_symmetric

        if self.proposal is None:
            if self.dimension is None:
                raise ValueError('UQpy: Either input proposal or dimension must be provided.')
            from UQpy.Distributions import JointInd, Normal
            self.proposal = JointInd([Normal()] * self.dimension)
            self.proposal_is_symmetric = True
        else:
            self._check_methods_proposal(self.proposal)

        if self.verbose:
            print('\nUQpy: Initialization of ' + self.__class__.__name__ + ' algorithm complete.')

        # If nsamples is provided, run the algorithm
        if (nsamples is not None) or (nsamples_per_chain is not None):
            self.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

    def run_one_iteration(self, current_state, current_log_pdf):
        """
        Run one iteration of the MCMC chain for MH algorithm, starting at current state - see ``MCMC`` class.
        """
        # Sample candidate
        candidate = current_state + self.proposal.rvs(nsamples=self.nchains, random_state=self.random_state)

        # Compute log_pdf_target of candidate sample
        log_p_candidate = self.evaluate_log_target(candidate)

        # Compute acceptance ratio
        if self.proposal_is_symmetric:  # proposal is symmetric
            log_ratios = log_p_candidate - current_log_pdf
        else:  # If the proposal is non-symmetric, one needs to account for it in computing acceptance ratio
            log_proposal_ratio = self.proposal.log_pdf(candidate - current_state) - \
                                 self.proposal.log_pdf(current_state - candidate)
            log_ratios = log_p_candidate - current_log_pdf - log_proposal_ratio

        # Compare candidate with current sample and decide or not to keep the candidate (loop over nc chains)
        accept_vec = np.zeros((self.nchains,))  # this vector will be used to compute accept_ratio of each chain
        unif_rvs = Uniform().rvs(nsamples=self.nchains, random_state=self.random_state).reshape((-1,))
        for nc, (cand, log_p_cand, r_) in enumerate(zip(candidate, log_p_candidate, log_ratios)):
            accept = np.log(unif_rvs[nc]) < r_
            if accept:
                current_state[nc, :] = cand
                current_log_pdf[nc] = log_p_cand
                accept_vec[nc] = 1.
        # Update the acceptance rate
        self._update_acceptance_rate(accept_vec)

        return current_state, current_log_pdf


####################################################################################################################

class MMH(MCMC):
    """

    Component-wise Modified Metropolis-Hastings algorithm.

    In this algorithm, candidate samples are drawn separately in each dimension, thus the proposal consists of a list
    of 1d distributions. The target pdf can be given as a joint pdf or a list of marginal pdfs in all dimensions. This
    will trigger two different algorithms.

    **References:**

    1. S.-K. Au and J. L. Beck,“Estimation of small failure probabilities in high dimensions by subset simulation,”
       Probabilistic Eng. Mech., vol. 16, no. 4, pp. 263–277, Oct. 2001.

    **Algorithm-specific inputs:**

    * **proposal** ((`list` of) ``Distribution`` object(s)):
        Proposal distribution(s) in one dimension, must have a log_pdf/pdf and rvs method.

        The proposal object may be a list of ``DistributionContinuous1D`` objects or a ``JointInd`` object.
        Default: standard normal

    * **proposal_is_symmetric** ((`list` of) `bool`):
        Indicates whether the proposal distribution is symmetric, affects computation of acceptance probability alpha
        Default: False, set to True if default proposal is used

    **Methods:**

    """
    def __init__(self, pdf_target=None, log_pdf_target=None, args_target=None, nburn=0, jump=1, dimension=None,
                 seed=None, save_log_pdf=False, concat_chains=True, nsamples=None, nsamples_per_chain=None,
                 proposal=None, proposal_is_symmetric=False, verbose=False, random_state=None, nchains=None):

        super().__init__(pdf_target=pdf_target, log_pdf_target=log_pdf_target, args_target=args_target,
                         dimension=dimension, seed=seed, nburn=nburn, jump=jump, save_log_pdf=save_log_pdf,
                         concat_chains=concat_chains, verbose=verbose, random_state=random_state, nchains=nchains)

        # If proposal is not provided: set it as a list of standard gaussians
        from UQpy.Distributions import Normal
        self.proposal = proposal
        self.proposal_is_symmetric = proposal_is_symmetric

        # set default proposal
        if self.proposal is None:
            self.proposal = [Normal(), ] * self.dimension
            self.proposal_is_symmetric = [True, ] * self.dimension
        # Proposal is provided, check it
        else:
            # only one Distribution is provided, check it and transform it to a list
            if isinstance(self.proposal, JointInd):
                self.proposal = [m for m in self.proposal.marginals]
                if len(self.proposal) != self.dimension:
                    raise ValueError('UQpy: Proposal given as a list should be of length dimension')
                [self._check_methods_proposal(p) for p in self.proposal]
            elif not isinstance(self.proposal, list):
                self._check_methods_proposal(self.proposal)
                self.proposal = [self.proposal] * self.dimension
            else:  # a list of proposals is provided
                if len(self.proposal) != self.dimension:
                    raise ValueError('UQpy: Proposal given as a list should be of length dimension')
                [self._check_methods_proposal(p) for p in self.proposal]

        # check the symmetry of proposal, assign False as default
        if isinstance(self.proposal_is_symmetric, bool):
            self.proposal_is_symmetric = [self.proposal_is_symmetric, ] * self.dimension
        elif not (isinstance(self.proposal_is_symmetric, list) and
                  all(isinstance(b_, bool) for b_ in self.proposal_is_symmetric)):
            raise TypeError('UQpy: Proposal_is_symmetric should be a (list of) boolean(s)')

        # check with algo type is used
        if self.evaluate_log_target_marginals is not None:
            self.target_type = 'marginals'
            self.current_log_pdf_marginals = None
        else:
            self.target_type = 'joint'

        if self.verbose:
            print('\nUQpy: Initialization of ' + self.__class__.__name__ + ' algorithm complete.')

        # If nsamples is provided, run the algorithm
        if (nsamples is not None) or (nsamples_per_chain is not None):
            self.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

    def run_one_iteration(self, current_state, current_log_pdf):
        """
        Run one iteration of the MCMC chain for MMH algorithm, starting at current state - see ``MCMC`` class.
        """
        # The target pdf is provided via its marginals
        accept_vec = np.zeros((self.nchains, ))
        if self.target_type == 'marginals':
            # Evaluate the current log_pdf
            if self.current_log_pdf_marginals is None:
                self.current_log_pdf_marginals = [self.evaluate_log_target_marginals[j](current_state[:, j, np.newaxis])
                                                  for j in range(self.dimension)]

            # Sample candidate (independently in each dimension)
            for j in range(self.dimension):
                candidate_j = current_state[:, j, np.newaxis] + self.proposal[j].rvs(
                    nsamples=self.nchains, random_state=self.random_state)

                # Compute log_pdf_target of candidate sample
                log_p_candidate_j = self.evaluate_log_target_marginals[j](candidate_j)

                # Compute acceptance ratio
                if self.proposal_is_symmetric[j]:  # proposal is symmetric
                    log_ratios = log_p_candidate_j - self.current_log_pdf_marginals[j]
                else:  # If the proposal is non-symmetric, one needs to account for it in computing acceptance ratio
                    log_prop_j = self.proposal[j].log_pdf
                    log_proposal_ratio = (log_prop_j(candidate_j - current_state[:, j, np.newaxis]) -
                                          log_prop_j(current_state[:, j, np.newaxis] - candidate_j))
                    log_ratios = log_p_candidate_j - self.current_log_pdf_marginals[j] - log_proposal_ratio

                # Compare candidate with current sample and decide or not to keep the candidate
                unif_rvs = Uniform().rvs(nsamples=self.nchains, random_state=self.random_state).reshape((-1,))
                for nc, (cand, log_p_cand, r_) in enumerate(
                        zip(candidate_j, log_p_candidate_j, log_ratios)):
                    accept = np.log(unif_rvs[nc]) < r_
                    if accept:
                        current_state[nc, j] = cand
                        self.current_log_pdf_marginals[j][nc] = log_p_cand
                        current_log_pdf = np.sum(self.current_log_pdf_marginals)
                        accept_vec[nc] += 1. / self.dimension

        # The target pdf is provided as a joint pdf
        else:
            candidate = np.copy(current_state)
            for j in range(self.dimension):
                candidate_j = current_state[:, j, np.newaxis] + self.proposal[j].rvs(
                    nsamples=self.nchains, random_state=self.random_state)
                candidate[:, j] = candidate_j[:, 0]

                # Compute log_pdf_target of candidate sample
                log_p_candidate = self.evaluate_log_target(candidate)

                # Compare candidate with current sample and decide or not to keep the candidate
                if self.proposal_is_symmetric[j]:  # proposal is symmetric
                    log_ratios = log_p_candidate - current_log_pdf
                else:  # If the proposal is non-symmetric, one needs to account for it in computing acceptance ratio
                    log_prop_j = self.proposal[j].log_pdf
                    log_proposal_ratio = (log_prop_j(candidate_j - current_state[:, j, np.newaxis]) -
                                          log_prop_j(current_state[:, j, np.newaxis] - candidate_j))
                    log_ratios = log_p_candidate - current_log_pdf - log_proposal_ratio
                unif_rvs = Uniform().rvs(nsamples=self.nchains, random_state=self.random_state).reshape((-1,))
                for nc, (cand, log_p_cand, r_) in enumerate(zip(candidate_j, log_p_candidate, log_ratios)):
                    accept = np.log(unif_rvs[nc]) < r_
                    if accept:
                        current_state[nc, j] = cand
                        current_log_pdf[nc] = log_p_cand
                        accept_vec[nc] += 1. / self.dimension
                    else:
                        candidate[:, j] = current_state[:, j]
        # Update the acceptance rate
        self._update_acceptance_rate(accept_vec)
        return current_state, current_log_pdf

####################################################################################################################


class Stretch(MCMC):
    """
    Affine-invariant sampler with Stretch moves, parallel implementation.

    **References:**

    1. J. Goodman and J. Weare, “Ensemble samplers with affine invariance,” Commun. Appl. Math. Comput. Sci.,vol.5,
       no. 1, pp. 65–80, 2010.
    2. Daniel Foreman-Mackey, David W. Hogg, Dustin Lang, and Jonathan Goodman. "emcee: The MCMC Hammer".
       Publications of the Astronomical Society of the Pacific, 125(925):306–312,2013.

    **Algorithm-specific inputs:**

    * **scale** (`float`):
        Scale parameter. Default: 2.

    **Methods:**

    """
    def __init__(self, pdf_target=None, log_pdf_target=None, args_target=None, nburn=0, jump=1, dimension=None,
                 seed=None, save_log_pdf=False, concat_chains=True, nsamples=None, nsamples_per_chain=None,
                 scale=2., verbose=False, random_state=None, nchains=None):

        flag_seed = False
        if seed is None:
            if dimension is None or nchains is None:
                raise ValueError('UQpy: Either `seed` or `dimension` and `nchains` must be provided.')
            flag_seed = True

        super().__init__(pdf_target=pdf_target, log_pdf_target=log_pdf_target, args_target=args_target,
                         dimension=dimension, seed=seed, nburn=nburn, jump=jump, save_log_pdf=save_log_pdf,
                         concat_chains=concat_chains, verbose=verbose, random_state=random_state, nchains=nchains)

        # Check nchains = ensemble size for the Stretch algorithm
        if flag_seed:
            self.seed = Uniform().rvs(nsamples=self.dimension * self.nchains, random_state=self.random_state).reshape(
                (self.nchains, self.dimension)
            )
        if self.nchains < 2:
            raise ValueError('UQpy: For the Stretch algorithm, a seed must be provided with at least two samples.')

        # Check Stretch algorithm inputs: proposal_type and proposal_scale
        self.scale = scale
        if not isinstance(self.scale, float):
            raise TypeError('UQpy: Input scale must be of type float.')

        if self.verbose:
            print('\nUQpy: Initialization of ' + self.__class__.__name__ + ' algorithm complete.')

        # If nsamples is provided, run the algorithm
        if (nsamples is not None) or (nsamples_per_chain is not None):
            self.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

    def run_one_iteration(self, current_state, current_log_pdf):
        """
        Run one iteration of the MCMC chain for Stretch algorithm, starting at current state - see ``MCMC`` class.
        """
        # Start the loop over nsamples - this code uses the parallel version of the stretch algorithm
        all_inds = np.arange(self.nchains)
        inds = all_inds % 2
        accept_vec = np.zeros((self.nchains,))
        # Separate the full ensemble into two sets, use one as a complementary ensemble to the other and vice-versa
        for split in range(2):
            set1 = (inds == split)

            # Get current and complementary sets
            sets = [current_state[inds == j, :] for j in range(2)]
            curr_set, comp_set = sets[split], sets[1 - split]  # current and complementary sets respectively
            ns, nc = len(curr_set), len(comp_set)

            # Sample new state for S1 based on S0 and vice versa
            unif_rvs = Uniform().rvs(nsamples=ns, random_state=self.random_state)
            zz = ((self.scale - 1.) * unif_rvs + 1) ** 2. / self.scale  # sample Z
            factors = (self.dimension - 1.) * np.log(zz)  # compute log(Z ** (d - 1))
            multi_rvs = Multinomial(n=1, p=[1. / nc, ] * nc).rvs(nsamples=ns, random_state=self.random_state)
            rint = np.nonzero(multi_rvs)[1]    # sample X_{j} from complementary set
            candidates = comp_set[rint, :] - (comp_set[rint, :] - curr_set) * np.tile(
                zz, [1, self.dimension])  # new candidates

            # Compute new likelihood, can be done in parallel :)
            logp_candidates = self.evaluate_log_target(candidates)

            # Compute acceptance rate
            unif_rvs = Uniform().rvs(nsamples=len(all_inds[set1]), random_state=self.random_state).reshape((-1,))
            for j, f, lpc, candidate, u_rv in zip(
                    all_inds[set1], factors, logp_candidates, candidates, unif_rvs):
                accept = np.log(u_rv) < f + lpc - current_log_pdf[j]
                if accept:
                    current_state[j] = candidate
                    current_log_pdf[j] = lpc
                    accept_vec[j] += 1.

        # Update the acceptance rate
        self._update_acceptance_rate(accept_vec)
        return current_state, current_log_pdf


####################################################################################################################


class DRAM(MCMC):
    """
    Delayed Rejection Adaptive Metropolis algorithm

    In this algorithm, the proposal density is Gaussian and its covariance C is being updated from samples as
    C = sp * C_sample where C_sample is the sample covariance. Also, the delayed rejection scheme is applied, i.e,
    if a candidate is not accepted another one is generated from the proposal with covariance gamma_2 ** 2 * C.

    **References:**

    1. Heikki Haario, Marko Laine, Antonietta Mira, and Eero Saksman. "DRAM: Efficient adaptive MCMC". Statistics
       and Computing, 16(4):339–354, 2006
    2. R.C. Smith, "Uncertainty Quantification - Theory, Implementation and Applications", CS&E, 2014

    **Algorithm-specific inputs:**

    * **initial_cov** (`ndarray`):
        Initial covariance for the gaussian proposal distribution. Default: I(dim)

    * **k0** (`int`):
        Rate at which covariance is being updated, i.e., every k0 iterations. Default: 100

    * **sp** (`float`):
        Scale parameter for covariance updating. Default: 2.38 ** 2 / dim

    * **gamma_2** (`float`):
        Scale parameter for delayed rejection. Default: 1 / 5

    * **save_cov** (`bool`):
        If True, updated covariance is saved in attribute `adaptive_covariance`. Default: False

    **Methods:**

    """

    def __init__(self, pdf_target=None, log_pdf_target=None, args_target=None, nburn=0, jump=1, dimension=None,
                 seed=None, save_log_pdf=False, concat_chains=True, nsamples=None, nsamples_per_chain=None,
                 initial_covariance=None, k0=100, sp=None, gamma_2=1/5, save_covariance=False, verbose=False,
                 random_state=None, nchains=None):

        super().__init__(pdf_target=pdf_target, log_pdf_target=log_pdf_target, args_target=args_target,
                         dimension=dimension, seed=seed, nburn=nburn, jump=jump, save_log_pdf=save_log_pdf,
                         concat_chains=concat_chains, verbose=verbose, random_state=random_state, nchains=nchains)

        # Check the initial covariance
        self.initial_covariance = initial_covariance
        if self.initial_covariance is None:
            self.initial_covariance = np.eye(self.dimension)
        elif not (isinstance(self.initial_covariance, np.ndarray)
                  and self.initial_covariance == (self.dimension, self.dimension)):
            raise TypeError('UQpy: Input initial_covariance should be a 2D ndarray of shape (dimension, dimension)')

        self.k0 = k0
        self.sp = sp
        if self.sp is None:
            self.sp = 2.38 ** 2 / self.dimension
        self.gamma_2 = gamma_2
        self.save_covariance = save_covariance
        for key, typ in zip(['k0', 'sp', 'gamma_2', 'save_covariance'], [int, float, float, bool]):
            if not isinstance(getattr(self, key), typ):
                raise TypeError('Input ' + key + ' must be of type ' + typ.__name__)

        # initialize the sample mean and sample covariance that you need
        self.current_covariance = np.tile(self.initial_covariance[np.newaxis, ...], (self.nchains, 1, 1))
        self.sample_mean = np.zeros((self.nchains, self.dimension, ))
        self.sample_covariance = np.zeros((self.nchains, self.dimension, self.dimension))
        if self.save_covariance:
            self.adaptive_covariance = [self.current_covariance.copy(), ]

        if self.verbose:
            print('\nUQpy: Initialization of ' + self.__class__.__name__ + ' algorithm complete.')

        # If nsamples is provided, run the algorithm
        if (nsamples is not None) or (nsamples_per_chain is not None):
            self.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

    def run_one_iteration(self, current_state, current_log_pdf):
        """
        Run one iteration of the MCMC chain for DRAM algorithm, starting at current state - see ``MCMC`` class.
        """
        from UQpy.Distributions import MVNormal
        mvp = MVNormal(mean=np.zeros(self.dimension, ), cov=1.)

        # Sample candidate
        candidate = np.zeros_like(current_state)
        for nc, current_cov in enumerate(self.current_covariance):
            mvp.update_params(cov=current_cov)
            candidate[nc, :] = current_state[nc, :] + mvp.rvs(
                nsamples=1, random_state=self.random_state).reshape((self.dimension, ))

        # Compute log_pdf_target of candidate sample
        log_p_candidate = self.evaluate_log_target(candidate)

        # Compare candidate with current sample and decide or not to keep the candidate (loop over nc chains)
        accept_vec = np.zeros((self.nchains, ))
        inds_delayed = []   # indices of chains that will undergo delayed rejection
        unif_rvs = Uniform().rvs(nsamples=self.nchains, random_state=self.random_state).reshape((-1,))
        for nc, (cand, log_p_cand, log_p_curr) in enumerate(zip(candidate, log_p_candidate, current_log_pdf)):
            accept = np.log(unif_rvs[nc]) < log_p_cand - log_p_curr
            if accept:
                current_state[nc, :] = cand
                current_log_pdf[nc] = log_p_cand
                accept_vec[nc] += 1.
            else:    # enter delayed rejection
                inds_delayed.append(nc)    # these indices will enter the delayed rejection part

        # Delayed rejection
        if len(inds_delayed) > 0:   # performed delayed rejection for some chains
            current_states_delayed = np.zeros((len(inds_delayed), self.dimension))
            candidates_delayed = np.zeros((len(inds_delayed), self.dimension))
            candidate2 = np.zeros((len(inds_delayed), self.dimension))
            # Sample other candidates closer to the current one
            for i, nc in enumerate(inds_delayed):
                current_states_delayed[i, :] = current_state[nc, :]
                candidates_delayed[i, :] = candidate[nc, :]
                mvp.update_params(cov=self.gamma_2 ** 2 * self.current_covariance[nc])
                candidate2[i, :] = current_states_delayed[i, :] + mvp.rvs(
                    nsamples=1, random_state=self.random_state).reshape((self.dimension, ))
            # Evaluate their log_target
            log_p_candidate2 = self.evaluate_log_target(candidate2)
            log_prop_cand_cand2 = mvp.log_pdf(candidates_delayed - candidate2)
            log_prop_cand_curr = mvp.log_pdf(candidates_delayed - current_states_delayed)
            # Accept or reject
            unif_rvs = Uniform().rvs(nsamples=len(inds_delayed), random_state=self.random_state).reshape((-1,))
            for (nc, cand2, log_p_cand2, j1, j2, u_rv) in zip(inds_delayed, candidate2, log_p_candidate2,
                                                              log_prop_cand_cand2, log_prop_cand_curr, unif_rvs):
                alpha_cand_cand2 = min(1., np.exp(log_p_candidate[nc] - log_p_cand2))
                alpha_cand_curr = min(1., np.exp(log_p_candidate[nc] - current_log_pdf[nc]))
                log_alpha2 = (log_p_cand2 - current_log_pdf[nc] + j1 - j2 +
                              np.log(max(1. - alpha_cand_cand2, 10 ** (-320))) -
                              np.log(max(1. - alpha_cand_curr, 10 ** (-320))))
                accept = np.log(u_rv) < min(0., log_alpha2)
                if accept:
                    current_state[nc, :] = cand2
                    current_log_pdf[nc] = log_p_cand2
                    accept_vec[nc] += 1.

        # Adaptive part: update the covariance
        for nc in range(self.nchains):
            # update covariance
            self.sample_mean[nc], self.sample_covariance[nc] = self._recursive_update_mean_covariance(
                n=self.niterations, new_sample=current_state[nc, :], previous_mean=self.sample_mean[nc],
                previous_covariance=self.sample_covariance[nc])
            if (self.niterations > 1) and (self.niterations % self.k0 == 0):
                self.current_covariance[nc] = self.sp * self.sample_covariance[nc] + 1e-6 * np.eye(self.dimension)
        if self.save_covariance and ((self.niterations > 1) and (self.niterations % self.k0 == 0)):
            self.adaptive_covariance.append(self.current_covariance.copy())

        # Update the acceptance rate
        self._update_acceptance_rate(accept_vec)
        return current_state, current_log_pdf

    @staticmethod
    def _recursive_update_mean_covariance(n, new_sample, previous_mean, previous_covariance=None):
        """
        Iterative formula to compute a new sample mean and covariance based on previous ones and new sample.

        New covariance is computed only of previous_covariance is provided.

        **Inputs:**

        * n (int): Number of samples used to compute the new mean
        * new_sample (ndarray (dim, )): new sample
        * previous_mean (ndarray (dim, )): Previous sample mean, to be updated with new sample value
        * previous_covariance (ndarray (dim, dim)): Previous sample covariance, to be updated with new sample value

        **Output/Returns:**

        * new_mean (ndarray (dim, )): Updated sample mean
        * new_covariance (ndarray (dim, dim)): Updated sample covariance

        """
        new_mean = (n - 1) / n * previous_mean + 1 / n * new_sample
        if previous_covariance is None:
            return new_mean
        dim = new_sample.size
        if n == 1:
            new_covariance = np.zeros((dim, dim))
        else:
            delta_n = (new_sample - previous_mean).reshape((dim, 1))
            new_covariance = (n - 2) / (n - 1) * previous_covariance + 1 / n * np.matmul(delta_n, delta_n.T)
        return new_mean, new_covariance

####################################################################################################################


class DREAM(MCMC):
    """
    DiffeRential Evolution Adaptive Metropolis algorithm

    **References:**

    1. J.A. Vrugt et al. "Accelerating Markov chain Monte Carlo simulation by differential evolution with
       self-adaptive randomized subspace sampling". International Journal of Nonlinear Sciences and Numerical
       Simulation, 10(3):273–290, 2009.[68]
    2. J.A. Vrugt. "Markov chain Monte Carlo simulation using the DREAM software package: Theory, concepts, and
       MATLAB implementation". Environmental Modelling & Software, 75:273–316, 2016.

    **Algorithm-specific inputs:**

    * **delta** (`int`):
        Jump rate. Default: 3

    * **c** (`float`):
        Differential evolution parameter. Default: 0.1

    * **c_star** (`float`):
        Differential evolution parameter, should be small compared to width of target. Default: 1e-6

    * **n_cr** (`int`):
        Number of crossover probabilities. Default: 3

    * **p_g** (`float`):
        Prob(gamma=1). Default: 0.2

    * **adapt_cr** (`tuple`):
        (iter_max, rate) governs adaptation of crossover probabilities (adapts every rate iterations if iter<iter_max).
        Default: (-1, 1), i.e., no adaptation

    * **check_chains** (`tuple`):
        (iter_max, rate) governs discarding of outlier chains (discard every rate iterations if iter<iter_max).
        Default: (-1, 1), i.e., no check on outlier chains

    **Methods:**

    """

    def __init__(self, pdf_target=None, log_pdf_target=None, args_target=None, nburn=0, jump=1, dimension=None,
                 seed=None, save_log_pdf=False, concat_chains=True, nsamples=None, nsamples_per_chain=None,
                 delta=3, c=0.1, c_star=1e-6, n_cr=3, p_g=0.2, adapt_cr=(-1, 1), check_chains=(-1, 1), verbose=False,
                 random_state=None, nchains=None):

        super().__init__(pdf_target=pdf_target, log_pdf_target=log_pdf_target, args_target=args_target,
                         dimension=dimension, seed=seed, nburn=nburn, jump=jump, save_log_pdf=save_log_pdf,
                         concat_chains=concat_chains, verbose=verbose, random_state=random_state, nchains=nchains)

        # Check nb of chains
        if self.nchains < 2:
            raise ValueError('UQpy: For the DREAM algorithm, a seed must be provided with at least two samples.')

        # Check user-specific algorithms
        self.delta = delta
        self.c = c
        self.c_star = c_star
        self.n_cr = n_cr
        self.p_g = p_g
        self.adapt_cr = adapt_cr
        self.check_chains = check_chains

        for key, typ in zip(['delta', 'c', 'c_star', 'n_cr', 'p_g'], [int, float, float, int, float]):
            if not isinstance(getattr(self, key), typ):
                raise TypeError('Input ' + key + ' must be of type ' + typ.__name__)
        if self.dimension is not None and self.n_cr > self.dimension:
            self.n_cr = self.dimension
        for key in ['adapt_cr', 'check_chains']:
            p = getattr(self, key)
            if not (isinstance(p, tuple) and len(p) == 2 and all(isinstance(i, (int, float)) for i in p)):
                raise TypeError('Inputs ' + key + ' must be a tuple of 2 integers.')
        if (not self.save_log_pdf) and (self.check_chains[0] > 0):
            raise ValueError('UQpy: Input save_log_pdf must be True in order to check outlier chains')

        # Initialize a few other variables
        self.j_ind, self.n_id = np.zeros((self.n_cr,)), np.zeros((self.n_cr,))
        self.cross_prob = np.ones((self.n_cr,)) / self.n_cr

        if self.verbose:
            print('UQpy: Initialization of ' + self.__class__.__name__ + ' algorithm complete.\n')

        # If nsamples is provided, run the algorithm
        if (nsamples is not None) or (nsamples_per_chain is not None):
            self.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

    def run_one_iteration(self, current_state, current_log_pdf):
        """
        Run one iteration of the MCMC chain for DREAM algorithm, starting at current state - see ``MCMC`` class.
        """
        r_diff = np.array([np.setdiff1d(np.arange(self.nchains), j) for j in range(self.nchains)])
        cross = np.arange(1, self.n_cr + 1) / self.n_cr

        # Dynamic part: evolution of chains
        unif_rvs = Uniform().rvs(nsamples=self.nchains * (self.nchains-1),
                                 random_state=self.random_state).reshape((self.nchains - 1, self.nchains))
        draw = np.argsort(unif_rvs, axis=0)
        dx = np.zeros_like(current_state)
        lmda = Uniform(scale=2 * self.c).rvs(nsamples=self.nchains, random_state=self.random_state).reshape((-1, ))
        std_x_tmp = np.std(current_state, axis=0)

        multi_rvs = Multinomial(n=1, p=[1./self.delta, ] * self.delta).rvs(
            nsamples=self.nchains, random_state=self.random_state)
        d_ind = np.nonzero(multi_rvs)[1]
        as_ = [r_diff[j, draw[slice(d_ind[j]), j]] for j in range(self.nchains)]
        bs_ = [r_diff[j, draw[slice(d_ind[j], 2 * d_ind[j], 1), j]] for j in range(self.nchains)]
        multi_rvs = Multinomial(n=1, p=self.cross_prob).rvs(nsamples=self.nchains, random_state=self.random_state)
        id_ = np.nonzero(multi_rvs)[1]
        # id = np.random.choice(self.n_CR, size=(self.nchains, ), replace=True, p=self.pCR)
        z = Uniform().rvs(nsamples=self.nchains * self.dimension,
                          random_state=self.random_state).reshape((self.nchains, self.dimension))
        subset_a = [np.where(z_j < cross[id_j])[0] for (z_j, id_j) in zip(z, id_)]  # subset A of selected dimensions
        d_star = np.array([len(a_j) for a_j in subset_a])
        for j in range(self.nchains):
            if d_star[j] == 0:
                subset_a[j] = np.array([np.argmin(z[j])])
                d_star[j] = 1
        gamma_d = 2.38 / np.sqrt(2 * (d_ind + 1) * d_star)
        g = Binomial(n=1, p=self.p_g).rvs(nsamples=self.nchains, random_state=self.random_state).reshape((-1, ))
        g[g == 0] = gamma_d[g == 0]
        norm_vars = Normal(loc=0., scale=1.).rvs(nsamples=self.nchains ** 2,
                                                 random_state=self.random_state).reshape((self.nchains, self.nchains))
        for j in range(self.nchains):
            for i in subset_a[j]:
                dx[j, i] = self.c_star * norm_vars[j, i] + \
                           (1 + lmda[j]) * g[j] * np.sum(current_state[as_[j], i] - current_state[bs_[j], i])
        candidates = current_state + dx

        # Evaluate log likelihood of candidates
        logp_candidates = self.evaluate_log_target(candidates)

        # Accept or reject
        accept_vec = np.zeros((self.nchains, ))
        unif_rvs = Uniform().rvs(nsamples=self.nchains, random_state=self.random_state).reshape((-1, ))
        for nc, (lpc, candidate, log_p_curr) in enumerate(zip(logp_candidates, candidates, current_log_pdf)):
            accept = np.log(unif_rvs[nc]) < lpc - log_p_curr
            if accept:
                current_state[nc, :] = candidate
                current_log_pdf[nc] = lpc
                accept_vec[nc] = 1.
            else:
                dx[nc, :] = 0
            self.j_ind[id_[nc]] = self.j_ind[id_[nc]] + np.sum((dx[nc, :] / std_x_tmp) ** 2)
            self.n_id[id_[nc]] += 1

        # Save the acceptance rate
        self._update_acceptance_rate(accept_vec)

        # update selection cross prob
        if self.niterations < self.adapt_cr[0] and self.niterations % self.adapt_cr[1] == 0:
            self.cross_prob = self.j_ind / self.n_id
            self.cross_prob /= sum(self.cross_prob)
        # check outlier chains (only if you have saved at least 100 values already)
        if (self.nsamples >= 100) and (self.niterations < self.check_chains[0]) and \
                (self.niterations % self.check_chains[1] == 0):
            self.check_outlier_chains(replace_with_best=True)

        return current_state, current_log_pdf

    def check_outlier_chains(self, replace_with_best=False):
        """
        Check outlier chains in DREAM algorithm.

        This function checks for outlier chains as part of the DREAM algorithm, potentially replacing outlier chains
        (i.e. the samples and log_pdf_values) with 'good' chains. The function does not have any returned output but it
        prints out the number of outlier chains.

        **Inputs:**

        * **replace_with_best** (`bool`):
            Indicates whether to replace outlier chains with the best (most probable) chain. Default: False

        """
        if not self.save_log_pdf:
            raise ValueError('UQpy: Input save_log_pdf must be True in order to check outlier chains')
        start_ = self.nsamples_per_chain // 2
        avgs_logpdf = np.mean(self.log_pdf_values[start_:self.nsamples_per_chain], axis=0)
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
                    self.samples[start_:, j, :] = self.samples[start_:, best_, :].copy()
                    self.log_pdf_values[start_:, j] = self.log_pdf_values[start_:, best_].copy()
                else:
                    print('UQpy: Chain {} is an outlier chain'.format(j))
        if self.verbose and outlier_num > 0:
            print('UQpy: Detected {} outlier chains'.format(outlier_num))


########################################################################################################################
########################################################################################################################
#                                         Importance Sampling
########################################################################################################################

class IS:
    """
    Sample from a user-defined target density using importance sampling.


    **Inputs:**

    * **nsamples** (`int`):
        Number of samples to generate - see ``run`` method. If not `None`, the `run` method is called when the object is
        created. Default is None.

    * **pdf_target** (callable):
        Callable that evaluates the pdf of the target distribution. Either log_pdf_target or pdf_target must be
        specified (the former is preferred).

    * **log_pdf_target** (callable)
        Callable that evaluates the log-pdf of the target distribution. Either log_pdf_target or pdf_target must be
        specified (the former is preferred).

    * **args_target** (`tuple`):
        Positional arguments of the target log_pdf / pdf callable.

    * **proposal** (``Distribution`` object):
        Proposal to sample from. This ``UQpy.Distributions`` object must have an rvs method and a log_pdf (or pdf)
        method.

    * **verbose** (`boolean`)
        Set ``verbose = True`` to print status messages to the terminal during execution.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.


    **Attributes:**

    * **samples** (`ndarray`):
        Set of samples, `ndarray` of shape (nsamples, dim)

    * **unnormalized_log_weights** (`ndarray`)
        Unnormalized log weights, i.e., log_w(x) = log_target(x) - log_proposal(x), `ndarray` of shape (nsamples, )

    * **weights** (`ndarray`):
        Importance weights, weighted so that they sum up to 1, `ndarray` of shape (nsamples, )

    **Methods:**
    """
    # Last Modified: 10/05/2020 by Audrey Olivier
    def __init__(self, nsamples=None, pdf_target=None, log_pdf_target=None, args_target=None,
                 proposal=None, verbose=False, random_state=None):
        # Initialize proposal: it should have an rvs and log pdf or pdf method
        self.proposal = proposal
        if not isinstance(self.proposal, Distribution):
            raise TypeError('UQpy: The proposal should be of type Distribution.')
        if not hasattr(self.proposal, 'rvs'):
            raise AttributeError('UQpy: The proposal should have an rvs method')
        if not hasattr(self.proposal, 'log_pdf'):
            if not hasattr(self.proposal, 'pdf'):
                raise AttributeError('UQpy: The proposal should have a log_pdf or pdf method')
            self.proposal.log_pdf = lambda x: np.log(np.maximum(self.proposal.pdf(x),
                                                                10 ** (-320) * np.ones((x.shape[0],))))

        # Initialize target
        self.evaluate_log_target = self._preprocess_target(log_pdf_=log_pdf_target, pdf_=pdf_target, args=args_target)

        self.verbose = verbose
        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

        # Initialize the samples and weights
        self.samples = None
        self.unnormalized_log_weights = None
        self.weights = None
        self.unweighted_samples = None

        # Run IS if nsamples is provided
        if nsamples is not None and nsamples != 0:
            self.run(nsamples)

    def run(self, nsamples):
        """
        Generate and weight samples.

        This function samples from the proposal and appends samples to existing ones (if any). It then weights the
        samples as log_w_unnormalized) = log(target)-log(proposal).

        **Inputs:**

        * **nsamples** (`int`)
            Number of weighted samples to generate.

        * **Output/Returns:**

        This function has no returns, but it updates the output attributes `samples`, `unnormalized_log_weights` and
        `weights` of the ``IS`` object.
        """

        if self.verbose:
            print('UQpy: Running Importance Sampling...')
        # Sample from proposal
        new_samples = self.proposal.rvs(nsamples=nsamples, random_state=self.random_state)
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
            print('UQpy: Importance Sampling performed successfully')

    # def resample(self, method='multinomial', nsamples=None):
    #     """
    #     Resample to get a set of un-weighted samples that represent the target pdf.
    #
    #     Utility function that creates a set of un-weighted samples from a set of weighted samples. Can be useful for
    #     plotting for instance.
    #
    #     **Inputs:**
    #
    #     * **method** (`str`)
    #         Resampling method, as of V3 only multinomial resampling is supported. Default: 'multinomial'.
    #     * **nsamples** (`int`)
    #         Number of un-weighted samples to generate. Default: None (same number of samples is generated as number of
    #         existing samples).
    #
    #     **Output/Returns:**
    #
    #     * (`ndarray`)
    #         Un-weighted samples that represent the target pdf, `ndarray` of shape (nsamples, dimension)
    #
    #     """
    #     from .Utilities import resample
    #     return resample(self.samples, self.weights, method=method, size=nsamples)

    def resample(self, method='multinomial', nsamples=None):
        """
        Resample to get a set of un-weighted samples that represent the target pdf.

        Utility function that creates a set of un-weighted samples from a set of weighted samples. Can be useful for
        plotting for instance.

        The ``resample`` method is not called automatically when instantiating the ``IS`` class or when invoking its
        ``run`` method.

        **Inputs:**

        * **method** (`str`)
            Resampling method, as of V3 only multinomial resampling is supported. Default: 'multinomial'.
        * **nsamples** (`int`)
            Number of un-weighted samples to generate. Default: None (sets `nsamples` equal to the number of
            existing samples).

        **Output/Returns:**

        The method has no returns, but it creates the attribute following attribute of the ``IS`` object.

        * **unweighted_samples** (`ndarray`)
            Un-weighted samples that represent the target pdf, `ndarray` of shape (nsamples, dimension)

        """

        if nsamples is None:
            nsamples = self.samples.shape[0]
        if method == 'multinomial':
            multinomial_run = np.random.multinomial(nsamples, self.weights, size=1)[0]
            idx = list()
            for j in range(self.samples.shape[0]):
                if multinomial_run[j] > 0:
                    idx.extend([j for _ in range(multinomial_run[j])])
            self.unweighted_samples = self.samples[idx, :]
        else:
            raise ValueError('Exit code: Current available method: multinomial')

    @staticmethod
    def _preprocess_target(log_pdf_, pdf_, args):
        """
        Preprocess the target pdf inputs.

        Utility function (static method), that transforms the log_pdf, pdf, args inputs into a function that evaluates
        log_pdf_target(x) for a given x.

        **Inputs:**

        * log_pdf_ ((list of) callables): Log of the target density function from which to draw random samples. Either
          pdf_target or log_pdf_target must be provided
        * pdf_ ((list of) callables): Target density function from which to draw random samples.
        * args (tuple): Positional arguments of the pdf target

        **Output/Returns:**

        * evaluate_log_pdf (callable): Callable that computes the log of the target density function

        """
        # log_pdf is provided
        if log_pdf_ is not None:
            if callable(log_pdf_):
                if args is None:
                    args = ()
                evaluate_log_pdf = (lambda x: log_pdf_(x, *args))
            else:
                raise TypeError('UQpy: log_pdf_target must be a callable')
        # pdf is provided
        elif pdf_ is not None:
            if callable(pdf_):
                if args is None:
                    args = ()
                evaluate_log_pdf = (lambda x: np.log(np.maximum(pdf_(x, *args), 10 ** (-320) * np.ones((x.shape[0],)))))
            else:
                raise TypeError('UQpy: pdf_target must be a callable')
        else:
            raise ValueError('UQpy: log_pdf_target or pdf_target should be provided.')
        return evaluate_log_pdf
