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
- ``AKMCS``: Class to perform adaptive Kriging Monte Carlo sampling.
- ``STS``: Class to perform stratified sampling.
- ``RSS``: Class to perform refined stratified sampling.
- ``Strata``: Class to perform stratification of the unit hypercube.
- ``Simplex``: Class to uniformly sample from a simplex.
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

            If the ``run`` method is invoked multiple times, the newly generated samples will overwrite the existing
            samples.

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
#                                         Class Strata
########################################################################################################################


class Strata:
    """
    Define a geometric decomposition of the n-dimensional unit hypercube into disjoint and space-filling strata.

    This is the parent class for all spatial stratifications. This parent class only provides the framework for
    stratification and cannot be used directly for the stratification. Stratification is done by calling the child
    class for the desired stratification.


    **Inputs:**

    * **seeds** (`ndarray`)
        Define the seed points for the strata. See specific subclass for definition of the seed points.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **verbose** (`Boolean`):
        A boolean declaring whether to write text to the terminal.


    **Attributes:**

    * **seeds** (`ndarray`)
        Seed points for the strata. See specific subclass for definition of the seed points.

    **Methods:**
    """

    def __init__(self, seeds=None, random_state=None, verbose=False):

        self.seeds = seeds
        self.volume = None
        self.verbose = verbose

        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif self.random_state is None:
            self.random_state = np.random.RandomState()
        elif not isinstance(self.random_state, np.random.RandomState):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

    def stratify(self):

        """
        Perform the stratification of the unit hypercube. It is overwritten by the subclass. This method must exist in
        any subclass of the ``Strata`` class.

        **Outputs/Returns:**

        The method has no returns, but it modifies the relevant attributes of the subclass.

        """

        return None


class RectangularStrata(Strata):
    """
    Define a geometric decomposition of the n-dimensional unit hypercube into disjoint and space-filling
    rectangular strata.

    ``RectangularStrata`` is a child class of the ``Strata`` class

    **Inputs:**

    * **nstrata** (`list` of `int`):
        A list of length `n` defining the number of strata in each of the `n` dimensions. Creates an equal
        stratification with strata widths equal to 1/`n_strata`. The total number of strata, `N`, is the product
        of the terms of `n_strata`.

        Example: `n_strata` = [2, 3, 2] creates a 3-dimensional stratification with:\n
                2 strata in dimension 0 with stratum widths 1/2\n
                3 strata in dimension 1 with stratum widths 1/3\n
                2 strata in dimension 2 with stratum widths 1/2\n

        The user must pass one of `nstrata` OR `input_file` OR `seeds` and `widths`

    * **input_file** (`str`):
        File path to an input file specifying stratum seeds and stratum widths.

        This is typically used to define irregular stratified designs.

        The user must pass one of `n_strata` OR `input_file` OR `seeds` and `widths`

    * **seeds** (`ndarray`):
        An array of dimension `N x n` specifying the seeds of all strata. The seeds of the strata are the
        coordinates of the stratum orthotope nearest the global origin.

        Example: A 2-dimensional stratification with 2 equal strata in each dimension:

            `origins` = [[0, 0], [0, 0.5], [0.5, 0], [0.5, 0.5]]

        The user must pass one of `n_strata` OR `input_file` OR `seeds` and `widths`

    * **widths** (`ndarray`):
        An array of dimension `N x n` specifying the widths of all strata in each dimension

        Example: A 2-dimensional stratification with 2 strata in each dimension

            `widths` = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]

        The user must pass one of `n_strata` OR `input_file` OR `seeds` and `widths`

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **verbose** (`Boolean`):
        A boolean declaring whether to write text to the terminal.


    **Attributes:**

    * **nstrata** (`list` of `int`):
        A list of length `n` defining the number of strata in each of the `n` dimensions. Creates an equal
        stratification with strata widths equal to 1/`n_strata`. The total number of strata, `N`, is the product
        of the terms of `n_strata`.

    * **seeds** (`ndarray`):
        An array of dimension `N x n` specifying the seeds of all strata. The seeds of the strata are the
        coordinates of the stratum orthotope nearest the global origin.

    * **widths** (`ndarray`):
        An array of dimension `N x n` specifying the widths of all strata in each dimension

    * **volume** (`ndarray`):
        An array of dimension `(nstrata, )` containing the volume of each stratum. Stratum volumes are equal to the
        product of the strata widths.

    **Methods:**
    """
    def __init__(self, nstrata=None, input_file=None, seeds=None, widths=None, random_state=None, verbose=False):
        super().__init__(random_state=random_state, seeds=seeds, verbose=verbose)

        self.input_file = input_file
        self.nstrata = nstrata
        self.widths = widths

        self.stratify()

    def stratify(self):
        """
        Performs the rectangular stratification.
        """
        if self.verbose:
            print('UQpy: Creating Rectangular stratification ...')

        if self.nstrata is None:
            if self.input_file is None:
                if self.widths is None or self.seeds is None:
                    raise RuntimeError('UQpy: The strata are not fully defined. Must provide `n_strata`, `input_file`, '
                                       'or `seeds` and `widths`.')

            else:
                # Read the strata from the specified input file
                # See documentation for input file formatting
                array_tmp = np.loadtxt(self.input_file)
                self.seeds = array_tmp[:, 0:array_tmp.shape[1] // 2]
                self.widths = array_tmp[:, array_tmp.shape[1] // 2:]

                # Check to see that the strata are space-filling
                space_fill = np.sum(np.prod(self.widths, 1))
                if 1 - space_fill > 1e-5:
                    raise RuntimeError('UQpy: The stratum design is not space-filling.')
                if 1 - space_fill < -1e-5:
                    raise RuntimeError('UQpy: The stratum design is over-filling.')

        # Define a rectilinear stratification by specifying the number of strata in each dimension via nstrata
        else:
            self.seeds = np.divide(self.fullfact(self.nstrata), self.nstrata)
            self.widths = np.divide(np.ones(self.seeds.shape), self.nstrata)

        self.volume = np.prod(self.widths, axis=1)

        if self.verbose:
            print('UQpy: Rectangular stratification created.')

    @staticmethod
    def fullfact(levels):

        """
        Create a full-factorial design

        Note: This function has been modified from pyDOE, released under BSD License (3-Clause)\n
        Copyright (C) 2012 - 2013 - Michael Baudin\n
        Copyright (C) 2012 - Maria Christopoulou\n
        Copyright (C) 2010 - 2011 - INRIA - Michael Baudin\n
        Copyright (C) 2009 - Yann Collette\n
        Copyright (C) 2009 - CEA - Jean-Marc Martinez\n
        Original source code can be found at:\n
        https://pythonhosted.org/pyDOE/#\n
        or\n
        https://pypi.org/project/pyDOE/\n
        or\n
        https://github.com/tisimst/pyDOE/\n

        **Input:**

        * **levels** (`list`):
            A list of integers that indicate the number of levels of each input design factor.

        **Output:**

        * **ff** (`ndarray`):
            Full-factorial design matrix.
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

    def plot_2d(self):
        """
        Plot the rectangular stratification.

        This is an instance method of the ``RectangularStrata`` class that can be called to plot the boundaries of a
        two-dimensional ``RectangularStrata`` object on :math:`[0, 1]^2`.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig = plt.figure()
        ax = fig.gca()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        for i in range(self.seeds.shape[0]):
            rect1 = patches.Rectangle(self.seeds[i, :], self.widths[i, 0], self.widths[i, 1], linewidth=1,
                                      edgecolor='b', facecolor='none')
            ax.add_patch(rect1)

        return fig


class VoronoiStrata(Strata):
    """
    Define a geometric decomposition of the n-dimensional unit hypercube into disjoint and space-filling
    Voronoi strata.

    ``VoronoiStrata`` is a child class of the ``Strata`` class.

    **Inputs:**

    * **seeds** (`ndarray`):
        An array of dimension `N x n` specifying the seeds of all strata. The seeds of the strata are the
        coordinates of the point inside each stratum that defines the stratum.

        The user must provide `seeds` or `nseeds` and `dimension`

    * **nseeds** (`int`):
        The number of seeds to randomly generate. Seeds are generated by random sampling on the unit hypercube.

        The user must provide `seeds` or `nseeds` and `dimension`

    * **dimension** (`ndarray`):
        The dimension of the unit hypercube in which to generate random seeds. Used only if `nseeds` is provided.

        The user must provide `seeds` or `nseeds` and `dimension`

    * **niters** (`int`)
        Number of iterations to perform to create a Centroidal Voronoi decomposition.

        If `niters = 0`, the Voronoi decomposition is based on the provided or generated seeds.

        If :math:`niters \ge 1`, the seed points are moved to the centroids of the Voronoi cells in each iteration and
        the a new Voronoi decomposition is performed. This process is repeated `niters` times to create a Centroidal
        Voronoi decomposition.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **verbose** (`Boolean`):
        A boolean declaring whether to write text to the terminal.


    **Attributes:**

    * **seeds** (`ndarray`):
        An array of dimension `N x n` containing the seeds of all strata. The seeds of the strata are the
        coordinates of the point inside each stratum that defines the stratum.

        If :math:`niters > 1` the `seeds` attribute will differ from the `seeds` input due to the iterations.

    * **vertices** (`list`)
        A list of the vertices for each Voronoi stratum on the unit hypercube.

    * **voronoi** (`object` of ``scipy.spatial.Voronoi``)
        Defines a Voronoi decomposition of the set of reflected points. When creating the Voronoi decomposition on
        the unit hypercube, the code reflects the points on the unit hypercube across all faces of the unit hypercube.
        This causes the Voronoi decomposition to create edges along the faces of the hypercube.

        This object is not the Voronoi decomposition of the unit hypercube. It is the Voronoi decomposition of all
        points and their reflections from which the unit hypercube is extracted.

        To access the vertices in the unit hypercube, see the attribute `vertices`.

    * **volume** (`ndarray`):
        An array of dimension `(nstrata, )` containing the volume of each Voronoi stratum in the unit hypercube.

    **Methods:**
    """

    def __init__(self, seeds=None, nseeds=None, dimension=None, niters=1, random_state=None, verbose=False):
        super().__init__(random_state=random_state, seeds=seeds, verbose=verbose)

        self.nseeds = nseeds
        self.dimension = dimension
        self.niters = niters
        self.voronoi = None
        self.vertices = []

        if self.seeds is not None:
            if self.nseeds is not None or self.dimension is not None:
                print("UQpy: Ignoring 'nseeds' and 'dimension' attributes because 'seeds' are provided")
            self.dimension = self.seeds.shape[1]

        self.stratify()

    def stratify(self):
        """
        Performs the Voronoi stratification.
        """
        if self.verbose:
            print('UQpy: Creating Voronoi stratification ...')

        initial_seeds = self.seeds
        if self.seeds is None:
            initial_seeds = stats.uniform.rvs(size=[self.nseeds, self.dimension], random_state=self.random_state)

        if self.niters == 0:
            self.voronoi, bounded_regions = self.voronoi_unit_hypercube(initial_seeds)

            cent, vol = [], []
            for region in bounded_regions:
                vertices = self.voronoi.vertices[region + [region[0]], :]
                centroid, volume = self.compute_voronoi_centroid_volume(vertices)
                self.vertices.append(vertices)
                cent.append(centroid[0, :])
                vol.append(volume)

            self.volume = np.asarray(vol)
        else:
            for i in range(self.niters):
                self.voronoi, bounded_regions = self.voronoi_unit_hypercube(initial_seeds)

                cent, vol = [], []
                for region in bounded_regions:
                    vertices = self.voronoi.vertices[region + [region[0]], :]
                    centroid, volume = self.compute_voronoi_centroid_volume(vertices)
                    self.vertices.append(vertices)
                    cent.append(centroid[0, :])
                    vol.append(volume)

                initial_seeds = np.asarray(cent)
                self.volume = np.asarray(vol)

        self.seeds = initial_seeds

        if self.verbose:
            print('UQpy: Voronoi stratification created.')

    @staticmethod
    def voronoi_unit_hypercube(seeds):
        """
        This function reflects the seeds across all faces of the unit hypercube and creates a Voronoi decomposition of
        using all the points and their reflections. This allows a Voronoi decomposition that is bounded on the unit
        hypercube to be extracted.

        **Inputs:**

        * **seeds** (`ndarray`):
            Coordinates of points in the unit hypercube from which to define the Voronoi decomposition.

        **Output/Returns:**

        * **vor** (``scipy.spatial.Voronoi`` object):
            Voronoi decomposition of the complete set of points and their reflections.

        * **bounded_regions** (see `regions` attribute of ``scipy.spatial.Voronoi``)
            Indices of the Voronoi vertices forming each Voronoi region for those regions lying inside the unit
            hypercube.
        """

        from scipy.spatial import Voronoi

        # Mirror the seeds in both low and high directions for each dimension
        bounded_points = seeds
        dimension = seeds.shape[1]
        for i in range(dimension):
            seeds_del = np.delete(bounded_points, i, 1)
            if i == 0:
                points_temp1 = np.hstack([np.atleast_2d(-bounded_points[:, i]).T, seeds_del])
                points_temp2 = np.hstack([np.atleast_2d(2 - bounded_points[:, i]).T, seeds_del])
            elif i == dimension - 1:
                points_temp1 = np.hstack([seeds_del, np.atleast_2d(-bounded_points[:, i]).T])
                points_temp2 = np.hstack([seeds_del, np.atleast_2d(2 - bounded_points[:, i]).T])
            else:
                points_temp1 = np.hstack(
                    [seeds_del[:, :i], np.atleast_2d(-bounded_points[:, i]).T, seeds_del[:, i:]])
                points_temp2 = np.hstack(
                    [seeds_del[:, :i], np.atleast_2d(2 - bounded_points[:, i]).T, seeds_del[:, i:]])
            seeds = np.append(seeds, points_temp1, axis=0)
            seeds = np.append(seeds, points_temp2, axis=0)

        vor = Voronoi(seeds, incremental=True)

        regions = [None] * bounded_points.shape[0]

        for i in range(bounded_points.shape[0]):
            regions[i] = vor.regions[vor.point_region[i]]

        bounded_regions = regions

        return vor, bounded_regions

    @staticmethod
    def compute_voronoi_centroid_volume(vertices):
        """
        This function computes the centroid and volume of a Voronoi cell from its vertices.

        **Inputs:**

        * **vertices** (`ndarray`):
            Coordinates of the vertices that define the Voronoi cell.

        **Output/Returns:**

        * **centroid** (`ndarray`):
            Centroid of the Voronoi cell.

        * **volume** (`ndarray`):
            Volume of the Voronoi cell.
        """

        from scipy.spatial import Delaunay, ConvexHull

        tess = Delaunay(vertices)
        dimension = np.shape(vertices)[1]

        w = np.zeros((tess.nsimplex, 1))
        cent = np.zeros((tess.nsimplex, dimension))
        for i in range(tess.nsimplex):
            ch = ConvexHull(tess.points[tess.simplices[i]])
            w[i] = ch.volume
            cent[i, :] = np.mean(tess.points[tess.simplices[i]], axis=0)

        volume = np.sum(w)
        centroid = np.matmul(np.divide(w, volume).T, cent)

        return centroid, volume


class DelaunayStrata(Strata):
    """
        Define a geometric decomposition of the n-dimensional unit hypercube into disjoint and space-filling
        Delaunay strata of n-dimensional simplexes.

        ``DelaunayStrata`` is a child class of the ``Strata`` class.

        **Inputs:**

        * **seeds** (`ndarray`):
            An array of dimension `N x n` specifying the seeds of all strata. The seeds of the strata are the
            coordinates of the vertices of the Delaunay cells.

            The user must provide `seeds` or `nseeds` and `dimension`

            Note that, if `seeds` does not include all corners of the unit hypercube, they are added.

        * **nseeds** (`int`):
            The number of seeds to randomly generate. Seeds are generated by random sampling on the unit hypercube. In
            addition, the class also adds seed points at all corners of the unit hypercube.

            The user must provide `seeds` or `nseeds` and `dimension`

        * **dimension** (`ndarray`):
            The dimension of the unit hypercube in which to generate random seeds. Used only if `nseeds` is provided.

            The user must provide `seeds` or `nseeds` and `dimension`

        * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
            Random seed used to initialize the pseudo-random number generator. Default is None.

            If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
            object itself can be passed directly.

        * **verbose** (`Boolean`):
            A boolean declaring whether to write text to the terminal.


        **Attributes:**

        * **seeds** (`ndarray`):
            An array of dimension `N x n` containing the seeds of all strata. The seeds of the strata are the
            coordinates of the vertices of the Delaunay cells.

        * **centroids** (`ndarray`)
            A list of the vertices for each Voronoi stratum on the unit hypercube.

        * **delaunay** (`object` of ``scipy.spatial.Delaunay``)
            Defines a Delaunay decomposition of the set of seed points and all corner points.

        * **volume** (`ndarray`):
            An array of dimension `(nstrata, )` containing the volume of each Delaunay stratum in the unit hypercube.

        **Methods:**
        """

    def __init__(self, seeds=None, nseeds=None, dimension=None, random_state=None, verbose=False):
        super().__init__(random_state=random_state, seeds=seeds, verbose=verbose)

        self.nseeds = nseeds
        self.dimension = dimension
        self.delaunay = None
        self.centroids = []

        if self.seeds is not None:
            if self.nseeds is not None or self.dimension is not None:
                print("UQpy: Ignoring 'nseeds' and 'dimension' attributes because 'seeds' are provided")
            self.nseeds, self.dimension = self.seeds.shape[0], self.seeds.shape[1]

        self.stratify()

    def stratify(self):
        import itertools
        from scipy.spatial import Delaunay

        if self.verbose:
            print('UQpy: Creating Delaaunay stratification ...')

        initial_seeds = self.seeds
        if self.seeds is None:
            initial_seeds = stats.uniform.rvs(size=[self.nseeds, self.dimension], random_state=self.random_state)

        # Modify seeds to include corner points of (0,1) space
        corners = list(itertools.product(*zip([0]*self.dimension, [1]*self.dimension)))
        initial_seeds = np.vstack([initial_seeds, corners])
        initial_seeds = np.unique([tuple(row) for row in initial_seeds], axis=0)

        self.delaunay = Delaunay(initial_seeds)
        self.centroids = np.zeros([0, self.dimension])
        self.volume = np.zeros([0])
        count = 0
        for sim in self.delaunay.simplices:  # extract simplices from Delaunay triangulation
            cent, vol = self.compute_delaunay_centroid_volume(self.delaunay.points[sim])
            self.centroids = np.vstack([self.centroids, cent])
            self.volume = np.hstack([self.volume, np.array([vol])])
            count = count + 1

        if self.verbose:
            print('UQpy: Delaunay stratification created.')

    @staticmethod
    def compute_delaunay_centroid_volume(vertices):
        """
        This function computes the centroid and volume of a Delaunay simplex from its vertices.

        **Inputs:**

        * **vertices** (`ndarray`):
            Coordinates of the vertices of the simplex.

        **Output/Returns:**

        * **centroid** (`numpy.ndarray`):
            Centroid of the Delaunay simplex.

        * **volume** (`numpy.ndarray`):
            Volume of the Delaunay simplex.
        """

        from scipy.spatial import ConvexHull

        ch = ConvexHull(vertices)
        volume = ch.volume
        # ch.volume: float = ch.volume
        centroid = np.mean(vertices, axis=0)

        return centroid, volume

########################################################################################################################
########################################################################################################################
#                                         Stratified Sampling  (STS)
########################################################################################################################


class STS:
    """
    Parent class for Stratified Sampling ([9]_).

    This is the parent class for all stratified sampling methods. This parent class only provides the framework for
    stratified sampling and cannot be used directly for the sampling. Sampling is done by calling the child
    class for the desired stratification.

    **Inputs:**

    * **dist_object** ((list of) ``Distribution`` object(s)):
        List of ``Distribution`` objects corresponding to each random variable.

    * **strata_object** (``Strata`` object)
        Defines the stratification of the unit hypercube. This must be provided and must be an object of a ``Strata``
        child class: ``RectangularStrata``, ``VoronoiStrata``, or ``DelaunayStrata``.

    * **nsamples_per_stratum** (`int` or `list`):
        Specifies the number of samples in each stratum. This must be either an integer, in which case an equal number
        of samples are drawn from each stratum, or a list. If it is provided as a list, the length of the list must be
        equal to the number of strata.

        If `nsamples_per_stratum` is provided when the class is defined, the ``run`` method will be executed
        automatically.  If neither `nsamples_per_stratum` or `nsamples` are provided when the class is defined, the user
        must call the ``run`` method to perform stratified sampling.

    * **nsamples** (`int`):
        Specify the total number of samples. If `nsamples` is specified, the samples will be drawn in proportion to
        the volume of the strata. Thus, each stratum will contain :math:`round(V_i*nsamples)` samples.

        If `nsamples` is provided when the class is defined, the ``run`` method will be executed
        automatically.  If neither `nsamples_per_stratum` or `nsamples` are provided when the class is defined, the user
        must call the ``run`` method to perform stratified sampling.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **verbose** (`Boolean`):
        A boolean declaring whether to write text to the terminal.

        Default value: False

    **Attributes:**

    * **samples** (`ndarray`):
        The generated samples following the prescribed distribution.

    * **samplesU01** (`ndarray`)
        The generated samples on the unit hypercube.

    * **weights** (`ndarray`)
        Individual sample weights.

    **Methods:**
    """
    def __init__(self, dist_object, strata_object, nsamples_per_stratum=None, nsamples=None, random_state=None,
                 verbose=False):

        self.verbose = verbose
        self.weights = None
        self.strata_object = strata_object
        self.nsamples_per_stratum = nsamples_per_stratum
        self.nsamples = nsamples
        self.samplesU01, self.samples = None, None

        # Check if a Distribution object is provided.
        from UQpy.Distributions import DistributionContinuous1D, JointInd

        if isinstance(dist_object, list):
            self.dimension = len(dist_object)
            for i in range(len(dist_object)):
                if not isinstance(dist_object[i], DistributionContinuous1D):
                    raise TypeError('UQpy: A DistributionContinuous1D object must be provided.')
        else:
            self.dimension = 1
            if not isinstance(dist_object, (DistributionContinuous1D, JointInd)):
                raise TypeError('UQpy: A DistributionContinuous1D or JointInd object must be provided.')

        self.dist_object = dist_object

        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')
        if self.random_state is None:
            self.random_state = self.strata_object.random_state

        if self.verbose:
            print("UQpy: STS object is created")

        # If nsamples_per_stratum or nsamples is provided, execute run method
        if self.nsamples_per_stratum is not None or self.nsamples is not None:
            self.run(nsamples_per_stratum=self.nsamples_per_stratum, nsamples=self.nsamples)

    def transform_samples(self, samples01):
        """
        Transform samples in the unit hypercube :math:`[0, 1]^n` to the prescribed distribution using the inverse CDF.

        **Inputs:**

        * **samplesU01** (`ndarray`):
            `ndarray` containing the generated samples on [0, 1]^dimension.

        **Outputs:**

        * **samples** (`ndarray`):
            `ndarray` containing the generated samples following the prescribed distribution.
        """

        samples_u_to_x = np.zeros_like(samples01)
        for j in range(0, samples01.shape[1]):
            samples_u_to_x[:, j] = self.dist_object[j].icdf(samples01[:, j])

        self.samples = samples_u_to_x

    def run(self, nsamples_per_stratum=None, nsamples=None):
        """
        Executes stratified sampling.

        This method performs the sampling for each of the child classes by running two methods:
        ``create_samplesu01``, and ``transform_samples``. The ``create_samplesu01`` method is
        unique to each child class and therefore must be overwritten when a new child class is defined. The
        ``transform_samples`` method is common to all stratified sampling classes and is therefore defined by the parent
        class. It does not need to be modified.

        If `nsamples` or `nsamples_per_stratum` is provided when the class is defined, the ``run`` method will be
        executed automatically.  If neither `nsamples_per_stratum` or `nsamples` are provided when the class is defined,
        the user must call the ``run`` method to perform stratified sampling.

        **Input:**

        * **nsamples_per_stratum** (`int` or `list`):
            Specifies the number of samples in each stratum. This must be either an integer, in which case an equal
            number of samples are drawn from each stratum, or a list. If it is provided as a list, the length of the
            list must be equal to the number of strata.

            If `nsamples_per_stratum` is provided when the class is defined, the ``run`` method will be executed
            automatically.  If neither `nsamples_per_stratum` or `nsamples` are provided when the class is defined, the
            user must call the ``run`` method to perform stratified sampling.

        * **nsamples** (`int`):
            Specify the total number of samples. If `nsamples` is specified, the samples will be drawn in proportion to
            the volume of the strata. Thus, each stratum will contain :math:`round(V_i*nsamples)` samples where
            :math:`V_i \le 1` is the volume of stratum `i` in the unit hypercube.

            If `nsamples` is provided when the class is defined, the ``run`` method will be executed
            automatically.  If neither `nsamples_per_stratum` or `nsamples` are provided when the class is defined, the
            user must call the ``run`` method to perform stratified sampling.

        **Outputs:**

        The ``run`` method has no output, although it modifies the `samples`, `samplesu01`, and `weights` attributes.
        """

        # Check inputs of run methods
        self.nsamples_per_stratum = nsamples_per_stratum
        self.nsamples = nsamples
        self._run_checks()

        if self.verbose:
            print("UQpy: Performing Stratified Sampling")

        # Call "create_sampleu01" method and generate samples in  the unit hypercube
        self.create_samplesu01(nsamples_per_stratum, nsamples)

        # Compute inverse cdf of samplesU01
        self.transform_samples(self.samplesU01)

        if self.verbose:
            print("UQpy: Stratified Sampling is completed")

    def _run_checks(self):
        if self.nsamples is not None:
            if not isinstance(self.nsamples, int):
                raise RuntimeError("UQpy: 'nsamples' must be an integer.")
            else:
                if isinstance(self.nsamples_per_stratum, (int, list)):
                    print("UQpy: STS class is executing proportional sampling, thus ignoring "
                          "'nsamples_per_stratum'.")
            self.nsamples_per_stratum = (self.strata_object.volume * self.nsamples).round()

        if self.nsamples_per_stratum is not None:
            if isinstance(self.nsamples_per_stratum, int):
                self.nsamples_per_stratum = [self.nsamples_per_stratum] * self.strata_object.volume.shape[0]
            elif isinstance(self.nsamples_per_stratum, list):
                if len(self.nsamples_per_stratum) != self.strata_object.volume.shape[0]:
                    raise ValueError("UQpy: Length of 'nsamples_per_stratum' must match the number of strata.")
            elif self.nsamples is None:
                raise ValueError("UQpy: 'nsamples_per_stratum' must be an integer or a list.")
        else:
            self.nsamples_per_stratum = [1] * self.strata_object.volume.shape[0]

    # Creating dummy method for create_samplesU01. These methods are overwritten in child classes.
    def create_samplesu01(self, nsamples_per_stratum, nsamples):
        """
        Executes the specific stratified sampling algorithm. This method is overwritten by each child class of ``STS``.

        **Input:**

        * **nsamples_per_stratum** (`int` or `list`):
            Specifies the number of samples in each stratum. This must be either an integer, in which case an equal
            number of samples are drawn from each stratum, or a list. If it is provided as a list, the length of the
            list must be equal to the number of strata.

            Either `nsamples_per_stratum` or `nsamples` must be provided.

        * **nsamples** (`int`):
            Specify the total number of samples. If `nsamples` is specified, the samples will be drawn in proportion to
            the volume of the strata. Thus, each stratum will contain :math:`round(V_i*nsamples)` samples where
            :math:`V_i \le 1` is the volume of stratum `i` in the unit hypercube.

            Either `nsamples_per_stratum` or `nsamples` must be provided.

        **Outputs:**

        The ``create_samplesu01`` method has no output, although it modifies the `samplesu01` and `weights` attributes.
        """
        return None


class RectangularSTS(STS):
    """
    Executes Stratified Sampling using Rectangular Stratification.

    ``RectangularSTS`` is a child class of ``STS``. ``RectangularSTS`` takes in all parameters defined in the parent
    ``STS`` class with differences note below. Only those inputs and attributes that differ from the parent class are
    listed below. See documentation for ``STS`` for additional details.

    **Inputs:**

    * **strata_object** (``RectangularStrata`` object):
        The `strata_object` for ``RectangularSTS`` must be an object of type ``RectangularStrata`` class.

    * **sts_criterion** (`str`):
        Random or Centered samples inside the rectangular strata.
        Options:
        1. 'random' - Samples are drawn randomly within the strata. \n
        2. 'centered' - Samples are drawn at the center of the strata. \n

        Default: 'random'

    **Methods:**

    """
    def __init__(self, dist_object, strata_object, nsamples_per_stratum=None, nsamples=None, sts_criterion="random",
                 verbose=False, random_state=None):
        if not isinstance(strata_object, RectangularStrata):
            raise NotImplementedError("UQpy: strata_object must be an object of RectangularStrata class")

        self.sts_criterion = sts_criterion
        if self.sts_criterion not in ['random', 'centered']:
            raise NotImplementedError("UQpy: Supported sts_criteria: 'random', 'centered'")
        if nsamples is not None:
            if self.sts_criterion == 'centered':
                if nsamples != len(strata_object.volume):
                    raise ValueError("UQpy: 'nsamples' attribute is not consistent with number of seeds for 'centered' "
                                     "sampling")
        if nsamples_per_stratum is not None:
            if self.sts_criterion == "centered":
                nsamples_per_stratum = [1] * strata_object.widths.shape[0]

        super().__init__(dist_object=dist_object, strata_object=strata_object,
                         nsamples_per_stratum=nsamples_per_stratum, nsamples=nsamples, random_state=random_state,
                         verbose=verbose)

    def create_samplesu01(self, nsamples_per_stratum=None, nsamples=None):
        """
        Overwrites the ``create_samplesu01`` method in the parent class to generate samples in rectangular strata on the
        unit hypercube. It has the same inputs and outputs as the ``create_samplesu01`` method in the parent class. See
        the ``STS`` class for additional details.
        """

        samples_in_strata, weights = [], []

        for i in range(self.strata_object.seeds.shape[0]):
            samples_temp = np.zeros([int(self.nsamples_per_stratum[i]), self.strata_object.seeds.shape[1]])
            for j in range(self.strata_object.seeds.shape[1]):
                if self.sts_criterion == "random":
                    samples_temp[:, j] = stats.uniform.rvs(loc=self.strata_object.seeds[i, j],
                                                           scale=self.strata_object.widths[i, j],
                                                           random_state=self.random_state,
                                                           size=int(self.nsamples_per_stratum[i]))
                else:
                    samples_temp[:, j] = self.strata_object.seeds[i, j] + self.strata_object.widths[i, j] / 2.
            samples_in_strata.append(samples_temp)

            if int(self.nsamples_per_stratum[i]) != 0:
                weights.extend(
                    [self.strata_object.volume[i] / self.nsamples_per_stratum[i]] * int(self.nsamples_per_stratum[i]))
            else:
                weights.extend([0] * int(self.nsamples_per_stratum[i]))

        self.weights = np.array(weights)
        self.samplesU01 = np.concatenate(samples_in_strata, axis=0)


class VoronoiSTS(STS):
    """
    Executes Stratified Sampling using Voronoi Stratification.

    ``VoronoiSTS`` is a child class of ``STS``. ``VoronoiSTS`` takes in all parameters defined in the parent
    ``STS`` class with differences note below. Only those inputs and attributes that differ from the parent class are
    listed below. See documentation for ``STS`` for additional details.

    **Inputs:**

    * **strata_object** (``VoronoiStrata`` object):
        The `strata_object` for ``VoronoiSTS`` must be an object of the ``VoronoiStrata`` class.

    **Methods:**

    """
    def __init__(self, dist_object, strata_object, nsamples_per_stratum=None, nsamples=None, random_state=None,
                 verbose=False):
        # Check strata_object
        if not isinstance(strata_object, VoronoiStrata):
            raise NotImplementedError("UQpy: strata_object must be an object of VoronoiStrata class")

        super().__init__(dist_object=dist_object, strata_object=strata_object,
                         nsamples_per_stratum=nsamples_per_stratum, nsamples=nsamples, random_state=random_state,
                         verbose=verbose)

    def create_samplesu01(self, nsamples_per_stratum=None, nsamples=None):
        """
        Overwrites the ``create_samplesu01`` method in the parent class to generate samples in Voronoi strata on the
        unit hypercube. It has the same inputs and outputs as the ``create_samplesu01`` method in the parent class. See
        the ``STS`` class for additional details.
        """
        from scipy.spatial import Delaunay, ConvexHull

        samples_in_strata, weights = list(), list()
        for j in range(len(self.strata_object.vertices)):  # For each bounded region (Voronoi stratification)
            vertices = self.strata_object.vertices[j][:-1, :]
            seed = self.strata_object.seeds[j, :].reshape(1, -1)
            seed_and_vertices = np.concatenate([vertices, seed])

            # Create Dealunay Triangulation using seed and vertices of each stratum
            delaunay_obj = Delaunay(seed_and_vertices)

            # Compute volume of each delaunay
            volume = list()
            for i in range(len(delaunay_obj.vertices)):
                vert = delaunay_obj.vertices[i]
                ch = ConvexHull(seed_and_vertices[vert])
                volume.append(ch.volume)

            temp_prob = np.array(volume) / sum(volume)
            a = list(range(len(delaunay_obj.vertices)))
            for k in range(int(self.nsamples_per_stratum[j])):
                simplex = self.random_state.choice(a, p=temp_prob)

                new_samples = Simplex(nodes=seed_and_vertices[delaunay_obj.vertices[simplex]], nsamples=1,
                                      random_state=self.random_state).samples

                samples_in_strata.append(new_samples)

            if int(self.nsamples_per_stratum[j]) != 0:
                weights.extend(
                    [self.strata_object.volume[j] / self.nsamples_per_stratum[j]] * int(self.nsamples_per_stratum[j]))
            else:
                weights.extend([0] * int(self.nsamples_per_stratum[j]))

        self.weights = weights
        self.samplesU01 = np.concatenate(samples_in_strata, axis=0)


class DelaunaySTS(STS):
    """
    Executes Stratified Sampling using Delaunay Stratification.

    ``DelaunaySTS`` is a child class of ``STS``. ``DelaunaySTS`` takes in all parameters defined in the parent
    ``STS`` class with differences note below. Only those inputs and attributes that differ from the parent class are
    listed below. See documentation for ``STS`` for additional details.

    **Inputs:**

    * **strata_object** (``DelaunayStrata`` object):
        The `strata_object` for ``DelaunaySTS`` must be an object of the ``DelaunayStrata`` class.

    **Methods:**

    """
    def __init__(self, dist_object, strata_object, nsamples_per_stratum=1, nsamples=None, random_state=None,
                 verbose=False):

        if not isinstance(strata_object, DelaunayStrata):
            raise NotImplementedError("UQpy: strata_object must be an object of DelaunayStrata class")

        super().__init__(dist_object=dist_object, strata_object=strata_object,
                         nsamples_per_stratum=nsamples_per_stratum, nsamples=nsamples, random_state=random_state,
                         verbose=verbose)

    def create_samplesu01(self, nsamples_per_stratum=None, nsamples=None):
        """
        Overwrites the ``create_samplesu01`` method in the parent class to generate samples in Delaunay strata on the
        unit hypercube. It has the same inputs and outputs as the ``create_samplesu01`` method in the parent class. See
        the ``STS`` class for additional details.
        """

        samples_in_strata, weights = [], []
        count = 0
        for simplex in self.strata_object.delaunay.simplices:  # extract simplices from Delaunay triangulation
            samples_temp = Simplex(nodes=self.strata_object.delaunay.points[simplex],
                                   nsamples=int(self.nsamples_per_stratum[count]), random_state=self.random_state)
            samples_in_strata.append(samples_temp.samples)
            if int(self.nsamples_per_stratum[count]) != 0:
                weights.extend(
                    [self.strata_object.volume[count] / self.nsamples_per_stratum[count]] * int(
                        self.nsamples_per_stratum[
                            count]))
            else:
                weights.extend([0] * int(self.nsamples_per_stratum[count]))
            count = count + 1

        self.weights = np.array(weights)
        self.samplesU01 = np.concatenate(samples_in_strata, axis=0)


########################################################################################################################
########################################################################################################################
#                                         Refined Stratified Sampling (RSS)
########################################################################################################################

class RSS:
    """
    Parent class for Refined Stratified Sampling [10]_, [11]_.

    This is the parent class for all refined stratified sampling methods. This parent class only provides the
    framework for refined stratified sampling and cannot be used directly for the sampling. Sampling is done by
    calling the child class for the desired algorithm.

    **Inputs:**

    * **sample_object** (``SampleMethods`` object(s)):
        Generally, this must be an object of a ``UQpy.SampleMethods`` class. Each child class of ``RSS`` has it's
        own constraints on which specific types of ``SampleMethods`` it can accept. These are described in the child
        class documentation below.

    * **runmodel_object** (``RunModel`` object):
        A ``RunModel`` object, which is used to evaluate the model.

        `runmodel_object` is optional. If it is provided, the specific ``RSS`` subclass with use it to compute the
        gradient of the model in each stratum for gradient-enhanced refined stratified sampling. If it is not
        provided, the ``RSS`` subclass will default to random stratum refinement.

    * **krig_object** (`class` object):
        A object defining a Kriging surrogate model, this object must have ``fit`` and ``predict`` methods.

        May be an object of the ``UQpy`` ``Kriging`` class or an object of the ``scikit-learn``
        ``GaussianProcessRegressor``

        `krig_object` is only used to compute the gradient in gradient-enhanced refined stratified sampling. It must
        be provided if a `runmodel_object` is provided.

    * **local** (`Boolean`):
        In gradient enhanced refined stratified sampling, the gradient is updated after each new sample is added.
        This parameter is used to determine whether the gradient is updated for every stratum or only locally in the
        strata nearest the refined stratum.

        If `local = True`, gradients are only updated in localized regions around the refined stratum.

        Used only in gradient-enhanced refined stratified sampling.

    * **max_train_size** (`int`):
        In gradient enhanced refined stratified sampling, if `local=True` `max_train_size` specifies the number of
        nearest points at which to update the gradient.

        Used only in gradient-enhanced refined stratified sampling.

    * **step_size** (`float`)
        Defines the size of the step to use for gradient estimation using central difference method.

        Used only in gradient-enhanced refined stratified sampling.

    * **qoi_name** (`dict`):
        Name of the quantity of interest from the `runmodel_object`. If the quantity of interest is a dictionary,
        this is used to convert it to a list

        Used only in gradient-enhanced refined stratified sampling.

    * **n_add** (`int`):
        Number of samples to be added per iteration.

        Default: 1.

    * **nsamples** (`int`):
        Total number of samples to be drawn (including the initial samples).

        If `nsamples` is provided when instantiating the class, the ``run`` method will automatically be called. If
        `nsamples` is not provided, an ``RSS`` subclass can be executed by invoking the ``run`` method and passing
        `nsamples`.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **verbose** (`Boolean`):
        A boolean declaring whether to write text to the terminal.

        Default value: False

    **Attributes:**

    Each of the above inputs are saved as attributes, in addition to the following created attributes.

        * **samples** (`ndarray`):
            The generated stratified samples following the prescribed distribution.

        * **samplesU01** (`ndarray`)
            The generated samples on the unit hypercube.

        * **weights** (`ndarray`)
            Individual sample weights.

        * **strata_object** (Object of ``Strata`` subclass)
            Defines the stratification of the unit hypercube. This is an object of the ``Strata`` subclass
            corresponding to the appropriate strata type.

        **Methods:**
        """
    def __init__(self, sample_object=None, runmodel_object=None, krig_object=None, local=False, max_train_size=None,
                 step_size=0.005, qoi_name=None, n_add=1, nsamples=None, random_state=None, verbose=False):

        # Initialize attributes that are common to all approaches
        self.sample_object = sample_object
        self.runmodel_object = runmodel_object
        self.verbose = verbose
        self.nsamples = nsamples
        self.training_points = self.sample_object.samplesU01
        self.samplesU01 = self.sample_object.samplesU01
        self.samples = self.sample_object.samples
        self.weights = None
        self.dimension = self.samples.shape[1]
        self.n_add = n_add

        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

        if self.runmodel_object is not None:
            if type(self.runmodel_object).__name__ not in ['RunModel']:
                raise NotImplementedError("UQpy Error: runmodel_object must be an object of the RunModel class.")

        if runmodel_object is not None:
            self.local = local
            self.max_train_size = max_train_size
            if krig_object is not None:
                if hasattr(krig_object, 'fit') and hasattr(krig_object, 'predict'):
                    self.krig_object = krig_object
                else:
                    raise NotImplementedError("UQpy Error: krig_object must have 'fit' and 'predict' methods.")
            self.qoi_name = qoi_name
            self.step_size = step_size
            if self.verbose:
                print('UQpy: GE-RSS - Running the initial sample set.')
            self.runmodel_object.run(samples=self.samples)
            if self.verbose:
                print('UQpy: GE-RSS - A RSS class object has been initiated.')
        else:
            if self.verbose:
                print('UQpy: RSS - A RSS class object has been initiated.')

        if self.nsamples is not None:
            if isinstance(self.nsamples, int) and self.nsamples > 0:
                self.run(nsamples=self.nsamples)
            else:
                raise NotImplementedError("UQpy: nsamples msut be a positive integer.")

    def run(self, nsamples):
        """
        Execute the random sampling in the ``RSS`` class.

        The ``run`` method is the function that performs random sampling in any ``RSS`` class. If `nsamples` is
        provided, the ``run`` method is automatically called when the ``RSS`` object is defined. The user may also call
        the ``run`` method directly to generate samples. The ``run`` method of the ``RSS`` class can be invoked many
        times and each time the generated samples are appended to the existing samples.

        The ``run`` method is inherited from the parent class and should not be modified by the subclass. It operates by
        calling a ``run_rss`` method that is uniquely defined for each subclass. All ``RSS`` subclasses must posses a
        ``run_rss`` method as defined below.

        **Input:**

        * **nsamples** (`int`):
            Total number of samples to be drawn.

            If the ``run`` method is invoked multiple times, the newly generated samples will be appended to the
            existing samples.

        **Output/Return:**

        The ``run`` method has no returns, although it creates and/or appends the `samples`, `samplesU01, `weights`, and
        `strata_object` attributes of the ``RSS`` class.
        """
        if isinstance(nsamples, int) and nsamples > 0:
            self.nsamples = nsamples
        else:
            raise RuntimeError("UQpy: nsamples must be a positive integer.")

        if self.nsamples <= self.samples.shape[0]:
            raise NotImplementedError('UQpy Error: The number of requested samples must be larger than the existing '
                                      'sample set.')

        self.run_rss()

    def estimate_gradient(self, x, y, xt):
        """
        Estimating gradients with a Kriging metamodel (surrogate).

        **Inputs:**

        * **x** (`ndarray`):
            Samples in the training data.

        * **y** (`ndarray`):
            Function values evaluated at the samples in the training data.

        * **xt** (`ndarray`):
            Samples where gradients need to be evaluated.

        **Outputs:**

        * **gr** (`ndarray`):
            First-order gradient evaluated at the points 'xt' using central difference.
        """
        if self.krig_object is not None:
            self.krig_object.fit(x, y)
            self.krig_object.nopt = 1
            tck = self.krig_object.predict
        else:
            from scipy.interpolate import LinearNDInterpolator
            tck = LinearNDInterpolator(x, y, fill_value=0).__call__

        gr = gradient(point=xt, runmodel_object=tck, order='first', df_step=self.step_size)
        return gr

    def update_samples(self, new_point):
        # Adding new sample to training points, samplesU01 and samples attributes
        self.training_points = np.vstack([self.training_points, new_point])
        self.samplesU01 = np.vstack([self.samplesU01, new_point])
        new_point_ = np.zeros_like(new_point)
        for k in range(self.dimension):
            new_point_[:, k] = self.sample_object.dist_object[k].icdf(new_point[:, k])
        self.samples = np.vstack([self.samples, new_point_])

    def identify_bins(self, strata_metric, p_):
        bin2break_, p_left = np.array([]), p_
        while np.where(strata_metric == strata_metric.max())[0].shape[0] < p_left:
            t = np.where(strata_metric == strata_metric.max())[0]
            bin2break_ = np.hstack([bin2break_, t])
            strata_metric[t] = 0
            p_left -= t.shape[0]

        tmp = self.random_state.choice(np.where(strata_metric == strata_metric.max())[0], p_left, replace=False)
        bin2break_ = np.hstack([bin2break_, tmp])
        bin2break_ = list(map(int, bin2break_))
        return bin2break_

    def run_rss(self):
        """
        This method is overwritten by each subclass in order to perform the refined stratified sampling.

        This must be an instance method of the class and, although it has no returns it should appropriately modify the
        following attributes of the class: `samples`, `samplesU01`, `weights`, `strata_object`.
        """

        pass


class RectangularRSS(RSS):
    """
    Executes Refined Stratified Sampling using Rectangular Stratification.

    ``RectangularRSS`` is a child class of ``RSS``. ``RectangularRSS`` takes in all parameters defined in the parent
    ``RSS`` class with differences note below. Only those inputs and attributes that differ from the parent class
    are listed below. See documentation for ``RSS`` for additional details.

    **Inputs:**

    * **sample_object** (``RectangularSTS`` object):
        The `sample_object` for ``RectangularRSS`` must be an object of the ``RectangularSTS`` class.

    **Methods:**
    """
    def __init__(self, sample_object=None, runmodel_object=None, krig_object=None, local=False, max_train_size=None,
                 step_size=0.005, qoi_name=None, n_add=1, nsamples=None, random_state=None, verbose=False):

        if not isinstance(sample_object, RectangularSTS):
            raise NotImplementedError("UQpy Error: sample_object must be an object of the RectangularSTS class.")

        self.strata_object = copy.deepcopy(sample_object.strata_object)

        super().__init__(sample_object=sample_object, runmodel_object=runmodel_object, krig_object=krig_object,
                         local=local, max_train_size=max_train_size, step_size=step_size, qoi_name=qoi_name,
                         n_add=n_add, nsamples=nsamples, random_state=random_state, verbose=verbose)

    def run_rss(self):
        """
        Overwrites the ``run_rss`` method in the parent class to perform refined stratified sampling with rectangular
        strata. It is an instance method that does not take any additional input arguments. See
        the ``RSS`` class for additional details.
        """
        if self.runmodel_object is not None:
            self._gerss()
        else:
            self._rss()

        self.weights = self.strata_object.volume

    def _gerss(self):
        """
        This method generates samples using Gradient Enhanced Refined Stratified Sampling.
        """
        if self.verbose:
            print('UQpy: Performing GE-RSS with rectangular stratification...')

        # Initialize the vector of gradients at each training point
        dy_dx = np.zeros((self.nsamples, np.size(self.training_points[1])))

        # Primary loop for adding samples and performing refinement.
        for i in range(self.samples.shape[0], self.nsamples, self.n_add):
            p = min(self.n_add, self.nsamples - i)  # Number of points to add in this iteration

            # If the quantity of interest is a dictionary, convert it to a list
            qoi = [None] * len(self.runmodel_object.qoi_list)
            if type(self.runmodel_object.qoi_list[0]) is dict:
                for j in range(len(self.runmodel_object.qoi_list)):
                    qoi[j] = self.runmodel_object.qoi_list[j][self.qoi_name]
            else:
                qoi = self.runmodel_object.qoi_list

            # ################################
            # --------------------------------
            # 1. Determine the strata to break
            # --------------------------------

            # Compute the gradients at the existing sample points
            if self.max_train_size is None or len(
                    self.training_points) <= self.max_train_size or i == self.samples.shape[0]:
                # Use the entire sample set to train the surrogate model (more expensive option)
                dy_dx[:i] = self.estimate_gradient(np.atleast_2d(self.training_points),
                                                   np.atleast_2d(np.array(qoi)),
                                                   self.strata_object.seeds +
                                                   0.5 * self.strata_object.widths)
            else:
                # Use only max_train_size points to train the surrogate model (more economical option)
                # Find the nearest neighbors to the most recently added point
                from sklearn.neighbors import NearestNeighbors
                knn = NearestNeighbors(n_neighbors=self.max_train_size)
                knn.fit(np.atleast_2d(self.training_points))
                neighbors = knn.kneighbors(np.atleast_2d(self.training_points[-1]), return_distance=False)

                # Recompute the gradient only at the nearest neighbor points.
                dy_dx[neighbors] = self.estimate_gradient(np.squeeze(self.training_points[neighbors]),
                                                          np.array(qoi)[neighbors][0],
                                                          np.squeeze(
                                                              self.strata_object.seeds[neighbors] +
                                                              0.5 * self.strata_object.widths[
                                                                  neighbors]))

            # Define the gradient vector for application of the Delta Method
            dy_dx1 = dy_dx[:i]

            # Estimate the variance within each stratum by assuming a uniform distribution over the stratum.
            # All input variables are independent
            var = (1 / 12) * self.strata_object.widths ** 2

            # Estimate the variance over the stratum by Delta Method
            s = np.zeros([i])
            for j in range(i):
                s[j] = np.sum(dy_dx1[j, :] * var[j, :] * dy_dx1[j, :]) * self.strata_object.volume[j] ** 2

            # 'p' is number of samples to be added in the current iteration
            bin2break = self.identify_bins(strata_metric=s, p_=p)

            # #############################################
            # ---------------------------------------------
            # 2. Update each strata and generate new sample
            # ---------------------------------------------
            new_points = np.zeros([p, self.dimension])
            # Update the strata_object for all new points
            for j in range(p):
                new_points[j, :] = self._update_stratum_and_generate_sample(bin2break[j])

            # ###########################
            # ---------------------------
            # 3. Update sample attributes
            # ---------------------------
            self.update_samples(new_point=new_points)

            # ###############################
            # -------------------------------
            # 4. Execute model at new samples
            # -------------------------------
            self.runmodel_object.run(samples=np.atleast_2d(self.samples[-self.n_add:]), append_samples=True)

            if self.verbose:
                print("Iteration:", i)

    def _rss(self):
        """
        This method generates samples using Refined Stratified Sampling.
        """

        if self.verbose:
            print('UQpy: Performing RSS with rectangular stratification...')

        # Primary loop for adding samples and performing refinement.
        for i in range(self.samples.shape[0], self.nsamples, self.n_add):
            p = min(self.n_add, self.nsamples - i)  # Number of points to add in this iteration
            # ################################
            # --------------------------------
            # 1. Determine the strata to break
            # --------------------------------
            # Estimate the weight corresponding to each stratum
            s = np.zeros(i)
            for j in range(i):
                s[j] = self.strata_object.volume[j] ** 2

            # 'p' is number of samples to be added in the current iteration
            bin2break = self.identify_bins(strata_metric=s, p_=p)

            # #############################################
            # ---------------------------------------------
            # 2. Update each strata and generate new sample
            # ---------------------------------------------
            new_points = np.zeros([p, self.dimension])
            # Update the strata_object for all new points, 'p' is number of samples to be added in the current iteration
            for j in range(p):
                new_points[j, :] = self._update_stratum_and_generate_sample(bin2break[j])

            # ###########################
            # ---------------------------
            # 3. Update sample attributes
            # ---------------------------
            self.update_samples(new_point=new_points)

            if self.verbose:
                print("Iteration:", i)

    def _update_stratum_and_generate_sample(self, bin_):
        # Cut the stratum in the direction of maximum length
        cut_dir_temp = self.strata_object.widths[bin_, :]
        dir2break = np.random.choice(np.argwhere(cut_dir_temp == np.amax(cut_dir_temp))[0])

        # Divide the stratum bin2break in the direction dir2break
        self.strata_object.widths[bin_, dir2break] = self.strata_object.widths[bin_, dir2break] / 2
        self.strata_object.widths = np.vstack([self.strata_object.widths, self.strata_object.widths[bin_, :]])
        self.strata_object.seeds = np.vstack([self.strata_object.seeds, self.strata_object.seeds[bin_, :]])
        # print(self.samplesU01[bin_, dir2break], self.strata_object.seeds[bin_, dir2break] + \
        #       self.strata_object.widths[bin_, dir2break])
        if self.samplesU01[bin_, dir2break] < self.strata_object.seeds[bin_, dir2break] + \
                self.strata_object.widths[bin_, dir2break]:
            self.strata_object.seeds[-1, dir2break] = self.strata_object.seeds[bin_, dir2break] + \
                                                      self.strata_object.widths[bin_, dir2break]
            # print("retain")
        else:
            self.strata_object.seeds[bin_, dir2break] = self.strata_object.seeds[bin_, dir2break] + \
                                                        self.strata_object.widths[bin_, dir2break]

        self.strata_object.volume[bin_] = self.strata_object.volume[bin_] / 2
        self.strata_object.volume = np.append(self.strata_object.volume, self.strata_object.volume[bin_])

        # Add a uniform random sample inside the new stratum
        new = stats.uniform.rvs(loc=self.strata_object.seeds[-1, :], scale=self.strata_object.widths[-1, :],
                                random_state=self.random_state)

        return new


class VoronoiRSS(RSS):
    """
    Executes Refined Stratified Sampling using Voronoi Stratification.

    ``VoronoiRSS`` is a child class of ``RSS``. ``VoronoiRSS`` takes in all parameters defined in the parent
    ``RSS`` class with differences note below. Only those inputs and attributes that differ from the parent class
    are listed below. See documentation for ``RSS`` for additional details.

    **Inputs:**

    * **sample_object** (``SampleMethods`` object):
        The `sample_object` for ``VoronoiRSS`` can be an object of any ``SampleMethods`` class that possesses the
        following attributes: `samples` and `samplesU01`

        This can be any ``SampleMethods`` object because ``VoronoiRSS`` creates its own `strata_object`. It does not use
        a `strata_object` inherited from an ``STS`` object.

    **Methods:**
    """

    def __init__(self, sample_object=None, runmodel_object=None, krig_object=None, local=False, max_train_size=None,
                 step_size=0.005, qoi_name=None, n_add=1, nsamples=None, random_state=None, verbose=False):

        if hasattr(sample_object, 'samplesU01'):
            self.strata_object = VoronoiStrata(seeds=sample_object.samplesU01)

        self.mesh = None
        self.mesh_vertices, self.vertices_in_U01 = [], []
        self.points_to_samplesU01, self.points = [], []

        super().__init__(sample_object=sample_object, runmodel_object=runmodel_object, krig_object=krig_object,
                         local=local, max_train_size=max_train_size, step_size=step_size, qoi_name=qoi_name,
                         n_add=n_add, nsamples=nsamples, random_state=random_state, verbose=verbose)

    def run_rss(self):
        """
        Overwrites the ``run_rss`` method in the parent class to perform refined stratified sampling with Voronoi
        strata. It is an instance method that does not take any additional input arguments. See
        the ``RSS`` class for additional details.
        """
        if self.runmodel_object is not None:
            self._gerss()
        else:
            self._rss()

        self.weights = self.strata_object.volume

    def _gerss(self):
        """
        This method generates samples using Gradient Enhanced Refined Stratified Sampling.
        """
        import math

        # Extract the boundary vertices and use them in the Delaunay triangulation / mesh generation
        self._add_boundary_points_and_construct_delaunay()

        dy_dx_old = 0
        self.mesh.old_vertices = self.mesh.vertices

        # Primary loop for adding samples and performing refinement.
        for i in range(self.samples.shape[0], self.nsamples, self.n_add):
            p = min(self.n_add, self.nsamples - i)  # Number of points to add in this iteration

            # Compute the centroids and the volumes of each simplex cell in the mesh
            self.mesh.centroids = np.zeros([self.mesh.nsimplex, self.dimension])
            self.mesh.volumes = np.zeros([self.mesh.nsimplex, 1])
            from scipy.spatial import qhull, ConvexHull
            for j in range(self.mesh.nsimplex):
                try:
                    ConvexHull(self.points[self.mesh.vertices[j]])
                    self.mesh.centroids[j, :], self.mesh.volumes[j] = \
                        DelaunayStrata.compute_delaunay_centroid_volume(self.points[self.mesh.vertices[j]])
                except qhull.QhullError:
                    self.mesh.centroids[j, :], self.mesh.volumes[j] = np.mean(self.points[self.mesh.vertices[j]]), 0

            # If the quantity of interest is a dictionary, convert it to a list
            qoi = [None] * len(self.runmodel_object.qoi_list)
            if type(self.runmodel_object.qoi_list[0]) is dict:
                for j in range(len(self.runmodel_object.qoi_list)):
                    qoi[j] = self.runmodel_object.qoi_list[j][self.qoi_name]
            else:
                qoi = self.runmodel_object.qoi_list

            # ################################
            # --------------------------------
            # 1. Determine the strata to break
            # --------------------------------

            # Compute the gradients at the existing sample points
            if self.max_train_size is None or len(self.training_points) <= self.max_train_size or \
                    i == self.samples.shape[0]:
                # Use the entire sample set to train the surrogate model (more expensive option)
                dy_dx = self.estimate_gradient(np.atleast_2d(self.training_points), qoi, self.mesh.centroids)
            else:
                # Use only max_train_size points to train the surrogate model (more economical option)
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
                knn.fit(np.atleast_2d(self.samplesU01))
                neighbors = knn.kneighbors(np.atleast_2d(self.samplesU01[-1]), return_distance=False)

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
                        if all(np.isin(self.vertices_in_U01, np.hstack([neighbors, np.atleast_2d(10 ** 18)]))):
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
                dy_dx[update_array, :] = self.estimate_gradient(np.squeeze(self.samplesU01[neighbors]),
                                                                np.atleast_2d(np.array(qoi)[neighbors]),
                                                                self.mesh.centroids[update_array])

            # Determine the simplex to break and draw a new sample

            # Estimate the variance over each simplex by Delta Method. Moments of the simplices are computed using
            # Eq. (19) from the following reference:
            # Good, I.J. and Gaskins, R.A. (1971). The Centroid Method of Numerical Integration. Numerische
            #       Mathematik. 16: 343--359.
            var = np.zeros((self.mesh.nsimplex, self.dimension))
            s = np.zeros(self.mesh.nsimplex)
            for j in range(self.mesh.nsimplex):
                for k in range(self.dimension):
                    std = np.std(self.points[self.mesh.vertices[j]][:, k])
                    var[j, k] = (self.mesh.volumes[j] * math.factorial(self.dimension) /
                                 math.factorial(self.dimension + 2)) * (self.dimension * std ** 2)
                s[j] = np.sum(dy_dx[j, :] * var[j, :] * dy_dx[j, :]) * (self.mesh.volumes[j] ** 2)
            dy_dx_old = dy_dx

            # 'p' is number of samples to be added in the current iteration
            bin2add = self.identify_bins(strata_metric=s, p_=p)

            # Create 'p' sub-simplex within the simplex with maximum variance
            new_points = np.zeros([p, self.dimension])
            for j in range(p):
                new_points[j, :] = self._generate_sample(bin2add[j])

            # ###########################
            # ---------------------------
            # 2. Update sample attributes
            # ---------------------------
            self.update_samples(new_point=new_points)

            # ###########################
            # ---------------------------
            # 3. Update strata attributes
            # ---------------------------
            self._update_strata(new_point=new_points)

            # ###############################
            # -------------------------------
            # 4. Execute model at new samples
            # -------------------------------
            self.runmodel_object.run(samples=self.samples[-self.n_add:])

            if self.verbose:
                print("Iteration:", i)

    def _rss(self):
        """
        This method generates samples using Refined Stratified Sampling.
        """

        # Extract the boundary vertices and use them in the Delaunay triangulation / mesh generation
        self._add_boundary_points_and_construct_delaunay()

        # Primary loop for adding samples and performing refinement.
        for i in range(self.samples.shape[0], self.nsamples, self.n_add):
            p = min(self.n_add, self.nsamples - i)  # Number of points to add in this iteration

            # ################################
            # --------------------------------
            # 1. Determine the strata to break
            # --------------------------------

            # Compute the centroids and the volumes of each simplex cell in the mesh
            self.mesh.centroids = np.zeros([self.mesh.nsimplex, self.dimension])
            self.mesh.volumes = np.zeros([self.mesh.nsimplex, 1])
            from scipy.spatial import qhull, ConvexHull
            for j in range(self.mesh.nsimplex):
                try:
                    ConvexHull(self.points[self.mesh.vertices[j]])
                    self.mesh.centroids[j, :], self.mesh.volumes[j] = \
                        DelaunayStrata.compute_delaunay_centroid_volume(self.points[self.mesh.vertices[j]])
                except qhull.QhullError:
                    self.mesh.centroids[j, :], self.mesh.volumes[j] = np.mean(self.points[self.mesh.vertices[j]]), 0

            # Determine the simplex to break and draw a new sample
            s = np.zeros(self.mesh.nsimplex)
            for j in range(self.mesh.nsimplex):
                s[j] = self.mesh.volumes[j] ** 2

            # 'p' is number of samples to be added in the current iteration
            bin2add = self.identify_bins(strata_metric=s, p_=p)

            # Create 'p' sub-simplex within the simplex with maximum variance
            new_points = np.zeros([p, self.dimension])
            for j in range(p):
                new_points[j, :] = self._generate_sample(bin2add[j])

            # ###########################
            # ---------------------------
            # 2. Update sample attributes
            # ---------------------------
            self.update_samples(new_point=new_points)

            # ###########################
            # ---------------------------
            # 3. Update strata attributes
            # ---------------------------
            self._update_strata(new_point=new_points)

            if self.verbose:
                print("Iteration:", i)

    def _generate_sample(self, bin_):
        """
        This method create a subsimplex inside a Dealaunay Triangle and generate a random sample inside it using
        Simplex class.


        **Input:**

        * **bin_** (`int or float`):
            Index of delaunay triangle.


        **Outputt:**

        * **new** (`ndarray`):
            An array of new sample.

        """
        import itertools
        tmp_vertices = self.points[self.mesh.simplices[int(bin_), :]]
        col_one = np.array(list(itertools.combinations(np.arange(self.dimension + 1), self.dimension)))
        self.mesh.sub_simplex = np.zeros_like(tmp_vertices)  # node: an array containing mid-point of edges
        for m in range(self.dimension + 1):
            self.mesh.sub_simplex[m, :] = np.sum(tmp_vertices[col_one[m] - 1, :], 0) / self.dimension

        # Using the Simplex class to generate a new sample in the sub-simplex
        new = Simplex(nodes=self.mesh.sub_simplex, nsamples=1, random_state=self.random_state).samples
        return new

    def _update_strata(self, new_point):
        """
        This method update the `mesh` and `strata_object` attributes of RSS class for each iteration.


        **Inputs:**

        * **new_point** (`ndarray`):
            An array of new samples generated at current iteration.
        """
        i_ = self.samples.shape[0]
        p_ = new_point.shape[0]
        # Update the matrices to have recognize the new point
        self.points_to_samplesU01 = np.hstack([self.points_to_samplesU01, np.arange(i_, i_ + p_)])
        self.mesh.old_vertices = self.mesh.vertices

        # Update the Delaunay triangulation mesh to include the new point.
        self.mesh.add_points(new_point)
        self.points = getattr(self.mesh, 'points')
        self.mesh_vertices = np.vstack([self.mesh_vertices, new_point])

        # Compute the strata weights.
        self.strata_object.voronoi, bounded_regions = VoronoiStrata.voronoi_unit_hypercube(self.samplesU01)

        self.strata_object.centroids = []
        self.strata_object.volume = []
        for region in bounded_regions:
            vertices = self.strata_object.voronoi.vertices[region + [region[0]]]
            centroid, volume = VoronoiStrata.compute_voronoi_centroid_volume(vertices)
            self.strata_object.centroids.append(centroid[0, :])
            self.strata_object.volume.append(volume)

    def _add_boundary_points_and_construct_delaunay(self):
        """
        This method add the corners of [0, 1]^dimension hypercube to the existing samples, which are used to construct a
        Delaunay Triangulation.
        """
        from scipy.spatial.qhull import Delaunay

        self.mesh_vertices = self.training_points.copy()
        self.points_to_samplesU01 = np.arange(0, self.training_points.shape[0])
        for i in range(np.shape(self.strata_object.voronoi.vertices)[0]):
            if any(np.logical_and(self.strata_object.voronoi.vertices[i, :] >= -1e-10,
                                  self.strata_object.voronoi.vertices[i, :] <= 1e-10)) or \
                    any(np.logical_and(self.strata_object.voronoi.vertices[i, :] >= 1 - 1e-10,
                                       self.strata_object.voronoi.vertices[i, :] <= 1 + 1e-10)):
                self.mesh_vertices = np.vstack(
                    [self.mesh_vertices, self.strata_object.voronoi.vertices[i, :]])
                self.points_to_samplesU01 = np.hstack([np.array([-1]), self.points_to_samplesU01, ])

        # Define the simplex mesh to be used for gradient estimation and sampling
        self.mesh = Delaunay(self.mesh_vertices, furthest_site=False, incremental=True, qhull_options=None)
        self.points = getattr(self.mesh, 'points')

########################################################################################################################
########################################################################################################################
#                                        Generating random samples inside a Simplex
########################################################################################################################


class Simplex:
    """
    Generate uniform random samples inside an n-dimensional simplex.


    **Inputs:**

    * **nodes** (`ndarray` or `list`):
        The vertices of the simplex.

    * **nsamples** (`int`):
        The number of samples to be generated inside the simplex.

        If `nsamples` is provided when the object is defined, the ``run`` method will be called automatically. If
        `nsamples` is not provided when the object is defined, the user must invoke the ``run`` method and specify
        `nsamples`.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    **Attributes:**

    * **samples** (`ndarray`):
        New random samples distributed uniformly inside the simplex.

    **Methods:**

    """

    def __init__(self, nodes=None, nsamples=None, random_state=None):
        self.nodes = np.atleast_2d(nodes)
        self.nsamples = nsamples

        if self.nodes.shape[0] != self.nodes.shape[1] + 1:
            raise NotImplementedError("UQpy: Size of simplex (nodes) is not consistent.")

        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

        if nsamples is not None:
            if self.nsamples <= 0 or type(self.nsamples).__name__ != 'int':
                raise NotImplementedError("UQpy: Number of samples to be generated 'nsamples' should be a positive "
                                          "integer.")
            self.samples = self.run(nsamples=nsamples)

    def run(self, nsamples):
        """
        Execute the random sampling in the ``Simplex`` class.

        The ``run`` method is the function that performs random sampling in the ``Simplex`` class. If `nsamples` is
        provided called when the ``Simplex`` object is defined, the ``run`` method is automatically. The user may also
        call the ``run`` method directly to generate samples. The ``run`` method of the ``Simplex`` class can be invoked
        many times and each time the generated samples are appended to the existing samples.

        **Input:**

        * **nsamples** (`int`):
            Number of samples to be generated inside the simplex.

            If the ``run`` method is invoked multiple times, the newly generated samples will be appended to the
            existing samples.

        **Output/Return:**

        The ``run`` method has no returns, although it creates and/or appends the `samples` attribute of the ``Simplex``
        class.

        """
        self.nsamples = nsamples
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
                    r[j] = stats.uniform.rvs(loc=0, scale=1, random_state=self.random_state) ** (1 / (dimension - j))
                d = np.cumprod(r)
                r_ = np.hstack((1, d))
                sample[i, :] = np.dot(ad, r_)
        else:
            a = min(self.nodes)
            b = max(self.nodes)
            sample = a + (b - a) * stats.uniform.rvs(size=[self.nsamples, dimension], random_state=self.random_state)
        return sample


########################################################################################################################
########################################################################################################################
#                                  Adaptive Kriging-Monte Carlo Simulation (AK-MCS)
########################################################################################################################
class AKMCS:
    """
    Adaptively sample for construction of a Kriging surrogate for different objectives including reliability,
    optimization, and global fit.


    **Inputs:**

    * **dist_object** ((list of) ``Distribution`` object(s)):
        List of ``Distribution`` objects corresponding to each random variable.

    * **runmodel_object** (``RunModel`` object):
        A ``RunModel`` object, which is used to evaluate the model.

    * **samples** (`ndarray`):
        The initial samples at which to evaluate the model.

        Either `samples` or `nstart` must be provided.

    * **krig_object** (`class` object):
        A Kriging surrogate model, this object must have ``fit`` and ``predict`` methods.

        May be an object of the ``UQpy`` ``Kriging`` class or an object of the ``scikit-learn``
        ``GaussianProcessRegressor``

    * **nsamples** (`int`):
        Total number of samples to be drawn (including the initial samples).

        If `nsamples` is provided when instantiating the class, the ``run`` method will automatically be called. If
        `nsamples` is not provided, ``AKMCS`` can be executed by invoking the ``run`` method and passing `nsamples`.

    * **nlearn** (`int`):
        Number of samples generated for evaluation of the learning function. Samples for the learning set are drawn
        using ``LHS``.

    * **nstart** (`int`):
        Number of initial samples, randomly generated using ``LHS``.

        Either `samples` or `nstart` must be provided.

    * **qoi_name** (`dict`):
        Name of the quantity of interest. If the quantity of interest is a dictionary, this is used to convert it to
        a list

    * **learning_function** (`str` or `function`):
        Learning function used as the selection criteria to identify new samples.

        Built-in options:
                    1. 'U' - U-function \n
                    2. 'EFF' - Expected Feasibility Function \n
                    3. 'Weighted-U' - Weighted-U function \n
                    4. 'EIF' - Expected Improvement Function \n
                    5. 'EGIF' - Expected Global Improvement Fit \n

        `learning_function` may also be passed as a user-defined callable function. This function must accept a Kriging
        surrogate model object with ``fit`` and ``predict`` methods, the set of learning points at which to evaluate the
        learning function, and it may also take an arbitrary number of additional parameters that are passed to
        ``AKMCS`` as `**kwargs`.

    * **n_add** (`int`):
            Number of samples to be added per iteration.

            Default: 1.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **verbose** (`Boolean`):
        A boolean declaring whether to write text to the terminal.

        Default value: False.

    * **kwargs**
        Used to pass parameters to `learning_function`.

        For built-in `learning_functions`, see the requisite inputs in the method list below.

        For user-defined `learning_functions`, these will be defined by the requisite inputs to the user-defined method.


    **Attributes:**

    * **samples** (`ndarray`):
        `ndarray` containing the samples at which the model is evaluated.

    * **lf_values** (`list`)
        The learning function evaluated at new sample points.


    **Methods:**

    """

    def __init__(self, dist_object, runmodel_object, krig_object, samples=None, nsamples=None, nlearn=10000,
                 nstart=None, qoi_name=None, learning_function='U', n_add=1, random_state=None, verbose=False,
                 **kwargs):

        # Initialize the internal variables of the class.
        self.runmodel_object = runmodel_object
        self.samples = np.array(samples)
        self.nlearn = nlearn
        self.nstart = nstart
        self.verbose = verbose
        self.qoi_name = qoi_name

        self.learning_function = learning_function
        self.learning_set = None
        self.dist_object = dist_object
        self.nsamples = nsamples

        self.moments = None
        self.n_add = n_add
        self.indicator = False
        self.pf = []
        self.cov_pf = []
        self.dimension = 0
        self.qoi = None
        self.krig_model = None
        self.kwargs = kwargs

        # Initialize and run preliminary error checks.
        if self.samples is not None:
            self.dimension = np.shape(self.samples)[1]
        else:
            self.dimension = len(self.dist_object)

        if type(self.learning_function).__name__ == 'function':
            self.learning_function = self.learning_function
        elif self.learning_function not in ['EFF', 'U', 'Weighted-U', 'EIF', 'EIGF']:
            raise NotImplementedError("UQpy Error: The provided learning function is not recognized.")
        elif self.learning_function == 'EIGF':
            self.learning_function = self.eigf
        elif self.learning_function == 'EIF':
            if 'eif_stop' not in self.kwargs:
                self.kwargs['eif_stop'] = 0.01
            self.learning_function = self.eif
        elif self.learning_function == 'U':
            if 'u_stop' not in self.kwargs:
                self.kwargs['u_stop'] = 2
            self.learning_function = self.u
        elif self.learning_function == 'Weighted-U':
            if 'u_stop' not in self.kwargs:
                self.kwargs['u_stop'] = 2
            self.learning_function = self.weighted_u
        else:
            if 'a' not in self.kwargs:
                self.kwargs['a'] = 0
            if 'epsilon' not in self.kwargs:
                self.kwargs['epsilon'] = 2
            if 'eff_stop' not in self.kwargs:
                self.kwargs['u_stop'] = 0.001
            self.learning_function = self.eff

        from UQpy.Distributions import DistributionContinuous1D, JointInd

        if isinstance(dist_object, list):
            for i in range(len(dist_object)):
                if not isinstance(dist_object[i], DistributionContinuous1D):
                    raise TypeError('UQpy: A DistributionContinuous1D object must be provided.')
        else:
            if not isinstance(dist_object, (DistributionContinuous1D, JointInd)):
                raise TypeError('UQpy: A DistributionContinuous1D or JointInd object must be provided.')

        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

        if hasattr(krig_object, 'fit') and hasattr(krig_object, 'predict'):
            self.krig_object = krig_object
        else:
            raise NotImplementedError("UQpy: krig_object must have 'fit' and 'predict' methods.")

        # If the initial sample design does not exists, run the initial calculations.
        if self.samples is None:
            if self.nstart is None:
                NotImplementedError("UQpy: User should provide either 'samples' or 'nstart' value.")
            if self.verbose:
                print('UQpy: AKMCS - Generating the initial sample set using Latin hypercube sampling.')
            self.samples = LHS(dist_object=self.dist_object, nsamples=self.nstart, random_state=random_state).samples

        if self.verbose:
            print('UQpy: AKMCS - Running the initial sample set using RunModel.')

        # Evaluate model at the training points
        if len(self.runmodel_object.qoi_list) == 0:
            self.runmodel_object.run(samples=self.samples)
        else:
            if len(self.runmodel_object.qoi_list) != self.samples.shape[0]:
                raise NotImplementedError("UQpy: There should be no model evaluation or Number of samples and model "
                                          "evaluation in RunModel object should be same.")

        if self.nsamples is not None:
            if self.nsamples <= 0 or type(self.nsamples).__name__ != 'int':
                raise NotImplementedError("UQpy: Number of samples to be generated 'nsamples' should be a positive "
                                          "integer.")
            self.run(nsamples=self.nsamples)

    def run(self, nsamples, samples=None, append_samples=True):
        """
        Execute the ``AKMCS`` learning iterations.

        The ``run`` method is the function that performs iterations in the ``AKMCS`` class. If `nsamples` is
        provided when defining the ``AKMCS`` object, the ``run`` method is automatically called. The user may also
        call the ``run`` method directly to generate samples. The ``run`` method of the ``AKMCS`` class can be invoked
        many times.

        **Inputs:**

        * **nsamples** (`int`):
            Total number of samples to be drawn (including the initial samples).

        * **samples** (`ndarray`):
            Samples at which to evaluate the model.

        * **append_samples** (`boolean`)
            Append new samples and model evaluations to the existing samples and model evaluations.

            If ``append_samples = False``, all previous samples and the corresponding quantities of interest from their
            model evaluations are deleted.

            If ``append_samples = True``, samples and their resulting quantities of interest are appended to the
            existing ones.

        **Output/Returns:**

        The ``run`` method has no returns, although it creates and/or appends the `samples` attribute of the
        ``AKMCS`` class.

        """

        self.nsamples = nsamples

        if samples is not None:
            # New samples are appended to existing samples, if append_samples is TRUE
            if append_samples:
                self.samples = np.vstack([self.samples, samples])
            else:
                self.samples = samples
                self.runmodel_object.qoi_list = []

            if self.verbose:
                print('UQpy: AKMCS - Evaluating the model at the sample set using RunModel.')

            self.runmodel_object.run(samples=samples, append_samples=append_samples)

        if self.verbose:
            print('UQpy: Performing AK-MCS design...')

        # If the quantity of interest is a dictionary, convert it to a list
        self.qoi = [None] * len(self.runmodel_object.qoi_list)
        if type(self.runmodel_object.qoi_list[0]) is dict:
            for j in range(len(self.runmodel_object.qoi_list)):
                self.qoi[j] = self.runmodel_object.qoi_list[j][self.qoi_name]
        else:
            self.qoi = self.runmodel_object.qoi_list

        # Train the initial Kriging model.
        self.krig_object.fit(self.samples, self.qoi)
        self.krig_model = self.krig_object.predict

        # kwargs = {"n_add": self.n_add, "parameters": self.kwargs, "samples": self.samples, "qoi": self.qoi,
        #           "dist_object": self.dist_object}

        # ---------------------------------------------
        # Primary loop for learning and adding samples.
        # ---------------------------------------------

        for i in range(self.samples.shape[0], self.nsamples):
            # Initialize the population of samples at which to evaluate the learning function and from which to draw
            # in the sampling.

            lhs = LHS(dist_object=self.dist_object, nsamples=self.nlearn, random_state=self.random_state)
            self.learning_set = lhs.samples.copy()

            # Find all of the points in the population that have not already been integrated into the training set
            rest_pop = np.array([x for x in self.learning_set.tolist() if x not in self.samples.tolist()])

            # Apply the learning function to identify the new point to run the model.

            # new_point, lf, ind = self.learning_function(self.krig_model, rest_pop, **kwargs)
            new_point, lf, ind = self.learning_function(self.krig_model, rest_pop, n_add=self.n_add,
                                                        parameters=self.kwargs, samples=self.samples, qoi=self.qoi,
                                                        dist_object=self.dist_object)

            # Add the new points to the training set and to the sample set.
            self.samples = np.vstack([self.samples, np.atleast_2d(new_point)])

            # Run the model at the new points
            self.runmodel_object.run(samples=new_point, append_samples=True)

            # If the quantity of interest is a dictionary, convert it to a list
            self.qoi = [None] * len(self.runmodel_object.qoi_list)
            if type(self.runmodel_object.qoi_list[0]) is dict:
                for j in range(len(self.runmodel_object.qoi_list)):
                    self.qoi[j] = self.runmodel_object.qoi_list[j][self.qoi_name]
            else:
                self.qoi = self.runmodel_object.qoi_list

            # Retrain the surrogate model
            self.krig_object.fit(self.samples, self.qoi, nopt=1)
            self.krig_model = self.krig_object.predict

            # Exit the loop, if error criteria is satisfied
            if ind:
                print("UQpy: Learning stops at iteration: ", i)
                break

            if self.verbose:
                print("Iteration:", i)

        if self.verbose:
            print('UQpy: AKMCS complete')

    # ------------------
    # LEARNING FUNCTIONS
    # ------------------
    @staticmethod
    def eigf(surr, pop, **kwargs):
        """
        Expected Improvement for Global Fit (EIGF) learning function. See [7]_ for a detailed explanation.


        **Inputs:**

        * **surr** (`class` object):
            A Kriging surrogate model, this object must have a ``predict`` method as defined in `krig_object` parameter.

        * **pop** (`ndarray`):
            An array of samples defining the learning set at which points the EIGF is evaluated

        * **n_add** (`int`):
            Number of samples to be added per iteration.

            Default: 1.

        * **parameters** (`dictionary`)
            Dictionary containing all necessary parameters and the stopping criterion for the learning function. For
            ``EIGF``, this dictionary is empty as no stopping criterion is specified.

        * **samples** (`ndarray`):
            The initial samples at which to evaluate the model.

        * **qoi** (`list`):
            A list, which contaains the model evaluations.

        * **dist_object** ((list of) ``Distribution`` object(s)):
            List of ``Distribution`` objects corresponding to each random variable.


        **Output/Returns:**

        * **new_samples** (`ndarray`):
            Samples selected for model evaluation.

        * **indicator** (`boolean`):
            Indicator for stopping criteria.

            `indicator = True` specifies that the stopping criterion has been met and the AKMCS.run method stops.

        * **eigf_lf** (`ndarray`)
            EIGF learning function evaluated at the new sample points.

        """
        samples = kwargs['samples']
        qoi = kwargs['qoi']
        n_add = kwargs['n_add']

        g, sig = surr(pop, True)

        # Remove the inconsistency in the shape of 'g' and 'sig' array
        g = g.reshape([pop.shape[0], 1])
        sig = sig.reshape([pop.shape[0], 1])

        # Evaluation of the learning function
        # First, find the nearest neighbor in the training set for each point in the population.
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(np.atleast_2d(samples))
        neighbors = knn.kneighbors(np.atleast_2d(pop), return_distance=False)

        # noinspection PyTypeChecker
        qoi_array = np.array([qoi[x] for x in np.squeeze(neighbors)])

        # Compute the learning function at every point in the population.
        u = np.square(g - qoi_array) + np.square(sig)
        rows = u[:, 0].argsort()[(np.size(g) - n_add):]

        indicator = False
        new_samples = pop[rows, :]
        eigf_lf = u[rows, :]

        return new_samples, eigf_lf, indicator

    @staticmethod
    def u(surr, pop, **kwargs):
        """
        U-function for reliability analysis. See [3] for a detailed explanation.


        **Inputs:**

        * **surr** (`class` object):
            A Kriging surrogate model, this object must have a ``predict`` method as defined in `krig_object` parameter.

        * **pop** (`ndarray`):
            An array of samples defining the learning set at which points the U-function is evaluated

        * **n_add** (`int`):
            Number of samples to be added per iteration.

            Default: 1.

        * **parameters** (`dictionary`)
            Dictionary containing all necessary parameters and the stopping criterion for the learning function. Here
            this includes the parameter `u_stop`.

        * **samples** (`ndarray`):
            The initial samples at which to evaluate the model.

        * **qoi** (`list`):
            A list, which contaains the model evaluations.

        * **dist_object** ((list of) ``Distribution`` object(s)):
            List of ``Distribution`` objects corresponding to each random variable.


        **Output/Returns:**

        * **new_samples** (`ndarray`):
            Samples selected for model evaluation.

        * **indicator** (`boolean`):
            Indicator for stopping criteria.

            `indicator = True` specifies that the stopping criterion has been met and the AKMCS.run method stops.

        * **u_lf** (`ndarray`)
            U learning function evaluated at the new sample points.

        """
        n_add = kwargs['n_add']
        parameters = kwargs['parameters']

        g, sig = surr(pop, True)

        # Remove the inconsistency in the shape of 'g' and 'sig' array
        g = g.reshape([pop.shape[0], 1])
        sig = sig.reshape([pop.shape[0], 1])

        u = abs(g) / sig
        rows = u[:, 0].argsort()[:n_add]

        indicator = False
        if min(u[:, 0]) >= parameters['u_stop']:
            indicator = True

        new_samples = pop[rows, :]
        u_lf = u[rows, 0]
        return new_samples, u_lf, indicator

    @staticmethod
    def weighted_u(surr, pop, **kwargs):
        """
        Probability Weighted U-function for reliability analysis. See [5]_ for a detailed explanation.


        **Inputs:**

        * **surr** (`class` object):
            A Kriging surrogate model, this object must have a ``predict`` method as defined in `krig_object` parameter.

        * **pop** (`ndarray`):
            An array of samples defining the learning set at which points the weighted U-function is evaluated

        * **n_add** (`int`):
            Number of samples to be added per iteration.

            Default: 1.

        * **parameters** (`dictionary`)
            Dictionary containing all necessary parameters and the stopping criterion for the learning function. Here
            this includes the parameter `u_stop`.

        * **samples** (`ndarray`):
            The initial samples at which to evaluate the model.

        * **qoi** (`list`):
            A list, which contaains the model evaluations.

        * **dist_object** ((list of) ``Distribution`` object(s)):
            List of ``Distribution`` objects corresponding to each random variable.

        **Output/Returns:**

        * **new_samples** (`ndarray`):
            Samples selected for model evaluation.

        * **w_lf** (`ndarray`)
            Weighted U learning function evaluated at the new sample points.

        * **indicator** (`boolean`):
            Indicator for stopping criteria.

            `indicator = True` specifies that the stopping criterion has been met and the AKMCS.run method stops.

        """
        n_add = kwargs['n_add']
        parameters = kwargs['parameters']
        samples = kwargs['samples']
        dist_object = kwargs['dist_object']

        g, sig = surr(pop, True)

        # Remove the inconsistency in the shape of 'g' and 'sig' array
        g = g.reshape([pop.shape[0], 1])
        sig = sig.reshape([pop.shape[0], 1])

        u = abs(g) / sig
        p1, p2 = np.ones([pop.shape[0], pop.shape[1]]), np.ones([samples.shape[0], pop.shape[1]])
        for j in range(samples.shape[1]):
            p1[:, j] = dist_object[j].pdf(np.atleast_2d(pop[:, j]).T)
            p2[:, j] = dist_object[j].pdf(np.atleast_2d(samples[:, j]).T)

        p1 = p1.prod(1).reshape(u.size, 1)
        max_p = max(p2.prod(1))
        u_ = u * ((max_p - p1) / max_p)
        rows = u_[:, 0].argsort()[:n_add]

        indicator = False
        if min(u[:, 0]) >= parameters['weighted_u_stop']:
            indicator = True

        new_samples = pop[rows, :]
        w_lf = u_[rows, :]
        return new_samples, w_lf, indicator

    @staticmethod
    def eff(surr, pop, **kwargs):
        """
        Expected Feasibility Function (EFF) for reliability analysis, see [6]_ for a detailed explanation.


        **Inputs:**

        * **surr** (`class` object):
            A Kriging surrogate model, this object must have a ``predict`` method as defined in `krig_object` parameter.

        * **pop** (`ndarray`):
            An array of samples defining the learning set at which points the EFF is evaluated

        * **n_add** (`int`):
            Number of samples to be added per iteration.

            Default: 1.

        * **parameters** (`dictionary`)
            Dictionary containing all necessary parameters and the stopping criterion for the learning function. Here
            these include `a`, `epsilon`, and `eff_stop`.

        * **samples** (`ndarray`):
            The initial samples at which to evaluate the model.

        * **qoi** (`list`):
            A list, which contaains the model evaluations.

        * **dist_object** ((list of) ``Distribution`` object(s)):
            List of ``Distribution`` objects corresponding to each random variable.


        **Output/Returns:**

        * **new_samples** (`ndarray`):
            Samples selected for model evaluation.

        * **indicator** (`boolean`):
            Indicator for stopping criteria.

            `indicator = True` specifies that the stopping criterion has been met and the AKMCS.run method stops.

        * **eff_lf** (`ndarray`)
            EFF learning function evaluated at the new sample points.

        """
        n_add = kwargs['n_add']
        parameters = kwargs['parameters']

        g, sig = surr(pop, True)

        # Remove the inconsistency in the shape of 'g' and 'sig' array
        g = g.reshape([pop.shape[0], 1])
        sig = sig.reshape([pop.shape[0], 1])
        # Reliability threshold: a_ = 0
        # EGRA method: epsilon = 2*sigma(x)
        a_, ep = parameters['eff_a'], parameters['eff_epsilon']*sig
        t1 = (a_ - g) / sig
        t2 = (a_ - ep - g) / sig
        t3 = (a_ + ep - g) / sig
        eff = (g - a_) * (2 * stats.norm.cdf(t1) - stats.norm.cdf(t2) - stats.norm.cdf(t3))
        eff += -sig * (2 * stats.norm.pdf(t1) - stats.norm.pdf(t2) - stats.norm.pdf(t3))
        eff += ep * (stats.norm.cdf(t3) - stats.norm.cdf(t2))
        rows = eff[:, 0].argsort()[-n_add:]

        indicator = False
        if max(eff[:, 0]) <= parameters['eff_stop']:
            indicator = True

        new_samples = pop[rows, :]
        eff_lf = eff[rows, :]
        return new_samples, eff_lf, indicator

    @staticmethod
    def eif(surr, pop, **kwargs):
        """
        Expected Improvement Function (EIF) for Efficient Global Optimization (EFO). See [4]_ for a detailed
        explanation.


        **Inputs:**

        * **surr** (`class` object):
            A Kriging surrogate model, this object must have a ``predict`` method as defined in `krig_object` parameter.

        * **pop** (`ndarray`):
            An array of samples defining the learning set at which points the EIF is evaluated

        * **n_add** (`int`):
            Number of samples to be added per iteration.

            Default: 1.

        * **parameters** (`dictionary`)
            Dictionary containing all necessary parameters and the stopping criterion for the learning function. Here
            this includes the parameter `eif_stop`.

        * **samples** (`ndarray`):
            The initial samples at which to evaluate the model.

        * **qoi** (`list`):
            A list, which contaains the model evaluations.

        * **dist_object** ((list of) ``Distribution`` object(s)):
            List of ``Distribution`` objects corresponding to each random variable.


        **Output/Returns:**

        * **new_samples** (`ndarray`):
            Samples selected for model evaluation.

        * **indicator** (`boolean`):
            Indicator for stopping criteria.

            `indicator = True` specifies that the stopping criterion has been met and the AKMCS.run method stops.

        * **eif_lf** (`ndarray`)
            EIF learning function evaluated at the new sample points.
        """
        n_add = kwargs['n_add']
        parameters = kwargs['parameters']
        qoi = kwargs['qoi']

        g, sig = surr(pop, True)

        # Remove the inconsistency in the shape of 'g' and 'sig' array
        g = g.reshape([pop.shape[0], 1])
        sig = sig.reshape([pop.shape[0], 1])

        fm = min(qoi)
        eif = (fm - g) * stats.norm.cdf((fm - g) / sig) + sig * stats.norm.pdf((fm - g) / sig)
        rows = eif[:, 0].argsort()[(np.size(g) - n_add):]

        indicator = False
        if max(eif[:, 0]) / abs(fm) <= parameters['eif_stop']:
            indicator = True

        new_samples = pop[rows, :]
        eif_lf = eif[rows, :]
        return new_samples, eif_lf, indicator

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
            final_nsamples, final_nsamples_per_chain = nsamples, nsamples_per_chain

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
            final_nsamples = nsamples + self.nsamples
            final_nsamples_per_chain = nsamples_per_chain + self.nsamples_per_chain

        return final_nsamples, final_nsamples_per_chain, current_state, current_log_pdf

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
        accept_vec = np.zeros((self.nchains, ))
        # Separate the full ensemble into two sets, use one as a complementary ensemble to the other and vice-versa
        for split in range(2):
            set1 = (inds == split)

            # Get current and complementary sets
            sets = [current_state[inds == j01, :] for j01 in range(2)]
            curr_set, comp_set = sets[split], sets[1 - split]  # current and complementary sets respectively
            ns, nc = len(curr_set), len(comp_set)

            # Sample new state for S1 based on S0
            unif_rvs = Uniform().rvs(nsamples=ns, random_state=self.random_state)
            zz = ((self.scale - 1.) * unif_rvs + 1.) ** 2. / self.scale  # sample Z
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

    * **unweighted_samples** (`ndarray`):
        Set of un-weighted samples (useful for instance for plotting), computed by calling the `resample` method

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

        # If a set of unweighted samples exist, delete them as they are not representative of the distribution anymore
        if self.unweighted_samples is not None:
            if self.verbose:
                print('UQpy: unweighted samples are being deleted, call the resample method to regenerate them')
            self.unweighted_samples = None

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
            existing weighted samples).

        **Output/Returns:**

        The method has no returns, but it computes the following attribute of the ``IS`` object.

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
