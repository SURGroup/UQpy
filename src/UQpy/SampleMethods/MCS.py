from UQpy.Distributions import *
import numpy as np

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
                    raise ValueError('UQpy: rvs method is missing.')
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

