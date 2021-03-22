import warnings
from inspect import isclass

import numpy as np

from UQpy.RunModel import RunModel
from UQpy.SampleMethods import *


########################################################################################################################
########################################################################################################################
#                                        Subset Simulation
########################################################################################################################


class SubsetSimulation:
    """
    Perform Subset Simulation to estimate probability of failure.

    This class estimates probability of failure for a user-defined model using Subset Simulation. The class can
    use one of several MCMC algorithms to draw conditional samples.

    **Input:**

    * **runmodel_object** (``RunModel`` object):
        The computational model. It should be of type `RunModel` (see ``RunModel`` class).

    * **mcmc_class** (Class of type ``SampleMethods.MCMC``)
        Specifies the MCMC algorithm.

        Must be a child class of the ``SampleMethods.MCMC`` parent class. Note: This is `not` and object of the class.
        This input specifies the class itself.

    * **samples_init** (`ndarray`)
        A set of samples from the specified probability distribution. These are the samples from the original
        distribution. They are not conditional samples. The samples must be an array of size
        `nsamples_per_ss x dimension`.

        If `samples_init` is not specified, the Subset_Simulation class will use the `mcmc_class` to draw the initial
        samples.

    * **p_cond** (`float`):
        Conditional probability for each conditional level.

    * **nsamples_per_ss** (`int`)
        Number of samples to draw in each conditional level.

    * **max_level** (`int`)
        Maximum number of allowable conditional levels.

    * **verbose** (Boolean):
        A boolean declaring whether to write text to the terminal.

    * **mcmc_kwargs** (`dict`)
        Any additional keyword arguments needed for the specific ``MCMC`` class.

    **Attributes:**

    * **samples** (`list` of `ndarrays`)
         A list of arrays containing the samples in each conditional level.

    * **g** (`list` of `ndarrays`)
        A list of arrays containing the evaluation of the performance function at each sample in each conditional level.

    * **g_level** (`list`)
        Threshold value of the performance function for each conditional level

    * **pf** (`float`)
        Probability of failure estimate

    * **cov1** (`float`)
        Coefficient of variation of the probability of failure estimate assuming independent chains

    * **cov2** (`float`)
        Coefficient of variation of the probability of failure estimate with dependent chains. From [4]_


    **Methods:**
    """

    def __init__(self, runmodel_object, mcmc_class=MMH, samples_init=None, p_cond=0.1, nsamples_per_ss=1000,
                 max_level=10, verbose=False, **mcmc_kwargs):

        # Store the MCMC object to create a new object of this type for each subset
        self.mcmc_kwargs = mcmc_kwargs
        self.mcmc_class = mcmc_class

        # Initialize other attributes
        self.runmodel_object = runmodel_object
        self.samples_init = samples_init
        self.p_cond = p_cond
        self.nsamples_per_ss = nsamples_per_ss
        self.max_level = max_level
        self.verbose = verbose

        # Check that a RunModel object is being passed in.
        if not isinstance(self.runmodel_object, RunModel):
            raise AttributeError(
                'UQpy: Subset simulation requires the user to pass a RunModel object')

        if 'random_state' in self.mcmc_kwargs:
            self.random_state = self.mcmc_kwargs['random_state']
            if isinstance(self.random_state, int):
                self.random_state = np.random.RandomState(self.random_state)
            elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
                raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')
        else:
            self.random_state = None

        # Perform initial error checks
        self._init_sus()

        # Initialize the mcmc_object from the specified class.
        mcmc_object = self.mcmc_class(**self.mcmc_kwargs)
        self.mcmc_objects = [mcmc_object]

        # Initialize new attributes/variables
        self.samples = list()
        self.g = list()
        self.g_level = list()

        if self.verbose:
            print('UQpy: Running Subset Simulation with MCMC of type: ' + str(type(mcmc_object)))

        [self.pf, self.cov1, self.cov2] = self.run()

        if self.verbose:
            print('UQpy: Subset Simulation Complete!')

    # -----------------------------------------------------------------------------------------------------------------------
    # The run function executes the chosen subset simulation algorithm
    def run(self):
        """
        Execute subset simulation

        This is an instance method that runs subset simulation. It is automatically called when the SubsetSimulation
        class is instantiated.

        **Output/Returns:**

        * **pf** (`float`)
            Probability of failure estimate

        * **cov1** (`float`)
            Coefficient of variation of the probability of failure estimate assuming independent chains

        * **cov2** (`float`)
            Coefficient of variation of the probability of failure estimate with dependent chains. From [4]_

        """

        step = 0
        n_keep = int(self.p_cond * self.nsamples_per_ss)
        d12 = list()
        d22 = list()

        # Generate the initial samples - Level 0
        # Here we need to make sure that we have good initial samples from the target joint density.
        if self.samples_init is None:
            warnings.warn('UQpy: You have not provided initial samples.\n Subset simulation is highly sensitive to the '
                          'initial sample set. It is recommended that the user either:\n'
                          '- Provide an initial set of samples (samples_init) known to follow the distribution; or\n'
                          '- Provide a robust MCMC object that will draw independent initial samples from the '
                          'distribution.')
            self.mcmc_objects[0].run(nsamples=self.nsamples_per_ss)
            self.samples.append(self.mcmc_objects[0].samples)
        else:
            self.samples.append(self.samples_init)

        # Run the model for the initial samples, sort them by their performance function, and identify the
        # conditional level
        self.runmodel_object.run(samples=np.atleast_2d(self.samples[step]))
        self.g.append(np.squeeze(self.runmodel_object.qoi_list))
        g_ind = np.argsort(self.g[step])
        self.g_level.append(self.g[step][g_ind[n_keep - 1]])

        # Estimate coefficient of variation of conditional probability of first level
        d1, d2 = self._cov_sus(step)
        d12.append(d1 ** 2)
        d22.append(d2 ** 2)

        if self.verbose:
            print('UQpy: Subset Simulation, conditional level 0 complete.')

        while self.g_level[step] > 0 and step < self.max_level:

            # Increment the conditional level
            step = step + 1

            # Initialize the samples and the performance function at the next conditional level
            self.samples.append(np.zeros_like(self.samples[step - 1]))
            self.samples[step][:n_keep] = self.samples[step - 1][g_ind[0:n_keep], :]
            self.g.append(np.zeros_like(self.g[step - 1]))
            self.g[step][:n_keep] = self.g[step - 1][g_ind[:n_keep]]

            # Unpack the attributes

            # Initialize a new MCMC object for each conditional level
            self.mcmc_kwargs['seed'] = np.atleast_2d(self.samples[step][:n_keep, :])
            self.mcmc_kwargs['random_state'] = self.random_state
            new_mcmc_object = self.mcmc_class(**self.mcmc_kwargs)
            self.mcmc_objects.append(new_mcmc_object)

            # Set the number of samples to propagate each chain (n_prop) in the conditional level
            n_prop_test = self.nsamples_per_ss / self.mcmc_objects[step].nchains
            if n_prop_test.is_integer():
                n_prop = self.nsamples_per_ss // self.mcmc_objects[step].nchains
            else:
                raise AttributeError(
                    'UQpy: The number of samples per subset (nsamples_per_ss) must be an integer multiple of '
                    'the number of MCMC chains.')

            # Propagate each chain n_prop times and evaluate the model to accept or reject.
            for i in range(n_prop - 1):

                # Propagate each chain
                if i == 0:
                    self.mcmc_objects[step].run(nsamples=2 * self.mcmc_objects[step].nchains)
                else:
                    self.mcmc_objects[step].run(nsamples=self.mcmc_objects[step].nchains)

                # Decide whether a new simulation is needed for each proposed state
                a = self.mcmc_objects[step].samples[i * n_keep:(i + 1) * n_keep, :]
                b = self.mcmc_objects[step].samples[(i + 1) * n_keep:(i + 2) * n_keep, :]
                test1 = np.equal(a, b)
                test = np.logical_and(test1[:, 0], test1[:, 1])

                # Pull out the indices of the false values in the test list
                ind_false = [i for i, val in enumerate(test) if not val]
                # Pull out the indices of the true values in the test list
                ind_true = [i for i, val in enumerate(test) if val]

                # Do not run the model for those samples where the MCMC state remains unchanged.
                self.samples[step][[x + (i + 1) * n_keep for x in ind_true], :] = \
                    self.mcmc_objects[step].samples[ind_true, :]
                self.g[step][[x + (i + 1) * n_keep for x in ind_true]] = self.g[step][ind_true]

                # Run the model at each of the new sample points
                x_run = self.mcmc_objects[step].samples[[x + (i + 1) * n_keep for x in ind_false], :]
                if x_run.size != 0:
                    self.runmodel_object.run(samples=x_run)

                    # Temporarily save the latest model runs
                    g_temp = np.asarray(self.runmodel_object.qoi_list[-len(x_run):])

                    # Accept the states with g <= g_level
                    ind_accept = np.where(g_temp <= self.g_level[step - 1])[0]
                    for ii in ind_accept:
                        self.samples[step][(i + 1) * n_keep + ind_false[ii]] = x_run[ii]
                        self.g[step][(i + 1) * n_keep + ind_false[ii]] = g_temp[ii]

                    # Reject the states with g > g_level
                    ind_reject = np.where(g_temp > self.g_level[step - 1])[0]
                    for ii in ind_reject:
                        self.samples[step][(i + 1) * n_keep + ind_false[ii]] = \
                            self.samples[step][i * n_keep + ind_false[ii]]
                        self.g[step][(i + 1) * n_keep + ind_false[ii]] = self.g[step][i * n_keep + ind_false[ii]]

            g_ind = np.argsort(self.g[step])
            self.g_level.append(self.g[step][g_ind[n_keep]])

            # Estimate coefficient of variation of conditional probability of first level
            d1, d2 = self._cov_sus(step)
            d12.append(d1 ** 2)
            d22.append(d2 ** 2)

            if self.verbose:
                print('UQpy: Subset Simulation, conditional level ' + str(step) + ' complete.')

        n_fail = len([value for value in self.g[step] if value < 0])

        pf = self.p_cond ** step * n_fail / self.nsamples_per_ss
        cov1 = np.sqrt(np.sum(d12))
        cov2 = np.sqrt(np.sum(d22))

        return pf, cov1, cov2

    # -----------------------------------------------------------------------------------------------------------------------
    # Support functions for subset simulation

    def _init_sus(self):
        """
        Check for errors in the SubsetSimulation class input

        This is an instance method that checks for errors in the input to the SubsetSimulation class. It is
        automatically called when the SubsetSimualtion class is instantiated.

        No inputs or returns.
        """

        # Check that an MCMC class is being passed in.
        if not isclass(self.mcmc_class):
            raise ValueError('UQpy: mcmc_class must be a child class of MCMC. Note it is not an instance of the class.')
        if not issubclass(self.mcmc_class, MCMC):
            raise ValueError('UQpy: mcmc_class must be a child class of MCMC.')

        # Check that a RunModel object is being passed in.
        if not isinstance(self.runmodel_object, RunModel):
            raise AttributeError(
                'UQpy: Subset simulation requires the user to pass a RunModel object')

        # Check that a valid conditional probability is specified.
        if type(self.p_cond).__name__ != 'float':
            raise AttributeError('UQpy: Invalid conditional probability. p_cond must be of float type.')
        elif self.p_cond <= 0. or self.p_cond >= 1.:
            raise AttributeError('UQpy: Invalid conditional probability. p_cond must be in (0, 1).')

        # Check that the number of samples per subset is properly defined.
        if type(self.nsamples_per_ss).__name__ != 'int':
            raise AttributeError('UQpy: Number of samples per subset (nsamples_per_ss) must be integer valued.')

        # Check that max_level is an integer
        if type(self.max_level).__name__ != 'int':
            raise AttributeError('UQpy: The maximum subset level (max_level) must be integer valued.')

    def _cov_sus(self, step):

        """
        Compute the coefficient of variation of the samples in a conditional level

        This is an instance method that is called after each conditional level is complete to compute the coefficient
        of variation of the conditional probability in that level.

        **Input:**

        :param step: Specifies the conditional level
        :type step: int

        **Output/Returns:**

        :param d1: Coefficient of variation in conditional level assuming independent chains
        :type d1: float

        :param d2: Coefficient of variation in conditional level with dependent chains
        :type d2: float
        """

        # Here, we assume that the initial samples are drawn to be uncorrelated such that the correction factors do not
        # need to be computed.
        if step == 0:
            d1 = np.sqrt((1 - self.p_cond) / (self.p_cond * self.nsamples_per_ss))
            d2 = np.sqrt((1 - self.p_cond) / (self.p_cond * self.nsamples_per_ss))

            return d1, d2
        else:
            n_c = int(self.p_cond * self.nsamples_per_ss)
            n_s = int(1 / self.p_cond)
            indicator = np.reshape(self.g[step] < self.g_level[step], (n_s, n_c))
            gamma = self._corr_factor_gamma(indicator, n_s, n_c)
            g_temp = np.reshape(self.g[step], (n_s, n_c))
            beta_hat = self._corr_factor_beta(g_temp, step)

            d1 = np.sqrt(((1 - self.p_cond) / (self.p_cond * self.nsamples_per_ss)) * (1 + gamma))
            d2 = np.sqrt(((1 - self.p_cond) / (self.p_cond * self.nsamples_per_ss)) * (1 + gamma + beta_hat))

            return d1, d2

    # Computes the conventional correlation factor gamma from Au and Beck
    def _corr_factor_gamma(self, indicator, n_s, n_c):
        """
        Compute the conventional correlation factor gamma from Au and Beck (Reference [1])

        This is an instance method that computes the correlation factor gamma used to estimate the coefficient of
        variation of the conditional probability estimate from a given conditional level. This method is called
        automatically within the _cov_sus method.

        **Input:**

        :param indicator: An array of booleans indicating whether the performance function is below the threshold for
                          the conditional probability.
        :type indicator: boolean array

        :param n_s: Number of samples drawn from each Markov chain in each conditional level
        :type n_s: int

        :param n_c: Number of Markov chains in each conditional level
        :type n_c: int

        **Output/Returns:**

        :param gam: Gamma factor in coefficient of variation estimate
        :type gam: float

        """

        gam = np.zeros(n_s - 1)
        r = np.zeros(n_s)

        ii = indicator * 1
        r_ = ii @ ii.T / n_c - self.p_cond ** 2
        for i in range(r_.shape[0]):
            r[i] = np.sum(np.diag(r_, i)) / (r_.shape[0] - i)

        r0 = 0.1 * (1 - 0.1)
        r = r / r0

        for i in range(n_s - 1):
            gam[i] = (1 - ((i + 1) / n_s)) * r[i + 1]
        gam = 2 * np.sum(gam)

        return gam

    # Computes the updated correlation factor beta from Shields et al.
    def _corr_factor_beta(self, g, step):
        """
        Compute the additional correlation factor beta from Shields et al. (Reference [2])

        This is an instance method that computes the correlation factor beta used to estimate the coefficient of
        variation of the conditional probability estimate from a given conditional level. This method is called
        automatically within the _cov_sus method.

        **Input:**

        :param g: An array containing the performance function evaluation at all points in the current conditional
                  level.
        :type g: numpy array

        :param step: Current conditional level
        :type step: int

        **Output/Returns:**

        :param beta: Beta factor in coefficient of variation estimate
        :type beta: float

        """

        beta = 0
        for i in range(np.shape(g)[1]):
            for j in range(i + 1, np.shape(g)[1]):
                if g[0, i] == g[0, j]:
                    beta = beta + 1
        beta = beta * 2

        ar = np.asarray(self.mcmc_objects[step].acceptance_rate)
        ar_mean = np.mean(ar)

        factor = 0
        for i in range(np.shape(g)[0] - 1):
            factor = factor + (1 - (i + 1) * np.shape(g)[0] / np.shape(g)[1]) * (1 - ar_mean)
        factor = factor * 2 + 1

        beta = beta / np.shape(g)[1] * factor

        return beta