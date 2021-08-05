import copy
import logging
import warnings
from inspect import isclass

import numpy as np

from UQpy.RunModel import RunModel
from UQpy.sampling import *


class SubsetSimulation:
    """
    Perform Subset Simulation to estimate probability of failure.

    This class estimates probability of failure for a user-defined model using Subset Simulation. The class can
    use one of several mcmc algorithms to draw conditional samples.

    **Input:**

    * **runmodel_object** (``RunModel`` object):
        The computational model. It should be of type `RunModel` (see ``RunModel`` class).

    * **mcmc_class** (Class of type ``sampling.mcmc``)
        Specifies the mcmc algorithm.

        Must be a child class of the ``sampling.mcmc`` parent class. Note: This is `not` and object of the class.
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
        Any additional keyword arguments needed for the specific ``mcmc`` class.

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

    def __init__(self, runmodel_object, mcmc_object, samples_init=None,
                 conditional_probability=0.1, nsamples_per_ss=1000, max_level=10):
        # Initialize other attributes
        self.runmodel_object = runmodel_object
        self.samples_init = samples_init
        self.conditional_probability = conditional_probability
        self.nsamples_per_ss = nsamples_per_ss
        self.max_level = max_level
        self.logger = logging.getLogger(__name__)

        # Check that a RunModel object is being passed in.
        if not isinstance(self.runmodel_object, RunModel):
            raise AttributeError(
                'UQpy: Subset simulation requires the user to pass a RunModel object')


        # Perform initial error checks
        self._verify_initialization_data()

        # Initialize the mcmc_object from the specified class.
        self.mcmc_objects = [mcmc_object]

        # Initialize new attributes/variables
        self.samples = list()
        self.g = list()
        self.g_level = list()

        self.logger.info('UQpy: Running Subset Simulation with mcmc of type: ' + str(type(mcmc_object)))

        [self.pf, self.cov1, self.cov2] = self.run()

        self.logger.info('UQpy: Subset Simulation Complete!')

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
        n_keep = int(self.conditional_probability * self.nsamples_per_ss)
        d12 = list()
        d22 = list()

        # Generate the initial samples - Level 0
        # Here we need to make sure that we have good initial samples from the target joint density.
        if self.samples_init is None:
            self.logger.warning('UQpy: You have not provided initial samples.\n Subset simulation is highly sensitive '
                                'to the initial sample set. It is recommended that the user either:\n'
                                '- Provide an initial set of samples (samples_init) known to follow the distribution; '
                                'or\n - Provide a robust mcmc object that will draw independent initial samples from '
                                'the distribution.')
            self.mcmc_objects[0].run(number_of_samples=self.nsamples_per_ss)
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
        d1, d2 = self._compute_coefficient_of_variation(step)
        d12.append(d1 ** 2)
        d22.append(d2 ** 2)

        self.logger.info('UQpy: Subset Simulation, conditional level 0 complete.')

        while self.g_level[step] > 0 and step < self.max_level:

            # Increment the conditional level
            step = step + 1

            # Initialize the samples and the performance function at the next conditional level
            self.samples.append(np.zeros_like(self.samples[step - 1]))
            self.samples[step][:n_keep] = self.samples[step - 1][g_ind[0:n_keep], :]
            self.g.append(np.zeros_like(self.g[step - 1]))
            self.g[step][:n_keep] = self.g[step - 1][g_ind[:n_keep]]

            # Unpack the attributes

            new_mcmc_object = copy.copy(self.mcmc_objects)
            self.mcmc_objects.append(new_mcmc_object)

            # Set the number of samples to propagate each chain (n_prop) in the conditional level
            n_prop_test = self.nsamples_per_ss / self.mcmc_objects[step].chains_number
            if n_prop_test.is_integer():
                n_prop = self.nsamples_per_ss // self.mcmc_objects[step].chains_number
            else:
                raise AttributeError(
                    'UQpy: The number of samples per subset (nsamples_per_ss) must be an integer multiple of '
                    'the number of mcmc chains.')

            # Propagate each chain n_prop times and evaluate the model to accept or reject.
            for i in range(n_prop - 1):

                # Propagate each chain
                if i == 0:
                    self.mcmc_objects[step].run(number_of_samples=2 * self.mcmc_objects[step].chains_number)
                else:
                    self.mcmc_objects[step].run(number_of_samples=self.mcmc_objects[step].chains_number)

                # Decide whether a new simulation is needed for each proposed state
                a = self.mcmc_objects[step].samples[i * n_keep:(i + 1) * n_keep, :]
                b = self.mcmc_objects[step].samples[(i + 1) * n_keep:(i + 2) * n_keep, :]
                test1 = np.equal(a, b)
                test = np.logical_and(test1[:, 0], test1[:, 1])

                # Pull out the indices of the false values in the test list
                ind_false = [i for i, val in enumerate(test) if not val]
                # Pull out the indices of the true values in the test list
                ind_true = [i for i, val in enumerate(test) if val]

                # Do not run the model for those samples where the mcmc state remains unchanged.
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
            d1, d2 = self._compute_coefficient_of_variation(step)
            d12.append(d1 ** 2)
            d22.append(d2 ** 2)

            self.logger.info('UQpy: Subset Simulation, conditional level ' + str(step) + ' complete.')

        n_fail = len([value for value in self.g[step] if value < 0])

        failure_probability = self.conditional_probability ** step * n_fail / self.nsamples_per_ss
        probability_cov_independent = np.sqrt(np.sum(d12))
        probability_cov_dependent = np.sqrt(np.sum(d22))

        return failure_probability, probability_cov_independent, probability_cov_dependent

    # -----------------------------------------------------------------------------------------------------------------------
    # Support functions for subset simulation

    def _verify_initialization_data(self):
        """
        Check for errors in the SubsetSimulation class input

        This is an instance method that checks for errors in the input to the SubsetSimulation class. It is
        automatically called when the SubsetSimulation class is instantiated.

        No inputs or returns.
        """

        # Check that an mcmc class is being passed in.
        if not isclass(self.mcmc_class):
            raise ValueError('UQpy: mcmc_class must be a child class of mcmc. Note it is not an instance of the class.')
        if not issubclass(self.mcmc_class, mcmc):
            raise ValueError('UQpy: mcmc_class must be a child class of mcmc.')

        # Check that a RunModel object is being passed in.
        if not isinstance(self.runmodel_object, RunModel):
            raise AttributeError(
                'UQpy: Subset simulation requires the user to pass a RunModel object')

        # Check that a valid conditional probability is specified.
        if type(self.conditional_probability).__name__ != 'float':
            raise AttributeError('UQpy: Invalid conditional probability. p_cond must be of float type.')
        elif self.conditional_probability <= 0. or self.conditional_probability >= 1.:
            raise AttributeError('UQpy: Invalid conditional probability. p_cond must be in (0, 1).')

        # Check that the number of samples per subset is properly defined.
        if type(self.nsamples_per_ss).__name__ != 'int':
            raise AttributeError('UQpy: Number of samples per subset (nsamples_per_ss) must be integer valued.')

        # Check that max_level is an integer
        if type(self.max_level).__name__ != 'int':
            raise AttributeError('UQpy: The maximum subset level (max_level) must be integer valued.')

    def _compute_coefficient_of_variation(self, step):

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
            independent_chains_cov = np.sqrt((1 - self.conditional_probability) /
                                             (self.conditional_probability * self.nsamples_per_ss))
            dependent_chains_cov = np.sqrt((1 - self.conditional_probability) /
                                           (self.conditional_probability * self.nsamples_per_ss))

            return independent_chains_cov, dependent_chains_cov
        else:
            n_c = int(self.conditional_probability * self.nsamples_per_ss)
            n_s = int(1 / self.conditional_probability)
            indicator = np.reshape(self.g[step] < self.g_level[step], (n_s, n_c))
            gamma = self._correlation_factor_gamma(indicator, n_s, n_c)
            g_temp = np.reshape(self.g[step], (n_s, n_c))
            beta_hat = self._correlation_factor_beta(g_temp, step)

            independent_chains_cov = \
                np.sqrt(((1 - self.conditional_probability) /
                         (self.conditional_probability * self.nsamples_per_ss)) * (1 + gamma))
            dependent_chains_cov =\
                np.sqrt(((1 - self.conditional_probability) /
                         (self.conditional_probability * self.nsamples_per_ss)) * (1 + gamma + beta_hat))

            return independent_chains_cov, dependent_chains_cov

    # Computes the conventional correlation factor gamma from Au and Beck
    def _correlation_factor_gamma(self, indicator, n_s, n_c):
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
        r_ = ii @ ii.T / n_c - self.conditional_probability ** 2
        for i in range(r_.shape[0]):
            r[i] = np.sum(np.diag(r_, i)) / (r_.shape[0] - i)

        r0 = 0.1 * (1 - 0.1)
        r = r / r0

        for i in range(n_s - 1):
            gam[i] = (1 - ((i + 1) / n_s)) * r[i + 1]
        gam = 2 * np.sum(gam)

        return gam

    # Computes the updated correlation factor beta from Shields et al.
    def _correlation_factor_beta(self, g, step):
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
