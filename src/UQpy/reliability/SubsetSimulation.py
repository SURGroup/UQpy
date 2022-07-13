import copy
import logging

from UQpy.run_model import RunModel
from UQpy.utilities.Utilities import process_random_state
from UQpy.sampling import *


class SubsetSimulation:

    @beartype
    def __init__(
        self,
        runmodel_object: RunModel,
        sampling: MCMC,
        samples_init: np.ndarray = None,
        conditional_probability: Annotated[Union[float, int], Is[lambda x: 0 <= x <= 1]] = 0.1,
        nsamples_per_subset: int = 1000,
        max_level: int = 10,
    ):
        """
        Perform Subset Simulation to estimate probability of failure.

        This class estimates probability of failure for a user-defined model using Subset Simulation. The class can
        use one of several mcmc algorithms to draw conditional samples.

        :param runmodel_object: The computational model. It should be of type :class:`.RunModel`.
        :param sampling: Specifies the :class:`MCMC` algorithm.
        :param samples_init: A set of samples from the specified probability distribution. These are the samples from
         the original distribution. They are not conditional samples. The samples must be an array of size
         :code:`samples_number_per_subset x dimension`.
         If `samples_init` is not specified, the :class:`.SubsetSimulation` class will use the :class:`.MCMC` class
         created with the aid of `sampling` to draw the initial samples.
        :param conditional_probability: Conditional probability for each conditional level.
        :param nsamples_per_subset: Number of samples to draw in each conditional level.
        :param max_level: Maximum number of allowable conditional levels.
        """
        # Initialize other attributes
        self._sampling_class = sampling
        self._random_state = process_random_state(sampling._random_state)
        self.runmodel_object = runmodel_object
        self.samples_init = samples_init
        self.conditional_probability = conditional_probability
        self.nsamples_per_subset = nsamples_per_subset
        self.max_level = max_level
        self.logger = logging.getLogger(__name__)

        self.mcmc_objects = [sampling]

        self.samples: list = list()
        """A list of arrays containing the samples in each conditional level. The size of the list is equal to the 
        number of levels."""
        self.performance_function_per_level: list = []
        """A list of arrays containing the evaluation of the performance function at each sample in each conditional 
        level. The size of the list is equal to the number of levels."""
        self.performance_threshold_per_level: list = []
        """Threshold value of the performance function for each conditional level. The size of the list is equal to the 
        number of levels."""

        self.logger.info("UQpy: Running Subset Simulation with mcmc of type: " + str(type(sampling)))
        self.failure_probability: float = None
        """Probability of failure estimate."""
        self.independent_chains_CoV: float = None
        """Coefficient of variation of the probability of failure estimate assuming independent chains."""
        self.dependent_chains_CoV: float = None
        """Coefficient of variation of the probability of failure estimate with dependent chains."""

        [self.failure_probability, self.independent_chains_CoV, self.dependent_chains_CoV] = self._run()

        self.logger.info("UQpy: Subset Simulation Complete!")

    # The run function executes the chosen subset simulation algorithm
    def _run(self):
        """
        Execute subset simulation

        This is an instance method that runs subset simulation. It is automatically called when the
        :class:`.SubsetSimulation` class is instantiated.
        """
        step = 0
        n_keep = int(self.conditional_probability * self.nsamples_per_subset)
        d12 = []
        d22 = []

        # Generate the initial samples - Level 0
        # Here we need to make sure that we have good initial samples from the target joint density.
        if self.samples_init is None:
            self.logger.warning(
                "UQpy: You have not provided initial samples.\n Subset simulation is highly sensitive "
                "to the initial sample set. It is recommended that the user either:\n"
                "- Provide an initial set of samples (samples_init) known to follow the distribution; "
                "or\n - Provide a robust mcmc object that will draw independent initial samples from "
                "the distribution."
            )
            self.mcmc_objects[0].run(nsamples=self.nsamples_per_subset)
            self.samples.append(self.mcmc_objects[0].samples)
        else:
            self.samples.append(self.samples_init)

        # Run the model for the initial samples, sort them by their performance function, and identify the
        # conditional level
        self.runmodel_object.run(samples=np.atleast_2d(self.samples[step]))
        self.performance_function_per_level.append(np.squeeze(self.runmodel_object.qoi_list))
        g_ind = np.argsort(self.performance_function_per_level[step])
        self.performance_threshold_per_level.append(self.performance_function_per_level[step][g_ind[n_keep - 1]])

        # Estimate coefficient of variation of conditional probability of first level
        d1, d2 = self._compute_coefficient_of_variation(step)
        d12.append(d1 ** 2)
        d22.append(d2 ** 2)

        self.logger.info("UQpy: Subset Simulation, conditional level 0 complete.")

        while self.performance_threshold_per_level[step] > 0 and step < self.max_level:

            # Increment the conditional level
            step += 1

            # Initialize the samples and the performance function at the next conditional level
            self.samples.append(np.zeros_like(self.samples[step - 1]))
            self.samples[step][:n_keep] = self.samples[step - 1][g_ind[0:n_keep], :]
            self.performance_function_per_level.append(np.zeros_like(self.performance_function_per_level[step - 1]))
            self.performance_function_per_level[step][:n_keep] = self.performance_function_per_level[step - 1][g_ind[:n_keep]]

            # Unpack the attributes
            new_sampler = copy.deepcopy(self._sampling_class)
            new_sampler.seed = np.atleast_2d(self.samples[step][:n_keep, :])
            new_sampler.n_chains = len(np.atleast_2d(self.samples[step][:n_keep, :]))
            new_sampler.random_state = process_random_state(self._random_state)
            self.mcmc_objects.append(new_sampler)

            # Set the number of samples to propagate each chain (n_prop) in the conditional level
            n_prop_test = (self.nsamples_per_subset / self.mcmc_objects[step].n_chains)
            if n_prop_test.is_integer():
                n_prop = (self.nsamples_per_subset // self.mcmc_objects[step].n_chains)
            else:
                raise AttributeError(
                    "UQpy: The number of samples per subset (nsamples_per_ss) must be an integer multiple of "
                    "the number of mcmc chains.")

            # Propagate each chain n_prop times and evaluate the model to accept or reject.
            for i in range(n_prop - 1):

                # Propagate each chain
                if i == 0:
                    self.mcmc_objects[step].run(nsamples=2 * self.mcmc_objects[step].n_chains)
                else:
                    self.mcmc_objects[step].run(nsamples=self.mcmc_objects[step].n_chains)

                # Decide whether a new simulation is needed for each proposed state
                a = self.mcmc_objects[step].samples[i * n_keep : (i + 1) * n_keep, :]
                b = self.mcmc_objects[step].samples[(i + 1) * n_keep : (i + 2) * n_keep, :]
                test1 = np.equal(a, b)
                test = np.logical_and(test1[:, 0], test1[:, 1])

                # Pull out the indices of the false values in the test list
                ind_false = [i for i, val in enumerate(test) if not val]
                # Pull out the indices of the true values in the test list
                ind_true = [i for i, val in enumerate(test) if val]

                # Do not run the model for those samples where the mcmc state remains unchanged.
                self.samples[step][[x + (i + 1) * n_keep for x in ind_true], :] = \
                    self.mcmc_objects[step].samples[ind_true, :]
                self.performance_function_per_level[step][[x + (i + 1) * n_keep for x in ind_true]] = \
                    self.performance_function_per_level[step][ind_true]

                # Run the model at each of the new sample points
                x_run = self.mcmc_objects[step].samples[[x + (i + 1) * n_keep for x in ind_false], :]
                if x_run.size != 0:
                    self.runmodel_object.run(samples=x_run)

                    # Temporarily save the latest model runs
                    g_temp = np.asarray(self.runmodel_object.qoi_list[-len(x_run) :])

                    # Accept the states with g <= g_level
                    ind_accept = np.where(g_temp <= self.performance_threshold_per_level[step - 1])[0]
                    for ii in ind_accept:
                        self.samples[step][(i + 1) * n_keep + ind_false[ii]] = x_run[ii]
                        self.performance_function_per_level[step][(i + 1) * n_keep + ind_false[ii]] = g_temp[ii]

                    # Reject the states with g > g_level
                    ind_reject = np.where(g_temp > self.performance_threshold_per_level[step - 1])[0]
                    for ii in ind_reject:
                        self.samples[step][(i + 1) * n_keep + ind_false[ii]] =\
                            self.samples[step][i * n_keep + ind_false[ii]]
                        self.performance_function_per_level[step][(i + 1) * n_keep + ind_false[ii]] = \
                            self.performance_function_per_level[step][i * n_keep + ind_false[ii]]

            g_ind = np.argsort(self.performance_function_per_level[step])
            self.performance_threshold_per_level.append(self.performance_function_per_level[step][g_ind[n_keep]])

            # Estimate coefficient of variation of conditional probability of first level
            d1, d2 = self._compute_coefficient_of_variation(step)
            d12.append(d1 ** 2)
            d22.append(d2 ** 2)

            self.logger.info("UQpy: Subset Simulation, conditional level " + str(step) + " complete.")

        n_fail = len([value for value in self.performance_function_per_level[step] if value < 0])

        failure_probability = (self.conditional_probability ** step * n_fail / self.nsamples_per_subset)
        probability_cov_independent = np.sqrt(np.sum(d12))
        probability_cov_dependent = np.sqrt(np.sum(d22))

        return failure_probability, probability_cov_independent, probability_cov_dependent

    def _compute_coefficient_of_variation(self, step):
        # Here, we assume that the initial samples are drawn to be uncorrelated such that the correction factors do not
        # need to be computed.
        if step == 0:
            independent_chains_cov = np.sqrt((1 - self.conditional_probability)
                                             / (self.conditional_probability * self.nsamples_per_subset))
            dependent_chains_cov = np.sqrt((1 - self.conditional_probability)
                                           / (self.conditional_probability * self.nsamples_per_subset))
        else:
            n_c = int(self.conditional_probability * self.nsamples_per_subset)
            n_s = int(1 / self.conditional_probability)
            indicator = np.reshape(self.performance_function_per_level[step] < self.performance_threshold_per_level[step], (n_s, n_c))
            gamma = self._correlation_factor_gamma(indicator, n_s, n_c)
            g_temp = np.reshape(self.performance_function_per_level[step], (n_s, n_c))
            beta_hat = self._correlation_factor_beta(g_temp, step)

            independent_chains_cov = np.sqrt(((1 - self.conditional_probability)
                                              / (self.conditional_probability * self.nsamples_per_subset))
                                             * (1 + gamma))
            dependent_chains_cov = np.sqrt(((1 - self.conditional_probability)
                                            / (self.conditional_probability * self.nsamples_per_subset))
                                           * (1 + gamma + beta_hat))

        return independent_chains_cov, dependent_chains_cov

    # Computes the conventional correlation factor gamma from Au and Beck
    def _correlation_factor_gamma(self, indicator, n_s, n_c):
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
        beta = 0
        for i in range(np.shape(g)[1]):
            for j in range(i + 1, np.shape(g)[1]):
                if g[0, i] == g[0, j]:
                    beta += 1
        beta *= 2

        ar = np.asarray(self.mcmc_objects[step].acceptance_rate)
        ar_mean = np.mean(ar)

        factor = sum((1 - (i + 1) * np.shape(g)[0] / np.shape(g)[1]) * (1 - ar_mean) for i in range(np.shape(g)[0] - 1))
        factor = factor * 2 + 1

        beta = beta / np.shape(g)[1] * factor

        return beta
