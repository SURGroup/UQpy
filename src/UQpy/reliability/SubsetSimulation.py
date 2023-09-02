from UQpy.sampling import *
from UQpy.utilities.Utilities import process_random_state
from UQpy.utilities.ValidationTypes import PositiveInteger


class SubsetSimulation:

    @beartype
    def __init__(
        self,
        runmodel_object: RunModel,
        sampling: MCMC,
        samples_init: np.ndarray = None,
        conditional_probability: Annotated[Union[float, int], Is[lambda x: 0 <= x <= 1]] = 0.1,
        nsamples_per_subset: PositiveInteger = 1000,
        max_level: PositiveInteger = 10,
    ):
        """
        Perform Subset Simulation to estimate probability of failure.

        This class estimates probability of failure for a user-defined model using Subset Simulation. The class can
        use one of several MCMC algorithms to draw conditional samples.

        :param runmodel_object: The computational model. It should be of type :class:`.RunModel`.
        :param sampling: Specifies the :class:`MCMC` algorithm.
        :param samples_init: A set of samples from the specified probability distribution. These are the samples from
         the original distribution. They are not conditional samples. The samples must be an array of size
         :code:`samples_number_per_subset x dimension`.
         If :code:`samples_init` is not specified, the :class:`.SubsetSimulation` class will use the :class:`.MCMC` class
         created with the aid of :code:`sampling` to draw the initial samples.
        :param conditional_probability: Conditional probability for each conditional level. Default: :math:`0.1`
        :param nsamples_per_subset: Number of samples to draw in each conditional level. Default: :math:`1000`
        :param max_level: Maximum number of allowable conditional levels. Default: :math:`10`
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

        self.dependent_chains_CoV: float = None
        """Coefficient of variation of the probability of failure estimate with dependent chains."""
        self.failure_probability: float = None
        """Probability of failure estimate."""
        self.independent_chains_CoV: float = None
        """Coefficient of variation of the probability of failure estimate assuming independent chains."""
        self.performance_function_per_level: list = []
        """A list of arrays containing the evaluation of the performance function at each sample in each conditional 
        level. The size of the list is equal to the number of levels."""
        self.performance_threshold_per_level: list = []
        """Threshold value of the performance function for each conditional level. The size of the list is equal to the 
        number of levels."""
        self.samples: list = []
        """A list of arrays containing the samples in each conditional level. The size of the list is equal to the 
        number of levels."""

        self.logger.info("UQpy: Running Subset Simulation with mcmc of type: " + str(type(sampling)))
        [self.failure_probability, self.independent_chains_CoV, self.dependent_chains_CoV] = self._run()
        self.logger.info("UQpy: Subset Simulation Complete!")

    def _run(self):
        """Execute subset simulation

        This is an instance method that runs subset simulation. It is automatically called when the
        :class:`.SubsetSimulation` class is instantiated.

        :return: failure_probability, probability_cov_independent, probability_cov_dependent
        """
        conditional_level = 0
        n_keep = int(self.conditional_probability * self.nsamples_per_subset)
        independent_chain_cov_squared = []
        dependent_chain_cov_squared = []

        # Generate the initial samples - Level 0
        # Here we need to make sure that we have good initial samples from the target joint density.
        if self.samples_init is None:
            self.logger.warning(
                "UQpy: You have not provided initial samples."
                "\n Subset simulation is highly sensitive to the initial sample set. It is recommended that the user either:"
                "\n- Provide an initial set of samples (samples_init) known to follow the joint distribution of the parameters; or"
                "\n- Provide a robust mcmc object that will draw independent initial samples from the joint distribution of the parameters."
            )
            self.mcmc_objects[0].run(nsamples=self.nsamples_per_subset)
            self.samples.append(self.mcmc_objects[0].samples)
        else:
            self.samples.append(self.samples_init)

        # Run the model with initial samples, sort by their performance function, and identify the conditional level
        self.runmodel_object.run(samples=np.atleast_2d(self.samples[conditional_level]))
        self.performance_function_per_level.append(np.squeeze(self.runmodel_object.qoi_list))
        g_ind = np.argsort(self.performance_function_per_level[conditional_level])
        self.performance_threshold_per_level.append(self.performance_function_per_level[conditional_level][g_ind[n_keep - 1]])

        # Estimate coefficient of variation of conditional probability of first level
        independent_intermediate_cov, dependent_intermediate_cov = self._compute_intermediate_cov(conditional_level)
        independent_chain_cov_squared.append(independent_intermediate_cov ** 2)
        dependent_chain_cov_squared.append(dependent_intermediate_cov ** 2)

        self.logger.info("UQpy: Subset Simulation, conditional level 0 complete.")

        while self.performance_threshold_per_level[conditional_level] > 0 and conditional_level < self.max_level:
            conditional_level += 1  # Increment the conditional level

            # Initialize the samples and the performance function at the next conditional level
            self.samples.append(np.zeros_like(self.samples[conditional_level - 1]))
            self.samples[conditional_level][:n_keep] = self.samples[conditional_level - 1][g_ind[0:n_keep], :]
            self.performance_function_per_level.append(np.zeros_like(self.performance_function_per_level[conditional_level - 1]))
            self.performance_function_per_level[conditional_level][:n_keep] = self.performance_function_per_level[conditional_level - 1][g_ind[:n_keep]]

            # Unpack the attributes
            new_sampler = copy.deepcopy(self._sampling_class)
            new_sampler.seed = np.atleast_2d(self.samples[conditional_level][:n_keep, :])
            new_sampler.n_chains = len(np.atleast_2d(self.samples[conditional_level][:n_keep, :]))
            new_sampler.random_state = process_random_state(self._random_state)
            self.mcmc_objects.append(new_sampler)

            # Set the number of samples to propagate each chain (n_prop) in the conditional level
            n_prop_test = self.nsamples_per_subset / self.mcmc_objects[conditional_level].n_chains
            if n_prop_test.is_integer():
                n_prop = self.nsamples_per_subset // self.mcmc_objects[conditional_level].n_chains
            else:
                raise AttributeError(
                    "UQpy: The number of samples per subset (nsamples_per_subset) must be an integer multiple of "
                    "the number of MCMC chains.")

            # Propagate each chain n_prop times and evaluate the model to accept or reject.
            for i in range(n_prop - 1):

                # Propagate each chain
                if i == 0:
                    self.mcmc_objects[conditional_level].run(nsamples=2 * self.mcmc_objects[conditional_level].n_chains)
                else:
                    self.mcmc_objects[conditional_level].run(nsamples=self.mcmc_objects[conditional_level].n_chains)

                # Decide whether a new simulation is needed for each proposed state
                a = self.mcmc_objects[conditional_level].samples[i * n_keep : (i + 1) * n_keep, :]
                b = self.mcmc_objects[conditional_level].samples[(i + 1) * n_keep : (i + 2) * n_keep, :]
                test1 = np.equal(a, b)
                test = np.logical_and(test1[:, 0], test1[:, 1])

                # Pull out the indices of the false values in the test list
                ind_false = [i for i, val in enumerate(test) if not val]
                # Pull out the indices of the true values in the test list
                ind_true = [i for i, val in enumerate(test) if val]

                # Do not run the model for those samples where the mcmc state remains unchanged.
                self.samples[conditional_level][[x + (i + 1) * n_keep for x in ind_true], :] = \
                    self.mcmc_objects[conditional_level].samples[ind_true, :]
                self.performance_function_per_level[conditional_level][[x + (i + 1) * n_keep for x in ind_true]] = \
                    self.performance_function_per_level[conditional_level][ind_true]

                # Run the model at each of the new sample points
                x_run = self.mcmc_objects[conditional_level].samples[[x + (i + 1) * n_keep for x in ind_false], :]
                if x_run.size != 0:
                    self.runmodel_object.run(samples=x_run)

                    # Temporarily save the latest model runs
                    response_function_values = np.asarray(self.runmodel_object.qoi_list[-len(x_run) :])

                    # Accept the states with g <= g_level
                    ind_accept = np.where(response_function_values <= self.performance_threshold_per_level[conditional_level - 1])[0]
                    for j in ind_accept:
                        self.samples[conditional_level][(i + 1) * n_keep + ind_false[j]] = x_run[j]
                        self.performance_function_per_level[conditional_level][(i + 1) * n_keep + ind_false[j]] = response_function_values[j]

                    # Reject the states with g > g_level
                    ind_reject = np.where(response_function_values > self.performance_threshold_per_level[conditional_level - 1])[0]
                    for k in ind_reject:
                        self.samples[conditional_level][(i + 1) * n_keep + ind_false[k]] =\
                            self.samples[conditional_level][i * n_keep + ind_false[k]]
                        self.performance_function_per_level[conditional_level][(i + 1) * n_keep + ind_false[k]] = \
                            self.performance_function_per_level[conditional_level][i * n_keep + ind_false[k]]

            g_ind = np.argsort(self.performance_function_per_level[conditional_level])
            self.performance_threshold_per_level.append(self.performance_function_per_level[conditional_level][g_ind[n_keep]])

            # Estimate coefficient of variation of conditional probability of first level
            independent_intermediate_cov, dependent_intermediate_cov = self._compute_intermediate_cov(conditional_level)
            independent_chain_cov_squared.append(independent_intermediate_cov ** 2)
            dependent_chain_cov_squared.append(dependent_intermediate_cov ** 2)

            self.logger.info("UQpy: Subset Simulation, conditional level " + str(conditional_level) + " complete.")

        n_fail = len([value for value in self.performance_function_per_level[conditional_level] if value < 0])

        failure_probability = (self.conditional_probability ** conditional_level * n_fail / self.nsamples_per_subset)
        probability_cov_independent = np.sqrt(np.sum(independent_chain_cov_squared))
        probability_cov_dependent = np.sqrt(np.sum(dependent_chain_cov_squared))

        return failure_probability, probability_cov_independent, probability_cov_dependent

    def _compute_intermediate_cov(self, conditional_level: int):
        """Computes the coefficient of variation of the intermediate failure probability

        Assumes the initial samples are uncorrelated so correction factors do not need to be computed.

        :param conditional_level: Index :math:`i` for the intermediate subset
        :return: independent_chains_cov, dependent_chains_cov
        """
        if conditional_level == 0:
            independent_chains_cov = np.sqrt((1 - self.conditional_probability)
                                             / (self.conditional_probability * self.nsamples_per_subset))
            dependent_chains_cov = np.sqrt((1 - self.conditional_probability)
                                           / (self.conditional_probability * self.nsamples_per_subset))
        else:
            n_chains = int(self.conditional_probability * self.nsamples_per_subset)
            n_samples_per_chain = int(1 / self.conditional_probability)
            indicator = np.reshape(self.performance_function_per_level[conditional_level] < self.performance_threshold_per_level[conditional_level],
                                   (n_samples_per_chain, n_chains))
            gamma = self._correlation_factor_gamma(indicator, n_samples_per_chain, n_chains)
            response_function_values = np.reshape(self.performance_function_per_level[conditional_level], (n_samples_per_chain, n_chains))
            beta_hat = self._correlation_factor_beta(response_function_values, conditional_level)

            independent_chains_cov = np.sqrt(((1 - self.conditional_probability)
                                              / (self.conditional_probability * self.nsamples_per_subset))
                                             * (1 + gamma))
            dependent_chains_cov = np.sqrt(((1 - self.conditional_probability)
                                            / (self.conditional_probability * self.nsamples_per_subset))
                                           * (1 + gamma + beta_hat))

        return independent_chains_cov, dependent_chains_cov

    def _correlation_factor_gamma(self, indicator: np.ndarray, n_samples_per_chain: int, n_chains: int):
        """Computes the conventional correlation factor :math:`\gamma` as defined by Au and Beck 2001

        :param indicator: Intermediate indicator function :math:`I_{Conditional Level}(\cdot)`
        :param n_samples_per_chain: Number of samples per chain in the MCMC algorithm
        :param n_chains: Number of chains in the MCMC algorithm
        :return:
        """
        gamma = np.zeros(n_samples_per_chain - 1)
        r = np.zeros(n_samples_per_chain)

        ii = indicator * 1
        r_ = ii @ ii.T / n_chains - self.conditional_probability ** 2
        for i in range(r_.shape[0]):
            r[i] = np.sum(np.diag(r_, i)) / (r_.shape[0] - i)

        r0 = 0.1 * (1 - 0.1)
        r = r / r0

        for i in range(n_samples_per_chain - 1):
            gamma[i] = (1 - ((i + 1) / n_samples_per_chain)) * r[i + 1]
        gamma = 2 * np.sum(gamma)

        return gamma

    def _correlation_factor_beta(self, response_function_values, conditional_level: int):
        """Computes the updated correlation factor :math:`beta` from Shields, Giovanis, Sundar 2021

        :param response_function_values: The response function :math:`G` evaluated at the conditional level :math:`i`
        :param conditional_level: Index :math:`i` for the intermediate subset
        :return:
        """
        beta = 0
        for i in range(np.shape(response_function_values)[1]):
            for j in range(i + 1, np.shape(response_function_values)[1]):
                if response_function_values[0, i] == response_function_values[0, j]:
                    beta += 1
        beta *= 2

        acceptance_rate = np.asarray(self.mcmc_objects[conditional_level].acceptance_rate)
        mean_acceptance_rate = np.mean(acceptance_rate)

        factor = sum((1 - (i + 1) * np.shape(response_function_values)[0] / np.shape(response_function_values)[1]) * (1 - mean_acceptance_rate)
                     for i in range(np.shape(response_function_values)[0] - 1))
        factor = factor * 2 + 1

        beta = beta / np.shape(response_function_values)[1] * factor

        return beta
