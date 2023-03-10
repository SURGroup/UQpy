import copy

import scipy.stats as stats

from UQpy.sampling.mcmc.MetropolisHastings import MetropolisHastings
from UQpy.distributions.collection.MultivariateNormal import MultivariateNormal
from UQpy.sampling.mcmc.baseclass.MCMC import *
from UQpy.sampling.mcmc.tempering_mcmc.baseclass.TemperingMCMC import TemperingMCMC


class SequentialTemperingMCMC(TemperingMCMC):

    @beartype
    def __init__(self, pdf_intermediate=None, log_pdf_intermediate=None, args_pdf_intermediate=(), seed=None,
                 distribution_reference: Distribution = None,
                 sampler: MCMC = None,
                 nsamples: PositiveInteger = None,
                 recalculate_weights: bool = False,
                 save_intermediate_samples=False,
                 percentage_resampling: int = 100,
                 random_state: RandomStateType = None,
                 resampling_burn_length: int = 0,
                 resampling_proposal: Distribution = None,
                 resampling_proposal_is_symmetric: bool = True):

        """
        Class for Sequential-Tempering MCMC

        :param sampler: :class:`MCMC` object: MCMC samplers used to draw the remaining samples for the intermediate
         distribution after the resampling step. Default to running a simple MH algorithm, where the proposal covariance
         is calculated as per the procedure given in Ching and Chen (2007) and Betz et. al. (2016).
        :param recalculate_weights: boolean: To be set to true if the resampling weights are to be recalculated after
         each point is generated during the resampling step. This is done so that the resampling weights are in
         accordance with the new sample generated after Metropolis Hastings is used for dispersion to ensure uniqueness
         of the samples.
        :param save_intermediate_samples: boolean: To be set to true to save the samples that are generated according to
         the intermediate distributions.
        :param percentage_resampling: float: Indicates what percentage of samples for a given intermediate distribution
         are to be generated through resampling from the set of samples generated for the previous intermediate
         distribution.
        :param resampling_burn_length: int: Burn-in length for the Metropolis Hastings dispersion step to ensure
         uniqueness.
        :param resampling_proposal: :class:`.Distribution` object. The proposal distribution for the Metropolis Hastings
         dispersion step.
        :param resampling_proposal_is_symmetric: boolean: Indicates whether the provided resampling proposal is
         symmetric.
        """

        self.proposal = resampling_proposal
        self.proposal_is_symmetric = resampling_proposal_is_symmetric
        self.resampling_burn_length = resampling_burn_length
        self.logger = logging.getLogger(__name__)

        super().__init__(pdf_intermediate=pdf_intermediate, log_pdf_intermediate=log_pdf_intermediate,
                         args_pdf_intermediate=args_pdf_intermediate, distribution_reference=distribution_reference,
                         random_state=random_state)
        self.logger = logging.getLogger(__name__)

        self.sampler = sampler
        # Initialize inputs
        self.save_intermediate_samples = save_intermediate_samples
        self.recalculate_weights = recalculate_weights

        self.resample_fraction = percentage_resampling / 100

        self.__dimension = sampler.dimension
        self.__n_chains = sampler.n_chains

        self.n_samples_per_chain = int(np.floor(((1 - self.resample_fraction) * nsamples) / self.__n_chains))
        self.n_resamples = int(nsamples - (self.n_samples_per_chain * self.__n_chains))

        # Initialize input distributions
        self.evaluate_log_reference, self.seed = self._preprocess_reference(dist_=distribution_reference,
                                                                            seed_=seed, nsamples=nsamples,
                                                                            dimension=self.__dimension,
                                                                            random_state=self.random_state)

        # Initialize flag that indicates whether default proposal is to be used (default proposal defined adaptively
        # during run)
        self.proposal_given_flag = self.proposal is not None
        # Initialize attributes
        self.tempering_parameters = None
        self.evidence = None
        self.evidence_CoV = None

        # Call the run function
        if nsamples is not None:
            self.run(nsamples=nsamples)
        else:
            raise ValueError('UQpy: a value for "nsamples" must be specified ')

    @beartype
    def run(self, nsamples: PositiveInteger = None):
        """
        Run the MCMC algorithm.

        This function samples from each intermediate distribution until samples from the target are generated. Samples
        cannot be appended to existing samples in this method. It leverages the `run_iterations` method specific to the
        sampler.

        :param nsamples: Number of samples to generate from the target (the same number of samples will be generated
         for all intermediate distributions).

        """
        self.logger.info('TMCMC Start')

        if self.samples is not None:
            raise RuntimeError('UQpy: run method cannot be called multiple times for the same object')

        points = self.seed  # Generated Samples from prior for zero-th tempering level

        # Initializing other variables
        current_tempering_parameter = 0.0  # Intermediate exponent
        previous_tempering_parameter = current_tempering_parameter
        self.tempering_parameters = np.array(current_tempering_parameter)
        pts_index = np.arange(nsamples)  # Array storing sample indices
        weights = np.zeros(nsamples)  # Array storing plausibility weights
        weight_probabilities = np.zeros(nsamples)  # Array storing plausibility weight probabilities
        expected_q0 = sum(
            np.exp(self.evaluate_log_intermediate(points[i, :].reshape((1, -1)), 0.0))
            for i in range(nsamples)) / nsamples

        evidence_estimator = expected_q0

        if self.save_intermediate_samples is True:
            self.intermediate_samples = []
            self.intermediate_samples += [points.copy()]

        # Calculate covariance matrix for the default proposal
        cov_scale = 0.2
        # Looping over all adaptively decided tempering levels
        while current_tempering_parameter < 1:

            # Copy the state of the points array
            points_copy = np.copy(points)

            # Adaptively set the tempering exponent for the current level
            previous_tempering_parameter = current_tempering_parameter
            current_tempering_parameter = self._find_temper_param(previous_tempering_parameter, points,
                                                                  self.evaluate_log_intermediate, nsamples)
            # d_exp = temper_param - temper_param_prev
            self.tempering_parameters = np.append(self.tempering_parameters, current_tempering_parameter)

            self.logger.info('beta selected')

            # Calculate the plausibility weights
            for i in range(nsamples):
                weights[i] = np.exp(self.evaluate_log_intermediate(points[i, :].reshape((1, -1)),
                                                                   current_tempering_parameter)
                                    - self.evaluate_log_intermediate(points[i, :].reshape((1, -1)),
                                                                     previous_tempering_parameter))

            # Calculate normalizing constant for the plausibility weights (sum of the weights)
            w_sum = np.sum(weights)
            # Calculate evidence from each tempering level
            evidence_estimator = evidence_estimator * (w_sum / nsamples)
            # Normalize plausibility weight probabilities
            weight_probabilities = (weights / w_sum)

            w_theta_sum = np.zeros(self.__dimension)
            for i in range(nsamples):
                for j in range(self.__dimension):
                    w_theta_sum[j] += weights[i] * points[i, j]
            sigma_matrix = np.zeros((self.__dimension, self.__dimension))
            for i in range(nsamples):
                points_deviation = np.zeros((self.__dimension, 1))
                for j in range(self.__dimension):
                    points_deviation[j, 0] = points[i, j] - (w_theta_sum[j] / w_sum)
                sigma_matrix += (weights[i] / w_sum) * np.dot(points_deviation,
                                                              points_deviation.T)  # Normalized by w_sum as per Betz et al
            sigma_matrix = cov_scale ** 2 * sigma_matrix

            mcmc_log_pdf_target = self._target_generator(self.evaluate_log_intermediate,
                                                         self.evaluate_log_reference, current_tempering_parameter)

            self.logger.info('Begin Resampling')
            # Resampling and MH-MCMC step
            for i in range(self.n_resamples):

                # Resampling from previous tempering level
                lead_index = int(self.random_state.choice(pts_index, p=weight_probabilities))
                lead = points_copy[lead_index]

                # Defining the default proposal
                if self.proposal_given_flag is False:
                    self.proposal = MultivariateNormal(lead, cov=sigma_matrix)

                # Single MH-MCMC step
                x = MetropolisHastings(dimension=self.__dimension, log_pdf_target=mcmc_log_pdf_target, seed=list(lead),
                                       nsamples=1, n_chains=1, burn_length=self.resampling_burn_length,
                                       proposal=self.proposal, random_state=self.random_state,
                                       proposal_is_symmetric=self.proposal_is_symmetric)

                # Setting the generated sample in the array
                points[i] = x.samples
                points_copy[lead_index] = x.samples

                if self.recalculate_weights:
                    weights[lead_index] = np.exp(
                        self.evaluate_log_intermediate(points[lead_index, :].reshape((1, -1)), current_tempering_parameter)
                        - self.evaluate_log_intermediate(points[lead_index, :].reshape((1, -1)), previous_tempering_parameter))
                    w_sum = np.sum(weights)
                    for j in range(nsamples):
                        weight_probabilities[j] = weights[j] / w_sum

            self.logger.info('Begin MCMC')
            mcmc_seed = self._mcmc_seed_generator(resampled_pts=points[0:self.n_resamples, :],
                                                  arr_length=self.n_resamples,
                                                  seed_length=self.__n_chains,
                                                  random_state=self.random_state)

            y = copy.deepcopy(self.sampler)
            self.update_target_and_seed(y, mcmc_seed, mcmc_log_pdf_target)
            y = self.sampler.__copy__(log_pdf_target=mcmc_log_pdf_target, seed=mcmc_seed,
                                      nsamples_per_chain=self.n_samples_per_chain,
                                      concatenate_chains=True, random_state=self.random_state)
            points[self.n_resamples:, :] = y.samples

            if self.save_intermediate_samples is True:
                self.intermediate_samples += [points.copy()]

            self.logger.info('Tempering level ended')

        # Setting the calculated values to the attributes
        self.samples = points
        self.evidence = evidence_estimator

    def update_target_and_seed(self, mcmc_class, mcmc_seed, mcmc_log_pdf_target):
        mcmc_class.seed = mcmc_seed
        mcmc_class.log_pdf_target = mcmc_log_pdf_target
        mcmc_class.pdf_target = None
        (mcmc_class.evaluate_log_target, mcmc_class.evaluate_log_target_marginals,) = \
            mcmc_class._preprocess_target(pdf_=None, log_pdf_=mcmc_class.log_pdf_target, args=None)

    def evaluate_normalization_constant(self):
        return self.evidence

    @staticmethod
    def _find_temper_param(temper_param_prev, samples, q_func, n, iter_lim=1000, iter_thresh=0.00001):
        """
        Find the tempering parameter for the next intermediate target using bisection search between 1.0 and the
        previous tempering parameter (taken to be 0.0 for the first level).

        **Inputs:**

        * **temper_param_prev** ('float'):
            The value of the previous tempering parameter

        * **samples** (`ndarray`):
            Generated samples from the previous intermediate target distribution

        * **q_func** (callable):
            The intermediate distribution (called 'self.evaluate_log_intermediate' in this code)

        * **n** ('int'):
            Number of samples

        * **iter_lim** ('int'):
            Number of iterations to run the bisection search algorithm for, to avoid infinite loops

        * **iter_thresh** ('float'):
            Threshold on the bisection interval, to avoid infinite loops
        """
        bot = temper_param_prev
        top = 1.0
        flag = 0  # Indicates when the tempering exponent has been found (flag = 1 => solution found)
        loop_counter = 0
        while flag == 0:
            loop_counter += 1
            q_scaled = np.zeros(n)
            temper_param_trial = ((bot + top) / 2)
            for i2 in range(n):
                q_scaled[i2] = np.exp(q_func(samples[i2, :].reshape((1, -1)), 1)
                                      - q_func(samples[i2, :].reshape((1, -1)), temper_param_prev))
            sigma_1 = np.std(q_scaled)
            mu_1 = np.mean(q_scaled)
            if sigma_1 < mu_1:
                flag = 1
                temper_param_trial = 1
                continue
            for i3 in range(n):
                q_scaled[i3] = np.exp(q_func(samples[i3, :].reshape((1, -1)), temper_param_trial)
                                      - q_func(samples[i3, :].reshape((1, -1)), temper_param_prev))
            sigma = np.std(q_scaled)
            mu = np.mean(q_scaled)
            if sigma < (0.9 * mu):
                bot = temper_param_trial
            elif sigma > (1.1 * mu):
                top = temper_param_trial
            else:
                flag = 1
            if loop_counter > iter_lim:
                flag = 2
                raise RuntimeError('UQpy: unable to find tempering exponent due to nonconvergence')
            if top - bot <= iter_thresh:
                flag = 3
                raise RuntimeError('UQpy: unable to find tempering exponent due to nonconvergence')
        return temper_param_trial

    def _preprocess_reference(self, dist_, seed_=None, nsamples=None, dimension=None, random_state=None):
        """
        Preprocess the target pdf inputs.

        Utility function (static method), that if given a distribution object, returns the log pdf of the target
        distribution of the first tempering level (the prior in a Bayesian setting), and generates the samples from this
        level. If instead the samples of the first level are passed, then the function passes these samples to the rest
        of the algorithm, and does a Kernel Density Approximation to estimate the log pdf of the target distribution for
        this level (as specified by the given sample points).

        **Inputs:**

        * seed_ ('ndarray'): The samples of the first tempering level
        * prior_ ('Distribution' object): Target distribution for the first tempering level
        * nsamples (int): Number of samples to be generated
        * dimension (int): The dimension  of the sample space

        **Output/Returns:**

        * evaluate_log_pdf (callable): Callable that computes the log of the target density function (the prior)
        """

        if dist_ is not None:
            if not (isinstance(dist_, Distribution)):
                raise TypeError('UQpy: A UQpy.Distribution object must be provided.')
            else:
                evaluate_log_pdf = (lambda x: dist_.log_pdf(x))
                if seed_ is not None:
                    if seed_.shape[0] == nsamples and seed_.shape[1] == dimension:
                        seed_values = seed_
                    else:
                        raise TypeError('UQpy: the seed values should be a numpy array of size (nsamples, dimension)')
                else:
                    seed_values = dist_.rvs(nsamples=nsamples, random_state=random_state)
        else:
            raise ValueError('UQpy: prior distribution must be provided')
        return evaluate_log_pdf, seed_values

    @staticmethod
    def _mcmc_seed_generator(resampled_pts, arr_length, seed_length, random_state):
        """
        Generates the seed from the resampled samples for the mcmc step

        Utility function (static method), that returns a selection of the resampled points (at any tempering level) to
        be used as the seed for the following mcmc exploration step.

        **Inputs:**

        * resampled_pts ('ndarray'): The resampled samples of the tempering level
        * arr_length (int): Length of resampled_pts
        * seed_length (int): Number of samples needed in the seed (same as nchains)

        **Output/Returns:**

        * evaluate_log_pdf (callable): Callable that computes the log of the target density function (the prior)
        """
        index_arr = np.arange(arr_length)
        seed_indices = random_state.choice(index_arr, size=seed_length, replace=False)
        mcmc_seed = resampled_pts[seed_indices, :]
        return mcmc_seed
