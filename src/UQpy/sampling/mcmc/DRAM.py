import logging
from beartype import beartype
from UQpy.sampling.mcmc.baseclass.MCMC import MCMC
from UQpy.distributions import *
from UQpy.sampling.input_data.DramInput import DramInput
from UQpy.utilities.ValidationTypes import *


class DRAM(MCMC):
    """
    Delayed Rejection Adaptive Metropolis algorithm

    In this algorithm, the proposal density is Gaussian and its covariance C is being updated from samples as
    C = sp * C_sample where C_sample is the sample covariance. Also, the delayed rejection scheme is applied, i.e,
    if a candidate is not accepted another one is generated from the proposal with covariance gamma_2 ** 2 * C.

    **References:**

    1. Heikki Haario, Marko Laine, Antonietta Mira, and Eero Saksman. "DRAM: Efficient adaptive mcmc".
       Statistics and Computing, 16(4):339–354, 2006
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

    @beartype
    def __init__(self,
                 dram_input: DramInput,
                 samples_number: int = None,
                 samples_number_per_chain: int = None):

        super().__init__(pdf_target=dram_input.pdf_target, log_pdf_target=dram_input.log_pdf_target,
                         args_target=dram_input.args_target, dimension=dram_input.dimension,
                         seed=dram_input.seed, burn_length=dram_input.burn_length, jump=dram_input.jump,
                         save_log_pdf=dram_input.save_log_pdf, concatenate_chains=dram_input.concatenate_chains,
                         random_state=dram_input.random_state, chains_number=dram_input.chains_number)

        self.logger = logging.getLogger(__name__)
        # Check the initial covariance
        self.initial_covariance = dram_input.initial_covariance
        if self.initial_covariance is None:
            self.initial_covariance = np.eye(self.dimension)
        elif not (isinstance(self.initial_covariance, np.ndarray)
                  and self.initial_covariance == (self.dimension, self.dimension)):
            raise TypeError('UQpy: Input initial_covariance should be a 2D ndarray of shape (dimension, dimension)')

        self.covariance_update_rate = dram_input.covariance_update_rate
        self.scale_parameter = dram_input.scale_parameter
        if self.scale_parameter is None:
            self.scale_parameter = 2.38 ** 2 / self.dimension
        self.delayed_rejection_scale = dram_input.delayed_rejection_scale
        self.save_covariance = dram_input.save_covariance
        for key, typ in zip(['covariance_update_rate', 'scale_parameter', 'delayed_rejection_scale', 'save_covariance'],
                            [int, float, float, bool]):
            if not isinstance(getattr(self, key), typ):
                raise TypeError('Input ' + key + ' must be of type ' + typ.__name__)

        # initialize the sample mean and sample covariance that you need
        self.current_covariance = np.tile(self.initial_covariance[np.newaxis, ...], (self.chains_number, 1, 1))
        self.sample_mean = np.zeros((self.chains_number, self.dimension,))
        self.sample_covariance = np.zeros((self.chains_number, self.dimension, self.dimension))
        if self.save_covariance:
            self.adaptive_covariance = [self.current_covariance.copy(), ]

        self.logger.info('\nUQpy: Initialization of ' + self.__class__.__name__ + ' algorithm complete.')

        # If nsamples is provided, run the algorithm
        if (samples_number is not None) or (samples_number_per_chain is not None):
            self.run(samples_number=samples_number, samples_number_per_chain=samples_number_per_chain)

    def run_one_iteration(self, current_state, current_log_pdf):
        """
        Run one iteration of the mcmc chain for DRAM algorithm, starting at current state -
        see ``mcmc`` class.
        """
        from UQpy.distributions import MultivariateNormal
        multivariate_normal = MultivariateNormal(mean=np.zeros(self.dimension, ), cov=1.)

        # Sample candidate
        candidate = np.zeros_like(current_state)
        for nc, current_cov in enumerate(self.current_covariance):
            multivariate_normal.update_parameters(cov=current_cov)
            candidate[nc, :] = current_state[nc, :] + multivariate_normal.rvs(
                nsamples=1, random_state=self.random_state).reshape((self.dimension,))

        # Compute log_pdf_target of candidate sample
        log_p_candidate = self.evaluate_log_target(candidate)

        # Compare candidate with current sample and decide or not to keep the candidate (loop over nc chains)
        accept_vec = np.zeros((self.chains_number,))
        delayed_chains_indices = []  # indices of chains that will undergo delayed rejection
        unif_rvs = Uniform().rvs(nsamples=self.chains_number, random_state=self.random_state).reshape((-1,))
        for nc, (cand, log_p_cand, log_p_curr) in enumerate(zip(candidate, log_p_candidate, current_log_pdf)):
            accept = np.log(unif_rvs[nc]) < log_p_cand - log_p_curr
            if accept:
                current_state[nc, :] = cand
                current_log_pdf[nc] = log_p_cand
                accept_vec[nc] += 1.
            else:  # enter delayed rejection
                delayed_chains_indices.append(nc)  # these indices will enter the delayed rejection part

        # Delayed rejection
        if len(delayed_chains_indices) > 0:  # performed delayed rejection for some chains
            current_states_delayed = np.zeros((len(delayed_chains_indices), self.dimension))
            candidates_delayed = np.zeros((len(delayed_chains_indices), self.dimension))
            candidate2 = np.zeros((len(delayed_chains_indices), self.dimension))
            # Sample other candidates closer to the current one
            for i, nc in enumerate(delayed_chains_indices):
                current_states_delayed[i, :] = current_state[nc, :]
                candidates_delayed[i, :] = candidate[nc, :]
                multivariate_normal \
                    .update_parameters(cov=self.delayed_rejection_scale ** 2 * self.current_covariance[nc])
                candidate2[i, :] = current_states_delayed[i, :] + multivariate_normal.rvs(
                    nsamples=1, random_state=self.random_state).reshape((self.dimension,))
            # Evaluate their log_target
            log_p_candidate2 = self.evaluate_log_target(candidate2)
            log_prop_cand_cand2 = multivariate_normal.log_pdf(candidates_delayed - candidate2)
            log_prop_cand_curr = multivariate_normal.log_pdf(candidates_delayed - current_states_delayed)
            # Accept or reject
            unif_rvs = Uniform().rvs(nsamples=len(delayed_chains_indices),
                                     random_state=self.random_state).reshape((-1,))
            for (nc, cand2, log_p_cand2, j1, j2, u_rv) in zip(delayed_chains_indices, candidate2, log_p_candidate2,
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
        for nc in range(self.chains_number):
            # update covariance
            self.sample_mean[nc], self.sample_covariance[nc] = self._recursive_update_mean_covariance(
                samples_number=self.iterations_number, new_sample=current_state[nc, :],
                previous_mean=self.sample_mean[nc], previous_covariance=self.sample_covariance[nc])
            if (self.iterations_number > 1) and (self.iterations_number % self.covariance_update_rate == 0):
                self.current_covariance[nc] = self.scale_parameter * self.sample_covariance[nc] + \
                                              1e-6 * np.eye(self.dimension)
        if self.save_covariance and ((self.iterations_number > 1) and
                                     (self.iterations_number % self.covariance_update_rate == 0)):
            self.adaptive_covariance.append(self.current_covariance.copy())

        # Update the acceptance rate
        self._update_acceptance_rate(accept_vec)
        return current_state, current_log_pdf

    @staticmethod
    def _recursive_update_mean_covariance(samples_number, new_sample, previous_mean, previous_covariance=None):
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
        new_mean = (samples_number - 1) / samples_number * previous_mean + 1 / samples_number * new_sample
        if previous_covariance is None:
            return new_mean
        dimensions = new_sample.size
        if samples_number == 1:
            new_covariance = np.zeros((dimensions, dimensions))
        else:
            delta_n = (new_sample - previous_mean).reshape((dimensions, 1))
            new_covariance = \
                (samples_number - 2) / (samples_number - 1) * previous_covariance + \
                1 / samples_number * np.matmul(delta_n, delta_n.T)
        return new_mean, new_covariance

    def __copy__(self):
        new = self.__class__(pdf_target=self.pdf_target,
                             log_pdf_target=self.log_pdf_target,
                             args_target=self.args_target,
                             burn_length=self.burn_length,
                             jump=self.jump,
                             dimension=self.dimension,
                             seed=self.seed,
                             save_log_pdf=self.save_log_pdf,
                             concatenate_chains=self.concatenate_chains,
                             initial_covariance=self.initial_covariance,
                             covariance_update_rate=self.covariance_update_rate,
                             scale_parameter=self.scale_parameter,
                             delayed_rejection_scale=self.delayed_rejection_scale,
                             save_covariance=self.save_covariance,
                             chains_number=self.chains_number,
                             random_state=self.random_state)
        new.__dict__.update(self.__dict__)

        return new
