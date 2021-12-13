import logging
from typing import Callable, List

from beartype import beartype
from UQpy.sampling.mcmc.baseclass.MCMC import MCMC
from UQpy.distributions import *
from UQpy.utilities.ValidationTypes import *


class DRAM(MCMC):

    @beartype
    def __init__(
            self,
            pdf_target: Union[Callable, List[Callable]] = None,
            log_pdf_target: Union[Callable, List[Callable]] = None,
            args_target: tuple = None,
            burn_length: Annotated[int, Is[lambda x: x >= 0]] = 0,
            jump: int = 1,
            dimension: int = None,
            seed: list = None,
            save_log_pdf: bool = False,
            concatenate_chains: bool = True,
            initial_covariance: float = None,
            covariance_update_rate: float = 100,
            scale_parameter: float = None,
            delayed_rejection_scale: float = 1 / 5,
            save_covariance: bool = False,
            random_state: RandomStateType = None,
            chains_number: int = None,
            samples_number: int = None,
            samples_number_per_chain: int = None,
    ):
        """
        Delayed Rejection Adaptive Metropolis algorithm :cite:`Dram1` :cite:`MCMC2`

        In this algorithm, the proposal density is Gaussian and its covariance C is being updated from samples as
        C = sp * C_sample where C_sample is the sample covariance. Also, the delayed rejection scheme is applied, i.e,
        if a candidate is not accepted another one is generated from the proposal with covariance gamma_2 ** 2 * C.

        :param pdf_target: Target density function from which to draw random samples. Either pdf_target or
         log_pdf_target must be provided (the latter should be preferred for better numerical stability).

         If pdf_target is a callable, it refers to the joint pdf to sample from, it must take at least one input x, which
         are the point(s) at which to evaluate the pdf. Within MCMC the pdf_target is evaluated as:
         p(x) = pdf_target(x, \*args_target)

         where x is a ndarray of shape (samples_number, dimension) and args_target are additional positional arguments that
         are provided to MCMC via its args_target input.

         If pdf_target is a list of callables, it refers to independent marginals to sample from. The marginal in dimension
         j is evaluated as: p_j(xj) = pdf_target[j](xj, \*args_target[j]) where x is a ndarray of shape (samples_number,
         dimension)
        :param log_pdf_target: Logarithm of the target density function from which to draw random samples.
         Either pdf_target or log_pdf_target must be provided (the latter should be preferred for better numerical
         stability).

         Same comments as for input pdf_target.
        :param args_target: Positional arguments of the pdf / log-pdf target function. See pdf_target
        :param burn_length: Length of burn-in - i.e., number of samples at the beginning of the chain to discard (note:
         no thinning during burn-in). Default is 0, no burn-in.
        :param jump: Thinning parameter, used to reduce correlation between samples. Setting jump=n corresponds to
         skipping n-1 states between accepted states of the chain. Default is 1 (no thinning).
        :param dimension: A scalar value defining the dimension of target density function. Either dimension and
         nchains or seed must be provided.
        :param seed: Seed of the Markov chain(s), shape (chains_number, dimension).
         Default: zeros(chains_number x dimension).

         If seed is not provided, both chains_number and dimension must be provided.
        :param save_log_pdf: Boolean that indicates whether to save log-pdf values along with the samples.
         Default: False
        :param concatenate_chains: Boolean that indicates whether to concatenate the chains after a run, i.e., samples
         are stored as an ndarray of shape (samples_number * chains_number, dimension) if True,
         (samples_number, chains_number, dimension) if False.
         Default: True
        :param chains_number: The number of Markov chains to generate. Either dimension and chains_number or seed must be
         provided.
        :param initial_covariance: Initial covariance for the gaussian proposal distribution. Default: I(dim)
        :param covariance_update_rate: Rate at which covariance is being updated, i.e., every k0 iterations.
         Default: 100
        :param scale_parameter: Scale parameter for covariance updating. Default: 2.38 ** 2 / dim
        :param delayed_rejection_scale: Scale parameter for delayed rejection. Default: 1 / 5
        :param save_covariance: If True, updated covariance is saved in attribute adaptive_covariance. Default: False
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is
         None.
        :param samples_number: Number of samples to generate.
        :param samples_number_per_chain: Number of samples to generate per chain.
        """
        super().__init__(
            pdf_target=pdf_target,
            log_pdf_target=log_pdf_target,
            args_target=args_target,
            dimension=dimension,
            seed=seed,
            burn_length=burn_length,
            jump=jump,
            save_log_pdf=save_log_pdf,
            concatenate_chains=concatenate_chains,
            random_state=random_state,
            chains_number=chains_number,
        )

        self.logger = logging.getLogger(__name__)
        # Check the initial covariance
        self.initial_covariance = initial_covariance
        if self.initial_covariance is None:
            self.initial_covariance = np.eye(self.dimension)
        elif not (isinstance(self.initial_covariance, np.ndarray)
                  and self.initial_covariance == (self.dimension, self.dimension)):
            raise TypeError(
                "UQpy: Input initial_covariance should be a 2D ndarray of shape (dimension, dimension)")

        self.covariance_update_rate = covariance_update_rate
        self.scale_parameter = scale_parameter
        if self.scale_parameter is None:
            self.scale_parameter = 2.38 ** 2 / self.dimension
        self.delayed_rejection_scale = delayed_rejection_scale
        self.save_covariance = save_covariance
        for key, typ in zip(
                [
                    "covariance_update_rate",
                    "scale_parameter",
                    "delayed_rejection_scale",
                    "save_covariance",
                ],
                [int, float, float, bool],
        ):
            if not isinstance(getattr(self, key), typ):
                raise TypeError("Input " + key + " must be of type " + typ.__name__)

        # initialize the sample mean and sample covariance that you need
        self.current_covariance = np.tile(
            self.initial_covariance[np.newaxis, ...], (self.chains_number, 1, 1))
        self.sample_mean = np.zeros((self.chains_number, self.dimension,))
        self.sample_covariance = np.zeros((self.chains_number, self.dimension, self.dimension))
        if self.save_covariance:
            self.adaptive_covariance = [self.current_covariance.copy(), ]

        self.logger.info("\nUQpy: Initialization of " + self.__class__.__name__ + " algorithm complete.")

        # If nsamples is provided, run the algorithm
        if (samples_number is not None) or (samples_number_per_chain is not None):
            self.run(samples_number=samples_number,
                     samples_number_per_chain=samples_number_per_chain, )

    def run_one_iteration(self, current_state, current_log_pdf):
        """
        Run one iteration of the mcmc chain for DRAM algorithm, starting at current state -
        see :class:`MCMC` class.
        """
        from UQpy.distributions import MultivariateNormal

        multivariate_normal = MultivariateNormal(mean=np.zeros(self.dimension, ), cov=1.0)

        # Sample candidate
        candidate = np.zeros_like(current_state)
        for nc, current_cov in enumerate(self.current_covariance):
            multivariate_normal.update_parameters(cov=current_cov)
            candidate[nc, :] = current_state[nc, :] + \
                               multivariate_normal.rvs(nsamples=1, random_state=self.random_state) \
                                   .reshape((self.dimension,))

        # Compute log_pdf_target of candidate sample
        log_p_candidate = self.evaluate_log_target(candidate)

        # Compare candidate with current sample and decide or not to keep the candidate (loop over nc chains)
        accept_vec = np.zeros((self.chains_number,))
        delayed_chains_indices = ([])  # indices of chains that will undergo delayed rejection
        unif_rvs = (Uniform().rvs(nsamples=self.chains_number, random_state=self.random_state)
                    .reshape((-1,)))
        for nc, (cand, log_p_cand, log_p_curr) in enumerate(
                zip(candidate, log_p_candidate, current_log_pdf)):
            accept = np.log(unif_rvs[nc]) < log_p_cand - log_p_curr
            if accept:
                current_state[nc, :] = cand
                current_log_pdf[nc] = log_p_cand
                accept_vec[nc] += 1.0
            else:  # enter delayed rejection
                delayed_chains_indices.append(nc)  # these indices will enter the delayed rejection part

        # Delayed rejection
        if delayed_chains_indices:  # performed delayed rejection for some chains
            current_states_delayed = np.zeros(
                (len(delayed_chains_indices), self.dimension))
            candidates_delayed = np.zeros((len(delayed_chains_indices), self.dimension))
            candidate2 = np.zeros((len(delayed_chains_indices), self.dimension))
            # Sample other candidates closer to the current one
            for i, nc in enumerate(delayed_chains_indices):
                current_states_delayed[i, :] = current_state[nc, :]
                candidates_delayed[i, :] = candidate[nc, :]
                multivariate_normal.update_parameters(
                    cov=self.delayed_rejection_scale ** 2 * self.current_covariance[nc])
                candidate2[i, :] = current_states_delayed[i, :] + \
                                   multivariate_normal.rvs(nsamples=1, random_state=self.random_state) \
                                       .reshape((self.dimension,))
            # Evaluate their log_target
            log_p_candidate2 = self.evaluate_log_target(candidate2)
            log_prop_cand_cand2 = multivariate_normal.log_pdf(candidates_delayed - candidate2)
            log_prop_cand_curr = multivariate_normal.log_pdf(candidates_delayed - current_states_delayed)
            # Accept or reject
            unif_rvs = (Uniform().rvs(nsamples=len(delayed_chains_indices),
                                      random_state=self.random_state).reshape((-1,)))
            for (nc, cand2, log_p_cand2, j1, j2, u_rv) in zip(
                    delayed_chains_indices,
                    candidate2,
                    log_p_candidate2,
                    log_prop_cand_cand2,
                    log_prop_cand_curr,
                    unif_rvs,
            ):
                alpha_cand_cand2 = min(1.0, np.exp(log_p_candidate[nc] - log_p_cand2))
                alpha_cand_curr = min(1.0, np.exp(log_p_candidate[nc] - current_log_pdf[nc]))
                log_alpha2 = (log_p_cand2 - current_log_pdf[nc] + j1 - j2
                              + np.log(max(1.0 - alpha_cand_cand2, 10 ** (-320)))
                              - np.log(max(1.0 - alpha_cand_curr, 10 ** (-320))))
                accept = np.log(u_rv) < min(0.0, log_alpha2)
                if accept:
                    current_state[nc, :] = cand2
                    current_log_pdf[nc] = log_p_cand2
                    accept_vec[nc] += 1.0

        # Adaptive part: update the covariance
        for nc in range(self.chains_number):
            # update covariance
            self.sample_mean[nc], self.sample_covariance[nc], = self._recursive_update_mean_covariance(
                samples_number=self.iterations_number,
                new_sample=current_state[nc, :],
                previous_mean=self.sample_mean[nc],
                previous_covariance=self.sample_covariance[nc], )
            if (self.iterations_number > 1) and (self.iterations_number % self.covariance_update_rate == 0):
                self.current_covariance[nc] = self.scale_parameter * self.sample_covariance[nc] + \
                                              1e-6 * np.eye(self.dimension)
        if self.save_covariance and \
                ((self.iterations_number > 1) and (self.iterations_number % self.covariance_update_rate == 0)):
            self.adaptive_covariance.append(self.current_covariance.copy())

        # Update the acceptance rate
        self._update_acceptance_rate(accept_vec)
        return current_state, current_log_pdf

    @staticmethod
    def _recursive_update_mean_covariance(
            samples_number, new_sample, previous_mean, previous_covariance=None
    ):
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
            new_covariance = (samples_number - 2) / (samples_number - 1) \
                             * previous_covariance + 1 / samples_number * np.matmul(delta_n, delta_n.T)
        return new_mean, new_covariance
