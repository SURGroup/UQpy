from UQpy.SampleMethods.MCMC.mcmc import MCMC
from UQpy.Distributions import *
import numpy as np

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
