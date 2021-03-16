from UQpy.SampleMethods.MCMC.mcmc import MCMC
from UQpy.Distributions import *
import numpy as np

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

