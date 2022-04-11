import logging
from typing import Callable
import warnings

import numpy as np

warnings.filterwarnings('ignore')

from beartype import beartype
from UQpy.sampling.mcmc.baseclass.MCMC import MCMC
from UQpy.distributions import *
from UQpy.utilities.ValidationTypes import *


class DREAM(MCMC):

    @beartype
    def __init__(
            self,
            pdf_target: Union[Callable, list[Callable]] = None,
            log_pdf_target: Union[Callable, list[Callable]] = None,
            args_target: tuple = None,
            burn_length: Annotated[int, Is[lambda x: x >= 0]] = 0,
            jump: PositiveInteger = 1,
            dimension: int = None,
            seed: list = None,
            save_log_pdf: bool = False,
            concatenate_chains: bool = True,
            jump_rate: int = 3,
            c: float = 0.1,
            c_star: float = 1e-6,
            crossover_probabilities_number: int = 3,
            gamma_probability: float = 0.2,
            crossover_adaptation: tuple = (-1, 1),
            check_chains: tuple = (-1, 1),
            random_state: RandomStateType = None,
            n_chains: int = None,
            nsamples: int = None,
            nsamples_per_chain: int = None,
    ):
        """
        DiffeRential Evolution Adaptive Metropolis algorithm :cite:`Dream1` :cite:`Dream2`

        :param pdf_target: Target density function from which to draw random samples. Either `pdf_target` or
         `log_pdf_target` must be provided (the latter should be preferred for better numerical stability).

         If `pdf_target` is a callable, it refers to the joint pdf to sample from, it must take at least one input
         **x**, which are the point(s) at which to evaluate the pdf. Within :class:`.MCMC` the `pdf_target` is evaluated
         as:
         :code:`p(x) = pdf_target(x, \*args_target)`

         where **x** is a :class:`numpy.ndarray  of shape :code:`(nsamples, dimension)` and `args_target` are additional
         positional arguments that are provided to :class:`.MCMC` via its `args_target` input.

         If `pdf_target` is a list of callables, it refers to independent marginals to sample from. The marginal in
         dimension :code:`j` is evaluated as: :code:`p_j(xj) = pdf_target[j](xj, \*args_target[j])` where **x** is a
         :class:`numpy.ndarray` of shape :code:`(nsamples, dimension)`
        :param log_pdf_target: Logarithm of the target density function from which to draw random samples.
         Either `pdf_target` or `log_pdf_target` must be provided (the latter should be preferred for better numerical
         stability).

         Same comments as for input `pdf_target`.
        :param args_target: Positional arguments of the pdf / log-pdf target function. See `pdf_target`
        :param burn_length: Length of burn-in - i.e., number of samples at the beginning of the chain to discard (note:
         no thinning during burn-in). Default is :math:`0`, no burn-in.
        :param jump: Thinning parameter, used to reduce correlation between samples. Setting :code:`jump=n` corresponds
         to skipping :code:`n-1` states between accepted states of the chain. Default is :math:`1` (no thinning).
        :param dimension: A scalar value defining the dimension of target density function. Either `dimension` and
         `n_chains` or `seed` must be provided.
        :param seed: Seed of the Markov chain(s), shape :code:`(n_chains, dimension)`.
         Default: :code:`zeros(n_chains x dimension)`.

         If `seed` is not provided, both `n_chains` and `dimension` must be provided.
        :param save_log_pdf: Boolean that indicates whether to save log-pdf values along with the samples.
         Default: :any:`False`
        :param concatenate_chains: Boolean that indicates whether to concatenate the chains after a run, i.e., samples
         are stored as a :class:`numpy.ndarray` of shape :code:`(nsamples * n_chains, dimension)` if :any:`True`,
         :code:`(nsamples, n_chains, dimension)` if :any:`False`.
         Default: :any:`True`
        :param n_chains: The number of Markov chains to generate. Either `dimension` and `n_chains` or `seed` must be
         provided.
        :param jump_rate: Jump rate. Default: :math:`3`
        :param c: Differential evolution parameter. Default: :math:`0.1`
        :param c_star: Differential evolution parameter, should be small compared to width of target.
         Default: :math:`1e-6`
        :param crossover_probabilities_number: Number of crossover probabilities. Default: :math:`3`
        :param gamma_probability: :code:`Prob(gamma=1)`. Default: :math:`0.2`
        :param crossover_adaptation: :code:`(iter_max, rate)` governs adaptation of crossover probabilities (adapts
         every rate iterations if :code:`iter<iter_max`). Default: :code:`(-1, 1)`, i.e., no adaptation
        :param check_chains: :code:`(iter_max, rate)` governs discarding of outlier chains (discard every rate
         iterations if :code:`iter<iter_max`). Default: :code:`(-1, 1)`, i.e., no check on outlier chains
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is
         :any:`None`.

        :param nsamples: Number of samples to generate.
        :param nsamples_per_chain: Number of samples to generate per chain.
        """
        self.nsamples = nsamples
        self.nsamples_per_chain = nsamples_per_chain
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
            n_chains=n_chains,
        )

        self.logger = logging.getLogger(__name__)
        # Check nb of chains
        if self.n_chains < 2:
            raise ValueError("UQpy: For the DREAM algorithm, a seed must be provided with at least two samples.")

        # Check user-specific algorithms
        self.jump_rate = jump_rate
        self.c = c
        self.c_star = c_star
        self.crossover_probabilities_number = crossover_probabilities_number
        self.gamma_probability = gamma_probability
        self.crossover_adaptation = crossover_adaptation
        self.check_chains = check_chains

        for key, typ in zip(
                [
                    "jump_rate",
                    "c",
                    "c_star",
                    "crossover_probabilities_number",
                    "gamma_probability",
                ],
                [int, float, float, int, float],
        ):
            if not isinstance(getattr(self, key), typ):
                raise TypeError("Input " + key + " must be of type " + typ.__name__)
        if (self.dimension is not None and self.crossover_probabilities_number > self.dimension):
            self.crossover_probabilities_number = self.dimension
        for key in ["crossover_adaptation", "check_chains"]:
            p = getattr(self, key)
            if not (isinstance(p, tuple) and len(p) == 2 and all(isinstance(i, (int, float)) for i in p)):
                raise TypeError("Inputs " + key + " must be a tuple of 2 integers.")
        if (not self.save_log_pdf) and (self.check_chains[0] > 0):
            raise ValueError("UQpy: Input save_log_pdf must be True in order to check outlier chains")

        # Initialize a few other variables
        self.j_ind = np.zeros((self.crossover_probabilities_number,))
        self.n_id = np.zeros((self.crossover_probabilities_number,))
        self.cross_prob = (
                np.ones((self.crossover_probabilities_number,))
                / self.crossover_probabilities_number)

        self.logger.info("UQpy: Initialization of " + self.__class__.__name__ + " algorithm complete.\n")

        if (nsamples is not None) or (nsamples_per_chain is not None):
            self.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain, )

    def run_one_iteration(self, current_state: np.ndarray, current_log_pdf: np.ndarray):
        """
        Run one iteration of the mcmc chain for DREAM algorithm, starting at current state -
        see :class:`MCMC` class.
        """
        r_diff = np.array([np.setdiff1d(np.arange(self.n_chains), j) for j in range(self.n_chains)])
        cross = (np.arange(1, self.crossover_probabilities_number + 1) / self.crossover_probabilities_number)

        # Dynamic part: evolution of chains
        unif_rvs = (Uniform().rvs(nsamples=self.n_chains * (self.n_chains - 1),
                                  random_state=self.random_state, )
                    .reshape((self.n_chains - 1, self.n_chains)))
        draw = np.argsort(unif_rvs, axis=0)
        dx = np.zeros_like(current_state)
        lmda = (Uniform(scale=2 * self.c).rvs(nsamples=self.n_chains, random_state=self.random_state)
                .reshape((-1,)))
        std_x_tmp = np.std(current_state, axis=0)

        multi_rvs = Multinomial(n=1, p=[1.0 / self.jump_rate, ] * self.jump_rate).rvs(
            nsamples=self.n_chains, random_state=self.random_state)
        d_ind = np.nonzero(multi_rvs)[1]
        as_ = [r_diff[j, draw[slice(d_ind[j]), j]] for j in range(self.n_chains)]
        bs_ = [r_diff[j, draw[slice(d_ind[j], 2 * d_ind[j], 1), j]] for j in range(self.n_chains)]
        multi_rvs = Multinomial(n=1, p=self.cross_prob).rvs(
            nsamples=self.n_chains, random_state=self.random_state)
        id_ = np.nonzero(multi_rvs)[1]
        # id = np.random.choice(self.n_CR, size=(self.nchains, ), replace=True, trial_probability=self.pCR)
        z = (Uniform().rvs(nsamples=self.n_chains * self.dimension,
                           random_state=self.random_state, )
             .reshape((self.n_chains, self.dimension)))
        subset_a = [np.where(z_j < cross[id_j])[0] for (z_j, id_j) in zip(z, id_)]  # subset A of selected dimensions
        d_star = np.array([len(a_j) for a_j in subset_a])
        for j in range(self.n_chains):
            if d_star[j] == 0:
                subset_a[j] = np.array([np.argmin(z[j])])
                d_star[j] = 1
        gamma_d = 2.38 / np.sqrt(2 * (d_ind + 1) * d_star)
        g = (Binomial(n=1, p=self.gamma_probability).rvs(nsamples=self.n_chains, random_state=self.random_state)
             .reshape((-1,)))
        g[g == 0] = gamma_d[g == 0]
        norm_vars = (Normal(loc=0.0, scale=1.0).rvs(nsamples=self.n_chains ** 2, random_state=self.random_state)
                     .reshape((self.n_chains, self.n_chains)))
        for j in range(self.n_chains):
            for i in subset_a[j]:
                dx[j, i] = self.c_star * norm_vars[j, i] + (1 + lmda[j]) * g[j] * np.sum(current_state[as_[j], i] -
                                                                                         current_state[bs_[j], i])
        candidates = current_state + dx

        # Evaluate log likelihood of candidates
        logp_candidates = self.evaluate_log_target(candidates)

        # Accept or reject
        accept_vec = np.zeros((self.n_chains,))
        unif_rvs = (Uniform().rvs(nsamples=self.n_chains, random_state=self.random_state)
                    .reshape((-1,)))
        for nc, (lpc, candidate, log_p_curr) in enumerate(
                zip(logp_candidates, candidates, current_log_pdf)):
            accept = np.log(unif_rvs[nc]) < lpc - log_p_curr
            if accept:
                current_state[nc, :] = candidate
                current_log_pdf[nc] = lpc
                accept_vec[nc] = 1.0
            else:
                dx[nc, :] = 0
            self.j_ind[id_[nc]] = self.j_ind[id_[nc]] + np.sum((dx[nc, :] / std_x_tmp) ** 2)
            self.n_id[id_[nc]] += 1

        # Save the acceptance rate
        self._update_acceptance_rate(accept_vec)

        # update selection cross prob
        if (self.iterations_number < self.crossover_adaptation[0]
                and self.iterations_number % self.crossover_adaptation[1] == 0):
            self.cross_prob = self.j_ind / self.n_id
            self.cross_prob /= sum(self.cross_prob)
        # check outlier chains (only if you have saved at least 100 values already)
        if ((self.samples_counter >= 100)
                and (self.iterations_number < self.check_chains[0])
                and (self.iterations_number % self.check_chains[1] == 0)):
            self.check_outlier_chains(replace_with_best=True)

        return current_state, current_log_pdf

    def check_outlier_chains(self, replace_with_best: bool = False):
        if not self.save_log_pdf:
            raise ValueError("UQpy: Input save_log_pdf must be True in order to check outlier chains")
        start_ = self.nsamples_per_chain // 2
        avgs_logpdf = np.mean(self.log_pdf_values[start_: self.nsamples_per_chain], axis=0)
        best_ = np.argmax(avgs_logpdf)
        avg_sorted = np.sort(avgs_logpdf)
        ind1, ind3 = (1 + round(0.25 * self.n_chains), 1 + round(0.75 * self.n_chains),)
        q1, q3 = avg_sorted[ind1], avg_sorted[ind3]
        qr = q3 - q1

        outlier_num = 0
        for j in range(self.n_chains):
            if avgs_logpdf[j] < q1 - 2.0 * qr:
                outlier_num += 1
                if replace_with_best:
                    self.samples[start_:, j, :] = self.samples[start_:, best_, :].copy()
                    self.log_pdf_values[start_:, j] = self.log_pdf_values[
                                                      start_:, best_
                                                      ].copy()
                else:
                    self.logger.info("UQpy: Chain {} is an outlier chain".format(j))
        if outlier_num > 0:
            self.logger.info("UQpy: Detected {} outlier chains".format(outlier_num))
