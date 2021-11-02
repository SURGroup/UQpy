import logging

from beartype import beartype

from UQpy.sampling.mcmc.baseclass.MCMC import MCMC
from UQpy.distributions import *

from UQpy.sampling.input_data.DreamInput import DreamInput
from UQpy.utilities.ValidationTypes import *


class DREAM(MCMC):

    @beartype
    def __init__(
        self,
        dream_input: DreamInput,
        samples_number: int = None,
        samples_number_per_chain: int = None,
    ):
        """
        DiffeRential Evolution Adaptive Metropolis algorithm

        **References:**

        1. J.A. Vrugt et al. "Accelerating Markov chain Monte Carlo simulation by differential evolution with
           self-adaptive randomized subspace sampling". International Journal of Nonlinear Sciences and Numerical
           Simulation, 10(3):273–290, 2009.[68]
        2. J.A. Vrugt. "Markov chain Monte Carlo simulation using the DREAM software package: Theory, concepts, and
           MATLAB implementation". Environmental Modelling & Software, 75:273–316, 2016.

        :param dream_input: Object that contains input data to the :class:`.DREAM` class.
         (See :class:`.DreamInput`)
        :param samples_number: Number of samples to generate.
        :param samples_number_per_chain: Number of samples to generate per chain.
        """
        super().__init__(
            pdf_target=dream_input.pdf_target,
            log_pdf_target=dream_input.log_pdf_target,
            args_target=dream_input.args_target,
            dimension=dream_input.dimension,
            seed=dream_input.seed,
            burn_length=dream_input.burn_length,
            jump=dream_input.jump,
            save_log_pdf=dream_input.save_log_pdf,
            concatenate_chains=dream_input.concatenate_chains,
            random_state=dream_input.random_state,
            chains_number=dream_input.chains_number,
        )

        self.logger = logging.getLogger(__name__)
        # Check nb of chains
        if self.chains_number < 2:
            raise ValueError(
                "UQpy: For the DREAM algorithm, a seed must be provided with at least two samples."
            )

        # Check user-specific algorithms
        self.jump_rate = dream_input.jump_rate
        self.c = dream_input.c
        self.c_star = dream_input.c_star
        self.crossover_probabilities_number = dream_input.crossover_probabilities_number
        self.gamma_probability = dream_input.gamma_probability
        self.crossover_adaptation = dream_input.crossover_adaptation
        self.check_chains = dream_input.check_chains

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
        if (
            self.dimension is not None
            and self.crossover_probabilities_number > self.dimension
        ):
            self.crossover_probabilities_number = self.dimension
        for key in ["crossover_adaptation", "check_chains"]:
            p = getattr(self, key)
            if not (
                isinstance(p, tuple)
                and len(p) == 2
                and all(isinstance(i, (int, float)) for i in p)
            ):
                raise TypeError("Inputs " + key + " must be a tuple of 2 integers.")
        if (not self.save_log_pdf) and (self.check_chains[0] > 0):
            raise ValueError(
                "UQpy: Input save_log_pdf must be True in order to check outlier chains"
            )

        # Initialize a few other variables
        self.j_ind = np.zeros((self.crossover_probabilities_number,))
        self.n_id = np.zeros((self.crossover_probabilities_number,))
        self.cross_prob = (
            np.ones((self.crossover_probabilities_number,))
            / self.crossover_probabilities_number
        )

        self.logger.info(
            "UQpy: Initialization of "
            + self.__class__.__name__
            + " algorithm complete.\n"
        )

        # If nsamples is provided, run the algorithm
        if (samples_number is not None) or (samples_number_per_chain is not None):
            self.run(
                samples_number=samples_number,
                samples_number_per_chain=samples_number_per_chain,
            )

    def run_one_iteration(self, current_state, current_log_pdf):
        """
        Run one iteration of the mcmc chain for DREAM algorithm, starting at current state -
        see :class:`MCMC` class.
        """
        r_diff = np.array(
            [
                np.setdiff1d(np.arange(self.chains_number), j)
                for j in range(self.chains_number)
            ]
        )
        cross = (
            np.arange(1, self.crossover_probabilities_number + 1)
            / self.crossover_probabilities_number
        )

        # Dynamic part: evolution of chains
        unif_rvs = (
            Uniform()
            .rvs(
                nsamples=self.chains_number * (self.chains_number - 1),
                random_state=self.random_state,
            )
            .reshape((self.chains_number - 1, self.chains_number))
        )
        draw = np.argsort(unif_rvs, axis=0)
        dx = np.zeros_like(current_state)
        lmda = (
            Uniform(scale=2 * self.c)
            .rvs(nsamples=self.chains_number, random_state=self.random_state)
            .reshape((-1,))
        )
        std_x_tmp = np.std(current_state, axis=0)

        multi_rvs = Multinomial(n=1, p=[1.0 / self.jump_rate,] * self.jump_rate).rvs(
            nsamples=self.chains_number, random_state=self.random_state
        )
        d_ind = np.nonzero(multi_rvs)[1]
        as_ = [r_diff[j, draw[slice(d_ind[j]), j]] for j in range(self.chains_number)]
        bs_ = [
            r_diff[j, draw[slice(d_ind[j], 2 * d_ind[j], 1), j]]
            for j in range(self.chains_number)
        ]
        multi_rvs = Multinomial(n=1, p=self.cross_prob).rvs(
            nsamples=self.chains_number, random_state=self.random_state
        )
        id_ = np.nonzero(multi_rvs)[1]
        # id = np.random.choice(self.n_CR, size=(self.nchains, ), replace=True, trial_probability=self.pCR)
        z = (
            Uniform()
            .rvs(
                nsamples=self.chains_number * self.dimension,
                random_state=self.random_state,
            )
            .reshape((self.chains_number, self.dimension))
        )
        subset_a = [
            np.where(z_j < cross[id_j])[0] for (z_j, id_j) in zip(z, id_)
        ]  # subset A of selected dimensions
        d_star = np.array([len(a_j) for a_j in subset_a])
        for j in range(self.chains_number):
            if d_star[j] == 0:
                subset_a[j] = np.array([np.argmin(z[j])])
                d_star[j] = 1
        gamma_d = 2.38 / np.sqrt(2 * (d_ind + 1) * d_star)
        g = (
            Binomial(n=1, p=self.gamma_probability)
            .rvs(nsamples=self.chains_number, random_state=self.random_state)
            .reshape((-1,))
        )
        g[g == 0] = gamma_d[g == 0]
        norm_vars = (
            Normal(loc=0.0, scale=1.0)
            .rvs(nsamples=self.chains_number ** 2, random_state=self.random_state)
            .reshape((self.chains_number, self.chains_number))
        )
        for j in range(self.chains_number):
            for i in subset_a[j]:
                dx[j, i] = self.c_star * norm_vars[j, i] + (1 + lmda[j]) * g[
                    j
                ] * np.sum(current_state[as_[j], i] - current_state[bs_[j], i])
        candidates = current_state + dx

        # Evaluate log likelihood of candidates
        logp_candidates = self.evaluate_log_target(candidates)

        # Accept or reject
        accept_vec = np.zeros((self.chains_number,))
        unif_rvs = (
            Uniform()
            .rvs(nsamples=self.chains_number, random_state=self.random_state)
            .reshape((-1,))
        )
        for nc, (lpc, candidate, log_p_curr) in enumerate(
            zip(logp_candidates, candidates, current_log_pdf)
        ):
            accept = np.log(unif_rvs[nc]) < lpc - log_p_curr
            if accept:
                current_state[nc, :] = candidate
                current_log_pdf[nc] = lpc
                accept_vec[nc] = 1.0
            else:
                dx[nc, :] = 0
            self.j_ind[id_[nc]] = self.j_ind[id_[nc]] + np.sum(
                (dx[nc, :] / std_x_tmp) ** 2
            )
            self.n_id[id_[nc]] += 1

        # Save the acceptance rate
        self._update_acceptance_rate(accept_vec)

        # update selection cross prob
        if (
            self.iterations_number < self.crossover_adaptation[0]
            and self.iterations_number % self.crossover_adaptation[1] == 0
        ):
            self.cross_prob = self.j_ind / self.n_id
            self.cross_prob /= sum(self.cross_prob)
        # check outlier chains (only if you have saved at least 100 values already)
        if (
            (self.samples_number >= 100)
            and (self.iterations_number < self.check_chains[0])
            and (self.iterations_number % self.check_chains[1] == 0)
        ):
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
            raise ValueError(
                "UQpy: Input save_log_pdf must be True in order to check outlier chains"
            )
        start_ = self.samples_number_per_chain // 2
        avgs_logpdf = np.mean(
            self.log_pdf_values[start_ : self.samples_number_per_chain], axis=0
        )
        best_ = np.argmax(avgs_logpdf)
        avg_sorted = np.sort(avgs_logpdf)
        ind1, ind3 = (
            1 + round(0.25 * self.chains_number),
            1 + round(0.75 * self.chains_number),
        )
        q1, q3 = avg_sorted[ind1], avg_sorted[ind3]
        qr = q3 - q1

        outlier_num = 0
        for j in range(self.chains_number):
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

    def __copy__(self):
        new = self.__class__(
            pdf_target=self.pdf_target,
            log_pdf_target=self.log_pdf_target,
            args_target=self.args_target,
            burn_length=self.burn_length,
            jump=self.jump,
            dimension=self.dimension,
            seed=self.seed,
            save_log_pdf=self.save_log_pdf,
            concatenate_chains=self.concatenate_chains,
            jump_rate=self.jump_rate,
            c=self.c,
            c_star=self.c_star,
            crossover_probabilities_number=self.crossover_probabilities_number,
            gamma_probability=self.gamma_probability,
            crossover_adaptation=self.crossover_adaptation,
            check_chains=self.check_chains,
            chains_number=self.chains_number,
            random_state=self.random_state,
        )
        new.__dict__.update(self.__dict__)

        return new
