import copy

import scipy.stats as stats

from UQpy.sampling.mcmc.MetropolisHastings import MetropolisHastings
from UQpy.distributions.collection.MultivariateNormal import MultivariateNormal
from UQpy.sampling.mcmc.baseclass.MCMC import *
from UQpy.sampling.tempering_mcmc.TemperingMCMC import TemperingMCMC


class SequentialTemperingMCMC(TemperingMCMC):
    """
    Sequential-Tempering MCMC

    This algorithms samples from a series of intermediate targets that are each tempered versions of the final/true
    target. In going from one intermediate distribution to the next, the existing samples are resampled according to
    some weights (similar to importance sampling). To ensure that there aren't a large number of duplicates, the
    resampling step is followed by a short (or even single-step) MCMC run that disperses the samples while remaining
    within the correct intermediate distribution. The final intermediate target is the required target distribution.

    **References**

    1. Ching and Chen, "Transitional Markov Chain Monte Carlo Method for Bayesian Model Updating,
       Model Class Selection, and Model Averaging", Journal of Engineering Mechanics/ASCE, 2007

    **Inputs:**

    Many inputs are similar to MCMC algorithms. Additional inputs are:

    * **mcmc_class**
    * **recalc_w**
    * **nburn_resample**
    * **nburn_mcmc**

    **Methods:**
    """

    @beartype
    def __init__(self, pdf_intermediate=None, log_pdf_intermediate=None, args_pdf_intermediate=(),
                 distribution_reference=None,
                 mcmc_class: MCMC = None,
                 dimension=None, seed=None,
                 nsamples: PositiveInteger = None,
                 recalc_w=False,
                 nburn_resample=0, save_intermediate_samples=False, nchains=1,
                 percentage_resampling=100, random_state=None,
                 proposal_is_symmetric=True):

        self.logger = logging.getLogger(__name__)

        super().__init__(pdf_intermediate=pdf_intermediate, log_pdf_intermediate=log_pdf_intermediate,
                         args_pdf_intermediate=args_pdf_intermediate, distribution_reference=distribution_reference,
                         dimension=dimension, random_state=random_state)

        # Initialize inputs
        self.save_intermediate_samples = save_intermediate_samples
        self.recalc_w = recalc_w
        self.nburn_resample = nburn_resample
        self.nchains = nchains
        self.resample_frac = percentage_resampling / 100
        self.proposal_is_symmetric=proposal_is_symmetric

        self.nspc = int(np.floor(((1 - self.resample_frac) * nsamples) / self.nchains))
        self.nresample = int(nsamples - (self.nspc * self.nchains))
        self.mcmc_class:MCMC = mcmc_class

        # Initialize input distributions
        self.evaluate_log_reference, self.seed = self._preprocess_reference(dist_=distribution_reference,
                                                                            seed_=seed, nsamples=nsamples,
                                                                            dimension=self.dimension)

        # Initialize flag that indicates whether default proposal is to be used (default proposal defined adaptively
        # during run)
        if self.proposal is None:
            self.proposal_given_flag = False
        else:
            self.proposal_given_flag = True

        # Initialize attributes
        self.temper_param_list = None
        self.evidence = None
        self.evidence_cov = None

        # Call the run function
        if nsamples is not None:
            self._run(nsamples=nsamples)
        else:
            raise ValueError('UQpy: a value for "nsamples" must be specified ')

    @beartype
    def _run(self, nsamples: PositiveInteger = None):

        self.logger.info('TMCMC Start')

        if self.samples is not None:
            raise RuntimeError('UQpy: run method cannot be called multiple times for the same object')

        pts = self.seed  # Generated Samples from prior for zero-th tempering level

        # Initializing other variables
        temper_param = 0.0  # Intermediate exponent
        temper_param_prev = temper_param
        self.temper_param_list = np.array(temper_param)
        pts_index = np.arange(nsamples)  # Array storing sample indices
        w = np.zeros(nsamples)  # Array storing plausibility weights
        wp = np.zeros(nsamples)  # Array storing plausibility weight probabilities
        exp_q0 = 0
        for i in range(nsamples):
            exp_q0 += np.exp(self.evaluate_log_intermediate(pts[i, :].reshape((1, -1)), 0.0))
        S = exp_q0 / nsamples

        if self.save_intermediate_samples is True:
            self.intermediate_samples = []
            self.intermediate_samples += [pts.copy()]

        # Looping over all adaptively decided tempering levels
        while temper_param < 1:

            # Adaptively set the tempering exponent for the current level
            temper_param_prev = temper_param
            temper_param = self._find_temper_param(temper_param_prev, pts, self.evaluate_log_intermediate, nsamples)
            # d_exp = temper_param - temper_param_prev
            self.temper_param_list = np.append(self.temper_param_list, temper_param)

            self.logger.info('beta selected')

            # Calculate the plausibility weights
            for i in range(nsamples):
                w[i] = np.exp(self.evaluate_log_intermediate(pts[i, :].reshape((1, -1)), temper_param)
                              - self.evaluate_log_intermediate(pts[i, :].reshape((1, -1)), temper_param_prev))

            # Calculate normalizing constant for the plausibility weights (sum of the weights)
            w_sum = np.sum(w)
            # Calculate evidence from each tempering level
            S = S * (w_sum / nsamples)
            # Normalize plausibility weight probabilities
            wp = (w / w_sum)

            # Calculate covariance matrix for the default proposal
            cov_scale = 0.2
            w_th_sum = np.zeros(self.dimension)
            for i in range(nsamples):
                for j in range(self.dimension):
                    w_th_sum[j] += w[i] * pts[i, j]
            sig_mat = np.zeros((self.dimension, self.dimension))
            for i in range(nsamples):
                pts_deviation = np.zeros((self.dimension, 1))
                for j in range(self.dimension):
                    pts_deviation[j, 0] = pts[i, j] - (w_th_sum[j] / w_sum)
                sig_mat += (w[i] / w_sum) * np.dot(pts_deviation,
                                                   pts_deviation.T)  # Normalized by w_sum as per Betz et al
            sig_mat = cov_scale * cov_scale * sig_mat

            mcmc_log_pdf_target = self._target_generator(self.evaluate_log_intermediate,
                                                         self.evaluate_log_reference, temper_param)

            self.logger.info('Begin Resampling')
            # Resampling and MH-MCMC step
            for i in range(self.nresample):

                # Resampling from previous tempering level
                lead_index = int(np.random.choice(pts_index, p=wp))
                lead = pts[lead_index]

                # Defining the default proposal
                if self.proposal_given_flag is False:
                    self.proposal = MultivariateNormal(lead, cov=sig_mat)

                # Single MH-MCMC step
                x = MetropolisHastings(dimension=self.dimension, log_pdf_target=mcmc_log_pdf_target, seed=lead,
                                       nsamples=1, nchains=1, nburn=self.nburn_resample, proposal=self.proposal,
                                       proposal_is_symmetric=self.proposal_is_symmetric)

                # Setting the generated sample in the array
                pts[i] = x.samples

                if self.recalc_w:
                    w[i] = np.exp(self.evaluate_log_intermediate(pts[i, :].reshape((1, -1)), temper_param)
                                  - self.evaluate_log_intermediate(pts[i, :].reshape((1, -1)), temper_param_prev))
                    wp[i] = w[i] / w_sum

            self.logger.info('Begin MCMC')
            mcmc_seed = self._mcmc_seed_generator(resampled_pts=pts[0:self.nresample, :], arr_length=self.nresample,
                                                  seed_length=self.nchains)

            y=copy.deepcopy(self.mcmc_class)
            self.update_target_and_seed(y, mcmc_seed, mcmc_log_pdf_target)
            # y = self.mcmc_class(log_pdf_target=mcmc_log_pdf_target, seed=mcmc_seed, dimension=self.dimension,
            #                     nchains=self.nchains, nsamples_per_chain=self.nspc, nburn=self.nburn_mcmc,
            #                     jump=self.jump_mcmc, concat_chains=True)
            pts[self.nresample:, :] = y.samples

            if self.save_intermediate_samples is True:
                self.intermediate_samples += [pts.copy()]

            self.logger.info('Tempering level ended')

        # Setting the calculated values to the attributes
        self.samples = pts
        self.evidence = S

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
            for i2 in range(0, n):
                q_scaled[i2] = np.exp(q_func(samples[i2, :].reshape((1, -1)), 1)
                                      - q_func(samples[i2, :].reshape((1, -1)), temper_param_prev))
            sigma_1 = np.std(q_scaled)
            mu_1 = np.mean(q_scaled)
            if sigma_1 < mu_1:
                flag = 1
                temper_param_trial = 1
                continue
            for i3 in range(0, n):
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

    def _preprocess_reference(self, dist_, seed_=None, nsamples=None, dimension=None):
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

        if dist_ is not None and seed_ is not None:
            raise ValueError('UQpy: both prior and seed values cannot be provided')
        elif dist_ is not None:
            if not (isinstance(dist_, Distribution)):
                raise TypeError('UQpy: A UQpy.Distribution object must be provided.')
            else:
                evaluate_log_pdf = (lambda x: dist_.log_pdf(x))
                seed_values = dist_.rvs(nsamples=nsamples)
        elif seed_ is not None:
            if seed_.shape[0] == nsamples and seed_.shape[1] == dimension:
                seed_values = seed_
                kernel = stats.gaussian_kde(seed_)
                evaluate_log_pdf = (lambda x: kernel.logpdf(x))
            else:
                raise TypeError('UQpy: the seed values should be a numpy array of size (nsamples, dimension)')
        else:
            raise ValueError('UQpy: either prior distribution or seed values must be provided')
        return evaluate_log_pdf, seed_values

    @staticmethod
    def _mcmc_seed_generator(resampled_pts, arr_length, seed_length):
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
        seed_indices = np.random.choice(index_arr, size=seed_length, replace=False)
        mcmc_seed = resampled_pts[seed_indices, :]
        return mcmc_seed
