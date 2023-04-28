import logging
from typing import Callable
from beartype import beartype
from UQpy.sampling.mcmc.baseclass.MCMC import MCMC
from UQpy.distributions import *
from UQpy.utilities.ValidationTypes import *
import warnings

warnings.filterwarnings('ignore')


class MetropolisHastings(MCMC):

    @beartype
    def __init__(
            self,
            pdf_target: Union[Callable, list[Callable]] = None,
            log_pdf_target: Union[Callable, list[Callable]] = None,
            args_target: tuple = None,
            burn_length: Annotated[int, Is[lambda x: x >= 0]] = 0,
            jump: int = 1,
            dimension: int = None,
            seed: list = None,
            save_log_pdf: bool = False,
            concatenate_chains: bool = True,
            n_chains: int = None,
            proposal: Distribution = None,
            proposal_is_symmetric: bool = False,
            random_state: RandomStateType = None,
            nsamples: PositiveInteger = None,
            nsamples_per_chain: PositiveInteger = None,
    ):
        """
        Metropolis-Hastings algorithm :cite:`MCMC1` :cite:`MCMC2`

        :param pdf_target: Target density function from which to draw random samples. Either `pdf_target` or
         `log_pdf_target` must be provided (the latter should be preferred for better numerical stability).

         If `pdf_target` is a callable, it refers to the joint pdf to sample from, it must take at least one input
         **x**, which are the point(s) at which to evaluate the pdf. Within :class:`.MCMC` the pdf_target is evaluated
         as:
         :code:`p(x) = pdf_target(x, \*args_target)`

         where **x** is a :class:`numpy.ndarray  of shape :code:`(nsamples, dimension)` and `args_target` are additional
         positional arguments that are provided to :class:`.MCMC` via its `args_target` input.

         If `pdf_target` is a list of callables, it refers to independent marginals to sample from. The marginal in
         dimension :code:`j` is evaluated as:
         :code:`p_j(xj) = pdf_target[j](xj, \*args_target[j])` where **x** is a :class:`numpy.ndarray` of shape
         :code:`(nsamples, dimension)`
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

         If seed is not provided, both n_chains and dimension must be provided.
        :param save_log_pdf: Boolean that indicates whether to save log-pdf values along with the samples.
         Default: :any:`False`
        :param concatenate_chains: Boolean that indicates whether to concatenate the chains after a run, i.e., samples
         are stored as an :class:`numpy.ndarray` of shape :code:`(nsamples * n_chains, dimension)` if :any:`True`,
         :code:`(nsamples, n_chains, dimension)` if :any:`False`.
         Default: :any:`True`
        :param n_chains: The number of Markov chains to generate. Either dimension and `n_chains` or `seed` must be
         provided.
        :param proposal: Proposal distribution, must have a log_pdf/pdf and rvs method. Default: standard
         multivariate normal
        :param proposal_is_symmetric: Indicates whether the proposal distribution is symmetric, affects computation of
         acceptance probability alpha Default: :any:`False`, set to :any:`True` if default proposal is used
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
        # Initialize algorithm specific inputs
        self.proposal = proposal
        self.proposal_is_symmetric = proposal_is_symmetric
        if self.proposal is None:
            if self.dimension is None:
                raise ValueError("UQpy: Either input proposal or dimension must be provided.")
            from UQpy.distributions import JointIndependent, Normal

            self.proposal = JointIndependent([Normal()] * self.dimension)
            self.proposal_is_symmetric = True
        else:
            self._check_methods_proposal(self.proposal)

        self.logger.info("\nUQpy: Initialization of " + self.__class__.__name__ + " algorithm complete.")

        if (nsamples is not None) or (nsamples_per_chain is not None):
            self.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain, )

    def run_one_iteration(self, current_state: np.ndarray, current_log_pdf: np.ndarray):
        """
        Run one iteration of the mcmc chain for MH algorithm, starting at current state -
        see :class:`MCMC` class.
        """
        # Sample candidate
        candidate = current_state + self.proposal.rvs(
            nsamples=self.n_chains, random_state=self.random_state)

        # Compute log_pdf_target of candidate sample
        log_p_candidate = self.evaluate_log_target(candidate)

        # Compute acceptance ratio
        if self.proposal_is_symmetric:  # proposal is symmetric
            log_ratios = log_p_candidate - current_log_pdf
        else:  # If the proposal is non-symmetric, one needs to account for it in computing acceptance ratio
            log_proposal_ratio = self.proposal.log_pdf(
                candidate - current_state
            ) - self.proposal.log_pdf(current_state - candidate)
            log_ratios = log_p_candidate - current_log_pdf - log_proposal_ratio

        # Compare candidate with current sample and decide or not to keep the candidate (loop over nc chains)
        accept_vec = np.zeros(
            (self.n_chains,)
        )  # this vector will be used to compute accept_ratio of each chain
        unif_rvs = (
            Uniform()
                .rvs(nsamples=self.n_chains, random_state=self.random_state)
                .reshape((-1,))
        )
        for nc, (cand, log_p_cand, r_) in enumerate(
                zip(candidate, log_p_candidate, log_ratios)
        ):
            accept = np.log(unif_rvs[nc]) < r_
            if accept:
                current_state[nc, :] = cand
                current_log_pdf[nc] = log_p_cand
                accept_vec[nc] = 1.0
        # Update the acceptance rate
        self._update_acceptance_rate(accept_vec)

        return current_state, current_log_pdf
