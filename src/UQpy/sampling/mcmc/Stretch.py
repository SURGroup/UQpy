import logging
from typing import Callable
import warnings

warnings.filterwarnings('ignore')

from beartype import beartype
from UQpy.sampling.mcmc.baseclass.MCMC import MCMC
from UQpy.distributions import *
from UQpy.utilities.ValidationTypes import *


class Stretch(MCMC):

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
            scale: float = 2.0,
            random_state: RandomStateType = None,
            n_chains: int = None,
            nsamples: PositiveInteger = None,
            nsamples_per_chain: PositiveInteger = None,
    ):
        """
        Affine-invariant sampler with Stretch moves, parallel implementation. :cite:`Stretch1` :cite:`Stretch2`

        :param pdf_target: Target density function from which to draw random samples. Either `pdf_target` or
         `log_pdf_target` must be provided (the latter should be preferred for better numerical stability).

         If `pdf_target` is a callable, it refers to the joint pdf to sample from, it must take at least one input
         **x**, which are the point(s) at which to evaluate the pdf. Within :class:`.MCMC` the `pdf_target` is evaluated
         as: :code:`p(x) = pdf_target(x, \*args_target)`

         where **x** is a :class:`numpy.ndarray` of shape :code:`(nsamples, dimension)` and `args_target` are additional
         positional arguments that are provided to :class:`.MCMC` via its `args_target` input.

         If `pdf_target` is a list of callables, it refers to independent marginals to sample from. The marginal in
         dimension :code:`j` is evaluated as: :code:`p_j(xj) = pdf_target[j](xj, \*args_target[j])` where **x** is a
         :class:`numpy.ndarray` of shape :code:`(nsamples, dimension)`
        :param log_pdf_target: Logarithm of the target density function from which to draw random samples.
         Either :code:`pdf_target` or :code:`log_pdf_target` must be provided (the latter should be preferred for better
         numerical stability).

         Same comments as for input `pdf_target`.
        :param args_target: Positional arguments of the pdf / log-pdf target function. See `pdf_target`
        :param burn_length: Length of burn-in - i.e., number of samples at the beginning of the chain to discard (note:
         no thinning during burn-in). Default is :math:`0`, no burn-in.
        :param jump: Thinning parameter, used to reduce correlation between samples. Setting :code:`jump=n` corresponds
         to skipping `n-1` states between accepted states of the chain. Default is :math:`1` (no thinning).
        :param dimension: A scalar value defining the dimension of target density function. Either `dimension` and
         `n_chains` or `seed` must be provided.
        :param seed: Seed of the Markov chain(s), shape :code:`(n_chains, dimension)`.
         Default: :code:`zeros(n_chains x dimension)`.

         If `seed` is not provided, both `n_chains` and `dimension` must be provided.
        :param save_log_pdf: Boolean that indicates whether to save log-pdf values along with the samples.
         Default: :any:`False`
        :param concatenate_chains: Boolean that indicates whether to concatenate the chains after a run, i.e., samples
         are stored as an :class:`numpy.ndarray` of shape :code:`(nsamples * n_chains, dimension)` if :any:`True`,
         :code:`(nsamples, n_chains, dimension)` if :any:`False`.
         Default: :any:`True`
        :param n_chains: The number of Markov chains to generate. Either `dimension` and `n_chains` or `seed` must be
         provided.
        :param scale: Scale parameter. Default: :math:`2`.
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is
         :any:`None`.
        :param nsamples: Number of samples to generate.
        :param nsamples_per_chain: Number of samples to generate per chain.
        """
        flag_seed = False
        if seed is None:
            if dimension is None or n_chains is None:
                raise ValueError("UQpy: Either `seed` or `dimension` and `n_chains` must be provided.")
            flag_seed = True

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
            n_chains=n_chains, )

        self.logger = logging.getLogger(__name__)
        # Check nchains = ensemble size for the Stretch algorithm
        if flag_seed:
            self.seed = (Uniform().rvs(nsamples=self.dimension * self.n_chains,
                                       random_state=self.random_state, )
                         .reshape((self.n_chains, self.dimension)))
        if self.n_chains < 2:
            raise ValueError("UQpy: For the Stretch algorithm, a seed must be provided with at least two samples.")

        # Check Stretch algorithm inputs: proposal_type and proposal_scale
        self.scale = scale
        if not isinstance(self.scale, float):
            raise TypeError("UQpy: Input scale must be of type float.")

        self.logger.info("\nUQpy: Initialization of " + self.__class__.__name__ + " algorithm complete.")

        if (nsamples is not None) or (nsamples_per_chain is not None):
            self.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain, )

    def run_one_iteration(self, current_state, current_log_pdf):
        """
        Run one iteration of the mcmc chain for Stretch algorithm, starting at current state -
        see :class:`.MCMC` class.
        """
        # Start the loop over nsamples - this code uses the parallel version of the stretch algorithm
        all_inds = np.arange(self.n_chains)
        inds = all_inds % 2
        accept_vec = np.zeros((self.n_chains,))
        # Separate the full ensemble into two sets, use one as a complementary ensemble to the other and vice-versa
        for split in range(2):
            set1 = inds == split

            # Get current and complementary sets
            sets = [current_state[inds == j01, :] for j01 in range(2)]
            curr_set, comp_set = (sets[split], sets[1 - split],)  # current and complementary sets respectively
            ns, nc = len(curr_set), len(comp_set)

            # Sample new state for S1 based on S0
            unif_rvs = Uniform().rvs(nsamples=ns, random_state=self.random_state)
            zz = ((self.scale - 1.0) * unif_rvs + 1.0) ** 2.0 / self.scale  # sample Z
            factors = (self.dimension - 1.0) * np.log(zz)  # compute log(Z ** (d - 1))
            multi_rvs = Multinomial(n=1, p=[1.0 / nc, ] * nc).rvs(
                nsamples=ns, random_state=self.random_state)
            rint = np.nonzero(multi_rvs)[1]  # sample X_{j} from complementary set
            candidates = comp_set[rint, :] - (comp_set[rint, :] - curr_set) * np.tile(
                zz, [1, self.dimension])  # new candidates

            # Compute new likelihood, can be done in parallel :)
            logp_candidates = self.evaluate_log_target(candidates)

            # Compute acceptance rate
            unif_rvs = (Uniform().rvs(nsamples=len(all_inds[set1]), random_state=self.random_state).reshape((-1,)))
            for j, f, lpc, candidate, u_rv in zip(all_inds[set1], factors, logp_candidates, candidates, unif_rvs):
                accept = np.log(u_rv) < f + lpc - current_log_pdf[j]
                if accept:
                    current_state[j] = candidate
                    current_log_pdf[j] = lpc
                    accept_vec[j] += 1.0

        # Update the acceptance rate
        self._update_acceptance_rate(accept_vec)
        return current_state, current_log_pdf
