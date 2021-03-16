from UQpy.SampleMethods.MCMC.mcmc import MCMC
from UQpy.Distributions import *
import numpy as np

class Stretch(MCMC):
    """
    Affine-invariant sampler with Stretch moves, parallel implementation.

    **References:**

    1. J. Goodman and J. Weare, “Ensemble samplers with affine invariance,” Commun. Appl. Math. Comput. Sci.,vol.5,
       no. 1, pp. 65–80, 2010.
    2. Daniel Foreman-Mackey, David W. Hogg, Dustin Lang, and Jonathan Goodman. "emcee: The MCMC Hammer".
       Publications of the Astronomical Society of the Pacific, 125(925):306–312,2013.

    **Algorithm-specific inputs:**

    * **scale** (`float`):
        Scale parameter. Default: 2.

    **Methods:**

    """
    def __init__(self, pdf_target=None, log_pdf_target=None, args_target=None, nburn=0, jump=1, dimension=None,
                 seed=None, save_log_pdf=False, concat_chains=True, nsamples=None, nsamples_per_chain=None,
                 scale=2., verbose=False, random_state=None, nchains=None):

        flag_seed = False
        if seed is None:
            if dimension is None or nchains is None:
                raise ValueError('UQpy: Either `seed` or `dimension` and `nchains` must be provided.')
            flag_seed = True

        super().__init__(pdf_target=pdf_target, log_pdf_target=log_pdf_target, args_target=args_target,
                         dimension=dimension, seed=seed, nburn=nburn, jump=jump, save_log_pdf=save_log_pdf,
                         concat_chains=concat_chains, verbose=verbose, random_state=random_state, nchains=nchains)

        # Check nchains = ensemble size for the Stretch algorithm
        if flag_seed:
            self.seed = Uniform().rvs(nsamples=self.dimension * self.nchains, random_state=self.random_state).reshape(
                (self.nchains, self.dimension)
            )
        if self.nchains < 2:
            raise ValueError('UQpy: For the Stretch algorithm, a seed must be provided with at least two samples.')

        # Check Stretch algorithm inputs: proposal_type and proposal_scale
        self.scale = scale
        if not isinstance(self.scale, float):
            raise TypeError('UQpy: Input scale must be of type float.')

        if self.verbose:
            print('\nUQpy: Initialization of ' + self.__class__.__name__ + ' algorithm complete.')

        # If nsamples is provided, run the algorithm
        if (nsamples is not None) or (nsamples_per_chain is not None):
            self.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

    def run_one_iteration(self, current_state, current_log_pdf):
        """
        Run one iteration of the MCMC chain for Stretch algorithm, starting at current state - see ``MCMC`` class.
        """
        # Start the loop over nsamples - this code uses the parallel version of the stretch algorithm
        all_inds = np.arange(self.nchains)
        inds = all_inds % 2
        accept_vec = np.zeros((self.nchains, ))
        # Separate the full ensemble into two sets, use one as a complementary ensemble to the other and vice-versa
        for split in range(2):
            set1 = (inds == split)

            # Get current and complementary sets
            sets = [current_state[inds == j01, :] for j01 in range(2)]
            curr_set, comp_set = sets[split], sets[1 - split]  # current and complementary sets respectively
            ns, nc = len(curr_set), len(comp_set)

            # Sample new state for S1 based on S0
            unif_rvs = Uniform().rvs(nsamples=ns, random_state=self.random_state)
            zz = ((self.scale - 1.) * unif_rvs + 1.) ** 2. / self.scale  # sample Z
            factors = (self.dimension - 1.) * np.log(zz)  # compute log(Z ** (d - 1))
            multi_rvs = Multinomial(n=1, p=[1. / nc, ] * nc).rvs(nsamples=ns, random_state=self.random_state)
            rint = np.nonzero(multi_rvs)[1]    # sample X_{j} from complementary set
            candidates = comp_set[rint, :] - (comp_set[rint, :] - curr_set) * np.tile(
                zz, [1, self.dimension])  # new candidates

            # Compute new likelihood, can be done in parallel :)
            logp_candidates = self.evaluate_log_target(candidates)

            # Compute acceptance rate
            unif_rvs = Uniform().rvs(nsamples=len(all_inds[set1]), random_state=self.random_state).reshape((-1,))
            for j, f, lpc, candidate, u_rv in zip(
                    all_inds[set1], factors, logp_candidates, candidates, unif_rvs):
                accept = np.log(u_rv) < f + lpc - current_log_pdf[j]
                if accept:
                    current_state[j] = candidate
                    current_log_pdf[j] = lpc
                    accept_vec[j] += 1.

        # Update the acceptance rate
        self._update_acceptance_rate(accept_vec)
        return current_state, current_log_pdf
