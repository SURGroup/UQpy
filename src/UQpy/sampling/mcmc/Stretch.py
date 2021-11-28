import logging
from beartype import beartype
from UQpy.sampling.mcmc.baseclass.MCMC import MCMC
from UQpy.distributions import *
from UQpy.sampling.input_data.StretchInput import StretchInput
from UQpy.utilities.ValidationTypes import *
from UQpy.sampling.input_data.SamplingInput import SamplingInput


class Stretch(MCMC):

    @beartype
    def __init__(
        self,
        stretch_input: StretchInput,
        samples_number: PositiveInteger = None,
        samples_number_per_chain: PositiveInteger = None,
    ):
        """
        Affine-invariant sampler with Stretch moves, parallel implementation. :cite:`Stretch1` :cite:`Stretch2`

        :param stretch_input: Object that contains input data to the :class:`.Stretch` class.
         (See :class:`.StretchInput`)
        :param samples_number: Number of samples to generate.
        :param samples_number_per_chain: Number of samples to generate per chain.
        """
        flag_seed = False
        if stretch_input.seed is None:
            if stretch_input.dimension is None or stretch_input.chains_number is None:
                raise ValueError(
                    "UQpy: Either `seed` or `dimension` and `nchains` must be provided."
                )
            flag_seed = True

        super().__init__(
            pdf_target=stretch_input.pdf_target,
            log_pdf_target=stretch_input.log_pdf_target,
            args_target=stretch_input.args_target,
            dimension=stretch_input.dimension,
            seed=stretch_input.seed,
            burn_length=stretch_input.burn_length,
            jump=stretch_input.jump,
            save_log_pdf=stretch_input.save_log_pdf,
            concatenate_chains=stretch_input.concatenate_chains,
            random_state=stretch_input.random_state,
            chains_number=stretch_input.chains_number,
        )

        self.logger = logging.getLogger(__name__)
        # Check nchains = ensemble size for the Stretch algorithm
        if flag_seed:
            self.seed = (
                Uniform()
                .rvs(
                    nsamples=self.dimension * self.chains_number,
                    random_state=self.random_state,
                )
                .reshape((self.chains_number, self.dimension))
            )
        if self.chains_number < 2:
            raise ValueError(
                "UQpy: For the Stretch algorithm, a seed must be provided with at least two samples."
            )

        # Check Stretch algorithm inputs: proposal_type and proposal_scale
        self.scale = stretch_input.scale
        if not isinstance(self.scale, float):
            raise TypeError("UQpy: Input scale must be of type float.")

        self.logger.info(
            "\nUQpy: Initialization of "
            + self.__class__.__name__
            + " algorithm complete."
        )

        # If nsamples is provided, run the algorithm
        if (samples_number is not None) or (samples_number_per_chain is not None):
            self.run(
                samples_number=samples_number,
                samples_number_per_chain=samples_number_per_chain,
            )

    def run_one_iteration(self, current_state, current_log_pdf):
        """
        Run one iteration of the mcmc chain for Stretch algorithm, starting at current state -
        see :class:`.MCMC` class.
        """
        # Start the loop over nsamples - this code uses the parallel version of the stretch algorithm
        all_inds = np.arange(self.chains_number)
        inds = all_inds % 2
        accept_vec = np.zeros((self.chains_number,))
        # Separate the full ensemble into two sets, use one as a complementary ensemble to the other and vice-versa
        for split in range(2):
            set1 = inds == split

            # Get current and complementary sets
            sets = [current_state[inds == j01, :] for j01 in range(2)]
            curr_set, comp_set = (
                sets[split],
                sets[1 - split],
            )  # current and complementary sets respectively
            ns, nc = len(curr_set), len(comp_set)

            # Sample new state for S1 based on S0
            unif_rvs = Uniform().rvs(nsamples=ns, random_state=self.random_state)
            zz = ((self.scale - 1.0) * unif_rvs + 1.0) ** 2.0 / self.scale  # sample Z
            factors = (self.dimension - 1.0) * np.log(zz)  # compute log(Z ** (d - 1))
            multi_rvs = Multinomial(n=1, p=[1.0 / nc,] * nc).rvs(
                nsamples=ns, random_state=self.random_state
            )
            rint = np.nonzero(multi_rvs)[1]  # sample X_{j} from complementary set
            candidates = comp_set[rint, :] - (comp_set[rint, :] - curr_set) * np.tile(
                zz, [1, self.dimension]
            )  # new candidates

            # Compute new likelihood, can be done in parallel :)
            logp_candidates = self.evaluate_log_target(candidates)

            # Compute acceptance rate
            unif_rvs = (
                Uniform()
                .rvs(nsamples=len(all_inds[set1]), random_state=self.random_state)
                .reshape((-1,))
            )
            for j, f, lpc, candidate, u_rv in zip(
                all_inds[set1], factors, logp_candidates, candidates, unif_rvs
            ):
                accept = np.log(u_rv) < f + lpc - current_log_pdf[j]
                if accept:
                    current_state[j] = candidate
                    current_log_pdf[j] = lpc
                    accept_vec[j] += 1.0

        # Update the acceptance rate
        self._update_acceptance_rate(accept_vec)
        return current_state, current_log_pdf


SamplingInput.input_to_class[StretchInput] = StretchInput
