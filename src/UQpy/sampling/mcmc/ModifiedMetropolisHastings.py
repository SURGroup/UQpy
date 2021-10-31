import logging

from beartype import beartype

from UQpy.sampling.mcmc.baseclass.MCMC import MCMC
from UQpy.distributions import *

from UQpy.sampling.input_data.MmhInput import MmhInput
from UQpy.utilities.ValidationTypes import *


class ModifiedMetropolisHastings(MCMC):
    @beartype
    def __init__(
        self,
        mmh_input: MmhInput,
        samples_number: PositiveInteger = None,
        samples_number_per_chain: PositiveInteger = None,
    ):
        """
        Component-wise Modified Metropolis-Hastings algorithm.

        In this algorithm, candidate samples are drawn separately in each dimension, thus the proposal consists of a list
        of 1d distributions. The target pdf can be given as a joint pdf or a list of marginal pdfs in all dimensions. This
        will trigger two different algorithms.

        **References:**

        1. S.-K. Au and J. L. Beck,“Estimation of small failure probabilities in high dimensions by subset simulation,”
           Probabilistic Eng. Mech., vol. 16, no. 4, pp. 263–277, Oct. 2001.

        :param mmh_input: Object that contains input data to the :class:`ΜοδιφιεδMetropolisHastings` class.
         (See :class:`.MμhInput`)
        :param samples_number: Number of samples to generate.
        :param samples_number_per_chain: Number of samples to generate per chain.
        """
        super().__init__(
            pdf_target=mmh_input.pdf_target,
            log_pdf_target=mmh_input.log_pdf_target,
            args_target=mmh_input.args_target,
            dimension=mmh_input.dimension,
            seed=mmh_input.seed,
            burn_length=mmh_input.burn_length,
            jump=mmh_input.jump,
            save_log_pdf=mmh_input.save_log_pdf,
            concatenate_chains=mmh_input.concatenate_chains,
            random_state=mmh_input.random_state,
            chains_number=mmh_input.chains_number,
        )

        self.logger = logging.getLogger(__name__)
        # If proposal is not provided: set it as a list of standard gaussians
        from UQpy.distributions import Normal

        self.proposal = mmh_input.proposal
        self.proposal_is_symmetric = mmh_input.proposal_is_symmetric

        # set default proposal
        if self.proposal is None:
            self.proposal = [Normal(),] * self.dimension
            self.proposal_is_symmetric = [True,] * self.dimension
        # Proposal is provided, check it
        else:
            # only one Distribution is provided, check it and transform it to a list
            if isinstance(self.proposal, JointIndependent):
                self.proposal = [m for m in self.proposal.marginals]
                if len(self.proposal) != self.dimension:
                    raise ValueError(
                        "UQpy: Proposal given as a list should be of length dimension"
                    )
                [self._check_methods_proposal(p) for p in self.proposal]
            elif not isinstance(self.proposal, list):
                self._check_methods_proposal(self.proposal)
                self.proposal = [self.proposal] * self.dimension
            else:  # a list of proposals is provided
                if len(self.proposal) != self.dimension:
                    raise ValueError(
                        "UQpy: Proposal given as a list should be of length dimension"
                    )
                [self._check_methods_proposal(p) for p in self.proposal]

        # check the symmetry of proposal, assign False as default
        if isinstance(self.proposal_is_symmetric, bool):
            self.proposal_is_symmetric = [self.proposal_is_symmetric,] * self.dimension
        elif not (
            isinstance(self.proposal_is_symmetric, list)
            and all(isinstance(b_, bool) for b_ in self.proposal_is_symmetric)
        ):
            raise TypeError(
                "UQpy: Proposal_is_symmetric should be a (list of) boolean(s)"
            )

        # check with algo type is used
        if self.evaluate_log_target_marginals is not None:
            self.target_type = "marginals"
            self.current_log_pdf_marginals = None
        else:
            self.target_type = "joint"

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
        Run one iteration of the mcmc chain for MMH algorithm, starting at current state -
        see :class:`MCMC` class.
        """
        # The target pdf is provided via its marginals
        accept_vec = np.zeros((self.chains_number,))
        if self.target_type == "marginals":
            # Evaluate the current log_pdf
            if self.current_log_pdf_marginals is None:
                self.current_log_pdf_marginals = [
                    self.evaluate_log_target_marginals[j](
                        current_state[:, j, np.newaxis]
                    )
                    for j in range(self.dimension)
                ]

            # Sample candidate (independently in each dimension)
            for j in range(self.dimension):
                candidate_j = current_state[:, j, np.newaxis] + self.proposal[j].rvs(
                    nsamples=self.chains_number, random_state=self.random_state
                )

                # Compute log_pdf_target of candidate sample
                log_p_candidate_j = self.evaluate_log_target_marginals[j](candidate_j)

                # Compute acceptance ratio
                if self.proposal_is_symmetric[j]:  # proposal is symmetric
                    log_ratios = log_p_candidate_j - self.current_log_pdf_marginals[j]
                else:  # If the proposal is non-symmetric, one needs to account for it in computing acceptance ratio
                    log_prop_j = self.proposal[j].log_pdf
                    log_proposal_ratio = log_prop_j(
                        candidate_j - current_state[:, j, np.newaxis]
                    ) - log_prop_j(current_state[:, j, np.newaxis] - candidate_j)
                    log_ratios = (
                        log_p_candidate_j
                        - self.current_log_pdf_marginals[j]
                        - log_proposal_ratio
                    )

                # Compare candidate with current sample and decide or not to keep the candidate
                unif_rvs = (
                    Uniform()
                    .rvs(nsamples=self.chains_number, random_state=self.random_state)
                    .reshape((-1,))
                )
                for nc, (cand, log_p_cand, r_) in enumerate(
                    zip(candidate_j, log_p_candidate_j, log_ratios)
                ):
                    accept = np.log(unif_rvs[nc]) < r_
                    if accept:
                        current_state[nc, j] = cand
                        self.current_log_pdf_marginals[j][nc] = log_p_cand
                        current_log_pdf = np.sum(self.current_log_pdf_marginals)
                        accept_vec[nc] += 1.0 / self.dimension

        # The target pdf is provided as a joint pdf
        else:
            candidate = np.copy(current_state)
            for j in range(self.dimension):
                candidate_j = current_state[:, j, np.newaxis] + self.proposal[j].rvs(
                    nsamples=self.chains_number, random_state=self.random_state
                )
                candidate[:, j] = candidate_j[:, 0]

                # Compute log_pdf_target of candidate sample
                log_p_candidate = self.evaluate_log_target(candidate)

                # Compare candidate with current sample and decide or not to keep the candidate
                if self.proposal_is_symmetric[j]:  # proposal is symmetric
                    log_ratios = log_p_candidate - current_log_pdf
                else:  # If the proposal is non-symmetric, one needs to account for it in computing acceptance ratio
                    log_prop_j = self.proposal[j].log_pdf
                    log_proposal_ratio = log_prop_j(
                        candidate_j - current_state[:, j, np.newaxis]
                    ) - log_prop_j(current_state[:, j, np.newaxis] - candidate_j)
                    log_ratios = log_p_candidate - current_log_pdf - log_proposal_ratio
                unif_rvs = (
                    Uniform()
                    .rvs(nsamples=self.chains_number, random_state=self.random_state)
                    .reshape((-1,))
                )
                for nc, (cand, log_p_cand, r_) in enumerate(
                    zip(candidate_j, log_p_candidate, log_ratios)
                ):
                    accept = np.log(unif_rvs[nc]) < r_
                    if accept:
                        current_state[nc, j] = cand
                        current_log_pdf[nc] = log_p_cand
                        accept_vec[nc] += 1.0 / self.dimension
                    else:
                        candidate[:, j] = current_state[:, j]
        # Update the acceptance rate
        self._update_acceptance_rate(accept_vec)
        return current_state, current_log_pdf

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
            proposal=self.proposal,
            proposal_is_symmetric=self.proposal_is_symmetric,
            chains_number=self.chains_number,
            random_state=self.random_state,
        )
        new.__dict__.update(self.__dict__)

        return new
