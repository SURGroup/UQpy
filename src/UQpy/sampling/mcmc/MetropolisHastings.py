import logging

from beartype import beartype

from UQpy.sampling.mcmc.baseclass.MCMC import MCMC
from UQpy.distributions import *

from UQpy.sampling.input_data.MhInput import MhInput
from UQpy.utilities.ValidationTypes import *


class MetropolisHastings(MCMC):
    """
    Metropolis-Hastings algorithm

    **References**

    1. Gelman et al., "Bayesian data analysis", Chapman and Hall/CRC, 2013
    2. R.C. Smith, "Uncertainty Quantification - Theory, Implementation and Applications", CS&E, 2014


    **Algorithm-specific inputs:**

    * **proposal** (``Distribution`` object):
        Proposal distribution, must have a log_pdf/pdf and rvs method. Default: standard multivariate normal

    * **proposal_is_symmetric** (`bool`):
        Indicates whether the proposal distribution is symmetric, affects computation of acceptance probability alpha
        Default: False, set to True if default proposal is used

    **Methods:**

    """
    @beartype
    def __init__(self,
                 mh_input: MhInput,
                 samples_number: PositiveInteger = None,
                 samples_number_per_chain: PositiveInteger = None):

        super().__init__(pdf_target=mh_input.pdf_target, log_pdf_target=mh_input.log_pdf_target,
                         args_target=mh_input.args_target, dimension=mh_input.dimension,
                         seed=mh_input.seed, burn_length=mh_input.burn_length, jump=mh_input.jump,
                         save_log_pdf=mh_input.save_log_pdf, concatenate_chains=mh_input.concatenate_chains,
                         random_state=mh_input.random_state, chains_number=mh_input.chains_number)

        self.logger = logging.getLogger(__name__)
        # Initialize algorithm specific inputs
        self.proposal = mh_input.proposal
        self.proposal_is_symmetric = mh_input.proposal_is_symmetric
        if self.proposal is None:
            if self.dimension is None:
                raise ValueError('UQpy: Either input proposal or dimension must be provided.')
            from UQpy.distributions import JointIndependent, Normal
            self.proposal = JointIndependent([Normal()] * self.dimension)
            self.proposal_is_symmetric = True
        else:
            self._check_methods_proposal(self.proposal)

        self.logger.info('\nUQpy: Initialization of ' + self.__class__.__name__ + ' algorithm complete.')

        # If nsamples is provided, run the algorithm
        if (samples_number is not None) or (samples_number_per_chain is not None):
            self.run(samples_number=samples_number, samples_number_per_chain=samples_number_per_chain)

    def run_one_iteration(self, current_state, current_log_pdf):
        """
        Run one iteration of the mcmc chain for MH algorithm, starting at current state -
        see ``mcmc`` class.
        """
        # Sample candidate
        candidate = current_state + self.proposal.rvs(nsamples=self.chains_number, random_state=self.random_state)

        # Compute log_pdf_target of candidate sample
        log_p_candidate = self.evaluate_log_target(candidate)

        # Compute acceptance ratio
        if self.proposal_is_symmetric:  # proposal is symmetric
            log_ratios = log_p_candidate - current_log_pdf
        else:  # If the proposal is non-symmetric, one needs to account for it in computing acceptance ratio
            log_proposal_ratio = self.proposal.log_pdf(candidate - current_state) - \
                                 self.proposal.log_pdf(current_state - candidate)
            log_ratios = log_p_candidate - current_log_pdf - log_proposal_ratio

        # Compare candidate with current sample and decide or not to keep the candidate (loop over nc chains)
        accept_vec = np.zeros((self.chains_number,))  # this vector will be used to compute accept_ratio of each chain
        unif_rvs = Uniform().rvs(nsamples=self.chains_number, random_state=self.random_state).reshape((-1,))
        for nc, (cand, log_p_cand, r_) in enumerate(zip(candidate, log_p_candidate, log_ratios)):
            accept = np.log(unif_rvs[nc]) < r_
            if accept:
                current_state[nc, :] = cand
                current_log_pdf[nc] = log_p_cand
                accept_vec[nc] = 1.
        # Update the acceptance rate
        self._update_acceptance_rate(accept_vec)

        return current_state, current_log_pdf

    def __copy__(self):
        new = self.__class__(pdf_target=self.pdf_target,
                             log_pdf_target=self.log_pdf_target,
                             args_target=self.args_target,
                             burn_length=self.burn_length,
                             jump=self.jump,
                             dimension=self.dimension,
                             seed=self.seed,
                             save_log_pdf=self.save_log_pdf,
                             concatenate_chains=self.concatenate_chains,
                             chains_number=self.chains_number,
                             proposal=self.proposal,
                             proposal_is_symmetric=self.proposal_is_symmetric,
                             random_state=self.random_state)
        new.__dict__.update(self.__dict__)

        return new
