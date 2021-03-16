from UQpy.SampleMethods.MCMC.mcmc import MCMC
from UQpy.Distributions import *
import numpy as np

class MMH(MCMC):
    """

    Component-wise Modified Metropolis-Hastings algorithm.

    In this algorithm, candidate samples are drawn separately in each dimension, thus the proposal consists of a list
    of 1d distributions. The target pdf can be given as a joint pdf or a list of marginal pdfs in all dimensions. This
    will trigger two different algorithms.

    **References:**

    1. S.-K. Au and J. L. Beck,“Estimation of small failure probabilities in high dimensions by subset simulation,”
       Probabilistic Eng. Mech., vol. 16, no. 4, pp. 263–277, Oct. 2001.

    **Algorithm-specific inputs:**

    * **proposal** ((`list` of) ``Distribution`` object(s)):
        Proposal distribution(s) in one dimension, must have a log_pdf/pdf and rvs method.

        The proposal object may be a list of ``DistributionContinuous1D`` objects or a ``JointInd`` object.
        Default: standard normal

    * **proposal_is_symmetric** ((`list` of) `bool`):
        Indicates whether the proposal distribution is symmetric, affects computation of acceptance probability alpha
        Default: False, set to True if default proposal is used

    **Methods:**

    """
    def __init__(self, pdf_target=None, log_pdf_target=None, args_target=None, nburn=0, jump=1, dimension=None,
                 seed=None, save_log_pdf=False, concat_chains=True, nsamples=None, nsamples_per_chain=None,
                 proposal=None, proposal_is_symmetric=False, verbose=False, random_state=None, nchains=None):

        super().__init__(pdf_target=pdf_target, log_pdf_target=log_pdf_target, args_target=args_target,
                         dimension=dimension, seed=seed, nburn=nburn, jump=jump, save_log_pdf=save_log_pdf,
                         concat_chains=concat_chains, verbose=verbose, random_state=random_state, nchains=nchains)

        # If proposal is not provided: set it as a list of standard gaussians
        from UQpy.Distributions import Normal
        self.proposal = proposal
        self.proposal_is_symmetric = proposal_is_symmetric

        # set default proposal
        if self.proposal is None:
            self.proposal = [Normal(), ] * self.dimension
            self.proposal_is_symmetric = [True, ] * self.dimension
        # Proposal is provided, check it
        else:
            # only one Distribution is provided, check it and transform it to a list
            if isinstance(self.proposal, JointInd):
                self.proposal = [m for m in self.proposal.marginals]
                if len(self.proposal) != self.dimension:
                    raise ValueError('UQpy: Proposal given as a list should be of length dimension')
                [self._check_methods_proposal(p) for p in self.proposal]
            elif not isinstance(self.proposal, list):
                self._check_methods_proposal(self.proposal)
                self.proposal = [self.proposal] * self.dimension
            else:  # a list of proposals is provided
                if len(self.proposal) != self.dimension:
                    raise ValueError('UQpy: Proposal given as a list should be of length dimension')
                [self._check_methods_proposal(p) for p in self.proposal]

        # check the symmetry of proposal, assign False as default
        if isinstance(self.proposal_is_symmetric, bool):
            self.proposal_is_symmetric = [self.proposal_is_symmetric, ] * self.dimension
        elif not (isinstance(self.proposal_is_symmetric, list) and
                  all(isinstance(b_, bool) for b_ in self.proposal_is_symmetric)):
            raise TypeError('UQpy: Proposal_is_symmetric should be a (list of) boolean(s)')

        # check with algo type is used
        if self.evaluate_log_target_marginals is not None:
            self.target_type = 'marginals'
            self.current_log_pdf_marginals = None
        else:
            self.target_type = 'joint'

        if self.verbose:
            print('\nUQpy: Initialization of ' + self.__class__.__name__ + ' algorithm complete.')

        # If nsamples is provided, run the algorithm
        if (nsamples is not None) or (nsamples_per_chain is not None):
            self.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

    def run_one_iteration(self, current_state, current_log_pdf):
        """
        Run one iteration of the MCMC chain for MMH algorithm, starting at current state - see ``MCMC`` class.
        """
        # The target pdf is provided via its marginals
        accept_vec = np.zeros((self.nchains, ))
        if self.target_type == 'marginals':
            # Evaluate the current log_pdf
            if self.current_log_pdf_marginals is None:
                self.current_log_pdf_marginals = [self.evaluate_log_target_marginals[j](current_state[:, j, np.newaxis])
                                                  for j in range(self.dimension)]

            # Sample candidate (independently in each dimension)
            for j in range(self.dimension):
                candidate_j = current_state[:, j, np.newaxis] + self.proposal[j].rvs(
                    nsamples=self.nchains, random_state=self.random_state)

                # Compute log_pdf_target of candidate sample
                log_p_candidate_j = self.evaluate_log_target_marginals[j](candidate_j)

                # Compute acceptance ratio
                if self.proposal_is_symmetric[j]:  # proposal is symmetric
                    log_ratios = log_p_candidate_j - self.current_log_pdf_marginals[j]
                else:  # If the proposal is non-symmetric, one needs to account for it in computing acceptance ratio
                    log_prop_j = self.proposal[j].log_pdf
                    log_proposal_ratio = (log_prop_j(candidate_j - current_state[:, j, np.newaxis]) -
                                          log_prop_j(current_state[:, j, np.newaxis] - candidate_j))
                    log_ratios = log_p_candidate_j - self.current_log_pdf_marginals[j] - log_proposal_ratio

                # Compare candidate with current sample and decide or not to keep the candidate
                unif_rvs = Uniform().rvs(nsamples=self.nchains, random_state=self.random_state).reshape((-1,))
                for nc, (cand, log_p_cand, r_) in enumerate(
                        zip(candidate_j, log_p_candidate_j, log_ratios)):
                    accept = np.log(unif_rvs[nc]) < r_
                    if accept:
                        current_state[nc, j] = cand
                        self.current_log_pdf_marginals[j][nc] = log_p_cand
                        current_log_pdf = np.sum(self.current_log_pdf_marginals)
                        accept_vec[nc] += 1. / self.dimension

        # The target pdf is provided as a joint pdf
        else:
            candidate = np.copy(current_state)
            for j in range(self.dimension):
                candidate_j = current_state[:, j, np.newaxis] + self.proposal[j].rvs(
                    nsamples=self.nchains, random_state=self.random_state)
                candidate[:, j] = candidate_j[:, 0]

                # Compute log_pdf_target of candidate sample
                log_p_candidate = self.evaluate_log_target(candidate)

                # Compare candidate with current sample and decide or not to keep the candidate
                if self.proposal_is_symmetric[j]:  # proposal is symmetric
                    log_ratios = log_p_candidate - current_log_pdf
                else:  # If the proposal is non-symmetric, one needs to account for it in computing acceptance ratio
                    log_prop_j = self.proposal[j].log_pdf
                    log_proposal_ratio = (log_prop_j(candidate_j - current_state[:, j, np.newaxis]) -
                                          log_prop_j(current_state[:, j, np.newaxis] - candidate_j))
                    log_ratios = log_p_candidate - current_log_pdf - log_proposal_ratio
                unif_rvs = Uniform().rvs(nsamples=self.nchains, random_state=self.random_state).reshape((-1,))
                for nc, (cand, log_p_cand, r_) in enumerate(zip(candidate_j, log_p_candidate, log_ratios)):
                    accept = np.log(unif_rvs[nc]) < r_
                    if accept:
                        current_state[nc, j] = cand
                        current_log_pdf[nc] = log_p_cand
                        accept_vec[nc] += 1. / self.dimension
                    else:
                        candidate[:, j] = current_state[:, j]
        # Update the acceptance rate
        self._update_acceptance_rate(accept_vec)
        return current_state, current_log_pdf
