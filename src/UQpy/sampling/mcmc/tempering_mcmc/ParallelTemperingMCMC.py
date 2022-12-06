from scipy.special import logsumexp
from scipy.integrate import trapz

from UQpy.sampling.mcmc import MetropolisHastings
from UQpy.sampling.mcmc.baseclass.MCMC import *
from UQpy.sampling.mcmc.tempering_mcmc.baseclass.TemperingMCMC import TemperingMCMC


class ParallelTemperingMCMC(TemperingMCMC):
    """
    Parallel-Tempering MCMC

    This algorithm runs the chains sampling from various tempered distributions in parallel. Periodically during the
    run, the different temperatures swap members of their ensemble in a way that
    preserves detailed balance.The chains closer to the reference chain (hot chains) can sample from regions that have
    low probability under the target and thus allow a better exploration of the parameter space, while the cold chains
    can better explore the regions of high likelihood.

    **References**

    1. Parallel Tempering: Theory, Applications, and New Perspectives, Earl and Deem
    2. Adaptive Parallel Tempering MCMC
    3. emcee the MCMC Hammer python package

    **Inputs:**

    Many inputs are similar to MCMC algorithms. Additional inputs are:

    * **niter_between_sweeps**

    * **mcmc_class**

    **Methods:**

    """

    @beartype
    def __init__(self, n_iterations_between_sweeps: PositiveInteger,
                 pdf_intermediate=None, log_pdf_intermediate=None, args_pdf_intermediate=(),
                 distribution_reference: Distribution = None,
                 save_log_pdf: bool = False, nsamples: PositiveInteger = None,
                 nsamples_per_chain: PositiveInteger = None,
                 random_state: RandomStateType = None,
                 tempering_parameters: list = None,
                 n_tempering_parameters: int = None,
                 samplers: Union[MCMC, list[MCMC]] = None):

        super().__init__(pdf_intermediate=pdf_intermediate, log_pdf_intermediate=log_pdf_intermediate,
                         args_pdf_intermediate=args_pdf_intermediate, distribution_reference=None,
                         save_log_pdf=save_log_pdf, random_state=random_state)
        self.logger = logging.getLogger(__name__)
        if not isinstance(samplers, list):
            self.samplers = [samplers.__copy__() for _ in range(len(tempering_parameters))]
        else:
            self.samplers = samplers

        self.distribution_reference = distribution_reference
        self.evaluate_log_reference = self._preprocess_reference(self.distribution_reference)

        # Initialize PT specific inputs: niter_between_sweeps and temperatures
        self.n_iterations_between_sweeps = n_iterations_between_sweeps
        self.tempering_parameters = tempering_parameters
        self.n_tempering_parameters = n_tempering_parameters
        if self.tempering_parameters is None:
            if self.n_tempering_parameters is None:
                raise ValueError('UQpy: either input temper_param_list or n_temper_params should be provided.')
            elif not (isinstance(self.n_tempering_parameters, int) and self.n_tempering_parameters >= 2):
                raise ValueError('UQpy: input n_temper_params should be a integer >= 2.')
            else:
                self.tempering_parameters = [1. / np.sqrt(2) ** i for i in
                                             range(self.n_tempering_parameters - 1, -1, -1)]
        elif (not isinstance(self.tempering_parameters, (list, tuple))
              or not (all(isinstance(t, (int, float)) and (0 < t <= 1.) for t in self.tempering_parameters))
                # or float(self.temperatures[0]) != 1.
        ):
            raise ValueError(
                'UQpy: temper_param_list should be a list of floats in [0, 1], starting at 0. and increasing to 1.')
        else:
            self.n_tempering_parameters = len(self.tempering_parameters)

        # default value
        for i, sampler in enumerate(self.samplers):
            if isinstance(sampler, MetropolisHastings) and sampler.proposal is None:
                from UQpy.distributions import JointIndependent, Normal
                self.samplers[i] = sampler.__copy__(proposal_is_symmetric=True,
                                                    proposal=JointIndependent(
                                                        [Normal(scale=1. / np.sqrt(self.tempering_parameters[i]))] *
                                                        sampler.dimension))

        # Initialize algorithm specific inputs: target pdfs
        self.thermodynamic_integration_results = None

        self.mcmc_samplers = []
        for i, temper_param in enumerate(self.tempering_parameters):
            log_pdf_target = (lambda x, temper_param=temper_param: self.evaluate_log_reference(
                x) + self.evaluate_log_intermediate(x, temper_param))
            self.mcmc_samplers.append(self.samplers[i].__copy__(log_pdf_target=log_pdf_target, concatenate_chains=True,
                                                                save_log_pdf=save_log_pdf,
                                                                random_state=self.random_state))

        self.logger.info('\nUQpy: Initialization of ' + self.__class__.__name__ + ' algorithm complete.')

        # If nsamples is provided, run the algorithm
        if (nsamples is not None) or (nsamples_per_chain is not None):
            self.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

    @beartype
    def run(self, nsamples: PositiveInteger = None, nsamples_per_chain: PositiveInteger = None):
        """
        Run the MCMC algorithm.

        This function samples from the MCMC chains and appends samples to existing ones (if any). This method leverages
        the ``run_iterations`` method that is specific to each algorithm.

        **Inputs:**

        * **nsamples** (`int`):
            Number of samples to generate.

        * **nsamples_per_chain** (`int`)
            Number of samples to generate per chain.

        Either `nsamples` or `nsamples_per_chain` must be provided (not both). Not that if `nsamples` is not a multiple
        of `nchains`, `nsamples` is set to the next largest integer that is a multiple of `nchains`.

        """
        current_state, current_log_pdf = [], []
        final_ns_per_chain = 0
        for i, mcmc_sampler in enumerate(self.mcmc_samplers):
            if mcmc_sampler.evaluate_log_target is None and mcmc_sampler.evaluate_log_target_marginals is None:
                (mcmc_sampler.evaluate_log_target, mcmc_sampler.evaluate_log_target_marginals,) = \
                    mcmc_sampler._preprocess_target(pdf_=mcmc_sampler.pdf_target,
                                                    log_pdf_=mcmc_sampler.log_pdf_target,
                                                    args=mcmc_sampler.args_target)
            ns, ns_per_chain, current_state_t, current_log_pdf_t = mcmc_sampler._initialize_samples(
                nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)
            current_state.append(current_state_t.copy())
            current_log_pdf.append(current_log_pdf_t.copy())
            if i == 0:
                final_ns_per_chain = ns_per_chain

        self.logger.info('UQpy: Running MCMC...')

        # Run nsims iterations of the MCMC algorithm, starting at current_state
        while self.mcmc_samplers[0].nsamples_per_chain < final_ns_per_chain:
            # update the total number of iterations

            # run one iteration of MCMC algorithms at various temperatures
            new_state, new_log_pdf = [], []
            for t, sampler in enumerate(self.mcmc_samplers):
                sampler.iterations_number += 1
                new_state_t, new_log_pdf_t = sampler.run_one_iteration(
                    current_state[t], current_log_pdf[t])
                new_state.append(new_state_t.copy())
                new_log_pdf.append(new_log_pdf_t.copy())

            # Do sweeps if necessary
            if self.mcmc_samplers[-1].iterations_number % self.n_iterations_between_sweeps == 0:
                for i in range(self.n_tempering_parameters - 1):
                    log_accept = (self.mcmc_samplers[i].evaluate_log_target(new_state[i + 1]) +
                                  self.mcmc_samplers[i + 1].evaluate_log_target(new_state[i]) -
                                  self.mcmc_samplers[i].evaluate_log_target(new_state[i]) -
                                  self.mcmc_samplers[i + 1].evaluate_log_target(new_state[i + 1]))
                    for nc, log_accept_chain in enumerate(log_accept):
                        if np.log(self.random_state.rand()) < log_accept_chain:
                            new_state[i][nc], new_state[i + 1][nc] = new_state[i + 1][nc], new_state[i][nc]
                            new_log_pdf[i][nc], new_log_pdf[i + 1][nc] = new_log_pdf[i + 1][nc], new_log_pdf[i][nc]

            # Update the chain, only if burn-in is over and the sample is not being jumped over
            # also increase the current number of samples and samples_per_chain
            if self.mcmc_samplers[-1].iterations_number > self.mcmc_samplers[-1].burn_length and \
                    (self.mcmc_samplers[-1].iterations_number -
                     self.mcmc_samplers[-1].burn_length) % self.mcmc_samplers[-1].jump == 0:
                for t, sampler in enumerate(self.mcmc_samplers):
                    sampler.samples[sampler.nsamples_per_chain, :, :] = new_state[t].copy()
                    if self.save_log_pdf:
                        sampler.log_pdf_values[sampler.nsamples_per_chain, :] = new_log_pdf[t].copy()
                    sampler.nsamples_per_chain += 1
                    sampler.samples_counter += sampler.n_chains

        self.logger.info('UQpy: MCMC run successfully !')

        # Concatenate chains maybe
        if self.mcmc_samplers[-1].concatenate_chains:
            for t, mcmc_sampler in enumerate(self.mcmc_samplers):
                mcmc_sampler._concatenate_chains()

        # Samples connect to posterior samples, i.e. the chain with beta=1.
        self.intermediate_samples = [sampler.samples for sampler in self.mcmc_samplers]
        self.samples = self.mcmc_samplers[-1].samples
        if self.save_log_pdf:
            self.log_pdf_values = self.mcmc_samplers[-1].log_pdf_values

    @beartype
    def evaluate_normalization_constant(self, compute_potential, log_Z0: float = None, nsamples_from_p0: int = None):
        """
        Evaluate new log free energy as

        :math:`\log{Z_{1}} = \log{Z_{0}} + \int_{0}^{1} E_{x~p_{beta}} \left[ U_{\beta}(x) \right] d\beta`

        References (for the Bayesian case):
        * https://emcee.readthedocs.io/en/v2.2.1/user/pt/

        **Inputs:**

        * **compute_potential** (callable):
            Function that takes three inputs (`x`, `log_factor_tempered_values`, `beta`) and computes the potential
            :math:`U_{\beta}(x)`. `log_factor_tempered_values` are the values saved during sampling of
            :math:`\log{p_{\beta}(x)}` at saved samples x.

        * **log_Z0** (`float`):
            Value of :math:`\log{Z_{0}}`

        * **nsamples_from_p0** (`int`):
            N samples from the reference distribution p0. Then :math:`\log{Z_{0}}` is evaluate via MC sampling
            as :math:`\frac{1}{N} \sum{p_{\beta=0}(x)}`. Used only if input *log_Z0* is not provided.

        """
        if not self.save_log_pdf:
            raise NotImplementedError('UQpy: the evidence cannot be computed when save_log_pdf is set to False.')
        if log_Z0 is None and nsamples_from_p0 is None:
            raise ValueError('UQpy: input log_Z0 or nsamples_from_p0 should be provided.')
        # compute average of log_target for the target at various temperatures
        log_pdf_averages = []
        for i, (temper_param, sampler) in enumerate(zip(self.tempering_parameters, self.mcmc_samplers)):
            log_factor_values = sampler.log_pdf_values - self.evaluate_log_reference(sampler.samples)
            potential_values = compute_potential(
                x=sampler.samples, temper_param=temper_param, log_intermediate_values=log_factor_values)
            log_pdf_averages.append(np.mean(potential_values))

        # use quadrature to integrate between 0 and 1
        temper_param_list_for_integration = np.copy(np.array(self.tempering_parameters))
        log_pdf_averages = np.array(log_pdf_averages)
        int_value = trapz(x=temper_param_list_for_integration, y=log_pdf_averages)
        if log_Z0 is None:
            samples_p0 = self.distribution_reference.rvs(nsamples=nsamples_from_p0)
            log_Z0 = np.log(1. / nsamples_from_p0) + logsumexp(
                self.evaluate_log_intermediate(x=samples_p0, temper_param=self.tempering_parameters[0]))

        self.thermodynamic_integration_results = {
            'log_Z0': log_Z0, 'temper_param_list': temper_param_list_for_integration,
            'expect_potentials': log_pdf_averages}

        return np.exp(int_value + log_Z0)
