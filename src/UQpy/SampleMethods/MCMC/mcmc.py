import numpy as np
from UQpy.Distributions import Distribution

class MCMC:
    """
    Generate samples from arbitrary user-specified probability density function using Markov Chain Monte Carlo.

    This is the parent class for all MCMC algorithms. This parent class only provides the framework for MCMC and cannot
    be used directly for sampling. Sampling is done by calling the child class for the specific MCMC algorithm.


    **Inputs:**

    * **dimension** (`int`):
        A scalar value defining the dimension of target density function. Either `dimension` and `nchains` or `seed`
        must be provided.

    * **pdf_target** ((`list` of) callables):
        Target density function from which to draw random samples. Either `pdf_target` or `log_pdf_target` must be
        provided (the latter should be preferred for better numerical stability).

        If `pdf_target` is a callable, it refers to the joint pdf to sample from, it must take at least one input `x`,
        which are the point(s) at which to evaluate the pdf. Within MCMC the `pdf_target` is evaluated as:
        ``p(x) = pdf_target(x, *args_target)``

        where `x` is a ndarray of shape (nsamples, dimension) and `args_target` are additional positional arguments that
        are provided to MCMC via its `args_target` input.

        If `pdf_target` is a list of callables, it refers to independent marginals to sample from. The marginal in
        dimension `j` is evaluated as: ``p_j(xj) = pdf_target[j](xj, *args_target[j])`` where `x` is a ndarray of shape
        (nsamples, dimension)

    * **log_pdf_target** ((`list` of) callables):
        Logarithm of the target density function from which to draw random samples. Either `pdf_target` or
        `log_pdf_target` must be provided (the latter should be preferred for better numerical stability).

        Same comments as for input `pdf_target`.

    * **args_target** ((`list` of) `tuple`):
        Positional arguments of the pdf / log-pdf target function. See `pdf_target`

    * **seed** (`ndarray`):
        Seed of the Markov chain(s), shape ``(nchains, dimension)``. Default: zeros(`nchains` x `dimension`).

        If `seed` is not provided, both `nchains` and `dimension` must be provided.

    * **nburn** (`int`):
        Length of burn-in - i.e., number of samples at the beginning of the chain to discard (note: no thinning during
        burn-in). Default is 0, no burn-in.

    * **jump** (`int`):
        Thinning parameter, used to reduce correlation between samples. Setting `jump=n` corresponds to	skipping `n-1`
        states between accepted states of the chain. Default is 1 (no thinning).

    * **nchains** (`int`):
        The number of Markov chains to generate. Either `dimension` and `nchains` or `seed` must be provided.

    * **save_log_pdf** (`bool`):
        Boolean that indicates whether to save log-pdf values along with the samples. Default: False

    * **verbose** (`boolean`)
        Set ``verbose = True`` to print status messages to the terminal during execution.

    * **concat_chains** (`bool`):
        Boolean that indicates whether to concatenate the chains after a run, i.e., samples are stored as an `ndarray`
        of shape (nsamples * nchains, dimension) if True, (nsamples, nchains, dimension) if False. Default: True

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.


    **Attributes:**

    * **samples** (`ndarray`)
        Set of MCMC samples following the target distribution, `ndarray` of shape (`nsamples` * `nchains`, `dimension`)
        or (nsamples, nchains, dimension) (see input `concat_chains`).

    * **log_pdf_values** (`ndarray`)
        Values of the log pdf for the accepted samples, `ndarray` of shape (nchains * nsamples,) or (nsamples, nchains)

    * **nsamples** (`list`)
        Total number of samples; The `nsamples` attribute tallies the total number of generated samples. After each
        iteration, it is updated by 1. At the end of the simulation, the `nsamples` attribute equals the user-specified
        value for input `nsamples` given to the child class.

    * **nsamples_per_chain** (`list`)
        Total number of samples per chain; Similar to the attribute `nsamples`, it is updated during iterations as new
        samples are saved.

    * **niterations** (`list`)
        Total number of iterations, updated on-the-fly as the algorithm proceeds. It is related to number of samples as
        niterations=nburn+jump*nsamples_per_chain.

    * **acceptance_rate** (`list`)
        Acceptance ratio of the MCMC chains, computed separately for each chain.

    **Methods:**
    """
    # Last Modified: 10/05/20 by Audrey Olivier

    def __init__(self, dimension=None, pdf_target=None, log_pdf_target=None, args_target=None, seed=None, nburn=0,
                 jump=1, nchains=None, save_log_pdf=False, verbose=False, concat_chains=True, random_state=None):

        if not (isinstance(nburn, int) and nburn >= 0):
            raise TypeError('UQpy: nburn should be an integer >= 0')
        if not (isinstance(jump, int) and jump >= 1):
            raise TypeError('UQpy: jump should be an integer >= 1')
        self.nburn, self.jump = nburn, jump
        self.seed = self._preprocess_seed(seed=seed, dim=dimension, nchains=nchains)
        self.nchains, self.dimension = self.seed.shape

        # Check target pdf
        self.evaluate_log_target, self.evaluate_log_target_marginals = self._preprocess_target(
            pdf_=pdf_target, log_pdf_=log_pdf_target, args=args_target)
        self.save_log_pdf = save_log_pdf
        self.concat_chains = concat_chains
        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')
        self.verbose = verbose

        self.log_pdf_target = log_pdf_target
        self.pdf_target = pdf_target
        self.args_target = args_target

        # Initialize a few more variables
        self.samples = None
        self.log_pdf_values = None
        self.acceptance_rate = [0.] * self.nchains
        self.nsamples, self.nsamples_per_chain = 0, 0
        self.niterations = 0  # total nb of iterations, grows if you call run several times

    def run(self, nsamples=None, nsamples_per_chain=None):
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
        # Initialize the runs: allocate space for the new samples and log pdf values
        final_nsamples, final_nsamples_per_chain, current_state, current_log_pdf = self._initialize_samples(
            nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

        if self.verbose:
            print('UQpy: Running MCMC...')

        # Run nsims iterations of the MCMC algorithm, starting at current_state
        while self.nsamples_per_chain < final_nsamples_per_chain:
            # update the total number of iterations
            self.niterations += 1
            # run iteration
            current_state, current_log_pdf = self.run_one_iteration(current_state, current_log_pdf)
            # Update the chain, only if burn-in is over and the sample is not being jumped over
            # also increase the current number of samples and samples_per_chain
            if self.niterations > self.nburn and (self.niterations - self.nburn) % self.jump == 0:
                self.samples[self.nsamples_per_chain, :, :] = current_state.copy()
                if self.save_log_pdf:
                    self.log_pdf_values[self.nsamples_per_chain, :] = current_log_pdf.copy()
                self.nsamples_per_chain += 1
                self.nsamples += self.nchains

        if self.verbose:
            print('UQpy: MCMC run successfully !')

        # Concatenate chains maybe
        if self.concat_chains:
            self._concatenate_chains()

    def run_one_iteration(self, current_state, current_log_pdf):
        """
        Run one iteration of the MCMC algorithm, starting at `current_state`.

        This method is over-written for each different MCMC algorithm. It must return the new state and associated
        log-pdf, which will be passed as inputs to the ``run_one_iteration`` method at the next iteration.

        **Inputs:**

        * **current_state** (`ndarray`):
            Current state of the chain(s), `ndarray` of shape ``(nchains, dimension)``.

        * **current_log_pdf** (`ndarray`):
            Log-pdf of the current state of the chain(s), `ndarray` of shape ``(nchains, )``.

        **Outputs/Returns:**

        * **new_state** (`ndarray`):
            New state of the chain(s), `ndarray` of shape ``(nchains, dimension)``.

        * **new_log_pdf** (`ndarray`):
            Log-pdf of the new state of the chain(s), `ndarray` of shape ``(nchains, )``.

        """
        return [], []

    ####################################################################################################################
    # Helper functions that can be used by all algorithms
    # Methods update_samples, update_accept_ratio and sample_candidate_from_proposal can be called in the run stage.
    # Methods preprocess_target, preprocess_proposal, check_seed and check_integers can be called in the init stage.

    def _concatenate_chains(self):
        """
        Concatenate chains.

        Utility function that reshapes (in place) attribute samples from (nsamples, nchains, dimension) to
        (nsamples * nchains, dimension), and log_pdf_values from (nsamples, nchains) to (nsamples * nchains, ).

        No input / output.

        """
        self.samples = self.samples.reshape((-1, self.dimension), order='C')
        if self.save_log_pdf:
            self.log_pdf_values = self.log_pdf_values.reshape((-1, ), order='C')
        return None

    def _unconcatenate_chains(self):
        """
        Inverse of concatenate_chains.

        Utility function that reshapes (in place) attribute samples from (nsamples * nchains, dimension) to
        (nsamples, nchains, dimension), and log_pdf_values from (nsamples * nchains) to (nsamples, nchains).

        No input / output.

        """
        self.samples = self.samples.reshape((-1, self.nchains, self.dimension), order='C')
        if self.save_log_pdf:
            self.log_pdf_values = self.log_pdf_values.reshape((-1, self.nchains), order='C')
        return None

    def _initialize_samples(self, nsamples, nsamples_per_chain):
        """
        Initialize necessary attributes and variables before running the chain forward.

        Utility function that allocates space for samples and log likelihood values, initialize sample_index,
        acceptance ratio. If some samples already exist, allocate space to append new samples to the old ones. Computes
        the number of forward iterations nsims to be run (depending on burnin and jump parameters).

        **Inputs:**

        * nchains (int): number of chains run in parallel
        * nsamples (int): number of samples to be generated
        * nsamples_per_chain (int): number of samples to be generated per chain

        **Output/Returns:**

        * nsims (int): Number of iterations to perform
        * current_state (ndarray of shape (nchains, dim)): Current state of the chain to start from.

        """
        if ((nsamples is not None) and (nsamples_per_chain is not None)) or (
                nsamples is None and nsamples_per_chain is None):
            raise ValueError('UQpy: Either nsamples or nsamples_per_chain must be provided (not both)')
        if nsamples_per_chain is not None:
            if not (isinstance(nsamples_per_chain, int) and nsamples_per_chain >= 0):
                raise TypeError('UQpy: nsamples_per_chain must be an integer >= 0.')
            nsamples = int(nsamples_per_chain * self.nchains)
        else:
            if not (isinstance(nsamples, int) and nsamples >= 0):
                raise TypeError('UQpy: nsamples must be an integer >= 0.')
            nsamples_per_chain = int(np.ceil(nsamples / self.nchains))
            nsamples = int(nsamples_per_chain * self.nchains)

        if self.samples is None:    # very first call of run, set current_state as the seed and initialize self.samples
            self.samples = np.zeros((nsamples_per_chain, self.nchains, self.dimension))
            if self.save_log_pdf:
                self.log_pdf_values = np.zeros((nsamples_per_chain, self.nchains))
            current_state = np.zeros_like(self.seed)
            np.copyto(current_state, self.seed)
            current_log_pdf = self.evaluate_log_target(current_state)
            if self.nburn == 0:    # if nburn is 0, save the seed, run one iteration less 
                self.samples[0, :, :] = current_state
                if self.save_log_pdf:
                    self.log_pdf_values[0, :] = current_log_pdf
                self.nsamples_per_chain += 1
                self.nsamples += self.nchains
            final_nsamples, final_nsamples_per_chain = nsamples, nsamples_per_chain

        else:    # fetch previous samples to start the new run, current state is last saved sample
            if len(self.samples.shape) == 2:   # the chains were previously concatenated
                self._unconcatenate_chains()
            current_state = self.samples[-1]
            current_log_pdf = self.evaluate_log_target(current_state)
            self.samples = np.concatenate(
                [self.samples, np.zeros((nsamples_per_chain, self.nchains, self.dimension))], axis=0)
            if self.save_log_pdf:
                self.log_pdf_values = np.concatenate(
                    [self.log_pdf_values, np.zeros((nsamples_per_chain, self.nchains))], axis=0)
            final_nsamples = nsamples + self.nsamples
            final_nsamples_per_chain = nsamples_per_chain + self.nsamples_per_chain

        return final_nsamples, final_nsamples_per_chain, current_state, current_log_pdf

    def _update_acceptance_rate(self, new_accept=None):
        """
        Update acceptance rate of the chains.

        Utility function, uses an iterative function to update the acceptance rate of all the chains separately.

        **Inputs:**

        * new_accept (list (length nchains) of bool): indicates whether the current state was accepted (for each chain
          separately).

        """
        self.acceptance_rate = [na / self.niterations + (self.niterations - 1) / self.niterations * a
                                for (na, a) in zip(new_accept, self.acceptance_rate)]

    @staticmethod
    def _preprocess_target(log_pdf_, pdf_, args):
        """
        Preprocess the target pdf inputs.

        Utility function (static method), that transforms the log_pdf, pdf, args inputs into a function that evaluates
        log_pdf_target(x) for a given x. If the target is given as a list of callables (marginal pdfs), the list of
        log margianals is also returned.

        **Inputs:**

        * log_pdf_ ((list of) callables): Log of the target density function from which to draw random samples. Either
          pdf_target or log_pdf_target must be provided.
        * pdf_ ((list of) callables): Target density function from which to draw random samples. Either pdf_target or
          log_pdf_target must be provided.
        * args (tuple): Positional arguments of the pdf target.

        **Output/Returns:**

        * evaluate_log_pdf (callable): Callable that computes the log of the target density function
        * evaluate_log_pdf_marginals (list of callables): List of callables to compute the log pdf of the marginals

        """
        # log_pdf is provided
        if log_pdf_ is not None:
            if callable(log_pdf_):
                if args is None:
                    args = ()
                evaluate_log_pdf = (lambda x: log_pdf_(x, *args))
                evaluate_log_pdf_marginals = None
            elif isinstance(log_pdf_, list) and (all(callable(p) for p in log_pdf_)):
                if args is None:
                    args = [()] * len(log_pdf_)
                if not (isinstance(args, list) and len(args) == len(log_pdf_)):
                    raise ValueError('UQpy: When log_pdf_target is a list, args should be a list (of tuples) of same '
                                     'length.')
                evaluate_log_pdf_marginals = list(
                    map(lambda i: lambda x: log_pdf_[i](x, *args[i]), range(len(log_pdf_))))
                evaluate_log_pdf = (lambda x: np.sum(
                    [log_pdf_[i](x[:, i, np.newaxis], *args[i]) for i in range(len(log_pdf_))]))
            else:
                raise TypeError('UQpy: log_pdf_target must be a callable or list of callables')
        # pdf is provided
        elif pdf_ is not None:
            if callable(pdf_):
                if args is None:
                    args = ()
                evaluate_log_pdf = (lambda x: np.log(np.maximum(pdf_(x, *args), 10 ** (-320) * np.ones((x.shape[0],)))))
                evaluate_log_pdf_marginals = None
            elif isinstance(pdf_, (list, tuple)) and (all(callable(p) for p in pdf_)):
                if args is None:
                    args = [()] * len(pdf_)
                if not (isinstance(args, (list, tuple)) and len(args) == len(pdf_)):
                    raise ValueError('UQpy: When pdf_target is given as a list, args should also be a list of same '
                                     'length.')
                evaluate_log_pdf_marginals = list(
                    map(lambda i: lambda x: np.log(np.maximum(pdf_[i](x, *args[i]),
                                                              10 ** (-320) * np.ones((x.shape[0],)))),
                        range(len(pdf_))
                        ))
                evaluate_log_pdf = (lambda x: np.sum(
                    [np.log(np.maximum(pdf_[i](x[:, i, np.newaxis], *args[i]), 10**(-320)*np.ones((x.shape[0],))))
                     for i in range(len(pdf_))]))
            else:
                raise TypeError('UQpy: pdf_target must be a callable or list of callables')
        else:
            raise ValueError('UQpy: log_pdf_target or pdf_target should be provided.')
        return evaluate_log_pdf, evaluate_log_pdf_marginals

    @staticmethod
    def _preprocess_seed(seed, dim, nchains):
        """
        Preprocess input seed.

        Utility function (static method), that checks the dimension of seed, assign [0., 0., ..., 0.] if not provided.

        **Inputs:**

        * seed (ndarray): seed for MCMC
        * dim (int): dimension of target density

        **Output/Returns:**

        * seed (ndarray): seed for MCMC
        * dim (int): dimension of target density

        """
        if seed is None:
            if dim is None or nchains is None:
                raise ValueError('UQpy: Either `seed` or `dimension` and `nchains` must be provided.')
            seed = np.zeros((nchains, dim))
        else:
            seed = np.atleast_1d(seed)
            if len(seed.shape) == 1:
                seed = np.reshape(seed, (1, -1))
            elif len(seed.shape) > 2:
                raise ValueError('UQpy: Input seed should be an array of shape (dimension, ) or (nchains, dimension).')
            if dim is not None and seed.shape[1] != dim:
                raise ValueError('UQpy: Wrong dimensions between seed and dimension.')
            if nchains is not None and seed.shape[0] != nchains:
                raise ValueError('UQpy: The number of chains and the seed shape are inconsistent.')
        return seed

    @staticmethod
    def _check_methods_proposal(proposal):
        """
        Check if proposal has required methods.

        Utility function (static method), that checks that the given proposal distribution has 1) a rvs method and 2) a
        log pdf or pdf method. If a pdf method exists but no log_pdf, the log_pdf methods is added to the proposal
        object. Used in the MH and MMH initializations.

        **Inputs:**

        * proposal (Distribution object): proposal distribution

        """
        if not isinstance(proposal, Distribution):
            raise TypeError('UQpy: Proposal should be a Distribution object')
        if not hasattr(proposal, 'rvs'):
            raise AttributeError('UQpy: The proposal should have an rvs method')
        if not hasattr(proposal, 'log_pdf'):
            if not hasattr(proposal, 'pdf'):
                raise AttributeError('UQpy: The proposal should have a log_pdf or pdf method')
            proposal.log_pdf = lambda x: np.log(np.maximum(proposal.pdf(x), 10 ** (-320) * np.ones((x.shape[0],))))


