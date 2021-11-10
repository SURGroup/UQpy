import logging
from beartype import beartype
from UQpy.distributions import Distribution
from UQpy.utilities.ValidationTypes import *
from UQpy.utilities.Utilities import process_random_state
from abc import ABC


class MCMC(ABC):

    # Last Modified: 10/05/20 by Audrey Olivier
    @beartype
    def __init__(
        self,
        dimension: Union[None, int] = None,
        pdf_target=None,
        log_pdf_target=None,
        args_target=None,
        seed=None,
        burn_length: Annotated[int, Is[lambda x: x >= 0]] = 0,
        jump: PositiveInteger = 1,
        chains_number: Union[None, int] = None,
        save_log_pdf: bool = False,
        concatenate_chains: bool = True,
        random_state: RandomStateType = None,
    ):
        """
        Generate samples from arbitrary user-specified probability density function using Markov Chain Monte Carlo.

        This is the parent class for all mcmc algorithms. This parent class only provides the framework for
        mcmc and cannot be used directly for sampling. Sampling is done by calling the child class for the specific
        mcmc algorithm.

        :param dimension: A scalar value defining the dimension of target density function. Either `dimension` and
         `chains_number` or `seed` must be provided.
        :param pdf_target: Target density function from which to draw random samples. Either `pdf_target` or
         `log_pdf_target` must be provided (the latter should be preferred for better numerical stability).
         If `pdf_target` is a callable, it refers to the joint pdf to sample from, it must take at least one input `x`,
         which are the point(s) at which to evaluate the pdf. Within mcmc the `pdf_target` is evaluated as:
         ``p(x) = pdf_target(x, *args_target)`` where `x` is a ndarray of shape (nsamples, dimension) and `args_target`
         are additional positional arguments that are provided to mcmc via its `args_target` input.
         If `pdf_target` is a list of callables, it refers to independent marginals to sample from. The marginal in
         dimension `j` is evaluated as: ``p_j(xj) = pdf_target[j](xj, *args_target[j])`` where `x` is a ndarray of shape
         (nsamples, dimension)
        :param log_pdf_target: Logarithm of the target density function from which to draw random samples. Either
         `pdf_target` or `log_pdf_target` must be provided (the latter should be preferred for better numerical
         stability).
        :param args_target: Positional arguments of the pdf / log-pdf target function. See `pdf_target`
        :param seed: Seed of the Markov chain(s), shape ``(nchains, dimension)``. Default: zeros(`nchains` x
         `dimension`). If `seed` is not provided, both `nchains` and `dimension` must be provided.
        :param burn_length: Length of burn-in - i.e., number of samples at the beginning of the chain to discard (note:
         no thinning during burn-in). Default is 0, no burn-in.
        :param jump: Thinning parameter, used to reduce correlation between samples. Setting `jump=n` corresponds to
         skipping `n-1` states between accepted states of the chain. Default is 1 (no thinning).
        :param chains_number: The number of Markov chains to generate. Either `dimension` and `nchains` or `seed` must
         be provided.
        :param save_log_pdf: Boolean that indicates whether to save log-pdf values along with the samples.
         Default: False
        :param concatenate_chains: Boolean that indicates whether to concatenate the chains after a run, i.e., samples
         are stored as an `ndarray` of shape (nsamples * nchains, dimension) if True, (nsamples, nchains, dimension) if
         False. Default: True
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is None.
         If an integer is provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise,
         the object itself can be passed directly.
        """
        self.burn_length, self.jump = burn_length, jump
        self.seed = self._preprocess_seed(
            seed=seed, dimensions=dimension, chains_number=chains_number
        )
        self.chains_number, self.dimension = self.seed.shape

        # Check target pdf
        (
            self.evaluate_log_target,
            self.evaluate_log_target_marginals,
        ) = self._preprocess_target(
            pdf_=pdf_target, log_pdf_=log_pdf_target, args=args_target
        )
        self.save_log_pdf = save_log_pdf
        self.concatenate_chains = concatenate_chains
        self.random_state = process_random_state(random_state)
        self.logger = logging.getLogger(__name__)

        self.log_pdf_target = log_pdf_target
        self.pdf_target = pdf_target
        self.args_target = args_target

        # Initialize a few more variables
        self.samples = None
        """Set of MCMC samples following the target distribution, `ndarray` of shape 
        (`samples_number` * `chains_number`, `dimension`)
        or (samples_number, chains_number, dimension) (see input `concatenate_chains`)."""
        self.log_pdf_values = None
        """Values of the log pdf for the accepted samples, `ndarray` of shape (chains_number * samples_number,) or 
        (samples_number, chains_number)"""
        self.acceptance_rate = [0.0] * self.chains_number
        self.samples_number = 0
        """Total number of samples; The `samples_number` attribute tallies the total number of generated samples. After 
        each iteration, it is updated by 1. At the end of the simulation, the `samples_number` attribute equals the 
        user-specified value for input `samples_number` given to the child class."""
        self.samples_number_per_chain = 0
        """Total number of samples per chain; Similar to the attribute `nsamples`, it is updated during iterations as new
        samples are saved."""
        self.iterations_number = (
            0  # total nb of iterations, grows if you call run several times
        )
        """Total number of iterations, updated on-the-fly as the algorithm proceeds. It is related to number of samples 
        as iterations_number=burn_length+jump*samples_number_per_chain."""

    def run(
        self, samples_number: PositiveInteger = None, samples_number_per_chain=None
    ):
        """
        Run the mcmc algorithm.

        This function samples from the mcmc chains and appends samples to existing ones (if any).
        This method leverages the :meth:`run_iterations` method that is specific to each algorithm.

        :param samples_number: Number of samples to generate.
        :param samples_number_per_chain: number of samples to generate per chain.

        Either `samples_number` or `samples_number_per_chain` must be provided (not both). Not that if `samples_number`
        is not a multiple of `chains_number`, `samples_number` is set to the next largest integer that is a multiple of
        `chains_number`.
        """

        # Initialize the runs: allocate space for the new samples and log pdf values
        (
            final_nsamples,
            final_nsamples_per_chain,
            current_state,
            current_log_pdf,
        ) = self._initialize_samples(
            number_of_samples=samples_number,
            samples_per_chain_number=samples_number_per_chain,
        )

        self.logger.info("UQpy: Running mcmc...")

        # Run nsims iterations of the mcmc algorithm, starting at current_state
        while self.samples_number_per_chain < final_nsamples_per_chain:
            # update the total number of iterations
            self.iterations_number += 1
            # run iteration
            current_state, current_log_pdf = self.run_one_iteration(
                current_state, current_log_pdf
            )
            # Update the chain, only if burn-in is over and the sample is not being jumped over
            # also increase the current number of samples and samples_per_chain
            if (
                self.iterations_number > self.burn_length
                and (self.iterations_number - self.burn_length) % self.jump == 0
            ):
                self.samples[self.samples_number_per_chain, :, :] = current_state.copy()
                if self.save_log_pdf:
                    self.log_pdf_values[
                        self.samples_number_per_chain, :
                    ] = current_log_pdf.copy()
                self.samples_number_per_chain += 1
                self.samples_number += self.chains_number

        self.logger.info("UQpy: mcmc run successfully !")

        # Concatenate chains maybe
        if self.concatenate_chains:
            self._concatenate_chains()

    def run_one_iteration(self, current_state, current_log_pdf):
        """
        Run one iteration of the mcmc algorithm, starting at `current_state`.

        This method is over-written for each different mcmc algorithm. It must return the new state and
        associated log-pdf, which will be passed as inputs to the :meth:`run_one_iteration` method at the next
        iteration.

        :param current_state: Current state of the chain(s), `ndarray` of shape ``(chains_number, dimension)``.
        :param current_log_pdf: Log-pdf of the current state of the chain(s), `ndarray` of shape ``(chains_number, )``.
        :return: New state of the chain(s) and Log-pdf of the new state of the chain(s)
        """
        return [], []

    def _concatenate_chains(self):
        self.samples = self.samples.reshape((-1, self.dimension), order="C")
        if self.save_log_pdf:
            self.log_pdf_values = self.log_pdf_values.reshape((-1,), order="C")
        return None

    def _unconcatenate_chains(self):
        self.samples = self.samples.reshape(
            (-1, self.chains_number, self.dimension), order="C"
        )
        if self.save_log_pdf:
            self.log_pdf_values = self.log_pdf_values.reshape(
                (-1, self.chains_number), order="C"
            )
        return None

    def _initialize_samples(self, number_of_samples, samples_per_chain_number):
        if (
            (number_of_samples is not None) and (samples_per_chain_number is not None)
        ) or (number_of_samples is None and samples_per_chain_number is None):
            raise ValueError(
                "UQpy: Either nsamples or nsamples_per_chain must be provided (not both)"
            )
        if samples_per_chain_number is not None:
            if not (
                isinstance(samples_per_chain_number, int)
                and samples_per_chain_number >= 0
            ):
                raise TypeError("UQpy: nsamples_per_chain must be an integer >= 0.")
            number_of_samples = int(samples_per_chain_number * self.chains_number)
        else:
            if not (isinstance(number_of_samples, int) and number_of_samples >= 0):
                raise TypeError("UQpy: nsamples must be an integer >= 0.")
            samples_per_chain_number = int(
                np.ceil(number_of_samples / self.chains_number)
            )
            number_of_samples = int(samples_per_chain_number * self.chains_number)

        if (
            self.samples is None
        ):  # very first call of run, set current_state as the seed and initialize self.samples
            self.samples = np.zeros(
                (samples_per_chain_number, self.chains_number, self.dimension)
            )
            if self.save_log_pdf:
                self.log_pdf_values = np.zeros(
                    (samples_per_chain_number, self.chains_number)
                )
            current_state = np.zeros_like(self.seed)
            np.copyto(current_state, self.seed)
            current_log_pdf = self.evaluate_log_target(current_state)
            if (
                self.burn_length == 0
            ):  # if nburn is 0, save the seed, run one iteration less
                self.samples[0, :, :] = current_state
                if self.save_log_pdf:
                    self.log_pdf_values[0, :] = current_log_pdf
                self.samples_number_per_chain += 1
                self.samples_number += self.chains_number
            final_nsamples, final_nsamples_per_chain = (
                number_of_samples,
                samples_per_chain_number,
            )

        else:  # fetch previous samples to start the new run, current state is last saved sample
            if len(self.samples.shape) == 2:  # the chains were previously concatenated
                self._unconcatenate_chains()
            current_state = self.samples[-1]
            current_log_pdf = self.evaluate_log_target(current_state)
            self.samples = np.concatenate(
                [
                    self.samples,
                    np.zeros(
                        (samples_per_chain_number, self.chains_number, self.dimension)
                    ),
                ],
                axis=0,
            )
            if self.save_log_pdf:
                self.log_pdf_values = np.concatenate(
                    [
                        self.log_pdf_values,
                        np.zeros((samples_per_chain_number, self.chains_number)),
                    ],
                    axis=0,
                )
            final_nsamples = number_of_samples + self.samples_number
            final_nsamples_per_chain = (
                samples_per_chain_number + self.samples_number_per_chain
            )

        return final_nsamples, final_nsamples_per_chain, current_state, current_log_pdf

    def _update_acceptance_rate(self, chain_state_acceptance=None):
        self.acceptance_rate = [
            na / self.iterations_number
            + (self.iterations_number - 1) / self.iterations_number * a
            for (na, a) in zip(chain_state_acceptance, self.acceptance_rate)
        ]

    @staticmethod
    def _preprocess_target(log_pdf_, pdf_, args):
        # log_pdf is provided
        if log_pdf_ is not None:
            if callable(log_pdf_):
                if args is None:
                    args = ()
                evaluate_log_pdf = lambda x: log_pdf_(x, *args)
                evaluate_log_pdf_marginals = None
            elif isinstance(log_pdf_, list) and (all(callable(p) for p in log_pdf_)):
                if args is None:
                    args = [()] * len(log_pdf_)
                if not (isinstance(args, list) and len(args) == len(log_pdf_)):
                    raise ValueError(
                        "UQpy: When log_pdf_target is a list, args should be a list (of tuples) of same "
                        "length."
                    )
                evaluate_log_pdf_marginals = list(
                    map(
                        lambda i: lambda x: log_pdf_[i](x, *args[i]),
                        range(len(log_pdf_)),
                    )
                )
                evaluate_log_pdf = lambda x: np.sum(
                    [
                        log_pdf_[i](x[:, i, np.newaxis], *args[i])
                        for i in range(len(log_pdf_))
                    ]
                )
            else:
                raise TypeError(
                    "UQpy: log_pdf_target must be a callable or list of callables"
                )
        # pdf is provided
        elif pdf_ is not None:
            if callable(pdf_):
                if args is None:
                    args = ()
                evaluate_log_pdf = lambda x: np.log(
                    np.maximum(pdf_(x, *args), 10 ** (-320) * np.ones((x.shape[0],)))
                )
                evaluate_log_pdf_marginals = None
            elif isinstance(pdf_, (list, tuple)) and (all(callable(p) for p in pdf_)):
                if args is None:
                    args = [()] * len(pdf_)
                if not (isinstance(args, (list, tuple)) and len(args) == len(pdf_)):
                    raise ValueError(
                        "UQpy: When pdf_target is given as a list, args should also be a list of same "
                        "length."
                    )
                evaluate_log_pdf_marginals = list(
                    map(
                        lambda i: lambda x: np.log(
                            np.maximum(
                                pdf_[i](x, *args[i]),
                                10 ** (-320) * np.ones((x.shape[0],)),
                            )
                        ),
                        range(len(pdf_)),
                    )
                )
                evaluate_log_pdf = lambda x: np.sum(
                    [
                        np.log(
                            np.maximum(
                                pdf_[i](x[:, i, np.newaxis], *args[i]),
                                10 ** (-320) * np.ones((x.shape[0],)),
                            )
                        )
                        for i in range(len(pdf_))
                    ]
                )
            else:
                raise TypeError(
                    "UQpy: pdf_target must be a callable or list of callables"
                )
        else:
            raise ValueError("UQpy: log_pdf_target or pdf_target should be provided.")
        return evaluate_log_pdf, evaluate_log_pdf_marginals

    @staticmethod
    def _preprocess_seed(seed, dimensions, chains_number):
        if seed is None:
            if dimensions is None or chains_number is None:
                raise ValueError(
                    "UQpy: Either `seed` or `dimension` and `nchains` must be provided."
                )
            seed = np.zeros((chains_number, dimensions))
        else:
            seed = np.atleast_1d(seed)
            if len(seed.shape) == 1:
                seed = np.reshape(seed, (1, -1))
            elif len(seed.shape) > 2:
                raise ValueError(
                    "UQpy: Input seed should be an array of shape (dimension, ) or (nchains, dimension)."
                )
            if dimensions is not None and seed.shape[1] != dimensions:
                raise ValueError("UQpy: Wrong dimensions between seed and dimension.")
            if chains_number is not None and seed.shape[0] != chains_number:
                raise ValueError(
                    "UQpy: The number of chains and the seed shape are inconsistent."
                )
        return seed

    @staticmethod
    def _check_methods_proposal(proposal_distribution):
        if not isinstance(proposal_distribution, Distribution):
            raise TypeError("UQpy: Proposal should be a Distribution object")
        if not hasattr(proposal_distribution, "rvs"):
            raise AttributeError("UQpy: The proposal should have an rvs method")
        if not hasattr(proposal_distribution, "log_pdf"):
            if not hasattr(proposal_distribution, "pdf"):
                raise AttributeError(
                    "UQpy: The proposal should have a log_pdf or pdf method"
                )
            proposal_distribution.log_pdf = lambda x: np.log(
                np.maximum(
                    proposal_distribution.pdf(x), 10 ** (-320) * np.ones((x.shape[0],))
                )
            )
