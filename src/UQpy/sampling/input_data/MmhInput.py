from dataclasses import dataclass

from UQpy.distributions.baseclass import Distribution
from UQpy.sampling.input_data.SamplingInput import SamplingInput
from UQpy.utilities.ValidationTypes import *


@dataclass
class MmhInput(SamplingInput):
    """
    :param pdf_target: Target density function from which to draw random samples. Either pdf_target or
     log_pdf_target must be provided (the latter should be preferred for better numerical stability).

     If pdf_target is a callable, it refers to the joint pdf to sample from, it must take at least one input x, which
     are the point(s) at which to evaluate the pdf. Within MCMC the pdf_target is evaluated as:
     p(x) = pdf_target(x, \*args_target)

     where x is a ndarray of shape (samples_number, dimension) and args_target are additional positional arguments that
     are provided to MCMC via its args_target input.

     If pdf_target is a list of callables, it refers to independent marginals to sample from. The marginal in dimension
     j is evaluated as: p_j(xj) = pdf_target[j](xj, \*args_target[j]) where x is a ndarray of shape (samples_number,
     dimension)
    :param log_pdf_target: Logarithm of the target density function from which to draw random samples.
     Either pdf_target or log_pdf_target must be provided (the latter should be preferred for better numerical
     stability).

     Same comments as for input pdf_target.
    :param args_target: Positional arguments of the pdf / log-pdf target function. See pdf_target
    :param burn_length: Length of burn-in - i.e., number of samples at the beginning of the chain to discard (note:
     no thinning during burn-in). Default is 0, no burn-in.
    :param jump: Thinning parameter, used to reduce correlation between samples. Setting jump=n corresponds to
     skipping n-1 states between accepted states of the chain. Default is 1 (no thinning).
    :param dimension: A scalar value defining the dimension of target density function. Either dimension and
     nchains or seed must be provided.
    :param seed: Seed of the Markov chain(s), shape (chains_number, dimension).
     Default: zeros(chains_number x dimension).

     If seed is not provided, both chains_number and dimension must be provided.
    :param save_log_pdf: Boolean that indicates whether to save log-pdf values along with the samples.
     Default: False
    :param concatenate_chains: Boolean that indicates whether to concatenate the chains after a run, i.e., samples
     are stored as an ndarray of shape (samples_number * chains_number, dimension) if True,
     (samples_number, chains_number, dimension) if False.
     Default: True
    :param chains_number: The number of Markov chains to generate. Either dimension and chains_number or seed must be
     provided.
    :param proposal: Proposal distribution, must have a log_pdf/pdf and rvs method. Default: standard
     multivariate normal
    :param proposal_is_symmetric: Indicates whether the proposal distribution is symmetric, affects computation of
     acceptance probability alpha Default: False, set to True if default proposal is used
    :param random_state: Random seed used to initialize the pseudo-random number generator. Default is
     None.
    """
    pdf_target: callable = None
    log_pdf_target: callable = None
    args_target: tuple = None
    burn_length: Annotated[int, Is[lambda x: x >= 0]] = 0
    jump: PositiveInteger = 1
    dimension: int = None
    seed: list = None
    save_log_pdf: bool = False
    concatenate_chains: bool = True
    proposal: Union[Distribution, list[Distribution]] = None
    proposal_is_symmetric: bool = False
    random_state: RandomStateType = None
    chains_number: int = None
