from dataclasses import dataclass
from UQpy.distributions.baseclass import Distribution
from UQpy.sampling.input_data.SamplingInput import SamplingInput
from UQpy.utilities.ValidationTypes import *


@dataclass
class ISInput(SamplingInput):
    """
    Class for providing input arguments to the :class:`.ImportanceSampling` class.

    :param pdf_target: Callable that evaluates the pdf of the target distribution.
     Either log_pdf_target or pdf_target must be specified (the former is preferred).
    :param log_pdf_target: Callable that evaluates the log-pdf of the target distribution.
     Either log_pdf_target or pdf_target must be specified (the former is preferred).
    :param args_target: Positional arguments of the target log_pdf / pdf callable.
    :param proposal: Proposal to sample from. This UQpy.Distributions object must have an rvs method and a
     log_pdf (or pdf) method.
    :param random_state:  Random seed used to initialize the pseudo-random number generator. Default is None.
     If an integer is provided, this sets the seed for an object of :class:`numpy.random.RandomState`.
     Otherwise, the object itself can be passed directly.
    """
    pdf_target: callable = None
    log_pdf_target: callable = None
    args_target: tuple = None
    proposal: Union[None, Distribution] = None
    random_state: RandomStateType = None
