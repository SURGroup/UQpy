from dataclasses import dataclass

from UQpy.distributions.baseclass import Distribution
from UQpy.sampling.input_data.SamplingInput import SamplingInput
from UQpy.utilities.ValidationTypes import *


@dataclass
class MhInput(SamplingInput):
    pdf_target: callable = None
    log_pdf_target: callable = None
    args_target: tuple = None
    burn_length: Annotated[int, Is[lambda x: x >= 0]] = 0
    jump: int = 1
    dimension: int = None
    seed: list = None
    save_log_pdf: bool = False
    concatenate_chains: bool = True
    chains_number: int = None
    proposal: Distribution = None
    proposal_is_symmetric: bool = False
    random_state: RandomStateType = None
