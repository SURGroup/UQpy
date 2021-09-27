from dataclasses import dataclass
from UQpy.sampling.input_data.SamplingInput import SamplingInput
from UQpy.utilities.ValidationTypes import *


@dataclass
class MhInput(SamplingInput):
    pdf_target = None
    log_pdf_target = None
    args_target = None
    burn_length: Annotated[int, Is[lambda x: x >= 0]] = 0
    jump = 1
    dimension: int = None
    seed = None
    save_log_pdf = False
    concatenate_chains: bool = True
    chains_number: int = None
    proposal = None
    proposal_is_symmetric: bool = False
    random_state: RandomStateType = None
