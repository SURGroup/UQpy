from dataclasses import dataclass
from UQpy.sampling.input_data.SamplingInput import SamplingInput
from UQpy.utilities.ValidationTypes import *


@dataclass
class MmhInput(SamplingInput):
    pdf_target = None
    log_pdf_target = None
    args_target = None
    burn_length: Annotated[int, Is[lambda x: x >= 0]] = 0
    jump: PositiveInteger = 1
    dimension: int = None
    seed = None
    save_log_pdf: bool = False
    concatenate_chains: bool = True
    proposal = None
    proposal_is_symmetric: bool = False
    random_state: RandomStateType = None
    chains_number: int = None
