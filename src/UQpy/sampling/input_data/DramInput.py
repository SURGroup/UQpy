from dataclasses import dataclass
from typing import Annotated

from beartype.vale import Is

from UQpy.sampling.input_data.SamplingInput import SamplingInput
from UQpy.utilities.ValidationTypes import *


@dataclass
class DramInput(SamplingInput):
    pdf_target = None
    log_pdf_target = None
    args_target = None
    burn_length: Annotated[int, Is[lambda x: x >= 0]] = 0
    jump: PositiveInteger = 1
    dimension: int = None
    seed = None
    save_log_pdf = False
    concatenate_chains = True
    initial_covariance: float = None
    covariance_update_rate: float = 100
    scale_parameter: float = None
    delayed_rejection_scale: float = 1 / 5
    save_covariance: bool = False
    random_state: RandomStateType = None
    chains_number: PositiveInteger = None
