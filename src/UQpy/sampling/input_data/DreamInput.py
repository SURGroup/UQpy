from dataclasses import dataclass
from typing import Annotated, Tuple

from beartype.vale import Is

from UQpy.sampling.input_data.SamplingInput import SamplingInput
from UQpy.utilities.ValidationTypes import *


@dataclass
class DreamInput(SamplingInput):
    pdf_target = None
    log_pdf_target = None
    args_target = None
    burn_length: Annotated[int, Is[lambda x: x >= 0]] = 0
    jump: PositiveInteger = 1
    dimension: int = None
    seed = None
    save_log_pdf = False
    concatenate_chains: bool = True
    jump_rate: int = 3
    c: float = 0.1
    c_star: float = 1e-6
    crossover_probabilities_number: int = 3
    gamma_probability: float = 0.2
    crossover_adaptation: Tuple = (-1, 1)
    check_chains: Tuple = (-1, 1)
    random_state: RandomStateType = None
    chains_number: int = None
