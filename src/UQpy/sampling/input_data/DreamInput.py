from dataclasses import dataclass
from UQpy.sampling.input_data.SamplingInput import SamplingInput
from UQpy.utilities.ValidationTypes import *


@dataclass
class DreamInput(SamplingInput):
    pdf_target: callable = None
    log_pdf_target: callable = None
    args_target: tuple = None
    burn_length: Annotated[int, Is[lambda x: x >= 0]] = 0
    jump: PositiveInteger = 1
    dimension: int = None
    seed: list = None
    save_log_pdf: bool = False
    concatenate_chains: bool = True
    jump_rate: int = 3
    c: float = 0.1
    c_star: float = 1e-6
    crossover_probabilities_number: int = 3
    gamma_probability: float = 0.2
    crossover_adaptation: tuple = (-1, 1)
    check_chains: tuple = (-1, 1)
    random_state: RandomStateType = None
    chains_number: int = None
