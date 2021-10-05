from dataclasses import dataclass
from UQpy.distributions.baseclass import Distribution
from UQpy.sampling.input_data.SamplingInput import SamplingInput
from UQpy.utilities.ValidationTypes import *


@dataclass
class ISInput(SamplingInput):
    pdf_target:  callable = None
    log_pdf_target: callable = None
    args_target: tuple = None
    proposal: Union[None, Distribution] = None
    random_state: RandomStateType = None
