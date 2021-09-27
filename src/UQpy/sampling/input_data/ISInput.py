from dataclasses import dataclass
from typing import Annotated

from beartype.vale import Is

from UQpy.distributions.baseclass import Distribution
from UQpy.sampling.input_data.SamplingInput import SamplingInput
from UQpy.utilities.ValidationTypes import *


@dataclass(init=True)
class ISInput(SamplingInput):
    pdf_target = None
    log_pdf_target = None
    args_target = None
    proposal: Union[None, Distribution] = None
    random_state: RandomStateType = None
