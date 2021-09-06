from typing import Union, Annotated

import numpy as np
from beartype.vale import Is

RandomStateType = Union[None, int, np.random.RandomState]
PositiveInteger = Annotated[int, Is[lambda number: number > 0]]
