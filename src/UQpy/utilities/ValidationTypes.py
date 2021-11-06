from typing import Union, Annotated

import numpy as np
from beartype.vale import Is

RandomStateType = Union[None, int, np.random.RandomState]
PositiveInteger = Annotated[int, Is[lambda number: number > 0]]
PositiveFloat = Annotated[float, Is[lambda number: number > 0]]
Numpy2DFloatArray = Annotated[
    np.ndarray,
    Is[lambda array: array.ndim == 2 and np.issubdtype(array.dtype, np.floating)],
]
NumpyFloatArray = Annotated[
    np.ndarray,
    Is[lambda array: np.issubdtype(array.dtype, np.floating)],
]
Numpy2DFloatArrayOrthonormal = Annotated[
    np.ndarray,
    Is[lambda array: array.ndim == 2 and np.issubdtype(array.dtype, np.floating) and
                     np.allclose(array.T @ array, np.eye(array.shape[1]))],
]

