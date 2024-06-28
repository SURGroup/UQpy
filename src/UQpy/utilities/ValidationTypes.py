import torch
import numpy as np
from beartype.vale import Is
from typing import Union, Annotated


RandomStateType = Union[None, int, np.random.RandomState]
PositiveInteger = Annotated[int, Is[lambda number: number > 0]]
NonNegativeInteger = Annotated[int, Is[lambda number: number >= 0]]
PositiveFloat = Annotated[float, Is[lambda number: number > 0]]
Numpy2DFloatArray = Annotated[
    np.ndarray,
    Is[lambda array: array.ndim == 2 and np.issubdtype(array.dtype, float)],
]
NumpyFloatArray = Annotated[
    np.ndarray,
    Is[lambda array: np.issubdtype(array.dtype, float)],
]
NumpyIntArray = Annotated[
    np.ndarray,
    Is[lambda array: np.issubdtype(array.dtype, int)],
]
Numpy2DFloatArrayOrthonormal = Annotated[
    np.ndarray,
    Is[
        lambda array: array.ndim == 2
        and np.issubdtype(array.dtype, float)
        and np.allclose(array.T @ array, np.eye(array.shape[1]))
    ],
]
Torch3DComplexTensor = Annotated[
    torch.Tensor, Is[lambda tensor: tensor.ndim == 3 and tensor.is_complex()]
]
Torch4DComplexTensor = Annotated[
    torch.Tensor, Is[lambda tensor: tensor.ndim == 4 and tensor.is_complex()]
]
Torch5DComplexTensor = Annotated[
    torch.Tensor, Is[lambda tensor: tensor.ndim == 5 and tensor.is_complex()]
]
