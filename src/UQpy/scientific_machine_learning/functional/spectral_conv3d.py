import torch
from typing import Annotated
from beartype import beartype
from beartype.vale import Is
from UQpy.utilities.ValidationTypes import PositiveInteger, Torch5DComplexTensor


@beartype
def spectral_conv3d(
    x: Annotated[torch.Tensor, Is[lambda tensor: tensor.ndim == 5]],
    weights: tuple[
        Torch5DComplexTensor,
        Torch5DComplexTensor,
        Torch5DComplexTensor,
        Torch5DComplexTensor,
    ],
    out_channels: PositiveInteger,
    modes: tuple[PositiveInteger, PositiveInteger, PositiveInteger],
) -> torch.Tensor:
    r"""Compute the 3d spectral convolution :math:`\mathcal{F}^{-1}(R (\mathcal{F}x) )`

    :param x: Tensor of shape :math:`(N, C_\text{in}, H, W, D)`
    :param weights: Tuple of four tensors each with shape :math:`(C_\text{in}, C_\text{out}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]})`.
     All weight tensors must have complex entries.
    :param modes: Tuple of Fourier modes to keep.
     At most :math:`(\lfloor H / 2 \rfloor + 1, \lfloor W / 2 \rfloor + 1, \lfloor D / 2 \rfloor + 1)`
    :param out_channels: :math:`C_\text{out}`, Number of channels in the output signal
    :return: Tensor :math:`\mathcal{F}^{-1}(R (\mathcal{F}x) )` of shape :math:`(N, C_\text{out}, H, W, D)`
    """
    batch_size, in_channels, height, width, depth = x.shape
    if modes[0] > (height // 2) + 1:
        raise ValueError(
            "UQpy: invalid `modes[0]`. `modes[0]` must be less than or equal to (height // 2) + 1"
        )
    if modes[1] > (width // 2) + 1:
        raise ValueError(
            "UQpy: invalid `modes[1]`. `modes[1]` must be less than or equal to (width // 2) + 1"
        )
    if modes[2] > (depth // 2) + 1:
        raise ValueError(
            "UQpy: invalid `modes[2]`. `modes[2]` must be less than or equal to (depth // 2) + 1"
        )
    # Apply Fourier transform
    x_ft = torch.fft.rfftn(x, s=(height, width, depth))
    # Apply linear transform in Fourier space
    out_shape = (
        batch_size,
        out_channels,
        (height // 2) + 1,
        (width // 2) + 1,
        (depth // 2) + 1,
    )
    out_ft = torch.zeros(out_shape, dtype=torch.cfloat)
    indices = [
        (
            slice(None),
            slice(None),
            slice(0, modes[0]),
            slice(0, modes[1]),
            slice(0, modes[2]),
        ),
        (
            slice(None),
            slice(None),
            slice(-modes[0], None),
            slice(0, modes[1]),
            slice(0, modes[2]),
        ),
        (
            slice(None),
            slice(None),
            slice(0, modes[0]),
            slice(-modes[1], None),
            slice(0, modes[2]),
        ),
        (
            slice(None),
            slice(None),
            slice(-modes[0], None),
            slice(-modes[1], None),
            slice(0, modes[2]),
        ),
    ]
    equation = "bixyz,ioxyz->boxyz"
    for i, w in zip(indices, weights):
        out_ft[i] = torch.einsum(equation, x_ft[i], w)
    # Return to physical space
    x = torch.fft.irfftn(out_ft, s=(height, width, depth))
    return x