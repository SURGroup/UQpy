import torch
from typing import Annotated
from beartype import beartype
from beartype.vale import Is
from UQpy.utilities.ValidationTypes import PositiveInteger

tensor5d = Annotated[torch.Tensor, Is[lambda x: x.ndim == 5]]


@beartype
def spectral_conv3d(
    x: tensor5d,
    weights: tuple[tensor5d, tensor5d, tensor5d, tensor5d],
    modes: tuple[PositiveInteger, PositiveInteger, PositiveInteger],
    out_channels: PositiveInteger,
) -> torch.Tensor:
    r"""Compute the 3d spectral convolution :math:`\mathcal{F}^{-1}(R (\mathcal{F}x) )`

    :param x: Tensor of shape :math:`(N, C_\text{in}, H, W, D)`
    :param weights: Tuple of four tensors each with shape :math:`(C_\\text{in}, C_\text{out}, \text{modes1}, \text{modes2}, \text{modes3})`
    :param modes: Tuple of three positive integers
    :param out_channels: :math:`C_\text{out}`, Number of channels in the output signal
    :return: Tensor :math:`\mathcal{F}^{-1}(R (\mathcal{F}x) )` of shape :math:`(N, C_\text{out}, H, W, D)`
    """
    batch_size, in_channels, height, width, depth = x.shape

    # Apply Fourier transform
    x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
    # Apply linear transform in Fourier space
    out_shape = (batch_size, out_channels, height, width, (depth // 2) + 1)
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
    ]  # ToDo: benchmark the 3d FNO.
    equation = "bihwd,iohwd->bohwd"  # "bixyz,ioxyz->boxyz"
    for i, w in zip(indices, weights):
        out_ft[i] = torch.einsum(equation, x_ft[i], w)
    # Return to physical space
    x = torch.fft.irfftn(out_ft, s=(height, width, depth))
    return x
