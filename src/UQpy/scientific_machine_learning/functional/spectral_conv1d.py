import torch
from typing import Annotated
from beartype import beartype
from beartype.vale import Is
from UQpy.utilities.ValidationTypes import PositiveInteger


@beartype
def spectral_conv1d(
    x: Annotated[torch.Tensor, Is[lambda x: x.ndim == 3]],
    weights: Annotated[torch.Tensor, Is[lambda x: x.ndim == 3]],
    out_channels: PositiveInteger,
    modes: PositiveInteger,
) -> torch.Tensor:
    """

    :param x:
    :param weights:
    :param out_channels:
    :param modes:
    :return:
    """
    batch_size, in_channels, length = x.shape
    # Apply Fourier transform
    x_ft = torch.fft.rfft(x)
    # Apply linear transform in Fourier space
    out_shape = (batch_size, out_channels, (length // 2 + 1))
    out_ft = torch.zeros(out_shape, dtype=torch.cfloat)
    equation = "bix,iox->box"
    indices = [slice(None), slice(None), slice(0, modes)]
    out_ft[indices] = torch.einsum(equation, x_ft[indices], weights.to(torch.cfloat))
    # Return to physical space
    x = torch.fft.irfft(out_ft, n=length)
    return x
