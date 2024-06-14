import torch
from typing import Annotated
from beartype import beartype
from beartype.vale import Is
from UQpy.utilities.ValidationTypes import PositiveInteger

tensor3d = Annotated[torch.Tensor, Is[lambda x: x.ndim == 3]]


@beartype
def spectral_conv1d(
    x: tensor3d,
    weights: tensor3d,
    out_channels: PositiveInteger,
    modes: PositiveInteger,
) -> torch.Tensor:
    r"""Compute the 1d spectral convolution :math:`\mathcal{F}^{-1}(R (\mathcal{F}x) )`

    :param x: Tensor of shape :math:`(N, C_\text{in}, L)`
    :param weights: Tensor of shape :math:`(C_\text{in}, C_\text{out}, L)`
    :param out_channels: :math:`C_\text{out}`, Number of channels in the output signal
    :param modes: Number of Fourier modes to keep, at most :math:`\lfloor L / 2 \rfloor + 1`
    :return: Tensor :math:`\mathcal{F}^{-1}(R (\mathcal{F}x) )` of shape :math:`(N, C_\text{out}, L)`
    """
    batch_size, in_channels, length = x.shape
    # Apply Fourier transform
    x_ft = torch.fft.rfft(x)
    # Apply linear transform in Fourier space
    out_shape = (batch_size, out_channels, (length // 2 + 1))
    out_ft = torch.zeros(out_shape, dtype=torch.cfloat)
    equation = "bix,iox->box"
    indices = [slice(None), slice(None), slice(0, modes)]  # :, :, :modes
    out_ft[indices] = torch.einsum(equation, x_ft[indices], weights.to(torch.cfloat))
    # Return to physical space
    x = torch.fft.irfft(out_ft, n=length)
    return x
