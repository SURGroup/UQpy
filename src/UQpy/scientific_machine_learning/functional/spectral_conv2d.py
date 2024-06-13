import torch
from typing import Annotated
from beartype import beartype
from beartype.vale import Is
from UQpy.utilities.ValidationTypes import PositiveInteger


tensor4d = Annotated[torch.Tensor, Is[lambda x: x.ndim == 4]]


@beartype
def spectral_conv2d(
    x: tensor4d,
    weights: tuple[tensor4d, tensor4d],
    out_channels: PositiveInteger,
    modes1: PositiveInteger,
    modes2: PositiveInteger,
) -> torch.Tensor:
    """Compute the 2d spectral convolution :math:`\mathcal{F}^{-1}(R (\mathcal{F}x) )`

    :param x: Tensor of shape :math:`(N, C_\text{in}, H, W)`
    :param weights: Tuple of two tensors each with shape :math:`(C_\\text{in}, C_\\text{out}, \\text{modes1}, \\text{modes2})`
    :param out_channels: :math:`C_\text{out}`, Number of channels in the output signal
    :param modes1: Number of Fourier modes to keep, at most :math:`\lfloor H / 2 \rfloor + 1`
    :param modes2: Number of Fourier modes to keep, at most :math:`\lfloor W / 2 \rfloor + 1`
    :return: Tensor :math:`\mathcal{F}^{-1}(R (\mathcal{F}x) )` of shape :math:`(N, C_\text{out}, H, W)`
    """
    batch_size, in_channels, height, width = x.shape
    # Fourier transform
    x_ft = torch.fft.rfft(x, 2, normalized=True, onesided=True)
    # Linear transform in Fourier space
    out_shape = (
        batch_size,
        out_channels,
        height,
        (width // 2) + 1,
    )
    out_ft = torch.zeros(out_shape, dtype=torch.cfloat)
    indices = [
        (slice(None), slice(None), slice(0, modes1), slice(0, modes2)),
        (slice(None), slice(None), slice(-modes1, None), slice(0, modes2)),
    ]
    equation = "bihw,iohw->bohw"  # "bixy,ioxy->boxy"
    for i, w in zip(indices, weights):
        out_ft[i] = torch.einsum(equation, x_ft[i], w)
    # Return to physical space
    x = torch.fft.irfft(
        out_ft, 2, normalize=True, onsided=True, signal_sizes=(height, width)
    )
    return x
