import torch
from typing import Annotated
from beartype import beartype
from beartype.vale import Is
from UQpy.utilities.ValidationTypes import PositiveInteger


@beartype
def spectral_conv2d(
    x: Annotated[torch.Tensor, Is[lambda x: x.ndim == 4]],
    weights1: Annotated[torch.Tensor, Is[lambda x: x.ndim == 4]],
    weights2: Annotated[torch.Tensor, Is[lambda x: x.ndim == 4]],
    out_channels: PositiveInteger,
    modes1: PositiveInteger,
    modes2: PositiveInteger,
):
    """ FixMe: how does this einsum work? upper and lower?

    :param x:
    :param weights:
    :param out_channels:
    :param modes1:
    :param modes2:
    :return:
    """
    batch_size, in_channels, height, width = x.shape
    # Fourier transform
    x_ft = torch.fft.rfft(x, 2, normalized=True, onesided=True)
    # Linear transform in Fourier space
    # out_shape = (
    #     batch_size,
    #     out_channels,
    #     height,
    #     (width // 2) + 1,
    #     2,
    # )  # FixMe: why is this 5d
    # out_ft = torch.zeros(out_shape, dtype=torch.cfloat)
    # # indices = [
    # i = (slice(None), slice(None), slice(0, modes1), slice(0, modes2))
    # j = (slice(None), slice(None), slice(-modes1, None), slice(0, modes2))
    # # ]
    # equation = "bixy,ioxy->boxy"
    #
    # out_ft[i] =
    # for i in indices:
    #     x_slice = x[i]
    #
    #     # out_ft[i] = torch.einsum(equation, x_ft[i], weights)
    #     upper_0 = torch.einsum(equation, x[..., 0], weights[..., 0])
    #     upper_1 = torch.einsum(equation, x[..., 1], weights[..., 1])
    #     upper = upper_0 - upper_1

    # Return to physical space
    x = torch.fft.irfft(
        out_ft, 2, normalize=True, onsided=True, signal_sizes=(height, width)
    )
    return x
