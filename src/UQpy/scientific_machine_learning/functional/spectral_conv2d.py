import torch
from typing import Annotated
from beartype import beartype
from beartype.vale import Is
from UQpy.utilities.ValidationTypes import PositiveInteger, Torch4DComplexTensor


@beartype
def spectral_conv2d(
    x: Annotated[torch.Tensor, Is[lambda tensor: tensor.ndim == 4]],
    weights: tuple[Torch4DComplexTensor, Torch4DComplexTensor],
    out_channels: PositiveInteger,
    modes: tuple[PositiveInteger, PositiveInteger],
) -> torch.Tensor:
    r"""Compute the 2d spectral convolution :math:`\mathcal{F}^{-1}(R (\mathcal{F}x) )`

    :param x: Tensor of shape :math:`(N, C_\text{in}, H, W)`
    :param weights: Tuple of two tensors each with shape :math:`(C_\text{in}, C_\text{out}, \text{modes[0]}, \text{modes[1]})`.
     Both weight tensors must have complex entries.
    :param out_channels: :math:`C_\text{out}`, Number of channels in the output signal
    :param modes: Tuple of Fourier modes to keep.
     At most :math:`(\lfloor H / 2 \rfloor + 1, \lfloor W / 2 \rfloor + 1)`
    :return: Tensor :math:`\mathcal{F}^{-1}(R (\mathcal{F}x) )` of shape :math:`(N, C_\text{out}, H, W)`
    """
    batch_size, in_channels, height, width = x.shape
    if modes[0] > (height // 2) + 1:
        raise ValueError(
            "UQpy: Invalid `modes[0]`. `modes[0]` must be less than or equal to (height // 2) + 1 "
        )
    if modes[1] > (width // 2) + 1:
        raise ValueError(
            "UQpy: Invalid `modes[1]`. `modes[1]` must be less than or equal to (width // 2) + 1"
        )
    correct_shape = torch.Size([in_channels, out_channels, modes[0], modes[1]])
    if weights[0].shape != correct_shape:
        raise AssertionError(
            "UQpy: Invalid shape for `weights[0]`. "
            "`weights[0]` must be of shape (in_channels, out_channels, modes[0], modes[1])"
        )
    if weights[1].shape != correct_shape:
        raise AssertionError(
            "UQpy: Invalid shape for `weights[1]`."
            "`weights[1] must be of shape (in_channels, out_channels, modes[0], modes[1])"
        )
    # Fourier transform
    # x_ft = torch.fft.rfft(x, 2, normalized=True, onesided=True)
    x_ft = torch.fft.rfft2(x, s=(height, width))
    # Linear transform in Fourier space
    out_shape = (
        batch_size,
        out_channels,
        (height // 2) + 1,
        (width // 2) + 1,
    )
    out_ft = torch.zeros(out_shape, dtype=torch.cfloat)
    indices = [
        (slice(None), slice(None), slice(0, modes[0]), slice(0, modes[1])),
        (slice(None), slice(None), slice(-modes[0], None), slice(0, modes[1])),
    ]
    equation = "bixy,ioxy->boxy"
    for i, w in zip(indices, weights):
        out_ft[i] = torch.einsum(equation, x_ft[i], w)
    # Return to physical space
    # x = torch.fft.irfft(
    #     out_ft, 2, normalize=True, onsided=True, signal_sizes=(height, width)
    # )
    x = torch.fft.irfft2(out_ft, s=(height, width))
    return x
