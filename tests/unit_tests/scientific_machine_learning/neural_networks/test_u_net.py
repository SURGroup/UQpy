import torch
import UQpy.scientific_machine_learning as sml


def test_output_shape():
    n_filters = [1, 64, 128]
    kernel_size = 3
    out_channels = 3
    unet = sml.Unet(n_filters, kernel_size, out_channels)

    x = torch.rand(1, 1, 512, 512)
    y = unet(x)

    assert y.shape == torch.Size((1, out_channels, 512, 512))