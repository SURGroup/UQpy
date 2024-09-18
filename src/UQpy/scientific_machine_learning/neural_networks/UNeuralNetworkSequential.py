import torch
import torch.nn as nn
import logging
from beartype import beartype
from beartype.vale import Is
from typing import Annotated, Union
from UQpy.scientific_machine_learning.baseclass.NeuralNetwork import NeuralNetwork
from UQpy.utilities.ValidationTypes import PositiveInteger


@beartype
class UNeuralNetworkSequential(NeuralNetwork):

    def __init__(
        self,
        filter_sizes: tuple,
        kernel_size: PositiveInteger,
        out_channels: PositiveInteger,
        layer_type: nn.Module = nn.Conv2d,
    ):
        r"""Initialize a U-Shaped Neural Network for a 2d signal

        :param filter_sizes: Size of filters in order.
        :param kernel_size: Size of the kernel used in ``layer_type``
        :param out_channels: Number of channels in the final convolution
        :param layer_type: Type of layer to use on each filter. Default :py:class:`torch.nn.Conv2d`

        Shape:

        - Input:
        - Output

        Attributes:

        - **encoder_i** (:py:class:`torch.nn.Module`): The "i"-th learnable encoder block.
          Note that :code:`encoder_i` is *not* an attribute of this class. There are encoder attributes with names
          :code:`encoder_1`, :code:`encoder_2`, ..., up to :math:`i=n_\text{filters} - 1`
        - **decoder_i** (:py:class:`torch.nn.Module`): The "i"-th learnable decoder block.
          Note that :code:`decoder_i` is *not* an attribute of this class. There are encoder attributes with names
          :code:`decoder_1`, :code:`decoder_2`, ..., up to :math:`i=n_\text{filters} - 1`
        - **decoder_upsample** (:py:class:`torch.nn.Module`):
        - **final_layer**  (:py:class:`torch.nn.Module`): Learnable :py:class:`torch.nn.Conv2d` applied after the last decoder block.
        """
        super().__init__()
        self.filter_sizes = filter_sizes
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.layer_type = layer_type

        self.n_filters = len(self.filter_sizes)
        self.logger = logging.getLogger(__name__)

        for i in range(1, self.n_filters):  # Encoding block initialization
            setattr(self, f"encoder_{i}", self.construct_encoder(i))
        for i in range(self.n_filters - 1, 1, -1):  # Decoding block initialization
            setattr(self, f"decoder_{i}", self.construct_decoder(i))
        self.final_layer = nn.Conv2d(
            self.filter_sizes[1], self.out_channels, kernel_size=1, padding=0
        )
        self.decoder_upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

    def construct_encoder(self, i: PositiveInteger) -> nn.Module:
        """Construct the ``i``-th encoder

        :param i: Index of the ``i``-th encoder
        :return: Trainable module defining the encoder
        """
        in_channels = 1 if i == 0 else self.filter_sizes[i - 1]
        out_channels = self.filter_sizes[i]
        layers = []
        if i != 1:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers += [
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.layer_type(
                in_channels,
                out_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.BatchNorm2d(out_channels),
            self.layer_type(
                out_channels,
                out_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        return nn.Sequential(*layers)

    def construct_decoder(self, i: PositiveInteger) -> nn.Module:
        """Construct the ``i``-th decoder

        :param i: Index of the ``i``-th decoder
        :return: Trainable module defining the decoder
        """
        combined_channels = self.filter_sizes[i] + self.filter_sizes[i - 1]
        # out_channels = self.filter_sizes[i - 1] if i > 0 else 1
        out_channels = self.filter_sizes[i - 1]
        layers = [
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            self.layer_type(
                combined_channels,
                out_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.layer_type(
                out_channels,
                out_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor, i: PositiveInteger) -> torch.Tensor:
        """Pass the tensor ``x`` through the ``i``-th decoder

        :param x: Tensor of shape compatible with :code:`encoder_i`
        :param i: Index of the encoder
        :return: Tensor of shape output by :code:`encoder_i`
        """
        if not hasattr(self, f"encoder_{i}"):
            raise AttributeError(
                f"UQpy: Invalid encoder index i={i}. "
                f"The index must be 1<= {i} < {self.n_filters},"
            )
        return getattr(self, f"encoder_{i}")(x)

    def decode(self, x: torch.Tensor, i: PositiveInteger) -> torch.Tensor:
        """Pass the tensor ``x`` through the ``i``-th decoder

        :param x: Tensor of shape compatible with :code:`decoder_i`
        :param i: Index of the decoder
        :return: Tensor of shape output by :code:`decoder_i`
        """
        if not hasattr(self, f"decoder_{i}"):
            raise AttributeError(
                f"UQpy: Invalid decoder index i={i}. "
                f"The index must be 1<= {i} < {self.n_filters},"
            )
        return getattr(self, f"decoder_{i}")(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass ``x`` through encoders and save each output. Then pass through decoders with skip connections.

        :param x: Tensor of shape # ToDo what shape is this?
        :return: Tensor of shape # ToDo what shape is this?
        """
        encoder_outputs = []
        for i in range(1, self.n_filters):  # Pass through encoders
            x = self.encode(x, i)
            encoder_outputs.append(x)

        for idx, i in enumerate(
            range(self.n_filters - 1, 1, -1)
        ):  # Pass through decoders
            x = self.decoder_upsample(x)
            skip_input = encoder_outputs[-(idx + 2)]
            x = torch.cat([x, skip_input], dim=1)
            x = self.decode(x, i)

        x = self.final_layer(x)
        return x


if __name__ == "__main__":
    n_filters = (1, 64, 128)
    kernel_size = 3
    out_channels = 3
    unet = UNeuralNetworkSequential(n_filters, kernel_size, out_channels)
    x = torch.rand(1, 1, 512, 512)
    print(unet)
    y = unet(x)
    print()
    print(x.shape)
    print(y.shape)
    # for m in unet.modules():
    #     print(m)
