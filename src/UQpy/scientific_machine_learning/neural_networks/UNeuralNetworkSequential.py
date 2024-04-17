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
        filter_sizes: Union[list[PositiveInteger]],
        kernel_size: PositiveInteger,
        out_channels: PositiveInteger,
        layer_type: nn.Module = nn.Conv2d,
        **kwargs,
    ):
        """Initialize a U-Shaped Neural Network

        :param filter_sizes: Size of filters in order. # ToDo: Should these be increasing multiples of 2?
        :param kernel_size: Size of the kernel used in ``layer_type``
        :param out_channels: Number of channels in the final convolution
        :param layer_type: Type of layer to use on each filter. Default ``torch.nn.Conv2d``
        """
        super().__init__(**kwargs)
        self.filter_sizes = filter_sizes
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.layer_type = layer_type

        self.n_filters = len(self.filter_sizes)
        self.logger = logging.getLogger(__name__)

        for i in range(1, self.n_filters):  # Encoding block initialization
            setattr(self, f"encoder_{i}", self._construct_encoder(i))
        for i in range(self.n_filters - 1, 1, -1):  # Decoding block initialization
            setattr(self, f"decoder_{i}", self._construct_decoder(i))
        self.final_layer = nn.Conv2d(
            self.filter_sizes[1], self.out_channels, kernel_size=1, padding=0
        )
        self.decoder_upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

    def _construct_encoder(self, i: PositiveInteger) -> nn.Module:
        """Construct the ``i``-th encoder

        :param i: Index of the ``i``-th encoder
        :return: Trainable module defining the encoder
        """
        in_channels = self.filter_sizes[i - 1] if i > 0 else 1
        out_channels = self.filter_sizes[i]
        if i == 1:
            layers = []
        else:  # i != 1
            layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
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

    def _construct_decoder(self, i: PositiveInteger) -> nn.Module:
        """Construct the ``i``-th decoder

        :param i: Index of the ``i``-th decoder
        :return: Trainable module defining the decoder
        """
        combined_channels = self.filter_sizes[i] + self.filter_sizes[i - 1]
        out_channels = self.filter_sizes[i - 1] if i > 0 else 1
        layers = [
            # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
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

    def encode(
        self,
        x: torch.Tensor,
        i: Annotated[PositiveInteger, Is[lambda n: n < self.n_filters]],
        # i: Annotated[PositiveInteger, Is[lambda n: hasattr(self, f"encoder_{i}"]],
    ) -> torch.Tensor:
        """

        :param x:
        :param i:
        :return:
        """
        return getattr(self, f"encoder_{i}")(x)

    def decode(self, x: torch.Tensor, i: PositiveInteger) -> torch.Tensor:
        """

        :param x:
        :param i:
        :return:
        """
        return getattr(self, f"decoder_{i}")(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation call of UNeuralNetwork

        :param x: Input tensor
        :return: Output tensor
        """
        encoder_outputs = []
        for i in range(1, self.n_filters):  # Pass through encoders
            x = self.encode(x, i)
            encoder_outputs.append(x)

        for idx, i in enumerate(range(self.n_filters - 1, 1, -1)):  # Pass through decoders
            x = self.decoder_upsample(x)
            skip_input = encoder_outputs[-(idx + 2)]
            x = torch.cat([x, skip_input], dim=1)
            x = self.decode(x, i)

        x = self.final_layer(x)
        return x
