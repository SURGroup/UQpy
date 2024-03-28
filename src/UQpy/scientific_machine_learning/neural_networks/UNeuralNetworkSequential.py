import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from UQpy.scientific_machine_learning.baseclass.NeuralNetwork import NeuralNetwork
from UQpy.utilities.ValidationTypes import PositiveInteger


class UNeuralNetworkSequential(NeuralNetwork):

    def __init__(
        self,
        filter_sizes: list[PositiveInteger],
        kernel_size: PositiveInteger,
        out_channels: PositiveInteger,
        layer_type: nn.Module = nn.Conv2d,
        additional_encode: nn.Module = None,
        additional_decode: nn.Module = None,
        **kwargs,
    ):
        """Initialize a U-Shaped Neural Network

        :param filter_sizes: Size of filters in order. # ToDo: Should these be increasing multiples of 2?
        :param kernel_size:
        :param out_channels: Number of channels in the final convolution
        :param layer_type: Type of layer to use on each filter. Default ``torch.nn.Conv2d``
        :param additional_encode: Optional additional layer to apply during each encoding. Default ``None``.
         Note this should be an instance of a ``torch.nn.Module`` that accepts two inputs (x, i)
         where x is a tensor and i is the encoding layer index.
        :param additional_decode: Optional additional layer to apply during each decoding. Default ``None``.
         Note this should be an instance of a ``torch.nn.Module`` that accepts two inputs (x, i)
          where x is a tensor and i is the decoding layer index.
        """
        super().__init__(**kwargs)
        self.filter_sizes = filter_sizes
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.layer_type = layer_type
        self.additional_encode = additional_encode
        self.additional_decode = additional_decode

        self.n_filters = len(self.filter_sizes)
        self.logger = logging.getLogger(__name__)

        for i in range(1, self.n_filters):  # Encoding block initialization
            encoder_layers = []
            in_channels = self.filter_sizes[i - 1] if i > 0 else 1
            out_channels = self.filter_sizes[i]
            if self.additional_encode:
                encoder_layers.append(self.additional_encode)
            if i != 1:
                encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            encoder_layers.append(
                self.layer_type(
                    in_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                )
            )
            encoder_layers.append(nn.BatchNorm2d(out_channels))
            encoder_layers.append(
                self.layer_type(
                    out_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                )
            )
            encoder_layers.append(nn.BatchNorm2d(out_channels))
            setattr(self, f"encoder_{i}", nn.Sequential(*encoder_layers))

        for i in range(self.n_filters - 1, 1, -1):  # Decoding block initialization
            decoder_layers = []
            combined_channels = self.filter_sizes[i] + self.filter_sizes[i - 1]
            out_channels = self.filter_sizes[i - 1] if i > 0 else 1
            decoder_layers.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # ToDo: Is this constant? Can this be written into the forward method?
            )
            # decoder_layers.append(skip)  # ToDo: how does this skip function work
            if self.additional_decode:
                decoder_layers.append(self.additional_decode)
            decoder_layers.append(
                self.layer_type(
                    combined_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                )
            )
            decoder_layers.append(nn.BatchNorm2d(out_channels)),
            decoder_layers.append(nn.ReLU()),
            decoder_layers.append(
                self.layer_type(
                    out_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                )
            )
            decoder_layers.append(nn.BatchNorm2d(out_channels))
            decoder_layers.append(nn.ReLU())
            setattr(self, f"decoder_{i}", nn.Sequential(*decoder_layers))
        self.final_layer = nn.Conv2d(self.filter_sizes[1], self.out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation call of UNO

        :param x: Input tensor
        :return: Output tensor
        """
        encoder_outputs = []
        for i in range(1, self.n_filters):  # Pass through encoders
            x = getattr(self, f"encoder_{i}")(x)
            encoder_outputs.append(x)

        for i in range(1, self.n_filters):
            # ToDo: does upsampling have to happen before skip? Should upsample be removed from the sequential? Are there any learnable parameters in upsample?
            # up_sample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)(x)
            # x = up_sample(x)
            x = getattr(self, f"decoder_{i}")(x)

        # Pass through decoder layers
        # for idx, i in enumerate(range(self.n_filters - 1, 1, -1)):
        # x = getattr(self, f"decoder_upsample_{i}")(x)
        # skip_input = encoder_outputs[-(idx + 2)]  # Get skip connection
        # x = torch.cat([x, skip_input], dim=1)
        # x = self.optional_step_dec(x, i)
        # x = getattr(self, f"decoder_conv_1_{i}")(x)
        # x = getattr(self, f"decoder_bn_1_{i}")(x)
        # x = F.relu(x)
        # x = getattr(self, f"decoder_conv_2_{i}")(x)
        # x = getattr(self, f"decoder_bn_2_{i}")(x)
        # x = F.relu(x)

        x = self.final_conv(x)
        return x

    @property
    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    @property
    def loss_function(self):
        return nn.MSELoss(reduction="mean")

    def learn(self, data_loader: torch.utils.data.DataLoader, epochs: int = 100):
        """

        :param data_loader:
        :param epochs:
        """
        self.logger.info(
            "UQpy: Scientific Machine Learning: Beginning training UNeuralNetwork"
        )
        raise NotImplementedError
