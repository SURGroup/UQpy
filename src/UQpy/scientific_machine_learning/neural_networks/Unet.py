import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from UQpy.scientific_machine_learning.baseclass.NeuralNetwork import NeuralNetwork
from UQpy.utilities.ValidationTypes import PositiveInteger


class Unet(NeuralNetwork):

    def __init__(
        self,
        n_filters: list[PositiveInteger],
        kernel_size: PositiveInteger,
        out_channels: PositiveInteger,
        layer_type: nn.Module = nn.Conv2d,
    ):
        r"""Construct U-net convolutional neural network for mean-field prediction

        :param n_filters: A list of positive integers specifying the number of filters at for each convolutional layer in the encoding and decoding paths.
         The length of the list determines the depth of the U-Net.
        :param kernel_size: The size of the convolutional kernels. This value is used for all convolutional layers.
         Standard kernel size options are 3, 6, or 9.
        :param out_channels: The number of output channels in the final convolutional layer.
        :param layer_type: The type of convolutional layer to use. The default is the ``nn.Conv2d``.
         It can be replaced with Bayesian layers for performing uncertainty quantification.

        .. note::
            A default value ``stride=2`` is used for the max pooling layers.
            A default padding value of ``kernel_size // 2`` is used for all convolutional layers.

        Shape:

        - Input: Tensor of shape :math:`(N, C_\text{in}, H, W)`
        - Output: Tensor of shape :math:`(N, C_\text{out}, H, W)`

        Attributes:

        Encoder Layers: The encoding blocks are created during initialization from the ``n_filters`` list
        for indices :math:`i=1, \dots, \text{len}(\texttt{n_filters})- 1`.

        - **encoder_maxpool_i** (:py:class:`torch.nn.MaxPool2d`): Max pooling layer for downsampling at encoder layer ``i`` (for ``i > 1``).
        - **encoder_conv_1_i** (:py:class:`torch.nn.Conv2d`): First convolutional layer at encoder layer ``i``.
        - **encoder_bn_1_i** (:py:class:`torch.nn.BatchNorm2d`): Batch normalization layer after ``encoder_conv_1_i``.
        - **encoder_conv_2_i** (:py:class:`torch.nn.Conv2d`): Second convolutional layer at encoder layer ``i``.
        - **encoder_bn_2_i** (:py:class:`torch.nn.BatchNorm2d`): Batch normalization layer after ``encoder_conv_2_i``.

        Decoder Layers:

        - **decoder_upsample_i** (:py:class:`torch.nn.Upsample`): Upsampling layer at decoder layer ``i``.
        - **decoder_conv_1_i** (:py:class:`torch.nn.Conv2d`): First convolutional at decoder layer ``i``.
        - **decoder_bn_1_i** (:py:class:`torch.nn.BatchNorm2d`): Batch normalization layer after ``decoder_conv_1_i``.
        - **decoder_conv_2_i** (:py:class:`torch.nn.Conv2d`): Second convolutional layer at decoder layer ``i``.
        - **decoder_bn_2_i** (:py:class:`torch.nn.BatchNorm2d`): Batch normalization layer after ``decoder_conv_2_i``.

        Final Convolution Layer:
        - **final_conv** (:py:class:'torch.nn.Conv2d'): Convolutional layer applied after the last decoder block. It maps the output to the desired number of channels.

        """
        super(Unet, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.layer_type = layer_type

        self.logger = logging.getLogger(__name__)

        for i in range(1, len(self.n_filters)):  # initialize encoding blocks
            in_channels = self.n_filters[i - 1]
            out_channels = self.n_filters[i]

            if i != 1:
                setattr(
                    self, f"encoder_maxpool_{i}", nn.MaxPool2d(kernel_size=2, stride=2)
                )

            setattr(
                self,
                f"encoder_conv_1_{i}",
                self.layer_type(
                    in_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                ),
            )
            setattr(self, f"encoder_bn_1_{i}", nn.BatchNorm2d(out_channels))
            setattr(
                self,
                f"encoder_conv_2_{i}",
                self.layer_type(
                    out_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                ),
            )
            setattr(self, f"encoder_bn_2_{i}", nn.BatchNorm2d(out_channels))

        for i in range(len(self.n_filters) - 1, 1, -1):  # initialize decoding blocks
            combined_channels = self.n_filters[i] + self.n_filters[i - 1]
            out_channels = self.n_filters[i - 1]
            setattr(
                self,
                f"decoder_upsample_{i}",
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            )
            setattr(
                self,
                f"decoder_conv_1_{i}",
                self.layer_type(
                    combined_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                ),
            )
            setattr(self, f"decoder_bn_1_{i}", nn.BatchNorm2d(out_channels))
            setattr(
                self,
                f"decoder_conv_2_{i}",
                self.layer_type(
                    out_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                ),
            )
            setattr(self, f"decoder_bn_2_{i}", nn.BatchNorm2d(out_channels))

        self.final_conv = nn.Conv2d(
            self.n_filters[1], self.out_channels, kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the U-Net model.

        The output is computed by passing the input through each encoding and decoding block together with the skip connections.

        :param x: Tensor of shape :math:`(N, C_\text{in}, H, W)`
        :return: Tensor of shape :math:`(N, C_\text{out}, H, W)`
        """
        encoder_outputs = []
        # Pass through encoder layers
        for i in range(1, len(self.n_filters)):
            x = self.optional_step_en(x, i)
            if i != 1:
                x = getattr(self, f"encoder_maxpool_{i}")(x)
            x = getattr(self, f"encoder_conv_1_{i}")(x)
            x = getattr(self, f"encoder_bn_1_{i}")(x)
            x = F.relu(x)
            x = getattr(self, f"encoder_conv_2_{i}")(x)
            x = getattr(self, f"encoder_bn_2_{i}")(x)
            x = F.relu(x)
            encoder_outputs.append(x)

        # Pass through decoder layers
        for idx, i in enumerate(range(len(self.n_filters) - 1, 1, -1)):
            x = getattr(self, f"decoder_upsample_{i}")(x)
            # get skip connection and concatenate to output
            skip_input = encoder_outputs[-(idx + 2)]
            x = torch.cat([x, skip_input], dim=1)

            # option dropout
            x = self.optional_step_dec(x, i)
            x = getattr(self, f"decoder_conv_1_{i}")(x)
            x = getattr(self, f"decoder_bn_1_{i}")(x)
            x = F.relu(x)
            x = getattr(self, f"decoder_conv_2_{i}")(x)
            x = getattr(self, f"decoder_bn_2_{i}")(x)
            x = F.relu(x)

        x = self.final_conv(x)
        return x

    def optional_step_en(self, x: torch.Tensor, i: int):
        """Optional method for additional operators during encoding

        Intended to be overridden by subclasses to apply operations like Monte Carlo Dropout based on the layer index i.

        :param x: Input tensor
        :param i: Index of the encoding block
        :return: Output tensor
        """
        return x

    def optional_step_dec(self, x, i):
        """Optional method for additional operations during decoding

        Intended to be overridden by subclasses to apply operations like Monte Carlo Dropout based on the layer index i.

        :param x: Input tensor
        :param i: Index of the decoding block
        :return: Output tensor
        """
        return x
