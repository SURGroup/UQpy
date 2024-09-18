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
        """
        Construct U-net convolutional neural network for mean-field prediction

        :param n_filters : A list of positive integers specifying the number of filters at for each convolutional layer in the encoding and decoding paths
        .. note:: The length of the list determines the depth of the U-Net. 
        :param kernel_size : The size of the convolutional kernels. This value is used for all convolutional layers. Standard kernel size options are :math(3, 6) or :math(9).
        out_channels : The number of output channels in the final convolutional layer. 
        layer_type : The type of convolutional layer to use. The default is the ``nn.Conv2d``. It can be replaced with Bayesian layers for performing uncertainty quantification.

        Shape: 
        - Input:Tensor of shape :math:(N, N_C, N_H, N_W) where:
        :math:N is the batch size or number of samples.
        :math:N_C is the number of input channels (e.g., RGB images, stress field maps etc).
        :math:N_H number of pixels in the height of the input image.
        :math:N_W number of pixels in the width of the input image.
        
        Attributes:
        The following encoding and decoding blocks are dynamically created during initialization based on the `n_filters` list up to :math:`i=n_\text{filters} - 1.

    **Encoder Layers**:

    - **encoder_maxpool_i** : nn.MaxPool2d
        Max pooling layer for downsampling at encoder layer ``i`` (for ``i > 1``).
    - **encoder_conv_1_i** : nn.Module
        First convolutional layer at encoder layer ``i``.
    - **encoder_bn_1_i** : nn.BatchNorm2d
        Batch normalization layer after `encoder_conv_1_i`.
    - **encoder_conv_2_i** : nn.Module
        Second convolutional layer at encoder layer ``i``.
    - **encoder_bn_2_i** : nn.BatchNorm2d
        Batch normalization layer after `encoder_conv_2_i`.

    **Decoder Layers**:

    - **decoder_upsample_i** : nn.Upsample
        Upsampling layer at decoder layer ``i``.
    - **decoder_conv_1_i** : nn.Module
        First convolutional layer at decoder layer ``i``.
    - **decoder_bn_1_i** : nn.BatchNorm2d
        Batch normalization layer after `decoder_conv_1_i`.
    - **decoder_conv_2_i** : nn.Module
        Second convolutional layer at decoder layer ``i``.
    - **decoder_bn_2_i** : nn.BatchNorm2d
        Batch normalization layer after `decoder_conv_2_i`.

    **Final Convolution Layer**:
    - **final_conv**: nn.Conv2d applied after the last decoder block. It maps the output to the desired number of channels.
    
    Methods
    -------
    - forward(x)
        Performs the forward pass through the U-Net model.

    - optional_step_en(x, i)

    - optional_step_dec(x, i)

    Optional methods for additional operations at the encoding and decoding path, respecitvely.
    These intended to be overridden by subclasses to apply operations like Monte Carlo Dropout(MCD) based on a boolean index i. The boolean index i has the same length as the number of filters. These should be specified in the sublass.

    .. note::
        By default, this method returns the input tensor unchanged.

    """

        super(Unet, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.layer_type = layer_type

        self.logger = logging.getLogger(__name__)

        """Encoding Block Initialization"""
        
        for i in range(1, len(self.n_filters)):
            in_channels = self.n_filters[i - 1] if i > 0 else 1
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

        """Decoding Block Initialization"""

        for i in range(len(self.n_filters) - 1, 1, -1):
            combined_channels = self.n_filters[i] + self.n_filters[i - 1]
            out_channels = self.n_filters[i - 1] if i > 0 else 1
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
            # self.decoder_bn_2.append(nn.BatchNorm2d(out_channels))
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

        """Final Convolution Layer"""

        self.final_conv = nn.Conv2d(
            self.n_filters[1], self.out_channels, kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the U-Net model. The networks is assembled by passing the input tensor through the operations of each encoding and decoding block together with the skip connections.

        :param x: Input tensor
        :return: Output tensor
        """
        encoder_outputs = []
        # Pass through encoder layers
        for i in range(1, len(self.n_filters)):
            """"Optional dropout"""
            x = self.optional_step_en(x, i)
            if i != 1:
                x = getattr(self, f"encoder_maxpool_{i}")(x)
            x = getattr(self, f"encoder_conv_1_{i}")(x)
            x = getattr(self, f"encoder_bn_1_{i}")(x)
            x = F.relu(
                x
            )
            x = getattr(self, f"encoder_conv_2_{i}")(x)
            x = getattr(self, f"encoder_bn_2_{i}")(x)
            x = F.relu(x)
            encoder_outputs.append(x)

        # Pass through decoder layers
        for idx, i in enumerate(range(len(self.n_filters) - 1, 1, -1)):
            x = getattr(self, f"decoder_upsample_{i}")(x)
            # Get skip connection
            skip_input = encoder_outputs[-(idx + 2)]

            before_cat_size = x.size()
            # print(f"Before concatenation size: {before_cat_size}")

            x = torch.cat([x, skip_input], dim=1)
            after_cat_size = x.size()
            # print(f"After concatenation size: {after_cat_size}")
            """"Optional dropout"""
            x = self.optional_step_dec(x, i)
            x = getattr(self, f"decoder_conv_1_{i}")(x)
            x = getattr(self, f"decoder_bn_1_{i}")(x)
            x = F.relu(x)
            x = getattr(self, f"decoder_conv_2_{i}")(x)
            x = getattr(self, f"decoder_bn_2_{i}")(x)
            x = F.relu(x)

        x = self.final_conv(x)
        return x
    
    """Optional methods for droout during the encoder and decoder stages"""
    
    def optional_step_en(self, x, i):

        """
        :param x:
        :param i:
        :return:
        """
        return x

    def optional_step_dec(self, x, i):
        """ Optional method for additional operations during the decoding stage.

        :param x:
        :param i:
        :return:
        """
        return x
