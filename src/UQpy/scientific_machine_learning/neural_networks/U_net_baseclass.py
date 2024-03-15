import torch
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d, Module, BatchNorm2d, Upsample
import sys

class UNet(Module):
    def __init__(self, nfilters, kernel_size, layer_type=Conv2d):
        # Initialize the torch.nn.Module superclass
        super(UNet, self).__init__()
        self.nfilters = nfilters
        self.kernel_size = kernel_size
        self.layer_type = layer_type
        self.encoder_maxpool = []
        self.encoder_convn_1 = []
        self.encoder_bn_1 = []
        self.encoder_convn_2 = []
        self.encoder_bn_2 = []
        self.decoder_upsample = []
        self.decoder_convolution_1 = []
        self.decoder_bn_1 = []
        self.decoder_convolution_2 = []
        self.decoder_bn_2 = []
        self.final_convolution = []

        # Encoding block initialization
        for i in range(1, len(self.nfilters)):
            in_channels = self.nfilters[i - 1] if i > 0 else 1
            out_channels = self.nfilters[i]

            if i != 1:
                setattr(self, f"encoder_maxpool_{i}", MaxPool2d(
                    kernel_size=2, stride=2)) 

            setattr(self, f"encoder_conv_1_{i}", self.layer_type(in_channels, out_channels,
                                                                 kernel_size=self.kernel_size,
                                                                 padding=self.kernel_size // 2))
            setattr(self, f"encoder_bn_1_{i}", BatchNorm2d(out_channels))
            setattr(self, f"encoder_conv_2_{i}", self.layer_type(
                out_channels, out_channels, kernel_size=self.kernel_size, padding=self.kernel_size // 2))
            setattr(self, f"encoder_bn_2_{i}", BatchNorm2d(out_channels))

        # Decoding block initialization
        for i in range(len(self.nfilters) - 1, 1, -1):
            combined_channels = self.nfilters[i] + self.nfilters[i - 1]
            out_channels = self.nfilters[i - 1] if i > 0 else 1
            setattr(self, f"decoder_upsample_{i}", Upsample(
                scale_factor=2, mode='bilinear', align_corners=True))
            setattr(self, f"decoder_conv_1_{i}", self.layer_type(
                combined_channels, out_channels, kernel_size=self.kernel_size, padding=self.kernel_size // 2))
            setattr(self, f"decoder_bn_1_{i}", BatchNorm2d(out_channels))
            self.decoder_bn_2.append(BatchNorm2d(out_channels))
            setattr(self, f"decoder_conv_2_{i}", self.layer_type(
                out_channels, out_channels, kernel_size=self.kernel_size, padding=self.kernel_size // 2))
            setattr(self, f"decoder_bn_2_{i}", BatchNorm2d(out_channels))

        self.final_conv = Conv2d(self.nfilters[1], 1, kernel_size=1, padding=0)
        
    def forward(self, x):
        encoder_outputs = []
        # Pass through encoder layers
        for i in range(1, len(self.nfilters)):
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
        for idx, i in enumerate(range(len(self.nfilters) - 1, 1, -1)):
            x = getattr(self, f"decoder_upsample_{i}")(x)
            # Get skip connection
            skip_input = encoder_outputs[-(idx+2)]
            
            before_cat_size = x.size()
            # print(f"Before concatenation size: {before_cat_size}")

            x = torch.cat([x, skip_input], dim=1)
            after_cat_size = x.size()
            # print(f"After concatenation size: {after_cat_size}")
            
            x = self.optional_step_dec(x, i)
            x = getattr(self, f"decoder_conv_1_{i}")(x)
            x = getattr(self, f"decoder_bn_1_{i}")(x)
            x = F.relu(x)
            x = getattr(self, f"decoder_conv_2_{i}")(x)
            x = getattr(self, f"decoder_bn_2_{i}")(x)
            x = F.relu(x)

        x = self.final_conv(x)
        return x

    def optional_step_en(self, x, i):
        return x

    def optional_step_dec(self, x, i):
        return x
