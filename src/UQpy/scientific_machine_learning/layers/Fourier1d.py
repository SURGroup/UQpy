import torch.nn as nn
import UQpy.scientific_machine_learning as sml
from UQpy.scientific_machine_learning.baseclass import FourierLayer


class Fourier1d(FourierLayer):
    @property
    def spectral_conv(self):
        return sml.SpectralConv1d(self.width, self.width, self.modes)

    @property
    def conv(self):
        return nn.Conv1d(self.width, self.width, 1)
