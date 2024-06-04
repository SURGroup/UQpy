import torch.nn as nn
import UQpy.scientific_machine_learning as sml
from UQpy.scientific_machine_learning.baseclass import FourierLayer


class Fourier3d(FourierLayer):

    @property
    def spectral_conv(self):
        return None

    @property
    def conv(self):
        return None
