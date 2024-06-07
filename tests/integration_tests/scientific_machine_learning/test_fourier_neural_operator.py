import torch
import torch.nn as nn
import UQpy.scientific_machine_learning as sml

width = 10
modes = 4
spectral_network = nn.Sequential(
    sml.FourierBlock1d(width, modes),
    sml.FourierBlock1d(width, modes),
    sml.FourierBlock1d(width, modes),
    sml.FourierBlock1d(width, modes, activation=None),
)

fno = sml.FourierNeuralOperator(
    spectral_network,
    upscale_network=nn.Linear(2, width),
    downscale_network=nn.Linear(width, 2),
)
