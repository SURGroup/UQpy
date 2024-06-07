import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import UQpy.scientific_machine_learning as sml
from dataset import load_data

import logging

logger = logging.getLogger("UQpy")
logger.setLevel(logging.INFO)
# if logger.hasHandlers():
#     logger.removeHandler(
#         logger.handlers[0]
#     )  # remove existing handlers to eliminate print statements
# file_handler = logging.FileHandler("POD_DeepOnet.log")
# logger.addHandler(file_handler)


class BranchNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, (8, 8), padding="same"),
            nn.AvgPool2d(2, 1, padding=0),
            nn.Conv2d(16, 16, (8, 8), padding="same"),
            nn.AvgPool2d(2, 1, padding=0),
            nn.Conv2d(16, 16, (8, 8), padding="same"),
            nn.AvgPool2d(2, 1, padding=0),
            nn.Conv2d(16, 16, (8, 8), padding="same"),
            nn.AvgPool2d(2, 1, padding=0),
        )
        self.dnn = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 24 * 24, 256), nn.Tanh(), nn.Linear(256, 180)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.swapaxes(x, 1, 3)
        x = self.conv_layers(x)
        x = self.dnn(x)
        return x


class TrunkNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x


branch_network = BranchNet()
trunk_network = TrunkNet()
model = sml.DeepOperatorNetwork(branch_network, trunk_network)


class PodDataSet(Dataset):
    """Load the dataset for POD example"""

    def __init__(self, u_basis, f, u):
        self.u_basis = u_basis
        self.f = f
        self.u = u

    def __len__(self):
        return int(self.f.shape[0])

    def __getitem__(self, i):
        return self.u_basis, self.f[i, :], self.u[i, :, 0]


modes = 180
file_name = "./data/Brusselator_data_KLE_lx_0.11_ly_0.15_v_0.15.npz"
F_train, U_train, F_test, U_test, X, u_mean, u_std, u_basis, lam_u = load_data(
    modes, file_name
)

train_data = DataLoader(
    PodDataSet(np.float32(u_basis), np.float32(F_train), np.float32(U_train)),
    batch_size=100,
    shuffle=True,
)
test_data = DataLoader(
    PodDataSet(np.float32(u_basis), np.float32(F_test), np.float32(U_train)),
    batch_size=100,
    shuffle=True,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
trainer = sml.Trainer(model, optimizer)
trainer.run(train_data=train_data, test_data=test_data, epochs=100, tolerance=1e-4)
