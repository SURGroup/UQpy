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
if logger.hasHandlers():
    logger.removeHandler(
        logger.handlers[0]
    )  # remove existing handlers to eliminate print statements
file_handler = logging.FileHandler("Bayesian_trainer.log")
logger.addHandler(file_handler)

priors = {
    "prior_mu": 0,
    "prior_sigma": 0.01,
    "posterior_mu_initial": (0, 0.1),
    "posterior_rho_initial": (-5, 0.1),
}


class BranchNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fnn = nn.Sequential(sml.BayesianLinear(101, 100, priors=priors), nn.Tanh())
        self.conv_layers = nn.Sequential(
            sml.BayesianConv2d(1, 16, (5, 5), padding="same", priors=priors),
            nn.AvgPool2d(2, 1, padding=0),
            sml.BayesianConvLayer(16, 16, (5, 5), padding="same", priors=priors),
            nn.AvgPool2d(2, 1, padding=0),
            sml.BayesianConv2d(16, 16, (5, 5), padding="same", priors=priors),
            nn.AvgPool2d(2, 1, padding=0),
            sml.BayesianConv2d(16, 64, (5, 5), padding="same", priors=priors),
            nn.AvgPool2d(2, 1, padding=0),
        )
        self.dnn = nn.Sequential(
            nn.Flatten(),
            sml.BayesianLinear(64 * 6 * 6, 512, priors=priors),
            nn.Tanh(),
            sml.BayesianLinear(512, 512, priors=priors),
            nn.Tanh(),
            sml.BayesianLinear(512, 200, priors=priors),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fnn(x)
        x = x.view(-1, 1, 10, 10)
        x = self.conv_layers(x)
        x = self.dnn(x)
        return x


class TrunkNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fnn = nn.Sequential(
            sml.BayesianLinear(2, 128, priors=priors),
            nn.Tanh(),
            sml.BayesianLinear(128, 128, priors=priors),
            nn.Tanh(),
            sml.BayesianLinear(128, 128, priors=priors),
            nn.Tanh(),
            sml.BayesianLinear(128, 200, priors=priors),
            nn.Tanh(),
        )
        self.Xmin = np.array([0.0, 0.0]).reshape((-1, 2))
        self.Xmax = np.array([1.0, 1.0]).reshape((-1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = 2.0 * (x - self.Xmin) / (self.Xmax - self.Xmin) - 1.0
        x = x.float()
        x = self.fnn(x)
        return x


branch_network = BranchNet()
trunk_network = TrunkNet()
model = sml.DeepOperatorNetwork(branch_network, trunk_network, 2)


class ElasticityDataSet(Dataset):
    """Load the Elasticity dataset"""

    def __init__(self, x, f_x, u_x, u_y):
        self.x = x
        self.f_x = f_x
        self.u_x = u_x
        self.u_y = u_y

    def __len__(self):
        return int(self.f_x.shape[0])

    def __getitem__(self, i):
        return self.x, self.f_x[i, :], (self.u_x[i, :, 0], self.u_y[i, :, 0])


(
    F_train,
    Ux_train,
    Uy_train,
    F_test,
    Ux_test,
    Uy_test,
    X,
    ux_train_mean,
    ux_train_std,
    uy_train_mean,
    uy_train_std,
) = load_data()
train_data = DataLoader(
    ElasticityDataSet(
        np.float32(X), np.float32(F_train), np.float32(Ux_train), np.float32(Uy_train)
    ),
    batch_size=100,
    shuffle=True,
)
test_data = DataLoader(
    ElasticityDataSet(
        np.float32(X), np.float32(F_test), np.float32(Ux_test), np.float32(Uy_test)
    ),
    batch_size=100,
    shuffle=True,
)


class LossFunction(nn.Module):
    def __init__(self, reduction: str = "mean", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reduction = reduction

    def forward(self, prediction, label):
        return F.mse_loss(
            prediction[0], label[0], reduction=self.reduction
        ) + F.mse_loss(prediction[1], label[1], reduction=self.reduction)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
trainer = sml.BBBTrainer(model, optimizer, LossFunction())
trainer.run(
    train_data=train_data,
    test_data=test_data,
    epochs=1000,
    tolerance=1e-4,
    beta=1e-5,
    num_samples=1,
)
torch.save(model.state_dict(), "./Bayesian_DeepOnet_LE.pt")
