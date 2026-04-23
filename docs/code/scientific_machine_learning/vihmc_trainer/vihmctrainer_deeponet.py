"""
Training a Bayesian operator network with VI-HMC
=============================================================
In this example we train a Bayesian operator network to learn the Burger's equation using VI-HMC
"""

# %% md
#
# First, we have to import the necessary modules.

# %%
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy
import matplotlib.pyplot as plt
import UQpy.scientific_machine_learning as sml
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger("UQpy")  # Optional, display UQpy logs to console
logger.setLevel(logging.INFO)


# %% md
# VI training
# =============================================================
# The first step is to train the Bayesian neural network using VI

# %%

# %% md
#
# We first define our training data for the DeepOperator network.
# We want to learn the solution of the Burger's equation and define the training data using the
# pytorch Dataset and Dataloader.
#
# For more information on defining the training data,
# see the pytorch documentation at https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

# %%
def get_data(N_train, N_valid, batch_size, p=10201):
    class BurgersDataSet(Dataset):
        """Load the Burgers dataset"""

        def __init__(self, branch_in, trunk_in, output, p):
            self.x = trunk_in.astype(np.float32)
            self.f_x = branch_in.astype(np.float32)
            self.u_x = output.astype(np.float32)
            self.p = p

        def __len__(self):
            return int(self.f_x.shape[0])

        def __getitem__(self, i):
            ind = np.random.choice(range(self.x.shape[0]), self.p, replace=False)
            return self.x[ind], np.expand_dims(self.f_x[i, :], axis=0), np.expand_dims(self.u_x[i, ind], axis=1)

    data_mat = scipy.io.loadmat("./DeepOnet_data.mat")
    train_data = BurgersDataSet(
        data_mat["branch_in"][0:N_train],
        data_mat["trunk_in"],
        data_mat["solution"][0:N_train],
        p=p,
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_dataset = BurgersDataSet(
        data_mat["branch_in"][N_train: N_train + N_valid],
        data_mat["trunk_in"],
        data_mat["solution"][N_train: N_train + N_valid],
        p=p,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    return train_loader, valid_loader


train_dataset, test_dataset = get_data(N_train=10, N_valid=10, batch_size=128, p=100)
data_noise = 1.0


# %% md
#
# Next we define a class to apply the necessary boundary conditions for the Burger's equation

# %%
class Apply_BC(nn.Module):
    def __init__(self):
        super(Apply_BC, self).__init__()

    def forward(self, x):
        x_bc = torch.stack(
            [
                torch.sin(2 * torch.pi * x[:, :, 1]),
                torch.sin(4 * torch.pi * x[:, :, 1]),
                torch.cos(2 * torch.pi * x[:, :, 1]),
                torch.cos(4 * torch.pi * x[:, :, 1]),
            ],
            dim=2,
        )
        return torch.cat([x[:, :, 0].unsqueeze(dim=2), x_bc], dim=2)


# %% md
#
# We then define the function to build a neural network. This function will be used to build the trunk and the branch
# networks of the DeepONet.

# %%

def build_nn(input_dim, output_dim, width, depth, is_bayesian, apply_bc):
    layer = sml.BayesianLinear if is_bayesian else nn.Linear
    modules = [Apply_BC()] if apply_bc else []
    modules.append(layer(input_dim, width))
    modules.append(nn.Tanh())
    for i in range(depth - 2):
        modules.append(layer(width, width))
        modules.append(nn.Tanh())
    modules.append(layer(width, output_dim))
    network = nn.Sequential(*modules)
    return network


# %% md
#
# Next we define the Gaussian negative log likelihood loss function with a fixed variance of the noise

# %%

class GaussianNLLLoss(nn.Module):
    def __init__(self, var, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var = var

    def forward(self, prediction, target):
        assert (
                prediction.shape == target.shape
        ), "Prediction does not match target shape in the loss function"
        return F.gaussian_nll_loss(
            prediction, target, torch.ones(prediction.shape) * self.var, reduction="sum"
        )


# %% md
#
# We now build the branch and trunk networks and pass them to the ``DeepOperatorNetwork`` class to construct our
# DeepOnet

# %%

branch_net = build_nn(
    input_dim=101, output_dim=100, width=100, depth=9, is_bayesian=True, apply_bc=False
)
trunk_net = build_nn(
    input_dim=5, output_dim=100, width=100, depth=9, is_bayesian=True, apply_bc=True
)
model = sml.DeepOperatorNetwork(branch_network=branch_net, trunk_network=trunk_net)

# %% md
#
# So far we have the operator network and training data. The ``BBBTrainer`` combines the two along with a pytorch
# optimization algorithm to learn the network parameters using VI. We instantiate the ``BBBTrainer`` and train the
# network.

# %%

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = sml.BBBTrainer(model, optimizer, loss_function=GaussianNLLLoss(var=data_noise ** 2))
trainer.run(train_data=train_dataset, test_data=test_dataset, epochs=10)

det_branch_net = build_nn(input_dim=101, output_dim=100, width=100, depth=9, is_bayesian=False, apply_bc=False)
det_trunk_net = build_nn(
    input_dim=5, output_dim=100, width=100, depth=9, is_bayesian=False, apply_bc=True
)
det_model = sml.DeepOperatorNetwork(branch_network=det_branch_net, trunk_network=det_trunk_net)

# %% md
#
# We define the necessary parameters to run HMC

# %%
# HMC Params
step_size = 2e-4
num_samples = 10
burn = num_samples // 5
post_var = 0.2024 ** 2
L = max(1, int(math.pi * post_var / (2 * step_size)))
loss = "NLL"  # regression or NLL
tau_out = (
        data_noise ** 2
)  # Measure of precision: 1/variance if Regression or variance if NLL
prior_sigma = 0.1

# %% md
#
# Finally the ``VIHMCTrainer`` samples the posterior distribution of parameters using HMC. This trainer uses the
# information from VI learned distributions to reduce the dimensions of the parameter space to run the HMC.

# %%

vihmc_trainer = sml.VIHMCTrainer(det_model=det_model, vi_model=model)
params_hmc, pred_list, _ = vihmc_trainer.run(
    train_data=train_dataset,
    valid_data=test_dataset,
    variance_threshold=0.90,
    step_size=step_size,
    num_samples=num_samples,
    burn=burn,
    prior_var=prior_sigma ** 2,
    num_steps=L,
    tau_out=tau_out,
    debug=False,
)
