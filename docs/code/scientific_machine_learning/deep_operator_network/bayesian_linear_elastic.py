"""
Learning a Linear Elastic system
================================

In this example, we train a Bayesian DeepOperatorNetwork to learn the behavior of a linear elastic system.
"""

# %% md
# In this example we use a deep operator network to approximate the solution to a linear elastic system.
# The dataset is provided by Goswami et al. :cite:`goswami2022elasticity` and our architecture closely follows their design.
# Using the dataset, we will
#
# 1. Construct a deep operator network
# 2. Load the training and testing data
# 3. Fit the network parameters to the training data
# 4. Plot the loss history
#
# First, import the necessary modules and set the logging behavior of UQpy.

# %%


import logging
import scipy.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import UQpy.scientific_machine_learning as sml

logger = logging.getLogger("UQpy")
logger.setLevel(logging.INFO)

# %% md
# **1. Construct a Bayesian deep operator network**
#
# A Bayesian Deep Operator Network is defined using the branch network, that encodes information about the domain :math:`x`,
# and the trunk network, that encodes information about the transformation.
#
# The branch and trunk networks can be defined using a ``torch.nn.Module`` or any subclass of it.
# Here we use subclasses of ``torch.nn.Module`` to define the networks. Both classes use standard
# torch activation functions and UQpy layers to define their operations.

# %%

class BranchNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fnn = nn.Sequential(sml.BayesianLinear(101, 100), nn.Tanh())
        self.conv_layers = nn.Sequential(
            sml.BayesianConv2d(1, 16, (5, 5), padding="same"),
            nn.AvgPool2d(2, 1, padding=0),
            sml.BayesianConv2d(16, 16, (5, 5), padding="same"),
            nn.AvgPool2d(2, 1, padding=0),
            sml.BayesianConv2d(16, 16, (5, 5), padding="same"),
            nn.AvgPool2d(2, 1, padding=0),
            sml.BayesianConv2d(16, 64, (5, 5), padding="same"),
            nn.AvgPool2d(2, 1, padding=0),
        )
        self.dnn = nn.Sequential(
            nn.Flatten(),
            sml.BayesianLinear(64 * 6 * 6, 512),
            nn.Tanh(),
            sml.BayesianLinear(512, 512),
            nn.Tanh(),
            sml.BayesianLinear(512, 200),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fnn(x)
        x = x.view(-1, 1, 10, 10)
        x = self.conv_layers(x)
        x = self.dnn(x)
        return x.unsqueeze(1)


class TrunkNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fnn = nn.Sequential(
            sml.BayesianLinear(2, 128),
            nn.Tanh(),
            sml.BayesianLinear(128, 128),
            nn.Tanh(),
            sml.BayesianLinear(128, 128),
            nn.Tanh(),
            sml.BayesianLinear(128, 200),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fnn(x)


branch_network = BranchNet()
trunk_network = TrunkNet()
model = sml.DeepOperatorNetwork(branch_network, trunk_network, 2)


# %% md
# **2. Load the training and testing data**
#
# With the model constructed, we turn our attention to the training data.
# Here we define a subclass of ``torch.nn.Dataset`` as outlined by the [torch documentation](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class).

# %%

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


elastic_data = io.loadmat('linear_elastic_data.mat')
train_dataset, test_dataset = random_split(ElasticityDataSet(
    elastic_data['X'], elastic_data['F'], elastic_data['Ux'],
    elastic_data['Uy']), [0.9, 0.1])

train_data = DataLoader(train_dataset,
                        batch_size=20,
                        shuffle=True,
                        )
test_data = DataLoader(test_dataset)

# %% md
# **3. Fit the network parameters to the training data**
#
# All that's left is to define a loss function, an optimizer, and run the BBBTrainer.
# We use the Mean Squared Error loss, KullbackLeibler divergence, and an Adam optimizer from torch.
# We assemble the model, optimizer, loss function, and training data with UQpy's BBBTrainer to learn the model parameters.

# %%


class LossFunction(nn.Module):
    def __init__(self, reduction: str = "mean", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reduction = reduction

    def forward(self, prediction, label):
        return F.mse_loss(
            prediction[:, :, 0], label[0], reduction=self.reduction
        ) + F.mse_loss(prediction[:, :, 1], label[1], reduction=self.reduction)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
trainer = sml.BBBTrainer(model, optimizer, loss_function=LossFunction(), scheduler=scheduler)
trainer.run(
    train_data=train_data,
    test_data=test_data,
    epochs=50,
    beta=1e-6,
    num_samples=5,
)


# %% md
# Finally, we plot the training history.

# %%

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.semilogy(trainer.history["train_loss"], label="Train Loss")
ax.semilogy(trainer.history["test_nll"], label="Test NLL")
ax.set_title("Bayesian DeepONet Training History")
ax.set(xlabel="Epoch", ylabel="Loss")
ax.legend()

plt.show()
