"""
Learning a Linear Elastic system
================================

In this example, we train a DeepOperatorNetwork to learn the behavior of a linear elastic system.
"""

# %% md
# In this example we use a deep operator network to approximate the solution to a linear elastic system.
# Using a given data set, we will
#
# 1. Construct a deep operator network
# 2. Load the training and testing data
# 3. Fit the network parameters to the training data
# 4. Plot the loss history
#
# First, import the necessary modules and set the logging behavior of UQpy.

# %%

# torch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# UQpy imports
import logging
import UQpy.scientific_machine_learning as sml
from local_elastic_data import load_data

logger = logging.getLogger("UQpy")
logger.setLevel(logging.INFO)


# %% md
# **1. Construct a deep operator network**
#
# A Deep Operator Network is defined using the branch network, that encodes information about the domain :math:`x`,
# and the trunk network, that encodes information about the transformation.
#
# The branch and trunk networks can be defined using a ``torch.nn.Module`` or any subclass of it.
# Here we use subclasses of ``torch.nn.Module`` to define the networks. Both classes use standard
# torch activation functions and layers to define their operations.

# %%

class BranchNetwork(nn.Module):
    """Construct the branch network for a deep operator network"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fnn = nn.Sequential(nn.Linear(101, 100), nn.Tanh())
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, (5, 5), padding="same"),
            nn.AvgPool2d(2, 1, padding=0),
            nn.Conv2d(16, 16, (5, 5), padding="same"),
            nn.AvgPool2d(2, 1, padding=0),
            nn.Conv2d(16, 16, (5, 5), padding="same"),
            nn.AvgPool2d(2, 1, padding=0),
            nn.Conv2d(16, 64, (5, 5), padding="same"),
            nn.AvgPool2d(2, 1, padding=0),
        )
        self.dnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 200),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fnn(x)
        x = x.view(-1, 1, 10, 10)
        x = self.conv_layers(x)
        x = self.dnn(x)
        return x.unsqueeze(1)


class TrunkNetwork(nn.Module):
    """Construct the trunk network for a deep operator network"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fnn = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 200),
            nn.Tanh(),
        )
        self.x_min = torch.tensor([[0.0, 0.0]])
        self.x_max = torch.tensor([[1.0, 1.0]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        x = x.float()
        x = self.fnn(x)
        return x


branch_network = BranchNetwork()
trunk_network = TrunkNetwork()
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
    ElasticityDataSet(X, F_train, Ux_train, Uy_train), batch_size=100, shuffle=True
)
test_data = DataLoader(
    ElasticityDataSet(X, F_test, Ux_test, Uy_test), batch_size=100, shuffle=True
)


# %% md
# **3. Fit the network parameters to the training data**
#
# All that's left is to define a loss function, an optimizer, and run the Trainer.
# We use the Mean Squared Error loss and an Adam optimizer from torch.
# We assemble the model, optimizer, loss function, and training data with UQpy's trainer to learn the model parameters.

# %%


class LossFunction(nn.Module):
    def __init__(self, reduction: str = "mean", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f = nn.MSELoss(reduction=reduction)

    def forward(self, prediction, label):
        return self.f(prediction[:, :, 0], label[0]) + self.f(
            prediction[:, :, 1],
            label[1],
        )


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
trainer = sml.Trainer(model, optimizer, loss_function=LossFunction(), scheduler=scheduler)
trainer.run(train_data=train_data, test_data=test_data, epochs=50)


# %% md
# Finally, we plot the training history.

# %%

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.semilogy(trainer.history["train_loss"], label="Train Loss")
ax.semilogy(trainer.history["test_loss"], label="Test Loss")
ax.set_title("DeepONet Training History")
ax.set(xlabel="Epoch", ylabel="MSE Loss")
ax.legend()

plt.show()