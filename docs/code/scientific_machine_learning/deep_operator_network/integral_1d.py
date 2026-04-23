"""
Learning the 1D Integral Operator
=================================

In this example, we train a Deep Operator Network to learn the operator :math:`\mathcal{L}f(x) = \int f(x) dx`.

"""

# %% md
# In this example we will approximate the integral operator :math:`\mathcal{L}f(x) = \int f(x) dx`
# Using a Deep Operator Network. This example
#
# 1. Generates training data from a stochastic process
# 2. Defines the architecture of a Deep Operator Network
# 3. Fits the network parameters to the training data
# 4. Visualizes the results
#
# First, import the necessary modules.

# %%

# Default imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# UQpy imports
import UQpy.scientific_machine_learning as sml
from UQpy.stochastic_process import SpectralRepresentation

# %% md
# **1. Generate Training Data**
#
# We generate random functions using the stochastic process module in UQpy.
# The stochastic process is sampled at 100 points over  using the function ``srm_samples``.
# We denote these input functions :math:`f`. They are sampled at 100 sensing points :math:`x`
# on the interval :math:`[0, 1]`.
# The operator :math:`\mathcal{L}f(x) = \int f(x) dx` is approximated using the function ``operator`` to
# numerically approximate the integral of :math:`f` using the cumulative summation.
# The class ``DONDataSet`` formats the dataset for training. It does not compute any new information.

# %%


def srm_samples(n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a 1D Gaussian process using the Spectral Representation Method

    :param n_samples: Number of samples. Each sample is one row
    :return: time, samples
    """
    max_time = 1
    n_time = 100
    max_frequency = 8 * np.pi
    n_frequency = 32

    delta_time = max_time / n_time
    delta_frequency = max_frequency / n_frequency
    frequency = np.linspace(0, max_frequency - delta_frequency, num=n_frequency)
    time = np.linspace(0, max_time - delta_time, num=n_time)
    power_spectrum = np.exp(-2 * frequency**2)
    srm = SpectralRepresentation(
        n_samples, power_spectrum, delta_time, delta_frequency, n_time, n_frequency
    )
    return (
        torch.tensor(time, dtype=torch.float).reshape(-1, 1),
        torch.tensor(srm.samples, dtype=torch.float),
    )


def operator(x: torch.Tensor, u_x: torch.Tensor) -> torch.Tensor:
    """Numerically approximate the integral operator

    :param x: Sensing points at which the function :math:`u` is evalated
    :param u_x: Evaluations of the function :math:`u` at points :math:`x`
    :return: Cumulative sum of ``u_x``
    """
    return torch.cumsum(u_x, axis=1) * (x[1] - x[0])


class DONDataSet(Dataset):
    """Format the data for UQpy Trainer"""

    def __init__(self, x, u_x, gu_x):
        self.x = x
        self.u_x = u_x
        self.gu_x = gu_x

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, i):
        return self.x, self.u_x[i, :], self.gu_x[i, :]


# %% md
# **2. Initialize Deep Operator Network**
#
# A Deep Operator Network is defined using the branch network, that encodes information about the domain :math:`x`,
# and the trunk network, that encodes information about the transformation :math:`\mathcal{L}:f \mapsto \int f`.
# This network maps a 1D function sampled at 100 points.
# The width of each hidden layer in the network is given in ``width`` and the final width of the branch
# and trunk networks before they are combined is given in ``final_width``.
#
# The branch and trunk networks can be defined using a ``torch.nn.Module`` or any subclass of it.
# Here we use the ``torch.nn.Sequential`` class to define the networks.

# %%

n_points = 100
n_dimension = 1
width = 20
final_width = 100  # Number of nodes on output of branch and trunk network
branch_network = nn.Sequential(
    nn.Linear(n_points, width),
    nn.ReLU(),
    nn.Linear(width, width),
    nn.ReLU(),
    nn.Linear(width, final_width),
)
trunk_network = nn.Sequential(
    nn.Linear(n_dimension, width),
    nn.ReLU(),
    nn.Linear(width, width),
    nn.ReLU(),
    nn.Linear(width, final_width),
    nn.ReLU(),
)
model = sml.DeepOperatorNetwork(branch_network, trunk_network)

# %% md
# **3. Train the network on the data**
#
# The ``Trainer`` class organizes the model, data, and a pytorch optimization algorithm to learn the model parameters.

# %%

x, f_x = srm_samples(300)
Lf_x = operator(x, f_x)
train_data = DataLoader(DONDataSet(x, f_x, Lf_x), batch_size=20, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
trainer = sml.Trainer(model, optimizer)
trainer.run(train_data=train_data, epochs=1_000, tolerance=1e-6)

# %% md
# **4. Visualize the results**
#
# Finally, we plot the loss history and the predictions made by the optimized network.

# %%

# Plot loss history
train_loss = trainer.history["train_loss"].detach().numpy()
fig, ax = plt.subplots()
ax.semilogy(train_loss)
ax.set_title("Training Loss of Deep Operator Network")
ax.set(xlabel="Epoch", ylabel="Loss")

# Plot deep operator network prediction and exact solution
colors = ("tab:blue", "tab:orange", "tab:green")
x_plot = x.detach().numpy().squeeze()
fig, ax = plt.subplots()
for i in range(3):
    prediction = model(x, f_x[i, :])
    ax.plot(
        x_plot,
        Lf_x[i, :].detach().numpy().squeeze(),
        label=f"$g_{i}:=\mathcal{{L}}f_{i} (x)$",
        color=colors[i],
        linestyle="dashed",
    )
    ax.plot(
        x_plot,
        prediction.detach().numpy(),
        label=f"DoN $\hat{{g}}_{i}$",
        color=colors[i],
    )
ax.set_title("Deep Operator Network Predictions")
ax.legend()

plt.show()
