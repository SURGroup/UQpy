"""
Learning the Laplacian Operator
===============================

In this example, we train a Deep Operator Network to learn the operator :math:`\mathcal{L} f(x,y) = \nabla^2 f(x, y)`

"""

# %% md
# In this example we will approximate the Laplacian operator :math:`\mathcal{L}f(x, y) = \nabla^2 f(x, y)`
# Using a Deep Operator Network. This example
#
# 1. Generates training data from a 2D stochastic process
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
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# UQpy imports
from UQpy.scientific_machine_learning.neural_networks import DeepOperatorNetwork
from UQpy.scientific_machine_learning.trainers import Trainer
from local_utilities import srm_2d_samples

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


def compute_laplacian(function, x, y):
    df_dx = torch.autograd.grad(function(x, y).sum(), x, create_graph=True)[0]
    d2f_dx2 = torch.autograd.grad(df_dx.sum(), x)[0]

    df_dy = torch.autograd.grad(function(x, y).sum(), y, create_graph=True)[0]
    d2f_dy2 = torch.autograd.grad(df_dy.sum(), y)[0]

    return d2f_dx2 + d2f_dy2


class DONTwoDimensionalDataSet(Dataset):
    """Format the data for the UQpy trainer"""

    def __init__(self, x, f, g):
        self.x = x
        self.f = f
        self.g = g

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, item):
        return self.x, self.f_x[item, ...], self.g_x[item, ...]

# %% md
# **2. Initialize Deep Operator Network**

# %%

n_points = 100
n_dimension = 1
width = 20
final_width = 100  # Number of nodes on output of branch and trunk network
branch_network = nn.Sequential(
    nn.Linear(n_points ** 2, width),
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
model = DeepOperatorNetwork(branch_network, trunk_network)


# %% md
# **3. Train the network on the data**
#
# The ``Trainer`` class organizes the model, data, and a pytorch optimization algorithm to learn the model parameters.

# %%

# x_vector = torch.linspace(1, 2, n_points, requires_grad=True)
# y_vector = torch.linspace(1, 2, n_points, requires_grad=True)
# x, y = torch.meshgrid(x_vector, y_vector)

x, f = srm_2d_samples(100)
g = compute_laplacian()


"""
how do I compute the laplacian of the SRM samples using autograd?
I need to implement SRM samples in pytorch don't I
"""
laplacian = compute_laplacian(lambda x, y: torch.exp(x) * torch.sin(y), x, y)


def harmonic_function(x, y):
    # return torch.exp(x) * torch.sin(y)
    return torch.log(x**2 - y**2)
    # return (x**3) - (3 * x * (y**2))
