"""
Training a Neural Network
==========================

This example shows how to train a simple neural network using UQpy's Trainer class.

"""

# %% md
# First, we import the necessary modules and, optionally, set UQpy to print logs to the console.

# %%

import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import UQpy.scientific_machine_learning as sml

torch.manual_seed(0)

# logger = logging.getLogger("UQpy")  # Optional, display UQpy logs to console
# logger.setLevel(logging.INFO)

# %% md
# Our neural network will approximate the function :math:`f(x)=0.4 \sin(4x) + 0.5 \cos(12x) + \epsilon` over the domain
# :math:`x \in [-1, 1]`. :math:`\epsilon` represents the noise in our measurement defined as the Gaussian random
# variable :math:`\epsilon \sim N(0, 0.05)`.
#
# Below we define the dataset by subclassing :py:class:`torch.utils.data.Dataset`.

# %%


class SinusoidalDataset(Dataset):
    def __init__(self, n_samples=20, noise_std=0.05):
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.x = torch.linspace(-1, 1, n_samples).reshape(-1, 1)
        self.y = 0.4 * torch.sin(4 * self.x) + 0.5 * torch.cos(12 * self.x)
        epsilon = torch.normal(torch.zeros_like(self.x), self.noise_std)
        self.y += epsilon

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return self.x[item], self.y[item]


# %% md
# Next, we define our model architecture using :py:class:`torch.nn.Sequential` and train the model.
# The model is trained using Pytorch's gradient descent algorithms passed to :py:class:`Trainer`.
# We include a scheduler to control the learning rate.

# %%


width = 30
network = nn.Sequential(
    nn.Linear(1, width),
    nn.ReLU(),
    nn.Linear(width, width),
    nn.ReLU(),
    nn.Linear(width, width),
    nn.ReLU(),
    nn.Linear(width, 1),
)
model = sml.FeedForwardNeuralNetwork(network)

dataset = SinusoidalDataset()
train_data = DataLoader(dataset, batch_size=20, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1_000)
trainer = sml.Trainer(model, optimizer, scheduler=scheduler)
print("Starting Training...", end="")
trainer.run(train_data=train_data, epochs=5_000)
print("done")

# %% md
# That's the hard part done! We defined our dataset, our model, and then fit the model to the data.
# The rest of this example plots the model prediction and exact solution for comparison.

# %%

# Plot training history
fig, ax = plt.subplots()
ax.semilogy(trainer.history["train_loss"])
ax.set_title("Bayes By Backpropagation Training Loss")
ax.set(xlabel="Epoch", ylabel="Loss")

# Get x and y data for plotting
x_noisy = dataset.x
y_noisy = dataset.y
x_exact = torch.linspace(-1, 1, 1000).reshape(-1, 1)
y_exact = 0.4 * torch.sin(4 * x_exact) + 0.5 * torch.cos(12 * x_exact)

# compute prediction from model
model.eval()
with torch.no_grad():
    prediction = model(x_exact)

# Plot model predictions

fig, ax = plt.subplots()
ax.scatter(x_noisy, y_noisy, label="Training Data", color="black")
ax.plot(
    x_exact,
    y_exact,
    label="Exact",
    color="black",
    linestyle="dashed",
)
ax.plot(x_exact, prediction, label="Model $\mu$", color="tab:blue")
ax.set_title("Neural Network Predictions")
ax.set(xlabel="x", ylabel="f(x)")
ax.legend()

plt.show()
