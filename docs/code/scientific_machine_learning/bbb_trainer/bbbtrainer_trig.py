"""
Training with Variational Inference
=============================================================

In this example, we train a Bayesian neural network using Variational Inference.

"""

# %% md
# First, we import the necessary modules and, optionally, set UQpy to print logs to the console.

# %%

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import UQpy.scientific_machine_learning as sml

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
        self.y = torch.tensor(
            0.4 * np.sin(4 * self.x)
            + 0.5 * np.cos(12 * self.x)
            + np.random.normal(0, self.noise_std, self.x.shape),
            dtype=torch.float,
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return self.x[item], self.y[item]


# %% md
# Next, we define our model architecture using UQpy's :py:class:`BayesianLinear` layers and train the model.
# The model is trained using the Bayes-by-backprop implementation in :py:class:`BBBTrainer`.

# %%


width = 20
network = nn.Sequential(
    sml.BayesianLinear(1, width),
    nn.ReLU(),
    sml.BayesianLinear(width, width),
    nn.ReLU(),
    sml.BayesianLinear(width, width),
    nn.ReLU(),
    sml.BayesianLinear(width, width),
    nn.ReLU(),
    sml.BayesianLinear(width, 1),
)
model = sml.FeedForwardNeuralNetwork(network)

dataset = SinusoidalDataset()
train_data = DataLoader(dataset, batch_size=20, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = sml.BBBTrainer(model, optimizer)
print("Starting Training...", end="")
trainer.run(train_data=train_data, epochs=5_000, beta=1e-6, num_samples=10)
print("done")

# %% md
# That's the hard part done! We defined our dataset, our model, and then fit the model to the data.
# The rest of this example plots the model predictions to compare them to the exact solution.

# %%

# Plot training history
fig, ax = plt.subplots()
ax.semilogy(trainer.history["train_loss"])
ax.set_title("Bayes By Backpropagation Training Loss")
ax.set(xlabel="Epoch", ylabel="Loss")

# Plot model predictions
x_noisy = dataset.x
y_noisy = dataset.y
x_exact = torch.linspace(-1, 1, 1000).reshape(-1, 1)
y_exact = 0.4 * torch.sin(4 * x_exact) + 0.5 * torch.cos(12 * x_exact)

# compute mean prediction from model
model.eval()
model.sample(False)
print("BNN is deterministic:", model.is_deterministic())
with torch.no_grad():
    mean_prediction = model(x_exact)

# compute stochastic prediction from model
model.sample(True)
print("BNN is deterministic:", model.is_deterministic())
n = 1_000
samples = torch.zeros(len(x_exact), n)
with torch.no_grad():
    for i in range(n):
        samples[:, i] = model(x_exact).squeeze()
standard_deviation = torch.std(samples, dim=1)

# convert tensors to numpy arrays for matplotlib
x_exact = x_exact.squeeze().detach().numpy()
y_exact = y_exact.squeeze().detach().numpy()
mean_prediction = mean_prediction.squeeze().detach().numpy()
standard_deviation = standard_deviation.squeeze().detach().numpy()

fig, ax = plt.subplots()
ax.scatter(x_noisy, y_noisy, label="Training Data", color="black")
ax.plot(
    x_exact,
    y_exact,
    label="Exact",
    color="black",
    linestyle="dashed",
)
ax.plot(x_exact, mean_prediction, label="Model $\mu$", color="tab:blue")
ax.fill_between(
    x_exact,
    mean_prediction - (3 * standard_deviation),
    mean_prediction + (3 * standard_deviation),
    label="$\mu \pm 3\sigma$,",
    color="tab:blue",
    alpha=0.3,
)
ax.set_title("Bayesian Neural Network Predictions")
ax.set(xlabel="x", ylabel="f(x)")
ax.legend()

plt.show()
