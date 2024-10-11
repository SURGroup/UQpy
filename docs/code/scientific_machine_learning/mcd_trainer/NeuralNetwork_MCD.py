"""
Probabilisitc Models with Monte Carlo Dropout
=============================================

This example shows how to create a probabilistic neural network using UQpy's ProbabilisticDropout layers.
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
        self.y =0.4 * torch.sin(4 * self.x) + 0.5 * torch.cos(12 * self.x)
        self.y += torch.normal(0, self.noise_std, self.x.shape)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return self.x[item], self.y[item]


# %% md
# We define our model as a fully connected feed-forward neural network.
# After each hidden layer, we add a :py:class:`ProbabilisticDropout` layer to randomly set some tensor elements to zero
# with probability :math:`p=1e-3`. These layers will be *inactive*  during training,
# and will only be turned on during evaluation.
#
# The model is trained using Pytorch's gradient descent algorithms passed to :py:class:`Trainer`.
# We include a scheduler to control the learning rate.

# %%

width = 30
p = 2e-3
network = nn.Sequential(
    nn.Linear(1, width),
    nn.ReLU(),
    nn.Linear(width, width),
    sml.ProbabilisticDropout(p=p),
    nn.ReLU(),
    nn.Linear(width, width),
    sml.ProbabilisticDropout(p=p),
    nn.ReLU(),
    nn.Linear(width, 1),
)
model = sml.FeedForwardNeuralNetwork(network)
model.drop(False)  # turn off all ProbabilisticDropout layers

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
# The rest of this example plots the model predictions to compare them to the exact solution.
# We also show how to activate the dropout layers to compute a probabilistic prediction

# %%

# Plot training history
fig, ax = plt.subplots()
ax.semilogy(trainer.history["train_loss"])
ax.set_title("Training Loss")
ax.set(xlabel="Epoch", ylabel="Loss")
fig.tight_layout()

# Plot model predictions
x_noisy = dataset.x
y_noisy = dataset.y
x_exact = torch.linspace(-1, 1, 1000).reshape(-1, 1)
y_exact = 0.4 * torch.sin(4 * x_exact) + 0.5 * torch.cos(12 * x_exact)

model.eval()
# compute deterministic prediction
model.drop(False)
print("Model is dropping:", model.dropping)
with torch.no_grad():
    deterministic_prediction = model(x_exact)

# compute probabilistic predictions
model.drop()  # turn on all dropout layers
print("Model is dropping:", model.dropping)
n = 500
samples = torch.zeros(n, len(x_exact))
for i in range(n):
    samples[i, :] = model(x_exact).detach().squeeze()
quantile_low = torch.quantile(samples, q=0.025, dim=0)
quantile_high = torch.quantile(samples, q=0.975, dim=0)

fig, ax = plt.subplots()
ax.scatter(x_noisy, y_noisy, label="Training Data", color="black")
ax.plot(
    x_exact,
    y_exact,
    label="Exact",
    color="black",
    linestyle="dashed",
)
ax.plot(
    x_exact,
    deterministic_prediction,
    label="Deterministic Model",
    color="tab:blue",
)
ax.fill_between(
    x_exact.squeeze(),
    quantile_low,
    quantile_high,
    label="Middle 95%",
    color="tab:blue",
    alpha=0.3,
)
ax.set_title("Monte Carlo Dropout Predictions")
ax.set(xlabel="x", ylabel="f(x)")
ax.legend()


# Plotting Results
# x_data = SinusoidalDataset().x.detach()
# y_data = SinusoidalDataset().y.detach()
# x_val = torch.linspace(-1, 1, 1000).view(-1, 1).detach()
# y_val = 0.4 * torch.sin(4 * x_val) + 0.5 * torch.cos(12 * x_val).detach()
# pred_val = model(x_val).detach()
#
# # %% Plot the deterministic model estimates
# fig, ax = plt.subplots()
# ax.scatter(x_data, y_data, label="Data", color="black", s=50)
# ax.plot(
#     x_val,
#     pred_val,
#     label="Final Prediction",
#     color="tab:orange",
# )
# ax.plot(
#     x_val.detach(),
#     y_val.detach(),
#     label="Target",
#     color="black",
#     linestyle="dashed",
# )
# ax.set_title("Deterministic Prediction")
# ax.set(xlabel="$x$", ylabel="$f(x)$")
# ax.legend()
# fig.tight_layout()
#
# train_loss = trainer.history["train_loss"].detach().numpy()
# fig, ax = plt.subplots()
# ax.semilogy(train_loss)
# ax.set_title("Training Loss")
# ax.set(xlabel="Epoch", ylabel="Loss")
# fig.tight_layout()
#
# # %%
# model.drop()  # activate the dropout layers
# n = 1_000
# samples = torch.zeros(n, len(x_val))
# for i in range(n):
#     samples[i, :] = model(x_val).detach().squeeze()
# mean = torch.mean(samples, dim=1)
# standard_deviation = torch.std(samples, dim=1)
#
# # Plotting Results
# fig, ax = plt.subplots()
# ax.plot(x_val, samples[:, 1], "tab:orange", label="Prediction 1")
# ax.plot(x_val, samples[:, 2], "tab:blue", label="Prediction 2")
# ax.scatter(x_data, y_data, color="black", label="Data")
# ax.plot(x_val, y_val, color="black", linestyle="dashed", label="Target")
# ax.set_title("Two Samples from Dropout NN")
# ax.set(xlabel="$x$", ylabel="$f(x)$")
# ax.legend()
# fig.tight_layout()
#
# # %%
# fig, ax = plt.subplots()
# ax.plot(x_val, mean, label="$\mu$")
# ax.fill_between(
#     x_val.view(-1),
#     torch.quantile(samples, q=0.025, dim=1),
#     torch.quantile(samples, q=0.975, dim=1),
#     label="95% Range",
#     # mean - (3 * standard_deviation),
#     # mean + (3 * standard_deviation),
#     # label="$\mu \pm 3\sigma$,",
#     alpha=0.3,
# )
# ax.plot(x_val, y_val, label="Target", color="black")
# ax.set_title("Dropout Neural Network 95% Range")
# ax.set(xlabel="x", ylabel="f(x)")
# ax.legend()

plt.show()
