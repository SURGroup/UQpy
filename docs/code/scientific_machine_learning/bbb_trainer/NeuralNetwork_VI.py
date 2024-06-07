"""
Training with Variational Inference
=============================================================

In this example, we train a Bayesian neural network using Variational Inference.

"""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import UQpy.scientific_machine_learning as sml


logger = logging.getLogger("UQpy")
logger.setLevel(logging.INFO)

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


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_data = DataLoader(SinusoidalDataset(), batch_size=20, shuffle=True)
trainer = sml.BBBTrainer(model, optimizer)
print("Starting Training...", end="")
trainer.run(train_data=train_data, epochs=5000, beta=1e-6, num_samples=10)
print("done")

# Plotting Results
x = SinusoidalDataset().x
y = SinusoidalDataset().y
x_val = torch.linspace(-1, 1, 1000).view(-1, 1)
y_val = 0.4 * torch.sin(4 * x_val) + 0.5 * torch.cos(12 * x_val)
model.train(False)
model.sample(False)
# final_prediction = model(x)
pred_val = model(x_val)
fig, ax = plt.subplots()
ax.plot(
    x.detach().numpy(),
    y.detach().numpy(),
    "*",
    label="Data",
    color="tab:blue",
)
ax.plot(
    x_val.detach().numpy(),
    pred_val.detach().numpy(),
    label="Final Prediction",
    color="tab:orange",
)
ax.plot(
    x_val.detach(),
    y_val.detach(),
    label="Exact",
    color="black",
    linestyle="dashed",
)

ax.set_title("Predictions")
ax.set(xlabel="x", ylabel="f(x)")
ax.legend()

train_loss = trainer.history["train_loss"].detach().numpy()
fig, ax = plt.subplots()
ax.plot(train_loss)
ax.set_title("Bayes By Backpropagation Training Loss")
ax.set(xlabel="Epoch", ylabel="Loss")

plt.show()

model.sample(False)
print("BNN is deterministic:", model.is_deterministic())
mean = model(x_val)
model.sample(True)
print("BNN is deterministic:", model.is_deterministic())
n = 1000
samples = torch.zeros(len(x_val), n)
for i in range(n):
    samples[:, i] = model(x_val).squeeze()
variance = torch.var(samples, dim=1)
standard_deviation = torch.sqrt(variance)

x_plot = x.squeeze().detach().numpy()
mu = mean.squeeze().detach().numpy()
sigma = standard_deviation.squeeze().detach().numpy()
fig, ax = plt.subplots()
ax.plot(x_val, mu, label="$\mu$")
ax.plot(
    x_val.detach(), y_val.detach(), label="Exact", color="black", linestyle="dashed"
)
ax.fill_between(
    x_val.view(-1).detach(),
    mu - (3 * sigma),
    mu + (3 * sigma),
    label="$\mu \pm 3\sigma$,",
    alpha=0.3,
)
ax.set_title("Bayesian Neural Network $\mu \pm 3\sigma$")
ax.set(xlabel="x", ylabel="f(x)")
ax.legend()

plt.show()
