# %%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import UQpy.scientific_machine_learning as sml
import logging

# %%
logger = logging.getLogger("UQpy")
logger.setLevel(logging.INFO)

width = 20
network = nn.Sequential(
    nn.Linear(1, width),
    nn.ReLU(),
    nn.Linear(width, width),
    sml.Dropout(p=0.1),
    nn.ReLU(),
    nn.Linear(width, width),
    sml.Dropout(p=0.1),
    nn.ReLU(),
    nn.Linear(width, width),
    sml.Dropout(p=0.1),
    nn.ReLU(),
    nn.Linear(width, 1),
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


# %% Train with dropout layers inactive
model.eval()  # this activates the dropout layers
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_data = DataLoader(SinusoidalDataset(), batch_size=20, shuffle=True)
trainer = sml.Trainer(model, optimizer)
print("Starting Training...", end="")
trainer.run(train_data=train_data, epochs=50000)
print("done")

# Plotting Results
x = SinusoidalDataset().x
y = SinusoidalDataset().y
x_val = torch.linspace(-1, 1, 1000).view(-1, 1)
y_val = 0.4 * torch.sin(4 * x_val) + 0.5 * torch.cos(12 * x_val)
pred_val = model(x_val)
# %% Plot the deterministic model estimates
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
ax.set_title("Deterministic training Loss")
ax.set(xlabel="Epoch", ylabel="Loss")
plt.show()
# %%
model.train()  # this activates the dropout layers
n = 10_000
samples = torch.zeros(len(x_val), n)
for i in range(n):
    samples[:, i] = model(x_val).detach().squeeze()
variance = torch.var(samples, dim=1)
standard_deviation = torch.sqrt(variance)
# Plotting Results
fig, ax = plt.subplots()
ax.plot(x, y, "bo", label="True Data")
ax.plot(x_val, samples[:, 1], "r-", label="Prediction 1")
ax.plot(x_val, samples[:, 2], "c--", label="Prediction 2")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.show()
# %%
x_plot = x.squeeze().detach().numpy()
mu = torch.mean(samples, dim=1)
sigma = standard_deviation.squeeze().detach().numpy()
fig, ax = plt.subplots()
ax.plot(x_val, mu, label="$\mu$")
ax.plot(x_val, y_val.detach().numpy(), label="Exact", color="black", linestyle="dashed")
ax.fill_between(
    x_val.view(-1),
    mu - (3 * sigma),
    mu + (3 * sigma),
    label="$\mu \pm 3\sigma$,",
    alpha=0.3,
)
ax.set_title("Bayesian Neural Network $\mu \pm 3\sigma$")
ax.set(xlabel="x", ylabel="f(x)")
ax.legend()

plt.show()
# %%