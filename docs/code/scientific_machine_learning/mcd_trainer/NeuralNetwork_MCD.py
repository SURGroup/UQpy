# %%

#!!!
import sys
import os

# Add the src directory to sys.path
src_path = '/Users/george/Documents/Main_Files/Scripts/UQpy/src'
if src_path not in sys.path:
    sys.path.append(src_path)
#!!!
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# UQpy imports
from UQpy.scientific_machine_learning.layers import Dropout
from UQpy.scientific_machine_learning.neural_networks import FeedForwardNeuralNetwork
from UQpy.scientific_machine_learning.trainers import Trainer
import logging
#%% 
logger = logging.getLogger("UQpy")
logger.setLevel(logging.INFO)

width = 20
network = nn.Sequential(
    nn.Linear(1, width),
    nn.ReLU(),
    nn.Linear(width, width),
    nn.ReLU(),
    nn.Linear(width, width),
    Dropout(drop_rate=0.5),
    nn.ReLU(),
    nn.Linear(width, width),
    Dropout(drop_rate=0.5),
    nn.ReLU(),
    nn.Linear(width, 1),
)
model = FeedForwardNeuralNetwork(network)


class SinusoidalDataset(Dataset):
    def __init__(self, n_samples=20, noise_std=0.05):
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.x = torch.linspace(-1, 1, n_samples).reshape(-1, 1)
        self.y = torch.tensor(0.4 * np.sin(4 * self.x) + 0.5 * np.cos(12 * self.x) +
                              np.random.normal(0, self.noise_std, self.x.shape), dtype=torch.float)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return self.x[item], self.y[item]


# %% Train with dropout layers inactive
model.eval()  # this activates the dropout layers
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_data = DataLoader(SinusoidalDataset(), batch_size=20, shuffle=True)
trainer = Trainer(model, optimizer)
print("Starting Training...", end="")
trainer.run(train_data=train_data, epochs=20000)
print("done")
# Plotting Results
x = SinusoidalDataset().x
y = SinusoidalDataset().y
final_prediction = model(x)
#%% Plot the deterministic model estimates
fig, ax = plt.subplots()
ax.plot(
    x.detach().numpy(),
    y.detach().numpy(), '*',
    label="Data",
    color="tab:blue",
)
ax.plot(
    x.detach().numpy(),
    final_prediction.detach().numpy(),
    label="Final Prediction",
    color="tab:orange",
)
ax.plot(
    x.detach().numpy(),
    (0.4 * np.sin(4 * x) + 0.5 * np.cos(12 * x)).detach().numpy(),
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
model.train() # this activates the dropout layers
n = 10_000
samples = torch.zeros(len(x), n)
for i in range(n):
    samples[:, i] = model(x).detach().squeeze()
variance = torch.var(samples, dim=1)
standard_deviation = torch.sqrt(variance)
# Plotting Results
fig, ax = plt.subplots()
ax.plot(x, y, 'bo', label='True Data')
ax.plot(x, samples[:,1], 'r-', label='Prediction 1')
ax.plot(x, samples[:,2], 'c--', label='Prediction 2')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
plt.show()
#%%
x_plot = x.squeeze().detach().numpy()
mu =  torch.mean(samples, dim=1)
sigma = standard_deviation.squeeze().detach().numpy()
fig, ax = plt.subplots()
ax.plot(x_plot, mu, label="$\mu$")
ax.plot(x_plot, y.detach().numpy(), label="Exact", color="black", linestyle="dashed")
ax.fill_between(
    x_plot, mu - (3 * sigma), mu + (3 * sigma), label="$\mu \pm 3\sigma$,", alpha=0.3
)
ax.set_title("Bayesian Neural Network $\mu \pm 3\sigma$")
ax.set(xlabel="x", ylabel="f(x)")
ax.legend()

plt.show()
# %%
