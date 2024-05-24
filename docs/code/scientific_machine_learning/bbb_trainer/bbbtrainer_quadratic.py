"""

Training a Bayesian neural network
=============================================================

In this example, we train a Bayesian neural network to learn the function :math:`f(x)=x^2`

"""

# %% md
#
# First, we have to import the necessary modules.

# %%

# Default imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# UQpy imports
import UQpy.scientific_machine_learning as sml

# %% md
#
# We define the network architecture using the ``nn.Sequential`` object
# and instantiate the ``BayesianNeuralNetwork``.

# %%

width = 5
network = nn.Sequential(
    sml.BayesianLinear(1, width),
    nn.ReLU(),
    sml.BayesianLinear(width, width),
    nn.ReLU(),
    sml.BayesianLinear(width, 1),
)
model = sml.FeedForwardNeuralNetwork(network)

# %% md
#
# With the neural network defined, we turn our attention to the training data.
# We want to learn the function :math:`f(x)=x^2` and define the training data using the
# pytorch Dataset and Dataloader.
#
# For more information on defining the training data,
# see the pytorch documentation at https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

# %%


class QuadraticDataset(Dataset):
    def __init__(self, n_samples=200):
        self.n_samples = n_samples
        self.x = torch.linspace(-5, 5, n_samples).reshape(-1, 1)
        self.y = self.x**2

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return self.x[item], self.y[item]


# %% md
#
# Before we continue with training the network, let's get the initial prediction of the neural network on the data.

# %%

initial_prediction = model(QuadraticDataset().x)

# %% md
#
# So far we have the neural network and training data. The ``BBBTrainer`` combines the two along with a
# pytorch optimization algorithm to learn the network parameters.
# We instantiate the ``BBBTrainer``, train the network, then print the initial and final loss alongside a model summary.

# %%

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_data = DataLoader(QuadraticDataset(), batch_size=20, shuffle=True)
trainer = sml.BBBTrainer(model, optimizer)
print("Starting Training...", end="")
trainer.run(train_data=train_data, epochs=300, beta=1e-6, num_samples=10)
print("done")

print("Initial loss:", trainer.history["train_loss"][0])
print("Final loss:", trainer.history["train_loss"][-1])
model.summary()

# %% md
#
# We compare the initial and final predictions and plot the loss history using matplotlib.

# %%

x = QuadraticDataset().x
y = QuadraticDataset().y
model.train(False)
model.sample(False)
final_prediction = model(x)
fig, ax = plt.subplots()
ax.plot(
    x.detach().numpy(),
    initial_prediction.detach().numpy(),
    label="Initial Prediction",
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
    y.detach().numpy(),
    label="Exact",
    color="black",
    linestyle="dashed",
)
ax.set_title("Initial and Final NN Predictions")
ax.set(xlabel="x", ylabel="f(x)")
ax.legend()

train_loss = trainer.history["train_loss"].detach().numpy()
fig, ax = plt.subplots()
ax.plot(train_loss)
ax.set_title("Bayes By Backpropagation Training Loss")
ax.set(xlabel="Epoch", ylabel="Loss")

plt.show()

# %% md
# The Bayesian neural network is a probabilistic model. Each of its parameters, in this case weights and biases,
# are governed by Gaussian distributions. We can get a deterministic output from the BNN by setting
# ``model.sample(False)``, which sets each parameter to the mean of its distribution.
#
# We can obtain error bars on model's output by sampling the parameters from their governing distribution.
# This is done by setting ``model.sample(True)`` and computing the forward model evaluation many times,
# then computing the sample variance

# %%


model.sample(False)
print("BNN is deterministic:", model.is_deterministic())
mean = model(x)

model.sample(True)
print("BNN is deterministic:", model.is_deterministic())
n = 10_000
samples = torch.zeros(len(x), n)
for i in range(n):
    samples[:, i] = model(x).squeeze()
variance = torch.var(samples, dim=1)
standard_deviation = torch.sqrt(variance)

x_plot = x.squeeze().detach().numpy()
mu = mean.squeeze().detach().numpy()
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
