"""
Training with Hamiltonian Monte Carlo
=============================================================

In this example, we train a Bayesian neural network using Hamiltonian Monte Carlo (HMC).
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
# import UQpy.scientific_machine_learning as sml

logger = logging.getLogger("HMC_Example")
logger.setLevel(logging.INFO)


#
true_coefs = torch.tensor([1., 2., 3.])
data = torch.randn(2000, 3)
dim = 3
labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()


def model(data):
    coefs_mean = torch.zeros(dim)
    coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(3)))
    y = pyro.sample('y', dist.Bernoulli(
        logits=(coefs * data).sum(-1)), obs=labels)
    return y


hmc_kernel = HMC(model, step_size=0.0855, num_steps=4)
mcmc = MCMC(hmc_kernel, num_samples=500, warmup_steps=100)
mcmc.run(data)
mcmc.get_samples()['beta'].mean(0)
tensor([0.9819,  1.9258,  2.9737])
# # Define a simple Bayesian neural network using Pyro
# class BayesianNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(BayesianNN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

#     def model(self, x_data, y_data=None):
#         # Priors over the network parameters
#         fc1w_prior = dist.Normal(loc=torch.zeros_like(self.fc1.weight), scale=torch.ones_like(self.fc1.weight))
#         fc1b_prior = dist.Normal(loc=torch.zeros_like(self.fc1.bias), scale=torch.ones_like(self.fc1.bias))

#         fc2w_prior = dist.Normal(loc=torch.zeros_like(self.fc2.weight), scale=torch.ones_like(self.fc2.weight))
#         fc2b_prior = dist.Normal(loc=torch.zeros_like(self.fc2.bias), scale=torch.ones_like(self.fc2.bias))

#         fc3w_prior = dist.Normal(loc=torch.zeros_like(self.fc3.weight), scale=torch.ones_like(self.fc3.weight))
#         fc3b_prior = dist.Normal(loc=torch.zeros_like(self.fc3.bias), scale=torch.ones_like(self.fc3.bias))

#         priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
#                   'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior,
#                   'fc3.weight': fc3w_prior, 'fc3.bias': fc3b_prior}

#         # Sample the network parameters
#         lifted_module = pyro.random_module("module", self, priors)
#         lifted_reg_model = lifted_module()

#         # Likelihood (given parameters)
#         prediction_mean = lifted_reg_model(x_data).squeeze(-1)
#         pyro.sample("obs", dist.Normal(prediction_mean, torch.tensor(0.1)), obs=y_data)

#         return prediction_mean

# # Create the dataset
# class SinusoidalDataset(Dataset):
#     def __init__(self, n_samples=20, noise_std=0.05):
#         self.n_samples = n_samples
#         self.noise_std = noise_std
#         self.x = torch.linspace(-1, 1, n_samples).reshape(-1, 1)
#         self.y = torch.tensor(
#             0.4 * np.sin(4 * self.x)
#             + 0.5 * np.cos(12 * self.x)
#             + np.random.normal(0, self.noise_std, self.x.shape),
#             dtype=torch.float,
#         )

#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, item):
#         return self.x[item], self.y[item]

# # Initialize the model, data, and trainer
# input_dim = 1
# hidden_dim = 20
# output_dim = 1
# model = BayesianNN(input_dim, hidden_dim, output_dim)

# train_data = SinusoidalDataset()
# train_loader = DataLoader(train_data, batch_size=20, shuffle=True)

# # Instantiate HMCTrainer
# hmc_trainer = HMCTrainer(
#     model=model.model,
#     step_size=0.1,
#     num_samples=1000,
#     warmup_steps=500,
#     num_steps=10,
# )

# # Run HMC training
# print("Starting HMC Training...", end="")
# hmc_trainer.run()
# print("done")

# # Plotting Results
# x = SinusoidalDataset().x
# y = SinusoidalDataset().y
# x_val = torch.linspace(-1, 1, 1000).view(-1, 1)
# y_val = 0.4 * torch.sin(4 * x_val) + 0.5 * torch.cos(12 * x_val)

# samples = hmc_trainer.history["samples"]
# mean_predictions = samples['module$$$fc3.bias'].mean(0)

# fig, ax = plt.subplots()
# ax.plot(
#     x.detach().numpy(),
#     y.detach().numpy(),
#     "*",
#     label="Data",
#     color="tab:blue",
# )
# ax.plot(
#     x_val.detach().numpy(),
#     mean_predictions.detach().numpy(),
#     label="Final Prediction",
#     color="tab:orange",
# )
# ax.plot(
#     x_val.detach(),
#     y_val.detach(),
#     label="Exact",
#     color="black",
#     linestyle="dashed",
# )

# ax.set_title("Predictions with HMC")
# ax.set(xlabel="x", ylabel="f(x)")
# ax.legend()

# plt.show()
