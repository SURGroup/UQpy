# %%
import numpy as np
import torch
import pyro
import pyro.distributions as dist
import torch.nn as nn
from pyro.nn import PyroModule, PyroSample
from pyro.infer import MCMC, HMC, Predictive
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# %%
# Define underflying fuction
def generate_data(x, noise_std=0.02):
    noise = noise_std * torch.randn_like(x)
    y = x + 0.4 * torch.sin(4 * x + noise) + 0.5 * torch.cos(2 * x + noise)
    return y

# %%
# Define dataset class
class SinusoidalDataset(Dataset):
    def __init__(self, cluster_centers, points_per_cluster=100, cluster_spread=0.05, noise_std=0.02):
        self.cluster_centers = cluster_centers
        self.points_per_cluster = points_per_cluster
        self.noise_std = noise_std
        self.cluster_spread = cluster_spread

        # Generate clustered x values around each cluster center
        self.x = torch.cat([
            torch.normal(center, cluster_spread *
                         torch.ones(points_per_cluster, 1))
            for center in cluster_centers
        ])

        # Generate corresponding y values based on the sinusoidal function with added noise
        self.y = generate_data(self.x, noise_std=noise_std)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


# %%
# Define data clusters
cluster_centers = [torch.tensor(
    [-0.8]), torch.tensor([-0.2]), torch.tensor([0.5]), torch.tensor([0.9])]
points_per_cluster = 20  # Points per cluster
cluster_spread = 0.05    # Spread of points around each cluster center

# Initialize dataset and DataLoader
dataset = SinusoidalDataset(cluster_centers=cluster_centers,
                            points_per_cluster=points_per_cluster, cluster_spread=cluster_spread)
train_data = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

# %%
# Define Bayesian Neural Network


class BNN(PyroModule):
    def __init__(self, in_dim=1, out_dim=1, hid_dim=5, prior_scale=10.):
        super().__init__()
        self.activation = nn.Tanh()
        self.layer1 = PyroModule[nn.Linear](in_dim, hid_dim)
        self.layer2 = PyroModule[nn.Linear](hid_dim, out_dim)

        # Set layer weights as random variables
        self.layer1.weight = PyroSample(dist.Normal(
            0., prior_scale).expand([hid_dim, in_dim]).to_event(2))
        self.layer1.bias = PyroSample(dist.Normal(
            0., prior_scale).expand([hid_dim]).to_event(1))
        self.layer2.weight = PyroSample(dist.Normal(
            0., prior_scale).expand([out_dim, hid_dim]).to_event(2))
        self.layer2.bias = PyroSample(dist.Normal(
            0., prior_scale).expand([out_dim]).to_event(1))

    def forward(self, x, y=None):
        x = x.reshape(-1, 1)
        x = self.activation(self.layer1(x))
        mu = self.layer2(x).squeeze()
        sigma = pyro.sample("sigma", dist.Gamma(0.5, 1.0))

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        return mu


# %%
# Convert data to PyTorch tensors for a single batch
x_train, y_train = next(iter(train_data))
x_train = x_train.float()
y_train = y_train.float()

# %%
# Define HMC kernel and MCMC sampler
hmc_kernel = HMC(BNN(), step_size=0.01, num_steps=10)
mcmc = MCMC(hmc_kernel, num_samples=50, warmup_steps=10)
mcmc.run(x_train, y_train)
# %%
# Prediction using MCMC samples
predictive = Predictive(
    model=BNN(), posterior_samples=mcmc.get_samples())
x_test = torch.linspace(-1.0, 1.5, 3000).reshape(-1, 1)
preds = predictive(x_test)

# %%
# Plot predictions with confidence intervals
def plot_predictions(preds):
    y_pred = preds['obs'].mean(dim=0).detach().numpy()
    y_std = preds['obs'].std(dim=0).detach().numpy()
    x_test_np = x_test.numpy().flatten()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_test_np, y_pred, '-', linewidth=3,
            color="#408765", label="Predictive mean")
    ax.fill_between(x_test_np, y_pred - 2 * y_std, y_pred +
                    2 * y_std, alpha=0.3, color='#86cfac')

    # Plot the true function and clustered observations
    x_true = np.linspace(-1.0, 1.5, 1000)
    y_true = generate_data(torch.from_numpy(x_true)).numpy()
    ax.plot(x_true, y_true, 'b-', linewidth=3, label="True function")
    ax.plot(x_train.numpy(), y_train.numpy(), 'ko',
            markersize=4, label="Observations")

    ax.set_xlim([-1.0, 1.5])
    ax.set_ylim([-1.5, 2.5])
    ax.set_xlabel("X", fontsize=30)
    ax.set_ylabel("Y", fontsize=30)
    ax.legend(loc=4, fontsize=15, frameon=False)
    plt.show()

plot_predictions(preds)
# %%
