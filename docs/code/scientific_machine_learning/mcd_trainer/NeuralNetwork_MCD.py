# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import UQpy.scientific_machine_learning as sml

# %%
plt.style.use("ggplot")
# import logging
# logger = logging.getLogger("UQpy")
# logger.setLevel(logging.INFO)

width = 20
p = 1e-3
network = nn.Sequential(
    nn.Linear(1, width),
    nn.ReLU(),
    nn.Linear(width, width),
    sml.Dropout(p=p),
    nn.ReLU(),
    nn.Linear(width, width),
    sml.Dropout(p=p),
    nn.ReLU(),
    nn.Linear(width, width),
    sml.Dropout(p=p),
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
            0.4 * torch.sin(4 * self.x) + 0.5 * torch.cos(12 * self.x)
        )
        self.y += torch.normal(0, self.noise_std, self.x.shape)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return self.x[item], self.y[item]


# %% Train with dropout layers inactive
model.drop(False)  # this turns off the dropout layers
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
train_data = DataLoader(SinusoidalDataset(), batch_size=5, shuffle=True)
trainer = sml.Trainer(model, optimizer, scheduler=scheduler)
print("Starting Training...", end="")
trainer.run(train_data=train_data, epochs=50_000, tolerance=0.0)
print("done")

# Plotting Results
x_data = SinusoidalDataset().x.detach()
y_data = SinusoidalDataset().y.detach()
x_val = torch.linspace(-1, 1, 1000).view(-1, 1).detach()
y_val = 0.4 * torch.sin(4 * x_val) + 0.5 * torch.cos(12 * x_val).detach()
pred_val = model(x_val).detach()

# %% Plot the deterministic model estimates
fig, ax = plt.subplots()
ax.scatter(
    x_data,
    y_data,
    label="Data",
    color="black",
    s=50
)
ax.plot(
    x_val,
    pred_val,
    label="Final Prediction",
    color="tab:orange",
)
ax.plot(
    x_val.detach(),
    y_val.detach(),
    label="Target",
    color="black",
    linestyle="dashed",
)
ax.set_title("Deterministic Prediction")
ax.set(xlabel="$x$", ylabel="$f(x)$")
ax.legend()
fig.tight_layout()

train_loss = trainer.history["train_loss"].detach().numpy()
fig, ax = plt.subplots()
ax.semilogy(train_loss)
ax.set_title("Training Loss")
ax.set(xlabel="Epoch", ylabel="Loss")
fig.tight_layout()

# %%
model.drop()  # activate the dropout layers
n = 1_000
samples = torch.zeros(n, len(x_val))
for i in range(n):
    samples[i, :] = model(x_val).detach().squeeze()
mean = torch.mean(samples, dim=1)
standard_deviation = torch.std(samples, dim=1)

# Plotting Results
fig, ax = plt.subplots()
ax.plot(x_val, samples[:, 1], "tab:orange", label="Prediction 1")
ax.plot(x_val, samples[:, 2], "tab:blue", label="Prediction 2")
ax.scatter(x_data, y_data, color="black", label="Data")
ax.plot(x_val, y_val, color="black", linestyle="dashed", label="Target")
ax.set_title("Two Samples from Dropout NN")
ax.set(xlabel="$x$", ylabel="$f(x)$")
ax.legend()
fig.tight_layout()

# %%
fig, ax = plt.subplots()
ax.plot(x_val, mean, label="$\mu$")
ax.fill_between(
    x_val.view(-1),
    torch.quantile(samples, q=0.025, dim=1),
    torch.quantile(samples, q=0.975, dim=1),
    label="95% Range",
    # mean - (3 * standard_deviation),
    # mean + (3 * standard_deviation),
    # label="$\mu \pm 3\sigma$,",
    alpha=0.3,
)
ax.plot(x_val, y_val, label="Target", color="black")
ax.set_title("Dropout Neural Network 95% Range")
ax.set(xlabel="x", ylabel="f(x)")
ax.legend()

plt.show()
# %%
