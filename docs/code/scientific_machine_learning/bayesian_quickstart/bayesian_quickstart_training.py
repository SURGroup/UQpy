"""
Bayesian Quickstart Training
============================
"""

# %% md
# This is the first half of a Bayesian version of the classification problem from this Pytorch Quickstart tutorial:
# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
#
# We strongly recommend reading the Pytorch quick start first to familiarize yourself with the problem.
# We include many of the comments from the Pytorch example, but assume the reader is familiar with model definitions
# and parameter optimization in Pytorch.
# In their tutorial, Pytorch implements a fully connected deterministic neural network to learn a classification of
# articles of clothing. Here, we implement a fully connected *Bayesian* neural network to learn the same classification.
#
# We import all the same packages, with the addition of UQpy's scientific machine learning module.
# Note that this demo requires the ``torchvision`` package.

# %%

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import UQpy.scientific_machine_learning as sml

# %% md
# The FashionMNIST dataset :cite:`xiao2017fashionMNIST` and dataloaders for our Bayesian classifier are identical to those used in the
# deterministic case.

# %%

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64
# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# %% md
# To construct a Bayesian classifier, we use UQpy's ``BayesianLinear`` layers in the ``nn.Module`` subclass.
# The ``BayesianLinear`` takes in the in features and out features just like ``nn.Linear``.
# The key difference is while the standard Linear layer has deterministic numbers for every element in its weight
# and bias tensors, a BayesianLinear layer has a representation of a random variable for each element in its tensors.

# %%

# Get cpu, gpu or mps device for training.
device = "cpu"
# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps" if torch.backends.mps.is_available() else "cpu"
# )
print(f"Using {device} device")


# Define model
class BayesianNeuralNetwork(nn.Module):
    """UQpy: Replace torch's nn.Linear with UQpy's sml.BayesianLinear"""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            sml.BayesianLinear(28 * 28, 512),  # nn.Linear(28 * 28, 512)
            nn.ReLU(),
            sml.BayesianLinear(512, 512),  # nn.Linear(512, 512)
            nn.ReLU(),
            sml.BayesianLinear(512, 512),  # nn.Linear(512, 512)
            nn.ReLU(),
            sml.BayesianLinear(512, 10),  # nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# %% md
# Just like in Torch tutorials, we define our network using an instance of this custom class.
# The network is wrapped inside the ``sml.FeedForwardNeuralNetwork`` object, which does not change the network but gives
# us better control over the random variable behavior. By default, the model is set to sampling mode, where the weights
# and biases of each Bayesian layer are sampled from their governing distribution. We can turn sampling off,
# which causes the model to use the means of its distributions (and consequently behave deterministically) with the
# command ``model.sample(False)``. To turn sampling back on, use ``model.sample(True)``.

# %%

network = BayesianNeuralNetwork().to(device)
model = sml.FeedForwardNeuralNetwork(network).to(device)
print(model)
model.sample(False)
print("model is in sampling mode:", model.sampling)
model.sample()
print("model is in sampling mode:", model.sampling)

# %% md
# With our model defined, we can turn our attention to training. Training a Bayesian neural network uses Torch's
# ``optim`` library, and we can reuse almost all of their training and testing functions.
# We use the testing function from torch's quickstart tutorial, with a small modification to the loss function.
# We add a divergence term to the loss, which represents the likelihood of the posterior distribution in the Bayesian
# update. The default prior distribution in a Bayesian layer is a zero mean Gaussian, so this effectively acts as a
# regularization term driving the weights and biases towards zero. We scale down the divergence by a factor of
# :math:`10^{-6}` so it doesn't dominate the total loss.
#
# The test function is nearly identical to torch's, with the inclusion of a single line to ensure the
# sampling mode is set to ``False``.

# %%

loss_fn = nn.CrossEntropyLoss()
# UQpy: define divergence function used in loss
divergence_fn = sml.GaussianKullbackLeiblerDivergence()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    """UQpy: Include a divergence term in the loss"""
    size = len(dataloader.dataset)
    model.train()
    # UQpy: Set to sample mode to True.
    model.sample()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # UQpy: Compute divergence and add it to the data loss
        beta = 1e-6
        loss += beta * divergence_fn(model)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    """UQpy: ensure model sampling is turned off."""
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    # UQpy: Set sampling mode to False
    model.sample(False)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "bayesian_model.pt")
print("Saved PyTorch Model State to bayesian_model.pt")


# %% md
# That's it! We defined the Bayesian neural network using UQpy's layers and added the model divergence to the loss
# function. That's the basics of defining and training a Bayesian neural network.
#
# In the next tutorial we make predictions with this trained Bayesian model.

# %%
