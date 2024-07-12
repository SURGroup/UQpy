"""
Bayesian Quickstart Testing
===========================

ToDo: finish this docstring
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import UQpy.scientific_machine_learning as sml

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
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
    """UQpy: We rename the class and replace torch's nn.Linear with UQpy's sml.BayesianLinear"""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.bayesian_linear_relu_stack = nn.Sequential(
            sml.BayesianLinear(28 * 28, 512),
            nn.ReLU(),
            sml.BayesianLinear(512, 512),
            nn.ReLU(),
            sml.BayesianLinear(512, 512),
            nn.ReLU(),
            sml.BayesianLinear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.bayesian_linear_relu_stack(x)
        return logits


network = BayesianNeuralNetwork().to(device)
model = sml.FeedForwardNeuralNetwork(network).to(device)  # UQpy: Place the network inside a UQpy.sml object
print(model)

loss_fn = nn.CrossEntropyLoss()
divergence_fn = sml.GaussianKullbackLeiblerDivergence()  # UQpy: define divergence function used in loss
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    """UQpy: Include a divergence term in the loss"""
    size = len(dataloader.dataset)
    model.train()
    model.sample()  # UQpy: Set to sample mode to True. Weights are sampled from their distributions
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # UQpy: Compute divergence and add it to the data loss
        loss += 1e-6 * divergence_fn(model)

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
    model.sample(False)  # UQpy: Set sampling mode to False. Use distribution means for weights
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

torch.save(model.state_dict(), "bayesian_model.pth")
print("Saved PyTorch Model State to bayesian_model.pth")
