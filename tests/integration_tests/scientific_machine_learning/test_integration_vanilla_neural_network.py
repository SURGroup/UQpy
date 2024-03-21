import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from UQpy.scientific_machine_learning.neural_networks.VanillaNeuralNetwork import (
    VanillaNeuralNetwork,
)

torch.manual_seed(0)


class QuadraticDataset(Dataset):
    def __init__(self, n_samples=200):
        self.n_samples = n_samples
        self.x = torch.linspace(-5, 5, n_samples).reshape(-1, 1)
        self.y = self.x**2

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return self.x[item], self.y[item]


def test_accuracy():
    """Test a single hidden layer neural network learning the functino f(x) = x ** 2"""
    width = 10
    network = nn.Sequential(
        nn.Linear(1, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, 1),
    )
    model = VanillaNeuralNetwork(network)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    training_dataset = QuadraticDataset()
    data_loader = DataLoader(training_dataset, batch_size=20, shuffle=True)
    model.learn(data_loader, epochs=3_000)

    final_loss = model.history["train loss"][-1]
    assert final_loss < 1e-2
