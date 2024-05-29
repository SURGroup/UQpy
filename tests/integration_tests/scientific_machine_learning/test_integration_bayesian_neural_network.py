import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import UQpy.scientific_machine_learning as sml

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
    """Test a single hidden layer neural network learning the function f(x) = x ** 2"""
    width = 10
    network = nn.Sequential(
        sml.BayesianLinear(1, width),
        nn.ReLU(),
        sml.BayesianLinear(width, width),
        nn.ReLU(),
        sml.BayesianLinear(width, 1),
    )
    model = sml.FeedForwardNeuralNetwork(network)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_data = DataLoader(QuadraticDataset(), batch_size=20, shuffle=True)
    trainer = sml.BBBTrainer(model, optimizer)
    trainer.run(train_data=train_data, epochs=1_000, beta=1e-6, num_samples=10)

    final_loss = trainer.history["train_nll"][-1]
    assert final_loss < 1
