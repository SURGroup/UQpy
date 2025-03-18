import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import UQpy.scientific_machine_learning as sml

torch.manual_seed(0)


class TestBBBTrainer:
    """Test the __init__ and run methods of the BBBTrainer

    Note:
        This test does *not* check if the trained model is accurate
    """

    x = torch.tensor([-1.0, 1.0])
    y = x**2
    dataset = torch.utils.data.TensorDataset(x, y)
    train_dataset, test_dataset = random_split(dataset, [1, 1])
    train_data = DataLoader(train_dataset)
    test_data = DataLoader(test_dataset)

    model = sml.FeedForwardNeuralNetwork(sml.BayesianLinear(1, 1))
    optimizer = torch.optim.Adam(model.parameters())
    epochs = 2
    trainer = sml.BBBTrainer(model, optimizer)
    trainer.run(train_data, test_data, epochs=epochs)

    def test_history_train_loss(self):
        """Passes if train_loss has the correct length and does not contain NaN"""
        assert len(self.trainer.history["train_loss"]) == self.epochs
        contains_nan = any(torch.isnan(self.trainer.history["train_loss"]))
        assert not contains_nan

    def test_history_train_nll(self):
        """Passes if train_nll has the correct length and does not contain NaN"""
        assert len(self.trainer.history["train_nll"]) == self.epochs
        contains_nan = any(torch.isnan(self.trainer.history["train_nll"]))
        assert not contains_nan

    def test_history_train_divergence(self):
        """Passes if train_divergence has the correct length and does not contain NaN"""
        assert len(self.trainer.history["train_divergence"]) == self.epochs
        contains_nan = any(torch.isnan(self.trainer.history["train_divergence"]))
        assert not contains_nan

    def test_history_test_nll(self):
        """Passes if test_nll has the correct length and does not contain NaN"""
        assert len(self.trainer.history["test_nll"]) == self.epochs
        contains_nan = any(torch.isnan(self.trainer.history["test_nll"]))
        assert not contains_nan
