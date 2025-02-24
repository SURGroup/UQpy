import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import UQpy.scientific_machine_learning as sml

torch.manual_seed(0)


class TestTrainer:
    """Test the __init__ and run methods of the Trainer

    Note:
     This test does *not* check if the trained model is accurate
    """

    x = torch.tensor([-1.0, 1.0])
    y = x**2
    dataset = torch.utils.data.TensorDataset(x, y)
    train_dataset, test_dataset = random_split(dataset, [1, 1])
    train_data = DataLoader(train_dataset)
    test_data = DataLoader(test_dataset)

    model = nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    trainer = sml.Trainer(model, optimizer, scheduler=scheduler)
    epochs = 2
    trainer.run(train_data, test_data, epochs=epochs)

    def test_history_test_loss(self):
        """Passes if test loss is the correct length and does not contain NaN"""
        assert len(self.trainer.history["test_loss"]) == self.epochs
        contains_nan = any(torch.isnan(self.trainer.history["test_loss"]))
        assert not contains_nan

    def test_history_train_loss(self):
        """Passes if train loss is the correct length and does not contain NaN"""
        assert len(self.trainer.history["train_loss"]) == self.epochs
        contains_nan = any(torch.isnan(self.trainer.history["train_loss"]))
        assert not contains_nan

